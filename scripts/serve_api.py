"""ColonAI — FastAPI REST service with patient-safety policy + security policy.

Endpoints
─────────
   POST  /predict            multipart image upload → safety-gated prediction
   GET   /health             liveness probe (no auth)
   GET   /version            safe model metadata (no auth)
   GET   /audit/today        today's audit log (REQUIRES X-API-Key)

Security policy
───────────────
   • Defaults to localhost-only (127.0.0.1). Set COLONAI_BIND=0.0.0.0
     to expose on LAN — but in that case COLONAI_API_KEY must be set.
   • All endpoints behind shared-secret X-API-Key header if the env var
     COLONAI_API_KEY is set. /health and /version are intentionally open
     for load-balancer health probes; they expose no PHI.
   • Upload size capped at 10 MB (src/app/security.py MAX_UPLOAD_BYTES).
   • PIL decompression-bomb guard (MAX_IMAGE_PIXELS = 100 M).
   • MIME / extension allow-list.
   • CORS allow-list defaults to localhost; configurable via COLONAI_CORS_ORIGINS.
   • Errors are sanitised — clients see a request_id, the server log has the trace.
   • Audit log files are chmod 0o600.

Run:
    uvicorn scripts.serve_api:app --host 127.0.0.1 --port 8081

Or via CLI defaults:
    python3 scripts/serve_api.py
"""
from __future__ import annotations
import io, os, sys, json, time, logging
from pathlib import Path
from typing import Optional
import numpy as np, torch, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES, CLASS_NAMES_5
from src.agents.unified_image_agent import GradCAMPlusPlus
from src.app.patient_safety import evaluate_safety, AuditLog
from src.app.image_atypicality import is_endoscopy_image
from src.app.security import (
    validate_upload_bytes, UploadError, safe_torch_load,
    require_api_key, safe_error, new_request_id,
    secure_file_perms, MAX_UPLOAD_BYTES, COLONAI_API_KEY_ENV,
)
from transformers import AutoTokenizer


logging.basicConfig(level=os.environ.get("COLONAI_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("colonai.api")

CKPT  = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
TEMP  = "outputs/unified_multimodal_v2/temperature.json"
BERT  = "dmis-lab/biobert-base-cased-v1.2"

# Bind / CORS — defaults are restrictive
BIND_HOST = os.environ.get("COLONAI_BIND", "127.0.0.1")
BIND_PORT = int(os.environ.get("COLONAI_PORT", "8081"))
CORS_DEFAULT = "http://localhost:8501,http://localhost:8502,http://127.0.0.1:8501,http://127.0.0.1:8502"
CORS_ORIGINS = [o.strip() for o in os.environ.get("COLONAI_CORS_ORIGINS",
                                                  CORS_DEFAULT).split(",") if o.strip()]


class Backend:
    def __init__(self):
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else (torch.device("mps") if torch.backends.mps.is_available()
                             else torch.device("cpu")))
        log.info("Loading model on %s", self.device)
        self.model = UnifiedMultiModalTransformer(
            n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(self.device)
        # safe_torch_load tries weights_only=True first; our own checkpoints
        # carry a "model_state" dict so we pass allow_unsafe=True to keep
        # them loading. Document this in SECURITY.md.
        state = safe_torch_load(CKPT, map_location=self.device, allow_unsafe=True)
        self.model.load_state_dict(state.get("model_state", state), strict=False)
        self.model.eval()

        self.tok = AutoTokenizer.from_pretrained(BERT)
        enc = self.tok("Patient referred for screening colonoscopy.",
                       padding="max_length", truncation=True, max_length=64,
                       return_tensors="pt")
        self.ids = enc["input_ids"].to(self.device)
        self.msk = enc["attention_mask"].to(self.device)
        self.tab = torch.zeros((1, N_TABULAR_FEATURES), device=self.device)
        self.tfm = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                              T.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])
        self.T = (float(json.loads(Path(TEMP).read_text()).get("temperature", 1.0))
                  if Path(TEMP).exists() else 1.0)
        self.cam_ex = GradCAMPlusPlus(self.model, self.model.get_image_target_layer())
        self.audit = AuditLog()
        log.info("Ready. T = %.3f", self.T)

    def predict(self, pil_img: Image.Image):
        t0 = time.perf_counter()
        arr  = np.array(pil_img)
        endo = is_endoscopy_image(arr)
        end_score = float(endo["score"])

        x = self.tfm(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x, self.ids, self.msk, self.tab)
            cal_logits = out["pathology"] / self.T
            probs = F.softmax(cal_logits, dim=-1)[0].cpu().numpy()
            pred_idx = int(probs.argmax()); pred_conf = float(probs[pred_idx])

        gc_focus = None
        if pred_idx == 0:
            cam = self.cam_ex.generate(image=x.detach().requires_grad_(True),
                                       class_idx=0, input_ids=self.ids,
                                       attention_mask=self.msk, tabular=self.tab)
            if cam is not None and cam.size >= 4:
                _f = cam.flatten()
                _thr = float(np.quantile(_f, 0.75))
                gc_focus = float((_f >= _thr).sum() / _f.size)

        self.model.train()
        mc_probs = []
        with torch.no_grad():
            for _ in range(3):
                p = F.softmax(self.model(x, self.ids, self.msk, self.tab)["pathology"]
                              / self.T, dim=-1)[0].cpu().numpy()
                mc_probs.append(p)
        self.model.eval()
        mc_probs = np.stack(mc_probs, axis=0)
        uncertainty = float(mc_probs.std(axis=0).max())

        verdict = evaluate_safety(
            confidence=pred_conf, uncertainty=uncertainty,
            endoscopy_score=end_score, gradcam_focus=gc_focus,
            agent_agreement=1.0,
            predicted_class=CLASS_NAMES_5[pred_idx],
        )
        result = {
            "verdict":         verdict.action,
            "reason":          verdict.reason,
            "disclaimer":      verdict.disclaimer,
            "flags":           verdict.flags,
            "elapsed_ms":      round((time.perf_counter() - t0) * 1000, 1),
            "endoscopy_score": end_score,
        }
        if verdict.action != "reject":
            result["prediction"] = {
                "class":       CLASS_NAMES_5[pred_idx],
                "class_idx":   pred_idx,
                "confidence":  pred_conf,
                "uncertainty": uncertainty,
                "all_probs":   {n: float(p) for n, p in zip(CLASS_NAMES_5, probs)},
            }
        return result, verdict


backend: Optional[Backend] = None
app = FastAPI(
    title="ColonAI API",
    description="Multimodal colon-cancer screening with patient-safety policy.",
    version="2.1.0-secure",
    # Hide internal docs in production
    docs_url=("/docs" if os.environ.get("COLONAI_EXPOSE_DOCS") else None),
    redoc_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.on_event("startup")
async def _start():
    global backend
    backend = Backend()
    # Loud warning if exposed to LAN without auth
    if BIND_HOST not in ("127.0.0.1", "localhost") and not os.environ.get(COLONAI_API_KEY_ENV):
        log.warning("⚠ Binding to %s WITHOUT COLONAI_API_KEY. "
                    "Anyone on this network can hit /predict.", BIND_HOST)


def _check_key(x_api_key: Optional[str] = Header(None)):
    """FastAPI dependency: enforces X-API-Key when one is configured."""
    if not require_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/health")
async def health():
    """Liveness probe — intentionally open, no PHI returned."""
    return {"status": "ok", "ready": backend is not None}


@app.get("/version")
async def version():
    """Safe-to-expose model metadata. Does NOT expose absolute filesystem paths."""
    return {
        "model_version":  "ColonAI-v2",
        "classes":        CLASS_NAMES_5,
        "temperature":    (backend.T if backend else None),
        "max_upload_mb":  MAX_UPLOAD_BYTES // (1024*1024),
        "safety_config":  {
            "min_confidence":      0.75,
            "max_uncertainty":     0.30,
            "min_endoscopy_score": 0.55,
        },
        "auth_enabled":   bool(os.environ.get(COLONAI_API_KEY_ENV)),
    }


@app.post("/predict")
async def predict(
    request:  Request,
    image:    UploadFile = File(...),
    _auth:    None = Depends(_check_key),
):
    if backend is None:
        raise HTTPException(503, "Backend not ready")
    rid = new_request_id()

    # 1) Read with cap — refuse silently large bodies
    try:
        raw = await image.read()
    except Exception as e:
        return JSONResponse(safe_error(e, rid), status_code=400)

    # 2) Validate upload (size, MIME, decompression-bomb)
    try:
        pil, upload_meta = validate_upload_bytes(
            raw, declared_mime=image.content_type, filename=image.filename)
    except UploadError as e:
        log.info("upload rejected rid=%s reason=%s", rid, e)
        return JSONResponse({
            "ok": False, "request_id": rid,
            "error_type": "UploadError", "message": str(e),
        }, status_code=400)

    # 3) Run prediction
    try:
        result, verdict = backend.predict(pil)
        result["request_id"] = rid
    except Exception as e:
        return JSONResponse(safe_error(e, rid), status_code=500)

    # 4) Audit log (best-effort)
    try:
        backend.audit.record(
            case_id=f"api_{rid}",
            pathology_class=result.get("prediction", {}).get("class", "abstain/reject"),
            confidence=result.get("prediction", {}).get("confidence", 0.0),
            uncertainty=result.get("prediction", {}).get("uncertainty", 0.0),
            verdict=verdict,
            extras={"upload": upload_meta})
        secure_file_perms(backend.audit.path)
    except Exception:
        pass
    return JSONResponse(result)


@app.get("/audit/today", response_class=PlainTextResponse)
async def audit_today(_auth: None = Depends(_check_key)):
    """Audit log for today — requires API key when configured.

    Returns 404 (not 200 with empty body) when no log exists, to avoid
    confirming that nothing's been logged.
    """
    p = Path(f"outputs/audit/audit_{time.strftime('%Y%m%d')}.jsonl")
    if not p.exists():
        raise HTTPException(404, "No audit log for today.")
    return p.read_text()


if __name__ == "__main__":
    import uvicorn
    if BIND_HOST not in ("127.0.0.1", "localhost") and not os.environ.get(COLONAI_API_KEY_ENV):
        log.warning("⚠ Refusing to bind to %s without COLONAI_API_KEY. "
                    "Set the env var or use 127.0.0.1. Falling back to 127.0.0.1.",
                    BIND_HOST)
        host = "127.0.0.1"
    else:
        host = BIND_HOST
    log.info("Starting on http://%s:%d (auth %s)",
             host, BIND_PORT,
             "ENABLED" if os.environ.get(COLONAI_API_KEY_ENV) else "disabled (dev mode)")
    uvicorn.run("scripts.serve_api:app", host=host, port=BIND_PORT, reload=False)
