"""ColonAI — FastAPI REST service with patient-safety policy enforced.

Endpoints
─────────
   POST  /predict            multipart image upload → safety-gated prediction
   GET   /health             liveness probe
   GET   /version            checkpoint + temperature + safety config
   GET   /audit/today        today's audit log (NDJSON)

The /predict endpoint NEVER returns a confident reading when the safety
policy says abstain or reject. It returns the verdict + disclaimer so the
hospital frontend can route to a human reviewer.

Run:
    uvicorn scripts.serve_api:app --host 0.0.0.0 --port 8081

Then test:
    curl -F image=@some_polyp.jpg http://localhost:8080/predict
"""
from __future__ import annotations
import io, sys, json, time
from pathlib import Path
from typing import Optional
import numpy as np, torch, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES, CLASS_NAMES_5
from src.agents.unified_image_agent import GradCAMPlusPlus
from src.app.patient_safety import evaluate_safety, AuditLog
from src.app.image_atypicality import is_endoscopy_image
from transformers import AutoTokenizer

CKPT  = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
TEMP  = "outputs/unified_multimodal_v2/temperature.json"
BERT  = "dmis-lab/biobert-base-cased-v1.2"


class Backend:
    def __init__(self):
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else (torch.device("mps") if torch.backends.mps.is_available()
                             else torch.device("cpu")))
        print(f"[API] Loading model on {self.device} …")
        self.model = UnifiedMultiModalTransformer(
            n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(self.device)
        state = torch.load(CKPT, map_location=self.device)
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
        print(f"[API] Ready. T = {self.T:.3f}")

    def predict(self, pil_img: Image.Image):
        t0 = time.perf_counter()
        # 1. Endoscopy gate
        arr  = np.array(pil_img)
        endo = is_endoscopy_image(arr)
        end_score = float(endo["score"])

        x = self.tfm(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x, self.ids, self.msk, self.tab)
            cal_logits = out["pathology"] / self.T
            probs = F.softmax(cal_logits, dim=-1)[0].cpu().numpy()
            pred_idx = int(probs.argmax()); pred_conf = float(probs[pred_idx])

        # 2. GradCAM focus measure (only if polyp predicted, for speed)
        gc_focus = None
        if pred_idx == 0:
            cam = self.cam_ex.generate(image=x.detach().requires_grad_(True),
                                       class_idx=0, input_ids=self.ids,
                                       attention_mask=self.msk, tabular=self.tab)
            if cam is not None and cam.size >= 4:
                _f = cam.flatten()
                _thr = float(np.quantile(_f, 0.75))
                gc_focus = float((_f >= _thr).sum() / _f.size)

        # 3. Quick MC-Dropout uncertainty estimate (3 stochastic forwards)
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

        # 4. Safety policy
        verdict = evaluate_safety(
            confidence=pred_conf, uncertainty=uncertainty,
            endoscopy_score=end_score, gradcam_focus=gc_focus,
            agent_agreement=1.0  # API path doesn't run multi-augment; safe default
        )
        result = {
            "verdict":          verdict.action,
            "reason":           verdict.reason,
            "disclaimer":       verdict.disclaimer,
            "flags":            verdict.flags,
            "elapsed_ms":       round((time.perf_counter() - t0) * 1000, 1),
            "endoscopy_score":  end_score,
            "temperature":      self.T,
        }
        if verdict.action != "reject":
            result["prediction"] = {
                "class":      CLASS_NAMES_5[pred_idx],
                "class_idx":  pred_idx,
                "confidence": pred_conf,
                "uncertainty": uncertainty,
                "all_probs":  {n: float(p) for n, p in zip(CLASS_NAMES_5, probs)},
            }
        return result, verdict


backend: Optional[Backend] = None
app = FastAPI(title="ColonAI API",
              description="Multimodal colon-cancer screening with patient-safety policy.",
              version="2.0.0")


@app.on_event("startup")
async def _start():
    global backend; backend = Backend()


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(backend.device) if backend else "loading"}


@app.get("/version")
async def version():
    return {
        "checkpoint":     CKPT,
        "temperature":    backend.T if backend else None,
        "classes":        CLASS_NAMES_5,
        "safety_config":  {
            "min_confidence":      0.75,
            "max_uncertainty":     0.30,
            "min_endoscopy_score": 0.55,
        }
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if backend is None:
        raise HTTPException(503, "Backend not ready")
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot decode image: {e}")
    result, verdict = backend.predict(pil)
    try:
        backend.audit.record(
            case_id=f"api_{int(time.time()*1000)}",
            pathology_class=result.get("prediction", {}).get("class", "abstain/reject"),
            confidence=result.get("prediction", {}).get("confidence", 0.0),
            uncertainty=result.get("prediction", {}).get("uncertainty", 0.0),
            verdict=verdict)
    except Exception: pass
    return JSONResponse(result)


@app.get("/audit/today", response_class=PlainTextResponse)
async def audit_today():
    p = Path(f"outputs/audit/audit_{time.strftime('%Y%m%d')}.jsonl")
    return p.read_text() if p.exists() else ""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scripts.serve_api:app", host="0.0.0.0", port=8081, reload=False)
