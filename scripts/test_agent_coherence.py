"""ColonAI — Agent Coherence & Deployment Readiness Test.

After retraining we want to PROVE the system isn't producing fake or
misaligned outputs. This script:

  Case A — Real polyp from ETIS-Larib (Pentax, the hardest vendor):
            * full orchestrator pipeline
            * GradCAM IoU vs ground-truth mask MUST be ≥ 0.30 (was 0.07 before)
            * pathology head MUST predict "polyps"
            * image agent emits a non-empty heatmap
            * text/tabular/fusion/xai/recommendation all run without exception

  Case B — Non-polyp endoscopy (HyperKvasir ulcerative-colitis):
            * pathology head MUST NOT predict "polyps"
            * endoscopy gate MUST return is_endoscopy = True

  Case C — Random non-endoscopy image (synthetic blue gradient):
            * endoscopy gate MUST return is_endoscopy = False
            * orchestrator should not be invoked (would produce a fake)

Writes outputs/unified_multimodal_v2/agent_coherence_report.json with
PASS/FAIL per check.
"""
from __future__ import annotations
import sys, json, traceback
from pathlib import Path
from typing import Dict

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES, CLASS_NAMES_5
from src.agents.unified_image_agent import GradCAMPlusPlus
from src.agents.multimodal_orchestrator import MultiModalOrchestrator
from src.app.image_atypicality import is_endoscopy_image
from transformers import AutoTokenizer


CHECKPOINT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
BERT = "dmis-lab/biobert-base-cased-v1.2"

# Test sample paths
POLYP_IMG = "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/images"
POLYP_MSK = "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/masks"


def _preprocess(pil):
    return T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(pil).unsqueeze(0)


def _iou(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 1 else 0.0


def _load_model(device):
    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    return model


def build_orchestrator(device):
    model = _load_model(device)
    tok   = AutoTokenizer.from_pretrained(BERT)
    orch  = MultiModalOrchestrator(model=model, tokenizer=tok, device=device,
                                   output_dir="outputs/unified_multimodal_v2/cases")
    return orch, model, tok


def make_inputs(pil_img: Image.Image, tok, device):
    image = _preprocess(pil_img).to(device)
    enc = tok("Patient referred for screening colonoscopy.",
              padding="max_length", truncation=True, max_length=64,
              return_tensors="pt")
    return (image,
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            torch.zeros((1, N_TABULAR_FEATURES), device=device))


def case_A_real_polyp(orch, model, tok, device) -> Dict:
    """A real Pentax polyp — the hardest cross-vendor case."""
    img_dir = Path(POLYP_IMG); msk_dir = Path(POLYP_MSK)
    img_files = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg"))
    if not img_files:
        return {"name": "A_real_polyp", "PASS": False,
                "fail_reason": "no test images found"}
    img_path = img_files[len(img_files)//2]   # mid-corpus sample
    msk_path = None
    for e in (".png", ".jpg", ".tif", ".tiff"):
        c = msk_dir / (img_path.stem + e)
        if c.exists(): msk_path = c; break
    pil_img  = Image.open(img_path).convert("RGB")
    gt_mask  = np.array(Image.open(msk_path).convert("L")) > 127

    checks = []

    # 1. Endoscopy gate must accept it
    arr = np.array(pil_img)
    endo = is_endoscopy_image(arr)
    checks.append(("endoscopy_gate_accepts", bool(endo["is_endoscopy"]),
                   f"score={endo['score']:.3f}"))

    # 2. Pathology must say polyps (class 0)
    image, ids, msk_attn, tab = make_inputs(pil_img, tok, device)
    with torch.no_grad():
        out = model(image, ids, msk_attn, tab)
        prob = F.softmax(out["pathology"], dim=-1)[0]
        pred = int(prob.argmax())
    checks.append(("pathology_predicts_polyp", pred == 0,
                   f"pred={CLASS_NAMES_5[pred]} p={prob[0].item():.3f}"))

    # 3. GradCAM IoU vs GT mask must be ≥ 0.30
    cam_ex = GradCAMPlusPlus(model, model.get_image_target_layer())
    cam = cam_ex.generate(image=image.detach().requires_grad_(True),
                          class_idx=0, input_ids=ids,
                          attention_mask=msk_attn, tabular=tab)
    iou_val = 0.0
    if cam is not None and cam.size >= 4:
        cam_r = cv2.resize(cam.astype(np.float32),
                           (pil_img.width, pil_img.height),
                           interpolation=cv2.INTER_LINEAR)
        thr = float(np.quantile(cam_r, 0.75))
        pred_m = cam_r >= thr
        iou_val = _iou(pred_m, gt_mask)
    checks.append(("gradcam_iou_ge_0.30", iou_val >= 0.30,
                   f"iou={iou_val:.3f} (baseline was 0.07 on Pentax)"))

    # 4. Full orchestrator runs without exception, all 6 agents emit results
    try:
        result = orch.run(image=image, input_ids=ids, attention_mask=msk_attn,
                          tabular=tab, raw_image_np=np.array(pil_img),
                          text="Patient referred for screening colonoscopy.",
                          save=False)
        has_img  = result.image_evidence is not None
        has_txt  = result.text_evidence  is not None
        has_tab  = result.tabular_evidence is not None
        has_fuse = result.fusion_diagnosis is not None
        has_xai  = result.xai_report is not None
        has_rec  = result.clinical_recommendation is not None
        all_ok   = all([has_img, has_txt, has_tab, has_fuse, has_xai, has_rec])
        checks.append(("orchestrator_all_6_agents_emitted", all_ok,
                       f"img={has_img} txt={has_txt} tab={has_tab} "
                       f"fuse={has_fuse} xai={has_xai} rec={has_rec}"))
    except Exception as e:
        checks.append(("orchestrator_all_6_agents_emitted", False,
                       f"EXCEPTION: {type(e).__name__}: {e}"))

    pass_count = sum(1 for _, ok, _ in checks if ok)
    return {
        "name":       "A_real_polyp_Pentax",
        "PASS":       pass_count == len(checks),
        "checks":     [{"name": n, "PASS": ok, "info": info}
                       for n, ok, info in checks],
        "pass_count": pass_count, "total": len(checks),
        "image_path": str(img_path),
    }


def case_B_non_polyp_endoscopy(orch, model, tok, device) -> Dict:
    """A HyperKvasir UC image — must NOT classify as polyps."""
    candidates = list(Path("data/processed/hyper_kvasir_clean/lower-gi-tract/"
                           "pathological-findings/ulcerative-colitis-grade-2").iterdir())
    candidates = [p for p in candidates if p.suffix.lower() in (".jpg",".png",".jpeg")]
    if not candidates:
        return {"name": "B_non_polyp_endoscopy", "PASS": False,
                "fail_reason": "no UC test images found"}
    img_path = candidates[len(candidates)//2]
    pil_img  = Image.open(img_path).convert("RGB")
    arr      = np.array(pil_img)

    checks = []
    endo = is_endoscopy_image(arr)
    checks.append(("endoscopy_gate_accepts", bool(endo["is_endoscopy"]),
                   f"score={endo['score']:.3f}"))

    image, ids, msk_attn, tab = make_inputs(pil_img, tok, device)
    with torch.no_grad():
        prob = F.softmax(model(image, ids, msk_attn, tab)["pathology"], dim=-1)[0]
        pred = int(prob.argmax())
    checks.append(("pathology_NOT_polyp", pred != 0,
                   f"pred={CLASS_NAMES_5[pred]} polyp_prob={prob[0].item():.3f}"))

    pass_count = sum(1 for _, ok, _ in checks if ok)
    return {
        "name":       "B_non_polyp_endoscopy",
        "PASS":       pass_count == len(checks),
        "checks":     [{"name": n, "PASS": ok, "info": info}
                       for n, ok, info in checks],
        "pass_count": pass_count, "total": len(checks),
        "image_path": str(img_path),
    }


def case_C_random_image(orch, model, tok, device) -> Dict:
    """Synthetic blue gradient — endoscopy gate MUST reject it."""
    rng = np.random.default_rng(42)
    # Blue-dominated photo (looks nothing like tissue) — many pixels with B>R
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        for x in range(512):
            arr[y, x] = [int(40 + rng.normal(0, 8)),
                         int(80 + rng.normal(0, 8)),
                         int(180 + rng.normal(0, 8))]
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    checks = []
    endo = is_endoscopy_image(arr)
    checks.append(("endoscopy_gate_REJECTS_non_endoscopy",
                   not bool(endo["is_endoscopy"]),
                   f"score={endo['score']:.3f}, signals={endo.get('signals', {})}"))

    pass_count = sum(1 for _, ok, _ in checks if ok)
    return {
        "name":       "C_random_non_endoscopy",
        "PASS":       pass_count == len(checks),
        "checks":     [{"name": n, "PASS": ok, "info": info}
                       for n, ok, info in checks],
        "pass_count": pass_count, "total": len(checks),
    }


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}")

    if not Path(CHECKPOINT).exists():
        print(f"  ERROR: checkpoint not found at {CHECKPOINT}.  "
              "Run scripts/retrain_deploy_grade.py first.")
        return

    print("\nBuilding orchestrator + all 6 agents …")
    orch, model, tok = build_orchestrator(device)

    print("\n┌──────────────────────────────────────────────────────────────┐")
    print("│ AGENT COHERENCE & DEPLOYMENT READINESS                        │")
    print("└──────────────────────────────────────────────────────────────┘")

    cases = []
    for case_fn, name in [(case_A_real_polyp,          "Case A — Pentax polyp"),
                          (case_B_non_polyp_endoscopy, "Case B — Non-polyp endo"),
                          (case_C_random_image,        "Case C — Non-endoscopy")]:
        print(f"\n→ {name}")
        try:
            r = case_fn(orch, model, tok, device)
        except Exception as e:
            r = {"name": name, "PASS": False, "exception": str(e),
                 "trace": traceback.format_exc()}
        cases.append(r)
        if "checks" in r:
            for c in r["checks"]:
                mark = "✓" if c["PASS"] else "✗"
                print(f"    {mark}  {c['name']}  ({c['info']})")
        print(f"    ► {'PASS' if r.get('PASS') else 'FAIL'}")

    total_pass = sum(1 for c in cases if c.get("PASS"))
    print(f"\n┌──────────────────────────────────────────────────────────────┐")
    print(f"│ RESULT: {total_pass}/{len(cases)} cases PASSED                                 │")
    print(f"└──────────────────────────────────────────────────────────────┘")

    out = Path("outputs/unified_multimodal_v2/agent_coherence_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": CHECKPOINT,
        "total_pass": total_pass,
        "total":      len(cases),
        "cases":      cases,
    }, indent=2))
    print(f"\nReport → {out}")


if __name__ == "__main__":
    main()
