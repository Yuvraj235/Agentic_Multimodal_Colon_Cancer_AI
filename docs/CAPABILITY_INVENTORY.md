# ColonAI — Capability Inventory (regression baseline)

Snapshot of **what the system does today**, captured at the start of the
clinician-extraction upgrade (branch `clinician-extraction`, 2026-06-14). Every
later phase must re-verify this list — nothing here may break. This is the
"upgrade, not rewrite" contract.

## Wizard pages (step → function in `app.py`)
| Step | Page | Function | What it does |
|---|---|---|---|
| 0 | Patient Info | `page_patient_info` | collect demographics + risk factors |
| 1 | Symptoms & Upload | `page_symptoms_upload` | symptom checker · image upload · existing-report upload |
| 2 | AI Analysis | `page_analysis` | runs the model + 6-agent pipeline; the core inference path |
| 3 | Results | `page_results` | verdict, GradCAM/seg localization, CADx card, rationale, staging, OOD/view-quality banners |
| 4 | Find Doctors | `page_doctor_finder` | locate specialists (geo) |
| 5 | Download Report | `page_report` | PDF report generation |
| 6 | Live Video Mode | `page_live_video` | upload clip / webcam → real-time YOLO polyp detection + tracking |
| 7 | Latest Research | `page_latest_research` | auto-updated cancer-news feed |

**Auxiliary pages (sidebar/help-routed, not in 0–7 wizard):**
`page_guide` (how-it-works walkthrough) · `page_recalibration` (clinician per-site
temperature calibration tool).

## Agents (`src/agents/`)
Orchestrators: `multimodal_orchestrator.py` (primary, coordinates all), `orchestrator.py`.
Specialists: `unified_image_agent.py` (GradCAM++), `image_agent.py`, `text_agent.py` /
`clinical_text_agent.py`, `tabular_risk_agent.py`, `fusion_reasoning_agent.py` /
`fusion_agent.py`, `pathology_agent.py`, `xai_agent.py` (MC-dropout uncertainty),
`clinical_recommendation_agent.py` (guideline next-steps), `clinical_explanation_agent.py` /
`explanation_agent.py`.

## Specialist app modules (`src/app/`) — key ones
- **Safety / honesty:** `patient_safety.py` (show/abstain/reject — "Requires human review"),
  `ood_gate.py` (out-of-scope detector), `view_quality.py` (poor-prep gate),
  `image_atypicality.py` (endoscopy-likeness gate; rejects radiology), `cross_check.py`
  (agent-agreement), `security.py` (upload validation), `reliability.py`.
- **Clinical outputs:** `staging.py` (AJCC from TNM), `crc_risk_model.py` (RR + APCS),
  `guideline_kb.py` (cited answers), `polyp_typing.py` (Paris/NICE/size band),
  `characterization.py` (CADx neoplastic/non-neoplastic), `histology.py` (H&E tissue),
  `smart_rationale.py` / `smart_inference.py`, `explanation_engine.py`, `decision_trace.py`,
  `recalibration.py`, `report_generator.py`, `patient_ui.py`.
- **Vision:** `segmentation.py` (seg decoder), `video_pipeline.py` (real-time detector +
  tracker), `strong_xai.py`, `modality_attribution.py`, `prototype_retrieval.py`,
  `counterfactual.py`.
- **Misc:** `geo.py`, `llm_refine.py` (optional Groq), `learning_log.py`, `ui_extras.py`.

## Trained model files (`outputs/unified_multimodal_v2/`, gitignored/local)
- `checkpoints/best_model.pth` — unified multimodal model (uc-fix promoted; downloaded on the
  Space from model repo `Yuvraj2319/colonai-v2`)
- `seg_head.pth` — segmentation decoder (multi-centre)
- `polyp_detector.pt` — YOLO real-time polyp detector (ETIS mAP50 0.876 / sens 0.822)
- `cadx_head.pth` — neoplastic-vs-non-neoplastic characterization
- `histology_head.pth` — 9-class H&E tissue
- `ood_head.pth` — real out-of-scope detector
- `view_quality_head.pth` — bowel-prep / view-quality gate
- (plus backups: `best_model_pre_ucfix.pth`, `seg_head_*` variants, `ood_head_*` variants)

## REST API (`scripts/serve_api.py`)
`GET /health` · `GET /version` · `POST /predict` (image → prediction + safety action) ·
`GET /audit/today` (audit log).

## Safety / abstain layer (the doctor's fail-safe — already implemented)
`patient_safety.evaluate_safety()` → action ∈ {show, abstain, reject}; abstain ⇒
"Requires human review". Thresholds: min_confidence 0.75, max_uncertainty 0.30,
gradcam-focus + agent-agreement checks. Sensitivity-first by design.

## Deployment
HF Space `Yuvraj2319/colonai` (docker SDK); heavy checkpoint pulled at runtime from model
repo `Yuvraj2319/colonai-v2`; small heads baked into the Space repo. See
[[deployment-hf-space]].

---
**Regression rule:** after every phase, confirm each page (0–7 + auxiliary), each REST
endpoint, and the safety layer still behave as above before proceeding.
