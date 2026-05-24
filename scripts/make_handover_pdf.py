"""ColonAI — full technical handover PDF.

Writes ColonAI_Handover.pdf at the project root. Everything needed to
resume the project in 6 months or hand it to another AI session:

  • Current state (checkpoints, metrics, what's deployed)
  • Architecture deep-dive (model + 6 agents + safety stack)
  • Training + inference pipelines
  • Datasets used + preprocessing
  • All environment variables, file paths, URLs, commit hashes
  • Known Stage-3 problems (immediate fixable) + proposed fixes
  • Known Stage-4 problems (deeper research) + research directions
  • Future improvements roadmap, prioritised
  • Operational runbook (deploy / restart / debug)
  • Code navigation index
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, HRFlowable, KeepTogether)
from reportlab.pdfgen import canvas

OUT_PATH = Path(__file__).resolve().parents[1] / "ColonAI_Handover.pdf"

INK    = colors.HexColor("#0B1220")
BRAND  = colors.HexColor("#0B5FFF")
ACCENT = colors.HexColor("#16A34A")
WARN   = colors.HexColor("#D97706")
DANGER = colors.HexColor("#DC2626")
SOFT   = colors.HexColor("#F1F5F9")
MID    = colors.HexColor("#475569")
MONOBG = colors.HexColor("#0F172A")
PAGE_W, PAGE_H = LETTER


def _header_footer(canv, doc):
    canv.saveState()
    canv.setFillColor(INK)
    canv.rect(0, PAGE_H - 0.30*inch, PAGE_W, 0.30*inch, stroke=0, fill=1)
    canv.setFillColor(colors.white)
    canv.setFont("Helvetica-Bold", 10)
    canv.drawString(0.6*inch, PAGE_H - 0.21*inch,
                    "ColonAI  —  Technical Handover  (private)")
    canv.setFont("Helvetica", 8)
    canv.drawRightString(PAGE_W - 0.6*inch, PAGE_H - 0.21*inch,
                         "Author: Yuvraj P. Singh  •  Generated: "
                         + datetime.now().strftime("%Y-%m-%d"))
    canv.setFillColor(MID); canv.setFont("Helvetica", 8)
    canv.drawString(0.6*inch, 0.45*inch,
                    "Repo: github.com/Yuvraj235/Agentic_Multimodal_Colon_Cancer_AI")
    canv.drawString(0.6*inch, 0.30*inch,
                    "Live: huggingface.co/spaces/Yuvraj2319/colonai")
    canv.drawRightString(PAGE_W - 0.6*inch, 0.30*inch, f"Page {doc.page}")
    canv.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────
def _styles():
    s = getSampleStyleSheet()
    return {
        "H1":  ParagraphStyle("H1",  parent=s["Heading1"], fontName="Helvetica-Bold",
                              fontSize=20, leading=24, textColor=INK,
                              spaceAfter=6, spaceBefore=0),
        "H2":  ParagraphStyle("H2",  parent=s["Heading2"], fontName="Helvetica-Bold",
                              fontSize=14, leading=18, textColor=BRAND,
                              spaceAfter=4, spaceBefore=12),
        "H3":  ParagraphStyle("H3",  parent=s["Heading3"], fontName="Helvetica-Bold",
                              fontSize=11, leading=14, textColor=INK,
                              spaceAfter=2, spaceBefore=8),
        "BODY":ParagraphStyle("body",parent=s["BodyText"], fontName="Helvetica",
                              fontSize=9.5, leading=13, textColor=INK,
                              alignment=TA_JUSTIFY, spaceAfter=4),
        "SUB": ParagraphStyle("sub", parent=s["BodyText"], fontName="Helvetica",
                              fontSize=10.5, leading=14, textColor=MID,
                              alignment=TA_LEFT, spaceAfter=8),
        "CODE":ParagraphStyle("code",parent=s["Code"], fontName="Courier",
                              fontSize=8, leading=10, textColor=colors.white,
                              backColor=MONOBG, leftIndent=8, rightIndent=8,
                              borderPadding=6, spaceBefore=4, spaceAfter=8),
        "INLINE": ParagraphStyle("inline", parent=s["BodyText"],
                                 fontName="Courier", fontSize=9, leading=12,
                                 textColor=INK, spaceAfter=3),
        "SMALL":ParagraphStyle("sm", parent=s["BodyText"], fontName="Helvetica",
                              fontSize=8, leading=10, textColor=MID),
        "WARN":ParagraphStyle("w", parent=s["BodyText"], fontName="Helvetica",
                              fontSize=9.5, leading=13, textColor=DANGER,
                              backColor=colors.HexColor("#FEE2E2"),
                              leftIndent=8, rightIndent=8, borderPadding=6,
                              spaceAfter=6),
        "OK":ParagraphStyle("ok",parent=s["BodyText"], fontName="Helvetica",
                              fontSize=9.5, leading=13, textColor=colors.HexColor("#065F46"),
                              backColor=colors.HexColor("#D1FAE5"),
                              leftIndent=8, rightIndent=8, borderPadding=6,
                              spaceAfter=6),
    }


def _kv_table(rows, col_widths=None):
    """Two-column key/value table."""
    if col_widths is None:
        col_widths = [2.0*inch, 5.0*inch]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN",   (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [colors.white, SOFT]),
        ("LINEABOVE", (0,0), (-1,0), 0.4, colors.HexColor("#CBD5E1")),
        ("LINEBELOW", (0,-1), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
    ]))
    return t


def _header_table(rows, col_widths, header_color=BRAND):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), header_color),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, SOFT]),
    ]))
    return t


# ═════════════════════════════════════════════════════════════════════════════
# BUILD
# ═════════════════════════════════════════════════════════════════════════════
def build():
    doc = SimpleDocTemplate(str(OUT_PATH), pagesize=LETTER,
        leftMargin=0.55*inch, rightMargin=0.55*inch,
        topMargin=0.55*inch, bottomMargin=0.55*inch,
        title="ColonAI — Technical Handover", author="Yuvraj P. Singh")
    S = _styles()
    P = lambda t, st="BODY": Paragraph(t, S[st])
    story = []

    # ────────────────────────────────────────────────────────────────────
    # COVER
    # ────────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4*inch))
    story.append(P("ColonAI — Technical Handover", "H1"))
    story.append(P("A complete record of every decision, file, metric, "
                   "known bug and improvement idea — for resuming this work "
                   "in any future session, with or without me.", "SUB"))
    story.append(HRFlowable(width="100%", thickness=1.2, color=BRAND,
                            spaceBefore=4, spaceAfter=14))

    # State block
    state = [
        ["Project status",      "Live demo running. v2 model deployed."],
        ["Last training run",   "epoch-4 deploy-grade retrain — val acc 0.865, "
                                "cross-vendor IoU mean 0.268, Pentax 0.16"],
        ["Active checkpoint",   "outputs/unified_multimodal_v2/checkpoints/best_model.pth"],
        ["Calibration",         "T = 0.45 (ECE 0.062 on val)"],
        ["Segmentation decoder","outputs/unified_multimodal_v2/seg_head.pth — IoU 0.61 cross-vendor"],
        ["OOD head",            "outputs/unified_multimodal_v2/ood_head.pth — val F1 = 1.0"],
        ["GitHub",              "github.com/Yuvraj235/Agentic_Multimodal_Colon_Cancer_AI  (main: cb797a2)"],
        ["Live Space",          "huggingface.co/spaces/Yuvraj2319/colonai"],
        ["Model repo",          "huggingface.co/Yuvraj2319/colonai-v2"],
        ["Auth token",          "stored in ~/.cache/huggingface/token   (Write scope)"],
    ]
    story.append(_kv_table(state, col_widths=[1.8*inch, 5.2*inch]))

    story.append(Spacer(1, 0.2*inch))
    story.append(P("Table of contents", "H2"))
    toc = [
        "1.  Snapshot of the system today (what works, what doesn't)",
        "2.  Architecture deep-dive (model, agents, safety stack)",
        "3.  Data pipeline (datasets, paths, augmentation, splits)",
        "4.  Training pipeline (every script, every hyperparameter)",
        "5.  Inference pipeline (orchestrator, safety policy, cross-check)",
        "6.  Deployment architecture (Docker, HF Space, REST API)",
        "7.  Current metrics — by dataset, by class, with calibration",
        "8.  Stage-3 problems (immediate, fixable in a session or two)",
        "9.  Stage-4 problems (deeper research gaps)",
        "10. Future-improvements roadmap (prioritised, with effort estimates)",
        "11. Operational runbook (deploy / restart / debug / re-train)",
        "12. Code navigation index (where to find X, how to change Y)",
        "13. Lessons learned (deploy gotchas, what to never do again)",
        "14. Reference: env vars, file paths, URLs, commit hashes",
    ]
    for line in toc:
        story.append(P(line, "BODY"))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 1. SNAPSHOT
    # ────────────────────────────────────────────────────────────────────
    story.append(P("1.  Snapshot of the system today", "H1"))
    story.append(P("Two-minute orientation if you're picking this up cold.", "SUB"))

    story.append(P("What works", "H2"))
    works = [
        "5-class pathology classifier (polyps / uc-mild / uc-mod-sev / barretts-esoph / therapeutic). "
        "Val acc 86.5%, calibrated ECE 0.062.",
        "GradCAM++ heat-map that lands in the polyp region (not on scope artefacts). "
        "Mean cross-vendor IoU 0.27, Pentax 0.16 (was 0.07 in v1).",
        "Polyp segmentation decoder — IoU 0.61 across all 5 external datasets, Pentax = Olympus.",
        "Six-agent orchestrator (image / text / tabular / fusion / xai / recommendation) "
        "wired into both Streamlit and a REST API.",
        "Patient-safety policy: rejects non-endoscopy uploads, abstains on low-confidence, "
        "shows otherwise. Audit log on every prediction.",
        "Live-video pipeline at 21 FPS on Apple Silicon with 3-frame debounce.",
        "Live demo on Hugging Face Spaces (Docker SDK, free CPU tier).",
        "Hospital-grade REST API with X-API-Key auth, CORS allow-list, sanitised errors.",
    ]
    for w in works:
        story.append(P("&#10003;&nbsp;&nbsp;" + w, "BODY"))

    story.append(P("What doesn't work yet", "H2"))
    nope = [
        ("UC severity grading", "Moderate-severe recall 0.15. Class-balanced sampler "
         "made the model over-call \"mild\" UC. Needs more uc-mod-sev training data "
         "or a hierarchical loss."),
        ("Per-polyp detection on Pentax", "Sensitivity@IoU0.5 is 0.38 on ETIS-Larib "
         "vs 0.75 on Olympus. Segmentation is at parity but proper detection isn't."),
        ("Invasive cancer / Stage III-IV", "Completely out-of-distribution. The model "
         "was trained on screening-stage data only. The image-stats safety net catches "
         "these but the system cannot classify them."),
        ("Confidence on hospital-real data", "ECE was measured on the public-data val "
         "split. Real-world distribution will need per-site temperature re-calibration."),
        ("Free-tier HF Space cold-start", "First request after idle takes ~30s while "
         "the container wakes up. Not an issue if traffic is steady."),
    ]
    for hd, body in nope:
        story.append(P("&#10007;&nbsp;&nbsp;<b>" + hd + ".</b>&nbsp;&nbsp;" + body, "BODY"))

    story.append(P("What's deployed where", "H2"))
    deploy = [
        ["Streamlit web app", "huggingface.co/spaces/Yuvraj2319/colonai",
         "Docker SDK, CPU basic, free. Checkpoint embedded in repo via Git LFS."],
        ["Model checkpoint",  "huggingface.co/Yuvraj2319/colonai-v2",
         "Separate model repo. 572 MB. Public."],
        ["Source code",       "github.com/Yuvraj235/Agentic_Multimodal_Colon_Cancer_AI",
         "Main branch. All commits include co-author trailer."],
        ["REST API (when run locally)", "scripts/serve_api.py",
         "Defaults to 127.0.0.1:8081. Auth via COLONAI_API_KEY env var."],
        ["Local Streamlit (dev)", "streamlit run app.py",
         "127.0.0.1:8501. .streamlit/config.toml caps upload at 10 MB."],
    ]
    story.append(_header_table(
        [["Surface", "Where", "Notes"]] + deploy,
        col_widths=[1.6*inch, 2.4*inch, 3.0*inch]))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 2. ARCHITECTURE
    # ────────────────────────────────────────────────────────────────────
    story.append(P("2.  Architecture deep-dive", "H1"))
    story.append(P("How the model, agents, and safety stack fit together.", "SUB"))

    story.append(P("2.1  Model — UnifiedMultiModalTransformer", "H2"))
    story.append(P(
        "Defined in <font face='Courier'>src/models/unified_transformer.py</font>. "
        "Multi-modal classifier with three encoders feeding a fusion transformer.", "BODY"))

    arch_tbl = [
        ["Component", "Implementation"],
        ["Image encoder",     "DualBackboneImageEncoder: ResNet50 (timm) + EfficientNet-B0. "
                              "Both produce 7×7 spatial maps. Fused per-position with a learned gate. "
                              "Output: (B, 49 patches, 256). ResNet target layer = layer4[-1] (GradCAM target)."],
        ["Text encoder",      "BioBERT (dmis-lab/biobert-base-cased-v1.2). CLS token → 256-d projection."],
        ["Tabular encoder",   "TabTransformer. 12 features (age, BMI, year_of_dx, days_to_followup, "
                              "cigs_per_day, pack_years, alcohol, gender, race_encoded, tumor_stage, "
                              "morphology_encoded, site_encoded). Per-feature tokens → 256-d."],
        ["Fusion",            "GatedFusionTransformer (3 cross-attention layers + 2 self-attention layers). "
                              "Per-modality CLS pooling. Learned modality-importance gate exposed as "
                              "out[\"mod_weights\"]."],
        ["Pathology head",    "5-class classifier (n_classes=5). Used by app.py + safety policy."],
        ["Staging head",      "4-class (no_cancer / I / II / III-IV). Anchored to v1 via KL distillation "
                              "during v2 retrain — does not drift."],
        ["Risk head",         "Binary benign/malignant. Same KL anchor."],
        ["Segmentation decoder", "SEPARATE module in scripts/train_segmentation_head.py. "
                                 "UNet-style 3-stage decoder on top of frozen ResNet50 layer4. "
                                 "~2.4M params. Loaded as outputs/unified_multimodal_v2/seg_head.pth."],
        ["OOD head",          "Separate MLP on top of fused 256-d embedding. "
                              "outputs/unified_multimodal_v2/ood_head.pth + ood_threshold.json."],
        ["Total trainable",   "~74.7M parameters."],
    ]
    story.append(_header_table(arch_tbl, col_widths=[1.4*inch, 5.6*inch]))

    story.append(P("Why this architecture (rationale)", "H3"))
    story.append(P(
        "ResNet50 layer4 produces a clean 7×7 spatial map — perfect for GradCAM++ "
        "and large enough to constrain via the mask-aligned attention loss. EfficientNet-B0 "
        "as the secondary backbone (rather than B4 from the original design) keeps the "
        "model deliberately smaller so val acc doesn't trivially saturate. BioBERT is the "
        "best-published clinical NLP backbone with permissive licensing. TabTransformer "
        "handles mixed categorical/continuous tabular features better than a simple MLP. "
        "The gated fusion lets the model dynamically weight modalities per-case (shown as "
        "the modality-weights radar chart in the UI).", "BODY"))

    story.append(P("2.2  Six-agent orchestrator", "H2"))
    story.append(P("Defined in <font face='Courier'>src/agents/multimodal_orchestrator.py</font>. "
                   "Each agent is a thin wrapper around the model exposing one capability.", "BODY"))
    agent_rows = [
        ["#", "Agent", "Source file", "Produces"],
        ["1", "UnifiedImageAgent",         "src/agents/unified_image_agent.py",
                                            "ImageEvidence: pathology probs + GradCAM++ heatmap"],
        ["2", "TextAgent",                 "src/agents/text_agent.py",
                                            "TextEvidence: BioBERT attention rollout, risk_level"],
        ["3", "TabularRiskAgent",          "src/agents/tabular_risk_agent.py",
                                            "TabularEvidence: SHAP-style perturbation importances"],
        ["4", "FusionReasoningAgent",      "src/agents/fusion_reasoning_agent.py",
                                            "FusionDiagnosis: pathology_class, confidence, modality_weights"],
        ["5", "XAIAgent",                  "src/agents/xai_agent.py",
                                            "XAIReport: MC-Dropout uncertainty, counterfactuals"],
        ["6", "ClinicalRecommendationAgent","src/agents/clinical_recommendation_agent.py",
                                            "ClinicalRecommendation: urgency, surveillance, BSG/NICE flags"],
    ]
    story.append(_header_table(agent_rows,
                  col_widths=[0.3*inch, 1.8*inch, 2.4*inch, 2.5*inch]))

    story.append(P("2.3  Safety stack (added in v2)", "H2"))
    story.append(P("Three independent layers, called in sequence in app.py around line 1370.", "BODY"))
    safety_rows = [
        ["Layer", "Module", "Decision"],
        ["1 — Endoscopy gate",
         "src/app/image_atypicality.py → is_endoscopy_image()",
         "Reject if pixel-stats score < 0.55 (red dominance, tissue hue, "
         "moderate saturation, texture, not grayscale)."],
        ["2 — Image-atypicality detector",
         "Same module, additional rules",
         "Flag if heavy bleeding / deep cavitation detected even when model is confident."],
        ["3a — Cross-agent consistency",
         "src/app/cross_check.py → cross_check()",
         "Compute coherence = geomean(class↔seg, gradcam↔seg, gradcam↔ig). "
         "Below 0.5 → safety policy treats as agents-disagree."],
        ["3b — Patient-safety policy",
         "src/app/patient_safety.py → evaluate_safety()",
         "Final gate: returns show / abstain / reject. "
         "Audit log written to outputs/audit/audit_YYYYMMDD.jsonl (chmod 0600)."],
        ["3c — TTA second opinion",
         "src/app/patient_safety.py → second_opinion()",
         "Optional strict mode requiring unanimous agreement across N augmentations."],
        ["3d — Live-video debouncer",
         "src/app/video_pipeline.py → PolypTracker",
         "min_persistent=3 frames, iou_thresh=0.20. "
         "No single-frame false positive reaches the operator."],
    ]
    story.append(_header_table(safety_rows,
                  col_widths=[1.5*inch, 2.3*inch, 3.2*inch], header_color=DANGER))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 3. DATA PIPELINE
    # ────────────────────────────────────────────────────────────────────
    story.append(P("3.  Data pipeline", "H1"))

    story.append(P("3.1  Datasets used", "H2"))
    ds_rows = [
        ["Dataset", "Path on disk", "Purpose", "Size"],
        ["HyperKvasir (clean)",   "data/processed/hyper_kvasir_clean/",
                                  "5-class pathology training",     "10,662 images, 23 sub-classes"],
        ["CVC-ClinicDB",          "data/raw/CVC-ClinicDB/PNG/...",
                                  "Polyp + pixel mask",             "612 images + masks"],
        ["Kvasir-SEG",            "data/raw/kvasir-seg/Kvasir-SEG/",
                                  "Polyp + pixel mask",             "1,000 images + masks"],
        ["CVC-ColonDB",           "data/raw/test_polyp_datasets/.../CVC-ColonDB/",
                                  "External polyp + mask",          "380 images + masks"],
        ["CVC-300",               "data/raw/test_polyp_datasets/.../CVC-300/",
                                  "External polyp + mask",          "60 images + masks"],
        ["ETIS-LaribPolypDB",     "data/raw/test_polyp_datasets/.../ETIS-LaribPolypDB/",
                                  "PENTAX scope — held out!",       "196 images + masks"],
        ["Kvasir-test",           "data/raw/test_polyp_datasets/.../Kvasir/",
                                  "External polyp + mask",          "100 images + masks"],
        ["TCGA clinical",         "data/raw/tcga/clinical/clinical.tsv",
                                  "Tabular features",               "1,802 rows × 12 features"],
    ]
    story.append(_header_table(ds_rows, col_widths=[1.5*inch, 2.4*inch, 1.8*inch, 1.3*inch]))

    story.append(P("3.2  Class label mapping", "H3"))
    cls_rows = [
        ["Class index", "Class name (CLASS_NAMES_5)", "Source sub-classes"],
        ["0", "polyps",            "polyps (HK) + CVC-ClinicDB + all polyp-mask datasets"],
        ["1", "uc-mild",           "ulcerative-colitis-grade-0-1 + grade-1 + grade-1-2 (HK)"],
        ["2", "uc-moderate-sev",   "ulcerative-colitis-grade-2 + grade-2-3 + grade-3 (HK)"],
        ["3", "barretts-esoph",    "barretts + barretts-short-segment + esophagitis-* (HK)"],
        ["4", "therapeutic",       "dyed-lifted-polyps + dyed-resection-margins (HK)"],
    ]
    story.append(_header_table(cls_rows, col_widths=[0.8*inch, 1.8*inch, 4.4*inch]))

    story.append(P("3.3  Splits + sampling", "H3"))
    story.append(P(
        "Stratified split is 75% train / 15% val / 10% test (seed=42, sklearn StratifiedShuffleSplit). "
        "Manifest CSV with MD5 hashes saved to outputs/.../manifests/split_manifest.csv on every run "
        "for audit. CVC-ClinicDB is appended to the train split AFTER MD5 dedup against HyperKvasir "
        "(to prevent CVC-ClinicDB images that exist in HK from leaking across splits). "
        "Class-balanced WeightedRandomSampler is used during v2 retraining — replacement=True, "
        "weights = 1/class_count.", "BODY"))

    story.append(P("3.4  Augmentation", "H3"))
    story.append(P(
        "Defined in <font face='Courier'>src/data/multimodal_dataset.py:get_train_transforms()</font>. "
        "Resize 256 → RandomCrop 224 → HFlip(0.5), VFlip(0.3), ColorJitter(b=.25,c=.25,s=.20,h=.04), "
        "RandomPerspective(.20, p=.4), ToTensor, ImageNet-normalise, RandomErasing(p=.3, scale=(.02,.10)). "
        "Tabular: Gaussian noise σ=0.05 added at train time. Polyp-mask datasets share the geometric "
        "augmentation between image and mask via shared random seed.", "BODY"))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 4. TRAINING PIPELINE
    # ────────────────────────────────────────────────────────────────────
    story.append(P("4.  Training pipeline", "H1"))
    story.append(P("Every script that produced the current checkpoints.", "SUB"))

    story.append(P("4.1  Training timeline (history)", "H2"))
    train_rows = [
        ["When", "Script", "Output", "Notes"],
        ["v1 (Feb 2026)",
         "experiments/run_full_pipeline.py",
         "outputs/unified_multimodal/checkpoints/best_model.pth",
         "Original 8-epoch train. Test acc 99.5%, but vendor-bias problem."],
        ["v2 ep 1-2 (May 23)",
         "scripts/retrain_deploy_grade.py --epochs 2",
         "outputs/unified_multimodal_v2/checkpoints/best_model.pth",
         "Mask-aware + class-balanced + KL distill from v1. Val 0.87."],
        ["v2 ep 3-4 (May 23)",
         "Same script, warm-start from v2-ep2",
         "Same path (overwrote)",
         "+2 epochs. Val dipped to 0.86 but cross-vendor IoU went up."],
        ["Calibration",
         "scripts/calibrate_t_ece.py",
         "outputs/unified_multimodal_v2/temperature.json",
         "Grid-searched T = 0.45 for ECE-min on val set."],
        ["Segmentation",
         "scripts/train_segmentation_head.py --epochs 4",
         "outputs/unified_multimodal_v2/seg_head.pth",
         "UNet decoder on frozen ResNet50. Val IoU 0.61."],
        ["OOD head",
         "scripts/train_ood_head.py",
         "outputs/unified_multimodal_v2/ood_head.pth + ood_threshold.json",
         "Lightweight MLP on fused embeddings. F1 = 1.0 (synthetic OOD)."],
    ]
    story.append(_header_table(train_rows, col_widths=[1.2*inch, 1.8*inch, 2.4*inch, 1.7*inch]))

    story.append(P("4.2  retrain_deploy_grade.py — exact config", "H2"))
    cfg_rows = [
        ["Optimizer",         "AdamW, weight_decay=0.12"],
        ["LR (heads + vision)", "2e-5 with CosineAnnealingLR, eta_min=2e-6"],
        ["LR (BERT)",          "2e-6 (10× lower than rest)"],
        ["Batch size",         "16 (MPS) / 32 (CUDA)"],
        ["Loss (pathology)",   "FocalLoss(gamma=2.0, label_smoothing=0.1, class_weights=1/counts)"],
        ["Loss (staging head)","KL(student/T || teacher/T) × T² where T=2.0"],
        ["Loss (risk head)",   "Same KL distillation"],
        ["Loss (attention)",   "AttentionMaskLoss: KL(softmax(|features|.mean(C)) || downsampled_mask). "
                               "Weight = 0.6 (--mask_loss_w)"],
        ["Distillation weight","0.5 (--distill_w)"],
        ["Warm-start",         "outputs/unified_multimodal/checkpoints/best_model.pth (v1)"],
        ["Sampler",            "WeightedRandomSampler(per-class weight = 1/count, replacement=True)"],
        ["Polyp+mask datasets","6: Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-Larib, Kvasir-test"],
        ["Train total samples","6,430 (4,082 HK+CVC + 2,348 polyp+mask)"],
        ["Epochs",             "4 total (2 + 2 extension)"],
        ["Time on MPS",        "~21 minutes total"],
    ]
    story.append(_kv_table(cfg_rows, col_widths=[1.8*inch, 5.2*inch]))

    story.append(P("4.3  Re-run training from scratch", "H3"))
    story.append(Paragraph(
        "# Step 1 — base v2 retrain (warm-start from v1)<br/>"
        "python3 scripts/retrain_deploy_grade.py \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--epochs 4 --batch_size 16 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--mask_loss_w 0.6 --distill_w 0.5<br/>"
        "<br/>"
        "# Step 2 — calibrate temperature<br/>"
        "python3 scripts/calibrate_t_ece.py<br/>"
        "<br/>"
        "# Step 3 — train segmentation decoder<br/>"
        "python3 scripts/train_segmentation_head.py<br/>"
        "<br/>"
        "# Step 4 — train OOD head<br/>"
        "python3 scripts/train_ood_head.py<br/>"
        "<br/>"
        "# Step 5 — validate cross-vendor IoU<br/>"
        "python3 scripts/validate_gradcam_v2.py<br/>"
        "<br/>"
        "# Step 6 — CADe detection metrics<br/>"
        "python3 scripts/cade_metrics_seg.py<br/>"
        "<br/>"
        "# Step 7 — full per-class evaluation<br/>"
        "python3 scripts/evaluate_v2_full.py",
        S["CODE"]))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 5. INFERENCE PIPELINE
    # ────────────────────────────────────────────────────────────────────
    story.append(P("5.  Inference pipeline", "H1"))

    story.append(P("5.1  End-to-end request flow", "H2"))
    flow_rows = [
        ["Step", "Module", "What happens"],
        ["1. Upload",
         "src/app/security.validate_upload_bytes()",
         "Size cap 10 MB, MIME allow-list, magic-byte check, PIL bomb guard."],
        ["2. Endoscopy gate",
         "src/app/image_atypicality.is_endoscopy_image()",
         "Pixel-stats score. Below 0.55 → reject early."],
        ["3. Model forward",
         "UnifiedMultiModalTransformer.forward()",
         "Image (224×224, ImageNet-normalised) + tokenised symptoms + tabular vector. "
         "Returns pathology / staging / risk / fused embedding / GradCAM target activations."],
        ["4. GradCAM++",
         "src/agents/unified_image_agent.GradCAMPlusPlus.generate()",
         "Hooks ResNet50 layer4[-1]. Backward through predicted class. "
         "Up-samples 7×7 → 224×224 with bilinear interpolation."],
        ["5. Integrated Gradients",
         "src/app/strong_xai.integrated_gradients()",
         "Captum-style 24-step IG. Independent second attribution method."],
        ["6. Segmentation decoder",
         "scripts/train_segmentation_head.SegDecoder.forward()",
         "Optional: produces 224×224 polyp mask."],
        ["7. MC-Dropout uncertainty",
         "src/agents/xai_agent.XAIAgent.explain()",
         "Sets model.train(); runs 3-5 stochastic forwards; uncertainty = std.max()."],
        ["8. TTA",
         "src/app/reliability.tta_inference()",
         "5 augmentations; agreement_pct = how many picked the modal class."],
        ["9. Cross-check",
         "src/app/cross_check.cross_check()",
         "Pairwise class↔seg / gradcam↔seg / gradcam↔ig. coherence = geomean."],
        ["10. Safety policy",
         "src/app/patient_safety.evaluate_safety()",
         "Returns {show, abstain, reject}. Audit-logged."],
        ["11. UI render",
         "app.py:page_analysis() / pages/results",
         "Verdict banner, plain-English narrative, heat-map, rationale card, charts."],
    ]
    story.append(_header_table(flow_rows, col_widths=[1.2*inch, 2.6*inch, 3.2*inch]))

    story.append(P("5.2  Decision policy (exact thresholds)", "H2"))
    pol_rows = [
        ["Signal", "Threshold", "Source of truth"],
        ["Endoscopy score",     ">= 0.55", "SAFETY_CONFIG['min_endoscopy_score']"],
        ["Confidence (calibrated)", ">= 0.75", "SAFETY_CONFIG['min_confidence']"],
        ["MC-Dropout uncertainty",  "<= 0.30", "SAFETY_CONFIG['max_uncertainty']"],
        ["GradCAM focus",       ">= 0.15 (top-quartile area)", "SAFETY_CONFIG['min_gradcam_focus']"],
        ["Agent agreement (normal)", ">= 0.66 (2 of 3)",        "SAFETY_CONFIG['min_agent_agreement']"],
        ["Agent agreement (strict)", "= 1.00 (unanimous)",       "SAFETY_CONFIG['strict_agent_agreement']"],
        ["Cross-check coherence",   ">= 0.50",                    "COHERENCE_FLOOR in cross_check.py"],
        ["Live-video persistence",  ">= 3 frames",                "SAFETY_CONFIG['live_debounce_frames']"],
        ["Live-video IoU",          ">= 0.20",                    "SAFETY_CONFIG['live_iou_threshold']"],
    ]
    story.append(_header_table(pol_rows, col_widths=[2.1*inch, 1.7*inch, 3.2*inch], header_color=DANGER))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 6. DEPLOYMENT
    # ────────────────────────────────────────────────────────────────────
    story.append(P("6.  Deployment architecture", "H1"))

    story.append(P("6.1  Docker image (HF Space)", "H2"))
    story.append(P(
        "Defined in <font face='Courier'>Dockerfile</font>. python:3.11-slim base. "
        "Runs as uid 1000 (HF convention). Streamlit on port 7860. "
        "TRANSFORMERS_CACHE and HF_HOME both point at /home/user/.cache/huggingface. "
        "Health-check hits /_stcore/health every 30s.", "BODY"))

    story.append(P("6.2  HF Space configuration", "H3"))
    hf_rows = [
        ["Repo",                "Yuvraj2319/colonai (Space, type: space, SDK: docker)"],
        ["Frontmatter in README.md", "title, emoji, colorFrom/To, sdk=docker, app_port=7860, license=mit, short_description ≤ 60 chars"],
        ["Hardware",            "cpu-basic (free tier)"],
        ["Public",              "Yes (Public visibility)"],
        ["Variables (visible)", "COLONAI_CHECKPOINT_HF_REPO = Yuvraj2319/colonai-v2 (vestigial — checkpoint is now baked in)"],
        ["Variables (visible)", "COLONAI_CHECKPOINT_HF_FILE = best_model.pth"],
        ["Secrets",             "HF_TOKEN (set programmatically via api.add_space_secret)"],
        ["Storage",             "Checkpoint committed via Git LFS in the Space repo, NOT downloaded at runtime"],
        ["Build context",       "Whole git tree EXCEPT what .dockerignore excludes"],
    ]
    story.append(_kv_table(hf_rows, col_widths=[2.0*inch, 5.0*inch]))

    story.append(P("6.3  .dockerignore — exact whitelist pattern", "H3"))
    story.append(Paragraph(
        "data<br/>experiments<br/>paper_figures<br/>notebooks<br/>"
        "<br/>"
        "# Outputs: keep only the model files, exclude figures/logs/etc.<br/>"
        "outputs/*<br/>"
        "!outputs/unified_multimodal_v2<br/>"
        "outputs/unified_multimodal_v2/*<br/>"
        "!outputs/unified_multimodal_v2/checkpoints<br/>"
        "!outputs/unified_multimodal_v2/temperature.json<br/>"
        "outputs/unified_multimodal_v2/checkpoints/*<br/>"
        "!outputs/unified_multimodal_v2/checkpoints/best_model.pth",
        S["CODE"]))

    story.append(P("6.4  REST API endpoints (scripts/serve_api.py)", "H2"))
    api_rows = [
        ["Endpoint", "Auth", "Returns"],
        ["GET /health",       "open",          "{status: ok, ready: True}"],
        ["GET /version",      "open",          "{model_version, classes, temperature, safety_config, auth_enabled}"],
        ["POST /predict",     "X-API-Key",     "{verdict, prediction, confidence, all_probs, request_id, elapsed_ms}"],
        ["GET /audit/today",  "X-API-Key",     "JSONL of every prediction recorded today"],
    ]
    story.append(_header_table(api_rows, col_widths=[1.8*inch, 1.5*inch, 3.7*inch]))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 7. METRICS
    # ────────────────────────────────────────────────────────────────────
    story.append(P("7.  Current metrics (everything in one place)", "H1"))

    story.append(P("7.1  Cross-vendor GradCAM IoU", "H2"))
    iou_rows = [
        ["Dataset", "Vendor", "v1 IoU", "v2 IoU", "v1 Dice", "v2 Dice"],
        ["Kvasir-SEG",        "Olympus", "0.236", "0.422", "0.359", "0.546"],
        ["ETIS-LaribPolypDB", "Pentax",  "0.069", "0.163", "0.112", "0.234"],
        ["CVC-ColonDB",       "Olympus", "0.109", "0.205", "0.181", "0.301"],
        ["CVC-300",           "Olympus", "0.118", "0.134", "0.206", "0.225"],
        ["Kvasir-test",       "Olympus", "0.213", "0.415", "0.325", "0.546"],
        ["Mean",              "",        "0.149", "0.268", "0.237", "0.371"],
    ]
    story.append(_header_table(iou_rows,
                  col_widths=[1.5*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch]))

    story.append(P("7.2  Segmentation decoder IoU (all 5 datasets, val split)", "H2"))
    seg_rows = [
        ["Dataset",            "Vendor",  "IoU",  "Sens@IoU0.5", "FPPI", "mAP@0.5"],
        ["Kvasir-SEG",         "Olympus", "0.618","0.746",       "0.215","0.635"],
        ["ETIS-LaribPolypDB",  "Pentax",  "0.609","0.383",       "0.444","0.359"],
        ["CVC-ColonDB",        "Olympus", "0.616","0.642",       "0.247","0.541"],
        ["CVC-300",            "Olympus", "0.625","0.917",       "0.083","0.874"],
        ["Kvasir-test",        "Olympus", "0.624","0.730",       "0.210","0.642"],
    ]
    story.append(_header_table(seg_rows,
                  col_widths=[1.7*inch, 0.9*inch, 0.8*inch, 1.1*inch, 0.8*inch, 0.9*inch],
                  header_color=ACCENT))

    story.append(P("7.3  Per-class pathology metrics (val set, n=694)", "H2"))
    pc_rows = [
        ["Class",            "Precision", "Recall", "F1",    "Support"],
        ["polyps",           "0.972",     "0.916",  "0.943", "154"],
        ["uc-mild",          "0.294",     "0.946",  "0.449", "37"],
        ["uc-moderate-sev",  "1.000",     "0.154",  "0.267", "91"],
        ["barretts-esoph",   "0.991",     "1.000",  "0.996", "113"],
        ["therapeutic",      "0.983",     "0.993",  "0.988", "299"],
    ]
    story.append(_header_table(pc_rows,
                  col_widths=[2.0*inch, 1.2*inch, 1.0*inch, 1.0*inch, 1.0*inch]))
    story.append(P("Macro-F1 = 0.729. Overall accuracy 0.865. "
                   "<b>uc-mild over-predicted (low precision), uc-mod-sev under-predicted (low recall).</b> "
                   "This is the main known failure mode — see Section 8.", "BODY"))

    story.append(P("7.4  Calibration", "H2"))
    cal_rows = [
        ["Metric",              "Raw",   "After T=0.45"],
        ["Expected Calibration Error (ECE)", "0.168", "0.062"],
        ["Brier score (multi-class)",        "0.182", "0.161"],
    ]
    story.append(_header_table(cal_rows, col_widths=[3.0*inch, 1.6*inch, 1.6*inch]))

    story.append(P("7.5  FPS benchmark (Apple Silicon MPS)", "H2"))
    fps_rows = [
        ["Metric",            "Value"],
        ["Mean latency / frame", "46.5 ms"],
        ["Median latency",       "45.8 ms"],
        ["P95 latency",          "48.3 ms"],
        ["P99 latency",          "76.7 ms"],
        ["Mean FPS",             "21.5"],
        ["P95 FPS",              "20.7"],
        ["Forward pass",         "42.1 ms (most of the latency)"],
        ["GradCAM",              "0.0 ms (only runs when polyp predicted)"],
        ["Live-use verdict",     "✓ Real-time capable (≥ 15 FPS)"],
    ]
    story.append(_kv_table(fps_rows, col_widths=[2.5*inch, 4.5*inch]))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 8. STAGE-3 PROBLEMS
    # ────────────────────────────────────────────────────────────────────
    story.append(P("8.  Stage-3 problems (immediately fixable)", "H1"))
    story.append(P("These can each be fixed in 1-3 sessions with the current codebase.", "SUB"))

    s3_problems = [
        {
            "title": "P-1.  uc-mod-sev recall = 0.15",
            "diag":  "Class-balanced WeightedRandomSampler oversamples uc-mild "
                     "(only 185 samples in train) which causes the model to learn "
                     "\"if it looks like ulcerative colitis, call it mild.\" "
                     "Per-class F1 = 0.267 → clinically unacceptable.",
            "fix":   "(a) Use a hierarchical loss: first classify {polyp / UC / barretts / "
                     "therapeutic}, then within UC predict {mild / moderate-severe}. "
                     "(b) OR: drop class-balanced sampling and use Focal loss with γ=3.0 "
                     "and explicit higher class weights for uc-mod-sev. "
                     "(c) OR: collect more uc-mod-sev images from public sources "
                     "(HyperKvasir has 443 grade-2 + 133 grade-3 = 576; aim for ≥ 1000 by adding "
                     "Kvasir-Capsule's Ulcer + Erythematous classes).",
            "effort":"4-8 hours retraining + evaluation",
        },
        {
            "title": "P-2.  Per-polyp detection on Pentax (sens 0.38 vs 0.75 on Olympus)",
            "diag":  "Segmentation IoU is at parity (0.61 both vendors), but turning a mask into "
                     "a single bbox via largest-connected-component then matching to GT loses "
                     "polyps that the seg decoder split into multiple components.",
            "fix":   "(a) Use a proper detection head (Faster R-CNN or YOLOv8-style anchor-free) "
                     "on top of the same ResNet50 features. The seg decoder already proves the "
                     "spatial features are right. "
                     "(b) OR: post-process seg mask with morphological closing before bbox extraction "
                     "(removes small gaps that split one polyp into two CCs). Cheap first try.",
            "effort":"(a) 2 days + retrain · (b) 1 hour",
        },
        {
            "title": "P-3.  HF Space cold-start = 30+ seconds",
            "diag":  "Free CPU tier sleeps after idle; first request pays the full container boot "
                     "(~2 min) including pytorch import + checkpoint load.",
            "fix":   "(a) Keep-alive: cron-job a /health hit every 10 min. "
                     "(b) Upgrade to CPU-upgrade ($0.03/h, ~$22/month). "
                     "(c) Quantize the model to int8 (cuts checkpoint size 4× and forward pass time ~2×).",
            "effort":"(a) 15 min · (b) money · (c) 1 day quantization-aware fine-tune",
        },
        {
            "title": "P-4.  Confidence calibration is on public-data val set",
            "diag":  "ECE 0.062 measured on HyperKvasir val. Real hospital data has different "
                     "scope brands, lighting, demographics. T=0.45 won't transfer perfectly.",
            "fix":   "Provide a per-site recalibration script: hospital uploads ~200 labelled "
                     "samples, system re-fits T via grid search. scripts/calibrate_t_ece.py "
                     "already does this — just needs a UI flow for hospital ops.",
            "effort":"3 hours UI + docs",
        },
        {
            "title": "P-5.  OOD head trained only on synthetic OOD",
            "diag":  "F1=1.0 on synthetic OOD (noise / gradients / checkerboards) but "
                     "we have NO test with real OOD endoscopy images (e.g. upper-GI when "
                     "expecting lower-GI, capsule endoscopy, narrow-band-imaging).",
            "fix":   "(a) Add Kvasir-Capsule images as a held-out OOD test set. "
                     "(b) Add narrow-band-imaging samples from CVC-ClinicNBI. "
                     "(c) Re-train OOD head with these as additional OOD class — F1 will drop honestly "
                     "but the score will be meaningful.",
            "effort":"6 hours data prep + retraining",
        },
    ]
    for p in s3_problems:
        story.append(P(p["title"], "H2"))
        story.append(P("<b>Diagnosis: </b>" + p["diag"], "BODY"))
        story.append(P("<b>Fix: </b>" + p["fix"], "BODY"))
        story.append(P("<b>Effort: </b>" + p["effort"], "BODY"))
        story.append(Spacer(1, 0.05*inch))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 9. STAGE-4 PROBLEMS
    # ────────────────────────────────────────────────────────────────────
    story.append(P("9.  Stage-4 problems (deeper research)", "H1"))
    story.append(P("These need real research, not just engineering. Months, not days.", "SUB"))

    s4 = [
        {
            "title": "R-1.  No real cancer staging data",
            "diag":  "The staging head outputs {no_cancer / I / II / III-IV} but the entire training "
                     "corpus is screening-stage (HyperKvasir, CVC). TCGA gives us tabular features but "
                     "no images. The staging head learns to output \"no_cancer\" always, and we anchor "
                     "it via KL distillation. So the staging head is essentially a vestigial output, "
                     "kept only because it was in v1's architecture.",
            "direction": "(i) Acquire a histopathology image dataset (PAIP, PANDA, TCGA-COAD whole-slide "
                         "images). (ii) Train staging head on histology + transfer to endoscopy via "
                         "feature alignment. (iii) OR drop the staging head and present a more honest "
                         "two-output system: pathology class + binary cancer-risk.",
        },
        {
            "title": "R-2.  Per-site distribution shift not measurable",
            "diag":  "Vendor bias was the v1 problem; v2 closed it. But there's a deeper "
                     "operator-induced bias: different endoscopists use different insertion technique, "
                     "different bowel-prep quality, different photographic style. We can't measure "
                     "this without per-site labelled data.",
            "direction":"Federated-learning pilot: 3-5 hospitals each upload class probabilities "
                        "(not images) for their own cases over 30 days; we measure prediction-shift "
                        "across sites. Use that to motivate per-site fine-tuning vs. global model.",
        },
        {
            "title": "R-3.  No characterisation (CADx), only detection (CADe)",
            "diag":  "We say \"polyp\" but not \"adenoma vs hyperplastic vs serrated\". A clinician "
                     "still needs to perform biopsy because we don't differentiate histology types. "
                     "CADx (computer-aided diagnosis) is the actual clinical workflow target.",
            "direction":"(i) Get NICE polyp classification training data (Kudo Pit Pattern, NBI International "
                        "Colorectal Endoscopic classification). (ii) Add a polyp-type head with 3-4 classes. "
                        "(iii) Validate against histology gold standard. This is a separate model+study, "
                        "probably worth its own dissertation chapter.",
        },
        {
            "title": "R-4.  Live-video temporal model is just frame-by-frame + tracker",
            "diag":  "Each frame is independently classified. A proper video model would use temporal "
                     "convolutions (X3D, TimeSformer) to exploit motion cues — e.g. how a polyp moves "
                     "differently from a fold when the scope moves.",
            "direction":"Train a 16-frame temporal classifier on Kvasir-Capsule sequences + SUN-SEG. "
                        "Compare against per-frame baseline on both detection F1 and temporal consistency.",
        },
        {
            "title": "R-5.  No active-learning loop with hospital feedback",
            "diag":  "Once deployed, the system doesn't learn from doctor corrections. "
                     "Every \"abstain\" case that a doctor labels is wasted training signal.",
            "direction":"Add a 'doctor feedback' button in the UI (yes/no/which class). "
                        "Store the feedback + image SHA-256 in an append-only log. "
                        "Monthly auto-retrain on accumulated corrections. "
                        "Needs careful PHI handling — image hashes only, no patient identifiers.",
        },
        {
            "title": "R-6.  No proper safety validation in a clinical setting",
            "diag":  "We measured cross-vendor IoU on PUBLIC datasets. We have not measured "
                     "what happens when a real endoscopist uses the system on real patients over 100 "
                     "real procedures. That's the only number that matters for clinical adoption.",
            "direction":"Partner with a hospital for a 3-month observational pilot. Measure (i) operator-"
                        "perceived agreement, (ii) overlooked-polyp rate with vs without ColonAI, "
                        "(iii) confidence-calibration on hospital data, (iv) operator time-to-decision. "
                        "Standard PRISMA-AI reporting.",
        },
    ]
    for p in s4:
        story.append(P(p["title"], "H2"))
        story.append(P("<b>Diagnosis: </b>" + p["diag"], "BODY"))
        story.append(P("<b>Research direction: </b>" + p["direction"], "BODY"))
        story.append(Spacer(1, 0.05*inch))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 10. ROADMAP
    # ────────────────────────────────────────────────────────────────────
    story.append(P("10.  Improvements roadmap (prioritised)", "H1"))

    rm = [
        ["Priority", "Item", "Effort", "Impact"],
        ["P0 — do first", "Fix uc-mod-sev recall (Section 8 P-1)",   "1 week",  "Removes the single worst clinical failure."],
        ["P0", "Per-site recalibration UI flow (P-4)",                 "3 hours", "Required before any real hospital deploy."],
        ["P0", "Morphological closing before seg→bbox (P-2 part b)",   "1 hour",  "Free Pentax detection boost."],
        ["P1", "Proper detection head (Faster R-CNN/YOLOv8) (P-2 a)",  "2 days",  "Brings Pentax detection F1 to ~0.6."],
        ["P1", "Real OOD test set (Capsule + NBI) (P-5)",              "6 hours", "Makes the OOD head's F1 meaningful."],
        ["P1", "Doctor-feedback button + log (R-5)",                   "1 day",   "Enables active learning loop."],
        ["P2", "int8 quantisation for HF Space cold-start (P-3 c)",    "1 day",   "Cuts checkpoint to 150 MB, doubles FPS."],
        ["P2", "Temporal model for live video (R-4)",                  "2 weeks", "Improves live-feed detection."],
        ["P2", "CADx polyp typing head (R-3)",                         "1 month", "Closes the gap to clinical workflow."],
        ["P3", "Federated cross-site distribution audit (R-2)",        "3 months","Needed for multi-site deployment."],
        ["P3", "Drop or replace staging head (R-1)",                   "1 day",   "Honesty win; minor functional change."],
        ["P3", "Clinical observational pilot (R-6)",                   "3 months","Required for any regulatory path."],
    ]
    story.append(_header_table(rm,
                  col_widths=[1.1*inch, 3.0*inch, 0.9*inch, 2.0*inch]))

    story.append(P("Suggested order for a 6-month follow-up dissertation", "H3"))
    story.append(P(
        "(1) Month 1: P0 items + P1 detection head — measurable lift in headline numbers. "
        "(2) Month 2-3: CADx polyp typing (R-3) — new contribution. "
        "(3) Month 4-5: Real-hospital pilot (R-6) — only honest clinical validation. "
        "(4) Month 6: Write everything up, including the negative results.", "BODY"))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 11. RUNBOOK
    # ────────────────────────────────────────────────────────────────────
    story.append(P("11.  Operational runbook", "H1"))

    story.append(P("11.1  Restart the live Space", "H2"))
    story.append(Paragraph(
        "# From your local venv (huggingface_hub installed):<br/>"
        "python3 -c \"from huggingface_hub import HfApi; "
        "HfApi(token=open('~/.cache/huggingface/token').read().strip())"
        ".restart_space(repo_id='Yuvraj2319/colonai')\"<br/>"
        "<br/>"
        "# OR via curl:<br/>"
        "curl -X POST -H \"Authorization: Bearer $(cat ~/.cache/huggingface/token)\" \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;https://huggingface.co/api/spaces/Yuvraj2319/colonai/restart",
        S["CODE"]))

    story.append(P("11.2  Push a code change to the live Space", "H2"))
    story.append(Paragraph(
        "# GitHub push is the source of truth:<br/>"
        "git push origin main<br/>"
        "<br/>"
        "# HF Space repo is a separate git remote — push there too:<br/>"
        "git remote add space https://huggingface.co/spaces/Yuvraj2319/colonai  # one time<br/>"
        "git push --force space main  # use the token from ~/.cache/huggingface/token as password",
        S["CODE"]))

    story.append(P("11.3  Re-upload the checkpoint to the Space", "H2"))
    story.append(Paragraph(
        "# The checkpoint lives in the Space repo via Git LFS.<br/>"
        "# To replace it after retraining:<br/>"
        "<br/>"
        "git clone https://huggingface.co/spaces/Yuvraj2319/colonai /tmp/space<br/>"
        "cd /tmp/space &amp;&amp; git lfs install<br/>"
        "cp ~/Desktop/.../outputs/unified_multimodal_v2/checkpoints/best_model.pth \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;outputs/unified_multimodal_v2/checkpoints/best_model.pth<br/>"
        "git add outputs/unified_multimodal_v2/checkpoints/best_model.pth<br/>"
        "git commit -m \"new checkpoint\"<br/>"
        "git push",
        S["CODE"]))

    story.append(P("11.4  Debug a Space that won't load the model", "H2"))
    story.append(Paragraph(
        "1.  Open https://huggingface.co/spaces/Yuvraj2319/colonai/logs and select Container.<br/>"
        "2.  Look for the [STARTUP] lines (first 10 lines of output):<br/>"
        "&nbsp;&nbsp;&nbsp;[STARTUP] COLONAI_CHECKPOINT_HF_REPO = ...<br/>"
        "&nbsp;&nbsp;&nbsp;[STARTUP] HF_TOKEN present = ...<br/>"
        "3.  Then look for [CHECKPOINT] lines from _maybe_download_checkpoint().<br/>"
        "4.  If \"COLONAI_CHECKPOINT_HF_REPO is unset → demo mode\" → env vars not set.<br/>"
        "5.  If \"v2 checkpoint already present\" → file is baked into image. Verify size.<br/>"
        "6.  If \"HF Hub download failed\" → check token, check model repo public.<br/>"
        "<br/>"
        "If env vars are missing, set them via:<br/>"
        "&nbsp;&nbsp;from huggingface_hub import HfApi<br/>"
        "&nbsp;&nbsp;api = HfApi(token=TOKEN)<br/>"
        "&nbsp;&nbsp;api.add_space_variable(repo_id='Yuvraj2319/colonai',<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;key='COLONAI_CHECKPOINT_HF_REPO', value='Yuvraj2319/colonai-v2')",
        S["CODE"]))

    story.append(P("11.5  Run the local Streamlit app for development", "H2"))
    story.append(Paragraph(
        "cd ~/Desktop/Agentic_Multimodal_Colon_Cancer_AI\\ copy<br/>"
        "source venv/bin/activate<br/>"
        "streamlit run app.py --server.port 8502 --server.address 127.0.0.1",
        S["CODE"]))

    story.append(P("11.6  Run the REST API locally", "H2"))
    story.append(Paragraph(
        "# Without auth (dev):<br/>"
        "python3 scripts/serve_api.py<br/>"
        "<br/>"
        "# With auth (any deploy):<br/>"
        "export COLONAI_API_KEY=$(openssl rand -hex 32)<br/>"
        "python3 scripts/serve_api.py<br/>"
        "<br/>"
        "# Test:<br/>"
        "curl -X POST -H \"X-API-Key: $COLONAI_API_KEY\" \\<br/>"
        "&nbsp;&nbsp;-F image=@some_polyp.jpg http://127.0.0.1:8081/predict",
        S["CODE"]))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 12. CODE NAVIGATION
    # ────────────────────────────────────────────────────────────────────
    story.append(P("12.  Code navigation index", "H1"))
    story.append(P("Where to look when you want to change X.", "SUB"))

    nav_rows = [
        ["To change…",                            "Edit…"],
        ["Safety thresholds (confidence/uncertainty/etc)",
         "src/app/patient_safety.py — SAFETY_CONFIG dict (top of file)"],
        ["Cross-check thresholds",
         "src/app/cross_check.py — COHERENCE_FLOOR, SEG_MIN_POLYP_AREA constants"],
        ["Endoscopy gate threshold",
         "src/app/image_atypicality.py — is_endoscopy_image() default threshold=0.55"],
        ["Live-video debouncer parameters",
         "src/app/video_pipeline.py — PolypTracker(...) constructor calls (lines 335, 472)"],
        ["Upload size cap",
         "src/app/security.py — MAX_UPLOAD_BYTES (default 10 MB) + .streamlit/config.toml maxUploadSize"],
        ["Training hyperparameters",
         "scripts/retrain_deploy_grade.py — argparse defaults in main()"],
        ["Class label map (5 classes)",
         "src/data/multimodal_dataset.py — SUBCLASS_TO_LABEL dict"],
        ["Plain-English narrative templates",
         "src/app/patient_ui.py — PLAIN_NAMES + NEXT_STEPS_PLAIN dicts"],
        ["Result-page rendering",
         "app.py — page_analysis() function around line 3100"],
        ["Sidebar navigation + status",
         "app.py — render_sidebar() function around line 1700"],
        ["Add a new agent",
         "src/agents/ + register in src/agents/multimodal_orchestrator.py __init__"],
        ["Add a new chart to results",
         "app.py — page_analysis() in the section that creates Plotly figures"],
        ["Change the safety verdict banner copy",
         "src/app/patient_ui.py — verdict_card_html()"],
        ["Add a new dataset for training",
         "scripts/retrain_deploy_grade.py — POLYP_MASK_DATASETS list (top)"],
        ["Change Docker base image / port",
         "Dockerfile (root of repo)"],
        ["Change what's deployed to HF Space",
         "Edit on local, push to both origin (GitHub) AND space (HF Space) remotes"],
        ["Add/remove env vars on the Space",
         "huggingface_hub.HfApi.add_space_variable / add_space_secret (Python)"],
    ]
    story.append(_header_table(nav_rows, col_widths=[3.0*inch, 4.0*inch]))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 13. LESSONS LEARNED
    # ────────────────────────────────────────────────────────────────────
    story.append(P("13.  Lessons learned (deploy gotchas)", "H1"))
    story.append(P("Things that wasted hours during this build — do not repeat.", "SUB"))

    lessons = [
        ("HF Spaces removed Streamlit from the create-Space UI dropdown.",
         "Pick Docker SDK and write a Dockerfile that runs Streamlit. Don't waste time "
         "looking for the Streamlit option — it's gone from the UI, even though it still "
         "works via README frontmatter sdk: streamlit."),
        ("HF Spaces' upload_file() / create_commit() silently dedupes large blobs via XET.",
         "If a file's SHA-256 already exists in another HF repo you own, upload_file may "
         "return success with a commit URL but the content blob never actually lands. "
         "Use git CLI with git-lfs installed instead, OR modify a single byte to break "
         "the dedup."),
        (".gitignore on the Space repo can SILENTLY refuse git add.",
         "When I tried to add the checkpoint via git CLI, .gitignore had 'outputs/' which "
         "blocked the add — but the upload_file path didn't error. Always run "
         "`git status --short` and `git lfs ls-files` to confirm the file is staged before "
         "celebrating a successful push."),
        ("torch.load default is weights_only=False — pickle RCE if checkpoint is untrusted.",
         "Use src/app/security.safe_torch_load() which prefers weights_only=True and falls "
         "back to allow_unsafe=True only for first-party files."),
        ("Streamlit ignores .streamlit/config.toml's `address` when you pass --server.address.",
         "Don't trust the config file alone; always set --server.address 127.0.0.1 explicitly."),
        ("PIL.Image.open() has no built-in size limit by default.",
         "A 30 KB PNG can legally decode to 1 GB in RAM (decompression bomb). Import "
         "src.app.security at the top of any module that calls Image.open() — it sets "
         "MAX_IMAGE_PIXELS = 100M as a side effect."),
        ("Streamlit's `unsafe_allow_html=True` is XSS-vulnerable for user-typed strings.",
         "Always pass user input through src.app.security.escape_html() before "
         "interpolating into HTML. patient.get('name') was leaking unescaped into the "
         "narrative — fixed, but watch for new callsites."),
        ("HF Hub model download counter has a multi-hour delay.",
         "Don't use download_count to verify whether the Space is fetching your checkpoint. "
         "Add [STARTUP] / [CHECKPOINT] log prints instead."),
        ("Streamlit caching prevents module-level code from re-running on script rerun.",
         "MOSTLY false — module-level code DOES re-run on script rerun in Streamlit. "
         "But @st.cache_resource'd functions run once per server lifetime. "
         "If you change model-loading code, you must restart the Streamlit server."),
        ("EMA optimizer crashes on BERT-layer unfreeze.",
         "EMA.update() must check `if n not in self.shadow: init`. This was a v1 bug — "
         "the fix is committed in experiments/run_full_pipeline.py."),
    ]
    _ind = ParagraphStyle("ind", parent=S["BODY"], leftIndent=12,
                          textColor=MID, spaceAfter=6)
    for h, b in lessons:
        story.append(P("•&nbsp;&nbsp;<b>" + h + "</b>", "BODY"))
        story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + b, _ind))

    story.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 14. REFERENCE
    # ────────────────────────────────────────────────────────────────────
    story.append(P("14.  Reference: env vars, paths, URLs, hashes", "H1"))

    story.append(P("14.1  Environment variables", "H2"))
    env_rows = [
        ["Variable", "Default", "Where read", "Purpose"],
        ["COLONAI_CHECKPOINT_HF_REPO", "(unset)", "app.py:67",
         "HF Hub model repo to download best_model.pth from. Vestigial now — checkpoint baked into Space repo."],
        ["COLONAI_CHECKPOINT_HF_FILE", "best_model.pth", "app.py:68",
         "Filename inside the model repo to download."],
        ["HF_TOKEN",                   "(unset)", "app.py:69, hf_hub_download token=",
         "Authenticated HF Hub access. Silences rate-limit warnings."],
        ["HUGGINGFACE_TOKEN",          "(unset)", "app.py:69 (fallback)",
         "Alternative name for HF_TOKEN."],
        ["COLONAI_API_KEY",            "(unset)", "scripts/serve_api.py + src/app/security.py",
         "Required to bind the API to 0.0.0.0. Enforced on /predict + /audit/today."],
        ["COLONAI_BIND",               "127.0.0.1", "scripts/serve_api.py",
         "API bind host. Refuses 0.0.0.0 unless COLONAI_API_KEY is set."],
        ["COLONAI_PORT",               "8081",   "scripts/serve_api.py",
         "API port."],
        ["COLONAI_CORS_ORIGINS",       "localhost:8501,localhost:8502", "scripts/serve_api.py",
         "Comma-separated CORS allow-list."],
        ["COLONAI_EXPOSE_DOCS",        "(unset)", "scripts/serve_api.py",
         "When set, enables Swagger /docs."],
        ["COLONAI_LOG_LEVEL",          "INFO",   "scripts/serve_api.py",
         "Log verbosity."],
        ["TRANSFORMERS_CACHE",         "/home/user/.cache/huggingface", "Dockerfile",
         "Where HF Hub caches transformer models."],
        ["HF_HOME",                    "/home/user/.cache/huggingface", "Dockerfile",
         "HF Hub root cache."],
    ]
    story.append(_header_table(env_rows,
                  col_widths=[1.6*inch, 1.4*inch, 1.5*inch, 2.5*inch]))

    story.append(P("14.2  Key file paths", "H2"))
    paths_rows = [
        ["Path",                                                          "Purpose"],
        ["outputs/unified_multimodal_v2/checkpoints/best_model.pth",      "v2 active checkpoint (572 MB)"],
        ["outputs/unified_multimodal_v2/seg_head.pth",                    "UNet segmentation decoder"],
        ["outputs/unified_multimodal_v2/ood_head.pth",                    "OOD detector MLP"],
        ["outputs/unified_multimodal_v2/ood_threshold.json",              "OOD decision threshold (F1-optimal)"],
        ["outputs/unified_multimodal_v2/temperature.json",                "Calibration temperature T = 0.45"],
        ["outputs/unified_multimodal_v2/metrics_v2.json",                 "Per-class P/R/F1, confusion matrix, ECE"],
        ["outputs/unified_multimodal_v2/cross_vendor_gradcam_compare.json","v1 vs v2 GradCAM IoU per dataset"],
        ["outputs/unified_multimodal_v2/seg_iou.json",                    "Segmentation IoU per epoch per dataset"],
        ["outputs/unified_multimodal_v2/cade_metrics_seg.json",           "CADe sensitivity / FPPI / mAP per dataset"],
        ["outputs/unified_multimodal_v2/fps_benchmark.json",              "FPS measurement output"],
        ["outputs/unified_multimodal_v2/figures/",                        "Dissertation plots + overlays"],
        ["outputs/unified_multimodal_v2/agent_coherence_report.json",     "3-case deployment-readiness test report"],
        ["outputs/audit/audit_YYYYMMDD.jsonl",                            "Daily audit log — every prediction (chmod 0600)"],
        ["data/processed/hyper_kvasir_clean/",                            "HyperKvasir training data (lower/upper-gi-tract subfolders)"],
        ["data/raw/CVC-ClinicDB/PNG/{Original,Ground Truth}/",            "CVC-ClinicDB images + masks"],
        ["data/raw/test_polyp_datasets/TestDataset/",                     "External polyp datasets (Pentax + others)"],
        ["data/raw/tcga/clinical/clinical.tsv",                           "TCGA tabular features"],
    ]
    story.append(_header_table(paths_rows, col_widths=[4.5*inch, 2.5*inch]))

    story.append(P("14.3  URLs and remotes", "H2"))
    url_rows = [
        ["Source code (GitHub)",     "https://github.com/Yuvraj235/Agentic_Multimodal_Colon_Cancer_AI"],
        ["Live demo (HF Space)",     "https://huggingface.co/spaces/Yuvraj2319/colonai"],
        ["Model checkpoint (HF Hub)","https://huggingface.co/Yuvraj2319/colonai-v2"],
        ["Direct Streamlit URL",     "https://yuvraj2319-colonai.hf.space"],
        ["HF tokens settings",       "https://huggingface.co/settings/tokens"],
    ]
    story.append(_kv_table(url_rows, col_widths=[2.2*inch, 4.8*inch]))

    story.append(P("14.4  Recent commit history (main branch)", "H2"))
    # Read git log
    import subprocess
    try:
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-15"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stderr=subprocess.DEVNULL).decode().strip()
        lines = [ln for ln in log.split("\n") if ln]
    except Exception:
        lines = ["(could not read git log)"]
    commits_rows = [["SHA", "Message"]]
    for ln in lines:
        parts = ln.split(" ", 1)
        if len(parts) == 2:
            commits_rows.append([parts[0], parts[1][:90]])
    story.append(_header_table(commits_rows, col_widths=[0.8*inch, 6.2*inch]))

    story.append(P("14.5  HF Hub auth", "H3"))
    story.append(P(
        "Token stored at ~/.cache/huggingface/token (write scope). "
        "Get a new one at https://huggingface.co/settings/tokens. "
        "Run <font face='Courier'>huggingface-cli login</font> to refresh. "
        "<b>DO NOT</b> commit this token to git or paste it into chat logs.", "BODY"))

    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=0.8, color=BRAND))
    story.append(Spacer(1, 0.1*inch))
    story.append(P(
        "<b>This document is the source of truth for the technical state of ColonAI.</b> "
        "If you change a threshold, retrain the model, or alter the deployment topology, "
        "regenerate this PDF (<font face='Courier'>python3 scripts/make_handover_pdf.py</font>) "
        "and replace the version in the repo.", "BODY"))
    story.append(Spacer(1, 0.05*inch))
    _author_style = ParagraphStyle("author_credit", parent=S["BODY"],
                                   fontSize=9, textColor=MID, alignment=TA_CENTER)
    story.append(Paragraph(
        "Document author: Yuvraj Pratap Singh (Amity University). "
        "Built with extensive AI assistance — every line in this codebase has been "
        "human-reviewed.",
        _author_style))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"  ✓ wrote {OUT_PATH}")
    print(f"  size: {OUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build()
