"""Builds the click-by-click presentation script as a polished A4 PDF.

Run:  python3 scripts/build_presentation_pdf.py
Output: outputs/ColonAI_Presentation_Script.pdf
"""

from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs" / "ColonAI_Presentation_Script.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether, PageBreak, ListFlowable, ListItem,
)
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

# ── Brand palette ────────────────────────────────────────────────────────
BRAND_BLUE = HexColor("#1A73E8")
BRAND_TEAL = HexColor("#00897B")
INK        = HexColor("#0F172A")
SUB        = HexColor("#475569")
SUBTLE     = HexColor("#94A3B8")
ACCENT     = HexColor("#FF5722")
SOFT_BG    = HexColor("#F4F8FF")
SOFT_CARD  = HexColor("#EEF4FF")
GREEN_OK   = HexColor("#15803D")
AMBER      = HexColor("#B45309")
RED        = HexColor("#B91C1C")

# ── Styles ───────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                    fontSize=22, leading=26, textColor=INK, spaceAfter=4, spaceBefore=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=14, leading=18, textColor=BRAND_BLUE, spaceAfter=4, spaceBefore=14)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold",
                    fontSize=11, leading=14, textColor=INK, spaceAfter=2, spaceBefore=10)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontName="Helvetica",
                      fontSize=10, leading=14.5, textColor=INK, alignment=TA_LEFT, spaceAfter=4)
SAY = ParagraphStyle("Say", parent=BODY, fontName="Helvetica-Oblique", textColor=SUB,
                     leftIndent=10, rightIndent=10, leading=14, fontSize=9.5)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, leading=11, textColor=SUB)
LABEL = ParagraphStyle("Label", parent=BODY, fontName="Helvetica-Bold", fontSize=9,
                       textColor=BRAND_BLUE, leading=12, spaceAfter=2)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=8.5, leading=11,
                         textColor=SUBTLE, alignment=TA_CENTER)


def hr(color=HexColor("#E2E8F0")):
    return HRFlowable(width="100%", thickness=0.6, color=color, spaceBefore=4, spaceAfter=4)


def cover_page(canv: canvas.Canvas, doc):
    canv.saveState()
    w, h = A4
    # Gradient-like banner (two bands)
    canv.setFillColor(BRAND_BLUE)
    canv.rect(0, h - 5*cm, w, 5*cm, fill=1, stroke=0)
    canv.setFillColor(BRAND_TEAL)
    canv.rect(0, h - 5*cm, w * 0.45, 5*cm, fill=1, stroke=0)

    canv.setFillColor(colors.white)
    canv.setFont("Helvetica-Bold", 26)
    canv.drawString(2*cm, h - 2.5*cm, "ColonAI · Presentation Script")
    canv.setFont("Helvetica", 12)
    canv.drawString(2*cm, h - 3.3*cm, "Click-by-click walkthrough · what to say, what to show, and why it works")
    canv.setFont("Helvetica", 9)
    canv.drawString(2*cm, h - 4.0*cm,
                    f"Agentic Multimodal Colon Cancer AI  ·  build {datetime.now():%d %b %Y}")

    # Footer brand bar
    canv.setFillColor(BRAND_BLUE)
    canv.rect(0, 0, w, 0.8*cm, fill=1, stroke=0)
    canv.setFillColor(colors.white)
    canv.setFont("Helvetica", 8)
    canv.drawString(2*cm, 0.30*cm,
                    "Research / educational use only. Not a medical device. Findings must be reviewed by a licensed clinician.")
    canv.restoreState()


def page_chrome(canv: canvas.Canvas, doc):
    canv.saveState()
    w, h = A4
    # Top brand bar
    canv.setFillColor(BRAND_BLUE)
    canv.rect(0, h - 0.55*cm, w, 0.55*cm, fill=1, stroke=0)
    canv.setFillColor(colors.white)
    canv.setFont("Helvetica-Bold", 9.5)
    canv.drawString(2*cm, h - 0.38*cm, "ColonAI — Presentation Script")
    canv.setFont("Helvetica", 8.5)
    canv.drawRightString(w - 2*cm, h - 0.38*cm, f"Page {doc.page}")
    # Bottom bar
    canv.setFillColor(HexColor("#F1F5F9"))
    canv.rect(0, 0, w, 0.55*cm, fill=1, stroke=0)
    canv.setFillColor(SUB)
    canv.setFont("Helvetica", 7.5)
    canv.drawString(2*cm, 0.20*cm, "Run with:  cd <project>  &&  ./run_app.command   (or)  python3 -m streamlit run app.py")
    canv.drawRightString(w - 2*cm, 0.20*cm, datetime.now().strftime("%d %b %Y"))
    canv.restoreState()


def card(title: str, body: list, bg=SOFT_CARD, border=BRAND_BLUE):
    """Render a soft-card flowable (Table-based for fill)."""
    cell = []
    cell.append(Paragraph(f"<b>{title}</b>", H3))
    for b in body:
        cell.append(b)
    t = Table([[cell]], colWidths=[16.6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("LINEBEFORE", (0,0), (-1,-1), 2.5, border),
        ("BOX",        (0,0), (-1,-1), 0.3, HexColor("#E2E8F0")),
        ("LEFTPADDING", (0,0), (-1,-1), 14),
        ("RIGHTPADDING",(0,0), (-1,-1), 14),
        ("TOPPADDING",  (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
    ]))
    return t


def step_row(time: str, click: str, you_say: str, screen: str):
    return Table([
        [
            Paragraph(f"<font color='#1A73E8'><b>{time}</b></font>", BODY),
            Paragraph(f"<b>{click}</b>", BODY),
            Paragraph(f"<i>{you_say}</i>", SAY),
            Paragraph(screen, SMALL),
        ]
    ], colWidths=[1.4*cm, 4.6*cm, 6.3*cm, 4.3*cm], style=TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LINEBELOW",(0,0),(-1,-1),0.3,HexColor("#E2E8F0")),
        ("LEFTPADDING",(0,0),(-1,-1),4), ("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))


def dot_legend():
    rows = [
        ("Click", "What you actually do in the app"),
        ("Say",   "The exact line to deliver — italic, slow, eye contact"),
        ("Screen","What the audience sees on screen"),
    ]
    return Table([[Paragraph(f"<b><font color='#1A73E8'>{a}</font></b> &nbsp;{b}", SMALL)] for a,b in rows],
                 colWidths=[16.6*cm], style=TableStyle([
                     ("BACKGROUND",(0,0),(-1,-1), HexColor("#F8FAFC")),
                     ("LINEBEFORE",(0,0),(-1,-1),2.0, BRAND_TEAL),
                     ("LEFTPADDING",(0,0),(-1,-1), 12), ("RIGHTPADDING",(0,0),(-1,-1), 12),
                     ("TOPPADDING",(0,0),(-1,-1), 6), ("BOTTOMPADDING",(0,0),(-1,-1), 6),
                 ]))


def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        title="ColonAI — Presentation Script",
    )

    flow = []

    # Cover page-ish opening (the cover is drawn by onFirstPage)
    flow.append(Spacer(1, 5*cm))  # leaves space for the cover banner
    flow.append(Spacer(1, 0.6*cm))
    flow.append(Paragraph("Why this exists", H2))
    flow.append(Paragraph(
        "This is the entire script for a 6–8 minute live demo of the ColonAI app. "
        "Every step has three things: <b>what to click</b>, <b>what to say</b>, and "
        "<b>what the audience sees</b>. If you follow it cold, your demo will land.",
        BODY))
    flow.append(Spacer(1, 6))
    flow.append(dot_legend())
    flow.append(Spacer(1, 12))

    # ── Section 1: Open & Setup ──────────────────────────────────────
    flow.append(card(
        "Before the demo · 60 seconds of prep",
        [
            Paragraph("Open a Terminal (Cmd-Space → \"Terminal\"). Run the launcher:", BODY),
            Paragraph(
                "<font face='Courier' size='9' color='#0F172A'>"
                "cd ~/Desktop/Agentic_Multimodal_Colon_Cancer_AI\\ copy &amp;&amp; ./run_app.command"
                "</font>", BODY),
            Paragraph(
                "Wait until the sidebar shows a green pulsing dot \"AI pipeline ready · checkpoint loaded\". "
                "That tells you the model weights loaded — not the demo fallback. "
                "If it stays grey, click any case on Step 1 to trigger loading.", BODY),
            Paragraph(
                "<b>Sanity check:</b> sidebar status dot should be green, not amber/red.", SMALL),
        ],
        bg=HexColor("#F0FDF4"), border=GREEN_OK,
    ))
    flow.append(Spacer(1, 12))

    # ── Section 2: The opener ────────────────────────────────────────
    flow.append(Paragraph("The opener · 30 seconds", H2))
    flow.append(card(
        "Say this verbatim",
        [Paragraph(
            "<i>\"Colorectal cancer is the third-most-common cancer worldwide — 90 % of "
            "early-stage cases are curable, but only 60 % are caught early. ColonAI is a "
            "multimodal AI screening tool that combines endoscopy images, clinical notes, "
            "and patient history to flag high-risk cases for the clinician — it doesn't "
            "replace them.\"</i>", BODY)],
    ))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(
        "Memorise the line. Say it slowly. Eye contact. The whole audience knows in 30 seconds "
        "what this is, what it solves, and what it isn't.", BODY))

    flow.append(PageBreak())

    # ── Section 3: The walkthrough ──────────────────────────────────
    flow.append(Paragraph("The 6-minute live walkthrough", H2))
    flow.append(Paragraph(
        "Each row tells you what to click, what to say, and what the audience sees. "
        "Times are cumulative.", SMALL))
    flow.append(Spacer(1, 6))

    # Header row
    header = Table([[
        Paragraph("<b>T+</b>", LABEL),
        Paragraph("<b>Click</b>", LABEL),
        Paragraph("<b>Say</b>", LABEL),
        Paragraph("<b>What's on screen</b>", LABEL),
    ]], colWidths=[1.4*cm, 4.6*cm, 6.3*cm, 4.3*cm], style=TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),HexColor("#EEF4FF")),
        ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    flow.append(header)

    steps = [
        ("0:00", "Sidebar visible · sidebar shows pulsing green dot",
         "Today I'll show how ColonAI screens for colorectal disease using three signals at once.",
         "App on Step 1 · sidebar ColonAI logo · status dot green"),
        ("0:20", "Click Site Guide → How to Present (then Back to App)",
         "There's a built-in presentation guide and 3 case studies.",
         "Brief glimpse of How-to-Present tab, then back to Step 1"),
        ("0:45", "On Step 1 click Load Case A · Sigmoid Polyp",
         "One click loads a real screening-FIT-positive patient — 58-year-old man, asymptomatic.",
         "Hero on Step 2 with image already in tab 2"),
        ("1:15", "Click the Upload Images tab to show the loaded image",
         "The image is from HyperKvasir; the symptoms come bundled with the case.",
         "Polyp endoscopy thumbnail visible"),
        ("1:35", "Click Analyse →",
         "Watch the 6 agents fire — each is autonomous; the orchestrator coordinates them.",
         "Pipeline animation, agent chips light up sequentially"),
        ("2:35", "Pipeline finishes · Diagnostic Results page loads",
         "Class probability 75 %+ for polyps, low risk score, low uncertainty.",
         "4 metric cards + Diagnosis tab default"),
        ("3:05", "Click 'Why this result?' tab (NEW)",
         "Plain-English explanation of the prediction. Per-modality evidence, confidence/uncertainty, and counterfactuals — what would have changed the answer.",
         "Reasoning panel + 3 evidence cards"),
        ("3:35", "Click GradCAM View tab",
         "This is the AI showing its work — the warm region is exactly what drove the prediction. A clinician can sanity-check this in two seconds.",
         "Original / heatmap side-by-side"),
        ("3:50", "Click Risk Charts tab",
         "Risk gauge plus a multi-dimensional radar combining AI risk with age, smoking, alcohol, family history and prior polyps.",
         "Gauge meter + radar chart"),
        ("4:20", "Click Recommendations tab",
         "The advice is tied to BSG and NICE pathways — surveillance interval, specialist referral, lifestyle. Not generic boilerplate.",
         "Urgency banner + primary action + surveillance + lifestyle"),
        ("4:50", "Click Find Doctors → type 'Noida' or 'New Delhi'",
         "The directory is location-aware — Delhi-NCR queries surface AIIMS, Sir Ganga Ram, Medanta, Apollo, Fortis, BLK-Max specialists. Each card shows WHY it matched.",
         "AI-tailored cards with 'Why recommended' chips + Google-Maps Directions link"),
        ("5:15", "Click Generate Report → Generate PDF Report",
         "The full clinical report, ready to send. With disclaimer and embedded GradCAM image.",
         "PDF download button appears"),
        ("5:45", "Optional: New Assessment → Load Case B (UC)",
         "Different patient, different prediction — UC mild, different recommendation.",
         "Pipeline runs again with bloody-diarrhoea case"),
    ]
    for t, c, s, sc in steps:
        flow.append(step_row(t, c, s, sc))

    flow.append(Spacer(1, 14))
    flow.append(card(
        "If you only have 3 minutes",
        [Paragraph(
            "Skip Find Doctors and PDF. Run the punch-line sequence: "
            "<b>Load Case A → Analyse → GradCAM → Recommendations</b>. "
            "That's the whole story in 90 seconds.", BODY)],
        bg=HexColor("#FFF8E1"), border=AMBER,
    ))
    flow.append(PageBreak())

    # ── Section 4: Technical slide ──────────────────────────────────
    flow.append(Paragraph("The technical slide · 90 seconds", H2))
    flow.append(Paragraph(
        "Show the <b>AI Explained</b> tab in Site Guide. Three numbers, three words — that's it.",
        BODY))
    flow.append(Spacer(1, 6))
    tech = Table([
        [Paragraph("<b>Architecture</b>", LABEL),
         Paragraph("Dual-backbone fusion: ResNet-50 + EfficientNet-B0 (image) → BioBERT (text) → "
                   "TabTransformer (12 TCGA features) → 3-stage gated cross-modal transformer "
                   "→ 3 task heads (5-class pathology, 4-class staging, binary risk).", BODY)],
        [Paragraph("<b>Performance</b>", LABEL),
         Paragraph("90.3 % test accuracy · 0.984 AUC-ROC · 0.81 macro-F1 on 1,066 held-out images. "
                   "Best epoch 7 of 60 (no overfit). ~150 M parameters.", BODY)],
        [Paragraph("<b>Explainability</b>", LABEL),
         Paragraph("GradCAM++ on ResNet layer4[-1] for the heatmap; BioBERT attention rollout on text; "
                   "SHAP-style perturbation on tabular; MC-Dropout (15 passes) for uncertainty.", BODY)],
        [Paragraph("<b>Datasets</b>", LABEL),
         Paragraph("HyperKvasir (10,662 images, Norway) + CVC-ClinicDB (612 polyp images, Spain) "
                   "+ TCGA clinical (461 patients, North America). Stratified split, no leakage.", BODY)],
    ], colWidths=[3.4*cm, 13.2*cm], style=TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LINEBELOW",(0,0),(-1,-1), 0.3, HexColor("#E2E8F0")),
        ("LEFTPADDING",(0,0),(-1,-1),6), ("RIGHTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    flow.append(tech)

    # ── Section 5: Honesty slide ────────────────────────────────────
    flow.append(Spacer(1, 14))
    flow.append(Paragraph("The honesty slide · 45 seconds", H2))
    flow.append(Paragraph(
        "Reviewers respect honesty more than spin. Lead with the limitations — it defuses ~80 % "
        "of hostile questions before they're asked.", BODY))
    flow.append(Spacer(1, 4))
    bullets = [
        "HyperKvasir is European (Norwegian), CVC-ClinicDB is Spanish, TCGA tabular is North-American — no external validation on Asian / African endoscopy databases yet.",
        "5 classes only — doesn't cover sessile-serrated lesions, CMV colitis, eosinophilic oesophagitis, or rarer entities.",
        "Staging is image-derived; in real practice T-stage needs CT/MRI plus biopsy.",
        "Currently research-grade — clinical deployment requires UKCA Class IIa, MHRA registration, post-market surveillance.",
    ]
    flow.append(ListFlowable(
        [ListItem(Paragraph(b, BODY)) for b in bullets],
        bulletType="bullet", start="•", leftIndent=14, bulletColor=BRAND_BLUE,
    ))

    flow.append(PageBreak())

    # ── Section 6: Likely Q&A ───────────────────────────────────────
    flow.append(Paragraph("The 8 questions you will be asked", H2))
    qa = [
        ("Have you validated this on real hospital data?",
         "No, not yet. The training data is European. External validation on at least two non-Western "
         "hospital datasets is the immediate next step. That's why every recommendation says 'review by a clinician'."),
        ("Why should a doctor use this instead of just looking at the image themselves?",
         "It doesn't replace looking — it adds a second signal. The AI reads the image, the clinical "
         "note, AND the patient's risk factors at the same time. The Modality Weights chart shows "
         "exactly how much each one mattered — explicit and auditable."),
        ("What about regulation?",
         "It would fall under SaMD (Software as a Medical Device). UKCA Class IIa under MHRA in the UK, "
         "FDA 510(k) De-Novo in the US, CE Class IIa under MDR in the EU. We're not pursuing certification "
         "yet because we don't have prospective external-validation data."),
        ("Why dual image backbones?",
         "ResNet-50 gives a clean 7×7 GradCAM target for interpretability; EfficientNet adds a parallel "
         "14×14 representation. A learned per-position gate fuses them, beating either alone by ~2 % macro-F1."),
        ("How do you avoid overfitting?",
         "Mixup α=0.3, label smoothing 0.1, weight decay 0.15, RandomPerspective + GaussianBlur + RandomErasing(p=0.4), "
         "Gaussian noise σ=0.05 on tabular features, EMA decay 0.9995, BERT freeze→unfreeze schedule. "
         "Best epoch was 7/60 — well before the curves flatten."),
        ("How does it handle uncertainty?",
         "MC-Dropout — 15 stochastic forward passes at inference. Predictive entropy is reported. "
         "Uncertainty > 0.6 triggers a 'seek expert review' note in the recommendation agent."),
        ("Why TCGA tabular when most patients aren't cancer patients?",
         "TCGA gives a realistic distribution of age × smoking × alcohol × BMI × stage. We sample one "
         "row per inference and overwrite the patient-known fields, so the model sees realistic correlations."),
        ("Could a clinician adopt this tomorrow?",
         "Not safely. To deploy: external validation on the target hospital's cases, prospective evaluation "
         "against histology gold-standard, regulatory approval, PACS / report-system integration, and ongoing "
         "post-market surveillance."),
    ]
    for q, a in qa:
        flow.append(Paragraph(f"<b>Q. {q}</b>", H3))
        flow.append(Paragraph(a, BODY))
        flow.append(Spacer(1, 4))

    flow.append(PageBreak())

    # ── Section 7: Three case studies ───────────────────────────────
    flow.append(Paragraph("Three case studies — memorise these", H2))
    cases = [
        ("Case A · Sigmoid polyp on screening FIT", BRAND_BLUE,
         "58-year-old man, asymptomatic, FIT 180 µg Hb/g positive on NHS bowel screening. Ex-smoker, BMI 26.5.",
         [
             ("Endoscopic finding", "14 mm sessile polyp in sigmoid colon."),
             ("AI prediction",      "polyps · ~88 % confidence · benign-risk score."),
             ("Clinical action",    "EMR (≥10 mm) · histology to MDT."),
             ("Surveillance",       "BSG/ACPGBI/PHE 2020: high-risk → repeat colonoscopy at 3 years; low-risk → return to FIT screening."),
             ("Why it matters",     "Adenoma-to-carcinoma sequence takes 5–15 yrs. Screening from 45–50 catches polyps before they're cancer."),
         ]),
        ("Case B · Bloody diarrhoea — suspected UC", ACCENT,
         "31-year-old woman, 6 weeks bloody diarrhoea (4–5/day), urgency, mild left-iliac-fossa cramping. CRP 22, faecal calprotectin 480.",
         [
             ("Endoscopic finding", "Granular mucosa, loss of vascular pattern, contact bleeding — Mayo endoscopic 1–2."),
             ("AI prediction",      "uc-mild · pathological-finding flag raised."),
             ("Differential",       "Infective colitis (rule out C. diff, Campylobacter), Crohn's, ischaemic colitis."),
             ("Clinical action",    "Topical + oral 5-ASA induction · flexi-sig with biopsies · gastro follow-up at 6 weeks."),
             ("Long-term",          "After 8–10 yrs of colitis, surveillance colonoscopy with chromoendoscopy."),
         ]),
        ("Case C · Long-standing GORD — Barrett's surveillance", HexColor("#9C27B0"),
         "62-year-old man, 15 yrs GORD on long-term PPI, BMI 31, ex-smoker. Surveillance OGD shows 4 cm tongue of columnar mucosa above GOJ.",
         [
             ("Endoscopic finding", "Prague C2M4 segment of intestinal metaplasia, no nodularity."),
             ("AI prediction",      "barretts-esoph · ~91 % confidence."),
             ("Histology protocol", "Seattle-protocol biopsies: 4-quadrant every 2 cm + targeted."),
             ("Clinical action",    "Non-dysplastic Barrett's ≥3 cm → 3-yearly surveillance OGD (BSG 2023)."),
             ("Escalation",         "Any LGD → expert path review + 6-month repeat. HGD or T1a → endoscopic eradication therapy (RFA ± EMR)."),
         ]),
    ]
    for title, color, vignette, rows in cases:
        flow.append(Paragraph(f"<font color='{color.hexval()}'><b>{title}</b></font>", H3))
        flow.append(Paragraph(f"<i>{vignette}</i>", BODY))
        rt = Table([
            [Paragraph(f"<b>{lab}</b>", LABEL), Paragraph(body, BODY)]
            for lab, body in rows
        ], colWidths=[3.6*cm, 13*cm], style=TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LINEBELOW",(0,0),(-1,-1), 0.25, HexColor("#E2E8F0")),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ]))
        flow.append(rt)
        flow.append(Spacer(1, 10))

    flow.append(PageBreak())

    # ── Section 8: How to launch (with command) ─────────────────────
    flow.append(Paragraph("Launching the app — every time, no surprises", H2))
    flow.append(Paragraph(
        "Two ways. The first is what to give a non-technical user. The second is for "
        "yourself if anything goes wrong.", BODY))
    flow.append(Spacer(1, 6))

    flow.append(card(
        "Option 1 · One-click (preferred)",
        [
            Paragraph("In Finder, double-click <b>run_app.command</b> in the project folder.", BODY),
            Paragraph(
                "It opens a Terminal, starts the server, and prints the URL. "
                "Visit <font face='Courier'>http://localhost:8501</font> in your browser. "
                "When you're done, press <b>Ctrl-C</b> in that Terminal to stop.", BODY),
        ],
        bg=HexColor("#F0FDF4"), border=GREEN_OK,
    ))
    flow.append(Spacer(1, 8))

    flow.append(card(
        "Option 2 · Manual (Terminal)",
        [
            Paragraph("Open Terminal, then:", BODY),
            Paragraph(
                "<font face='Courier' size='9'>"
                "cd ~/Desktop/Agentic_Multimodal_Colon_Cancer_AI\\ copy<br/>"
                "python3 -m streamlit run app.py --server.port 8501"
                "</font>", BODY),
            Paragraph("That's the only command you need.", SMALL),
        ],
        bg=HexColor("#FFF8E1"), border=AMBER,
    ))
    flow.append(Spacer(1, 8))

    flow.append(card(
        "Troubleshooting · 60-second fixes",
        [
            Paragraph(
                "<b>Port 8501 already in use:</b> the launcher tries 8502, 8503 etc. automatically. "
                "Or kill the previous server: "
                "<font face='Courier'>lsof -ti:8501 | xargs kill -9</font>", BODY),
            Paragraph(
                "<b>'streamlit not found':</b> install once: "
                "<font face='Courier'>pip3 install -r requirements.txt</font>", BODY),
            Paragraph(
                "<b>Sidebar shows red 'Model load failed':</b> the checkpoint file is missing. "
                "Confirm <font face='Courier'>outputs/unified_multimodal/checkpoints/best_model.pth</font> exists. "
                "Without it, the app falls into demo mode (still presentable, just simulated).", BODY),
            Paragraph(
                "<b>App is slow on first analysis:</b> normal — the model loads ~20 s on first run, "
                "then is cached. The Quick-demo cards trigger this loading early.", BODY),
        ],
        bg=HexColor("#FEF2F2"), border=RED,
    ))

    flow.append(PageBreak())

    # ── Section 9: Closing ──────────────────────────────────────────
    flow.append(Paragraph("The closer", H2))
    flow.append(card(
        "Say this and stop talking",
        [Paragraph(
            "<i>\"ColonAI is not a black box and not a replacement for a clinician — it's a second "
            "pair of eyes that looks at the image, the patient, and the symptoms together. The win "
            "is in catching the cases that get missed when only one signal is examined.\"</i>", BODY)],
    ))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(
        "Then take a breath. Q&A starts. Don't fill silence with extra information — wait for "
        "the question.", BODY))

    flow.append(Spacer(1, 16))
    flow.append(hr())
    flow.append(Paragraph("Brand and references", H2))
    flow.append(Paragraph(
        "Architecture: dual-backbone (ResNet-50 + EfficientNet-B0) + BioBERT + TabTransformer + "
        "3-stage gated cross-modal transformer; three task heads. "
        "Datasets: HyperKvasir, CVC-ClinicDB, TCGA. "
        "Guidelines cited: NICE NG12 (suspected cancer 2-week-wait, last update 2023); "
        "BSG/ACPGBI/PHE 2020 (post-polypectomy surveillance); BSG 2023 (Barrett's surveillance); "
        "USPSTF 2021 (CRC screening); ESGE 2020/2024 (polypectomy). "
        "All clinical content is for educational use only — every finding must be confirmed by a licensed clinician.",
        SMALL))

    doc.build(flow, onFirstPage=cover_page, onLaterPages=page_chrome)
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()
