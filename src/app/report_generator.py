"""
PDF Report Generator for the Colon Cancer AI Diagnostic System.
Uses reportlab for professional clinical-grade output.
"""

import io
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, KeepTogether, PageBreak
    )
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics import renderPDF
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages


# ─── Colour palette ────────────────────────────────────────────────────────
BRAND_BLUE   = colors.HexColor("#1A73E8") if HAS_REPORTLAB else None
BRAND_TEAL   = colors.HexColor("#00897B") if HAS_REPORTLAB else None
RISK_GREEN   = colors.HexColor("#2E7D32") if HAS_REPORTLAB else None
RISK_YELLOW  = colors.HexColor("#F9A825") if HAS_REPORTLAB else None
RISK_ORANGE  = colors.HexColor("#E65100") if HAS_REPORTLAB else None
RISK_RED     = colors.HexColor("#B71C1C") if HAS_REPORTLAB else None
LIGHT_GREY   = colors.HexColor("#F5F5F5") if HAS_REPORTLAB else None
MID_GREY     = colors.HexColor("#9E9E9E") if HAS_REPORTLAB else None
DARK_TEXT    = colors.HexColor("#212121") if HAS_REPORTLAB else None


def _risk_color(risk_score: float):
    """Return a ReportLab color based on risk score 0–1."""
    if risk_score < 0.25:
        return RISK_GREEN
    if risk_score < 0.5:
        return RISK_YELLOW
    if risk_score < 0.75:
        return RISK_ORANGE
    return RISK_RED


def _risk_label(risk_score: float) -> str:
    if risk_score < 0.25:
        return "LOW"
    if risk_score < 0.5:
        return "MODERATE"
    if risk_score < 0.75:
        return "HIGH"
    return "CRITICAL"


def _numpy_to_rl_image(arr: np.ndarray, width_cm: float = 8.0, height_cm: float = 6.0) -> Optional[object]:
    """Convert numpy image array to ReportLab Image flowable."""
    if not HAS_REPORTLAB or arr is None:
        return None
    try:
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return RLImage(buf, width=width_cm * cm, height=height_cm * cm)
    except Exception:
        return None


# ─── Main entry point ──────────────────────────────────────────────────────

def generate_pdf_report(
    patient_data:    Dict[str, Any],
    symptoms:        List[str],
    symptom_text:    str,
    analysis:        Optional[Dict[str, Any]],
    doctors:         List[Dict[str, Any]],
    gradcam_overlay: Optional[np.ndarray] = None,
    original_image:  Optional[np.ndarray] = None,
) -> bytes:
    """
    Generate a comprehensive clinical PDF report.

    Returns:
        PDF bytes ready for st.download_button.
    """
    if HAS_REPORTLAB:
        return _generate_reportlab(
            patient_data, symptoms, symptom_text,
            analysis, doctors, gradcam_overlay, original_image
        )
    # Fallback: matplotlib-based PDF
    return _generate_matplotlib_pdf(
        patient_data, symptoms, symptom_text,
        analysis, doctors, gradcam_overlay, original_image
    )


# ─── ReportLab implementation ──────────────────────────────────────────────

def _generate_reportlab(
    patient_data, symptoms, symptom_text,
    analysis, doctors, gradcam_overlay, original_image
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="AI Diagnostic Report",
        author="ColonAI Diagnostic System",
    )

    styles = getSampleStyleSheet()
    # Custom styles
    h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                        textColor=BRAND_BLUE, fontSize=20, spaceAfter=6,
                        fontName="Helvetica-Bold")
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        textColor=BRAND_TEAL, fontSize=14, spaceAfter=4,
                        fontName="Helvetica-Bold", spaceBefore=12)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"],
                        textColor=DARK_TEXT, fontSize=11, spaceAfter=3,
                        fontName="Helvetica-Bold")
    normal = ParagraphStyle("Normal2", parent=styles["Normal"],
                             fontSize=10, leading=15, textColor=DARK_TEXT)
    small = ParagraphStyle("Small", parent=styles["Normal"],
                            fontSize=8, textColor=MID_GREY, leading=12)
    center = ParagraphStyle("Center", parent=styles["Normal"],
                             alignment=TA_CENTER, fontSize=10)
    disclaimer_style = ParagraphStyle("Disc", parent=styles["Normal"],
                                       fontSize=8, textColor=MID_GREY,
                                       leading=12, alignment=TA_JUSTIFY)

    story = []
    W = A4[0] - 4 * cm  # usable width

    # ── Header ──────────────────────────────────────────────────────────
    story.append(Paragraph("AI Diagnostic Report", h1))
    story.append(Paragraph("Agentic Multimodal Colon Cancer Screening System", center))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_BLUE, spaceAfter=10))
    story.append(Spacer(1, 0.3 * cm))

    # Report meta
    now = datetime.now()
    meta_data = [
        ["Report ID:", f"RPT-{now.strftime('%Y%m%d%H%M%S')}",
         "Generated:", now.strftime("%d %b %Y, %H:%M")],
        ["Model Version:", "UnifiedMultiModalTransformer v1.0",
         "Pipeline:", "6-Agent Agentic Analysis"],
    ]
    meta_table = Table(meta_data, colWidths=[3 * cm, 6 * cm, 2.5 * cm, 5.5 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), MID_GREY),
        ("TEXTCOLOR", (2, 0), (2, -1), MID_GREY),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── Patient Information ───────────────────────────────────────────
    story.append(Paragraph("1. Patient Information", h2))
    p = patient_data
    pt_data = [
        ["Name:", p.get("name", "N/A"), "Age:", str(p.get("age", "N/A"))],
        ["Gender:", p.get("gender", "N/A"), "BMI:", str(p.get("bmi", "N/A"))],
        ["City:", p.get("city", "N/A"), "Country:", p.get("country", "N/A")],
        ["Smoking:", p.get("smoking", "No"), "Alcohol:", p.get("alcohol", "No")],
        ["Family History:", p.get("family_history", "No"), "Previous Polyps:", p.get("prev_polyps", "No")],
    ]
    pt_table = Table(pt_data, colWidths=[3 * cm, 5.5 * cm, 3 * cm, 5.5 * cm])
    pt_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), MID_GREY),
        ("TEXTCOLOR", (2, 0), (2, -1), MID_GREY),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.3, MID_GREY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(pt_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── Reported Symptoms ─────────────────────────────────────────────
    story.append(Paragraph("2. Reported Symptoms", h2))
    if symptoms:
        symp_text = " • ".join(symptoms)
        story.append(Paragraph(f"<b>Checked symptoms:</b> {symp_text}", normal))
    if symptom_text:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(f"<b>Additional description:</b> {symptom_text}", normal))
    if not symptoms and not symptom_text:
        story.append(Paragraph("No symptoms reported.", normal))
    story.append(Spacer(1, 0.4 * cm))

    # ── AI Analysis Results ───────────────────────────────────────────
    story.append(Paragraph("3. AI Analysis Results", h2))

    if analysis:
        risk_score  = analysis.get("risk_score", 0.0)
        risk_color  = _risk_color(risk_score)
        risk_lbl    = _risk_label(risk_score)
        path_class  = analysis.get("pathology_class", "Unknown")
        stage       = analysis.get("stage", "Unknown")
        confidence  = analysis.get("confidence", 0.0)
        uncertainty = analysis.get("uncertainty", 0.0)

        # Summary banner
        banner_data = [[
            f"FINDING\n{path_class.upper().replace('-', ' ')}",
            f"STAGE\n{stage}",
            f"RISK\n{risk_lbl}  {risk_score:.0%}",
            f"CONFIDENCE\n{confidence:.0%}",
        ]]
        banner_table = Table(banner_data, colWidths=[W / 4] * 4)
        banner_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (0, 0), BRAND_BLUE),
            ("BACKGROUND", (1, 0), (1, 0), BRAND_TEAL),
            ("BACKGROUND", (2, 0), (2, 0), risk_color),
            ("BACKGROUND", (3, 0), (3, 0), colors.HexColor("#37474F")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 0.4 * cm))

        # Probability breakdown
        probs = analysis.get("pathology_probs", {})
        if probs:
            story.append(Paragraph("Class Probability Breakdown", h3))
            prob_data = [["Class", "Probability", "Bar"]]
            for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
                bar_len = int(prob * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                prob_data.append([cls.replace("-", " ").title(),
                                   f"{prob:.1%}", bar])
            prob_table = Table(prob_data, colWidths=[5 * cm, 3 * cm, W - 8 * cm])
            prob_table.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
                ("GRID", (0, 0), (-1, -1), 0.25, MID_GREY),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(prob_table)
            story.append(Spacer(1, 0.3 * cm))

        # Modality weights
        img_w  = analysis.get("image_weight", 0.33)
        txt_w  = analysis.get("text_weight",  0.33)
        tab_w  = analysis.get("tabular_weight", 0.34)
        story.append(Paragraph("Modality Contribution Weights", h3))
        mod_data = [
            ["Imaging (GradCAM++)",    f"{img_w:.1%}"],
            ["Clinical Text (BioBERT)", f"{txt_w:.1%}"],
            ["Patient Data (TCGA)",    f"{tab_w:.1%}"],
        ]
        mod_table = Table(mod_data, colWidths=[8 * cm, 4 * cm])
        mod_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GREY, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.25, MID_GREY),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(mod_table)
        story.append(Spacer(1, 0.3 * cm))

        # Uncertainty
        uncert_label = "Low" if uncertainty < 0.3 else "Moderate" if uncertainty < 0.6 else "High"
        story.append(Paragraph(
            f"<b>Model Uncertainty:</b> {uncert_label} ({uncertainty:.2f}) &nbsp;&nbsp; "
            f"<b>Inference Time:</b> {analysis.get('inference_time_ms', 0):.0f} ms",
            normal
        ))
        story.append(Spacer(1, 0.4 * cm))

        # GradCAM images (side by side)
        img_row = []
        if original_image is not None:
            rl_orig = _numpy_to_rl_image(original_image, width_cm=7.5, height_cm=6.0)
            if rl_orig:
                img_row.append(rl_orig)
        if gradcam_overlay is not None:
            rl_cam = _numpy_to_rl_image(gradcam_overlay, width_cm=7.5, height_cm=6.0)
            if rl_cam:
                img_row.append(rl_cam)
        if img_row:
            story.append(Paragraph("Endoscopy Image & GradCAM++ Attention", h3))
            caption_row = []
            if original_image is not None:
                caption_row.append(Paragraph("Original Endoscopy Image", center))
            if gradcam_overlay is not None:
                caption_row.append(Paragraph("GradCAM++ Heatmap (AI Focus Region)", center))
            while len(img_row) < 2:
                img_row.append("")
                caption_row.append("")
            img_table = Table([img_row, caption_row], colWidths=[W / 2, W / 2])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 0),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 0.4 * cm))

    else:
        story.append(Paragraph("No image was uploaded. AI analysis not performed.", normal))
        story.append(Spacer(1, 0.4 * cm))

    # ── Clinical Recommendation ───────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Clinical Recommendations", h2))

    if analysis and analysis.get("recommendation"):
        rec = analysis["recommendation"]
        urgency     = rec.get("urgency", "Routine")
        urgency_colors = {
            "Routine": RISK_GREEN, "Elective": RISK_YELLOW,
            "Urgent": RISK_ORANGE, "Emergency": RISK_RED
        }
        u_color = urgency_colors.get(urgency, RISK_GREEN)

        # Urgency badge
        urg_table = Table([[f"URGENCY: {urgency.upper()}"]],
                          colWidths=[W])
        urg_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), u_color),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 13),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(urg_table)
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph(f"<b>Primary Action:</b> {rec.get('primary_action', 'N/A')}", normal))
        story.append(Paragraph(f"<b>Surveillance Plan:</b> {rec.get('surveillance', 'N/A')}", normal))
        story.append(Spacer(1, 0.3 * cm))

        # Referrals
        referrals = rec.get("referrals", [])
        if referrals:
            story.append(Paragraph("Specialist Referrals", h3))
            for r in referrals:
                story.append(Paragraph(f"• {r}", normal))

        # Investigations
        investigations = rec.get("investigations", [])
        if investigations:
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph("Recommended Investigations", h3))
            for inv in investigations:
                story.append(Paragraph(f"• {inv}", normal))

        # Lifestyle
        lifestyle = rec.get("lifestyle_advice", [])
        if lifestyle:
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph("Lifestyle Recommendations", h3))
            for lf in lifestyle:
                story.append(Paragraph(f"• {lf}", normal))

        story.append(Spacer(1, 0.4 * cm))
    else:
        story.append(Paragraph(
            "Please consult a gastroenterologist for a full clinical assessment "
            "and personalized treatment recommendations.", normal))
        story.append(Spacer(1, 0.4 * cm))

    # ── Suggested Doctors ─────────────────────────────────────────────
    if doctors:
        story.append(Paragraph("5. Suggested Specialists Near You", h2))
        for i, dr in enumerate(doctors[:5], 1):
            doc_data = [
                [f"{i}. {dr['name']}", f"Rating: {dr['rating']:.1f}/5.0"],
                [f"{dr['specialty']}", f"{dr['experience_years']} yrs exp"],
                [dr['hospital'], ""],
                [f"{dr['city']}, {dr['country']}",
                 f"Tel: {dr.get('phone', 'N/A')}"],
            ]
            doc_table = Table(doc_data, colWidths=[W - 4 * cm, 4 * cm])
            doc_table.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GREY),
                ("GRID", (0, 0), (-1, -1), 0.25, MID_GREY),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("SPAN", (0, 2), (1, 2)),
            ]))
            story.append(doc_table)
            story.append(Spacer(1, 0.2 * cm))
        story.append(Spacer(1, 0.3 * cm))

    # ── Disclaimer ───────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=MID_GREY, spaceAfter=8))
    story.append(Paragraph(
        "<b>IMPORTANT DISCLAIMER:</b> This report is generated by an AI system for "
        "informational and screening purposes ONLY. It does NOT constitute a medical "
        "diagnosis, treatment plan, or professional medical advice. All findings must be "
        "reviewed and verified by a qualified, licensed medical professional before any "
        "clinical decisions are made. The AI model has been trained on research datasets "
        "(HyperKvasir, CVC-ClinicDB, TCGA) and may not generalise to all patient "
        "populations. Do not delay seeking professional medical advice based on this report.",
        disclaimer_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─── Matplotlib fallback ───────────────────────────────────────────────────

def _generate_matplotlib_pdf(
    patient_data, symptoms, symptom_text,
    analysis, doctors, gradcam_overlay, original_image
) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: Header + Patient + Symptoms + Results
        fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        for ax in axes:
            ax.axis("off")

        p = patient_data or {}
        axes[0].text(0.5, 0.7, "AI Diagnostic Report", ha="center", va="center",
                     fontsize=18, fontweight="bold", color="#1A73E8")
        axes[0].text(0.5, 0.3, "Agentic Multimodal Colon Cancer Screening System",
                     ha="center", va="center", fontsize=11, color="#555")
        axes[0].axhline(0.05, color="#1A73E8", linewidth=2)

        pt_str = (f"Patient: {p.get('name','N/A')}  |  Age: {p.get('age','N/A')}  |  "
                  f"Gender: {p.get('gender','N/A')}  |  City: {p.get('city','N/A')}")
        axes[1].text(0.02, 0.7, "Patient Information", fontsize=12, fontweight="bold", color="#00897B")
        axes[1].text(0.02, 0.3, pt_str, fontsize=9, color="#333")

        symp_str = "Symptoms: " + (", ".join(symptoms) if symptoms else "None reported")
        axes[2].text(0.02, 0.7, "Reported Symptoms", fontsize=12, fontweight="bold", color="#00897B")
        axes[2].text(0.02, 0.3, symp_str, fontsize=9, color="#333", wrap=True)

        if analysis:
            res_str = (
                f"Finding: {analysis.get('pathology_class','N/A').replace('-',' ').title()}  |  "
                f"Stage: {analysis.get('stage','N/A')}  |  "
                f"Risk Score: {analysis.get('risk_score',0):.1%}  |  "
                f"Confidence: {analysis.get('confidence',0):.1%}"
            )
            axes[3].text(0.02, 0.7, "AI Analysis Results", fontsize=12, fontweight="bold", color="#00897B")
            axes[3].text(0.02, 0.3, res_str, fontsize=9, color="#333")
        else:
            axes[3].text(0.02, 0.5, "No image uploaded — AI analysis not performed.",
                         fontsize=9, color="#999")

        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # Page 2: Images side by side if available
        if gradcam_overlay is not None or original_image is not None:
            fig2, axs2 = plt.subplots(1, 2, figsize=(8.27, 5.0))
            fig2.patch.set_facecolor("white")
            if original_image is not None:
                axs2[0].imshow(original_image)
                axs2[0].set_title("Original Image", fontsize=10, fontweight="bold")
            else:
                axs2[0].axis("off")
            axs2[0].axis("off")
            if gradcam_overlay is not None:
                axs2[1].imshow(gradcam_overlay)
                axs2[1].set_title("GradCAM++ Heatmap", fontsize=10, fontweight="bold")
            else:
                axs2[1].axis("off")
            axs2[1].axis("off")
            plt.tight_layout()
            pdf.savefig(fig2, dpi=150)
            plt.close(fig2)

    buf.seek(0)
    return buf.read()
