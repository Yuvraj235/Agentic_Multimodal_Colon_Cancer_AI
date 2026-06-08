"""ColonAI — patient-friendly UI helpers.

Three things this module gives the rest of the app:

  1. `plain_english_diagnosis()`
        Converts a 5-class pathology label + confidence into a sentence
        a 14-year-old can understand. Avoids any clinical jargon
        (no "ulcerative colitis", no "Barrett's", no "polyp" without
        an inline definition).

  2. `accessibility_css()`
        Returns a CSS string that, when injected via st.markdown(...,
        unsafe_allow_html=True), applies:
           • Larger base font (18px)
           • High-contrast colour palette
           • Dyslexia-friendly font (OpenDyslexic if installed, fallback
             to a clean sans-serif)
           • Generous line-height (1.7) and letter-spacing
           • Bigger tap-targets (≥44 px)
        Behind an opt-in toggle so power users keep the dense layout.

  3. `verdict_card_html()`
        Renders the result page's top banner: SHOW / ABSTAIN / REJECT
        with patient-appropriate copy (no flag codes, no jargon).

  4. `rationale_card_html()`
        Bulleted "why this answer" panel that consumes the structured
        rationale list from cross_check.CrossCheckReport.
"""
from __future__ import annotations
from typing import List, Optional, Dict
from src.app.security import escape_html as _esc


PLAIN_NAMES = {
    "polyps":          "growth on the bowel wall (a polyp)",
    "uc-mild":         "mild inflammation of the bowel lining",
    "uc-moderate-sev": "moderate-to-severe inflammation of the bowel lining",
    "barretts-esoph":  "changes in the food pipe (Barrett's-type cells)",
    "therapeutic":     "site where a previous procedure was performed",
}

NEXT_STEPS_PLAIN = {
    "polyps":          "Your doctor will usually want to remove this and look at it under a microscope.",
    "uc-mild":         "Often managed with diet changes and/or anti-inflammatory medicine.",
    "uc-moderate-sev": "Usually needs prompt review by a gastroenterologist.",
    "barretts-esoph":  "Surveillance endoscopies every 1–3 years are recommended.",
    "therapeutic":     "Routine follow-up — no immediate action usually needed.",
}


def plain_english_diagnosis(pathology_class: str, confidence: float) -> Dict[str, str]:
    """Return a dict with `name`, `next_steps`, `confidence_phrase`.

    Never returns medical jargon a layperson would not recognise.
    """
    known = pathology_class in PLAIN_NAMES
    if not known:
        # Out-of-distribution / unrecognised class — NEVER claim confidence in a
        # label we don't have. The caller drops the confidence sentence entirely.
        return {
            "name":              "a finding outside its trained categories",
            "next_steps":        "Please have a clinician review this directly.",
            "confidence_phrase": "could not confidently classify",
            "confidence_pct":    f"{confidence * 100:.0f}%",
            "known":             False,
        }
    name = PLAIN_NAMES[pathology_class]
    if confidence >= 0.85:
        conf_phrase = "is quite sure it sees"
    elif confidence >= 0.70:
        conf_phrase = "thinks it most likely sees"
    else:
        conf_phrase = "is not very confident, but is leaning toward"
    return {
        "name":              name,
        "next_steps":        NEXT_STEPS_PLAIN.get(pathology_class,
                                                   "Please discuss with your doctor."),
        "confidence_phrase": conf_phrase,
        "confidence_pct":    f"{confidence * 100:.0f}%",
        "known":             True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Accessibility CSS
# ─────────────────────────────────────────────────────────────────────────────
def accessibility_css() -> str:
    """Inject this with st.markdown(accessibility_css(), unsafe_allow_html=True)
    when the user enables Accessibility Mode in the sidebar."""
    return """
    <style id="colonai-accessibility-css">
      /* Larger base font for legibility */
      html, body, [class^="st-"] {
        font-size: 18px !important;
        line-height: 1.7 !important;
        letter-spacing: 0.01em !important;
      }
      h1 { font-size: 2.2rem !important; }
      h2 { font-size: 1.8rem !important; }
      h3 { font-size: 1.5rem !important; }
      h4 { font-size: 1.25rem !important; }
      p, li, label, .stMarkdown { font-size: 1.05rem !important; }

      /* Higher-contrast palette */
      body, .main, [data-testid="stAppViewContainer"] {
        background: #FFFFFF !important;
        color: #0B1220 !important;
      }
      a, [data-testid="stMarkdown"] a {
        color: #0B5FFF !important;
        text-decoration: underline !important;
      }

      /* Larger tap targets — buttons, inputs */
      button, [role="button"], .stButton > button {
        min-height: 48px !important;
        font-size: 1.05rem !important;
        padding: 12px 18px !important;
      }
      input, textarea, select, [data-baseweb="select"] {
        min-height: 44px !important;
        font-size: 1.05rem !important;
      }

      /* Strong focus ring for keyboard users */
      *:focus-visible {
        outline: 3px solid #FFB300 !important;
        outline-offset: 2px !important;
      }

      /* Dyslexia-friendly font when available */
      body, .main, h1, h2, h3, h4, p, li {
        font-family: "OpenDyslexic", "Atkinson Hyperlegible",
                     -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Helvetica, Arial, sans-serif !important;
      }
    </style>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Verdict banner — replaces flag-codes with plain language
# ─────────────────────────────────────────────────────────────────────────────
def verdict_card_html(safety_verdict: Dict, plain_dx: Optional[Dict] = None) -> str:
    """Build the top banner of the results page based on the safety policy
    verdict. All strings are escaped before interpolation.
    """
    if not safety_verdict: return ""
    action = safety_verdict.get("action", "show")
    flags  = safety_verdict.get("flags", []) or []

    if action == "reject":
        bg = "linear-gradient(135deg,#FEE2E2 0%,#FECACA 100%)"
        bd = "#DC2626"; fg = "#7F1D1D"; icon = "🚫"
        title  = "We can't analyse this image"
        body   = ("The picture you uploaded does not look like a colonoscopy "
                  "or endoscopy frame. Please upload a real medical image, "
                  "or try one of the example cases.")
    elif action == "abstain":
        bg = "linear-gradient(135deg,#FEF3C7 0%,#FDE68A 100%)"
        bd = "#D97706"; fg = "#78350F"; icon = "⚠️"
        title  = "Please ask a doctor to review this"
        body   = ("The AI isn't confident enough to give a clear answer for "
                  "this image. This isn't a sign that something is wrong — "
                  "it just means a trained doctor should look at it.")
    else:
        bg = "linear-gradient(135deg,#DCFCE7 0%,#BBF7D0 100%)"
        bd = "#16A34A"; fg = "#14532D"; icon = "✅"
        title  = "AI analysis complete"
        body   = "All our internal safety checks passed."
        if plain_dx:
            if plain_dx.get("known", True):
                body += (f" The AI {_esc(plain_dx['confidence_phrase'])} "
                         f"<b>{_esc(plain_dx['name'])}</b> "
                         f"({_esc(plain_dx['confidence_pct'])} confidence).")
            else:
                # Unknown/out-of-distribution class — no fake confidence claim.
                body += (" The AI <b>could not confidently classify</b> this finding, "
                         "so it is flagging it for a clinician to review directly.")

    return f"""
    <div role="alert" aria-live="polite"
         style="background:{bg};border:2px solid {bd};border-radius:14px;
                padding:18px 22px;box-shadow:0 4px 14px rgba(0,0,0,0.06);
                margin:14px 0;">
      <div style="font-size:1.25rem;font-weight:800;color:{fg};margin-bottom:6px;">
        {icon} {_esc(title)}
      </div>
      <div style="color:{fg};font-size:1rem;line-height:1.55;">
        {body}
      </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# "Why this answer" rationale card
# ─────────────────────────────────────────────────────────────────────────────
def rationale_card_html(rationale_lines: List[str],
                        flags: Optional[List[str]] = None) -> str:
    """Render the cross-check rationale + flags as a patient-friendly card.

    Each `rationale_lines` entry is a sentence describing what the AI
    actually looked at (lesion size, location, attribution agreement).
    """
    if not rationale_lines and not flags: return ""
    bullets = "".join(f"<li style='margin:6px 0;'>{_esc(r)}</li>"
                      for r in rationale_lines or [])
    flag_text = ""
    if flags:
        flag_text = (
            "<div style='margin-top:10px;padding:10px 14px;background:#FEF3C7;"
            "border-left:4px solid #D97706;border-radius:8px;font-size:0.92rem;"
            "color:#78350F;'>"
            "<b>Cross-check flags:</b><ul style='margin:4px 0 0 18px;padding:0;'>"
            + "".join(f"<li>{_esc(f.replace('_',' '))}</li>" for f in flags)
            + "</ul></div>")
    return f"""
    <div style="background:#FFF;border:1px solid #E2E8F0;border-radius:14px;
                padding:18px 22px;margin:14px 0;
                box-shadow:0 2px 8px rgba(15,23,42,0.04);">
      <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.7px;
                  color:#0B5FFF;font-weight:800;margin-bottom:8px;">
        Why the AI reached this answer
      </div>
      <ul style="font-size:1rem;color:#1F2937;line-height:1.55;
                 margin:0 0 0 18px;padding:0;">
        {bullets}
      </ul>
      {flag_text}
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(plain_english_diagnosis("polyps", 0.87))
    print(plain_english_diagnosis("uc-mild", 0.62))
    print(verdict_card_html({"action": "abstain", "flags": ["low_confidence"]})[:200], "…")
    print(rationale_card_html(["Lesion shape is round.",
                                "GradCAM and IG agree on the centre."])[:200], "…")
