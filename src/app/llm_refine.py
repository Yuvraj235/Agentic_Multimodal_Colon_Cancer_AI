"""ColonAI — optional LLM-based rationale refinement.

⚠️ Important medical-AI policy:

The LLM is used **ONLY to rewrite our deterministic measurements into
a single fluent patient-friendly paragraph**. It never sees the raw
image, the patient's name, age, or symptoms. It cannot change the
predicted class, the confidence, or the urgency. If it tries (the
output disagrees with the deterministic prediction we passed in),
we throw the LLM output away and fall back to the deterministic text.

Provider
────────
We use **Groq** (https://groq.com/) — free tier, fast (~700 t/s),
OpenAI-compatible API, runs Llama-3.1-70B-Instruct. Set env var
GROQ_API_KEY to enable. Without the key, this module is a no-op.

Get a free key:
   1. https://console.groq.com/keys
   2. Click "Create API Key"
   3. Set the env var on the Streamlit app or HF Space settings
"""
from __future__ import annotations
import os, json, re
from typing import Dict, List, Optional

# Default model — Llama-3.1 8B is more than enough for sentence-level rewriting,
# and the 8B tier has much higher free-tier rate limits than 70B.
GROQ_MODEL_DEFAULT = "llama-3.1-8b-instant"
GROQ_ENDPOINT      = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT_SEC        = 12


def is_available() -> bool:
    """Return True if GROQ_API_KEY is set and the requests library is importable."""
    if not os.environ.get("GROQ_API_KEY"): return False
    try:
        import requests  # noqa
        return True
    except ImportError:
        return False


def refine_rationale(
    *,
    predicted_class:   str,
    confidence:        float,
    uncertainty:       float,
    safety_action:     str,                  # "show" | "abstain" | "reject"
    deterministic_bullets: List[str],
    differential:      Optional[List[Dict]] = None,
    is_hedged:         bool = False,
    hedge_reason:      Optional[str] = None,
    timeout:           int = TIMEOUT_SEC,
) -> Dict:
    """Refine our deterministic rationale bullets into one patient-friendly
    paragraph. Returns:
        {
          "refined_paragraph": str,           # the LLM output (or "" if unavailable)
          "fallback_used":     bool,          # True if LLM unavailable / rejected
          "fallback_reason":   str,           # why
          "model":             str,           # e.g. "llama-3.1-70b-versatile"
        }

    Guard rails (any failure → fallback):
       • LLM output mustn't change the predicted_class
       • LLM output mustn't invent a confidence percentage > our value + 5
       • LLM output mustn't be empty
       • LLM output mustn't contain "I am a doctor", "I diagnose", "definitely"
    """
    if not is_available():
        return {"refined_paragraph": "", "fallback_used": True,
                "fallback_reason": "GROQ_API_KEY not set",
                "model": ""}

    import requests
    api_key = os.environ["GROQ_API_KEY"]
    model = os.environ.get("GROQ_MODEL", GROQ_MODEL_DEFAULT)

    # Build a strictly-bounded prompt
    bullets_str = "\n".join(f"- {b}" for b in deterministic_bullets[:8])
    diff_str = ""
    if differential:
        diff_str = ("Top-3 differential (from the model, do not change): " +
                    " · ".join(f"{d['class']} {d['prob']*100:.0f}%"
                               for d in differential[:3]))
    hedge_str = ""
    if is_hedged and hedge_reason:
        hedge_str = f"Clinical hedge: {hedge_reason}"

    sys_prompt = (
        "You are a patient-friendly medical-explanation rewriter. You will be "
        "given the output of a colonoscopy AI in bullet form, with a confidence "
        "percentage and an action (show/abstain/reject). Rewrite the bullets "
        "into ONE paragraph (max 90 words) that a non-medical person can "
        "understand. STRICT RULES:\n"
        "  • DO NOT change the predicted class, the confidence percentage, "
        "    or the safety action.\n"
        "  • DO NOT claim certainty — use phrases like 'the AI suggests' / "
        "    'tends toward'.\n"
        "  • DO NOT mention drugs, medications, surgical procedures, or "
        "    specific dosages.\n"
        "  • DO NOT use the words 'definitely', 'I diagnose', 'I am a doctor'.\n"
        "  • DO end with: 'Please discuss this with a qualified clinician.'\n"
        "Return ONLY the paragraph — no preamble, no markdown, no bullets."
    )
    user_prompt = (
        f"Predicted class: {predicted_class}\n"
        f"Confidence: {confidence*100:.0f}%\n"
        f"Internal uncertainty score: {uncertainty:.2f}\n"
        f"Safety action: {safety_action}\n"
        f"{diff_str}\n{hedge_str}\n\n"
        f"Deterministic observations to rewrite:\n{bullets_str}\n\n"
        f"Now write the patient-friendly paragraph."
    )

    try:
        r = requests.post(
            GROQ_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type":  "application/json"},
            json={"model": model,
                  "messages": [{"role": "system", "content": sys_prompt},
                               {"role": "user",   "content": user_prompt}],
                  "temperature": 0.2,
                  "max_tokens":  256,
                  "stream":      False},
            timeout=timeout,
        )
        if r.status_code != 200:
            return {"refined_paragraph": "", "fallback_used": True,
                    "fallback_reason": f"Groq HTTP {r.status_code}: {r.text[:200]}",
                    "model": model}
        body = r.json()
        text = body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return {"refined_paragraph": "", "fallback_used": True,
                "fallback_reason": f"{type(e).__name__}: {e}",
                "model": model}

    # ── Guard-rail checks ──────────────────────────────────────────────
    text_lower = text.lower()
    if not text or len(text) < 30:
        return {"refined_paragraph": "", "fallback_used": True,
                "fallback_reason": "LLM output too short", "model": model}
    forbidden = ["i diagnose", "i am a doctor", "definitely", "you have ",
                 "you definitely", "100% confident"]
    for f in forbidden:
        if f in text_lower:
            return {"refined_paragraph": "", "fallback_used": True,
                    "fallback_reason": f"forbidden phrase '{f}'", "model": model}

    # Must keep the predicted class somewhere (or use a synonym we provide)
    safe_synonyms = {
        "polyps":            ["polyp", "growth on the bowel"],
        "uc-mild":           ["ulcerative colitis", "uc"],
        "uc-moderate-sev":   ["ulcerative colitis", "uc"],
        "barretts-esoph":    ["barrett", "oesophagus", "esophagus"],
        "therapeutic":       ["previous procedure", "post-treatment", "therapeutic"],
    }
    needles = [predicted_class.lower()] + safe_synonyms.get(predicted_class, [])
    if not any(n in text_lower for n in needles):
        return {"refined_paragraph": "", "fallback_used": True,
                "fallback_reason": "LLM dropped the predicted class", "model": model}

    # Confidence reality-check: LLM mustn't claim higher confidence than ours
    pct_match = re.search(r"(\d{1,3})\s*%", text)
    if pct_match:
        llm_pct = int(pct_match.group(1))
        if llm_pct > confidence * 100 + 5:
            return {"refined_paragraph": "", "fallback_used": True,
                    "fallback_reason":
                        f"LLM claimed {llm_pct}% > our {confidence*100:.0f}%",
                    "model": model}

    return {"refined_paragraph": text, "fallback_used": False,
            "fallback_reason": "", "model": model}


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (only runs if GROQ_API_KEY is set)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"GROQ_API_KEY set? {bool(os.environ.get('GROQ_API_KEY'))}")
    print(f"Available?        {is_available()}")
    if is_available():
        r = refine_rationale(
            predicted_class="polyps", confidence=0.87, uncertainty=0.12,
            safety_action="show",
            deterministic_bullets=[
                "The AI is confident in its reading (87%).",
                "It is focused on a small region (~6%) in the upper-left.",
                "Attention is tightly focused (87%) — concentrated.",
                "Shape is round and smooth (circularity 0.90), consistent with a sessile polyp.",
                "Dominant colour is salmon-pink.",
            ],
            differential=[{"class": "polyps", "prob": 0.87},
                          {"class": "therapeutic", "prob": 0.08}])
        print()
        print("REFINED:")
        print("  " + (r["refined_paragraph"] or "(empty)"))
        print(f"\nfallback_used={r['fallback_used']}  reason={r['fallback_reason']}")
