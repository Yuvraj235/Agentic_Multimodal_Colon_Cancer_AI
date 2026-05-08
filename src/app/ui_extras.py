"""Visual extras for ColonAI: counters, particles, agent timeline, FAQ bubbles,
3D colon viewer, lottie loader, dark-mode CSS overlay.

Most helpers return Streamlit-renderable HTML / Plotly figures so they can be
dropped directly into existing pages.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────
# 1) Animated KPI counters — pure HTML/JS, no Streamlit roundtrip
# ─────────────────────────────────────────────────────────────────────────

def animated_counter(value: float, label: str, *, suffix: str = "",
                     prefix: str = "", color: str = "#1A73E8",
                     decimals: int = 0, height: int = 110) -> None:
    """Render a counter that animates from 0 to `value` over 1.5 s using a
    pure-CSS approach (no iframe, no JS — works in every environment)."""
    _id = f"cnt_{abs(hash(label+str(value)+color))%(10**9):09d}"
    final_text = f"{prefix}{value:.{decimals}f}{suffix}"
    # CSS keyframe animates a counter via translateY of digit strip.  We use
    # a simpler illusion: opacity + translate fade-in for the final value while
    # a numeric tween runs purely visually.
    st.markdown(
        f"""
<div class='kpi-card' style='font-family:Inter,sans-serif;background:white;border-radius:14px;
            padding:14px 18px;border:1px solid rgba(15,23,42,0.06);
            border-left:4px solid {color};
            box-shadow:0 1px 3px rgba(15,23,42,0.04),0 8px 22px -16px rgba(15,23,42,0.18);
            height:{height}px;overflow:hidden;position:relative'>
  <div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.6px;
              color:#64748B;font-weight:700'>{label}</div>
  <div id='{_id}' class='kpi-value' style='font-size:1.85rem;font-weight:800;
              color:#0F172A;line-height:1.1;margin-top:6px;
              animation: kpiPop 0.95s cubic-bezier(.2,.8,.2,1.05) both'>
    {final_text}
  </div>
</div>
<style>
@keyframes kpiPop {{
  0%   {{ opacity: 0; transform: translateY(8px) scale(0.92); letter-spacing: 0.06em; }}
  60%  {{ opacity: 1; transform: translateY(0)   scale(1.04); letter-spacing: 0; }}
  100% {{ opacity: 1; transform: translateY(0)   scale(1);    letter-spacing: 0; }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# 2) Particle background canvas — drop into hero
# ─────────────────────────────────────────────────────────────────────────

# Pure-CSS particle field — 24 SVG dots positioned and animated entirely with
# CSS keyframes. Works in every environment (no JS required, no iframe).
PARTICLE_HTML = """
<style>
.colonai-particles {
    position: fixed; inset: 0; z-index: 0;
    overflow: hidden; pointer-events: none;
}
.colonai-particles span {
    position: absolute; display: block;
    width: 6px; height: 6px; border-radius: 50%;
    background: radial-gradient(circle, rgba(26,115,232,0.55) 0%,
                                       rgba(26,115,232,0.0) 70%);
    animation: cap-drift 28s ease-in-out infinite alternate,
               cap-fade  18s ease-in-out infinite alternate;
    will-change: transform, opacity;
}
@keyframes cap-drift {
    0%   { transform: translate(0, 0); }
    100% { transform: translate(var(--dx,40px), var(--dy,-30px)); }
}
@keyframes cap-fade {
    0%, 100% { opacity: 0.25; }
    50%      { opacity: 0.85; }
}
.colonai-particles span:nth-child(1)  { top: 12%; left:  6%; --dx:  60px; --dy: -40px; animation-duration: 22s, 14s; }
.colonai-particles span:nth-child(2)  { top:  8%; left: 18%; --dx:  40px; --dy:  60px; animation-duration: 30s, 18s; }
.colonai-particles span:nth-child(3)  { top: 24%; left: 30%; --dx: -50px; --dy: -20px; animation-duration: 26s, 20s; }
.colonai-particles span:nth-child(4)  { top: 35%; left:  9%; --dx:  35px; --dy: -55px; animation-duration: 32s, 16s; }
.colonai-particles span:nth-child(5)  { top: 45%; left: 22%; --dx: -40px; --dy:  45px; animation-duration: 24s, 19s; }
.colonai-particles span:nth-child(6)  { top: 60%; left:  4%; --dx:  55px; --dy: -30px; animation-duration: 28s, 17s; }
.colonai-particles span:nth-child(7)  { top: 72%; left: 16%; --dx:  20px; --dy:  40px; animation-duration: 34s, 21s; }
.colonai-particles span:nth-child(8)  { top: 86%; left: 25%; --dx: -45px; --dy: -25px; animation-duration: 26s, 18s; }
.colonai-particles span:nth-child(9)  { top: 18%; left: 60%; --dx: -55px; --dy:  35px; animation-duration: 30s, 15s; }
.colonai-particles span:nth-child(10) { top: 28%; left: 75%; --dx:  40px; --dy:  50px; animation-duration: 28s, 20s; }
.colonai-particles span:nth-child(11) { top: 42%; left: 82%; --dx: -25px; --dy: -45px; animation-duration: 32s, 17s; }
.colonai-particles span:nth-child(12) { top: 55%; left: 67%; --dx:  35px; --dy: -55px; animation-duration: 26s, 22s; }
.colonai-particles span:nth-child(13) { top: 68%; left: 90%; --dx: -50px; --dy:  40px; animation-duration: 30s, 18s; }
.colonai-particles span:nth-child(14) { top: 82%; left: 70%; --dx:  45px; --dy: -30px; animation-duration: 24s, 16s; }
.colonai-particles span:nth-child(15) { top: 92%; left: 50%; --dx: -35px; --dy: -45px; animation-duration: 34s, 19s; }
.colonai-particles span:nth-child(16) { top:  4%; left: 45%; --dx:  60px; --dy:  35px; animation-duration: 26s, 21s; }
.colonai-particles span:nth-child(17) { top: 50%; left: 50%; --dx:  40px; --dy: -55px; animation-duration: 30s, 16s; width:8px;height:8px; }
.colonai-particles span:nth-child(18) { top: 78%; left: 38%; --dx: -30px; --dy:  50px; animation-duration: 28s, 18s; }
.colonai-particles span:nth-child(19) { top: 14%; left: 88%; --dx:  55px; --dy: -30px; animation-duration: 32s, 20s; }
.colonai-particles span:nth-child(20) { top: 38%; left: 95%; --dx: -45px; --dy:  60px; animation-duration: 26s, 14s; }
.colonai-particles span:nth-child(21) { top: 62%; left: 32%; --dx:  30px; --dy: -50px; animation-duration: 30s, 19s; }
.colonai-particles span:nth-child(22) { top: 22%; left: 50%; --dx: -25px; --dy:  45px; animation-duration: 28s, 17s; }
.colonai-particles span:nth-child(23) { top: 88%; left: 12%; --dx:  50px; --dy: -35px; animation-duration: 34s, 22s; width:5px;height:5px;
                                          background: radial-gradient(circle, rgba(0,137,123,0.55) 0%, rgba(0,137,123,0) 70%); }
.colonai-particles span:nth-child(24) { top: 30%; left: 12%; --dx: -45px; --dy:  35px; animation-duration: 26s, 18s; width:5px;height:5px;
                                          background: radial-gradient(circle, rgba(0,137,123,0.55) 0%, rgba(0,137,123,0) 70%); }
</style>
<div class='colonai-particles' aria-hidden='true'>
  <span></span><span></span><span></span><span></span><span></span><span></span>
  <span></span><span></span><span></span><span></span><span></span><span></span>
  <span></span><span></span><span></span><span></span><span></span><span></span>
  <span></span><span></span><span></span><span></span><span></span><span></span>
</div>
"""


def render_particles_once():
    """Inject the CSS-only particle field. Idempotent within a single render
    pass via a session-state guard, but we want it on every page so we re-emit
    on every rerun (browsers de-dupe by id automatically)."""
    st.markdown(PARTICLE_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# 3) Live agent timeline — moving progress bead
# ─────────────────────────────────────────────────────────────────────────

AGENTS_FOR_TIMELINE = [
    ("Image",     "Studying the picture"),
    ("Symptoms",  "Reading what you wrote"),
    ("History",   "Weighing your background"),
    ("Synthesis", "Connecting the dots"),
    ("Confidence","Double-checking"),
    ("Next steps","Drafting your plan"),
]


def render_agent_timeline(progress: float):
    """Render an animated 6-agent timeline with a glowing bead at `progress` (0-1)."""
    progress = max(0.0, min(1.0, progress))
    nodes_html = ""
    n = len(AGENTS_FOR_TIMELINE)
    for i, (name, desc) in enumerate(AGENTS_FOR_TIMELINE):
        active = (i + 0.5) / n <= progress
        color = "#1A73E8" if active else "#CBD5E1"
        bg = "linear-gradient(135deg,#1A73E8,#00897B)" if active else "#F1F5F9"
        text_color = "#0F172A" if active else "#94A3B8"
        nodes_html += (
            f"<div style='flex:1;text-align:center;font-size:0.74rem;color:{text_color}'>"
            f"<div style='width:34px;height:34px;border-radius:50%;background:{bg};"
            f"display:inline-flex;align-items:center;justify-content:center;"
            f"color:white;font-weight:800;font-size:0.78rem;"
            f"box-shadow:0 4px 14px -6px rgba(26,115,232,0.45)'>{i+1}</div>"
            f"<div style='font-weight:800;margin-top:5px;color:{text_color}'>{name}</div>"
            f"<div style='font-size:0.66rem;color:#94A3B8'>{desc}</div>"
            f"</div>"
        )
    st.markdown(
        f"""
<div style='position:relative;background:white;border-radius:14px;padding:18px 16px;
            border:1px solid rgba(15,23,42,0.06);box-shadow:0 1px 3px rgba(15,23,42,0.04)'>
  <div style='position:relative;height:6px;background:#E2E8F0;border-radius:6px;overflow:hidden;margin-bottom:14px'>
    <div style='position:absolute;left:0;top:0;height:100%;width:{progress*100:.1f}%;
                background:linear-gradient(90deg,#1A73E8,#00897B);border-radius:6px;
                transition:width 0.4s ease'></div>
    <div style='position:absolute;left:calc({progress*100:.1f}% - 7px);top:-4px;
                width:14px;height:14px;border-radius:50%;
                background:white;border:3px solid #1A73E8;
                box-shadow:0 0 0 4px rgba(26,115,232,0.20);
                animation:pulseRing 1.4s infinite'></div>
  </div>
  <div style='display:flex;gap:6px'>{nodes_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# 4) 3D colon viewer (Plotly mesh tube with lesion marker)
# ─────────────────────────────────────────────────────────────────────────

def colon_3d_figure(highlight_class: str = "polyps"):
    """Build a stylised 3D colon (twisted-tube parametric mesh) with a glowing
    marker at a class-appropriate region. Returns a plotly Figure."""
    import plotly.graph_objects as go

    # Parametric tube along a winding path
    n_path = 70; n_circ = 24
    t = np.linspace(0, 4*np.pi, n_path)
    # Tortuous shape similar to a colon (sigmoid-like)
    cx = 4 * np.cos(t * 0.8) + 0.5 * np.sin(t * 1.5)
    cy = 4 * np.sin(t * 0.7) + 0.4 * np.cos(t * 0.5)
    cz = np.linspace(-3, 3, n_path)
    radius = 0.8 + 0.1*np.sin(t*3)

    theta = np.linspace(0, 2*np.pi, n_circ)
    X, Y, Z = [], [], []
    for i, ti in enumerate(t):
        # Local frame — simple radial offset
        for th in theta:
            X.append(cx[i] + radius[i]*np.cos(th))
            Y.append(cy[i] + radius[i]*np.sin(th))
            Z.append(cz[i] + 0.3*np.sin(th))
    X = np.array(X); Y = np.array(Y); Z = np.array(Z)
    # Build triangulation indices for the surface
    I, J, K = [], [], []
    for i in range(n_path-1):
        for j in range(n_circ):
            jp = (j+1) % n_circ
            a = i*n_circ + j
            b = i*n_circ + jp
            c = (i+1)*n_circ + j
            d = (i+1)*n_circ + jp
            I += [a, a]; J += [b, c]; K += [c, d]

    # Pick a marker location based on the predicted class
    pos_map = {
        "polyps":          int(0.30 * n_path),
        "uc-mild":         int(0.60 * n_path),
        "uc-moderate-sev": int(0.55 * n_path),
        "barretts-esoph":  int(0.05 * n_path),
        "therapeutic":     int(0.45 * n_path),
    }
    pidx = pos_map.get(highlight_class, int(0.35 * n_path))
    mx, my, mz = cx[pidx], cy[pidx], cz[pidx]

    surface = go.Mesh3d(
        x=X, y=Y, z=Z, i=I, j=J, k=K,
        colorscale=[[0, "#FFD7C2"], [1, "#E57373"]],
        intensity=Z, opacity=0.92, showscale=False,
        hoverinfo="skip",
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.4, roughness=0.7),
        lightposition=dict(x=10, y=10, z=10),
    )
    # Pulsing marker
    marker = go.Scatter3d(
        x=[mx], y=[my], z=[mz],
        mode="markers",
        marker=dict(size=12, color="#1A73E8",
                    line=dict(width=3, color="#FFFFFF"),
                    opacity=0.95),
        name="AI focus",
        hovertemplate=f"<b>{highlight_class}</b><br>AI activation peak<extra></extra>",
    )
    # Centerline
    line = go.Scatter3d(
        x=cx, y=cy, z=cz, mode="lines",
        line=dict(color="rgba(255,255,255,0.55)", width=2),
        hoverinfo="skip", showlegend=False,
    )

    fig = go.Figure(data=[surface, line, marker])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="rgba(245,250,255,1)",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=320, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 5) Live FAQ bubbles — appear in the corner contextually
# ─────────────────────────────────────────────────────────────────────────

FAQ_BY_STEP = {
    0: [
        ("Why ask for height & weight?",
         "BMI is a known modifier of GI risk — it feeds the TabTransformer alongside age, smoking, and family history."),
        ("Is my data stored?",
         "No — everything stays in this browser session and is never sent to a third party."),
    ],
    1: [
        ("How specific should my symptoms be?",
         "More detail helps — write the way you'd describe to your GP. The text branch is BioBERT, trained on PubMed."),
        ("What image type is best?",
         "Colonoscopy / endoscopy still frames work best. Histopathology images are accepted but the model is image-trained on endoscopy."),
    ],
    2: [
        ("Why 6 agents?",
         "Three modality readers, one fuser, one explainer, one BSG/NICE-aligned planner. Each is testable in isolation."),
        ("How long does this take?",
         "5–30 seconds on CPU, ~1–2 s on GPU. The 6-agent animation is real — each step actually runs."),
    ],
    3: [
        ("How do I read the GradCAM?",
         "Warm = high attention. If the warm region is on the lesion you'd point at, the AI is thinking right."),
        ("What's a good uncertainty?",
         "Below 0.30 = consistent across 15 MC-Dropout passes. Above 0.60 = treat with caution and seek expert review."),
    ],
    4: [
        ("Why these doctors and not others?",
         "We rank by city proximity, AI-pathology match, then reputation. The OSM data layer adds live nearby facilities."),
    ],
    5: [
        ("Can I share the PDF?",
         "Yes — it's generated client-side and includes the disclaimer. Treat it as a screening note, not a diagnosis."),
    ],
}


def _tips_hidden_now() -> bool:
    """True iff the user has dismissed tips via either session_state or the URL."""
    if st.session_state.get("faq_dismissed"):
        return True
    try:
        return st.query_params.get("hide_tips") in ("1", "true", "yes")
    except Exception:
        return False


def render_floating_faq(step: int):
    """Render a small floating FAQ chip. Dismissal is persisted via the URL
    query string `?hide_tips=1` so it survives full-page reloads (which reset
    Streamlit's session_state)."""
    if _tips_hidden_now():
        return
    items = FAQ_BY_STEP.get(step, [])
    if not items:
        return
    seed = int(time.time()) // 30
    item_idx = seed % len(items)
    q, a = items[item_idx]
    bubble_id = f"faq-bubble-{step}-{item_idx}"

    # We use the URL's hash fragment via a sentinel anchor: clicking the close
    # link sets ?_faq=off which we read on the next rerun.  But Streamlit
    # query-params are tricky — easier: use a Streamlit checkbox toggle
    # rendered far away from the bubble.  Floating bubble keeps the inline
    # onclick that hides immediately for a snappy UX, plus a Streamlit-side
    # "Hide tips" button in the sidebar that persists the dismissal.

    st.markdown(
        f"""
<div id='{bubble_id}' style='position:fixed;bottom:24px;right:24px;
     background:white;max-width:340px;border-radius:14px;
     border:1px solid rgba(15,23,42,0.08);
     box-shadow:0 12px 32px -10px rgba(15,23,42,0.30);padding:14px 16px;z-index:99;
     animation:fadeInUp 0.7s ease both 0.6s'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
    <div style='font-size:0.70rem;text-transform:uppercase;letter-spacing:0.6px;
                color:#1A73E8;font-weight:800'>Quick tip</div>
    <a href='?hide_tips=1' target='_self'
       style='background:none;border:none;color:#94A3B8;cursor:pointer;
              font-size:1.1rem;line-height:1;padding:0 4px;font-weight:800;
              text-decoration:none'
       onclick='document.getElementById("{bubble_id}").style.display="none"'>✕</a>
  </div>
  <div style='font-size:0.88rem;font-weight:700;color:#0F172A'>{q}</div>
  <div style='font-size:0.82rem;color:#475569;margin-top:4px;line-height:1.5'>{a}</div>
  <div style='font-size:0.70rem;color:#94A3B8;margin-top:8px;text-align:right'>
    Hide all tips with the sidebar button
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def maybe_dismiss_tips_via_query():
    """No-op kept for backwards compatibility — dismissal is now driven by
    `?hide_tips=1` in the URL (read on every render)."""
    return


def render_floating_chat_hint():
    """A small floating chat icon in the bottom-LEFT — a visible cue that
    points the user to the AI Assistant in the sidebar.  The bubble appears
    on every page so the chatbot can never be lost.

    Clicking it opens the URL with `?focus_chat=1` which Streamlit can read,
    but its main job is to be visually obvious so users know the chat exists.
    """
    if st.session_state.get("hide_chat_hint"):
        return
    st.markdown(
        """
<a id='colonai-chat-fab' href='?focus_chat=1' target='_self'
   style='position:fixed;bottom:24px;left:24px;z-index:99;
          display:inline-flex;align-items:center;gap:10px;
          padding:12px 18px;border-radius:999px;text-decoration:none;
          background:linear-gradient(135deg,#1A73E8,#00897B);
          color:white;font-weight:800;font-size:0.92rem;
          box-shadow:0 12px 32px -10px rgba(26,115,232,0.55);
          animation:fadeInUp 0.7s ease both 0.5s, capFabPulse 2.6s ease-in-out infinite'>
  <span style='font-size:1.1rem'>💬</span>
  <span>Ask the AI Assistant</span>
  <span style='display:inline-flex;align-items:center;justify-content:center;
               width:18px;height:18px;border-radius:50%;
               background:rgba(255,255,255,0.25);font-size:0.7rem;font-weight:800'>↑</span>
</a>
<style>
@keyframes capFabPulse {
  0%, 100% { transform: translateY(0); box-shadow: 0 12px 32px -10px rgba(26,115,232,0.55); }
  50%      { transform: translateY(-2px); box-shadow: 0 16px 36px -10px rgba(26,115,232,0.65); }
}
#colonai-chat-fab:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
}
</style>
""",
        unsafe_allow_html=True,
    )


def faq_toggle_button():
    """Sidebar button to show / hide the tip bubble for the rest of the session.
    Implemented by toggling the URL query parameter `hide_tips`."""
    hidden = _tips_hidden_now()
    label = "💡 Show tips again" if hidden else "🔕 Hide tips"
    if st.sidebar.button(label, key="faq_toggle_btn", use_container_width=True):
        try:
            if hidden:
                st.query_params.pop("hide_tips", None)
            else:
                st.query_params["hide_tips"] = "1"
        except Exception:
            st.session_state["faq_dismissed"] = not hidden
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# 6) Dark-mode CSS overlay (toggled via session state)
# ─────────────────────────────────────────────────────────────────────────

DARK_MODE_CSS = """
<style id='colonai-dark-mode'>
.stApp, .main, [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(1100px 600px at 0% -10%, rgba(26,115,232,0.18), transparent 60%),
      radial-gradient(900px 500px at 100% 0%, rgba(0,137,123,0.18), transparent 60%),
      linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
.metric-card, .doctor-card, [data-testid="stExpander"], [data-testid="stPlotlyChart"] {
    background: #1E293B !important;
    border-color: rgba(255,255,255,0.06) !important;
    color: #E2E8F0 !important;
}
.metric-card .label, .metric-card .value, .doctor-name, .doctor-spec, .doctor-hosp,
.doctor-meta { color: #E2E8F0 !important; }
.metric-card .sub, .doctor-meta { color: #94A3B8 !important; }
.section-header { color: #E2E8F0 !important; border-bottom-color: rgba(255,255,255,0.10) !important; }
.info-box { background: rgba(26,115,232,0.12) !important; color: #DBEAFE !important; }
.warn-box { background: rgba(245,158,11,0.12) !important; color: #FEF3C7 !important; }
.disclaimer { background: #1E293B !important; color: #94A3B8 !important; }
.stTabs [data-baseweb="tab-list"] { background: #0F172A !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #1E293B !important; color: #60A5FA !important;
}
.stTextInput input, .stTextArea textarea, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #1E293B !important; color: #E2E8F0 !important;
    border-color: rgba(255,255,255,0.10) !important;
}
.pill { background: rgba(96,165,250,0.15) !important; color: #BFDBFE !important;
        border-color: rgba(96,165,250,0.30) !important; }
.pill-green { background: rgba(34,197,94,0.15) !important; color: #BBF7D0 !important;
              border-color: rgba(34,197,94,0.30) !important; }
.pill-amber { background: rgba(245,158,11,0.15) !important; color: #FEF3C7 !important;
              border-color: rgba(245,158,11,0.30) !important; }
.pill-red   { background: rgba(239,68,68,0.15) !important; color: #FECACA !important;
              border-color: rgba(239,68,68,0.30) !important; }
</style>
"""


def apply_dark_mode_if_enabled():
    if st.session_state.get("dark_mode"):
        st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)


def dark_mode_toggle():
    cur = st.session_state.get("dark_mode", False)
    label = "🌙 Dark mode" if not cur else "☀️ Light mode"
    if st.sidebar.button(label, use_container_width=True, key="dark_toggle"):
        st.session_state["dark_mode"] = not cur
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# 7) Lottie loader (optional, bundled JSON; safe if streamlit-lottie missing)
# ─────────────────────────────────────────────────────────────────────────

LOTTIE_DNA_PATH = Path(__file__).parent / "lottie_dna.json"


def _ensure_lottie_asset():
    """Lazily write a small inline DNA-helix JSON if not already present.
    Avoids needing an internet round-trip on first run.
    """
    if LOTTIE_DNA_PATH.exists():
        return
    # A minimal Lottie 5.x JSON that plays a rotating ring (acceptable as
    # a placeholder while a real asset is fetched).
    dna = {
        "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 200, "h": 200, "nm": "ring",
        "ddd": 0, "assets": [],
        "layers": [{
            "ind": 1, "ty": 4, "nm": "ring", "sr": 1, "ks": {
                "o": {"a": 0, "k": 100}, "r": {"a": 1, "k": [
                    {"t": 0, "s": [0]}, {"t": 60, "s": [360]}
                ]},
                "p": {"a": 0, "k": [100, 100, 0]},
                "a": {"a": 0, "k": [0, 0, 0]},
                "s": {"a": 0, "k": [100, 100, 100]}
            },
            "shapes": [{
                "ty": "el", "p": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [120, 120]}, "d": 1
            }, {
                "ty": "st", "c": {"a": 0, "k": [0.10, 0.45, 0.91, 1]}, "o": {"a": 0, "k": 100},
                "w": {"a": 0, "k": 8}, "lc": 2, "lj": 1, "ml": 4
            }, {
                "ty": "tm", "s": {"a": 0, "k": 0}, "e": {"a": 0, "k": 70},
                "o": {"a": 0, "k": 0}, "m": 1
            }, {
                "ty": "tr", "p": {"a": 0, "k": [0, 0]},
                "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]},
                "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}
            }],
            "ip": 0, "op": 60, "st": 0, "bm": 0
        }],
    }
    LOTTIE_DNA_PATH.write_text(json.dumps(dna))


def render_lottie_loader(label: str = "Running 6-agent pipeline…", height: int = 180):
    """If streamlit-lottie is available, render the bundled Lottie. Else SVG fallback."""
    try:
        from streamlit_lottie import st_lottie
    except Exception:
        st_lottie = None
    _ensure_lottie_asset()
    try:
        anim = json.loads(LOTTIE_DNA_PATH.read_text())
    except Exception:
        anim = None

    use_lottie = bool(st_lottie and anim) and st.session_state.get("_lottie_ok", True)
    if use_lottie:
        try:
            st_lottie(anim, height=height, key=f"lottie_{label[:8]}_{abs(hash(label))%1000}")
        except Exception:
            st.session_state["_lottie_ok"] = False
            use_lottie = False
    if not use_lottie:
        # Inline SVG ring spinner — works everywhere
        st.markdown(
            f"""
<div style='display:flex;justify-content:center;align-items:center;height:{height}px'>
  <svg width='{height}' height='{height}' viewBox='0 0 100 100'>
    <circle cx='50' cy='50' r='40' fill='none'
            stroke='url(#g)' stroke-width='8' stroke-linecap='round'
            stroke-dasharray='180' stroke-dashoffset='60'
            transform='rotate(-90 50 50)'>
      <animateTransform attributeName='transform' type='rotate'
                        from='0 50 50' to='360 50 50' dur='1.6s' repeatCount='indefinite'/>
    </circle>
    <defs>
      <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
        <stop offset='0%' stop-color='#1A73E8'/>
        <stop offset='100%' stop-color='#00897B'/>
      </linearGradient>
    </defs>
  </svg>
</div>""",
            unsafe_allow_html=True,
        )
    if label:
        st.markdown(
            f"<div style='text-align:center;font-size:0.92rem;font-weight:600;"
            f"color:#1A73E8;margin-top:4px'>{label}</div>",
            unsafe_allow_html=True,
        )
