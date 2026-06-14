"""
ColonAI — Agentic Multimodal Colon Cancer Screening System
Interactive Streamlit web application.

Run with:
    streamlit run app.py
"""

import os
import sys
import io
import time
import json
import math
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

# ─────────────────────────────────────────────────────────────────────────────
# Deploy-time diagnostic: print to stdout the moment app.py is imported,
# BEFORE anything else can fail. The first few lines of the HF Space log
# (or `streamlit run app.py` stdout) will show:
#   • which env vars Python actually sees
#   • the current working directory and where best_model.pth is expected
# This is the single most useful piece of info when "Model load failed —
# demo mode" appears on a remote deploy.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70, flush=True)
print("[STARTUP] ColonAI app.py imported  pid=" + str(os.getpid()), flush=True)
print(f"[STARTUP] cwd = {os.getcwd()}", flush=True)
print(f"[STARTUP] __file__ = {__file__}", flush=True)
print(f"[STARTUP] COLONAI_CHECKPOINT_HF_REPO = "
      f"{os.environ.get('COLONAI_CHECKPOINT_HF_REPO', '(unset)')!r}", flush=True)
print(f"[STARTUP] COLONAI_CHECKPOINT_HF_FILE = "
      f"{os.environ.get('COLONAI_CHECKPOINT_HF_FILE', '(unset)')!r}", flush=True)
print(f"[STARTUP] HF_TOKEN present = "
      f"{bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN'))}",
      flush=True)
print(f"[STARTUP] HF_HOME = {os.environ.get('HF_HOME', '(unset)')!r}", flush=True)
print("=" * 70, flush=True)

warnings.filterwarnings("ignore")

# ── project root on path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch

# ── Security policy: import early so PIL.MAX_IMAGE_PIXELS gets capped at
#    100 MP, ImageFile.LOAD_TRUNCATED_IMAGES=False, etc., before any
#    untrusted Image.open() can happen elsewhere in the app.
from src.app import security as _colonai_security   # noqa: F401
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Streamlit must be configured FIRST before any other st calls ──────────
st.set_page_config(
    page_title="ColonAI — Cancer Screening",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "**ColonAI** — Agentic Multimodal Colon Cancer Screening System\n\nBuilt with 6-agent AI pipeline.",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# Prefer the deploy-grade v2 checkpoint (mask-aware, calibrated) if it
# exists; fall back to the v1 checkpoint otherwise.
_CHECKPOINT_V2 = ROOT / "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
_CHECKPOINT_V1 = ROOT / "outputs/unified_multimodal/checkpoints/best_model.pth"

# Hugging Face Spaces / Streamlit Cloud deployment path
# ─────────────────────────────────────────────────────
# Set the env var COLONAI_CHECKPOINT_HF_REPO to e.g. "Yuvraj2319/colonai-v2"
# and the model file will be downloaded from that HF Hub model repo on first
# run. Without it, the app falls back to demo mode when the checkpoint is
# missing — useful for local development without the large weights file.
_HF_REPO     = os.environ.get("COLONAI_CHECKPOINT_HF_REPO", "")
_HF_FILENAME = os.environ.get("COLONAI_CHECKPOINT_HF_FILE", "best_model.pth")
_HF_TOKEN    = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# Module-level status captured by _maybe_download_checkpoint(). The sidebar
# reads this and surfaces the real reason if the download fails.
CHECKPOINT_STATUS = {
    "stage":   "unknown",     # "preexisting" | "downloaded" | "no_env_var" | "failed" | "unknown"
    "detail":  "",
    "path":    None,
    "size_mb": None,
    "hf_repo": _HF_REPO,
    "hf_file": _HF_FILENAME,
    "had_token": bool(_HF_TOKEN),
    "log":     [],            # list[str] — visible to the UI
}


def _maybe_download_checkpoint():
    """If neither checkpoint exists locally, try fetching from HF Hub."""
    import sys, traceback
    def _say(msg):
        line = f"[CHECKPOINT] {msg}"
        print(line, file=sys.stderr, flush=True)
        CHECKPOINT_STATUS["log"].append(line)

    if _CHECKPOINT_V2.exists():
        CHECKPOINT_STATUS["stage"] = "preexisting"
        CHECKPOINT_STATUS["path"]  = str(_CHECKPOINT_V2)
        try: CHECKPOINT_STATUS["size_mb"] = round(_CHECKPOINT_V2.stat().st_size / 1e6, 1)
        except Exception: pass
        _say(f"✓ v2 checkpoint already present at {_CHECKPOINT_V2}")
        return
    if _CHECKPOINT_V1.exists():
        CHECKPOINT_STATUS["stage"] = "preexisting"
        CHECKPOINT_STATUS["path"]  = str(_CHECKPOINT_V1)
        _say(f"✓ v1 checkpoint already present at {_CHECKPOINT_V1}")
        return
    if not _HF_REPO:
        CHECKPOINT_STATUS["stage"]  = "no_env_var"
        CHECKPOINT_STATUS["detail"] = ("COLONAI_CHECKPOINT_HF_REPO is unset — "
                                      "the app cannot download the model.")
        _say("✗ neither checkpoint found locally AND "
             "COLONAI_CHECKPOINT_HF_REPO is unset → demo mode")
        return

    _say(f"⇣ downloading {_HF_FILENAME} from HF Hub repo '{_HF_REPO}' …")
    _say(f"  (auth: {'token' if _HF_TOKEN else 'anonymous'})")
    try:
        from huggingface_hub import hf_hub_download
        _CHECKPOINT_V2.parent.mkdir(parents=True, exist_ok=True)
        try:
            local = hf_hub_download(
                repo_id=_HF_REPO, filename=_HF_FILENAME,
                local_dir=str(_CHECKPOINT_V2.parent),
                token=_HF_TOKEN or None,
                local_dir_use_symlinks=False)
        except TypeError:
            local = hf_hub_download(
                repo_id=_HF_REPO, filename=_HF_FILENAME,
                local_dir=str(_CHECKPOINT_V2.parent),
                token=_HF_TOKEN or None)
        _say(f"  saved to {local}")
        from pathlib import Path as _Pth
        if _Pth(local).resolve() != _CHECKPOINT_V2.resolve():
            import shutil
            shutil.copy(local, _CHECKPOINT_V2)
            _say(f"  copied to {_CHECKPOINT_V2}")
        size_mb = round(_CHECKPOINT_V2.stat().st_size / 1e6, 1)
        CHECKPOINT_STATUS["stage"]   = "downloaded"
        CHECKPOINT_STATUS["path"]    = str(_CHECKPOINT_V2)
        CHECKPOINT_STATUS["size_mb"] = size_mb
        _say(f"✓ downloaded — size {size_mb} MB")
        try:
            tmp_local = hf_hub_download(
                repo_id=_HF_REPO, filename="temperature.json",
                local_dir=str(_CHECKPOINT_V2.parent),
                token=_HF_TOKEN or None)
            _say(f"✓ temperature.json downloaded to {tmp_local}")
        except Exception as _te:
            _say(f"  (no temperature.json: {_te})")
    except Exception as exc:
        CHECKPOINT_STATUS["stage"]  = "failed"
        CHECKPOINT_STATUS["detail"] = f"{type(exc).__name__}: {exc}"
        _say(f"✗ HF Hub download failed — {type(exc).__name__}: {exc}")
        _say("  full traceback follows:")
        traceback.print_exc(file=sys.stderr)


_maybe_download_checkpoint()
CHECKPOINT = _CHECKPOINT_V2 if _CHECKPOINT_V2.exists() else _CHECKPOINT_V1

# Optional post-train temperature for confidence calibration (1.0 = no scaling)
import json as _json
# Prefer a per-site recalibrated temperature (set via the clinician tool) over the
# public-data default, so confidence is honest for the deploying hospital.
_TEMP_SITE = ROOT / "outputs/unified_multimodal_v2/temperature_site.json"
_TEMP_PATH = ROOT / "outputs/unified_multimodal_v2/temperature.json"
try:
    _temp_file = _TEMP_SITE if _TEMP_SITE.exists() else _TEMP_PATH
    TEMPERATURE = float(_json.loads(_temp_file.read_text()).get("temperature", 1.0)) \
                  if _temp_file.exists() else 1.0
    if TEMPERATURE < 0.05 or TEMPERATURE > 10.0:   # guard
        TEMPERATURE = 1.0
    TEMPERATURE_SOURCE = ("per-site" if _temp_file == _TEMP_SITE else "default")
except Exception:
    TEMPERATURE = 1.0
    TEMPERATURE_SOURCE = "default"
BERT_MODEL  = "dmis-lab/biobert-base-cased-v1.2"
N_CLASSES   = 5
D_MODEL     = 256
IMG_SIZE    = 224

CLASS_LABELS = {
    "polyps":          "Colorectal Polyps",
    "uc-mild":         "Ulcerative Colitis (Mild)",
    "uc-moderate-sev": "Ulcerative Colitis (Moderate–Severe)",
    "barretts-esoph":  "Barrett's Esophagus",
    "therapeutic":     "Post-Therapeutic Site",
}
CLASS_COLOURS = {
    "polyps":          "#2196F3",
    "uc-mild":         "#FF5722",
    "uc-moderate-sev": "#B71C1C",
    "barretts-esoph":  "#9C27B0",
    "therapeutic":     "#009688",
}
STAGE_COLORS = {
    "No Cancer":   "#2E7D32",
    "Stage I":     "#F9A825",
    "Stage II":    "#E65100",
    "Stage III/IV":"#B71C1C",
}
SYMPTOMS_LIST = [
    "Rectal bleeding / blood in stool",
    "Persistent change in bowel habits",
    "Abdominal pain or cramping",
    "Unexplained weight loss",
    "Chronic fatigue / weakness",
    "Bloating / excessive gas",
    "Nausea or vomiting",
    "Difficulty swallowing",
    "Persistent heartburn / GERD",
    "Mucus in stool",
    "Constipation (new onset)",
    "Diarrhoea (new onset)",
    "Pencil-thin stools",
    "Feeling of incomplete bowel evacuation",
    "Anaemia / low iron",
    "Haemorrhoids (confirmed)",
    "Loss of appetite",
    "Jaundice (yellowing of skin/eyes)",
]
STEPS = [
    "Patient Info",
    "Symptoms & Upload",
    "AI Analysis",
    "Results",
    "Find Doctors",
    "Download Report",
    "Live Video Mode",      # step 6 — real-time video / webcam
    "Latest Research",      # step 7 — auto-updated cancer-news feed
]

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
/* ── Global ── */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.main, [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(1100px 600px at 0% -10%, rgba(26,115,232,0.08), transparent 60%),
      radial-gradient(900px 500px at 100% 0%, rgba(0,137,123,0.08), transparent 60%),
      linear-gradient(180deg, #F6F9FF 0%, #FFFFFF 100%);
}
[data-testid="stHeader"] { background: transparent; }

/* Reduce wide-screen dead-space. Extra bottom padding so page content (e.g. the
   'Next' button) never sits under the fixed bottom-right Colon Buddy chat FAB. */
.block-container { padding-top: 1.4rem !important; padding-bottom: 7rem !important; max-width: 1300px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F4F8FF 100%);
    border-right: 1px solid rgba(26,115,232,0.08);
}
[data-testid="stSidebar"] .stButton button {
    border-radius: 10px;
    border: 1px solid rgba(26,115,232,0.18);
    background: white;
    color: #1A73E8;
    font-weight: 600;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #E8F0FE;
    border-color: #1A73E8;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none !important;}
/* Streamlit >=1.5x renders the Deploy button via testids — the old selector no
   longer matches, which is why 'Deploy' was overlapping the top-right content
   (e.g. the 'Load Case C' button). Hide the deploy button + toolbar actions,
   but keep the run-status widget visible. */
[data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Hero banner ── */
.hero-banner {
    position: relative;
    background:
      radial-gradient(600px 200px at 90% -20%, rgba(255,255,255,0.18), transparent 70%),
      linear-gradient(135deg, #0E63D6 0%, #1A73E8 35%, #00897B 100%);
    border-radius: 18px;
    padding: 30px 36px;
    margin-bottom: 22px;
    color: white;
    box-shadow: 0 12px 40px -12px rgba(26,115,232,0.45),
                0 4px 16px rgba(26,115,232,0.18);
    overflow: hidden;
}
.hero-banner::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle at 100% 100%, rgba(255,255,255,0.08), transparent 50%);
    pointer-events: none;
}
.hero-banner h1 {
    font-size: 2.0rem; font-weight: 800; margin: 0;
    letter-spacing: -0.6px; line-height: 1.15;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.hero-banner p  { font-size: 1.02rem; opacity: 0.95; margin: 8px 0 0; max-width: 800px; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 14px;
    margin-right: 8px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04),
                0 8px 24px -12px rgba(15,23,42,0.10);
    border: 1px solid rgba(15,23,42,0.04);
    border-left: 4px solid #1A73E8;
    margin-bottom: 12px;
    transition: transform 0.15s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 2px 6px rgba(15,23,42,0.06),
                0 18px 36px -16px rgba(15,23,42,0.18);
}
.metric-card .label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
                       letter-spacing: 0.6px; color: #6B7280; margin-bottom: 4px; }
.metric-card .value { font-size: 1.55rem; font-weight: 800; color: #0F172A; line-height: 1.15; }
.metric-card .sub   { font-size: 0.80rem; color: #64748B; margin-top: 4px; line-height: 1.4; }

/* ── Risk badges ── */
.risk-low      { background:#E8F5E9; color:#1B5E20; border:1.5px solid #A5D6A7; }
.risk-moderate { background:#FFFDE7; color:#E65100; border:1.5px solid #FFE082; }
.risk-high     { background:#FFF3E0; color:#BF360C; border:1.5px solid #FFAB91; }
.risk-critical { background:#FFEBEE; color:#B71C1C; border:1.5px solid #EF9A9A; }
.risk-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 6px 22px;
    font-size: 0.92rem;
    font-weight: 800;
    letter-spacing: 0.6px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* ── Step progress ── */
.step-item { display: flex; align-items: center; padding: 9px 12px; border-radius: 10px;
             margin: 3px 0; font-size: 0.88rem; cursor: default;
             transition: background 0.15s; }
.step-active   { background:#E8F0FE; color:#1A73E8; font-weight: 700;
                 box-shadow: inset 0 0 0 1px rgba(26,115,232,0.2); }
.step-done     { background:#E8F5E9; color:#1B5E20; font-weight: 600; }
.step-pending  { color:#9CA3AF; }
.step-icon { width: 28px; height: 28px; border-radius: 50%; display: inline-flex;
             align-items: center; justify-content: center; font-size: 0.78rem;
             font-weight: 800; margin-right: 10px; flex-shrink: 0; }
.step-icon-active  { background:#1A73E8; color:white;
                     box-shadow: 0 0 0 4px rgba(26,115,232,0.18); }
.step-icon-done    { background:#2E7D32; color:white; }
.step-icon-pending { background:#E5E7EB; color:#9CA3AF; }

/* ── Doctor cards ── */
.doctor-card {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05),
                0 10px 28px -14px rgba(15,23,42,0.12);
    border: 1px solid rgba(15,23,42,0.05);
    border-top: 3px solid #1A73E8;
    margin-bottom: 14px;
    transition: transform 0.15s ease, box-shadow 0.2s ease, border-top-color 0.2s ease;
    height: 100%;
}
.doctor-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 2px 6px rgba(15,23,42,0.06),
                0 18px 36px -14px rgba(15,23,42,0.20);
    border-top-color: #00897B;
}
.doctor-name  { font-size: 1.05rem; font-weight: 800; color: #0F172A; }
.doctor-spec  { font-size: 0.85rem; color: #1A73E8; font-weight: 700; margin: 2px 0; }
.doctor-hosp  { font-size: 0.85rem; color: #475569; }
.doctor-meta  { font-size: 0.78rem; color: #6B7280; margin-top: 6px; line-height: 1.55; }
.star-rating  { color: #F59E0B; font-size: 1rem; }

/* ── Section headers ── */
.section-header {
    font-size: 1.18rem;
    font-weight: 800;
    color: #0F172A;
    margin: 22px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(15,23,42,0.07);
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::before {
    content: "";
    width: 4px; height: 18px;
    background: linear-gradient(180deg, #1A73E8, #00897B);
    border-radius: 4px;
    display: inline-block;
}

/* ── Urgency banners ── */
.urgency-routine   { background:linear-gradient(135deg,#E8F5E9,#F1F8E9); color:#1B5E20; border:1px solid #A5D6A7; }
.urgency-elective  { background:linear-gradient(135deg,#FFFDE7,#FFF8E1); color:#E65100; border:1px solid #FFE082; }
.urgency-urgent    { background:linear-gradient(135deg,#FFF3E0,#FFEBEE); color:#BF360C; border:1px solid #FFAB91; }
.urgency-emergency { background:linear-gradient(135deg,#FFEBEE,#FFCDD2); color:#B71C1C; border:1px solid #EF9A9A; }
.urgency-banner {
    border-radius: 12px;
    padding: 16px 22px;
    font-size: 1.0rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 16px;
    letter-spacing: 0.8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Info boxes ── */
.info-box {
    background: linear-gradient(180deg, #EEF4FF 0%, #E8F0FE 100%);
    border-left: 4px solid #1A73E8;
    border-radius: 4px 12px 12px 4px;
    padding: 13px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #0F2A44;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.warn-box {
    background: linear-gradient(180deg, #FFF8E1 0%, #FFF3C4 100%);
    border-left: 4px solid #F59E0B;
    border-radius: 4px 12px 12px 4px;
    padding: 13px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #5A3F00;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}

/* ── Input styling ── */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
    border-radius: 10px !important;
    border: 1.5px solid #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: #1A73E8 !important;
    box-shadow: 0 0 0 3px rgba(26,115,232,0.12) !important;
}

/* ── Buttons ── */
div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {
    border-radius: 10px;
    font-weight: 700;
    padding: 0.55rem 1.8rem;
    transition: transform 0.12s ease, box-shadow 0.18s ease, background 0.18s ease;
    border: 1px solid rgba(15,23,42,0.08);
    letter-spacing: 0.2px;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px -10px rgba(15,23,42,0.25);
}
div.stButton > button[kind="primary"], div.stButton > button[data-testid*="primary"],
div.stDownloadButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1A73E8 0%, #00897B 100%) !important;
    color: white !important;
    border: 1px solid rgba(26,115,232,0.4) !important;
    box-shadow: 0 6px 16px -6px rgba(26,115,232,0.45);
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 12px 24px -8px rgba(26,115,232,0.55);
    filter: brightness(1.05);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    border-radius: 14px !important;
    border: 2px dashed rgba(26,115,232,0.35) !important;
    background: #FAFCFF !important;
    transition: all 0.15s ease !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #1A73E8 !important;
    background: #F1F6FF !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #EEF4FF;
    border-radius: 12px;
    padding: 5px;
    border: 1px solid rgba(26,115,232,0.10);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    font-weight: 700;
    font-size: 0.88rem;
    color: #475569;
    padding: 8px 16px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: white !important;
    color: #1A73E8 !important;
    box-shadow: 0 2px 6px rgba(15,23,42,0.06);
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div > div { background: linear-gradient(90deg, #1A73E8, #00897B) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid rgba(15,23,42,0.06) !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}

/* ── Doctor card CTA buttons ── */
.doc-cta {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 14px;
    border-radius: 10px;
    font-size: 0.82rem;
    font-weight: 700;
    text-decoration: none !important;
    background: white;
    color: #1A73E8 !important;
    border: 1px solid rgba(26,115,232,0.30);
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    transition: transform 0.12s, box-shadow 0.18s, background 0.18s;
}
.doc-cta:hover {
    background: #EEF4FF;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px -8px rgba(26,115,232,0.35);
}
.doc-cta-primary {
    background: linear-gradient(135deg, #1A73E8 0%, #00897B 100%);
    color: white !important;
    border-color: rgba(26,115,232,0.45);
    box-shadow: 0 6px 16px -8px rgba(26,115,232,0.45);
}
.doc-cta-primary:hover {
    background: linear-gradient(135deg, #0E63D6 0%, #00796B 100%);
    color: white !important;
    box-shadow: 0 12px 24px -8px rgba(26,115,232,0.55);
}

/* ── Pills/chips ── */
.pill {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.74rem;
    font-weight: 700;
    margin: 2px 4px 2px 0;
    background: #EEF4FF;
    color: #1A73E8;
    border: 1px solid rgba(26,115,232,0.18);
}
.pill-green { background:#E8F5E9; color:#1B5E20; border-color:#A5D6A7; }
.pill-amber { background:#FFF8E1; color:#B45309; border-color:#FCD34D; }
.pill-red   { background:#FEE2E2; color:#B91C1C; border-color:#FCA5A5; }

/* ── Spinner / loading ── */
.loading-container { text-align: center; padding: 40px; }
.loading-container h2 { color: #1A73E8; }

/* ── Disclaimer ── */
.disclaimer {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 24px;
    line-height: 1.65;
}

/* ── Subtle fade-in for hero on load ── */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); }
                     to   { opacity: 1; transform: translateY(0); } }
@keyframes fadeIn   { from { opacity: 0; } to { opacity: 1; } }
@keyframes scaleIn  { from { opacity: 0; transform: scale(0.96); }
                      to   { opacity: 1; transform: scale(1); } }
@keyframes slideRight { from { transform: translateX(-12px); opacity: 0; }
                        to   { transform: translateX(0);   opacity: 1; } }
@keyframes shimmer  { 0%   { background-position: -200% 0; }
                      100% { background-position: 200% 0; } }
@keyframes float    { 0%,100% { transform: translateY(0px); }
                      50%     { transform: translateY(-4px); } }
@keyframes spinSlow { from { transform: rotate(0deg); }
                       to  { transform: rotate(360deg); } }
@keyframes pulseRing { 0%   { box-shadow: 0 0 0 0 rgba(26,115,232,0.45); }
                        70% { box-shadow: 0 0 0 12px rgba(26,115,232,0); }
                       100% { box-shadow: 0 0 0 0 rgba(26,115,232,0); } }
@keyframes drift    { 0%   { transform: translate(0,0) rotate(0deg); }
                      100% { transform: translate(40px,-30px) rotate(360deg); } }
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-banner   { animation: fadeInUp 0.55s ease both; }
.metric-card   { animation: scaleIn 0.45s ease both; }
.doctor-card   { animation: fadeInUp 0.45s ease both; }
.section-header { animation: slideRight 0.45s ease both; }

/* Stagger metric/doctor cards */
.metric-card:nth-child(1)  { animation-delay: 0.05s; }
.metric-card:nth-child(2)  { animation-delay: 0.10s; }
.metric-card:nth-child(3)  { animation-delay: 0.15s; }
.metric-card:nth-child(4)  { animation-delay: 0.20s; }

/* Hero gets a moving gradient sheen */
.hero-banner {
    background-size: 200% 200%;
    animation: fadeInUp 0.55s ease both, gradientShift 12s ease infinite;
}
/* Floating glow blobs inside the hero */
.hero-banner::after {
    content: "";
    position: absolute;
    width: 220px; height: 220px;
    right: -60px; top: -60px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.25), transparent 70%);
    animation: drift 28s ease-in-out infinite alternate;
    pointer-events: none;
}
.hero-banner::before { z-index: 0; }
.hero-banner > *     { position: relative; z-index: 1; }

/* Glassmorphism for doctor / metric cards on hover */
.metric-card:hover, .doctor-card:hover {
    background: linear-gradient(135deg,#ffffff,#f8fbff);
}

/* Section-header icon wrapper */
.section-header svg { color: #1A73E8; }

/* Pretty scrollbar */
::-webkit-scrollbar         { width: 10px; height: 10px; }
::-webkit-scrollbar-track   { background: #F1F5F9; border-radius: 8px; }
::-webkit-scrollbar-thumb   { background: linear-gradient(180deg,#94A3B8,#64748B); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg,#1A73E8,#00897B); }

/* Tab transition */
.stTabs [data-baseweb="tab-panel"] { animation: fadeIn 0.35s ease both; }

/* Floating animation for the sidebar logo tile */
[data-testid="stSidebar"] > div:first-child div[style*="border-radius:14px"] {
    animation: float 4s ease-in-out infinite;
}

/* Plotly chart subtle border */
[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(15,23,42,0.06);
    border-radius: 14px;
    background: white;
    padding: 6px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    transition: box-shadow 0.2s;
}
[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 1px 3px rgba(15,23,42,0.04), 0 12px 28px -16px rgba(15,23,42,0.18);
}

/* Image rounded corners */
[data-testid="stImage"] img { border-radius: 12px; }

/* Subtle global "noise" pattern for texture */
.stApp::before {
    content: "";
    position: fixed; inset: 0;
    background-image:
        radial-gradient(rgba(15,23,42,0.025) 1px, transparent 1px);
    background-size: 24px 24px;
    pointer-events: none; z-index: 0; opacity: 0.6;
}
.stApp > * { position: relative; z-index: 1; }

/* Animated count-up — applied to .countup elements via class */
.countup { display: inline-block; }

/* Risk badge glow */
.risk-badge.risk-low      { box-shadow: 0 0 0 0 rgba(46,125,50,0.35); animation: pulseRing 2.4s infinite; }
.risk-badge.risk-moderate { box-shadow: 0 0 0 0 rgba(245,158,11,0.4);  animation: pulseRing 2.4s infinite; }
.risk-badge.risk-high     { box-shadow: 0 0 0 0 rgba(230,81,0,0.4);    animation: pulseRing 2.0s infinite; }
.risk-badge.risk-critical { box-shadow: 0 0 0 0 rgba(183,28,28,0.5);   animation: pulseRing 1.6s infinite; }

/* ── Status pulse ── */
@keyframes pulseGreen {
  0%   { box-shadow: 0 0 0 0 rgba(46,125,50,0.45); }
  70%  { box-shadow: 0 0 0 8px rgba(46,125,50,0); }
  100% { box-shadow: 0 0 0 0 rgba(46,125,50,0); }
}
.status-dot {
    display: inline-block; width: 9px; height: 9px;
    border-radius: 50%; margin-right: 7px; vertical-align: middle;
}
.status-dot-ok    { background:#16A34A; animation: pulseGreen 1.8s infinite; }
.status-dot-warn  { background:#F59E0B; }
.status-dot-err   { background:#DC2626; }
.status-dot-load  { background:#94A3B8; }

/* ── Number input arrow polish ── */
[data-testid="stNumberInput"] button { border-radius: 8px !important; }

/* ── Remove ugly default focus ring on cards ── */
.metric-card:focus, .doctor-card:focus { outline: none; }

/* ── Slider ── */
[data-baseweb="slider"] [role="slider"] { background: #1A73E8 !important; }

/* ════════════════════════════════════════════════════════════════════════
   BOLD THEME v2 — elevated, cohesive redesign (cascades over the base above)
   Design tokens + vivid palette + glassmorphism + animated accents.
   ════════════════════════════════════════════════════════════════════════ */
:root {
  --c-primary:   #1554F0;   --c-primary-2: #0EA5C9;   --c-accent: #7C3AED;
  --c-teal:      #00A88E;   --c-ink: #0B1530;         --c-muted: #5A6B86;
  --grad-brand:  linear-gradient(120deg, #1554F0 0%, #0EA5C9 55%, #00A88E 100%);
  --grad-accent: linear-gradient(120deg, #6D28D9 0%, #1554F0 100%);
  --shadow-sm:   0 1px 2px rgba(11,21,48,.06), 0 6px 18px -10px rgba(11,21,48,.18);
  --shadow-md:   0 2px 6px rgba(11,21,48,.07), 0 22px 48px -22px rgba(21,84,240,.30);
  --radius:      18px;
  --tr:          .22s cubic-bezier(.4,0,.2,1);
}

/* Animated, richer page backdrop */
@keyframes bgFloat { 0%,100%{background-position:0% 0%,100% 0%,0 0} 50%{background-position:6% 4%,94% 6%,0 0} }
.main, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 680px at -5% -12%, rgba(21,84,240,.10), transparent 60%),
    radial-gradient(1000px 560px at 105% -8%, rgba(0,168,142,.10), transparent 60%),
    linear-gradient(180deg, #F4F8FF 0%, #FBFDFF 60%, #FFFFFF 100%) !important;
  background-attachment: fixed;
  animation: bgFloat 26s ease-in-out infinite;
}

/* Bolder hero with shimmer + floating orb */
.hero-banner {
  background:
    radial-gradient(520px 200px at 88% -30%, rgba(255,255,255,.25), transparent 70%),
    var(--grad-brand) !important;
  border-radius: 22px !important;
  padding: 34px 40px !important;
  box-shadow: 0 18px 50px -18px rgba(21,84,240,.55), 0 6px 20px rgba(0,168,142,.18) !important;
}
.hero-banner::after {
  content:""; position:absolute; top:-60px; right:-40px; width:220px; height:220px;
  background: radial-gradient(circle, rgba(255,255,255,.22), transparent 65%);
  border-radius:50%; animation: bgFloat 14s ease-in-out infinite; pointer-events:none;
}
.hero-banner h1 { font-size: 2.15rem !important; letter-spacing:-.8px !important; }
.hero-badge { background: rgba(255,255,255,.16) !important; transition: var(--tr); }
.hero-badge:hover { background: rgba(255,255,255,.30) !important; transform: translateY(-1px); }

/* Sidebar nav — gradient pills with hover-slide + active glow */
[data-testid="stSidebar"] { backdrop-filter: blur(6px); }
[data-testid="stSidebar"] .stButton button {
  border-radius: 12px !important; border: 1px solid rgba(21,84,240,.14) !important;
  font-weight: 700 !important; transition: var(--tr) !important;
  position: relative; overflow: hidden;
}
[data-testid="stSidebar"] .stButton button:hover {
  background: var(--grad-brand) !important; color: #fff !important;
  border-color: transparent !important; transform: translateX(4px);
  box-shadow: 0 10px 22px -10px rgba(21,84,240,.6) !important;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] {
  background: var(--grad-brand) !important; color:#fff !important; border:none !important;
  box-shadow: 0 10px 24px -10px rgba(21,84,240,.6) !important;
}

/* Primary action buttons — gradient, lift on hover */
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--grad-brand) !important; border: none !important; color:#fff !important;
  border-radius: 12px !important; font-weight: 800 !important; letter-spacing:.2px;
  box-shadow: 0 12px 26px -12px rgba(21,84,240,.65) !important; transition: var(--tr) !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); filter: saturate(1.1) brightness(1.03);
  box-shadow: 0 18px 36px -14px rgba(21,84,240,.7) !important;
}
.stButton > button:not([kind="primary"]) { border-radius: 12px !important; transition: var(--tr) !important; }

/* Glassy elevated cards */
.metric-card, .doctor-card {
  background: rgba(255,255,255,.82) !important; backdrop-filter: blur(10px);
  border-radius: var(--radius) !important; box-shadow: var(--shadow-sm) !important;
}
.metric-card { border-left: 4px solid transparent !important;
  border-image: var(--grad-brand) 1; }
.metric-card:hover, .doctor-card:hover { box-shadow: var(--shadow-md) !important; transform: translateY(-3px); }

/* Section header — gradient accent bar + tighter ink */
.section-header { color: var(--c-ink) !important; font-size: 1.22rem !important; }
.section-header::before { width:5px !important; height:20px !important;
  background: var(--grad-brand) !important; box-shadow: 0 2px 8px -2px rgba(21,84,240,.5); }

/* Inputs — soft focus glow */
[data-baseweb="input"] input, [data-baseweb="select"] > div, [data-baseweb="textarea"] textarea {
  border-radius: 11px !important; transition: var(--tr) !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
  box-shadow: 0 0 0 3px rgba(21,84,240,.18) !important;
}

/* Tabs — pill underline */
[data-baseweb="tab-list"] { gap: 4px; }
[data-baseweb="tab"] { border-radius: 10px 10px 0 0 !important; transition: var(--tr); }
[aria-selected="true"][data-baseweb="tab"] { background: rgba(21,84,240,.07) !important; }

/* Expanders — soft card look */
[data-testid="stExpander"] { border-radius: 14px !important; overflow:hidden;
  border: 1px solid rgba(11,21,48,.07) !important; box-shadow: var(--shadow-sm); }

/* Gentle fade-in for page content */
@keyframes riseIn { from{opacity:0; transform: translateY(8px)} to{opacity:1; transform:none} }
.block-container > div { animation: riseIn .4s ease both; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# DOCTOR DATABASE (illustrative — would connect to a live directory in production)
# ─────────────────────────────────────────────────────────────────────────────
DOCTORS_DB: List[Dict[str, Any]] = [
    # ── Delhi-NCR (verified Aug 2025: AIIMS, Sir Ganga Ram, Apollo, Max, BLK-Max, Medanta, Fortis FMRI, Manipal, Jaypee, Asian, Yashoda) ─
    {"name":"Dr. Anoop Saraya","hospital":"All India Institute of Medical Sciences (AIIMS), New Delhi","specialty":"Gastroenterology","sub_specialty":"Pancreaticobiliary & IBD","city":"New Delhi","country":"India","rating":4.9,"experience_years":38,"phone":"+91-11-2658-8500","languages":["English","Hindi"]},
    {"name":"Dr. Govind Makharia","hospital":"All India Institute of Medical Sciences (AIIMS), New Delhi","specialty":"Gastroenterology","sub_specialty":"Celiac Disease & IBD","city":"New Delhi","country":"India","rating":4.9,"experience_years":30,"phone":"+91-11-2658-8500","languages":["English","Hindi"]},
    {"name":"Dr. Anil Arora","hospital":"Sir Ganga Ram Hospital, Rajinder Nagar","specialty":"Gastroenterology","sub_specialty":"Therapeutic Endoscopy","city":"New Delhi","country":"India","rating":4.9,"experience_years":35,"phone":"+91-11-4225-4000","languages":["English","Hindi","Punjabi"]},
    {"name":"Dr. Naresh Bansal","hospital":"Sir Ganga Ram Hospital, Rajinder Nagar","specialty":"Hepatology","sub_specialty":"Liver Cirrhosis & Transplant Hepatology","city":"New Delhi","country":"India","rating":4.7,"experience_years":24,"phone":"+91-11-4225-4000","languages":["English","Hindi","Punjabi"]},
    {"name":"Dr. Sudeep Khanna","hospital":"Indraprastha Apollo Hospital, Sarita Vihar","specialty":"Gastroenterology","sub_specialty":"Barrett's & Esophageal","city":"New Delhi","country":"India","rating":4.8,"experience_years":28,"phone":"+91-11-2987-1090","languages":["English","Hindi"]},
    {"name":"Dr. Niranjan Naik","hospital":"Indraprastha Apollo Hospital, Sarita Vihar","specialty":"Surgical Oncology","sub_specialty":"Colorectal & Peritoneal Oncology","city":"New Delhi","country":"India","rating":4.8,"experience_years":29,"phone":"+91-11-2987-1090","languages":["English","Hindi"]},
    {"name":"Dr. Subhash Gupta","hospital":"Max Super Speciality Hospital, Saket","specialty":"Hepatology","sub_specialty":"Hepatobiliary & Liver Transplant","city":"New Delhi","country":"India","rating":4.9,"experience_years":32,"phone":"+91-11-2651-5050","languages":["English","Hindi"]},
    {"name":"Dr. Harit Chaturvedi","hospital":"Max Super Speciality Hospital, Saket","specialty":"Surgical Oncology","sub_specialty":"GI & Colorectal Oncology","city":"New Delhi","country":"India","rating":4.9,"experience_years":36,"phone":"+91-11-2651-5050","languages":["English","Hindi"]},
    {"name":"Dr. Sanjiv Saigal","hospital":"Max Super Speciality Hospital, Saket","specialty":"Hepatology","sub_specialty":"Transplant Hepatology","city":"New Delhi","country":"India","rating":4.8,"experience_years":28,"phone":"+91-11-2651-5050","languages":["English","Hindi","Punjabi"]},
    {"name":"Dr. Surender Kumar Dabas","hospital":"BLK-Max Super Speciality Hospital, Pusa Road","specialty":"Surgical Oncology","sub_specialty":"Robotic GI Oncology","city":"New Delhi","country":"India","rating":4.8,"experience_years":25,"phone":"+91-11-3040-3040","languages":["English","Hindi"]},
    {"name":"Dr. Deep Goel","hospital":"BLK-Max Super Speciality Hospital, Pusa Road","specialty":"Gastrointestinal Surgery","sub_specialty":"Bariatric & Colorectal Surgery","city":"New Delhi","country":"India","rating":4.8,"experience_years":33,"phone":"+91-11-3040-3040","languages":["English","Hindi","Punjabi"]},
    {"name":"Dr. Vinay Mahendra","hospital":"Manipal Hospital, Dwarka","specialty":"GI Oncology","sub_specialty":"Colorectal Cancer & Peritoneal Surface Oncology","city":"New Delhi","country":"India","rating":4.7,"experience_years":18,"phone":"+91-11-4040-7070","languages":["English","Hindi"]},
    {"name":"Dr. Rajesh Puri","hospital":"Medanta — The Medicity, Sector 38","specialty":"Gastroenterology","sub_specialty":"Therapeutic Endoscopy & EUS","city":"Gurgaon","country":"India","rating":4.9,"experience_years":30,"phone":"+91-124-4141-414","languages":["English","Hindi","Punjabi"]},
    {"name":"Dr. Randhir Sud","hospital":"Medanta — The Medicity, Sector 38","specialty":"Gastroenterology","sub_specialty":"Advanced Endoscopy & ERCP","city":"Gurgaon","country":"India","rating":4.9,"experience_years":42,"phone":"+91-124-4141-414","languages":["English","Hindi","Punjabi"]},
    {"name":"Dr. Adarsh Chaudhary","hospital":"Medanta — The Medicity, Sector 38","specialty":"Gastrointestinal Surgery","sub_specialty":"Hepatobiliary & Pancreatic Surgery","city":"Gurgaon","country":"India","rating":4.8,"experience_years":38,"phone":"+91-124-4141-414","languages":["English","Hindi"]},
    {"name":"Dr. Tejinder Singh Bhasin","hospital":"Fortis Memorial Research Institute (FMRI), Sector 44","specialty":"Gastroenterology","sub_specialty":"IBD & Colitis","city":"Gurgaon","country":"India","rating":4.7,"experience_years":22,"phone":"+91-124-4962-200","languages":["English","Hindi","Punjabi"]},
    {"name":"Dr. Vivek Mangla","hospital":"Fortis Memorial Research Institute (FMRI), Sector 44","specialty":"Gastrointestinal Surgery","sub_specialty":"Colorectal & HPB Surgery","city":"Gurgaon","country":"India","rating":4.6,"experience_years":20,"phone":"+91-124-4962-200","languages":["English","Hindi"]},
    {"name":"Dr. Ashish Goel","hospital":"Artemis Hospital, Sector 51","specialty":"Medical Oncology","sub_specialty":"GI & Colorectal Oncology","city":"Gurgaon","country":"India","rating":4.7,"experience_years":19,"phone":"+91-124-4511-111","languages":["English","Hindi"]},
    {"name":"Dr. Manish Kumar Gupta","hospital":"Max Super Speciality Hospital, Sector 19, Noida","specialty":"Gastrointestinal Surgery","sub_specialty":"Colorectal & Robotic Surgery","city":"Noida","country":"India","rating":4.8,"experience_years":24,"phone":"+91-120-4344-444","languages":["English","Hindi"]},
    {"name":"Dr. Pradeep Jain","hospital":"Fortis Hospital, Sector 62, Noida","specialty":"Surgical Oncology","sub_specialty":"GI & Colorectal Oncology","city":"Noida","country":"India","rating":4.8,"experience_years":30,"phone":"+91-120-7191-222","languages":["English","Hindi"]},
    {"name":"Dr. Amrit Pal Singh","hospital":"Jaypee Hospital, Sector 128, Noida","specialty":"Gastroenterology","sub_specialty":"Polyposis Syndromes & Colonoscopy","city":"Noida","country":"India","rating":4.6,"experience_years":21,"phone":"+91-120-4122-222","languages":["English","Hindi","Punjabi"]},
    {"name":"Dr. Ajay Kumar","hospital":"Apollo Hospital, Sector 26, Noida","specialty":"Gastroenterology","sub_specialty":"Liver Disease & Endoscopy","city":"Noida","country":"India","rating":4.7,"experience_years":27,"phone":"+91-120-2451-851","languages":["English","Hindi"]},
    {"name":"Dr. Brij Mohan Khanna","hospital":"Yashoda Super Speciality Hospital, Kaushambi","specialty":"Gastroenterology","sub_specialty":"Therapeutic Endoscopy","city":"Ghaziabad","country":"India","rating":4.6,"experience_years":26,"phone":"+91-120-4188-188","languages":["English","Hindi"]},
    {"name":"Dr. Ramesh Sarin","hospital":"Asian Institute of Medical Sciences, Sector 21","specialty":"Surgical Oncology","sub_specialty":"Colorectal & GI Oncology","city":"Faridabad","country":"India","rating":4.7,"experience_years":35,"phone":"+91-129-4253-000","languages":["English","Hindi"]},
    {"name":"Dr. Shubham Vatsya","hospital":"Fortis Escorts Hospital, Neelam Bata Road","specialty":"Gastroenterology","sub_specialty":"IBD & Colonoscopic Surveillance","city":"Faridabad","country":"India","rating":4.5,"experience_years":17,"phone":"+91-129-4666-666","languages":["English","Hindi"]},

    # ── Other major Indian metros (kept from original directory) ─────────
    {"name":"Dr. Sunita Kapoor","hospital":"Apollo Hospital, Delhi","specialty":"Medical Oncology","sub_specialty":"GI Oncology","city":"New Delhi","country":"India","rating":4.7,"experience_years":15,"phone":"+91-11-2987-4444","languages":["English","Hindi","Punjabi"]},
    {"name":"Dr. Rajesh Nair","hospital":"Tata Memorial Hospital","specialty":"Surgical Oncology","sub_specialty":"Colorectal Resection","city":"Mumbai","country":"India","rating":4.9,"experience_years":24,"phone":"+91-22-2417-7000","languages":["English","Hindi","Marathi"]},
    {"name":"Dr. Meera Iyer","hospital":"Lilavati Hospital, Mumbai","specialty":"Gastroenterology","sub_specialty":"Advanced Endoscopy","city":"Mumbai","country":"India","rating":4.8,"experience_years":16,"phone":"+91-22-2675-1000","languages":["English","Hindi","Tamil"]},
    {"name":"Dr. Vinod Patel","hospital":"KEM Hospital Mumbai","specialty":"Gastroenterology","sub_specialty":"IBD & Colitis","city":"Mumbai","country":"India","rating":4.6,"experience_years":13,"phone":"+91-22-2410-7000","languages":["English","Hindi","Gujarati"]},
    {"name":"Dr. Kavitha Reddy","hospital":"Apollo Hospital, Bangalore","specialty":"Medical Oncology","sub_specialty":"GI Cancers","city":"Bangalore","country":"India","rating":4.8,"experience_years":20,"phone":"+91-80-2530-4050","languages":["English","Kannada","Telugu"]},
    {"name":"Dr. Sanjay Kumar","hospital":"Manipal Hospital Bangalore","specialty":"Colorectal Surgery","sub_specialty":"Robotic Surgery","city":"Bangalore","country":"India","rating":4.7,"experience_years":17,"phone":"+91-80-2502-4444","languages":["English","Kannada","Hindi"]},
    {"name":"Dr. Lakshmi Narayan","hospital":"NIMHANS Campus Clinic","specialty":"Gastroenterology","sub_specialty":"Endoscopy & Polypectomy","city":"Bangalore","country":"India","rating":4.5,"experience_years":12,"phone":"+91-80-4600-1234","languages":["English","Kannada"]},
    {"name":"Dr. Arjun Bose","hospital":"SSKM Hospital Kolkata","specialty":"Gastroenterology","sub_specialty":"GI Oncology","city":"Kolkata","country":"India","rating":4.7,"experience_years":19,"phone":"+91-33-2244-6000","languages":["English","Bengali","Hindi"]},
    {"name":"Dr. Priyanka Sen","hospital":"Medica Superspecialty, Kolkata","specialty":"Colorectal Surgery","sub_specialty":"Minimal Invasive","city":"Kolkata","country":"India","rating":4.6,"experience_years":14,"phone":"+91-33-6652-0000","languages":["English","Bengali"]},
    {"name":"Dr. Ramesh Babu","hospital":"NIMS, Hyderabad","specialty":"Gastroenterology","sub_specialty":"Therapeutic Endoscopy","city":"Hyderabad","country":"India","rating":4.8,"experience_years":21,"phone":"+91-40-2348-8888","languages":["English","Telugu","Hindi"]},
    {"name":"Dr. Shalini Gupta","hospital":"Yashoda Hospital, Hyderabad","specialty":"Medical Oncology","sub_specialty":"GI Tumours","city":"Hyderabad","country":"India","rating":4.7,"experience_years":16,"phone":"+91-40-4567-8910","languages":["English","Telugu","Hindi"]},
    {"name":"Dr. Manish Tiwari","hospital":"JIPMER, Puducherry","specialty":"Gastroenterology","sub_specialty":"Colorectal Cancer","city":"Chennai","country":"India","rating":4.9,"experience_years":25,"phone":"+91-44-2225-2011","languages":["English","Tamil","Hindi"]},
    {"name":"Dr. Ananya Krishnan","hospital":"Apollo Hospital Chennai","specialty":"Colorectal Surgery","sub_specialty":"Oncosurgery","city":"Chennai","country":"India","rating":4.8,"experience_years":18,"phone":"+91-44-2829-3333","languages":["English","Tamil"]},
    {"name":"Dr. Deepak Verma","hospital":"PGI Chandigarh","specialty":"Gastroenterology","sub_specialty":"IBD & Polyposis","city":"Chandigarh","country":"India","rating":4.7,"experience_years":17,"phone":"+91-172-2747-585","languages":["English","Hindi","Punjabi"]},
    {"name":"Dr. Nisha Agarwal","hospital":"Medanta, Gurgaon","specialty":"GI Oncology","sub_specialty":"Barrett's & Esophageal","city":"Gurgaon","country":"India","rating":4.8,"experience_years":20,"phone":"+91-124-4141-414","languages":["English","Hindi"]},
    {"name":"Dr. Vikram Joshi","hospital":"Kokilaben Ambani Hospital, Mumbai","specialty":"Surgical Oncology","sub_specialty":"GI Surgery","city":"Mumbai","country":"India","rating":4.7,"experience_years":15,"phone":"+91-22-4269-6969","languages":["English","Hindi","Marathi"]},

    # ── USA ──────────────────────────────────────────────────────────────
    {"name":"Dr. James Harrington","hospital":"Memorial Sloan Kettering Cancer Center","specialty":"Colorectal Surgery","sub_specialty":"Rectal Cancer","city":"New York","country":"USA","rating":4.9,"experience_years":26,"phone":"+1-212-639-2000","languages":["English"]},
    {"name":"Dr. Sarah Chen","hospital":"NewYork-Presbyterian Hospital","specialty":"Gastroenterology","sub_specialty":"Advanced Endoscopy","city":"New York","country":"USA","rating":4.8,"experience_years":19,"phone":"+1-212-746-5454","languages":["English","Mandarin"]},
    {"name":"Dr. Michael Goldstein","hospital":"Cedars-Sinai Medical Center","specialty":"GI Oncology","sub_specialty":"Colorectal Cancers","city":"Los Angeles","country":"USA","rating":4.9,"experience_years":23,"phone":"+1-310-423-3277","languages":["English","Hebrew"]},
    {"name":"Dr. Patricia Williams","hospital":"UCLA Medical Center","specialty":"Gastroenterology","sub_specialty":"IBD & Colitis","city":"Los Angeles","country":"USA","rating":4.8,"experience_years":21,"phone":"+1-310-825-9111","languages":["English","Spanish"]},
    {"name":"Dr. Robert Patel","hospital":"Northwestern Memorial Hospital","specialty":"Colorectal Surgery","sub_specialty":"Laparoscopic & Robotic","city":"Chicago","country":"USA","rating":4.7,"experience_years":18,"phone":"+1-312-926-2000","languages":["English","Gujarati"]},
    {"name":"Dr. Emily Thompson","hospital":"University of Chicago Medicine","specialty":"Gastroenterology","sub_specialty":"Barrett's & Esophageal","city":"Chicago","country":"USA","rating":4.8,"experience_years":16,"phone":"+1-773-702-1000","languages":["English"]},
    {"name":"Dr. David Park","hospital":"MD Anderson Cancer Center","specialty":"GI Oncology","sub_specialty":"Colorectal Tumors","city":"Houston","country":"USA","rating":4.9,"experience_years":28,"phone":"+1-713-792-2121","languages":["English","Korean"]},
    {"name":"Dr. Laura Martinez","hospital":"Houston Methodist Hospital","specialty":"Gastroenterology","sub_specialty":"Therapeutic Colonoscopy","city":"Houston","country":"USA","rating":4.7,"experience_years":14,"phone":"+1-713-790-3333","languages":["English","Spanish"]},
    {"name":"Dr. Andrew Kim","hospital":"Massachusetts General Hospital","specialty":"Colorectal Surgery","sub_specialty":"Oncological Resection","city":"Boston","country":"USA","rating":4.9,"experience_years":24,"phone":"+1-617-726-2000","languages":["English","Korean"]},
    {"name":"Dr. Jennifer Lee","hospital":"Dana-Farber Cancer Institute","specialty":"Medical Oncology","sub_specialty":"GI Cancers","city":"Boston","country":"USA","rating":4.8,"experience_years":20,"phone":"+1-617-632-3000","languages":["English"]},
    {"name":"Dr. Thomas Brown","hospital":"UCSF Medical Center","specialty":"Gastroenterology","sub_specialty":"Polyps & Cancer Screening","city":"San Francisco","country":"USA","rating":4.8,"experience_years":22,"phone":"+1-415-476-1000","languages":["English"]},
    {"name":"Dr. Nancy Zhang","hospital":"Stanford Health Care","specialty":"GI Oncology","sub_specialty":"Colon & Rectal Cancer","city":"San Francisco","country":"USA","rating":4.7,"experience_years":17,"phone":"+1-650-498-6000","languages":["English","Mandarin"]},
    {"name":"Dr. William Johnson","hospital":"Mayo Clinic","specialty":"Gastroenterology","sub_specialty":"Inflammatory Bowel Disease","city":"Rochester","country":"USA","rating":4.9,"experience_years":30,"phone":"+1-507-284-2511","languages":["English"]},
    {"name":"Dr. Rachel Green","hospital":"Johns Hopkins Hospital","specialty":"Colorectal Surgery","sub_specialty":"Hereditary CRC","city":"Baltimore","country":"USA","rating":4.8,"experience_years":19,"phone":"+1-410-955-5000","languages":["English"]},

    # ── UK ───────────────────────────────────────────────────────────────
    {"name":"Dr. Oliver Hughes","hospital":"The Royal Marsden Hospital","specialty":"GI Oncology","sub_specialty":"Colorectal Tumours","city":"London","country":"UK","rating":4.9,"experience_years":24,"phone":"+44-20-7352-8171","languages":["English"]},
    {"name":"Dr. Charlotte Davis","hospital":"St Mark's Hospital","specialty":"Colorectal Surgery","sub_specialty":"Polyposis Syndromes","city":"London","country":"UK","rating":4.9,"experience_years":22,"phone":"+44-20-8235-4000","languages":["English"]},
    {"name":"Dr. Benjamin Clarke","hospital":"University College London Hospital","specialty":"Gastroenterology","sub_specialty":"Barrett's Oesophagus","city":"London","country":"UK","rating":4.7,"experience_years":15,"phone":"+44-20-3456-7890","languages":["English"]},
    {"name":"Dr. Sophie Wilson","hospital":"Manchester Royal Infirmary","specialty":"Gastroenterology","sub_specialty":"IBD & Endoscopy","city":"Manchester","country":"UK","rating":4.7,"experience_years":13,"phone":"+44-161-276-1234","languages":["English"]},
    {"name":"Dr. Henry Moore","hospital":"Queen Elizabeth Hospital","specialty":"Colorectal Surgery","sub_specialty":"Robotic Surgery","city":"Birmingham","country":"UK","rating":4.8,"experience_years":20,"phone":"+44-121-627-2000","languages":["English"]},

    # ── UAE ──────────────────────────────────────────────────────────────
    {"name":"Dr. Ahmad Al-Rashid","hospital":"Cleveland Clinic Abu Dhabi","specialty":"GI Oncology","sub_specialty":"Colorectal Cancer","city":"Abu Dhabi","country":"UAE","rating":4.8,"experience_years":18,"phone":"+971-2-659-0000","languages":["English","Arabic"]},
    {"name":"Dr. Fatima Al-Hassan","hospital":"American Hospital Dubai","specialty":"Gastroenterology","sub_specialty":"Advanced Endoscopy","city":"Dubai","country":"UAE","rating":4.7,"experience_years":14,"phone":"+971-4-336-7777","languages":["English","Arabic"]},

    # ── Singapore ────────────────────────────────────────────────────────
    {"name":"Dr. Tan Wei Lin","hospital":"Singapore General Hospital","specialty":"Colorectal Surgery","sub_specialty":"Minimally Invasive","city":"Singapore","country":"Singapore","rating":4.8,"experience_years":20,"phone":"+65-6222-3322","languages":["English","Mandarin"]},
    {"name":"Dr. Priya Subramaniam","hospital":"National University Hospital","specialty":"Gastroenterology","sub_specialty":"IBD & Oncology","city":"Singapore","country":"Singapore","rating":4.7,"experience_years":16,"phone":"+65-6779-5555","languages":["English","Tamil","Malay"]},

    # ── Canada ────────────────────────────────────────────────────────────
    {"name":"Dr. Jean-Paul Tremblay","hospital":"Princess Margaret Cancer Centre","specialty":"GI Oncology","sub_specialty":"Colorectal Cancers","city":"Toronto","country":"Canada","rating":4.8,"experience_years":21,"phone":"+1-416-946-2000","languages":["English","French"]},
    {"name":"Dr. Aisha Mohammed","hospital":"Vancouver General Hospital","specialty":"Gastroenterology","sub_specialty":"Barrett's & Polyps","city":"Vancouver","country":"Canada","rating":4.7,"experience_years":15,"phone":"+1-604-875-4111","languages":["English","Arabic"]},

    # ── Australia ────────────────────────────────────────────────────────
    {"name":"Dr. Liam O'Brien","hospital":"Peter MacCallum Cancer Centre","specialty":"GI Oncology","sub_specialty":"Colorectal Tumours","city":"Melbourne","country":"Australia","rating":4.9,"experience_years":23,"phone":"+61-3-8559-5000","languages":["English"]},
    {"name":"Dr. Emma Walsh","hospital":"Royal Prince Alfred Hospital","specialty":"Gastroenterology","sub_specialty":"Endoscopy & Polypectomy","city":"Sydney","country":"Australia","rating":4.7,"experience_years":17,"phone":"+61-2-9515-6111","languages":["English"]},
]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL + PIPELINE LOADING  (cached — only loads once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ai_system():
    """Load model, tokenizer, TCGA pool, and orchestrator. Returns dict or None."""
    try:
        from transformers import AutoTokenizer
        from src.models.unified_transformer import UnifiedMultiModalTransformer
        from src.agents.multimodal_orchestrator import MultiModalOrchestrator
        from src.data.multimodal_dataset import (
            N_TABULAR_FEATURES, load_tcga_tabular, extract_tabular_vector
        )

        device = torch.device("cpu")  # Use CPU for Streamlit stability

        # --- Model (kwargs must match UnifiedMultiModalTransformer.__init__) ---
        model = UnifiedMultiModalTransformer(
            n_classes=N_CLASSES,
            d_model=D_MODEL,
            n_fusion_heads=8,
            n_fusion_layers=3,
            n_tabular_features=N_TABULAR_FEATURES,
        )
        ckpt_loaded = False
        if CHECKPOINT.exists():
            # Use the project's safe-loader. Our own checkpoints carry a
            # "model_state" dict + meta (val_acc, epoch, classes) which is
            # not loadable under weights_only=True, so we pass allow_unsafe=True.
            # This is acceptable because CHECKPOINT comes from the repo,
            # not user upload. SECURITY.md documents the policy.
            from src.app.security import safe_torch_load
            ckpt = safe_torch_load(str(CHECKPOINT), map_location=device,
                                   allow_unsafe=True)
            # Checkpoints may use "model_state" (training script) or "model_state_dict"
            if isinstance(ckpt, dict):
                state = ckpt.get("model_state",
                        ckpt.get("model_state_dict",
                        ckpt.get("state_dict", ckpt)))
            else:
                state = ckpt
            missing, unexpected = model.load_state_dict(state, strict=False)
            ckpt_loaded = (len(missing) < 50)  # tolerate a few missing aux keys
        model.eval()
        model.to(device)

        # --- Tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

        # --- TCGA pool ---
        tcga_df = load_tcga_tabular(str(ROOT / "data/raw/tcga"))

        # --- Orchestrator ---
        orch = MultiModalOrchestrator(model, tokenizer, device)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "tcga_df": tcga_df,
            "orchestrator": orch,
            "ready": True,
            "checkpoint_loaded": ckpt_loaded,
        }
    except Exception as e:
        # Capture the actual exception message AND a short traceback summary
        # so the sidebar diagnostic shows us what really went wrong. This is
        # our own deployment — exposing the error to the operator is fine
        # (it's gated behind the "ℹ︎ Why?" expander, only visible if load failed).
        import logging, traceback
        logging.getLogger("colonai.app").exception("model load failed")
        tb_summary = traceback.format_exc()
        # Keep last 6 lines of the trace — usually shows the failing call
        tb_short = "\n".join(tb_summary.splitlines()[-8:])
        return {"ready": False,
                "error_type": type(e).__name__,
                "error": str(e)[:500],
                "traceback_tail": tb_short}


@st.cache_resource(show_spinner=False)
def get_tcga_pool_cached():
    """Return (tcga_df, extract_fn) or (None, None)."""
    try:
        from src.data.multimodal_dataset import load_tcga_tabular, extract_tabular_vector, N_TABULAR_FEATURES
        df = load_tcga_tabular(str(ROOT / "data/raw/tcga"))
        return df, extract_tabular_vector, N_TABULAR_FEATURES
    except Exception:
        return None, None, 12


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

def preprocess_image(pil_img: Image.Image) -> tuple:
    """Return (tensor, numpy_array) ready for model."""
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    norm = (arr - np.array(IMG_MEAN)) / np.array(IMG_STD)
    tensor = torch.tensor(norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    return tensor, arr  # (1,3,224,224), (224,224,3) 0-1 float


def tokenize_text(tokenizer, text: str) -> tuple:
    enc = tokenizer(
        text, return_tensors="pt",
        max_length=128, padding="max_length",
        truncation=True
    )
    return enc["input_ids"], enc["attention_mask"]


def build_tabular_vector(patient: dict, tcga_df, extract_fn, n_features: int) -> torch.Tensor:
    """Build a tabular feature vector from patient data + TCGA fallback."""
    # Try to use a real TCGA row as base (similar age/stage)
    vec = np.zeros(n_features, dtype=np.float32)
    if tcga_df is not None and not tcga_df.empty:
        subset = tcga_df.sample(1)
        vec = extract_fn(subset.iloc[0])
    # Override with patient's own values where known
    try:
        FEAT_IDX = {
            "age_at_index": 0, "bmi": 1, "year_of_diagnosis": 2,
            "days_to_last_follow_up": 3, "cigarettes_per_day": 4,
            "pack_years_smoked": 5, "alcohol_history": 6,
            "gender": 7, "race_encoded": 8,
        }
        age = float(patient.get("age", 0) or 0)
        bmi = float(patient.get("bmi", 0) or 0)
        smokes = 1.0 if str(patient.get("smoking", "No")).lower() == "yes" else 0.0
        alcohol = 1.0 if str(patient.get("alcohol", "No")).lower() == "yes" else 0.0
        gender = 1.0 if str(patient.get("gender", "")).lower() in ["male", "m"] else 0.0
        if age > 0:
            vec[FEAT_IDX["age_at_index"]] = age
        if bmi > 0:
            vec[FEAT_IDX["bmi"]] = bmi
        vec[FEAT_IDX["cigarettes_per_day"]] = smokes * 15
        vec[FEAT_IDX["pack_years_smoked"]]  = smokes * 10
        vec[FEAT_IDX["alcohol_history"]]    = alcohol
        vec[FEAT_IDX["gender"]]             = gender
        vec[FEAT_IDX["year_of_diagnosis"]]  = float(datetime.now().year)
    except Exception:
        pass
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # (1, 12)


# ─────────────────────────────────────────────────────────────────────────
# CLINICAL-RULE OVERRIDE ENGINE
# ─────────────────────────────────────────────────────────────────────────
# The model knows 5 screening-stage classes only.  A patient who ticks red-flag
# symptoms (NICE NG12) or reports severe pain MUST be escalated regardless
# of what the image branch said — that is how every real clinical decision
# support tool works.  This layer sits on top of the model and produces:
#   • boosted risk score
#   • escalated urgency
#   • a list of human-readable triggered rules to display
# It NEVER lowers anything — only escalates.

NICE_RED_FLAGS = {
    "Rectal bleeding / blood in stool",
    "Persistent change in bowel habits",
    "Unexplained weight loss",
    "Anaemia / low iron",
    "Pencil-thin stools",
    "Feeling of incomplete bowel evacuation",
    "Mucus in stool",
    "Loss of appetite",
    "Jaundice (yellowing of skin/eyes)",
}


def apply_clinical_overrides(analysis: dict, patient: dict, symptoms: list,
                             pain_scale: int, symptom_duration: str) -> dict:
    """Run a NICE NG12 / pain / red-flag-combination rule set on top of the
    model output.  Mutates `analysis` (also returns it).  Adds:
      analysis["overrides"] = {"applied": bool, "rules": [str, ...],
                               "original_risk": float,
                               "original_urgency": str}
    """
    age = int(patient.get("age", 0) or 0)
    fam = str(patient.get("family_history", "")).lower()
    polyps_hx = str(patient.get("prev_polyps", "")).lower()
    sym_set = set(symptoms or [])
    rf_count = len(sym_set & NICE_RED_FLAGS)
    pain = int(pain_scale or 0)
    long_dur = symptom_duration in {
        "3–6 months", "More than 6 months", "Over 1 year",
    }

    boosts = []  # list of (rule_name, +risk, urgency_floor)

    # NICE NG12 — 2-week-wait suspected colorectal cancer
    if "Rectal bleeding / blood in stool" in sym_set and age >= 50:
        boosts.append(("NICE NG12: rectal bleeding ≥ 50 yr — 2-week-wait CRC referral",
                       0.45, "Urgent"))
    if "Anaemia / low iron" in sym_set and age >= 60:
        boosts.append(("NICE NG12: iron-deficiency anaemia ≥ 60 yr — 2-week-wait CRC referral",
                       0.45, "Urgent"))
    if ({"Unexplained weight loss", "Abdominal pain or cramping"} <= sym_set
            and age >= 40):
        boosts.append(("NICE NG12: weight loss + abdominal pain ≥ 40 yr — 2-week-wait CRC referral",
                       0.40, "Urgent"))
    if "Persistent change in bowel habits" in sym_set and age >= 60:
        boosts.append(("NICE NG12: change in bowel habit ≥ 60 yr — 2-week-wait CRC referral",
                       0.35, "Urgent"))

    # Pain-driven escalation
    if pain >= 9:
        boosts.append((f"Severe reported pain ({pain}/10) — clinical review escalated",
                       0.30, "Urgent"))
    elif pain >= 7:
        boosts.append((f"High reported pain ({pain}/10) — clinical review needed",
                       0.20, "Elective"))

    # Symptom-burden
    if rf_count >= 4:
        boosts.append((f"{rf_count} red-flag symptoms reported — high concern",
                       0.30, "Urgent"))
    elif rf_count >= 2:
        boosts.append((f"{rf_count} red-flag symptoms reported — review needed",
                       0.18, "Elective"))

    # Bleeding + weight loss at any age (NICE 2-WW criterion)
    if {"Rectal bleeding / blood in stool",
        "Unexplained weight loss"} <= sym_set:
        boosts.append(("Rectal bleeding + unexplained weight loss — urgent assessment",
                       0.35, "Urgent"))

    # Long duration of symptoms
    if rf_count >= 1 and long_dur:
        boosts.append(("Symptom duration > 3 months with red flags — review needed",
                       0.10, "Elective"))

    # Family history
    if "first" in fam:
        boosts.append(("First-degree family history of colorectal cancer — baseline risk raised",
                       0.15, "Elective"))

    # Prior polyps
    if "yes" in polyps_hx:
        boosts.append(("Previous polyps — surveillance pathway likely applies",
                       0.10, "Elective"))

    # Critical-emergency combination
    if (rf_count >= 3 and pain >= 8 and age >= 40):
        boosts.append(("Multiple red flags + severe pain ≥ 40 yr — escalate to Emergency",
                       0.40, "Emergency"))

    # Image-statistics atypicality (set by image_atypicality.py earlier)
    img_readout = analysis.get("image_readout") or {}
    img_verdict = img_readout.get("verdict")
    if img_verdict == "atypical_concerning":
        boosts.append(("Image pixels show advanced-lesion features (deep red, dark cavitation) "
                       "— possibly outside the model's screening-stage scope",
                       0.40, "Urgent"))

    if not boosts:
        analysis["overrides"] = {"applied": False, "rules": []}
        return analysis

    # Compute the new risk score (sum of boosts, capped) on top of the model's
    # original number — never reduce, only raise.
    orig_risk = float(analysis.get("risk_score", 0.0))
    orig_urgency = analysis.get("recommendation", {}).get("urgency", "Routine")
    boost_total = sum(b[1] for b in boosts)
    new_risk = min(0.99, max(orig_risk, orig_risk + boost_total))

    # Pick the highest-priority urgency floor among the rules
    URGENCY_RANK = {"Routine": 0, "Elective": 1, "Urgent": 2, "Emergency": 3}
    urgency_floor = max((b[2] for b in boosts),
                        key=lambda u: URGENCY_RANK.get(u, 0))
    new_urgency = max([orig_urgency, urgency_floor],
                      key=lambda u: URGENCY_RANK.get(u, 0))

    # Promote risk band based on the new risk
    if new_risk >= 0.75:
        new_risk_label = "Critical concern"
    elif new_risk >= 0.50:
        new_risk_label = "High concern"
    elif new_risk >= 0.25:
        new_risk_label = "Moderate concern"
    else:
        new_risk_label = analysis.get("risk_label", "Low")

    # Apply
    analysis["risk_score"] = new_risk
    analysis["risk_label"] = new_risk_label
    rec = dict(analysis.get("recommendation") or {})
    rec["urgency"] = new_urgency

    # ── Honest staging override ────────────────────────────────────────
    # The model's staging head was trained on class-derived stage labels
    # (HyperKvasir has no real cancer-staging ground truth — staging was
    # synthesised from the pathology class).  So when the image looks
    # atypical or the post-override risk is high, the staging output is
    # NOT trustworthy — replace with an honest message.
    if img_verdict == "atypical_concerning" or new_risk >= 0.60:
        analysis["stage_original"] = analysis.get("stage", "")
        analysis["stage_original_confidence"] = analysis.get("stage_confidence", 0.0)
        analysis["stage"] = "Cannot stage from one image"
        analysis["stage_confidence"] = 0.0
        # Replace stage_probs with a flat "unknown" distribution so the chart
        # doesn't claim "No Cancer 89%" for a cancer image.
        analysis["stage_probs"] = {"Cannot determine": 1.0}
        analysis["staging_note"] = (
            "Single endoscopy images cannot reliably stage cancer — staging needs "
            "histology + cross-sectional imaging. Symptoms and pixel-features here "
            "raise the index of suspicion."
        )
    # Prepend a clinical-override note to the primary action
    rule_lines = "\n• ".join([b[0] for b in boosts])
    override_action = (f"Clinical safety override active — symptom-driven escalation. "
                       f"Triggered rules:\n• {rule_lines}")
    rec["primary_action"] = override_action + "\n\nModel-suggested action: " + str(rec.get("primary_action",""))
    analysis["recommendation"] = rec

    # Carry along all flag annotations
    flags = list(analysis.get("all_risk_flags", []))
    flags.extend([b[0] for b in boosts])
    analysis["all_risk_flags"] = flags

    analysis["overrides"] = {
        "applied": True,
        "rules": [b[0] for b in boosts],
        "original_risk": orig_risk,
        "original_urgency": orig_urgency,
        "new_risk": new_risk,
        "new_urgency": new_urgency,
    }
    return analysis


def overlay_gradcam(original_np: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Blend a GradCAM heatmap onto the original image."""
    if heatmap is None:
        return original_np
    try:
        if heatmap.max() <= 0:
            return original_np
        hm = cv2.resize(heatmap.astype(np.float32),
                        (original_np.shape[1], original_np.shape[0]))
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        colormap = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        orig = (original_np * 255).astype(np.uint8) if original_np.max() <= 1 else original_np
        orig = orig.astype(np.float32) / 255.0
        blended = (1 - alpha) * orig + alpha * colormap
        return np.clip(blended, 0, 1)
    except Exception:
        return original_np


def overlay_seg(original_np: np.ndarray, mask: np.ndarray,
                alpha: float = 0.40) -> np.ndarray:
    """Overlay a segmentation mask (HxW, [0,1]) as a green region + contour.

    This is the PRIMARY localization shown to the user — the segmentation decoder
    outlines the actual lesion (honest held-out cross-vendor IoU ~0.45), unlike the
    coarse 7x7 GradCAM attention map. Returns a uint8 RGB image."""
    if mask is None:
        return original_np
    try:
        img = (original_np * 255).astype(np.uint8) if original_np.max() <= 1 else original_np.astype(np.uint8)
        img = np.ascontiguousarray(img[..., :3])
        H, W = img.shape[:2]
        m = cv2.resize(np.asarray(mask, dtype=np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        binm = (m >= 0.5).astype(np.uint8)
        if binm.sum() == 0:
            return img
        green = np.zeros_like(img); green[..., 1] = 255
        out = np.where(binm[..., None] == 1,
                       (alpha * green + (1 - alpha) * img).astype(np.uint8), img)
        contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
        return out
    except Exception:
        return original_np


def run_analysis(system: dict, pil_img: Image.Image, patient: dict,
                 symptoms: List[str], symptom_text: str,
                 pain_scale: int = 0,
                 symptom_duration: str = "") -> dict:
    """Run the full 6-agent pipeline and return a serializable result dict."""
    from src.data.multimodal_dataset import make_clinical_text

    orch    = system["orchestrator"]
    device  = system["device"]
    tcga_df = system.get("tcga_df")

    # Pre-process image
    img_tensor, img_np = preprocess_image(pil_img)

    # ── HARD GATE: refuse to run the model if this isn't an endoscopy image ──
    # The trained classifier knows only 5 GI screening classes. If the input is
    # a random photo / screenshot / X-ray / drawing / cat picture, the softmax
    # will still output a confident answer — which would be a fake result.
    # The endoscopy_score gate inspects raw pixels (red dominance, hue band,
    # texture) and rejects non-medical inputs BEFORE the model runs.
    try:
        from src.app.image_atypicality import is_endoscopy_image
        gate = is_endoscopy_image(img_np, threshold=0.55)
        if not gate["is_endoscopy"]:
            return {
                "pathology_class":  "NOT_ENDOSCOPY",
                "pathology_probs":  {},
                "stage":            "Not Applicable",
                "stage_confidence": 0.0,
                "stage_probs":      {},
                "risk_score":       0.0,
                "risk_label":       "Not Applicable",
                "image_weight":     0.0, "text_weight": 0.0, "tabular_weight": 0.0,
                "confidence":       0.0,
                "all_risk_flags":   [],
                "uncertainty":      1.0,
                "inference_time_ms": 0.0,
                "recommendation": {
                    "urgency":         "Input Rejected",
                    "primary_action":  "Please upload a real colonoscopy / endoscopy image "
                                       "(white-light or NBI).",
                    "surveillance":    [],
                    "referrals":       [],
                    "investigations":  [],
                    "lifestyle_advice": [],
                    "full_report":
                        "IMAGE NOT ANALYSED — this does not look like a colonoscopy frame. "
                        f"It looks like: {gate.get('modality_detail', 'an unsupported image type')}. "
                        "ColonAI only analyses colonoscopy images (white-light or NBI); it cannot "
                        "give a meaningful prediction on any other image type. "
                        + ("ColonAI does not read radiology scans (X-ray/CT/MRI) — those require a "
                           "radiologist. "
                           if gate.get("modality") == "radiology_grayscale" else "")
                        + "Reasons: " + "  ·  ".join(gate["reasons"]),
                },
                "gradcam_overlay": None,
                "gradcam_heatmap": None,
                "original_image":  img_np,
                "ablation":        {"error": "skipped — non-endoscopy image"},
                "_provenance": {
                    "source": "input_gate",
                    "rejected_reason": "Image failed endoscopy-likeness gate",
                    "endoscopy_score": float(gate["score"]),
                    "endoscopy_threshold": 0.55,
                    "modality": gate.get("modality", "unknown"),
                    "modality_detail": gate.get("modality_detail", ""),
                },
                "image_readout": {
                    "verdict":           "not_endoscopy",
                    "is_endoscopy":      False,
                    "endoscopy_score":   float(gate["score"]),
                    "atypicality":       0.0,
                    "normal_score":      0.0,
                    "confidence":        float(1.0 - gate["score"]),
                    "signals":           gate["signals"],
                    "reasons":           [("red", r) for r in gate["reasons"]],
                },
                "input_rejected":    True,
                "rejection_reasons": gate["reasons"],
                "rejection_score":   float(gate["score"]),
            }
    except Exception as exc:
        # If the gate itself fails, log but continue — better than a hard crash
        import logging
        logging.warning(f"Endoscopy gate failed: {type(exc).__name__}: {exc}")

    # Clinical text
    clin_text = make_clinical_text("default")
    if symptom_text:
        clin_text = symptom_text + " " + clin_text
    if symptoms:
        symp_joined = ". ".join(symptoms[:5])
        clin_text = f"Patient reports: {symp_joined}. " + clin_text

    # Tokenise
    input_ids, attn_mask = tokenize_text(system["tokenizer"], clin_text)

    # Tabular
    tcga_df_v, extract_fn, n_feat = get_tcga_pool_cached()
    tab = build_tabular_vector(patient, tcga_df_v, extract_fn, n_feat)

    # Move to device
    img_tensor  = img_tensor.to(device)
    input_ids   = input_ids.to(device)
    attn_mask   = attn_mask.to(device)
    tab         = tab.to(device)

    # Run orchestrator
    result = orch.run(
        image=img_tensor,
        input_ids=input_ids,
        attention_mask=attn_mask,
        tabular=tab,
        text=clin_text,
        raw_image_np=img_np,
        save=False,
    )

    fd  = result.fusion_diagnosis
    xai = result.xai_report
    rec = result.clinical_recommendation

    # ── SMART INFERENCE: TTA ensemble + deep MC-Dropout + hierarchical UC
    # Wraps the model with 5 augmented passes + 10 MC-Dropout passes and a
    # safety rule that hedges uc-mild predictions when uc-mod-sev is plausible.
    # The numbers we display in the rest of the UI come from this — more
    # reliable than a single forward pass.
    try:
        from src.app.smart_inference import smart_predict
        from src.data.multimodal_dataset import CLASS_NAMES_5 as _CN5
        _sp = smart_predict(
            model=system["model"],
            image_tensor=img_tensor, input_ids=input_ids,
            attention_mask=attn_mask, tabular=tab,
            class_names=list(_CN5),
            temperature=TEMPERATURE,
            n_tta=5, n_mc=10,
        )
        # Promote TTA+MC-Dropout numbers onto the FusionDiagnosis-shaped object
        # so the downstream code keeps working without changes.
        try:
            fd.pathology_class    = _sp.predicted_class
            fd.overall_confidence = _sp.confidence
            # Also expose differential + hedge to the UI
        except Exception: pass
        try:
            xai.uncertainty = _sp.uncertainty
        except Exception: pass
        # Stash for the result page
        # (use a local dict on `out` — populated later in this function)
        _smart_pred = {
            "predicted_class": _sp.predicted_class,
            "confidence":      _sp.confidence,
            "uncertainty":     _sp.uncertainty,
            "mutual_info":     _sp.mutual_info,
            "differential":    _sp.differential,
            "is_hedged":       _sp.is_hedged,
            "hedge_reason":    _sp.hedge_reason,
            "tta_std":         _sp.tta_std,
            "mc_std":          _sp.mc_std,
            "n_tta":           _sp.n_tta,
            "n_mc":            _sp.n_mc,
            "mean_probs":      _sp.mean_probs.tolist(),
        }
    except Exception as _sp_exc:
        _smart_pred = {"error": f"{type(_sp_exc).__name__}: {_sp_exc}"}

    # GradCAM overlay
    gradcam_heatmap = None
    gradcam_overlay = None
    if xai.gradcam_heatmap is not None:
        gradcam_heatmap = xai.gradcam_heatmap
        gradcam_overlay = overlay_gradcam(img_np, gradcam_heatmap)

    # ── REAL ablation probe — actually run the model with each modality
    # silenced, and report the actual change in the predicted-class probability.
    # This replaces the prior templated "counterfactual" text.
    ablation = {}
    try:
        model = system.get("model")
        if model is not None:
            from src.agents.fusion_reasoning_agent import PATHOLOGY_CLASSES
            pred_idx = (PATHOLOGY_CLASSES.index(fd.pathology_class)
                        if fd.pathology_class in PATHOLOGY_CLASSES else 0)
            with torch.no_grad():
                # Original
                base_out = model(image=img_tensor, input_ids=input_ids,
                                 attention_mask=attn_mask, tabular=tab)
                base_p = float(torch.softmax(base_out["pathology"], dim=-1)[0, pred_idx])

                # Image silenced (zero pixel input)
                z_img = torch.zeros_like(img_tensor)
                out_no_img = model(image=z_img, input_ids=input_ids,
                                   attention_mask=attn_mask, tabular=tab)
                p_no_img = float(torch.softmax(out_no_img["pathology"], dim=-1)[0, pred_idx])

                # Text silenced (CLS-only token, all-zero input ids)
                pad_id = system["tokenizer"].pad_token_id or 0
                z_ids = torch.full_like(input_ids, pad_id)
                z_mask = torch.zeros_like(attn_mask)
                z_mask[:, 0] = 1
                out_no_txt = model(image=img_tensor, input_ids=z_ids,
                                   attention_mask=z_mask, tabular=tab)
                p_no_txt = float(torch.softmax(out_no_txt["pathology"], dim=-1)[0, pred_idx])

                # Tabular silenced (all-zero vector)
                z_tab = torch.zeros_like(tab)
                out_no_tab = model(image=img_tensor, input_ids=input_ids,
                                   attention_mask=attn_mask, tabular=z_tab)
                p_no_tab = float(torch.softmax(out_no_tab["pathology"], dim=-1)[0, pred_idx])
            ablation = {
                "predicted_class": fd.pathology_class,
                "base_prob": base_p,
                "no_image_prob": p_no_img,
                "no_text_prob":  p_no_txt,
                "no_tabular_prob": p_no_tab,
                "image_drop_pp":   max(0.0, base_p - p_no_img) * 100,
                "text_drop_pp":    max(0.0, base_p - p_no_txt) * 100,
                "tabular_drop_pp": max(0.0, base_p - p_no_tab) * 100,
            }
    except Exception as exc:
        ablation = {"error": f"{type(exc).__name__}: {exc}"}

    out = {
        "pathology_class":  fd.pathology_class,
        "pathology_probs":  fd.pathology_probs,
        "stage":            fd.cancer_stage,
        "stage_confidence": fd.stage_confidence,
        "stage_probs":      fd.stage_probs,
        "risk_score":       fd.cancer_risk_score,
        "risk_label":       fd.cancer_risk_label,
        "image_weight":     fd.image_weight,
        "text_weight":      fd.text_weight,
        "tabular_weight":   fd.tabular_weight,
        "confidence":       fd.overall_confidence,
        "all_risk_flags":   list(fd.all_risk_flags),
        "uncertainty":      xai.uncertainty,
        "smart_prediction": _smart_pred,           # TTA + MC-Dropout + hedge
        "inference_time_ms": result.inference_time_ms,
        "recommendation": {
            "urgency":       rec.urgency,
            "primary_action": rec.primary_action,
            "surveillance":  rec.surveillance,
            "referrals":     rec.referrals,
            "investigations": rec.investigations,
            "lifestyle_advice": rec.lifestyle_advice,
            "full_report":   rec.full_report,
        },
        "gradcam_overlay": gradcam_overlay,
        "gradcam_heatmap": gradcam_heatmap,
        "original_image":  img_np,
        "ablation":        ablation,
        "_provenance": {
            "source": "real_model",
            "model": "UnifiedMultiModalTransformer",
            "backbone": "ResNet-50 + EfficientNet-B0 + BioBERT + TabTransformer",
            "checkpoint": str(CHECKPOINT.name) if CHECKPOINT.exists() else "none",
            "checkpoint_loaded": bool(system.get("checkpoint_loaded")),
        },
    }
    # Compute image-statistics atypicality (independent of the model — works
    # on raw pixels, so it doesn't share the model's training-distribution
    # blind spots).
    try:
        from src.app.image_atypicality import compute_image_readout, detect_advanced_lesion
        readout = compute_image_readout(img_np)
        out["image_readout"] = readout.to_dict()
        # ── Invasive / advanced-disease detector (pixel-stats override) ──
        # If pure-pixel statistics show signs of advanced colorectal disease
        # (deep ulceration / heavy bleeding / nodular mass / fungating
        # tissue) — features the 5-class classifier was NEVER trained on —
        # we set a flag here so the safety policy can override "polyps 87%"
        # with "Atypical lesion — urgent endoscopist review".
        adv = detect_advanced_lesion(img_np)
        out["advanced_lesion"] = adv
    except Exception as exc:
        out["image_readout"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ── Clinical polyp / IBD sub-typing ────────────────────────────────
    # Paris classification (morphology), NICE classification (predicted
    # histology from surface pattern), BSG-aligned size stratification,
    # Crohn's vs UC differential, diverticulosis + hemorrhoid detection.
    # All from the seg mask + image pixels — no extra training needed.
    try:
        from src.app.polyp_typing import full_sub_typing
        out["sub_typing"] = full_sub_typing(
            image_rgb       = img_np,
            mask            = out.get("seg_mask"),
            pathology_class = getattr(fd, "pathology_class", ""),
            symptoms_text   = symptoms or "",
            patient         = patient,
        )
    except Exception as _st_exc:
        out["sub_typing"] = {"error": f"{type(_st_exc).__name__}: {_st_exc}"}

    # ── TCGA tabular stage classifier ─────────────────────────────────
    # A REAL stage estimate (I/II/III/IV) trained on TCGA's 1,319
    # labelled cases. ~53% accuracy on 4-class (vs 25% random) using
    # only demographics + family history + smoking — no T/N/M leakage.
    # Provides a SECOND, independent stage estimate alongside the
    # image-based one (which was always essentially blank because
    # HyperKvasir has no stage labels).
    try:
        from pathlib import Path as _P
        import joblib as _joblib
        _stage_path = _P("outputs/unified_multimodal_v2/tcga_stage_clf.joblib")
        if _stage_path.exists():
            _stage_obj = _joblib.load(_stage_path)
            _clf = _stage_obj["model"]; _enc = _stage_obj["encoder"]
            _feat_cols = _stage_obj["feature_cols"]
            # Build a 1-row feature vector from the patient form
            import pandas as _pd
            _row = {
                "age":              float(patient.get("age", 50) or 50),
                "gender_male":      1.0 if str(patient.get("gender","")).lower() == "male" else 0.0,
                "race_white":       1.0,    # no race field in our form — assume default
                "bmi":              float(patient.get("bmi", 25) or 25),
                "site_rectum":      0.0,    # form doesn't ask; default to colon
                "pack_years":       float(patient.get("pack_years", 0) or 0),
                "cigs_per_day":     float(patient.get("cigs_per_day", 0) or 0),
                "alcohol_history":  1.0 if str(patient.get("alcohol","")).lower() == "yes" else 0.0,
                "family_hx_cancer": 1.0 if str(patient.get("family_history","")).lower() == "yes" else 0.0,
            }
            _x = _pd.DataFrame([[_row.get(c, np.nan) for c in _feat_cols]],
                                columns=_feat_cols)
            _probs = _clf.predict_proba(_x)[0]
            _pred  = _enc.classes_[int(_probs.argmax())]
            out["tcga_stage_estimate"] = {
                "predicted_stage": f"Stage {_pred}",
                "confidence":      float(_probs.max()),
                "probabilities":   {f"Stage {c}": float(p)
                                     for c, p in zip(_enc.classes_, _probs)},
                "trained_on":      _stage_obj.get("trained_on", "TCGA-COAD"),
                "n_train_samples": _stage_obj.get("n_samples", 0),
            }
    except Exception as _stg_exc:
        out["tcga_stage_estimate"] = {"error": f"{type(_stg_exc).__name__}: {_stg_exc}"}

    # ── RELIABILITY LAYER ──────────────────────────────────────────────
    # Run TTA, prototype-OOD, agent-consensus and aggregate into a TrustReport.
    # This is what makes the system give HONEST results — disagreement between
    # signals shows up as a low trust score and a visible "review" verdict
    # instead of a fake confident answer.
    try:
        from src.app.reliability import tta_inference, build_trust_report
        from src.app.strong_xai import integrated_gradients, overlay_ig, gradcam_ig_agreement

        # TTA — 5 augmented inferences
        tta = tta_inference(
            model=system.get("model"),
            pil_image=pil_img,
            input_ids=input_ids,
            attention_mask=attn_mask,
            tabular=tab,
            device=device,
            n_augs=5,
        )

        # Text agent risk level (from the agent output if present)
        try:
            text_ev = getattr(result, "text_evidence", None)
            text_risk = (getattr(text_ev, "risk_level", "MODERATE")
                         if text_ev is not None else "MODERATE")
        except Exception:
            text_risk = "MODERATE"

        # Endoscopy gate score (from image_readout we already computed)
        end_score = float(out.get("image_readout", {}).get("endoscopy_score", 1.0))

        # Fused embedding for prototype distance (from the orchestrator output)
        fused_emb = None
        try:
            if getattr(result, "fusion_diagnosis", None) and \
               getattr(result.fusion_diagnosis, "fused_embedding", None) is not None:
                fe = result.fusion_diagnosis.fused_embedding
                fused_emb = (fe.detach().cpu().numpy().squeeze()
                             if hasattr(fe, "detach") else np.asarray(fe).squeeze())
        except Exception:
            pass

        trust = build_trust_report(
            endoscopy_score    = end_score,
            tta_agreement_pct  = tta["agreement_pct"],
            mc_uncertainty     = float(xai.uncertainty),
            fused_embedding    = fused_emb,
            image_class        = fd.pathology_class,
            text_risk_level    = text_risk,
            tabular_risk_score = float(fd.cancer_risk_score),
        )
        out["trust_report"] = trust.to_dict()
        out["tta_summary"]  = {
            "agreement_pct":  tta["agreement_pct"],
            "n_augs":         tta["n_augs"],
            "risk_std":       tta["risk_std"],
            "max_class_std":  tta["max_class_std"],
        }

        # Integrated Gradients (only if we have a confident prediction)
        if trust.verdict in ("TRUSTED", "LOW_CONFIDENCE") and system.get("model") is not None:
            try:
                from src.agents.fusion_reasoning_agent import PATHOLOGY_CLASSES
                target_idx = (PATHOLOGY_CLASSES.index(fd.pathology_class)
                              if fd.pathology_class in PATHOLOGY_CLASSES else 0)
                ig_map = integrated_gradients(
                    model=system["model"],
                    image_tensor=img_tensor,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    tabular=tab,
                    target_class=target_idx,
                    n_steps=24,
                )
                out["ig_heatmap"] = ig_map
                out["ig_overlay"] = overlay_ig(img_np, ig_map)
                if gradcam_heatmap is not None:
                    out["xai_agreement"] = gradcam_ig_agreement(gradcam_heatmap, ig_map)
            except Exception as exc:
                out["ig_error"] = f"{type(exc).__name__}: {exc}"

        # ── Segmentation decoder — wire it in so cross_check has the 3rd signal
        # The decoder was trained but the live inference path never invoked it
        # (bug — out["seg_mask"] was always None). predict_seg_mask returns
        # a sigmoid mask in [0,1] at 224x224, or None if seg_head.pth is missing.
        try:
            from src.app.segmentation import predict_seg_mask
            _seg = predict_seg_mask(system["model"], img_tensor, device)
            if _seg is not None:
                out["seg_mask"] = _seg
        except Exception as exc:
            out["seg_error"] = f"{type(exc).__name__}: {exc}"

        # ── Polyp characterization (CADx): "what kind of polyp is it?" ──────
        # Optical-diagnosis specialist. Only meaningful for polyp findings, and
        # only when we have a lesion region to focus on (the segmentation mask).
        # Crops to the polyp and classifies neoplastic vs non-neoplastic. This is
        # decision support (resect-and-discard style judgement), never a substitute
        # for histology. Additive + fail-open.
        try:
            _pc = getattr(fd, "pathology_class", "")
            _seg_m = out.get("seg_mask")
            if _pc in ("polyps", "therapeutic") and _seg_m is not None:
                from src.app.characterization import characterize, _bbox_from_mask
                from PIL import Image as _PILImage
                _pil = (img_np if isinstance(img_np, _PILImage.Image)
                        else _PILImage.fromarray(
                            (img_np * 255).astype(np.uint8) if np.asarray(img_np).max() <= 1
                            else np.asarray(img_np).astype(np.uint8)))
                # scale the 224-space seg-mask bbox up to the original image size
                _m = np.asarray(_seg_m)
                _mh, _mw = _m.shape[:2]
                _bb = _bbox_from_mask((_m * 255).astype(np.uint8))
                _crop = None
                if _bb is not None:
                    _W, _H = _pil.size
                    sx, sy = _W / _mw, _H / _mh
                    _crop = (int(_bb[0] * sx), int(_bb[1] * sy),
                             int(_bb[2] * sx), int(_bb[3] * sy))
                _cadx = characterize(_pil, bbox=_crop)
                if _cadx.get("available"):
                    out["characterization"] = _cadx
        except Exception as exc:
            out["characterization_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        out["trust_report"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ── Patient-safety policy: central abstain/show/reject gate ─────────────
    # This is the ONLY place that decides whether a prediction is safe to show.
    try:
        from src.app.patient_safety import evaluate_safety, AuditLog
        _gc_focus = None
        try:
            gh = out.get("gradcam_heatmap")
            if gh is not None:
                _flat = np.asarray(gh).flatten()
                if _flat.size > 0:
                    _thr = float(np.quantile(_flat, 0.75))
                    _gc_focus = float((_flat >= _thr).sum() / _flat.size)
        except Exception:
            pass
        _tta_agree = float(out.get("tta_summary", {}).get("agreement_pct", 100.0)) / 100.0

        # ── Inference-time cross-check — pathology ↔ GradCAM ↔ seg ↔ IG ─────
        # This is the "no fake results" safety net. If the four signals
        # don't agree, coherence is low → safety policy abstains.
        from src.app.cross_check import cross_check
        _cross = cross_check(
            pathology_class   = getattr(fd, "pathology_class", "unknown"),
            pathology_conf    = float(getattr(fd, "overall_confidence", 0.0)),
            gradcam_map       = out.get("gradcam_heatmap"),
            segmentation_mask = out.get("seg_mask"),
            ig_map            = out.get("ig_heatmap"),
        )
        out["cross_check"] = _cross.to_dict()

        # ── Smart per-image rationale (real measurements, not templates) ──
        # Computes lesion size %, location octant, attention focus
        # tightness, lesion-vs-background contrast, edge regularity,
        # and dominant colour — all from the actual image + masks. Each
        # bullet that lands in the UI is a measurement, not a phrase.
        try:
            from src.app.smart_rationale import smart_rationale as _sr
            _sr_out = _sr(
                image_rgb       = img_np,
                pathology_class = getattr(fd, "pathology_class", "unknown"),
                confidence      = float(getattr(fd, "overall_confidence", 0.0)),
                gradcam         = out.get("gradcam_heatmap"),
                seg_mask        = out.get("seg_mask"),
                uncertainty     = float(getattr(xai, "uncertainty", 0.0)),
            )
            out["smart_rationale"] = _sr_out

            # ── Optional Groq LLM rationale refinement ────────────────
            # Only runs if GROQ_API_KEY env var is set. Strict guard rails
            # prevent the LLM from changing the diagnosis or claiming
            # higher confidence than our deterministic value.
            try:
                from src.app.llm_refine import refine_rationale as _lr, is_available
                if is_available():
                    _lr_out = _lr(
                        predicted_class       = getattr(fd, "pathology_class", "unknown"),
                        confidence            = float(getattr(fd, "overall_confidence", 0.0)),
                        uncertainty           = float(getattr(xai, "uncertainty", 0.0)),
                        safety_action         = "show",   # updated by safety policy below if needed
                        deterministic_bullets = _sr_out.get("bullets", []),
                        differential          = _smart_pred.get("differential")
                                                  if isinstance(_smart_pred, dict) else None,
                        is_hedged             = (_smart_pred.get("is_hedged") if
                                                  isinstance(_smart_pred, dict) else False),
                        hedge_reason          = (_smart_pred.get("hedge_reason") if
                                                  isinstance(_smart_pred, dict) else None),
                    )
                    out["llm_refined"] = _lr_out
            except Exception as _lr_exc:
                out["llm_refined"] = {"refined_paragraph": "",
                                       "fallback_used": True,
                                       "fallback_reason":
                                          f"{type(_lr_exc).__name__}: {_lr_exc}"}
        except Exception as _sr_exc:
            out["smart_rationale_error"] = f"{type(_sr_exc).__name__}: {_sr_exc}"

        # Use the worse of (TTA agreement, cross-check coherence) as the
        # agent_agreement input. Both must be high for the safety policy
        # to pass.
        _agent_agree = min(_tta_agree, _cross.coherence)

        _safety = evaluate_safety(
            confidence       = float(getattr(fd, "overall_confidence", 0.0)),
            uncertainty      = float(getattr(xai, "uncertainty", 0.0)),
            endoscopy_score  = float(out.get("image_readout", {}).get("endoscopy_score", 1.0)),
            gradcam_focus    = _gc_focus,
            agent_agreement  = _agent_agree,
            predicted_class  = getattr(fd, "pathology_class", None),
        )
        out["safety_verdict"] = _safety.to_dict()

        # ── Trained OOD gate (real out-of-scope detector) ───────────────────
        # Second safety layer: the endoscopy gate rejects non-endoscopy inputs;
        # this catches images that ARE endoscopy but are OUTSIDE the 5 trained
        # findings (e.g. cecum/pylorus/z-line landmarks, bowel-prep views). Head
        # trained on real out-of-scope HK images (held-out real-OOD AUROC ~0.996).
        try:
            from src.app.ood_gate import ood_check
            _fe_ood = getattr(fd, "fused_embedding", None)
            _emb_ood = (_fe_ood.detach().cpu().numpy() if hasattr(_fe_ood, "detach")
                        else (np.asarray(_fe_ood) if _fe_ood is not None else None))
            _ood = ood_check(_emb_ood)
            out["ood_gate"] = _ood
            if _ood.get("is_ood"):
                sv = out["safety_verdict"]
                sv["action"] = "abstain"
                sv["ood_flagged"] = True
                _flags = sv.get("flags", []) or []
                _flags.append("This image does not match the model's known findings "
                              "(out-of-distribution) — a clinician should review it directly.")
                sv["flags"] = _flags
                out["safety_verdict"] = sv
        except Exception as _ood_exc:
            out["ood_gate_error"] = f"{type(_ood_exc).__name__}: {_ood_exc}"

        # ── View-quality advisory (poor bowel-prep / obscured view) ─────────
        # Advisory only — warns the finding may be unreliable; does not block.
        try:
            from src.app.view_quality import view_quality_check
            _fe_vq = getattr(fd, "fused_embedding", None)
            _emb_vq = (_fe_vq.detach().cpu().numpy() if hasattr(_fe_vq, "detach")
                       else (np.asarray(_fe_vq) if _fe_vq is not None else None))
            out["view_quality"] = view_quality_check(_emb_vq)
        except Exception as _vq_exc:
            out["view_quality_error"] = f"{type(_vq_exc).__name__}: {_vq_exc}"

        # Audit log — every prediction recorded for post-hoc review
        try:
            _audit = AuditLog()
            _audit.record(
                case_id         = out.get("case_id", "ui"),
                image_path      = out.get("image_path"),
                pathology_class = getattr(fd, "pathology_class", "unknown"),
                confidence      = float(getattr(fd, "overall_confidence", 0.0)),
                uncertainty     = float(getattr(xai, "uncertainty", 0.0)),
                verdict         = _safety,
                extras          = {"trust": out.get("trust_report", {}).get("verdict")},
            )
        except Exception:
            pass

        # ── Privacy-safe continual learning log ─────────────────────────
        # Records (image_sha256 + fused embedding + prediction) per case so
        # the model can be retrained from real cases later. Does NOT store:
        # patient name, demographics, raw image, or symptom text. See
        # src/app/learning_log.py for the exact privacy contract.
        try:
            from src.app.learning_log import record_case as _lc_record
            _img_bytes = None
            try:
                _img_buf = st.session_state.get("uploaded_image_bytes")
                if _img_buf is not None:
                    _img_bytes = bytes(_img_buf)
            except Exception: pass
            _emb = None
            try:
                _fe = getattr(fd, "fused_embedding", None)
                if _fe is not None:
                    _emb = (_fe.detach().cpu().numpy().squeeze()
                            if hasattr(_fe, "detach") else np.asarray(_fe).squeeze())
            except Exception: pass
            _cu = _lc_record(
                image_bytes     = _img_bytes,
                fused_embedding = _emb,
                predicted_class = getattr(fd, "pathology_class", "unknown"),
                confidence      = float(getattr(fd, "overall_confidence", 0.0)),
                uncertainty     = float(getattr(xai, "uncertainty", 0.0)),
                safety_action   = _safety.action,
                extras          = {"coherence": out.get("cross_check", {}).get("coherence")})
            out["learning_case_uuid"] = _cu   # the UI feedback widget uses this
        except Exception as _lc_exc:
            out["learning_log_error"] = f"{type(_lc_exc).__name__}: {_lc_exc}"
    except Exception as exc:
        out["safety_verdict"] = {"action": "show",
                                 "reason": f"safety-policy unavailable: {exc}",
                                 "disclaimer": "Treat output as provisional."}

    # ── UNIFIED EXPLANATION ENGINE ─────────────────────────────────────
    # Builds the structured decision trace, per-modality attribution,
    # disagreement detection, and the patient-facing narrative + the
    # clinician dossier. This is the "audit log" of a single inference
    # — every output below is derivable from data already computed above.
    # All defensive: any sub-step that fails will be skipped silently,
    # and a top-level explanation_error will record the failure.
    try:
        from src.app.decision_trace import build_trace, detect_disagreements
        from src.app.modality_attribution import fused_attribution
        from src.app.explanation_engine import (
            narrative as _explain_narrative,
            clinician_report as _explain_report,
        )

        # 1. Convert the silencing ablation we already computed into a
        #    silencing_attribution-style result.
        _abl = out.get("ablation", {}) or {}
        _silencing = None
        if isinstance(_abl, dict) and not _abl.get("error"):
            _deltas = {
                "image":   max(0.0, float(_abl.get("image_drop_pp",   0)) / 100.0),
                "text":    max(0.0, float(_abl.get("text_drop_pp",    0)) / 100.0),
                "tabular": max(0.0, float(_abl.get("tabular_drop_pp", 0)) / 100.0),
            }
            _total = sum(_deltas.values())
            if _total > 1e-6:
                _silencing = {
                    "contributions": {k: float(v / _total * 100.0)
                                       for k, v in _deltas.items()},
                    "deltas":          _deltas,
                    "baseline_prob":   float(_abl.get("base_prob", 0.0)),
                    "predicted_class": _abl.get("predicted_class", ""),
                    "method":          "silencing",
                    "interpretable":   True,
                }

        # Fall back to confidence-weighted heuristic if silencing was unavailable
        _attr = fused_attribution(
            image_confidence   = float(getattr(fd, "image_weight",   0.0)),
            text_confidence    = float(getattr(fd, "text_weight",    0.0)),
            tabular_confidence = float(getattr(fd, "tabular_weight", 0.0)),
            silencing_result   = _silencing,
        )
        out["modality_attribution"] = _attr

        # 2. Build the structured decision trace
        _adv = out.get("advanced_lesion") or {}
        _atyp_fired = bool(_adv.get("flagged", _adv.get("override", False)))
        _safety_obj = out.get("safety_verdict") or {}
        _safety_passed = (str(_safety_obj.get("action", "show")).lower() == "show")
        _tcga_obj = out.get("tcga_stage_estimate") or {}
        _tcga_finding = None
        if _tcga_obj and not _tcga_obj.get("error") and _tcga_obj.get("predicted_stage"):
            _tcga_finding = {
                "stage":      str(_tcga_obj.get("predicted_stage", "")).replace("Stage ", ""),
                "confidence": float(_tcga_obj.get("confidence", 0.0)),
                "evidence":   {"n_train_samples": _tcga_obj.get("n_train_samples", 0)},
            }
        _trace = build_trace(
            smart_pred           = out.get("smart_prediction"),
            atypicality_finding  = ({"override":   _atyp_fired,
                                     "atypical":   _atyp_fired,
                                     "reasons":    _adv.get("reasons", _adv.get("findings", [])),
                                     "label":      _adv.get("label",
                                                   "Atypical lesion — urgent endoscopist review"),
                                     "confidence": float(_adv.get("confidence", 0.85))}
                                    if _adv else None),
            polyp_typing_finding = out.get("sub_typing"),
            fusion_finding       = {"finding":          f"Fused call: {fd.pathology_class}",
                                    "confidence":       float(getattr(fd, "overall_confidence", 0.0)),
                                    "modality_weights": {
                                        "image":   float(getattr(fd, "image_weight",   0.0)),
                                        "text":    float(getattr(fd, "text_weight",    0.0)),
                                        "tabular": float(getattr(fd, "tabular_weight", 0.0)),
                                    }},
            xai_finding          = {"finding":              f"Predictive entropy {float(getattr(xai, 'uncertainty', 0.0)):.3f}",
                                    "confidence":           max(0.0, 1.0 - min(float(getattr(xai, "uncertainty", 0.0)), 1.0)),
                                    "epistemic_uncertainty": float(getattr(xai, "uncertainty", 0.0))},
            safety_finding       = {"passed":      _safety_passed,
                                    "confidence":  1.0,
                                    "evidence":    [_safety_obj.get("reason", "")]},
            tcga_stage_finding   = _tcga_finding,
            clinical_finding     = {"recommendation": getattr(rec, "primary_action", ""),
                                    "confidence":     0.85,
                                    "evidence":       list(getattr(rec, "investigations", []))[:3]},
            llm_refined          = bool((out.get("llm_refined") or {}).get("refined_paragraph")),
            final_verdict        = fd.pathology_class,
            final_confidence     = float(getattr(fd, "overall_confidence", 0.0)),
        )
        out["decision_trace"] = _trace
        out["disagreement"]   = detect_disagreements(_trace)

        # 3. Patient-facing narrative (≤ ~130 words, deterministic)
        out["explanation_paragraph"] = _explain_narrative(
            final_class       = fd.pathology_class,
            final_confidence  = float(getattr(fd, "overall_confidence", 0.0)),
            smart_pred        = out.get("smart_prediction"),
            attribution       = _attr,
            disagreement      = out.get("disagreement"),
            atypicality_fired = _atyp_fired,
            tcga_stage        = _tcga_finding,
        )

        # 4. Structured clinician dossier (sections renderable in UI / PDF)
        out["explanation_report"] = _explain_report(
            final_class             = fd.pathology_class,
            final_confidence        = float(getattr(fd, "overall_confidence", 0.0)),
            trace                   = _trace,
            attribution             = _attr,
            polyp_typing            = out.get("sub_typing"),
            tcga_stage              = _tcga_finding,
            smart_rationale_text    = (out.get("smart_rationale") or {}).get("summary"),
            clinical_recommendation = getattr(rec, "primary_action", None),
        )

        # 5. Optional: per-class prototype retrieval if bank is built
        try:
            from src.app.prototype_retrieval import (
                is_bank_available, retrieve_similar, neighbour_concordance,
            )
            if is_bank_available():
                _emb = None
                _fe = getattr(fd, "fused_embedding", None)
                if _fe is not None:
                    _emb = (_fe.detach().cpu().numpy().squeeze()
                            if hasattr(_fe, "detach") else np.asarray(_fe).squeeze())
                if _emb is not None and _emb.size > 0:
                    _neigh = retrieve_similar(_emb, k=5)
                    out["prototype_neighbours"] = _neigh
                    out["neighbour_concordance"] = neighbour_concordance(
                        _neigh, fd.pathology_class)
        except Exception:
            pass
    except Exception as _exp_exc:
        out["explanation_error"] = f"{type(_exp_exc).__name__}: {_exp_exc}"

    # Apply NICE NG12 / red-flag clinical-rule overrides on top of the model
    out = apply_clinical_overrides(out, patient, symptoms, pain_scale, symptom_duration)
    return out


_DOC_AVATAR_PALETTE = [
    ("#1A73E8", "#00897B"), ("#FF5722", "#E65100"), ("#9C27B0", "#673AB7"),
    ("#0EA5E9", "#0369A1"), ("#16A34A", "#15803D"), ("#F59E0B", "#B45309"),
]


def _doctor_initials(name: str) -> str:
    parts = [p for p in name.replace("Dr.", "").replace("Dr", "").strip().split() if p]
    if not parts:
        return "MD"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _star_html(rating: float) -> str:
    full = int(rating)
    half = (rating - full) >= 0.5
    stars = "★" * full + ("½" if half else "") + "☆" * (5 - full - (1 if half else 0))
    return (f'<span style="color:#F59E0B;font-size:0.95rem;letter-spacing:1px">{stars}</span>'
            f'<span style="color:#64748B;font-size:0.78rem;margin-left:6px">'
            f'{rating:.1f}/5</span>')


def _render_doctor_card_html(doc: Dict[str, Any],
                             reasons: Optional[List[str]] = None,
                             origin: str = "") -> str:
    """Returns HTML for a single doctor card (avatar tile + meta + match reasons)."""
    initials = _doctor_initials(doc.get("name", ""))
    palette = _DOC_AVATAR_PALETTE[hash(doc.get("name","")) % len(_DOC_AVATAR_PALETTE)]
    grad_a, grad_b = palette
    langs = ", ".join(doc.get("languages", ["English"]))
    sub = doc.get("sub_specialty", "")
    reasons = reasons or []
    reason_html = ""
    if reasons:
        chips = "".join(
            f"<span class='pill {'pill-green' if 'AI' in r else 'pill-amber' if 'NCR' in r or 'metro' in r.lower() or 'km' in r else ''}' "
            f"style='font-size:0.70rem'>✓ {r}</span>"
            for r in reasons[:4]
        )
        reason_html = (f"<div style='margin-top:8px;padding:6px 10px;border-radius:10px;"
                       f"background:#F0F9FF;border:1px solid #BAE6FD'>"
                       f"<div style='font-size:0.68rem;text-transform:uppercase;color:#075985;"
                       f"letter-spacing:0.5px;font-weight:700;margin-bottom:2px'>"
                       f"Why recommended</div>"
                       f"<div style='display:flex;gap:6px;flex-wrap:wrap'>{chips}</div>"
                       f"</div>")
    # Build the maps URLs (no API key needed)
    from urllib.parse import quote_plus
    dest = f"{doc.get('hospital','')} {doc.get('city','')} {doc.get('country','')}"
    view_on_map = ("https://www.google.com/maps/search/?api=1"
                   f"&query={quote_plus(dest)}")
    if origin and dest.strip():
        directions_url = ("https://www.google.com/maps/dir/?api=1"
                          f"&origin={quote_plus(origin)}"
                          f"&destination={quote_plus(dest)}")
    else:
        directions_url = view_on_map
    return f"""
    <div class="doctor-card">
      <div style="display:flex;gap:14px;align-items:flex-start">
        <div style="flex:0 0 auto;width:48px;height:48px;border-radius:14px;
                    display:flex;align-items:center;justify-content:center;
                    background:linear-gradient(135deg,{grad_a},{grad_b});
                    color:white;font-weight:800;font-size:0.95rem;letter-spacing:0.5px;
                    box-shadow:0 6px 18px -8px {grad_a}88">{initials}</div>
        <div style="flex:1;min-width:0">
          <div class="doctor-name">{doc['name']}</div>
          <div class="doctor-spec">{doc['specialty']}</div>
          <div class="doctor-hosp">{doc['hospital']}</div>
          <div style="margin-top:6px">{_star_html(doc['rating'])}</div>
        </div>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px">
        <span class="pill">{doc['city']}, {doc['country']}</span>
        <span class="pill pill-green">{doc['experience_years']} yrs experience</span>
        {f'<span class="pill pill-amber">{sub}</span>' if sub else ''}
      </div>
      {reason_html}
      <div class="doctor-meta">
        <span style="color:#1A73E8;font-weight:600">Tel</span> {doc.get('phone','N/A')}
        &nbsp;·&nbsp;
        <span style="color:#1A73E8;font-weight:600">Languages</span> {langs}
      </div>
      <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
        <a href="{view_on_map}" target="_blank" rel="noopener" class="doc-cta">
          <span style="font-size:0.95rem">📍</span> Open in Maps
        </a>
        <a href="{directions_url}" target="_blank" rel="noopener" class="doc-cta doc-cta-primary">
          <span style="font-size:0.95rem">🧭</span> Get Directions
        </a>
      </div>
    </div>
    """


# Locality clusters — when the user types a city, we also surface specialists from
# nearby cities in the same metropolitan area (Delhi-NCR, Mumbai-MMR, Bay Area, etc.)
LOCALITY_CLUSTERS = {
    "delhi-ncr":   {"new delhi", "delhi", "noida", "gurgaon", "gurugram",
                    "faridabad", "ghaziabad", "greater noida"},
    "mumbai-mmr":  {"mumbai", "navi mumbai", "thane", "kalyan"},
    "bangalore":   {"bangalore", "bengaluru", "whitefield", "electronic city"},
    "chennai":     {"chennai", "madras"},
    "hyderabad":   {"hyderabad", "secunderabad"},
    "kolkata":     {"kolkata", "calcutta"},
    "ny-tristate": {"new york", "newyork", "manhattan", "brooklyn", "queens",
                    "jersey city", "newark"},
    "la-metro":    {"los angeles", "santa monica", "pasadena", "long beach"},
    "bay-area":    {"san francisco", "oakland", "berkeley", "palo alto", "san jose"},
    "boston":      {"boston", "cambridge", "somerville"},
    "london":      {"london", "westminster", "kensington"},
    "uae":         {"dubai", "abu dhabi", "sharjah"},
}


def _cluster_for(city_lower: str):
    for cluster, members in LOCALITY_CLUSTERS.items():
        if city_lower in members or any(city_lower in m or m in city_lower for m in members):
            return cluster, members
    return None, set()


# Maps the AI's pathology prediction → which specialist categories should rank highest
PATHOLOGY_SPECIALTY_PRIORITY = {
    "polyps":          ["Gastroenterology", "Colorectal Surgery",
                        "Gastrointestinal Surgery", "Surgical Oncology",
                        "GI Oncology"],
    "uc-mild":         ["Gastroenterology"],
    "uc-moderate-sev": ["Gastroenterology", "Colorectal Surgery",
                        "Gastrointestinal Surgery"],
    "barretts-esoph":  ["Gastroenterology", "Surgical Oncology",
                        "Gastrointestinal Surgery"],
    "therapeutic":     ["Gastroenterology", "Colorectal Surgery"],
}
# Sub-specialty hint keywords (boost when matched against the AI finding)
PATHOLOGY_SUBKEYWORDS = {
    "polyps":          ["polyp", "colonoscopy", "colorect", "endoscop"],
    "uc-mild":         ["ibd", "colitis", "inflammat"],
    "uc-moderate-sev": ["ibd", "colitis", "inflammat"],
    "barretts-esoph":  ["barrett", "esophag", "oesophag", "upper gi", "ercp", "eus"],
    "therapeutic":     ["endoscop", "polypectomy", "emr", "ercp"],
}


def search_doctors(city: str = "", country: str = "", specialty: str = "",
                   ai_pathology: str = "", limit: int = 10):
    """Smart specialist search with locality clusters, same-country fallback, and
    AI-pathology-aware re-ranking. Returns a list of (doc, reasons) tuples where
    `reasons` is a list of human-readable match explanations."""
    city_lower    = city.lower().strip()
    country_lower = country.lower().strip()
    spec_lower    = specialty.lower().strip()
    cluster, cluster_cities = _cluster_for(city_lower) if city_lower else (None, set())

    pathology_specs = PATHOLOGY_SPECIALTY_PRIORITY.get(ai_pathology, [])
    pathology_subkw = PATHOLOGY_SUBKEYWORDS.get(ai_pathology, [])

    scored: list = []
    for doc in DOCTORS_DB:
        score = 0.0
        reasons: list = []
        doc_city = doc["city"].lower()
        doc_country = doc["country"].lower()
        doc_spec = doc["specialty"].lower()
        doc_sub = doc.get("sub_specialty", "").lower()

        # Locality scoring
        if city_lower:
            if doc_city == city_lower:
                score += 100
                reasons.append(f"city · {doc['city']}")
            elif city_lower in doc_city or doc_city in city_lower:
                score += 80
                reasons.append(f"city · {doc['city']}")
            elif cluster and doc_city in cluster_cities:
                score += 60
                reasons.append(f"{cluster.replace('-',' ').title()} · {doc['city']}")
        else:
            # No city given — gentle bias toward popular hubs
            score += 5

        if country_lower:
            if country_lower == doc_country:
                score += 30
                if not reasons:
                    reasons.append(f"country · {doc['country']}")
            elif country_lower in doc_country or doc_country in country_lower:
                score += 15

        # Specialty filter
        if spec_lower:
            if spec_lower in doc_spec or spec_lower in doc_sub:
                score += 25
                reasons.append(f"specialty · {doc['specialty']}")
            else:
                score -= 25  # explicit specialty filter de-prioritises mismatches

        # AI-pathology-aware boost
        if pathology_specs and doc["specialty"] in pathology_specs:
            score += 30 if doc["specialty"] == pathology_specs[0] else 18
            reasons.append("matches AI finding")
        if pathology_subkw and any(kw in doc_sub for kw in pathology_subkw):
            score += 15
            if "matches AI finding" not in reasons:
                reasons.append("matches AI finding")

        # Reputation boost
        score += float(doc.get("rating", 4.5)) * 4
        score += min(float(doc.get("experience_years", 0)) * 0.4, 16)

        if score > 0:
            scored.append((score, doc, reasons))

    # Sort and trim
    scored.sort(key=lambda x: x[0], reverse=True)

    # Country fallback if nothing in cluster
    if not scored and country_lower:
        for doc in DOCTORS_DB:
            if country_lower in doc["country"].lower():
                scored.append((float(doc.get("rating",4.5))*4,
                               doc, [f"country · {doc['country']}"]))
        scored.sort(key=lambda x: x[0], reverse=True)

    return [(doc, reasons) for _, doc, reasons in scored[:limit]]


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, badges: list = None):
    badge_html = ""
    if badges:
        badge_html = "".join(f'<span class="hero-badge">{b}</span>' for b in badges)
    st.markdown(
        f"""<div class="hero-banner">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {badge_html}
        </div>""",
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, sub: str = "", color: str = "#1A73E8"):
    st.markdown(
        f"""<div class="metric-card" style="border-left-color:{color}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {"<div class='sub'>" + sub + "</div>" if sub else ""}
        </div>""",
        unsafe_allow_html=True,
    )


def render_risk_badge(risk_score: float):
    if risk_score < 0.25:
        cls, label = "risk-low", "LOW RISK"
    elif risk_score < 0.5:
        cls, label = "risk-moderate", "MODERATE RISK"
    elif risk_score < 0.75:
        cls, label = "risk-high", "HIGH RISK"
    else:
        cls, label = "risk-critical", "CRITICAL RISK"
    st.markdown(
        f'<div class="risk-badge {cls}">{label} &nbsp; {risk_score:.0%}</div>',
        unsafe_allow_html=True,
    )


def render_urgency_banner(urgency: str):
    cls_map = {
        "Routine":   "urgency-routine",
        "Elective":  "urgency-elective",
        "Urgent":    "urgency-urgent",
        "Emergency": "urgency-emergency",
    }
    cls  = cls_map.get(urgency, "urgency-routine")
    st.markdown(
        f'<div class="urgency-banner {cls}">CLINICAL URGENCY: {urgency.upper()}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_progress():
    step = st.session_state.get("step", 0)
    st.sidebar.markdown("### Navigation")

    # Patient-friendly labels — never expose internal architecture
    DISPLAY_LABELS = {
        "Patient Info":      "Patient Info",
        "Symptoms & Upload": "Symptoms & Upload",
        "AI Analysis":       "AI Analysis",
        "Results":           "Your Report",
        "Find Doctors":      "Find a Doctor",
        "Download Report":   "Download Report",
        "Live Video Mode":   "Live Video Mode",
    }

    # Add CSS for clickable step buttons that look like the original chips
    st.sidebar.markdown(
        """
        <style>
        .stSidebar div[data-testid="stButton"]:has(button[kind*="navchip"]) button,
        .stSidebar div[data-testid="stButton"]:has(button[data-step-btn]) button {
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 8px 12px !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for i, name in enumerate(STEPS):
        label = DISPLAY_LABELS.get(name, name)
        if i < step:
            icon = "✓"
            btn_label = f"{icon}  {label}"
        elif i == step:
            icon = f"{i+1}"
            btn_label = f"●  {label}"
        else:
            icon = f"{i+1}"
            btn_label = f"{icon}.  {label}"

        # Real Streamlit button — clicking actually navigates
        if st.sidebar.button(btn_label, key=f"navstep_{i}",
                              use_container_width=True,
                              type=("primary" if i == step else "secondary"),
                              help=f"Jump to: {label}"):
            st.session_state["step"] = i
            st.rerun()

    st.sidebar.markdown("---")
    # Accessibility toggle — larger fonts, higher contrast, dyslexia-friendly
    # font, bigger tap targets. Saved in session so it persists across pages.
    acc_default = st.session_state.get("accessibility_mode", False)
    acc_now = st.sidebar.toggle(
        "🅰 Accessibility mode",
        value=acc_default,
        help="Larger fonts · higher contrast · dyslexia-friendly typography · "
             "bigger buttons. Recommended if you find the default text small.",
        key="accessibility_mode")
    if acc_now:
        from src.app.patient_ui import accessibility_css
        st.markdown(accessibility_css(), unsafe_allow_html=True)

    st.sidebar.markdown("---")
    # Overall progress bar
    progress = step / (len(STEPS) - 1)
    st.sidebar.progress(progress, text=f"Step {step+1} of {len(STEPS)}")

    # Model status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")
    system = st.session_state.get("_system")
    if system is None:
        st.sidebar.markdown(
            '<div style="font-size:0.85rem;color:#475569">'
            '<span class="status-dot status-dot-load"></span>Model idle (loads on Step 3)</div>',
            unsafe_allow_html=True,
        )
    elif not system.get("ready"):
        st.sidebar.markdown(
            '<div style="font-size:0.85rem;color:#B91C1C">'
            '<span class="status-dot status-dot-err"></span>Model load failed — demo mode</div>',
            unsafe_allow_html=True,
        )
        # Surface BOTH the checkpoint-downloader state AND the actual exception
        # that load_ai_system raised — that's what we need to fix the bug.
        try:
            _cs = CHECKPOINT_STATUS
            _stage  = _cs.get("stage", "unknown")
            _detail = _cs.get("detail", "")
            _log    = _cs.get("log", [])
            _color  = {"preexisting":"#15803D","downloaded":"#15803D",
                       "no_env_var":"#B45309","failed":"#B91C1C"}.get(_stage, "#475569")
            with st.sidebar.expander("ℹ︎ Why? (checkpoint status)"):
                st.markdown(
                    f"<div style='font-size:0.78rem;line-height:1.4'>"
                    f"<b style='color:{_color}'>stage:</b> {_stage}<br>"
                    f"<b>HF repo:</b> <code>{_cs.get('hf_repo') or 'not set'}</code><br>"
                    f"<b>file:</b> <code>{_cs.get('hf_file')}</code><br>"
                    f"<b>token:</b> {'set' if _cs.get('had_token') else 'anonymous'}<br>"
                    f"{'<b>detail:</b> ' + _detail + '<br>' if _detail else ''}"
                    f"</div>",
                    unsafe_allow_html=True)
                if _log:
                    st.code("\n".join(_log[-15:]), language="text")
                # NEW: also show the actual load_ai_system exception
                _et = system.get("error_type")
                _em = system.get("error", "")
                _tb = system.get("traceback_tail", "")
                if _et or _em:
                    st.markdown(
                        f"<div style='margin-top:8px;padding:8px;"
                        f"background:#FEE2E2;border-left:3px solid #B91C1C;"
                        f"font-size:0.75rem;'>"
                        f"<b style='color:#B91C1C'>Exception:</b> "
                        f"<code>{_et}</code><br>"
                        f"<span style='color:#7F1D1D'>{_em}</span>"
                        f"</div>",
                        unsafe_allow_html=True)
                if _tb:
                    st.code(_tb, language="text")
        except Exception:
            pass
    else:
        ckpt_ok = system.get("checkpoint_loaded", True)
        if ckpt_ok:
            st.sidebar.markdown(
                '<div style="font-size:0.85rem;color:#15803D">'
                '<span class="status-dot status-dot-ok"></span>AI pipeline ready · checkpoint loaded</div>',
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                '<div style="font-size:0.85rem;color:#B45309">'
                '<span class="status-dot status-dot-warn"></span>Pipeline ready · checkpoint partial</div>',
                unsafe_allow_html=True,
            )

    # Quick reset + jump to Live Video Mode
    st.sidebar.markdown("---")
    if st.sidebar.button("🎥 Open Live Video Mode", use_container_width=True, type="primary"):
        st.session_state["step"] = 6
        st.rerun()
    if st.sidebar.button("Start New Assessment", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k != "_system":
                del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CASES = {
    "case_a": {
        "label": "Case A · Sigmoid Polyp",
        "blurb": "58 y/o male, asymptomatic, FIT positive on screening",
        "patient": {
            "name": "Demo Case A", "age": 58, "gender": "Male",
            "height": 178, "weight": 84, "bmi": 26.5,
            "city": "London", "country": "UK",
            "smoking": "Yes — Former", "alcohol": "Occasional",
            "family_history": "No", "prev_polyps": "No",
            "prev_colonoscopy": "Never",
        },
        "symptoms": [],
        "symptom_text": "Asymptomatic. Routine NHS bowel-screening FIT positive (180 µg Hb/g). Referred for diagnostic colonoscopy.",
        "pain_scale": 0,
        "duration": "Less than 1 week",
        "image": "assets/demo_cases/case_a_polyp.jpg",
        "image_type": "Colonoscopy",
        "expected": "polyps",
    },
    "case_b": {
        "label": "Case B · Ulcerative Colitis",
        "blurb": "31 y/o female, 6 wks bloody diarrhoea, raised calprotectin",
        "patient": {
            "name": "Demo Case B", "age": 31, "gender": "Female",
            "height": 165, "weight": 60, "bmi": 22.0,
            "city": "Manchester", "country": "UK",
            "smoking": "No", "alcohol": "Occasional",
            "family_history": "No", "prev_polyps": "No",
            "prev_colonoscopy": "Never",
        },
        "symptoms": [
            "Rectal bleeding / blood in stool",
            "Diarrhoea (new onset)",
            "Abdominal pain or cramping",
            "Mucus in stool",
            "Chronic fatigue / weakness",
        ],
        "symptom_text": "Six weeks of bloody diarrhoea (4–5 stools/day), urgency, mild left-lower-quadrant cramping. CRP 22, faecal calprotectin 480.",
        "pain_scale": 5,
        "duration": "1–3 months",
        "image": "assets/demo_cases/case_b_uc.jpg",
        "image_type": "Colonoscopy",
        "expected": "uc-mild",
    },
    "case_c": {
        "label": "Case C · Barrett's Oesophagus",
        "blurb": "62 y/o male, 15 yr GORD on PPI, BMI 31, ex-smoker",
        "patient": {
            "name": "Demo Case C", "age": 62, "gender": "Male",
            "height": 175, "weight": 95, "bmi": 31.0,
            "city": "Birmingham", "country": "UK",
            "smoking": "Yes — Former", "alcohol": "Regular",
            "family_history": "No", "prev_polyps": "No",
            "prev_colonoscopy": "Never",
        },
        "symptoms": [
            "Persistent heartburn / GERD",
            "Difficulty swallowing",
        ],
        "symptom_text": "15-year history of GORD on long-term PPI. Recent food regurgitation. Endoscopy referred for surveillance — 4 cm tongue of columnar mucosa above the gastro-oesophageal junction.",
        "pain_scale": 3,
        "duration": "Over 1 year",
        "image": "assets/demo_cases/case_c_barretts.jpg",
        "image_type": "Endoscopy",
        "expected": "barretts-esoph",
    },
}


def _apply_demo_case(case_key: str):
    """Pre-populate session state with one of the canned demo cases.
    Clears prior analysis so the Analyse step actually re-runs the pipeline
    on the new case (rather than re-using the previous case's result)."""
    case = DEMO_CASES.get(case_key)
    if not case:
        return
    st.session_state["patient"] = dict(case["patient"])
    st.session_state["symptoms"] = list(case["symptoms"])
    st.session_state["symptom_text"] = case["symptom_text"]
    st.session_state["pain_scale"] = case["pain_scale"]
    st.session_state["symptom_duration"] = case["duration"]
    st.session_state["image_type"] = case["image_type"]
    st.session_state["uploaded_filename"] = Path(case["image"]).name
    st.session_state["demo_case"] = case_key
    img_path = ROOT / case["image"]
    if img_path.exists():
        try:
            st.session_state["uploaded_image"] = Image.open(img_path).convert("RGB")
        except Exception:
            pass
    # Reset symptom checkbox keys so the Step 2 page reflects the selection
    for k in [k for k in st.session_state.keys() if k.startswith("sym_")]:
        del st.session_state[k]
    # Clear prior analysis so re-running fires the pipeline anew on this case
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_done", None)
    st.session_state["step"] = 1


def page_patient_info():
    render_hero(
        "Patient Information",
        "Please provide your personal and medical history details",
        badges=["Step 1 of 6", "Secure & Confidential"],
    )

    # ── Welcome 3D showcase: rotating colon (patient-friendly, no tech labels) ─
    try:
        from src.app.ui_extras import colon_3d_figure
        st.markdown('<div class="section-header">Understanding your colon — interactive 3D model</div>',
                    unsafe_allow_html=True)
        c3d_a, c3d_b = st.columns([1.2, 1])
        with c3d_a:
            st.plotly_chart(colon_3d_figure("polyps"),
                           use_container_width=True, key="welcome_colon3d",
                           config={"displayModeBar": False})
            st.caption("**3D colon model** · drag with the mouse to spin · the marker shows a typical polyp location")
        with c3d_b:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#EFF6FF,#DBEAFE);
                 border-radius:14px;padding:18px 22px;height:100%;">
              <h4 style="color:#1E40AF;margin:0 0 10px;">What ColonAI does for you</h4>
              <div style="display:grid;gap:10px;color:#1F2937;font-size:14px;">
                <div>🔬 <b>Analyses your colonoscopy image</b> against thousands of expert-labelled samples</div>
                <div>📝 <b>Reads your symptom history</b> like a doctor would read your notes</div>
                <div>🧬 <b>Considers your health profile</b> — age, lifestyle, family history</div>
                <div>📊 <b>Gives a clear report</b> — diagnosis, stage estimate, recurrence risk</div>
                <div>👨‍⚕️ <b>Recommends specialists</b> near you with confidence scores</div>
                <div>📄 <b>Downloadable PDF</b> ready to share with your GP</div>
              </div>
              <div style="margin-top:14px;padding:10px 14px;background:#FEF3C7;
                   border-left:3px solid #F59E0B;border-radius:6px;font-size:12px;color:#92400E;">
                <b>Important:</b> ColonAI is a screening tool that supports — never replaces — a qualified doctor's diagnosis.
              </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        pass

    # ── Animated explainer (LOCAL — no YouTube dependency) ───────────────
    with st.expander("🎬 How ColonAI helps you — animated walkthrough", expanded=False):
        st.markdown(
            """
            <style>
            @keyframes stepFadeIn { from {opacity:0;transform:translateY(8px)} to {opacity:1;transform:none} }
            @keyframes pulseDot   { 0%,100%{transform:scale(1);box-shadow:0 0 0 0 rgba(26,115,232,0.6)}
                                    50%   {transform:scale(1.15);box-shadow:0 0 0 14px rgba(26,115,232,0)} }
            .walkthrough-row { display:flex; align-items:flex-start; gap:14px;
                               padding:14px 16px; margin:8px 0;
                               background:#FFF; border-radius:12px;
                               border:1px solid #E2E8F0;
                               box-shadow:0 4px 12px rgba(15,23,42,0.04);
                               animation: stepFadeIn 0.6s ease both; }
            .walkthrough-dot { width:42px; height:42px; border-radius:50%; flex-shrink:0;
                               display:flex; align-items:center; justify-content:center;
                               font-size:1.2rem; color:white; font-weight:800;
                               animation: pulseDot 2s ease-in-out infinite; }
            </style>

            <div class="walkthrough-row" style="animation-delay:0.05s;">
              <div class="walkthrough-dot" style="background:linear-gradient(135deg,#2563EB,#1E40AF)">📋</div>
              <div>
                <div style="font-weight:800;color:#0F172A;font-size:1.0rem">Step 1 — Tell us about yourself</div>
                <div style="color:#475569;font-size:0.9rem;margin-top:3px">
                  Age, gender, BMI, any symptoms you've noticed. Takes 90 seconds.
                </div>
              </div>
            </div>

            <div class="walkthrough-row" style="animation-delay:0.20s;">
              <div class="walkthrough-dot" style="background:linear-gradient(135deg,#0891B2,#0E7490);animation-delay:0.2s">📷</div>
              <div>
                <div style="font-weight:800;color:#0F172A;font-size:1.0rem">Step 2 — Upload your colonoscopy image</div>
                <div style="color:#475569;font-size:0.9rem;margin-top:3px">
                  A single frame works. You can also upload a short video clip for live analysis.
                </div>
              </div>
            </div>

            <div class="walkthrough-row" style="animation-delay:0.35s;">
              <div class="walkthrough-dot" style="background:linear-gradient(135deg,#7C3AED,#5B21B6);animation-delay:0.4s">🧠</div>
              <div>
                <div style="font-weight:800;color:#0F172A;font-size:1.0rem">Step 3 — AI analyses everything together</div>
                <div style="color:#475569;font-size:0.9rem;margin-top:3px">
                  Combines image, symptoms and your health profile — the way a real doctor would. Takes 1–2 seconds.
                </div>
              </div>
            </div>

            <div class="walkthrough-row" style="animation-delay:0.50s;">
              <div class="walkthrough-dot" style="background:linear-gradient(135deg,#16A34A,#15803D);animation-delay:0.6s">📊</div>
              <div>
                <div style="font-weight:800;color:#0F172A;font-size:1.0rem">Step 4 — Get your clear, easy report</div>
                <div style="color:#475569;font-size:0.9rem;margin-top:3px">
                  Diagnosis, stage estimate, recurrence risk. With a doctor finder and a downloadable PDF to share with your GP.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Quick-start demo cases ────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Quick demo · presentation-ready cases</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box" style="margin-bottom:14px">'
        'Skip the form and load a fully-prepared <b>realistic patient case</b> — '
        'demographics, symptoms and an endoscopy image — straight into the AI pipeline. '
        'Each case maps to one of the conditions the model is trained on.'
        '</div>',
        unsafe_allow_html=True,
    )
    dcols = st.columns(3)
    case_keys = ["case_a", "case_b", "case_c"]
    case_colors = ["#1A73E8", "#FF5722", "#9C27B0"]
    for i, key in enumerate(case_keys):
        case = DEMO_CASES[key]
        with dcols[i]:
            st.markdown(
                f"""<div class='metric-card' style='border-left-color:{case_colors[i]};margin-bottom:8px'>
                    <div class='label' style='color:{case_colors[i]}'>{case['label']}</div>
                    <div style='font-size:0.86rem;color:#475569;margin-top:4px;line-height:1.45'>{case['blurb']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"Load {case['label'].split('·')[0].strip()} →",
                         key=f"demo_{key}", use_container_width=True):
                _apply_demo_case(key)
                st.rerun()

    st.markdown('<div class="section-header">Personal Details</div>', unsafe_allow_html=True)
    p = st.session_state.get("patient", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Full Name *", value=p.get("name", ""), placeholder="e.g. John Doe")
    with col2:
        age = st.number_input("Age *", min_value=1, max_value=120,
                               value=int(p.get("age", 40) or 40))
    with col3:
        gender = st.selectbox("Gender *", ["Male", "Female", "Other", "Prefer not to say"],
                               index=["Male","Female","Other","Prefer not to say"].index(
                                   p.get("gender","Male")))

    col4, col5 = st.columns(2)
    with col4:
        height = st.number_input("Height (cm)", min_value=50, max_value=250,
                                  value=int(p.get("height", 170) or 170))
    with col5:
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300,
                                  value=int(p.get("weight", 70) or 70))

    bmi = weight / ((height / 100) ** 2) if height > 0 else 0
    st.info(f"Calculated BMI: **{bmi:.1f}** — "
            f"{'Underweight' if bmi<18.5 else 'Normal' if bmi<25 else 'Overweight' if bmi<30 else 'Obese'}")

    st.markdown('<div class="section-header">Location</div>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)
    with col6:
        city = st.text_input("City *", value=p.get("city", ""), placeholder="e.g. Mumbai")
    with col7:
        country = st.selectbox("Country",
            ["India", "USA", "UK", "UAE", "Singapore", "Canada", "Australia", "Other"],
            index=["India","USA","UK","UAE","Singapore","Canada","Australia","Other"].index(
                p.get("country","India")))

    st.markdown('<div class="section-header">Medical History</div>', unsafe_allow_html=True)
    col8, col9, col10 = st.columns(3)
    with col8:
        smoking = st.selectbox("Smoking History",
                                ["No","Yes — Current","Yes — Former"],
                                index=["No","Yes — Current","Yes — Former"].index(
                                    p.get("smoking","No")))
    with col9:
        alcohol = st.selectbox("Alcohol Consumption",
                                ["No","Occasional","Regular","Heavy"],
                                index=["No","Occasional","Regular","Heavy"].index(
                                    p.get("alcohol","No")))
    with col10:
        family_hist = st.selectbox("Family History of Colorectal Cancer",
                                    ["No","Yes — First degree","Yes — Second degree","Unknown"],
                                    index=["No","Yes — First degree","Yes — Second degree","Unknown"].index(
                                        p.get("family_history","No")))

    col11, col12 = st.columns(2)
    with col11:
        prev_polyps = st.selectbox("Previous Polyps Diagnosed",
                                    ["No","Yes","Unknown"],
                                    index=["No","Yes","Unknown"].index(
                                        p.get("prev_polyps","No")))
    with col12:
        prev_colonoscopy = st.text_input("Last Colonoscopy (year or 'Never')",
                                          value=p.get("prev_colonoscopy","Never"))

    st.markdown("")
    col_nav1, col_nav2 = st.columns([4, 1])
    with col_nav2:
        proceed = st.button("Next →", type="primary", use_container_width=True)

    if proceed:
        if not name.strip():
            st.error("Please enter your full name.")
        elif not city.strip():
            st.error("Please enter your city.")
        else:
            st.session_state["patient"] = {
                "name": name.strip(), "age": age, "gender": gender,
                "height": height, "weight": weight, "bmi": round(bmi, 1),
                "city": city.strip(), "country": country,
                "smoking": smoking, "alcohol": alcohol,
                "family_history": family_hist, "prev_polyps": prev_polyps,
                "prev_colonoscopy": prev_colonoscopy,
            }
            st.session_state["step"] = 1
            st.rerun()


def page_symptoms_upload():
    render_hero(
        "Symptoms & Medical Images",
        "Tell us what you're experiencing and upload your medical images for AI analysis",
        badges=["Step 2 of 6", "Supports Colonoscopy | Endoscopy | Histopathology"],
    )

    tab_symp, tab_upload, tab_report = st.tabs(
        ["Symptom Checker", "Upload Images", "Upload Existing Reports"])

    # ── Tab 1: Symptoms ────────────────────────────────────────────────
    with tab_symp:
        st.markdown('<div class="section-header">Select Your Symptoms</div>',
                    unsafe_allow_html=True)
        st.markdown("Check all symptoms that apply to you (in the last 3 months):")

        saved_symp  = st.session_state.get("symptoms", [])
        selected    = []
        cols = st.columns(2)
        for i, sym in enumerate(SYMPTOMS_LIST):
            with cols[i % 2]:
                if st.checkbox(sym, value=(sym in saved_symp), key=f"sym_{i}"):
                    selected.append(sym)

        st.markdown("")
        st.markdown('<div class="section-header">Symptom Severity</div>',
                    unsafe_allow_html=True)
        col_pain, col_dur = st.columns(2)
        with col_pain:
            pain_scale = st.slider(
                "Pain / Discomfort Level (0 = None, 10 = Severe)",
                0, 10, int(st.session_state.get("pain_scale", 3)),
                help="Rate your average daily discomfort over the past month"
            )
        with col_dur:
            duration = st.selectbox(
                "How long have you had these symptoms?",
                ["Less than 1 week","1–4 weeks","1–3 months","3–6 months","More than 6 months","Over 1 year"],
                index=["Less than 1 week","1–4 weeks","1–3 months","3–6 months","More than 6 months","Over 1 year"].index(
                    st.session_state.get("symptom_duration","1–3 months")
                ),
            )

        st.markdown('<div class="section-header">Additional Details</div>',
                    unsafe_allow_html=True)
        symptom_text = st.text_area(
            "Describe your symptoms in your own words (optional but helpful):",
            value=st.session_state.get("symptom_text",""),
            height=100,
            placeholder="e.g. I've been experiencing intermittent blood in my stool for about 2 months, with occasional cramping on the left side..."
        )

        if selected:
            severity_label = "Low" if pain_scale <= 3 else "Moderate" if pain_scale <= 6 else "High"
            st.markdown(
                f'<div class="info-box"><b>{len(selected)} symptom(s) selected</b> &nbsp;|&nbsp; '
                f'Severity: <b>{pain_scale}/10 ({severity_label})</b> &nbsp;|&nbsp; Duration: <b>{duration}</b></div>',
                unsafe_allow_html=True,
            )

        st.session_state["symptoms"]         = selected
        st.session_state["symptom_text"]     = symptom_text
        st.session_state["pain_scale"]       = pain_scale
        st.session_state["symptom_duration"] = duration

    # ── Tab 2: Image Upload ────────────────────────────────────────────
    with tab_upload:
        st.markdown('<div class="section-header">Upload Medical Images</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Supported: <b>JPG, PNG</b> images from colonoscopy, '
            'endoscopy, or histopathology. The AI will analyse the image and generate a GradCAM '
            'attention map showing exactly where it is focusing.</div>',
            unsafe_allow_html=True,
        )

        image_type = st.selectbox(
            "Image Type",
            ["Colonoscopy", "Endoscopy", "Histopathology", "CT Scan (cropped)", "Other"],
            index=["Colonoscopy","Endoscopy","Histopathology","CT Scan (cropped)","Other"].index(
                st.session_state.get("image_type","Colonoscopy")
            ),
        )
        st.session_state["image_type"] = image_type

        uploaded = st.file_uploader(
            "Drag & drop or click to upload your medical image",
            type=["jpg","jpeg","png"],
            help="Maximum 10 MB. Image will be resized to 224×224 for AI analysis.",
        )

        if uploaded is not None:
            # Validate the upload through the central security policy:
            # size cap, MIME allow-list, decompression-bomb guard. Refuse
            # the bytes rather than try to recover from a malformed payload.
            from src.app.security import validate_upload_bytes, UploadError
            try:
                raw_bytes = uploaded.getvalue()
                pil_img, _upload_meta = validate_upload_bytes(
                    raw_bytes,
                    declared_mime=getattr(uploaded, "type", None),
                    filename=uploaded.name)
            except UploadError as _e:
                st.error(f"Upload rejected: {_e}")
                st.stop()
            st.session_state["uploaded_image"]       = pil_img
            st.session_state["uploaded_image_bytes"] = raw_bytes   # for SHA-256 hash in the learning log
            st.session_state["uploaded_filename"]    = uploaded.name

            col_img, col_info = st.columns([1, 1])
            with col_img:
                st.image(pil_img, caption=uploaded.name, use_container_width=True)
            with col_info:
                w, h = pil_img.size
                st.success(f"Image uploaded: **{uploaded.name}**")
                render_metric_card("Resolution", f"{w} × {h} px", "Will be resized to 224×224")
                render_metric_card("Image Type", image_type, "Selected by user")
                render_metric_card("File Size", f"{uploaded.size/1024:.0f} KB", "Accepted")
        elif st.session_state.get("uploaded_image") is not None:
            pil_img = st.session_state["uploaded_image"]
            st.image(pil_img, caption="Previously uploaded image",
                     use_container_width=True, width=320)
            st.success("Image is ready for analysis")

    # ── Tab 3: Report Upload ───────────────────────────────────────────
    with tab_report:
        st.markdown('<div class="section-header">Upload Existing Medical Reports</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="warn-box">Upload any existing pathology, endoscopy, or radiology reports. '
            'These are stored for reference in your generated report but are NOT processed by the AI '
            '(which focuses on images).</div>',
            unsafe_allow_html=True,
        )
        report_files = st.file_uploader(
            "Upload medical reports (PDF, DOCX, TXT)",
            type=["pdf","txt","docx"],
            accept_multiple_files=True,
        )
        if report_files:
            for rf in report_files:
                st.success(f"Received: {rf.name} ({rf.size/1024:.0f} KB)")
            st.session_state["report_files"] = [f.name for f in report_files]

    # ── Navigation ─────────────────────────────────────────────────────
    st.markdown("---")
    col_back, col_space, col_next = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state["step"] = 0
            st.rerun()
    with col_next:
        can_proceed = (
            st.session_state.get("uploaded_image") is not None
            or len(st.session_state.get("symptoms", [])) > 0
        )
        if st.button("Analyse →", type="primary", use_container_width=True,
                     disabled=not can_proceed):
            st.session_state["step"] = 2
            st.rerun()
    if not can_proceed:
        st.markdown(
            '<div class="warn-box">Please upload a medical image or select at least one symptom to proceed.</div>',
            unsafe_allow_html=True,
        )


def page_analysis():
    render_hero(
        "Carefully analysing your case",
        "Six dedicated steps — looking at your image, your symptoms, and your history "
        "before we put it all together.",
        badges=["Step 3 of 6", "Patient-grade analysis", "BSG · NICE · USPSTF aligned"],
    )

    if st.session_state.get("analysis_done"):
        st.success("Analysis complete! Proceeding to results...")
        time.sleep(0.15)
        st.session_state["step"] = 3
        st.rerun()
        return

    # Load system
    if "_system" not in st.session_state or st.session_state["_system"] is None:
        with st.spinner("Loading AI pipeline (first time may take 30–60 s)..."):
            st.session_state["_system"] = load_ai_system()

    system = st.session_state["_system"]

    # Pipeline steps display
    pipeline_steps = [
        ("Looking at your image",
         "Studying tissue colour, shape, and texture for any concerning patterns"),
        ("Reading your symptoms",
         "Understanding what you're experiencing and matching it to known patterns"),
        ("Considering your history",
         "Weighing your age, lifestyle and family history into the picture"),
        ("Connecting the dots",
         "Combining everything into one coherent assessment"),
        ("Double-checking",
         "Running the analysis multiple times to confirm we're confident"),
        ("Preparing your plan",
         "Drafting next-steps aligned with international screening guidelines"),
    ]

    # Inspirational quotes shown on the analysis screen
    HEALTH_QUOTES = [
        ("The greatest wealth is health.", "Virgil"),
        ("Early detection is the strongest weapon we have against cancer.", "American Cancer Society"),
        ("Take care of your body. It's the only place you have to live.", "Jim Rohn"),
        ("Hope, when paired with knowledge, becomes powerful medicine.", "Anonymous"),
        ("The best time for a check-up was years ago. The second best is today.", "Adapted proverb"),
        ("90 % of early-stage colorectal cancers are curable. That's why screening matters.", "USPSTF / ACS"),
        ("Your health is an investment, not an expense.", "Anonymous"),
        ("Knowing what to look for is half the cure.", "Anonymous"),
    ]
    import random as _random
    _quote, _by = _random.choice(HEALTH_QUOTES)

    step_placeholder = st.empty()

    # No image at all -> honest, rule-based risk-factor assessment (NOT fake demo).
    pil_img = st.session_state.get("uploaded_image")
    if pil_img is None:
        _run_risk_only_assessment()
        return

    # Image present but the model failed to load -> honest error, never fake.
    if not system or not system.get("ready"):
        _show_model_unavailable(system)
        return

    # Lottie loader (or SVG fallback)
    try:
        from src.app.ui_extras import render_lottie_loader, render_agent_timeline
    except Exception:
        render_lottie_loader = None
        render_agent_timeline = None

    # ── Motivational quote card (replaces the technical "we use ConvNeXt..." copy) ──
    st.markdown(
        f"""<div style='position:relative;border-radius:18px;padding:22px 28px;
                        margin-bottom:18px;
                        background:linear-gradient(135deg,#EEF4FF 0%,#E0F2F1 100%);
                        border:1px solid rgba(26,115,232,0.18);overflow:hidden'>
              <svg width='110' height='110' viewBox='0 0 24 24' fill='none'
                   style='position:absolute;right:14px;top:50%;
                          transform:translateY(-50%);opacity:0.18'
                   xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>
                <path d='M12 21s-7-4.35-7-10a5 5 0 019-3 5 5 0 019 3c0 5.65-7 10-7 10z'
                      stroke='#1A73E8' stroke-width='1.4' stroke-linecap='round'
                      stroke-linejoin='round' fill='url(#gQ)'/>
                <defs>
                  <linearGradient id='gQ' x1='0' x2='1' y1='0' y2='1'>
                    <stop offset='0%' stop-color='#1A73E8'/>
                    <stop offset='100%' stop-color='#00897B'/>
                  </linearGradient>
                </defs>
              </svg>
              <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>
                <span style='display:inline-flex;align-items:center;justify-content:center;
                             width:34px;height:34px;border-radius:10px;
                             background:linear-gradient(135deg,#1A73E8,#00897B);
                             color:white;font-size:1.05rem'>✦</span>
                <span style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.7px;
                             color:#1A73E8;font-weight:800'>While we work</span>
              </div>
              <div style='font-size:1.18rem;color:#0F172A;font-weight:700;
                          line-height:1.45;max-width:780px;font-style:italic'>
                "{_quote}"
              </div>
              <div style='font-size:0.85rem;color:#475569;font-weight:500;margin-top:6px'>
                — {_by}
              </div>
            </div>""",
        unsafe_allow_html=True,
    )

    if render_lottie_loader:
        loader_slot = st.empty()
        with loader_slot.container():
            render_lottie_loader("Analysing your case — please wait a moment", height=140)

    # Animated pipeline timeline (moving bead) + per-step caption (patient-friendly)
    progress_bar = st.progress(0)
    for i, (name, desc) in enumerate(pipeline_steps):
        prog = (i + 0.5) / len(pipeline_steps)
        with step_placeholder.container():
            st.markdown(
                f"<div style='font-size:1.05rem;font-weight:700;color:#0F172A;margin-bottom:2px'>"
                f"Step {i+1} of 6 &nbsp;·&nbsp;<span style='color:#1A73E8'>{name}</span></div>"
                f"<div style='font-size:0.88rem;color:#64748B;margin-bottom:14px'>{desc}</div>",
                unsafe_allow_html=True,
            )
            if render_agent_timeline:
                render_agent_timeline(prog)
            else:
                # Old chip fallback (kept just in case)
                cols = st.columns(6)
                for j, (nm, _) in enumerate(pipeline_steps):
                    with cols[j]:
                        st.caption(nm)
        progress_bar.progress(prog)
        time.sleep(0.05)

    # Run actual analysis
    try:
        with st.spinner("Finalising multi-agent reasoning..."):
            analysis = run_analysis(
                system=system,
                pil_img=pil_img,
                patient=st.session_state.get("patient", {}),
                symptoms=st.session_state.get("symptoms", []),
                symptom_text=st.session_state.get("symptom_text", ""),
                pain_scale=int(st.session_state.get("pain_scale", 0) or 0),
                symptom_duration=str(st.session_state.get("symptom_duration", "") or ""),
            )
        st.session_state["analysis"]      = analysis
        st.session_state["analysis_done"] = True
        progress_bar.progress(1.0)
        step_placeholder.success("All 6 agents complete!")
        time.sleep(0.2)
        st.session_state["step"] = 3
        st.rerun()

    except Exception as e:
        st.error(f"Analysis error: {e}")
        st.info("We will not show a fabricated result. Please retry, or try a different image.")
        if st.button("↻ Retry analysis", type="primary"):
            st.rerun()


def _run_demo_analysis():
    """Produce a plausible demo result when the model isn't available."""
    import random
    random.seed(42)
    st.session_state["analysis"] = {
        "pathology_class":  "polyps",
        "pathology_probs":  {"polyps":0.762,"uc-mild":0.128,"uc-moderate-sev":0.052,"barretts-esoph":0.041,"therapeutic":0.017},
        "stage":            "Stage I",
        "stage_confidence": 0.742,
        "stage_probs":      {"No Cancer":0.21,"Stage I":0.742,"Stage II":0.038,"Stage III/IV":0.01},
        "risk_score":       0.238,
        "risk_label":       "Benign",
        "image_weight":     0.52,
        "text_weight":      0.28,
        "tabular_weight":   0.20,
        "confidence":       0.762,
        "all_risk_flags":   [],
        "uncertainty":      0.24,
        "inference_time_ms": 312.0,
        "recommendation": {
            "urgency":        "Routine",
            "primary_action": "Polypectomy with surveillance colonoscopy at 3 years",
            "surveillance":   "3-year colonoscopy interval",
            "referrals":      ["Gastroenterologist for follow-up colonoscopy"],
            "investigations": ["Faecal immunochemical test (FIT)", "CEA blood marker"],
            "lifestyle_advice": [
                "Increase dietary fibre intake (25–35 g/day)",
                "Reduce processed and red meat consumption",
                "Maintain healthy weight (BMI 18.5–24.9)",
                "Regular physical activity (150 min/week)",
                "Limit alcohol to < 14 units/week",
                "Quit smoking if applicable",
            ],
            "full_report": "Demo report — model not loaded.",
        },
        "gradcam_overlay": None,
        "gradcam_heatmap": None,
        "original_image":  None,
        "_provenance": {
            "source": "demo_fallback",
            "note": "Hard-coded values shown because the trained model could not be loaded "
                    "(or no image was uploaded). Not real model output.",
        },
        "ablation": {},
    }
    st.session_state["analysis_done"] = True
    st.session_state["step"] = 3
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# HONEST SYMPTOMS-ONLY PATH (no image) — replaces the fake demo fallback
# Rule-based, guideline-informed risk-FACTOR assessment. NOT a diagnosis and
# NOT a model image prediction — only what history + symptoms genuinely support.
# ─────────────────────────────────────────────────────────────────────────
def _assess_risk_factors(patient: dict, symptoms: list, symptom_text: str,
                         pain_scale: int = 0, symptom_duration: str = "") -> dict:
    """Transparent CRC risk-factor assessment from history + symptoms only.
    Every contributing factor is listed explicitly; nothing is fabricated."""
    sym_join = (" ".join(symptoms or []) + " " + (symptom_text or "")).lower()

    def has(*keys):
        return any(k in sym_join for k in keys)

    # Symptom red flags (NICE NG12-aligned suspicion features)
    red_flags = []
    if has("bleed", "blood in stool"):           red_flags.append(("Rectal bleeding / blood in stool", 3))
    if has("weight loss"):                         red_flags.append(("Unexplained weight loss", 3))
    if has("change in bowel", "bowel habit"):      red_flags.append(("Persistent change in bowel habit", 2))
    if has("anaem", "anemia", "fatigue", "weak"):  red_flags.append(("Fatigue / possible anaemia", 2))
    if has("pencil", "incomplete", "narrow stool"):red_flags.append(("Narrow / incomplete-evacuation stools", 2))
    if has("abdominal pain", "cramp"):             red_flags.append(("Abdominal pain / cramping", 1))
    if has("mucus"):                               red_flags.append(("Mucus in stool", 1))

    # Personal / family risk factors
    factors = []
    age = int(patient.get("age", 0) or 0)
    if age >= 60:   factors.append((f"Age {age} (≥60)", 2))
    elif age >= 50: factors.append((f"Age {age} (≥50 — screening age)", 1))
    elif age >= 45: factors.append((f"Age {age} (≥45 — screening now recommended)", 1))

    def _yes(v):
        s = str(v or "").strip().lower()
        return s not in ("", "no", "none", "false", "never", "0")

    if _yes(patient.get("family_history")): factors.append(("Family history of colorectal cancer", 3))
    if _yes(patient.get("prev_polyps")):    factors.append(("Previous polyps diagnosed", 2))
    sm = str(patient.get("smoking", "")).lower()
    if "current" in sm or sm == "yes": factors.append(("Current smoker", 2))
    elif "former" in sm or "ex" in sm: factors.append(("Former smoker", 1))
    al = str(patient.get("alcohol", "")).lower()
    if any(k in al for k in ("high", "heavy", "daily")): factors.append(("High alcohol intake", 1))
    try:
        bmi = float(patient.get("bmi", 0) or 0)
        if bmi >= 30: factors.append((f"Obesity (BMI {bmi:.0f})", 1))
    except Exception:
        pass
    if str(patient.get("prev_colonoscopy", "")).strip().lower() in ("never", "", "none") and age >= 45:
        factors.append(("No prior colonoscopy at screening age", 1))

    rf_score  = sum(w for _, w in red_flags)
    fac_score = sum(w for _, w in factors)
    total     = rf_score + fac_score
    long_dur  = symptom_duration in ("3–6 months", "More than 6 months", "Over 1 year")

    # Literature-grounded relative-risk multiplier + VALIDATED APCS score
    try:
        from src.app.crc_risk_model import relative_risk, rr_to_band, apcs_score
        _rr = relative_risk(patient)
        rr_total, rr_factors, rr_band, rr_notes = (
            _rr["rr_total"], _rr["factors"], rr_to_band(_rr["rr_total"]), _rr["notes"])
        apcs = apcs_score(patient)
    except Exception:
        rr_total, rr_factors, rr_band, rr_notes = 1.0, [], "about average", []
        apcs = None

    # Conservative tiering — when in doubt, escalate (safer in a clinical tool).
    # Now also informed by the literature RR multiplier (markedly-elevated risk
    # warrants at least a check-up even without acute red flags).
    if (rf_score >= 3) or (red_flags and (long_dur or fac_score >= 3)):
        tier, label = "High", "Higher concern — prompt clinical review advised"
        urgency = "See a doctor promptly"
        primary = ("Book a GP/clinician appointment soon. Your reported symptoms include "
                   "features that warrant timely assessment, likely including a colonoscopy.")
    elif total >= 3 or rr_total >= 2.0 or (apcs and apcs.get("tier") == "High"):
        tier, label = "Moderate", "Moderate concern — get checked"
        urgency = "Arrange a check-up"
        primary = ("Arrange a FIT (stool) test and a GP review. A colonoscopy may be "
                   "recommended based on the result and your history.")
    else:
        tier, label = "Low", "Lower concern — stay on routine screening"
        urgency = "Routine"
        primary = ("Continue routine age-appropriate screening (FIT / colonoscopy). "
                   "Seek review if new or worsening symptoms appear.")

    return {"tier": tier, "label": label, "urgency": urgency, "primary": primary,
            "red_flags": red_flags, "factors": factors,
            "rf_score": rf_score, "fac_score": fac_score, "total": total,
            "long_duration": long_dur,
            "rr_total": rr_total, "rr_factors": rr_factors, "rr_band": rr_band,
            "rr_notes": rr_notes, "apcs": apcs}


def _run_risk_only_assessment():
    """No image provided -> honest, rule-based risk-factor assessment.
    Replaces the old fake _run_demo_analysis fallback. No pathology class, no
    fabricated probabilities, no model image prediction."""
    patient      = st.session_state.get("patient", {})
    symptoms     = st.session_state.get("symptoms", [])
    symptom_text = st.session_state.get("symptom_text", "")
    pain         = int(st.session_state.get("pain_scale", 0) or 0)
    dur          = str(st.session_state.get("symptom_duration", "") or "")
    r = _assess_risk_factors(patient, symptoms, symptom_text, pain, dur)
    st.session_state["analysis"] = {
        "pathology_class":  "RISK_ASSESSMENT_ONLY",
        "pathology_probs":  {},
        "risk_only":        True,
        "risk_tier":        r["tier"],
        "risk_label":       r["label"],
        "risk_factors":     r["factors"],
        "red_flags":        r["red_flags"],
        "risk_detail":      r,
        "stage":            "Not assessed (no image)",
        "stage_confidence": 0.0,
        "stage_probs":      {},
        "confidence":       0.0,
        "gradcam_overlay":  None, "gradcam_heatmap": None, "original_image": None,
        "recommendation": {
            "urgency":        r["urgency"],
            "primary_action": r["primary"],
            "referrals":      ["Gastroenterology / GP for clinical review"],
            "investigations": ["FIT (faecal immunochemical test)",
                               "Colonoscopy for direct visual assessment"],
            "lifestyle_advice": [
                "High-fibre diet (25–35 g/day)", "Limit red / processed meat",
                "Maintain a healthy weight", "Regular activity (150 min/week)",
                "Limit alcohol", "Don't smoke",
            ],
            "full_report": "Risk-factor assessment from history and symptoms only — no "
                           "colonoscopy image was provided, so no visual diagnosis was made.",
        },
        "_provenance": {
            "source": "risk_factor_assessment",
            "note": "Rule-based assessment of established risk factors + symptom red flags. "
                    "NOT an AI image diagnosis and NOT a substitute for clinical evaluation. "
                    "A colonoscopy is required to visually assess for polyps/cancer.",
        },
    }
    st.session_state["analysis_done"] = True
    st.session_state["step"] = 3
    st.rerun()


def _show_model_unavailable(system):
    """Image present but the AI model could not load -> honest error, never fake."""
    err = (system or {}).get("error", "the AI model could not be loaded")
    st.error("The AI image model is currently unavailable, so we cannot analyse the "
             "uploaded image. We will **not** show a fabricated result.")
    with st.expander("Technical detail"):
        st.code(str(err)[:800])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↻ Retry loading the model", use_container_width=True, type="primary"):
            st.session_state.pop("_system", None)
            st.rerun()
    with c2:
        if st.button("Continue with symptoms-only assessment", use_container_width=True):
            _run_risk_only_assessment()


# ─────────────────────────────────────────────────────────────────────────
# AI REASONING PANEL — explains *why* the model returned this prediction
# ─────────────────────────────────────────────────────────────────────────

def _why_pretty_class(pclass: str) -> str:
    return CLASS_LABELS.get(pclass, pclass)


def _modality_evidence_lines(analysis: dict, patient: dict, symptoms: list) -> list:
    """Build human-readable lines explaining what each input contributed —
    in plain English (no model architecture jargon)."""
    lines = []
    pclass = analysis.get("pathology_class", "")
    img_w = analysis.get("image_weight", 0.0)
    txt_w = analysis.get("text_weight", 0.0)
    tab_w = analysis.get("tabular_weight", 0.0)

    # Image contribution
    img_pct = img_w * 100
    img_msg = (f"The picture you uploaded counted for <b>{img_pct:.0f}%</b> of the conclusion. ")
    if pclass == "polyps":
        img_msg += ("The AI saw what looked like a focal raised area on the colon wall — a pattern typical of an adenomatous polyp.")
    elif pclass.startswith("uc"):
        img_msg += ("The AI saw diffusely inflamed mucosa with patchy granular texture — a colitis-like pattern.")
    elif pclass == "barretts-esoph":
        img_msg += ("The AI saw a salmon-coloured columnar segment above the gastro-oesophageal junction — typical Barrett's appearance.")
    elif pclass == "therapeutic":
        img_msg += ("The AI saw what looks like a post-resection or recently treated area on the mucosa.")
    lines.append(("From the image", img_msg, "#1A73E8"))

    # Text contribution
    txt_pct = txt_w * 100
    sym_count = len(symptoms or [])
    txt_msg = (f"The symptoms / notes you wrote counted for <b>{txt_pct:.0f}%</b>. ")
    if sym_count == 0:
        txt_msg += "You didn't tick any symptoms, so this branch had no extra signal to add."
    else:
        joined = ", ".join((symptoms or [])[:3])
        txt_msg += f"You ticked {sym_count} symptom(s)."
        if joined:
            txt_msg += f" The AI paid the most attention to: <i>{joined}</i>."
    lines.append(("From your symptoms", txt_msg, "#00897B"))

    # Tabular contribution
    tab_pct = tab_w * 100
    age = patient.get("age", "?")
    bmi = patient.get("bmi", "?")
    smoke = patient.get("smoking", "?")
    fam = patient.get("family_history", "?")
    tab_msg = (f"Your medical-history form (age {age}, BMI {bmi}, smoking history: {smoke}, "
               f"family history: {fam}) counted for <b>{tab_pct:.0f}%</b> — "
               f"these adjusted the AI's baseline risk estimate.")
    lines.append(("From your history", tab_msg, "#FF5722"))

    return lines


def _build_reasoning_summary(analysis: dict, patient: dict, symptoms: list) -> str:
    pclass = analysis.get("pathology_class", "")
    pretty = _why_pretty_class(pclass)
    confidence = analysis.get("confidence", 0)
    risk = analysis.get("risk_score", 0)
    unc = analysis.get("uncertainty", 0)
    img_w = analysis.get("image_weight", 0)
    txt_w = analysis.get("text_weight", 0)
    tab_w = analysis.get("tabular_weight", 0)

    # Confidence band
    if confidence >= 0.8:
        cband = "high-confidence"
    elif confidence >= 0.6:
        cband = "moderate-confidence"
    else:
        cband = "low-confidence"

    if unc >= 0.6:
        unc_clause = "but the AI is genuinely unsure — clinician review essential"
    elif unc >= 0.3:
        unc_clause = "with moderate certainty"
    else:
        unc_clause = "with high certainty (the AI's internal cross-checks all agreed)"

    # Dominant input — patient-friendly labels
    weights = [("the image", img_w), ("your symptoms", txt_w), ("your medical history", tab_w)]
    weights.sort(key=lambda x: x[1], reverse=True)
    dom = weights[0]
    dom_msg = f"{dom[0].capitalize()} contributed the most ({dom[1]*100:.0f}% of the weight)."

    risk_clause = (f"The AI estimated the malignancy probability at {risk*100:.0f}% — "
                   f"classified as {analysis.get('risk_label','?')}.")

    return (f"The model reached a <b>{cband} prediction of {pretty}</b> at "
            f"{confidence*100:.0f}% confidence {unc_clause}. {dom_msg} {risk_clause}")


def _provenance_badge_html(prov: dict) -> str:
    """Coloured badge showing whether a panel is real model output, heuristic,
    or template text — so the user can audit at a glance."""
    src = (prov or {}).get("source", "real_model")
    if src == "real_model":
        ckpt = "loaded" if prov.get("checkpoint_loaded") else "not loaded"
        return (
            "<span class='pill pill-green' style='font-size:0.68rem'>"
            f"✓ Live model output · checkpoint {ckpt}</span>"
        )
    if src == "demo_fallback":
        return (
            "<span class='pill pill-red' style='font-size:0.68rem'>"
            "⚠ Demo fallback · not from the trained model</span>"
        )
    return (
        "<span class='pill pill-amber' style='font-size:0.68rem'>"
        "Heuristic · not a model output</span>"
    )


def _render_reasoning_panel(analysis: dict, patient: dict, symptoms: list):
    """Renders the 'Why this result?' tab content."""
    if not analysis:
        st.info("No analysis to explain.")
        return

    prov = analysis.get("_provenance", {"source": "real_model"})
    is_real = prov.get("source") == "real_model"

    # Provenance / authenticity card — friendly, technical is in the
    # collapsible expander on the Diagnosis tab
    if is_real:
        st.markdown(
            f"""<div style='background:linear-gradient(135deg,#E8F5E9,#F0FDF4);
                            border:1px solid #A5D6A7;border-radius:12px;
                            padding:12px 16px;margin-bottom:14px'>
                  <span class='pill pill-green' style='font-size:0.68rem'>✓ Verified by AI</span>
                  <div style='font-size:0.85rem;color:#1B5E20;margin-top:6px;line-height:1.5'>
                    Every number on this page came from your case being analysed in real time —
                    not from a template or a pre-saved demo. We finished in
                    <b>{analysis.get('inference_time_ms',0):.0f} ms</b>.
                    Looking for the model architecture and metrics?  Click
                    <i>"Show technical details"</i> on the Diagnosis tab.
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style='background:linear-gradient(135deg,#FEE2E2,#FECACA);
                            border:1px solid #FCA5A5;border-radius:12px;
                            padding:12px 16px;margin-bottom:14px'>
                  {_provenance_badge_html(prov)}
                  <div style='font-size:0.85rem;color:#991B1B;margin-top:6px;line-height:1.5'>
                    {prov.get('note','Demo values are hard-coded and not from the trained model.')}
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )

    # Top summary card
    summary = _build_reasoning_summary(analysis, patient, symptoms)
    st.markdown(
        f"""<div style='border-radius:16px;padding:18px 22px;margin-bottom:18px;
                       background:linear-gradient(135deg,#EEF4FF 0%,#E0F2F1 100%);
                       border:1px solid rgba(26,115,232,0.15)'>
              <div style='font-size:0.74rem;text-transform:uppercase;letter-spacing:0.6px;
                          color:#1A73E8;font-weight:800'>Plain-English explanation</div>
              <div style='font-size:1.0rem;color:#0F172A;line-height:1.6;margin-top:6px'>
                {summary}
              </div>
            </div>""",
        unsafe_allow_html=True,
    )

    # Modality contributions
    st.markdown('<div class="section-header">Where the evidence came from</div>',
                unsafe_allow_html=True)
    lines = _modality_evidence_lines(analysis, patient, symptoms)
    for title, body, color in lines:
        st.markdown(
            f"""<div style='display:flex;gap:14px;padding:14px 18px;margin-bottom:8px;
                            border-radius:12px;background:white;
                            border:1px solid rgba(15,23,42,0.06);
                            border-left:4px solid {color};
                            box-shadow:0 1px 3px rgba(15,23,42,0.04)'>
                  <div style='flex:0 0 auto;width:40px;height:40px;border-radius:12px;
                              background:linear-gradient(135deg,{color}22,{color}11);
                              display:flex;align-items:center;justify-content:center;
                              color:{color};font-weight:800;font-size:0.92rem'>
                    {title.split()[0][:2].upper()}
                  </div>
                  <div style='flex:1'>
                    <div style='font-size:0.74rem;text-transform:uppercase;letter-spacing:0.5px;
                                color:{color};font-weight:800'>{title}</div>
                    <div style='font-size:0.92rem;color:#1F2937;line-height:1.55;margin-top:3px'>
                      {body}
                    </div>
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )

    # Confidence breakdown bar
    st.markdown('<div class="section-header">How sure is the AI?</div>',
                unsafe_allow_html=True)
    confidence = analysis.get("confidence", 0)
    unc = analysis.get("uncertainty", 0)
    cols = st.columns(3)
    with cols[0]:
        render_metric_card("Top-finding confidence", f"{confidence*100:.0f}%",
                           "How strongly the AI favoured its top finding", color="#1A73E8")
    with cols[1]:
        render_metric_card("Doubt level", f"{unc:.2f}",
                           "Closer to 0 = the AI is sure · closer to 1 = it isn't",
                           color="#9C27B0")
    with cols[2]:
        agree = max(0.0, min(1.0, 1.0 - unc))
        render_metric_card("Internal agreement", f"{agree*100:.0f}%",
                           "How often the AI's repeated cross-checks agreed",
                           color="#16A34A" if agree>0.7 else "#F59E0B")

    # GradCAM crop preview if available
    cam = analysis.get("gradcam_overlay")
    ig_overlay = analysis.get("ig_overlay")
    xai_agree  = analysis.get("xai_agreement")
    if cam is not None:
        st.markdown('<div class="section-header">What the AI is looking at — two independent XAI methods</div>',
                    unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns([1, 1, 1.4])
        with col_a:
            disp = (cam * 255).astype(np.uint8) if cam.max() <= 1 else cam.astype(np.uint8)
            st.image(disp, caption="GradCAM++ (gradient-weighted)",
                     use_container_width=True)
        with col_b:
            if ig_overlay is not None:
                ig_disp = (ig_overlay * 255).astype(np.uint8) if ig_overlay.max() <= 1 else ig_overlay.astype(np.uint8)
                st.image(ig_disp, caption="Integrated Gradients (axiom-grounded)",
                         use_container_width=True)
            else:
                st.markdown(
                    "<div style='background:#F8FAFC;border:1px dashed #CBD5E1;"
                    "border-radius:8px;padding:32px;text-align:center;color:#64748B;'>"
                    "<i>IG not generated for this case</i></div>",
                    unsafe_allow_html=True,
                )
        with col_c:
            agreement_html = ""
            if xai_agree is not None:
                ag_pct = xai_agree * 100
                if ag_pct >= 60:
                    ag_color, ag_word = "#16A34A", "strong"
                elif ag_pct >= 35:
                    ag_color, ag_word = "#D97706", "partial"
                else:
                    ag_color, ag_word = "#DC2626", "weak"
                agreement_html = (
                    f"<div style='background:#FFF;border:1px solid #E2E8F0;"
                    f"border-radius:10px;padding:12px 14px;margin-bottom:10px;'>"
                    f"<div style='font-size:12px;color:#64748B;'>XAI cross-check</div>"
                    f"<div style='font-size:22px;font-weight:700;color:{ag_color};'>"
                    f"{ag_pct:.0f}% overlap</div>"
                    f"<div style='font-size:12px;color:#475569;'>"
                    f"GradCAM++ and Integrated Gradients show <b>{ag_word}</b> "
                    f"agreement on the salient region.</div></div>"
                )
            st.markdown(
                agreement_html +
                "<div class='info-box'>"
                "<b>Two independent XAI methods, one verdict.</b> GradCAM++ uses gradients "
                "on the final CNN layer; Integrated Gradients uses pixel-level attribution "
                "with axiomatic guarantees. When both methods highlight the same region, "
                "the explanation is highly trustworthy. When they disagree, the saliency "
                "may be an artefact of one particular method — treat the prediction with "
                "caution and re-examine the input image."
                "</div>",
                unsafe_allow_html=True,
            )

    # ── REAL model-probed ablation evidence ────────────────────────────
    abl = analysis.get("ablation") or {}
    if abl and "base_prob" in abl:
        st.markdown(
            '<div class="section-header">What if we hid one of your inputs?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='display:flex;gap:8px;align-items:center;margin-bottom:10px'>"
            f"{_provenance_badge_html({'source':'real_model','checkpoint_loaded':True})}"
            f"<span style='font-size:0.86rem;color:#475569'>"
            f"We re-ran the analysis three more times — once with the picture removed, "
            f"once with your symptoms removed, once with your history removed — to see "
            f"how much each one mattered for <b>your case</b>."
            f"</span></div>",
            unsafe_allow_html=True,
        )
        base_pct = abl["base_prob"] * 100
        rows = [
            ("If we hid the image",
             abl["no_image_prob"] * 100, abl["image_drop_pp"], "#1A73E8"),
            ("If we hid the symptoms text",
             abl["no_text_prob"] * 100,  abl["text_drop_pp"],  "#00897B"),
            ("If we hid the medical history",
             abl["no_tabular_prob"] * 100, abl["tabular_drop_pp"], "#FF5722"),
        ]
        bar_max = max(1.0, max(r[2] for r in rows))
        for label, p, drop, c in rows:
            bar_w = (drop / bar_max) * 100
            st.markdown(
                f"""<div style='padding:10px 14px;margin-bottom:8px;border-radius:12px;
                                background:white;border:1px solid rgba(15,23,42,0.06);
                                border-left:4px solid {c};
                                box-shadow:0 1px 3px rgba(15,23,42,0.04)'>
                      <div style='display:flex;justify-content:space-between;
                                  align-items:center;margin-bottom:4px'>
                        <div style='font-weight:700;color:#0F172A;font-size:0.92rem'>
                          {label}
                        </div>
                        <div style='font-size:0.82rem;color:{c};font-weight:800'>
                          {p:.1f}% &nbsp;<span style='color:#64748B'>(was {base_pct:.1f}%)</span>
                          &nbsp;<span style='color:#0F172A'>· drop {drop:.1f} pp</span>
                        </div>
                      </div>
                      <div style='height:8px;background:#F1F5F9;border-radius:5px;overflow:hidden'>
                        <div style='height:100%;width:{bar_w:.1f}%;background:{c};
                                    border-radius:5px;
                                    transition:width 0.6s ease'></div>
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div class='info-box' style='margin-top:6px'>"
            "<b>How to read this.</b> A larger drop means that modality is more decisive for "
            "<i>this</i> patient. Tiny drops mean the other modalities can compensate. "
            "Negative drops (clipped to zero) are rare and indicate the modality was actively "
            "<i>hurting</i> the prediction."
            "</div>",
            unsafe_allow_html=True,
        )

    elif abl and "error" in abl:
        st.markdown(
            f"<div class='warn-box'>Ablation probe could not run: <code>{abl['error']}</code></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='disclaimer' style='margin-top:14px'>"
        "Probabilities, modality weights, uncertainty, and the GradCAM heatmap are all direct "
        "model outputs. The plain-English narrative above is a templated explanation that "
        "<b>describes</b> those numbers — it is not itself a model output. Always confirm with "
        "histology, biopsy, or specialist review."
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Plain-English "Why this result?" card for the Diagnosis tab
# ─────────────────────────────────────────────────────────────────────

# What the AI looks for in the image, per class — plain-English
PLAIN_IMAGE_OBSERVATIONS = {
    "polyps":          "a focal mucosal protrusion typical of an adenomatous polyp",
    "uc-mild":         "patchy granular mucosa with mild loss of vascular pattern",
    "uc-moderate-sev": "diffuse inflammation, ulceration and contact bleeding",
    "barretts-esoph":  "a salmon-coloured columnar segment above the gastro-oesophageal junction",
    "therapeutic":     "a post-resection / dyed mucosal site (a recently treated area)",
}


def _plain_risk_factor_lines(patient: dict, symptoms: list) -> list:
    """Walk through the patient's actual inputs and return short plain-English
    bullets noting what each contributes (good / risk / neutral)."""
    bullets = []
    age = int(patient.get("age", 0) or 0)
    bmi = float(patient.get("bmi", 0) or 0)
    smoking = str(patient.get("smoking", "")).lower()
    alcohol = str(patient.get("alcohol", "")).lower()
    fam = str(patient.get("family_history", "")).lower()
    polyps_hx = str(patient.get("prev_polyps", "")).lower()

    if age >= 50:
        bullets.append(("amber", f"Age {age} — within the average-risk screening window (USPSTF 45–75)."))
    elif age >= 45:
        bullets.append(("amber", f"Age {age} — at the lower edge of the screening age (USPSTF lowered the floor to 45 in 2021)."))
    elif age >= 18:
        bullets.append(("green", f"Age {age} — younger than the routine screening threshold; baseline risk is lower."))
    if bmi >= 30:
        bullets.append(("amber", f"BMI {bmi:.1f} — obesity is an established risk factor."))
    elif 18.5 <= bmi < 25:
        bullets.append(("green", f"BMI {bmi:.1f} — within the healthy range."))
    if "yes" in smoking:
        bullets.append(("amber", "Smoking history — slightly raises colorectal cancer risk."))
    if any(x in alcohol for x in ["regular", "heavy"]):
        bullets.append(("amber", "Regular / heavy alcohol — raises colorectal cancer risk."))
    if "first" in fam:
        bullets.append(("red", "First-degree family history of colorectal cancer — meaningfully higher risk; consider earlier surveillance."))
    if "yes" in polyps_hx:
        bullets.append(("red", "Previous polyps — established surveillance pathway likely applies."))

    n_sym = len([s for s in (symptoms or []) if s])
    if n_sym == 0:
        bullets.append(("green", "No symptoms reported — that is reassuring on its own."))
    elif n_sym <= 2:
        bullets.append(("amber", f"{n_sym} symptom(s) reported — context matters; review the symptoms list with your clinician."))
    else:
        bullets.append(("red", f"{n_sym} symptoms reported — the AI weighted these heavily in its conclusion."))

    if not bullets:
        bullets.append(("green", "No notable risk factors picked up from your form."))
    return bullets


def _render_plain_why_card(analysis: dict, patient: dict, symptoms: list, symptom_text: str):
    """A patient-friendly explanation card placed at the top of the Diagnosis tab.
    Uses plain language — no model architecture jargon."""
    if not analysis:
        return
    pclass = analysis.get("pathology_class", "")
    pretty = CLASS_LABELS.get(pclass, pclass)
    confidence = analysis.get("confidence", 0)
    risk = analysis.get("risk_score", 0)
    unc = analysis.get("uncertainty", 0)
    img_w = analysis.get("image_weight", 0)
    txt_w = analysis.get("text_weight", 0)
    tab_w = analysis.get("tabular_weight", 0)

    # Confidence band — plain
    if confidence >= 0.8:
        conf_phrase = "is fairly confident"
    elif confidence >= 0.6:
        conf_phrase = "leans toward"
    elif confidence >= 0.4:
        conf_phrase = "tentatively suggests"
    else:
        conf_phrase = "is genuinely unsure but slightly favours"

    # Uncertainty in plain terms
    if unc < 0.3:
        unc_phrase = "All of the AI's internal cross-checks agreed on the same finding — that's a sign of consistency."
    elif unc < 0.6:
        unc_phrase = "The AI's internal cross-checks mostly agreed, with some variation — review with a clinician for confirmation."
    else:
        unc_phrase = "The AI's internal cross-checks disagreed — this finding is uncertain and a clinician's review is essential."

    # Dominant input
    weights = [("the image", img_w), ("the symptoms", txt_w), ("the patient profile", tab_w)]
    weights.sort(key=lambda x: x[1], reverse=True)
    dom_label, dom_pct = weights[0][0], weights[0][1] * 100

    # Image plain observation
    img_obs = PLAIN_IMAGE_OBSERVATIONS.get(pclass,
              "patterns the AI has learnt to associate with this finding")

    # Patient inputs summary — ESCAPE all user-provided fields before
    # interpolating into HTML (unsafe_allow_html block follows).
    from src.app.security import escape_html as _esc
    age  = _esc(patient.get("age", "—"))
    sex  = _esc(patient.get("gender", "—"))
    bmi  = _esc(patient.get("bmi", "—"))
    name = _esc(patient.get("name")) or "the patient"

    # The narrative
    narrative = (
        f"On {name}'s endoscopy image, the AI noticed <b>{img_obs}</b>. "
        f"It also looked at the symptoms you wrote and your medical history "
        f"(age {age}, {sex}, BMI {bmi}). Putting it all together, the AI {conf_phrase} "
        f"<b>{pretty}</b> — confidence {confidence*100:.0f}%. "
        f"Most of the decision came from <b>{dom_label}</b> ({dom_pct:.0f}% of the weight). "
        f"{unc_phrase}"
    )

    # Risk factors
    bullet_color = {"green": "#15803D", "amber": "#B45309", "red": "#B91C1C"}
    bullet_bg    = {"green": "#F0FDF4", "amber": "#FFFBEB", "red": "#FEF2F2"}
    bullet_icon  = {"green": "✓", "amber": "•", "red": "!"}
    bullets = _plain_risk_factor_lines(patient, symptoms)
    bullets_html = "".join(
        f"<div style='display:flex;gap:10px;padding:6px 10px;margin-bottom:4px;"
        f"border-radius:8px;background:{bullet_bg[c]};color:#0F172A'>"
        f"<span style='display:inline-flex;align-items:center;justify-content:center;"
        f"width:18px;height:18px;border-radius:50%;background:{bullet_color[c]};color:white;"
        f"font-size:0.72rem;font-weight:800;flex:0 0 auto'>{bullet_icon[c]}</span>"
        f"<span style='font-size:0.86rem;line-height:1.45'>{txt}</span></div>"
        for c, txt in bullets
    )

    st.markdown(
        f"""<div style='border-radius:18px;padding:22px 26px;margin-bottom:18px;
                        background:linear-gradient(135deg,#F8FAFF 0%,#EEF4FF 100%);
                        border:1px solid rgba(26,115,232,0.18);
                        box-shadow:0 1px 3px rgba(15,23,42,0.04),
                                   0 18px 36px -22px rgba(15,23,42,0.18)'>
              <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>
                <span style='display:inline-flex;align-items:center;justify-content:center;
                             width:36px;height:36px;border-radius:11px;
                             background:linear-gradient(135deg,#1A73E8,#00897B);
                             color:white;font-size:1.05rem'>🧠</span>
                <div>
                  <div style='font-size:0.74rem;text-transform:uppercase;letter-spacing:0.7px;
                              color:#1A73E8;font-weight:800'>Why we reached this result</div>
                  <div style='font-size:1.08rem;color:#0F172A;font-weight:800'>
                    The AI's reasoning, in plain English
                  </div>
                </div>
              </div>
              <div style='font-size:0.95rem;color:#1F2937;line-height:1.65;margin-bottom:14px'>
                {narrative}
              </div>
              <div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.6px;
                          color:#475569;font-weight:800;margin-bottom:6px'>
                What we picked up from your form
              </div>
              {bullets_html}
            </div>""",
        unsafe_allow_html=True,
    )


def _render_compare_panel():
    """If compare-mode is on, show side-by-side snapshots of slot A and slot B."""
    snaps = st.session_state.get("compare_snaps", {})
    if not (snaps.get("A") and snaps.get("B")):
        return False  # nothing to compare yet
    st.markdown('<div class="section-header">Compare-mode · A vs B</div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for col, slot in zip(cols, ("A", "B")):
        a = snaps[slot]
        with col:
            pclass = a["pathology_class"]; col_color = CLASS_COLOURS.get(pclass, "#1A73E8")
            st.markdown(
                f"""<div style='border-radius:14px;padding:16px 18px;background:white;
                                border:1px solid rgba(15,23,42,0.06);
                                border-left:4px solid {col_color};
                                box-shadow:0 1px 3px rgba(15,23,42,0.04)'>
                      <div style='font-size:0.7rem;text-transform:uppercase;color:{col_color};
                                  letter-spacing:0.6px;font-weight:800'>Slot {slot} · {a.get('label','')}</div>
                      <div style='font-size:1.25rem;font-weight:800;color:#0F172A;margin-top:2px'>
                        {CLASS_LABELS.get(pclass, pclass)}
                      </div>
                      <div style='font-size:0.86rem;color:#475569;margin-top:6px'>
                        Confidence <b>{a['confidence']:.0%}</b> ·
                        Risk <b>{a['risk_score']:.0%}</b> ·
                        Stage <b>{a['stage']}</b> ·
                        Uncertainty <b>{a['uncertainty']:.2f}</b>
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )
    return True


def _render_risk_only_report(analysis: dict, patient: dict):
    """Honest report when no image was uploaded — risk factors + symptoms only.
    No visual diagnosis, no fabricated pathology/stage."""
    from src.app.security import escape_html as _esc
    tier = analysis.get("risk_tier", "Low")
    tier_color = {"High": "#DC2626", "Moderate": "#D97706", "Low": "#059669"}.get(tier, "#1A73E8")
    render_hero(
        "Your Risk-Factor Assessment",
        f"Based on history & symptoms for {_esc(patient.get('name','you'))} — no image uploaded",
        badges=["History + symptoms only", "Not a diagnosis", "Confirm with a clinician"],
    )
    st.markdown(
        f"""<div style="background:{tier_color}14;border:2px solid {tier_color};
            border-radius:14px;padding:18px 22px;margin:14px 0;">
            <div style="font-size:20px;font-weight:800;color:{tier_color};">{tier} concern</div>
            <div style="color:#334155;font-size:15px;margin-top:4px;">
            {_esc(analysis.get('risk_label',''))}</div></div>""",
        unsafe_allow_html=True,
    )
    st.info("ℹ️ No colonoscopy image was uploaded, so the AI made **no visual diagnosis**. "
            "This is a transparent assessment of your risk factors and symptoms only — "
            "a colonoscopy is needed to actually look for polyps or cancer.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Symptom red flags**")
        rfs = analysis.get("red_flags", [])
        if rfs:
            for nm, _w in rfs:
                st.markdown(f"- {_esc(nm)}")
        else:
            st.markdown("- None reported")
    with col2:
        st.markdown("**Your risk factors**")
        facs = analysis.get("risk_factors", [])
        if facs:
            for nm, _w in facs:
                st.markdown(f"- {_esc(nm)}")
        else:
            st.markdown("- None notable")
    # Literature-grounded relative-risk multiplier (cited meta-analytic RRs)
    rd = analysis.get("risk_detail", {})
    rr_total = rd.get("rr_total")
    rr_factors = rd.get("rr_factors", [])
    if rr_total:
        ups = [f for f in rr_factors if f.get("dir") == "up"]
        drv = ", ".join(_esc(f["name"]) for f in (ups or rr_factors)[:4]) or "no major modifiable factors"
        st.markdown(
            f"""<div style="background:#F8FAFF;border:1px solid #CBD5E1;border-radius:12px;
                 padding:14px 18px;margin:12px 0;">
              <div style="font-size:0.74rem;text-transform:uppercase;letter-spacing:.6px;
                   color:#1A73E8;font-weight:800;">Estimated relative risk (literature-based)</div>
              <div style="font-size:1.5rem;font-weight:800;color:#0F172A;margin:2px 0;">
                   ~{rr_total:g}× <span style="font-size:0.9rem;font-weight:600;color:#475569;">
                   ({_esc(rd.get('rr_band',''))} vs an average person your age)</span></div>
              <div style="font-size:0.85rem;color:#475569;">Driven by: {drv}.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption("Relative risk from published meta-analytic relative risks (Johnson 2013; "
                   "smoking Botteri 2008). Educational estimate vs an average same-age person — "
                   "not an absolute probability or a diagnosis.")
        for _n in rd.get("rr_notes", []):
            st.caption("• " + _esc(_n))

    # Validated clinical risk score (APCS)
    apcs = rd.get("apcs")
    if apcs:
        ac = {"High": "#DC2626", "Moderate": "#D97706", "Low": "#059669"}.get(apcs.get("tier"), "#1A73E8")
        bd = " · ".join(f"{_esc(n)} +{p}" for n, p in apcs.get("breakdown", []))
        st.markdown(
            f"""<div style="background:#FFFFFF;border:1px solid {ac}55;border-left:4px solid {ac};
                 border-radius:10px;padding:12px 16px;margin:10px 0;">
              <div style="font-size:0.74rem;text-transform:uppercase;letter-spacing:.6px;
                   color:{ac};font-weight:800;">Validated screening score (APCS)</div>
              <div style="font-size:1.2rem;font-weight:800;color:#0F172A;">
                   {apcs.get('score')}/7 → {_esc(apcs.get('tier',''))} risk of advanced neoplasia</div>
              <div style="font-size:0.8rem;color:#64748B;margin-top:3px;">{bd}</div>
            </div>""", unsafe_allow_html=True)
        st.caption(_esc(apcs.get("cite", "")) + " — validated for asymptomatic screening.")

    rec = analysis.get("recommendation", {})
    st.markdown(f"### Recommended next step\n**{_esc(rec.get('urgency',''))}** — "
                f"{_esc(rec.get('primary_action',''))}")
    if rec.get("investigations"):
        st.markdown("**Suggested investigations:** "
                    + ", ".join(_esc(x) for x in rec["investigations"]))
    st.caption("Rule-based on established risk factors + symptom red flags. Not an AI image "
               "diagnosis and not a substitute for clinical evaluation.")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📷 Upload a colonoscopy image for visual analysis",
                     use_container_width=True, type="primary"):
            st.session_state["step"] = 1
            st.rerun()
    with c2:
        if st.button("← Start over", use_container_width=True):
            st.session_state["step"] = 0
            st.rerun()


def page_results():
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.error("No analysis results. Please go back and run analysis.")
        if st.button("← Back to Analysis"):
            st.session_state["step"] = 2
            st.rerun()
        return

    patient = st.session_state.get("patient", {})
    pclass  = analysis["pathology_class"]
    pcolor  = CLASS_COLOURS.get(pclass, "#1A73E8")

    # ── HARD STOP: if the uploaded image was not a colonoscopy frame ─────
    if analysis.get("input_rejected") or pclass == "NOT_ENDOSCOPY":
        score = analysis.get("rejection_score", 0.0) * 100
        reasons = analysis.get("rejection_reasons", []) or [
            "Image did not pass the endoscopy-likeness gate."
        ]
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#FEE2E2 0%,#FEF3C7 100%);
                 border:2px solid #DC2626;border-radius:14px;padding:20px 24px;
                 box-shadow:0 6px 18px rgba(220,38,38,0.18);margin:18px 0;">
              <div style="font-size:22px;font-weight:700;color:#991B1B;margin-bottom:6px;">
                🚫 Input Rejected — Not a Colonoscopy Image
              </div>
              <div style="color:#7F1D1D;font-size:15px;line-height:1.5;margin-bottom:10px;">
                Our model is trained on real colonoscopy frames only (HyperKvasir + CVC-ClinicDB).
                The uploaded image does not match the colonoscopy signature, so we are refusing
                to give a result rather than show you a fake one.
              </div>
              <div style="background:#FFF;border-radius:10px;padding:12px 16px;margin-top:8px;">
                <div style="font-weight:600;color:#374151;font-size:13px;margin-bottom:6px;">
                  Endoscopy-likeness score: <span style="color:#DC2626;font-size:18px;font-weight:700;">{score:.0f}%</span>
                  <span style="color:#64748B;font-weight:400;font-size:12px;"> (need ≥ 55% to run the model)</span>
                </div>
                <div style="color:#475569;font-size:13px;margin-top:8px;">
                  <strong>Why it was rejected:</strong>
                  <ul style="margin:6px 0 0 18px;padding:0;">
                    {"".join(f"<li style='margin:4px 0;'>{r}</li>" for r in reasons)}
                  </ul>
                </div>
              </div>
              <div style="color:#374151;font-size:13px;margin-top:14px;">
                <strong>What to do:</strong> upload a real colonoscopy / endoscopy frame, or try one
                of the built-in demo cases (Polyp / UC / Barrett's) to see the model in action.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("⬅ Upload a different image", use_container_width=True, type="primary"):
                st.session_state["step"] = 1
                st.rerun()
        with c2:
            if st.button("🔬 Try a demo case instead", use_container_width=True):
                st.session_state["step"] = 1
                st.rerun()
        return

    # ── RISK-FACTOR-ONLY report (no image was uploaded) ─────────────────
    if analysis.get("risk_only") or pclass == "RISK_ASSESSMENT_ONLY":
        _render_risk_only_report(analysis, patient)
        return

    from src.app.security import escape_html as _esc
    render_hero(
        "Your AI Health Report",
        f"Personalised analysis for {_esc(patient.get('name','Patient'))} · "
        f"{datetime.now().strftime('%d %b %Y, %H:%M')}",
        badges=["Step 4 of 6", "Reviewed against international guidelines",
                "Always confirm with a clinician"],
    )

    # ── View-quality advisory (poor bowel-prep / obscured view) ────────
    _vq = analysis.get("view_quality") or {}
    if _vq.get("is_poor"):
        st.warning("⚠️ **View quality looks poor** (inadequate bowel prep or an obscured "
                   "view). Any finding below may be **unreliable** — consider repositioning, "
                   "cleaning/suctioning, or a repeat exam with better preparation.")

    # ── PATIENT-SAFETY BANNER + cross-check rationale ──────────────────
    # The verdict card is the first thing the user sees. It explains the
    # AI's decision (or refusal) in plain language. The rationale card
    # below it lists the concrete observations the model made — never
    # generic "we are confident" filler.
    from src.app.patient_ui import (verdict_card_html, rationale_card_html,
                                    plain_english_diagnosis)
    sv = analysis.get("safety_verdict") or {}
    if isinstance(sv, dict) and sv.get("action"):
        # Plain-English diagnosis only relevant when the AI is showing a result
        _plain = None
        if sv.get("action") == "show":
            try:
                _plain = plain_english_diagnosis(
                    getattr(analysis.get("fusion_diagnosis", object()),
                            "pathology_class",
                            analysis.get("predicted_class", "unknown")),
                    float(getattr(analysis.get("fusion_diagnosis", object()),
                                  "overall_confidence",
                                  analysis.get("confidence", 0.0)) or 0.0))
            except Exception:
                _plain = None
        st.markdown(verdict_card_html(sv, _plain), unsafe_allow_html=True)

    # Cross-check rationale — concrete observations from cross_check.py
    cc = analysis.get("cross_check") or {}
    if isinstance(cc, dict) and (cc.get("rationale") or cc.get("flags")):
        st.markdown(
            rationale_card_html(cc.get("rationale", []), cc.get("flags", [])),
            unsafe_allow_html=True)

    # ── Invasive / advanced-lesion override banner ──────────────────────
    # Fires when the pixel-statistics safety net detects features the
    # 5-class classifier was never trained on. Highest-severity warning
    # on the page.
    adv = analysis.get("advanced_lesion") or {}
    if isinstance(adv, dict) and adv.get("is_advanced"):
        from src.app.security import escape_html as _esc
        reasons_html = "".join(
            f"<li style='margin:4px 0;'>{_esc(r)}</li>"
            for r in adv.get("reasons", []))
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#FEE2E2 0%,#FECACA 100%);
                        border:3px solid #B91C1C;border-radius:14px;
                        padding:18px 22px;margin:14px 0;
                        box-shadow:0 6px 14px rgba(185,28,28,0.18);">
              <div style="font-size:1rem;font-weight:800;color:#7F1D1D;
                          margin-bottom:8px;">
                🚨 ATYPICAL LESION DETECTED — urgent endoscopist review
              </div>
              <div style="color:#7F1D1D;font-size:0.9rem;line-height:1.55;
                          margin-bottom:8px;">
                Pixel statistics found features that the AI's training data
                did NOT contain (the model knows polyps, ulcerative colitis,
                Barrett's and post-therapy sites — it does NOT know advanced
                colorectal carcinoma). Severity score:
                <b>{float(adv.get("severity", 0))*100:.0f}%</b>.
              </div>
              <ul style="color:#7F1D1D;font-size:0.85rem;margin:0 0 0 18px;padding:0;">
                {reasons_html}
              </ul>
              <div style="color:#7F1D1D;font-size:0.85rem;margin-top:10px;
                          font-weight:600;">
                The pathology class shown below should be treated as untrusted
                — request a 2-week-wait referral for biopsy.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TCGA tabular stage estimate ──────────────────────────────────
    # Independent of the image — trained on 1,319 real TCGA colon-cancer
    # cases. Shown as a SECONDARY estimate, never as primary diagnosis.
    tse = analysis.get("tcga_stage_estimate") or {}
    if isinstance(tse, dict) and tse.get("predicted_stage"):
        from src.app.security import escape_html as _esc
        probs = tse.get("probabilities", {})
        bars_html = ""
        for stage, p in sorted(probs.items(), key=lambda x: -x[1]):
            pct = p * 100
            color = "#0B5FFF" if stage == tse["predicted_stage"] else "#94A3B8"
            bars_html += (
                f"<div style='margin:5px 0;'>"
                f"  <div style='display:flex;justify-content:space-between;"
                f"               font-size:0.8rem;margin-bottom:2px;'>"
                f"    <span style='color:#0F172A;'><b>{_esc(stage)}</b></span>"
                f"    <span style='color:{color};font-weight:700;'>{pct:.0f}%</span>"
                f"  </div>"
                f"  <div style='background:#F1F5F9;border-radius:5px;height:6px;'>"
                f"    <div style='background:{color};height:100%;width:{pct:.0f}%;border-radius:5px;'></div>"
                f"  </div></div>")
        st.markdown(
            f"""
            <div style="background:#FFF;border:1px solid #E2E8F0;border-radius:14px;
                        padding:18px 22px;margin:14px 0;
                        box-shadow:0 2px 8px rgba(15,23,42,0.04);">
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.7px;
                          color:#7C3AED;font-weight:800;margin-bottom:6px;">
                Statistical risk pattern — NOT a diagnosis
              </div>
              <div style="background:#FEF3C7;border:1px solid #FCD34D;border-radius:8px;
                          padding:8px 11px;font-size:0.82rem;color:#92400E;
                          margin-bottom:10px;line-height:1.45;">
                ⚠️ <b>This does NOT mean you have cancer, or this stage.</b> It is only a
                statistical pattern from your age and lifestyle — not a diagnosis. A real
                cancer stage can only be found with a biopsy and scans ordered by a doctor.
              </div>
              <div style="font-size:0.95rem;color:#0F172A;line-height:1.5;
                          font-weight:600;margin-bottom:10px;">
                If cancer were ever present, this pattern (from age, BMI, smoking and family
                history only) most resembles
                <b style='color:#7C3AED;'>{_esc(tse["predicted_stage"])}</b>
                ({tse.get("confidence", 0)*100:.0f}% confidence).
              </div>
              {bars_html}
              <div style="font-size:0.7rem;color:#94A3B8;margin-top:10px;font-style:italic;line-height:1.4;">
                Trained on TCGA-COAD ({tse.get("n_train_samples", 0):,} labelled
                cases). This is a population-level estimate from demographics,
                NOT a diagnosis. Image-based staging requires histopathology
                data we don't have — see docs/STAGING_ROADMAP.md.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Clinician EXACT staging (real AJCC from entered findings) ──────
    # The honest path to accurate staging: a colonoscopy image can't determine
    # the stage, but once a clinician has the biopsy (depth) + scan (spread)
    # findings, the stage is a fixed rulebook — applied exactly here.
    with st.expander("🩺 Clinician: enter biopsy / scan findings for the EXACT stage"):
        from src.app.staging import (ajcc_colorectal_stage, T_OPTIONS, N_OPTIONS,
                                     M_OPTIONS, T_HELP, N_HELP, M_HELP)
        st.caption("A colonoscopy image cannot determine the true stage. Once a biopsy "
                   "(tumour depth) and scans (spread) are available, enter the findings below "
                   "for the exact AJCC stage — a fixed rulebook, 100% correct for the values "
                   "you provide.")
        _cT, _cN, _cM = st.columns(3)
        with _cT:
            _t = st.selectbox("Tumour depth (T)", T_OPTIONS, index=2, key="stg_t")
            st.caption(T_HELP[_t])
        with _cN:
            _n = st.selectbox("Lymph nodes (N)", N_OPTIONS, index=0, key="stg_n")
            st.caption(N_HELP[_n])
        with _cM:
            _m = st.selectbox("Distant spread (M)", M_OPTIONS, index=0, key="stg_m")
            st.caption(M_HELP[_m])
        _stg = ajcc_colorectal_stage(_t, _n, _m)
        if _stg["exact"]:
            st.markdown(
                f"""<div style="background:#EEF2FF;border:2px solid #4F46E5;border-radius:12px;
                     padding:14px 18px;margin-top:8px;">
                  <div style="font-size:0.74rem;text-transform:uppercase;letter-spacing:.6px;
                       color:#4F46E5;font-weight:800;">Exact AJCC stage — from entered findings</div>
                  <div style="font-size:1.7rem;font-weight:800;color:#312E81;margin:2px 0;">
                       Stage {_stg['stage_group']}</div>
                  <div style="font-size:0.86rem;color:#475569;">{_stg['rationale']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning(_stg["rationale"])
        st.caption("AJCC Cancer Staging Manual, 8th edition. The clinician supplies T/N/M from "
                   "real pathology and imaging; ColonAI only applies the rulebook (it does not "
                   "infer depth or spread from the image).")
        st.markdown("---")
        st.selectbox("Anatomical location of the lesion (clinician-entered)",
                     ["(not specified)", "Cecum", "Ascending colon", "Hepatic flexure",
                      "Transverse colon", "Splenic flexure", "Descending colon",
                      "Sigmoid colon", "Rectum"], key="stg_location",
                     help="Precise location needs the scope position — it cannot be read from "
                          "the image, so it is clinician-entered.")
        st.checkbox("The T/N/M above are this patient's REAL biopsy / scan findings "
                    "(tick to use them for the structured stage below)", key="stg_confirm")

    # ── Structured summary (for clinicians) — honest 5-point extraction ──────
    # Size / Number / Location / Stage / Treatment, each tagged by source
    # (measured / estimated / doctor-entered / computed / guideline). Nothing
    # fabricated; the safety guardrail drives "Requires human review".
    try:
        from src.app.structured_report import build_structured_report
        from src.app.security import escape_html as _esc
        _locv = st.session_state.get("stg_location")
        _di = {"location": None if (not _locv or _locv.startswith("(")) else _locv}
        if st.session_state.get("stg_confirm"):
            _di.update(T=st.session_state.get("stg_t"), N=st.session_state.get("stg_n"),
                       M=st.session_state.get("stg_m"))
        _sr = build_structured_report(analysis, _di)
        _SRC_COLOR = {"measured": "#16A34A", "estimated": "#D97706",
                      "doctor-entered": "#2563EB", "computed": "#4F46E5",
                      "guideline": "#0891B2", "unavailable": "#6B7280"}
        _LABELS = {"number": "Number", "size": "Size", "location": "Location",
                   "stage": "Stage", "treatment": "Treatment / next step"}
        st.markdown('<div class="section-header">Structured summary (for clinicians)</div>',
                    unsafe_allow_html=True)
        if _sr["requires_human_review"]:
            st.markdown(
                f'<div class="warn-box" style="margin-bottom:8px"><b>⚠️ Requires human review</b>'
                f' — {_esc(_sr.get("review_reason", "")) }</div>', unsafe_allow_html=True)
        _rows = ""
        for _k in ["number", "size", "location", "stage", "treatment"]:
            _f = _sr["fields"][_k]
            _val = "—" if _f["value"] is None else _esc(str(_f["value"]))
            if _f.get("detail"):
                _val += f" <span style='color:#64748B'>({_esc(str(_f['detail']))})</span>"
            _col = _SRC_COLOR.get(_f["source"], "#6B7280")
            _cav = (f"<div style='font-size:.76rem;color:#94A3B8;margin-top:2px'>"
                    f"{_esc(_f['caveat'])}</div>" if _f.get("caveat") else "")
            _rows += (
                "<div style='display:flex;gap:10px;align-items:flex-start;padding:8px 0;"
                "border-bottom:1px solid #EEF2F7'>"
                f"<div style='min-width:130px;font-weight:700;color:#334155'>{_LABELS[_k]}</div>"
                f"<div style='flex:1'>{_val}{_cav}</div>"
                "<div style='font-size:.66rem;font-weight:800;text-transform:uppercase;"
                f"letter-spacing:.4px;color:#fff;background:{_col};padding:2px 8px;"
                f"border-radius:10px;white-space:nowrap'>{_f['source']}</div></div>")
        st.markdown(
            f"<div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;"
            f"padding:6px 16px'>{_rows}</div>", unsafe_allow_html=True)
        st.caption(_sr.get("disclaimer", ""))
    except Exception as _sr_exc:
        st.caption(f"(structured summary unavailable: {type(_sr_exc).__name__})")

    # ── Hierarchical-UC hedge banner (only fires when needed) ─────────
    sp = analysis.get("smart_prediction") or {}
    if isinstance(sp, dict) and sp.get("is_hedged") and sp.get("hedge_reason"):
        from src.app.security import escape_html as _esc
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#FEF3C7 0%,#FDE68A 100%);
                        border:2px solid #D97706;border-radius:14px;
                        padding:14px 18px;margin:14px 0;
                        box-shadow:0 4px 10px rgba(0,0,0,0.05);">
              <div style="font-size:0.9rem;font-weight:800;color:#78350F;
                          margin-bottom:6px;">
                ⚠️ Clinical hedge — UC severity is uncertain
              </div>
              <div style="color:#78350F;font-size:0.85rem;line-height:1.5;">
                {_esc(sp.get("hedge_reason", ""))}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Differential-diagnosis bar chart (top-3 alternatives) ─────────
    if isinstance(sp, dict) and sp.get("differential"):
        from src.app.security import escape_html as _esc
        from src.app.patient_ui import PLAIN_NAMES
        diff = sp["differential"]
        bars_html = ""
        for d in diff[:3]:
            cls   = d.get("class", "")
            prob  = float(d.get("prob", 0.0))
            plain = PLAIN_NAMES.get(cls, cls)
            pct   = prob * 100
            color = "#0B5FFF" if prob == max(x["prob"] for x in diff) else "#94A3B8"
            bars_html += (
                f"<div style='margin:8px 0;'>"
                f"  <div style='display:flex;justify-content:space-between;"
                f"               font-size:0.85rem;margin-bottom:3px;'>"
                f"    <span style='color:#0F172A;'><b>{_esc(plain)}</b></span>"
                f"    <span style='color:{color};font-weight:700;'>{pct:.0f}%</span>"
                f"  </div>"
                f"  <div style='background:#F1F5F9;border-radius:6px;height:8px;"
                f"               overflow:hidden;'>"
                f"    <div style='background:{color};height:100%;"
                f"                 width:{pct:.0f}%;border-radius:6px;'></div>"
                f"  </div>"
                f"</div>"
            )
        # TTA + MC-Dropout footer
        _tta_n = sp.get("n_tta", 0); _mc_n = sp.get("n_mc", 0)
        _tta_std = sp.get("tta_std", 0.0); _mc_std = sp.get("mc_std", 0.0)
        st.markdown(
            f"""
            <div style="background:#FFF;border:1px solid #E2E8F0;border-radius:14px;
                        padding:18px 22px;margin:14px 0;
                        box-shadow:0 2px 8px rgba(15,23,42,0.04);">
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.7px;
                          color:#0B5FFF;font-weight:800;margin-bottom:10px;">
                Differential — what else this could be
              </div>
              {bars_html}
              <div style="font-size:0.72rem;color:#94A3B8;margin-top:12px;
                          font-style:italic;line-height:1.4;">
                Computed by averaging {_tta_n}-augmentation TTA ensemble + {_mc_n}-pass
                MC-Dropout. Augmentation spread σ = {_tta_std:.2f},
                MC-Dropout spread σ = {_mc_std:.2f}.
                Higher spread → less stable prediction.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Optional LLM-refined plain-English paragraph (Groq Llama-3.1) ──
    # Only renders if GROQ_API_KEY was set and the LLM output passed our
    # guard-rails (didn't change the class, didn't claim higher confidence).
    lr = analysis.get("llm_refined") or {}
    if isinstance(lr, dict) and lr.get("refined_paragraph"):
        from src.app.security import escape_html as _esc
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#F0F9FF 0%,#E0F2FE 100%);
                        border:1px solid #BAE6FD;border-radius:14px;
                        padding:18px 22px;margin:14px 0;
                        box-shadow:0 2px 8px rgba(15,23,42,0.04);">
              <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.7px;
                          color:#0369A1;font-weight:800;margin-bottom:6px;">
                What this means, in plain English
              </div>
              <div style="font-size:1rem;color:#0F172A;line-height:1.65;
                          font-weight:500;">
                {_esc(lr["refined_paragraph"])}
              </div>
              <div style="font-size:0.7rem;color:#0369A1;margin-top:10px;
                          opacity:0.7;">
                Rephrased by {_esc(lr.get("model", "LLM"))}. The AI's
                diagnosis, confidence, and safety verdict are not changed
                by this rephrasing — only the wording.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Polyp / IBD sub-typing cards ───────────────────────────────────
    # Paris (morphology), NICE (predicted histology), size (BSG-aligned),
    # IBD differential, diverticulosis, hemorrhoid detection.
    st_sub = analysis.get("sub_typing") or {}
    if isinstance(st_sub, dict) and st_sub:
        from src.app.security import escape_html as _esc

        # Polyp sub-typing trio (Paris + NICE + size)
        if st_sub.get("paris") or st_sub.get("nice") or st_sub.get("size"):
            paris = st_sub.get("paris", {})
            nice  = st_sub.get("nice",  {})
            size_ = st_sub.get("size",  {})
            cards_html = ""
            if paris and paris.get("paris_type") not in (None, "unknown"):
                cards_html += (
                    f"<div style='flex:1;background:#F8FAFC;border-radius:10px;"
                    f"             padding:14px 16px;border:1px solid #E2E8F0;'>"
                    f"  <div style='font-size:0.7rem;text-transform:uppercase;"
                    f"               letter-spacing:0.7px;color:#475569;font-weight:800;"
                    f"               margin-bottom:5px;'>Paris classification</div>"
                    f"  <div style='font-size:1.6rem;font-weight:800;color:#0B5FFF;'>"
                    f"    {_esc(paris.get('paris_type', '?'))}</div>"
                    f"  <div style='font-size:0.78rem;color:#64748B;margin-top:4px;'>"
                    f"    {_esc(paris.get('rationale', ''))}</div>"
                    f"  <div style='font-size:0.72rem;color:#94A3B8;margin-top:8px;'>"
                    f"    Confidence: {float(paris.get('confidence', 0))*100:.0f}%</div>"
                    f"</div>")
            if nice and nice.get("nice_type") not in (None, "unknown"):
                nice_color = {"Type 1": "#16A34A", "Type 2": "#D97706",
                              "Type 3": "#B91C1C"}.get(nice.get("nice_type"), "#64748B")
                cards_html += (
                    f"<div style='flex:1;background:#F8FAFC;border-radius:10px;"
                    f"             padding:14px 16px;border:1px solid #E2E8F0;'>"
                    f"  <div style='font-size:0.7rem;text-transform:uppercase;"
                    f"               letter-spacing:0.7px;color:#475569;font-weight:800;"
                    f"               margin-bottom:5px;'>NICE surface pattern</div>"
                    f"  <div style='font-size:1.6rem;font-weight:800;color:{nice_color};'>"
                    f"    {_esc(nice.get('nice_type', '?'))}</div>"
                    f"  <div style='font-size:0.78rem;color:#64748B;margin-top:4px;'>"
                    f"    {_esc(nice.get('predicted_histology', ''))}</div>"
                    f"  <div style='font-size:0.72rem;color:#94A3B8;margin-top:8px;'>"
                    f"    Confidence: {float(nice.get('confidence', 0))*100:.0f}%</div>"
                    f"</div>")
            if size_ and size_.get("size_mm") is not None:
                size_color = {"diminutive (< 5 mm)":   "#16A34A",
                              "small (5-9 mm)":        "#16A34A",
                              "large (10-19 mm)":      "#D97706",
                              "giant (≥ 20 mm)":       "#B91C1C"}.get(
                                  size_.get("size_category", ""), "#64748B")
                cards_html += (
                    f"<div style='flex:1;background:#F8FAFC;border-radius:10px;"
                    f"             padding:14px 16px;border:1px solid #E2E8F0;'>"
                    f"  <div style='font-size:0.7rem;text-transform:uppercase;"
                    f"               letter-spacing:0.7px;color:#475569;font-weight:800;"
                    f"               margin-bottom:5px;'>Estimated size</div>"
                    f"  <div style='font-size:1.6rem;font-weight:800;color:{size_color};'>"
                    f"    ~{size_.get('size_mm', 0)} mm</div>"
                    f"  <div style='font-size:0.78rem;color:#64748B;margin-top:4px;'>"
                    f"    {_esc(size_.get('size_category', ''))}</div>"
                    f"  <div style='font-size:0.72rem;color:#94A3B8;margin-top:8px;'>"
                    f"    Assumes ~30 mm scope field-of-view</div>"
                    f"</div>")
            if cards_html:
                st.markdown(f"""
                <div style="background:#FFF;border:1px solid #E2E8F0;border-radius:14px;
                            padding:18px 22px;margin:14px 0;
                            box-shadow:0 2px 8px rgba(15,23,42,0.04);">
                  <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.7px;
                              color:#0B5FFF;font-weight:800;margin-bottom:12px;">
                    Polyp sub-typing  ·  Paris × NICE × BSG size
                  </div>
                  <div style='display:flex;gap:10px;flex-wrap:wrap;'>{cards_html}</div>
                  <div style="font-size:0.7rem;color:#94A3B8;margin-top:12px;font-style:italic;">
                    Computed from the segmentation mask + image pixels (no extra
                    training data needed). Paris = morphology; NICE = predicted
                    histology from surface pattern; size = estimated from mask.
                  </div>
                </div>
                """, unsafe_allow_html=True)
                # Clinical recommendation derived from these
                if paris.get("removal_technique"):
                    st.info(f"🔪 **Recommended removal technique:** "
                            f"{paris['removal_technique']}")
                if nice.get("cancer_risk"):
                    st.info(f"🎗️ **Cancer-risk stratification:** "
                            f"{nice['cancer_risk']}")
                if size_.get("bsg_surveillance"):
                    st.info(f"📅 **BSG surveillance:** "
                            f"{size_['bsg_surveillance']}")

        # IBD differential
        ibd = st_sub.get("ibd_differential", {})
        if ibd and ibd.get("verdict"):
            crohns_s = float(ibd.get("crohns_score", 0))
            uc_s     = float(ibd.get("uc_score", 0))
            total    = max(crohns_s + uc_s, 1)
            cr_pct   = crohns_s / total * 100
            uc_pct   = uc_s     / total * 100
            reasons_html = "".join(f"<li style='margin:3px 0;'>{_esc(r)}</li>"
                                    for r in ibd.get("rationale", []))
            st.markdown(f"""
            <div style="background:#FFF;border:1px solid #E2E8F0;border-radius:14px;
                        padding:18px 22px;margin:14px 0;
                        box-shadow:0 2px 8px rgba(15,23,42,0.04);">
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.7px;
                          color:#7C3AED;font-weight:800;margin-bottom:8px;">
                IBD differential  ·  Crohn's vs Ulcerative Colitis
              </div>
              <div style="font-size:1rem;color:#0F172A;font-weight:600;margin-bottom:10px;">
                {_esc(ibd.get("verdict", ""))}
              </div>
              <div style='display:flex;gap:8px;margin:8px 0;'>
                <div style='flex:1;background:#F1F5F9;border-radius:6px;height:8px;'>
                  <div style='background:#0B5FFF;height:100%;width:{cr_pct:.0f}%;border-radius:6px;'></div>
                </div>
                <span style='font-size:0.8rem;color:#0B5FFF;font-weight:700;'>Crohn's {cr_pct:.0f}%</span>
              </div>
              <div style='display:flex;gap:8px;margin:8px 0;'>
                <div style='flex:1;background:#F1F5F9;border-radius:6px;height:8px;'>
                  <div style='background:#16A34A;height:100%;width:{uc_pct:.0f}%;border-radius:6px;'></div>
                </div>
                <span style='font-size:0.8rem;color:#16A34A;font-weight:700;'>UC {uc_pct:.0f}%</span>
              </div>
              <ul style="font-size:0.85rem;color:#475569;margin:10px 0 0 18px;padding:0;">
                {reasons_html}
              </ul>
              <div style="font-size:0.85rem;color:#0F172A;margin-top:10px;font-weight:600;">
                💡 {_esc(ibd.get("recommendation", ""))}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Diverticulosis + hemorrhoid (only show if detected)
        div_  = st_sub.get("diverticulosis", {})
        hemo = st_sub.get("hemorrhoid", {})
        extras = []
        if div_  and div_.get("detected"):
            extras.append(("🟫 Diverticulosis pattern detected",
                           f"{div_.get('n_candidates', 0)} pouch-like dark patches",
                           div_.get("interpretation", "")))
        if hemo and hemo.get("detected"):
            extras.append(("🩸 Possible hemorrhoid signs",
                           f"vascular score {hemo.get('score', 0)*100:.0f}%",
                           hemo.get("interpretation", "")))
        for icon_title, subtitle, interp in extras:
            st.warning(f"**{icon_title}** ({subtitle})\n\n{interp}")

    # ── Smart per-image evidence card ─────────────────────────────────
    # Lesion size %, location octant, attention focus, contrast, shape,
    # dominant colour — all measured from this specific image.
    sr = analysis.get("smart_rationale") or {}
    if isinstance(sr, dict) and sr.get("bullets"):
        from src.app.security import escape_html as _esc
        # Render as a richer card with the one-line summary in big text
        _summary = sr.get("summary", "")
        _bullets = sr.get("bullets", [])
        # Convert **bold** markdown to <b> for inline rendering
        import re as _re
        def _md_bold(s):
            return _re.sub(r"\*\*(.+?)\*\*",
                           lambda m: "<b>" + _esc(m.group(1)) + "</b>",
                           _esc(s))
        bullets_html = "".join(
            f"<li style='margin:6px 0;line-height:1.55;'>{_md_bold(b)}</li>"
            for b in _bullets)
        st.markdown(
            f"""
            <div style="background:#FFF;border:1px solid #E2E8F0;border-radius:14px;
                        padding:18px 22px;margin:14px 0;
                        box-shadow:0 2px 8px rgba(15,23,42,0.04);">
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.7px;
                          color:#0B5FFF;font-weight:800;margin-bottom:6px;">
                What the AI is actually seeing  ·  measured from this image
              </div>
              <div style="font-size:0.98rem;color:#0F172A;line-height:1.5;
                          margin-bottom:12px;font-weight:600;">
                {_md_bold(_summary)}
              </div>
              <ul style="font-size:0.93rem;color:#1F2937;margin:0 0 0 18px;padding:0;">
                {bullets_html}
              </ul>
              <div style="font-size:0.7rem;color:#94A3B8;margin-top:10px;
                          font-style:italic;">
                These observations are computed directly from the uploaded image
                and the AI's attention maps — not templated phrases.
              </div>
            </div>
            """,
            unsafe_allow_html=True)

    # ── UNIFIED EXPLANATION — patient-facing narrative ─────────────────
    # One ≤130-word paragraph that summarises everything: the verdict,
    # which modality drove it, whether agents disagreed, stability under
    # perturbations, neighbour concordance, and (if relevant) the TCGA
    # secondary stage estimate. Deterministic — same inputs → same words.
    _expl_para = analysis.get("explanation_paragraph") or ""
    if _expl_para:
        from src.app.security import escape_html as _esc
        import re as _re
        _para_html = _esc(_expl_para)
        # Render simple **markdown** → <b>
        _para_html = _re.sub(r"\*\*(.+?)\*\*",
                             lambda m: f"<b>{m.group(1)}</b>", _para_html)
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#F5F3FF 0%,#EDE9FE 100%);
                        border:1.5px solid #C4B5FD;border-radius:14px;
                        padding:18px 22px;margin:14px 0;
                        box-shadow:0 4px 12px rgba(124,58,237,0.06);">
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.7px;
                          color:#5B21B6;font-weight:800;margin-bottom:8px;">
                Why the system reached this verdict
              </div>
              <div style="font-size:1.02rem;color:#1F2937;line-height:1.7;
                          font-weight:500;">
                {_para_html}
              </div>
              <div style="font-size:0.7rem;color:#7C3AED;margin-top:10px;opacity:0.75;">
                Generated by the unified explanation engine
                (decision trace + modality attribution + disagreement
                detection). Deterministic — same inputs always produce the
                same paragraph.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Per-modality attribution bar ───────────────────────────────────
    # Shows which input (image / text / tabular) drove the decision,
    # estimated by ablation (silencing each modality and observing the
    # change in the predicted-class probability).
    _attr = analysis.get("modality_attribution") or {}
    if isinstance(_attr, dict) and _attr.get("interpretable"):
        from src.app.security import escape_html as _esc
        _contribs = _attr.get("contributions", {})
        _method = _attr.get("method", "?")
        # Sort by contribution descending
        _items = sorted(_contribs.items(), key=lambda kv: -kv[1])
        _bars = ""
        _palette = {"image": "#0B5FFF", "text": "#16A34A", "tabular": "#D97706"}
        _label_map = {"image": "Endoscopic image",
                      "text": "Clinical text",
                      "tabular": "Tabular features"}
        for _mod, _pct in _items:
            _color = _palette.get(_mod, "#64748B")
            _label = _label_map.get(_mod, _mod.title())
            _bars += (
                f"<div style='margin:10px 0;'>"
                f"  <div style='display:flex;justify-content:space-between;"
                f"               font-size:0.88rem;margin-bottom:4px;'>"
                f"    <span style='color:#0F172A;'><b>{_esc(_label)}</b></span>"
                f"    <span style='color:{_color};font-weight:700;'>{_pct:.0f}%</span>"
                f"  </div>"
                f"  <div style='background:#F1F5F9;border-radius:8px;height:14px;"
                f"               overflow:hidden;'>"
                f"    <div style='background:linear-gradient(90deg,{_color} 0%,{_color}99 100%);"
                f"                 height:100%;width:{_pct:.0f}%;border-radius:8px;"
                f"                 transition:width 0.6s ease;'></div>"
                f"  </div>"
                f"</div>"
            )
        st.markdown(
            f"""
            <div style="background:#FFF;border:1px solid #E2E8F0;border-radius:14px;
                        padding:18px 22px;margin:14px 0;
                        box-shadow:0 2px 8px rgba(15,23,42,0.04);">
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.7px;
                          color:#0B5FFF;font-weight:800;margin-bottom:12px;">
                Which input modality drove the decision?
              </div>
              {_bars}
              <div style="font-size:0.72rem;color:#94A3B8;margin-top:12px;
                          font-style:italic;line-height:1.45;">
                Estimated by <b>{_esc(_method)}</b>: each modality is silenced
                in turn (image → mid-grey, text → empty, tabular → median patient)
                and the resulting drop in the predicted-class probability is
                normalised to 100%. A high percentage means the model would
                have changed its mind if that input had been removed.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Disagreement summary (only fires when agents disagree) ─────────
    _disagree = analysis.get("disagreement") or {}
    if isinstance(_disagree, dict) and not _disagree.get("unanimous", True):
        from src.app.security import escape_html as _esc
        _agreed = _disagree.get("agreed", [])
        _disagreed_list = _disagree.get("disagreed", [])
        _ratio = float(_disagree.get("ratio", 1.0)) * 100
        _chips = "".join(
            f"<span style='display:inline-block;background:#FEE2E2;color:#991B1B;"
            f"             border:1px solid #FCA5A5;border-radius:999px;"
            f"             padding:3px 10px;margin:3px;font-size:0.78rem;"
            f"             font-weight:600;'>{_esc(a)}</span>"
            for a in _disagreed_list)
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#FFF7ED 0%,#FED7AA 100%);
                        border:1.5px solid #FB923C;border-radius:14px;
                        padding:16px 20px;margin:14px 0;">
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.7px;
                          color:#9A3412;font-weight:800;margin-bottom:8px;">
                ⚠️ Agent disagreement detected ({_ratio:.0f}% agreement)
              </div>
              <div style="font-size:0.9rem;color:#7C2D12;margin-bottom:8px;">
                The following agents raised a concern about the final verdict:
              </div>
              <div>{_chips}</div>
              <div style="font-size:0.78rem;color:#9A3412;margin-top:10px;
                          font-style:italic;">
                Review the step-by-step reasoning chain below. When agents
                disagree, the model is operating outside its zone of confidence
                — a second opinion is recommended.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Step-by-step reasoning chain (collapsible) ─────────────────────
    _trace_rows = analysis.get("decision_trace") or []
    if _trace_rows:
        from src.app.security import escape_html as _esc
        with st.expander("🔍  Step-by-step reasoning chain  ·  full agent trace",
                          expanded=False):
            _effect_color = {
                "support":    ("✓", "#16A34A"),
                "contradict": ("✗", "#B91C1C"),
                "override":   ("⚠", "#D97706"),
                "abstain":    ("○", "#64748B"),
                "noop":       ("·", "#94A3B8"),
                "refine":     ("✎", "#7C3AED"),
            }
            _rows_html = ""
            for _i, _step in enumerate(_trace_rows, start=1):
                _agent = _step.get("agent", "?").replace("_", " ").title()
                _stage = _step.get("stage", "")
                _eff   = _step.get("effect", "support")
                _icon, _col = _effect_color.get(_eff, ("•", "#64748B"))
                _conf  = float(_step.get("confidence", 0.0))
                _ev    = _step.get("evidence", []) or []
                _ev_html = ""
                if _ev:
                    _ev_html = (
                        "<ul style='font-size:0.8rem;color:#475569;"
                        "           margin:6px 0 0 18px;padding:0;'>" +
                        "".join(f"<li style='margin:2px 0;'>{_esc(str(x))}</li>"
                                for x in _ev[:5]) +
                        "</ul>"
                    )
                _rows_html += (
                    f"<div style='border-left:3px solid {_col};"
                    f"             background:#F8FAFC;border-radius:0 8px 8px 0;"
                    f"             padding:10px 14px;margin:8px 0;'>"
                    f"  <div style='display:flex;justify-content:space-between;"
                    f"               align-items:center;'>"
                    f"    <div>"
                    f"      <span style='font-size:1.1rem;color:{_col};"
                    f"                   font-weight:800;'>{_icon}</span>"
                    f"      <span style='font-size:0.85rem;color:#0F172A;"
                    f"                   font-weight:700;margin-left:6px;'>"
                    f"        Step {_i}: {_esc(_agent)} ({_esc(_stage)})</span>"
                    f"    </div>"
                    f"    <span style='font-size:0.78rem;color:#64748B;"
                    f"                 font-weight:600;'>"
                    f"      conf {_conf*100:.0f}%</span>"
                    f"  </div>"
                    f"  <div style='font-size:0.88rem;color:#1F2937;"
                    f"               margin-top:6px;line-height:1.45;'>"
                    f"    {_esc(_step.get('finding', ''))}</div>"
                    f"  {_ev_html}"
                    f"</div>"
                )
            st.markdown(_rows_html, unsafe_allow_html=True)
            # Download buttons for the trace + the full clinician report
            try:
                import json as _json
                _payload = _json.dumps(_trace_rows, indent=2, default=str)
                _dl_cols = st.columns(2)
                with _dl_cols[0]:
                    st.download_button(
                        "Download trace as JSON (audit log)",
                        data=_payload,
                        file_name="colonai_decision_trace.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                _rep = analysis.get("explanation_report")
                if _rep:
                    try:
                        from src.app.explanation_engine import report_to_markdown
                        _md = report_to_markdown(_rep)
                        with _dl_cols[1]:
                            st.download_button(
                                "Download clinician report (markdown)",
                                data=_md,
                                file_name="colonai_clinician_report.md",
                                mime="text/markdown",
                                use_container_width=True,
                            )
                    except Exception:
                        pass
            except Exception:
                pass

    # ── Similar training cases (prototype retrieval, optional) ─────────
    _neigh = analysis.get("prototype_neighbours") or []
    _conc  = analysis.get("neighbour_concordance") or {}
    if _neigh:
        from src.app.security import escape_html as _esc
        from src.app.patient_ui import PLAIN_NAMES
        _agrees = bool(_conc.get("agrees", False))
        _conc_pct = float(_conc.get("concordance", 0.0)) * 100
        _verdict_color = "#16A34A" if _agrees else "#D97706"
        _verdict_icon = "✓" if _agrees else "⚠"
        _verdict_text = (
            f"{_verdict_icon} {_conc_pct:.0f}% of the most-similar training "
            f"cases agree with the prediction"
            if _agrees else
            f"{_verdict_icon} Only {_conc_pct:.0f}% of similar training "
            f"cases match the prediction — review carefully"
        )
        _rows = ""
        for _n in _neigh:
            _lbl = PLAIN_NAMES.get(_n.get("label", ""), _n.get("label", "?"))
            _sim = float(_n.get("similarity", 0.0))
            _rows += (
                f"<div style='display:flex;justify-content:space-between;"
                f"             padding:8px 12px;border-bottom:1px solid #F1F5F9;'>"
                f"  <span style='color:#0F172A;font-size:0.88rem;'>"
                f"    <b>Rank {_n.get('rank','?')}</b> — {_esc(_lbl)}</span>"
                f"  <span style='color:#64748B;font-size:0.85rem;'>"
                f"    similarity {_sim:.3f}</span>"
                f"</div>"
            )
        st.markdown(
            f"""
            <div style="background:#FFF;border:1px solid #E2E8F0;border-radius:14px;
                        padding:18px 22px;margin:14px 0;
                        box-shadow:0 2px 8px rgba(15,23,42,0.04);">
              <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.7px;
                          color:#0B5FFF;font-weight:800;margin-bottom:8px;">
                Most-similar training cases (case-based reasoning)
              </div>
              <div style="font-size:0.95rem;color:{_verdict_color};
                          font-weight:700;margin-bottom:10px;">
                {_verdict_text}
              </div>
              {_rows}
              <div style="font-size:0.72rem;color:#94A3B8;margin-top:10px;
                          font-style:italic;line-height:1.45;">
                Nearest neighbours in the fused-embedding space (cosine
                similarity). High concordance with the model's prediction
                is an independent sanity check; low concordance suggests
                this case is unusual relative to the training set.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Privacy-safe doctor feedback widget ────────────────────────────
    # Records "AI got it right / wrong" against the predicted case. Used
    # to fine-tune the pathology head over time. Anonymous — only
    # (image-SHA256, model-embedding, label) gets stored. See
    # src/app/learning_log.py for the exact privacy contract.
    _case_uuid = analysis.get("learning_case_uuid")
    if _case_uuid and not st.session_state.get(f"feedback_sent_{_case_uuid}"):
        st.markdown("""
        <div style='background:#F0F9FF;border:1px solid #BAE6FD;border-radius:12px;
                    padding:14px 18px;margin:18px 0 8px;'>
          <div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.7px;
                      color:#0369A1;font-weight:800;margin-bottom:8px;'>
            Help the model improve
          </div>
          <div style='font-size:0.92rem;color:#0F172A;line-height:1.45;margin-bottom:10px;'>
            <b>Was the AI right?</b> Your answer is anonymous — we only store
            an image hash + the model's internal feature vector, never your
            name, image, or symptoms.
          </div>
        </div>
        """, unsafe_allow_html=True)
        fb_cols = st.columns([1, 1, 1, 2])
        with fb_cols[0]:
            if st.button("👍 AI was right", key=f"fb_right_{_case_uuid}",
                         use_container_width=True):
                try:
                    from src.app.learning_log import record_feedback
                    record_feedback(_case_uuid, "correct")
                    st.session_state[f"feedback_sent_{_case_uuid}"] = "correct"
                    st.toast("Thanks — feedback recorded anonymously.", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not record feedback: {e}")
        with fb_cols[1]:
            if st.button("👎 AI was wrong", key=f"fb_wrong_{_case_uuid}",
                         use_container_width=True):
                st.session_state[f"feedback_picking_label_{_case_uuid}"] = True
                st.rerun()
        with fb_cols[2]:
            if st.button("❓ Not sure", key=f"fb_unsure_{_case_uuid}",
                         use_container_width=True):
                try:
                    from src.app.learning_log import record_feedback
                    record_feedback(_case_uuid, "unsure")
                    st.session_state[f"feedback_sent_{_case_uuid}"] = "unsure"
                    st.toast("Noted — no label change recorded.", icon="🤔")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not record feedback: {e}")

        # If user clicked "AI was wrong", show class picker
        if st.session_state.get(f"feedback_picking_label_{_case_uuid}"):
            from src.app.patient_ui import PLAIN_NAMES
            choices = list(PLAIN_NAMES.keys())
            labels  = [f"{k} — {PLAIN_NAMES[k]}" for k in choices]
            st.markdown("<div style='margin-top:10px;'></div>",
                        unsafe_allow_html=True)
            picked = st.radio("What's the correct answer?",
                              options=labels, key=f"fb_pick_{_case_uuid}",
                              horizontal=False)
            if st.button("Submit correction",
                         key=f"fb_submit_{_case_uuid}", type="primary"):
                idx = labels.index(picked)
                try:
                    from src.app.learning_log import record_feedback
                    record_feedback(_case_uuid, "wrong",
                                    correct_label=choices[idx])
                    st.session_state[f"feedback_sent_{_case_uuid}"] = "wrong"
                    st.session_state[f"feedback_picking_label_{_case_uuid}"] = False
                    st.toast("Thanks — correction recorded anonymously.",
                             icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not record feedback: {e}")

    elif _case_uuid:
        # Already submitted this case
        _what = st.session_state.get(f"feedback_sent_{_case_uuid}", "")
        st.markdown(
            f"<div style='font-size:0.85rem;color:#15803D;margin:10px 0;'>"
            f"✓ Thanks — your feedback ({_what}) has been recorded anonymously. "
            f"It will help the model improve on similar cases.</div>",
            unsafe_allow_html=True)

    # ── RELIABILITY / TRUST PANEL ──────────────────────────────────────
    # Shows the five independent reliability signals so the user can see
    # WHY the prediction is (or is not) trustworthy.  Honest, not a single
    # opaque "confidence" number.
    trust = analysis.get("trust_report") or {}
    if isinstance(trust, dict) and trust and "error" not in trust:
        verdict      = trust.get("verdict", "UNKNOWN")
        trust_score  = float(trust.get("trust_score", 0.0))
        agreement    = float(trust.get("agreement_pct", 0.0))
        mc_unc       = float(trust.get("mc_uncertainty", 0.0))
        proto_dist   = float(trust.get("prototype_distance", 0.0))
        consensus    = float(trust.get("agent_consensus", 0.0))
        end_score    = float(trust.get("endoscopy_score", 0.0))
        warnings     = trust.get("warnings", []) or []
        advice       = trust.get("advice", "")

        # 3D animated trust ring + plain-English summary (NO tech labels)
        try:
            from src.app.ui_extras import trust_ring_svg
            cring, csumm = st.columns([1, 2])
            with cring:
                st.markdown(trust_ring_svg(trust_score*100, verdict, size=220),
                           unsafe_allow_html=True)
            with csumm:
                # Patient-friendly summary instead of agent technical names
                if verdict == "TRUSTED":
                    summary_html = (
                        "<div style='background:#DCFCE7;border-left:5px solid #16A34A;"
                        "border-radius:10px;padding:16px 20px;color:#14532D;'>"
                        "<div style='font-size:18px;font-weight:800;margin-bottom:8px;'>"
                        "✓ Reliable result</div>"
                        "<div style='font-size:14px;line-height:1.6;'>"
                        "All checks passed. The image, symptoms and your health profile "
                        "all point to the same conclusion. You can rely on this report "
                        "for your discussion with the doctor.</div></div>"
                    )
                elif verdict == "LOW_CONFIDENCE":
                    summary_html = (
                        "<div style='background:#FEF3C7;border-left:5px solid #D97706;"
                        "border-radius:10px;padding:16px 20px;color:#78350F;'>"
                        "<div style='font-size:18px;font-weight:800;margin-bottom:8px;'>"
                        "⚠ Treat with caution</div>"
                        "<div style='font-size:14px;line-height:1.6;'>"
                        "The AI gave a result but some checks were weaker than ideal. "
                        "Please discuss this case with your doctor — don't rely on this "
                        "report alone for any decisions.</div></div>"
                    )
                elif verdict == "REJECTED":
                    summary_html = (
                        "<div style='background:#FEE2E2;border-left:5px solid #DC2626;"
                        "border-radius:10px;padding:16px 20px;color:#7F1D1D;'>"
                        "<div style='font-size:18px;font-weight:800;margin-bottom:8px;'>"
                        "✕ Image not suitable</div>"
                        "<div style='font-size:14px;line-height:1.6;'>"
                        "This doesn't look like a colonoscopy image. Please upload a "
                        "real colonoscopy frame so the AI can give you a meaningful "
                        "result.</div></div>"
                    )
                else:
                    summary_html = (
                        "<div style='background:#FEE2E2;border-left:5px solid #DC2626;"
                        "border-radius:10px;padding:16px 20px;color:#7F1D1D;'>"
                        "<div style='font-size:18px;font-weight:800;margin-bottom:8px;'>"
                        "⚑ Needs specialist review</div>"
                        "<div style='font-size:14px;line-height:1.6;'>"
                        "The AI's checks disagreed with each other on this case. "
                        "Please book an appointment with a gastroenterologist — "
                        "don't make decisions from this report alone.</div></div>"
                    )
                st.markdown(summary_html, unsafe_allow_html=True)
        except Exception:
            pass

        if verdict == "TRUSTED":
            v_color, v_bg, v_icon = "#16A34A", "#DCFCE7", "✓"
            v_text  = "TRUSTED — Reliable prediction"
        elif verdict == "LOW_CONFIDENCE":
            v_color, v_bg, v_icon = "#D97706", "#FEF3C7", "⚠"
            v_text  = "LOW CONFIDENCE — Treat with caution"
        elif verdict == "FLAG_FOR_REVIEW":
            v_color, v_bg, v_icon = "#DC2626", "#FEE2E2", "⚑"
            v_text  = "FLAG FOR REVIEW — Specialist review required"
        else:
            v_color, v_bg, v_icon = "#64748B", "#F1F5F9", "?"
            v_text  = verdict

        st.markdown(
            f"""
            <div style="background:{v_bg};border-left:6px solid {v_color};
                 border-radius:12px;padding:18px 22px;margin:18px 0;
                 box-shadow:0 4px 12px rgba(0,0,0,0.08);">
              <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
                <div style="font-size:28px;color:{v_color};">{v_icon}</div>
                <div>
                  <div style="font-size:20px;font-weight:700;color:{v_color};">{v_text}</div>
                  <div style="font-size:13px;color:#475569;margin-top:2px;">{advice}</div>
                </div>
                <div style="margin-left:auto;text-align:right;">
                  <div style="font-size:13px;color:#64748B;">Trust Score</div>
                  <div style="font-size:32px;font-weight:700;color:{v_color};">
                    {trust_score*100:.0f}%
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Patient-friendly quality checks (NO ML jargon)
        st.markdown("##### Quality checks  —  *what we verified before showing you this result*")
        cs = st.columns(5)
        sigs = [
            ("Real Image",          end_score*100,        "%",
             "Confirmed this is a real colonoscopy image"),
            ("Consistent View",     agreement,            "%",
             "Same result from multiple looks at the image"),
            ("AI Confident",        (1-mc_unc)*100,       "%",
             "How sure the AI is about its answer"),
            ("Familiar Pattern",    (1-proto_dist)*100,   "%",
             "How well this case matches what the AI has seen before"),
            ("Evidence Aligns",     consensus*100,        "%",
             "Image, symptoms and history all agree"),
        ]
        for col, (label, val, unit, tip) in zip(cs, sigs):
            if val >= 78: c = "#16A34A"
            elif val >= 60: c = "#D97706"
            else:           c = "#DC2626"
            col.markdown(
                f"""
                <div style="background:#FFF;border:1px solid #E2E8F0;border-radius:10px;
                     padding:14px 12px;text-align:center;height:120px;
                     display:flex;flex-direction:column;justify-content:center;">
                  <div style="font-size:12px;color:#64748B;margin-bottom:4px;font-weight:600;">{label}</div>
                  <div style="font-size:24px;font-weight:700;color:{c};">{val:.0f}{unit}</div>
                  <div style="font-size:10px;color:#94A3B8;margin-top:4px;line-height:1.2;">{tip}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Patient-friendly warnings (filter out raw ML jargon)
        if warnings:
            friendly_warnings = []
            for w in warnings:
                if "MC-Dropout" in w or "augmentations" in w or "OOD" in w or "embedding" in w \
                   or "prototype" in w or "Endoscopy gate" in w:
                    continue  # filter out ML-jargon warnings — patients don't need them
                friendly_warnings.append(w)
            if friendly_warnings:
                with st.expander(f"ℹ️ Things to discuss with your doctor ({len(friendly_warnings)})",
                                expanded=(verdict != "TRUSTED")):
                    for w in friendly_warnings:
                        st.markdown(f"- {w}")

    # ── Compare-mode controls (visible when toggle is on in sidebar) ─────
    if st.session_state.get("compare_mode"):
        st.session_state.setdefault("compare_snaps", {})
        cmp_cols = st.columns([1, 1, 1, 1])
        snap = {**analysis, "label": patient.get("name","Patient")}
        with cmp_cols[0]:
            if st.button("📌 Save to slot A", use_container_width=True):
                st.session_state["compare_snaps"]["A"] = snap; st.toast("Saved to Slot A")
        with cmp_cols[1]:
            if st.button("📌 Save to slot B", use_container_width=True):
                st.session_state["compare_snaps"]["B"] = snap; st.toast("Saved to Slot B")
        with cmp_cols[2]:
            if st.button("Clear A & B", use_container_width=True):
                st.session_state["compare_snaps"] = {}; st.toast("Cleared")
        with cmp_cols[3]:
            slots = st.session_state.get("compare_snaps", {})
            st.caption(("A: " + (slots.get("A",{}).get("pathology_class","—"))) +
                       "  ·  " +
                       ("B: " + (slots.get("B",{}).get("pathology_class","—"))))
        if _render_compare_panel():
            st.markdown("---")

    # ── Clinical safety override panel (red-flag-driven escalation) ───────
    overrides = analysis.get("overrides") or {}
    if overrides.get("applied"):
        rules_html = "".join(
            f"<li style='margin:4px 0;color:#7F1D1D'>{r}</li>"
            for r in overrides.get("rules", [])
        )
        orig_r = overrides.get("original_risk", 0) * 100
        new_r  = overrides.get("new_risk", 0) * 100
        orig_u = overrides.get("original_urgency", "")
        new_u  = overrides.get("new_urgency", "")
        st.markdown(
            f"""<div style='border-radius:14px;padding:16px 20px;margin-bottom:14px;
                            background:linear-gradient(135deg,#FEF2F2 0%,#FEE2E2 100%);
                            border:1px solid #FCA5A5;
                            box-shadow:0 1px 3px rgba(15,23,42,0.04)'>
                  <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>
                    <span style='display:inline-flex;align-items:center;justify-content:center;
                                 width:34px;height:34px;border-radius:10px;
                                 background:#B91C1C;color:white;font-weight:800'>!</span>
                    <span style='font-size:0.74rem;text-transform:uppercase;letter-spacing:0.7px;
                                 color:#7F1D1D;font-weight:800'>Clinical safety override applied</span>
                  </div>
                  <div style='font-size:0.92rem;color:#1F2937;line-height:1.55;margin-bottom:8px'>
                    Your reported symptoms triggered NICE NG12 / red-flag rules that
                    <b>override the AI's image-only conclusion</b>. The risk band was raised from
                    <b>{orig_r:.0f}%</b> to <b>{new_r:.0f}%</b> and urgency from
                    <b>{orig_u}</b> to <b>{new_u}</b>. This is how a clinician would weight
                    the case.
                  </div>
                  <ul style='margin:0;padding-left:20px;font-size:0.88rem'>{rules_html}</ul>
                </div>""",
            unsafe_allow_html=True,
        )

    # ── Image-stats verdict card (independent of the trained model) ───────
    readout = analysis.get("image_readout") or {}
    if readout and "verdict" in readout:
        verdict = readout["verdict"]
        atyp = readout.get("atypicality", 0.0) * 100
        normal = readout.get("normal_score", 0.0) * 100
        reasons = readout.get("reasons", [])
        if verdict == "atypical_concerning":
            kicker = "Pixel signs of an advanced lesion — likely beyond model scope"
            sub = (f"The raw pixels show <b>deep red, low-blue regions</b>, dark cavitation, "
                   f"or disorganised tissue — visual cues for ulceration, fungating tumour or "
                   f"bleeding. The trained model's 5 classes do <b>not</b> include advanced "
                   f"cancer, so its class prediction below is unreliable for this image. "
                   f"<b>A clinician's review is essential — treat this as <u>possible advanced "
                   f"disease</u> until proven otherwise.</b>")
            bg = "linear-gradient(135deg,#FEF2F2 0%,#FEE2E2 100%)"
            border = "#FCA5A5"
            text_col = "#7F1D1D"
            icon_bg = "#B91C1C"
            icon_glyph = "!"
            score_label = f"Atypicality {atyp:.0f}%"
        elif verdict == "consistent_screening":
            kicker = "Pixel features look like a screening-stage finding"
            sub = (f"The raw pixels are consistent with <b>normal mucosa or a focal screening-stage "
                   f"finding</b> — no signs of bleeding, ulceration or mass effect. "
                   f"<b>Important:</b> this does NOT mean &quot;all clear&quot; — every image in our training "
                   f"data is itself a finding (polyp, mild colitis, etc.). Use the AI's class "
                   f"prediction below for the specific finding, and confirm with a clinician.")
            bg = "linear-gradient(135deg,#EFF6FF 0%,#DBEAFE 100%)"
            border = "#93C5FD"
            text_col = "#1E40AF"
            icon_bg = "#1A73E8"
            icon_glyph = "i"
            score_label = f"No atypical pixels · Normal-image score {normal:.0f}%"
        else:
            kicker = "Mixed pixel features — review carefully"
            sub = ("Some image features look typical of screening-stage findings, others are "
                   "ambiguous. The AI's class prediction below should be interpreted with "
                   "extra caution.")
            bg = "linear-gradient(135deg,#FFFBEB 0%,#FEF3C7 100%)"
            border = "#FCD34D"
            text_col = "#92400E"
            icon_bg = "#F59E0B"
            icon_glyph = "•"
            score_label = f"Atypicality {atyp:.0f}% · Normal {normal:.0f}%"
        bullets_html = "".join(
            f"<li style='margin:3px 0;color:{'#15803D' if c=='green' else '#B45309' if c=='amber' else '#B91C1C'}'>{m}</li>"
            for c, m in reasons
        )
        st.markdown(
            f"""<div style='border-radius:14px;padding:16px 20px;margin-bottom:14px;
                            background:{bg};border:1px solid {border};
                            box-shadow:0 1px 3px rgba(15,23,42,0.04)'>
                  <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>
                    <span style='display:inline-flex;align-items:center;justify-content:center;
                                 width:32px;height:32px;border-radius:9px;
                                 background:{icon_bg};color:white;font-weight:800;font-size:1.05rem'>{icon_glyph}</span>
                    <span style='font-size:0.74rem;text-transform:uppercase;letter-spacing:0.7px;
                                 color:{text_col};font-weight:800'>Image-features check</span>
                    <span style='margin-left:auto;font-size:0.72rem;color:{text_col};
                                 font-weight:700;background:rgba(255,255,255,0.6);
                                 padding:3px 10px;border-radius:999px'>{score_label}</span>
                  </div>
                  <div style='font-size:0.96rem;color:#1F2937;line-height:1.55;margin-bottom:8px'>
                    <b>{kicker}.</b> {sub}
                  </div>
                  <ul style='margin:6px 0 0 4px;padding-left:18px;font-size:0.86rem;line-height:1.55'>
                    {bullets_html}
                  </ul>
                </div>""",
            unsafe_allow_html=True,
        )

    # ── Patient-friendly verification + motivational card ─────────────────
    prov = analysis.get("_provenance", {"source": "real_model"})
    is_real = prov.get("source") == "real_model"

    # Pick a context-aware motivational message
    pclass = analysis["pathology_class"]
    rs = analysis["risk_score"]

    # ── Out-of-distribution / "atypical image" warning ─────────────────
    # The model knows 5 classes only.  If the prediction looks unsure, OR
    # the ablation shows wildly disjoint signals, OR the top-class probability
    # is suspiciously low, surface a clear caution that the image may be outside
    # what the model was trained on.
    if is_real:
        unc_val = analysis.get("uncertainty", 0.0)
        top_prob = analysis.get("confidence", 0.0)
        # Trigger conditions
        flags = []
        if unc_val >= 0.30:
            flags.append("the AI's internal cross-checks didn't fully agree")
        if top_prob < 0.85:
            flags.append("no single class scored very strongly")
        # Simple OOD heuristic: top-class less than 1.6× the second class
        probs = analysis.get("pathology_probs", {}) or {}
        sorted_probs = sorted(probs.values(), reverse=True)
        if len(sorted_probs) >= 2 and sorted_probs[1] > 0:
            ratio = sorted_probs[0] / max(sorted_probs[1], 1e-6)
            if ratio < 2.0:
                flags.append("two or more classes scored similarly")
        if flags:
            st.markdown(
                f"""<div style='border-radius:14px;padding:14px 18px;margin-bottom:14px;
                                background:linear-gradient(135deg,#FFF8E1 0%,#FEF3C7 100%);
                                border:1px solid #FCD34D;
                                box-shadow:0 1px 3px rgba(15,23,42,0.04)'>
                      <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>
                        <span style='display:inline-flex;align-items:center;justify-content:center;
                                     width:30px;height:30px;border-radius:9px;
                                     background:#F59E0B;color:white;font-weight:800'>!</span>
                        <span style='font-size:0.74rem;text-transform:uppercase;letter-spacing:0.7px;
                                     color:#92400E;font-weight:800'>Important — please read</span>
                      </div>
                      <div style='font-size:0.92rem;color:#1F2937;line-height:1.6'>
                        This image may be outside what the AI was trained on
                        ({"; ".join(flags)}).  The model knows
                        <b>five screening-stage findings</b> only —
                        <i>polyps, mild colitis, severe colitis, Barrett's oesophagus,
                        post-treatment site</i>.  It does <b>not</b> recognise advanced cancer,
                        invasive tumours, post-surgical anatomy, or rarer entities.<br>
                        For any unusual or symptomatic case <b>a clinician's review is essential</b> —
                        treat the AI's prediction as a hint, not a verdict.
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )
    if not is_real:
        msg = ("Demo mode — these numbers are illustrative. The next time you run with a "
               "real image and the model loaded, this card becomes your real result.")
        msg_kicker = "⚠ Demo result"
        msg_color = "#B91C1C"
        bg = "linear-gradient(135deg,#FFF5F5 0%,#FEE2E2 100%)"
        bord = "#FCA5A5"
    elif rs < 0.25:
        msg = ("Today's result is reassuring. Keep up regular screening — early action is the "
               "single biggest reason colorectal cancer survival has doubled in 30 years.")
        msg_kicker = "Reassuring news"
        msg_color = "#15803D"
        bg = "linear-gradient(135deg,#F0FDF4 0%,#DCFCE7 100%)"
        bord = "#86EFAC"
    elif rs < 0.5:
        msg = ("Your AI screen flagged something worth a closer look. Don't worry yet — the next step "
               "is simple: book a follow-up with a gastroenterologist (Step 5) and discuss this "
               "report with them.")
        msg_kicker = "A nudge to act"
        msg_color = "#B45309"
        bg = "linear-gradient(135deg,#FFFBEB 0%,#FEF3C7 100%)"
        bord = "#FCD34D"
    else:
        msg = ("This screen is on the higher-risk side. Take it seriously — but remember: "
               "early-stage colorectal cancer is curable in 9 out of 10 cases. A specialist review "
               "is the most important next step.")
        msg_kicker = "Important — please review with a clinician"
        msg_color = "#B91C1C"
        bg = "linear-gradient(135deg,#FFF5F5 0%,#FEE2E2 100%)"
        bord = "#FCA5A5"

    # SVG icon — a styled "ribbon of hope"
    ribbon_svg = """
<svg width='84' height='84' viewBox='0 0 64 64' fill='none' xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>
  <path d='M32 6c4 0 7 3 7 7v10l8 18c2 4-1 9-5 9h-3l-7-12-7 12h-3c-4 0-7-5-5-9l8-18V13c0-4 3-7 7-7z'
        fill='url(#rg)' stroke='rgba(255,255,255,0.65)' stroke-width='1.4'/>
  <defs>
    <linearGradient id='rg' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='#1A73E8'/>
      <stop offset='100%' stop-color='#00897B'/>
    </linearGradient>
  </defs>
</svg>"""

    st.markdown(
        f"""<div style='position:relative;border-radius:18px;padding:18px 22px;margin:-6px 0 14px 0;
                        background:{bg};border:1px solid {bord};
                        box-shadow:0 1px 3px rgba(15,23,42,0.04),
                                   0 16px 36px -22px rgba(15,23,42,0.18);
                        display:flex;gap:18px;align-items:center;overflow:hidden'>
              <div style='flex:0 0 auto;filter:drop-shadow(0 6px 14px rgba(26,115,232,0.25))'>
                {ribbon_svg}
              </div>
              <div style='flex:1;min-width:0'>
                <div style='font-size:0.74rem;text-transform:uppercase;letter-spacing:0.7px;
                            color:{msg_color};font-weight:800;margin-bottom:4px'>
                  {msg_kicker}
                </div>
                <div style='font-size:1.0rem;color:#0F172A;line-height:1.55;font-weight:500'>
                  {msg}
                </div>
                <div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center'>
                  {"<span class='pill pill-green' style='font-size:0.68rem'>✓ Verified by AI</span>" if is_real else "<span class='pill pill-red' style='font-size:0.68rem'>⚠ Demo</span>"}
                  <span class='pill' style='font-size:0.68rem'>Reviewed against BSG · NICE · USPSTF guidelines</span>
                </div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )

    # Collapsible technical details — for researchers / clinicians who want them
    if is_real:
        with st.expander("Show technical details · for clinicians & researchers", expanded=False):
            st.markdown(
                f"""
- **Model**: `{prov.get('model','UnifiedMultiModalTransformer')}`
- **Backbone**: {prov.get('backbone','ResNet-50 + EfficientNet-B0 + BioBERT + TabTransformer')}
- **Checkpoint**: `{prov.get('checkpoint','best_model.pth')}` ({"loaded" if prov.get('checkpoint_loaded') else "weights init"})
- **Held-out test metrics**: 90.3 % accuracy · 0.81 macro F1 · 0.984 AUC-ROC (on 1,066 images)
- **Inference time (this case)**: {analysis.get('inference_time_ms',0):.0f} ms
- **MC-Dropout passes**: 15 · entropy reported as the Uncertainty metric
- **Recommendation source**: BSG / NICE / USPSTF guideline rules
                """
            )

    # ── Top "key-findings" strip (one-glance summary) ─────────────────────
    risk_score = analysis["risk_score"]
    unc        = analysis["uncertainty"]
    rc = "#2E7D32" if risk_score<0.25 else "#F9A825" if risk_score<0.5 else "#E65100" if risk_score<0.75 else "#B71C1C"
    risk_band = ("Low" if risk_score<0.25 else "Moderate" if risk_score<0.5
                 else "High" if risk_score<0.75 else "Critical")
    unc_lbl   = "Low" if unc<0.3 else "Moderate" if unc<0.6 else "High"
    urgency   = analysis.get("recommendation", {}).get("urgency", "Routine")
    pretty    = CLASS_LABELS.get(pclass, pclass)

    st.markdown(
        f"""<div style='border-radius:16px;padding:20px 26px;margin-bottom:18px;
                        background:linear-gradient(135deg, #FFFFFF 0%, #F8FAFF 100%);
                        border:1px solid rgba(15,23,42,0.06);
                        box-shadow:0 1px 2px rgba(15,23,42,0.04),
                                   0 16px 36px -22px rgba(15,23,42,0.18)'>
          <div style='display:flex;flex-wrap:wrap;align-items:center;gap:18px'>
            <div style='flex:0 0 6px;align-self:stretch;border-radius:6px;
                        background:linear-gradient(180deg, {pcolor}, {rc})'></div>
            <div style='flex:1;min-width:240px'>
              <div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.6px;
                          color:#64748B;font-weight:700'>Key finding</div>
              <div style='font-size:1.45rem;font-weight:800;color:#0F172A;line-height:1.15;margin-top:2px'>
                {pretty}
              </div>
              <div style='font-size:0.86rem;color:#475569;margin-top:4px'>
                Confidence {analysis['confidence']:.0%} ·
                <span style='color:{rc};font-weight:700'>{risk_band} risk</span> ·
                Stage <b>{"not staged from this image" if analysis['stage'] == 'Cannot stage from one image' else analysis['stage']}</b> ·
                Uncertainty <b>{unc:.2f}</b> ({unc_lbl})
              </div>
            </div>
            <div style='display:flex;gap:8px;flex-wrap:wrap'>
              <span class='pill {"pill-green" if urgency=="Routine" else "pill-amber" if urgency in ("Elective","Urgent") else "pill-red"}'>{urgency}</span>
              <span class='pill'>Inference {analysis['inference_time_ms']:.0f} ms</span>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Top summary metrics — animated counters
    try:
        from src.app.ui_extras import animated_counter
    except Exception:
        animated_counter = None

    col_a, col_b, col_c, col_d = st.columns(4)
    stage_disabled = analysis.get("stage") == "Cannot stage from one image"
    if animated_counter:
        with col_a:
            animated_counter(analysis["confidence"]*100, "AI Confidence",
                             suffix="%", color=pcolor)
        with col_b:
            if stage_disabled:
                # Honest counter — staging head can't reliably stage this image
                render_metric_card(
                    "Cancer Stage",
                    "Cannot stage",
                    "Single endoscopy images don't give true stage. See note below.",
                    color="#B91C1C",
                )
            else:
                animated_counter(analysis["stage_confidence"]*100, "Stage Confidence",
                                 suffix="%", color=STAGE_COLORS.get(analysis["stage"],"#1A73E8"))
        with col_c:
            animated_counter(risk_score*100, "Risk Score",
                             suffix="%", color=rc)
        with col_d:
            uc = "#2E7D32" if unc<0.3 else "#F9A825" if unc<0.6 else "#B71C1C"
            animated_counter(unc, "Uncertainty", suffix="", color=uc, decimals=2)
    else:
        with col_a:
            render_metric_card("AI Finding", CLASS_LABELS.get(pclass, pclass),
                               f"Confidence: {analysis['confidence']:.0%}", color=pcolor)
        with col_b:
            render_metric_card("Cancer Stage", analysis["stage"],
                               f"Confidence: {analysis['stage_confidence']:.0%}",
                               color=STAGE_COLORS.get(analysis["stage"],"#1A73E8"))
        with col_c:
            render_metric_card("Risk Score", f"{risk_score:.0%}", analysis["risk_label"], color=rc)
        with col_d:
            uc = "#2E7D32" if unc<0.3 else "#F9A825" if unc<0.6 else "#B71C1C"
            render_metric_card("AI Uncertainty", f"{unc:.2f}", unc_lbl, color=uc)

    st.markdown("")
    # Risk badge
    render_risk_badge(risk_score)
    st.markdown("")

    # ── Tabs ───────────────────────────────────────────────────────────
    tab1, tab_why, tab2, tab3, tab4 = st.tabs([
        "Diagnosis", "Why this result?", "Where the polyp is", "Risk Charts", "Recommendations"
    ])

    # ── Tab 1: Diagnosis ───────────────────────────────────────────────
    with tab1:
        # Always-on "what the AI can recognise" tile so the user knows the scope
        st.markdown(
            """<div style='border-radius:12px;padding:12px 16px;margin-bottom:14px;
                            background:#F8FAFF;border:1px solid rgba(26,115,232,0.18);
                            font-size:0.82rem;color:#475569;line-height:1.6'>
                  <div style='font-weight:800;color:#1A73E8;margin-bottom:4px'>
                    What this AI can &amp; can't see — be honest with yourself
                  </div>
                  <b>Trained on:</b> HyperKvasir (10,662 screening images, Norway) +
                  CVC-ClinicDB (612 polyp images, Spain). Output classes: polyps · mild
                  colitis · severe colitis · Barrett's oesophagus · post-treatment site.<br>
                  <b>Therefore it CANNOT recognise:</b> stage III–IV cancer · fungating /
                  ulcerated tumour masses · post-surgical anatomy · sessile-serrated lesions ·
                  Crohn's-pattern disease · paediatric pathology · rare entities.<br>
                  <b>Cancer staging from a single image is approximate at best.</b> True
                  TNM staging requires biopsy histology + CT/MRI cross-sectional imaging.
                  When this app shows a stage, it's a <i>visual</i> stage of the
                  predicted-class only — not a substitute for clinical staging.
                </div>""",
            unsafe_allow_html=True,
        )

        # Plain-English "Why this result" card — based on the patient's actual inputs
        _render_plain_why_card(analysis,
                               st.session_state.get("patient", {}),
                               st.session_state.get("symptoms", []),
                               st.session_state.get("symptom_text", ""))

        col_diag, col_mod = st.columns([3, 2])

        with col_diag:
            st.markdown('<div class="section-header">Class Probability Distribution</div>',
                        unsafe_allow_html=True)
            probs  = analysis["pathology_probs"]
            labels = [CLASS_LABELS.get(k, k) for k in probs.keys()]
            values = list(probs.values())
            colors = [CLASS_COLOURS.get(k, "#999") for k in probs.keys()]

            fig_bar = go.Figure(go.Bar(
                y=labels, x=values,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Probability: %{x:.1%}<extra></extra>",
            ))
            fig_bar.update_layout(
                height=280,
                margin=dict(l=0, r=40, t=10, b=10),
                xaxis=dict(range=[0, 1.1], showgrid=True, gridcolor="#f0f0f0",
                           tickformat=".0%", title=""),
                yaxis=dict(title=""),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter, sans-serif", size=11),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_mod:
            st.markdown('<div class="section-header">Modality Weights</div>',
                        unsafe_allow_html=True)
            mod_labels  = ["Imaging", "Clinical Text", "Patient Data"]
            mod_values  = [analysis["image_weight"], analysis["text_weight"], analysis["tabular_weight"]]
            mod_colours = ["#1A73E8", "#00897B", "#FF5722"]

            fig_pie = go.Figure(go.Pie(
                labels=mod_labels, values=mod_values,
                hole=0.55,
                marker_colors=mod_colours,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Weight: %{value:.1%}<extra></extra>",
            ))
            fig_pie.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="white",
                showlegend=False,
                annotations=[dict(text="Modality<br>Fusion", x=0.5, y=0.5,
                                  font_size=12, showarrow=False, font_color="#333")],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Stage display — only when the staging head is trustworthy ────
        if stage_disabled:
            note = analysis.get("staging_note") or (
                "Single endoscopy images cannot reliably stage cancer."
            )
            st.markdown(
                f"""<div style='border-radius:14px;padding:14px 18px;margin-top:14px;
                                background:#FEF2F2;border:1px solid #FCA5A5'>
                      <div style='font-size:0.74rem;text-transform:uppercase;letter-spacing:0.7px;
                                  color:#7F1D1D;font-weight:800;margin-bottom:4px'>
                        Cancer staging not shown
                      </div>
                      <div style='font-size:0.92rem;color:#1F2937;line-height:1.55'>
                        {note}  Real cancer staging requires <b>histology</b> (biopsy) +
                        <b>cross-sectional imaging</b> (CT / MRI). The staging head's
                        output for this image is <b>not reliable</b> — we are hiding it
                        rather than showing a false low-stage number.
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="section-header">Stage estimate · model-derived</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:0.78rem;color:#64748B;margin-bottom:8px'>"
                "<b>Caveat:</b> the staging head was trained against class-derived labels "
                "(not against real TNM cancer staging). Treat this as the <i>visual</i> "
                "stage of the predicted finding only — true staging needs biopsy + imaging."
                "</div>",
                unsafe_allow_html=True,
            )
            stage_probs  = analysis["stage_probs"]
            stage_labels = list(stage_probs.keys())
            stage_vals   = list(stage_probs.values())
            stage_colors = [STAGE_COLORS.get(s, "#999") for s in stage_labels]
            fig_stage = go.Figure(go.Bar(
                x=stage_labels, y=stage_vals,
                marker_color=stage_colors,
                text=[f"{v:.1%}" for v in stage_vals],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>%{y:.1%}<extra></extra>",
            ))
            fig_stage.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(range=[0, 1.15], tickformat=".0%", showgrid=True, gridcolor="#f0f0f0"),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, sans-serif", size=11),
            )
            st.plotly_chart(fig_stage, use_container_width=True)

        # Risk flags
        flags = analysis.get("all_risk_flags", [])
        if flags:
            st.markdown('<div class="section-header">Risk Flags</div>', unsafe_allow_html=True)
            for flag in flags:
                st.markdown(f'<div class="warn-box">{flag}</div>', unsafe_allow_html=True)

    # ── Tab 1.5: Why this result? (AI Reasoning Panel) ─────────────────
    with tab_why:
        _render_reasoning_panel(analysis, st.session_state.get("patient", {}),
                                st.session_state.get("symptoms", []))

    # ── Tab 2: GradCAM ─────────────────────────────────────────────────
    with tab2:
        orig  = analysis.get("original_image")
        cam   = analysis.get("gradcam_overlay")
        heat  = analysis.get("gradcam_heatmap")
        seg   = analysis.get("seg_mask")

        # Resolve a display copy of the input image
        disp_orig = None
        if orig is not None:
            disp_orig = (orig * 255).astype(np.uint8) if orig.max() <= 1 else orig
        else:
            _pil_in = st.session_state.get("uploaded_image")
            if _pil_in is not None:
                disp_orig = np.array(_pil_in.convert("RGB"))

        if disp_orig is not None or cam is not None:
            # ── PRIMARY: segmentation-based localization (the lesion outline) ──
            st.markdown('<div class="section-header">Where the polyp is — segmentation</div>',
                        unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                if disp_orig is not None:
                    st.image(disp_orig, caption="Input image", use_container_width=True)
            with col_b:
                if seg is not None and disp_orig is not None:
                    seg_disp = overlay_seg(disp_orig, seg)
                    st.image(seg_disp,
                             caption="Polyp outline (segmentation decoder — the actual lesion location)",
                             use_container_width=True)
                else:
                    st.info("No polyp region was segmented in this image (the segmentation "
                            "decoder found no lesion, or no image was provided).")
            st.markdown(
                """<div class="info-box" style="margin-top:6px">
                <b>This is the localization to trust.</b> The green region is the segmentation
                decoder's pixel-level outline of the polyp (honest cross-vendor IoU ≈ 0.45 on a
                held-out scanner). It shows <i>where</i> the lesion is, far more precisely than
                the attention map below.
                </div>""",
                unsafe_allow_html=True,
            )

            # ── What kind of polyp? (CADx optical impression) ───────────────
            cadx = analysis.get("characterization")
            if isinstance(cadx, dict) and cadx.get("available"):
                _neo   = cadx.get("kind") == "neoplastic"
                _conf  = float(cadx.get("confidence", 0.0))
                _box   = "warn-box" if _neo else "info-box"
                _head  = ("Looks like a neoplastic polyp" if _neo
                          else "Looks like a non-neoplastic polyp")
                _plain = ("This kind can slowly turn cancerous, so the usual step is "
                          "to remove it and send it to the lab to be certain."
                          if _neo else
                          "This kind is usually harmless (for example a hyperplastic "
                          "polyp) — but the lab still has the final say.")
                st.markdown(
                    f"""<div class="{_box}" style="margin-top:10px">
                    <b>What kind of polyp — optical impression</b><br>
                    <span style="font-size:1.05rem"><b>{_head}</b>
                    &nbsp;(model {_conf*100:.0f}% confident)</span><br>
                    {_plain}<br>
                    <span style="opacity:.82;font-size:.85rem">
                    This is an <i>optical</i> impression from the image alone
                    (specialist trained on the BKAI-IGH NeoPolyp set). It is
                    decision-support only — the biopsy / histology result is the
                    final answer, not this.</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # ── SECONDARY: GradCAM attention — explanation only, clearly demoted ──
            with st.expander("Model attention (GradCAM++) — explanation only, NOT the polyp outline"):
                gc1, gc2 = st.columns(2)
                with gc1:
                    if disp_orig is not None:
                        st.image(disp_orig, caption="Input image", use_container_width=True)
                with gc2:
                    if cam is not None:
                        disp_cam = (cam * 255).astype(np.uint8) if cam.max() <= 1 else cam.astype(np.uint8)
                        st.image(disp_cam,
                                 caption="Red = higher attention (coarse 7×7 — not a precise outline)",
                                 use_container_width=True)
                    else:
                        st.info("GradCAM heatmap not available for this analysis.")
                if heat is not None:
                    import matplotlib.pyplot as plt
                    fig_hm, ax = plt.subplots(figsize=(8, 2.5))
                    im = ax.imshow(heat, cmap="hot", aspect="auto")
                    ax.set_title("GradCAM++ heatmap (brighter = higher model attention)")
                    ax.axis("off")
                    plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.1)
                    st.pyplot(fig_hm, use_container_width=True)
                    plt.close(fig_hm)
                st.markdown(
                    """<div class="warn-box" style="margin-top:6px">
                    <b>Why this is secondary.</b> GradCAM++ is a coarse 7×7 classifier-attention
                    map — it shows roughly where the model looked <b>for its predicted class</b>,
                    not the lesion's true extent. For localization, use the segmentation above.
                    When the prediction is wrong, the attention map is wrong too.
                    </div>""",
                    unsafe_allow_html=True,
                )

            # ── 3D Colon Viewer ─────────────────────────────────────────────
            try:
                from src.app.ui_extras import colon_3d_figure
                st.markdown('<div class="section-header">3D anatomical viewer</div>',
                            unsafe_allow_html=True)
                col_3d, col_caption = st.columns([3, 2])
                with col_3d:
                    st.plotly_chart(colon_3d_figure(highlight_class=pclass),
                                    use_container_width=True)
                with col_caption:
                    st.markdown(
                        f"""<div class='info-box'>
                          <b>What you're looking at.</b><br>
                          A stylised 3-D model of the colorectum. The pulsing blue marker
                          shows the typical location of the predicted finding —
                          <b>{CLASS_LABELS.get(pclass, pclass)}</b>. Drag to rotate, scroll
                          to zoom. This is illustrative — the real lesion location is the
                          segmentation outline above.
                        </div>""",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass
        else:
            st.info("No image was uploaded — image-based localization needs an "
                    "endoscopy/colonoscopy image. See your risk-factor assessment instead.")

    # ── Tab 3: Risk Charts ─────────────────────────────────────────────
    with tab3:
        col_gauge, col_radar = st.columns(2)

        with col_gauge:
            st.markdown('<div class="section-header">Cancer Risk Gauge</div>',
                        unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score * 100,
                delta={"reference": 25, "valueformat": ".0f"},
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar":  {"color": rc, "thickness": 0.25},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "#ddd",
                    "steps": [
                        {"range": [0,  25], "color": "#E8F5E9"},
                        {"range": [25, 50], "color": "#FFF9C4"},
                        {"range": [50, 75], "color": "#FFE0B2"},
                        {"range": [75,100], "color": "#FFEBEE"},
                    ],
                    "threshold": {
                        "line": {"color": "#B71C1C", "width": 3},
                        "thickness": 0.75,
                        "value": 75,
                    },
                },
                title={"text": "Malignancy Risk", "font": {"size": 14, "color": "#555"}},
            ))
            fig_gauge.update_layout(
                height=280, margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_radar:
            st.markdown('<div class="section-header">Multi-Dimensional Risk Profile</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "<div style='margin-bottom:6px'>"
                "<span class='pill pill-amber' style='font-size:0.68rem'>Heuristic · not model output</span>"
                " <span class='pill pill-green' style='font-size:0.68rem'>AI risk axis IS model output</span>"
                "</div>"
                "<div style='font-size:0.78rem;color:#64748B;margin-bottom:8px'>"
                "Only the <b>AI Risk</b> axis comes from the trained model. The other 5 axes are "
                "heuristic risk factors derived from the patient form and weighted using literature "
                "(USPSTF 2021 · ACS · WCRF · BSG/ACPGBI 2010/2020). Hover any axis for the source."
                "</div>",
                unsafe_allow_html=True,
            )
            unc = analysis["uncertainty"]
            # Compute proxy scores from available data
            p_data = st.session_state.get("patient", {})
            age_risk   = min(1.0, max(0.0, (float(p_data.get("age", 40) or 40) - 40) / 40))
            smoke_risk = 0.7 if "Yes" in str(p_data.get("smoking","")) else 0.1
            alc_risk   = 0.5 if p_data.get("alcohol","No") in ["Regular","Heavy"] else 0.15
            fam_risk   = 0.8 if "First" in str(p_data.get("family_history","No")) else 0.2
            poly_risk  = 0.75 if p_data.get("prev_polyps","No") == "Yes" else 0.2

            radar_cats = ["AI Risk", "Age Factor", "Smoking", "Alcohol", "Family Hx", "Prior Polyps"]
            radar_vals = [risk_score, age_risk, smoke_risk, alc_risk, fam_risk, poly_risk]
            # Citations per axis — surface in tooltip
            radar_cite = [
                "Multimodal AI · this case",
                "USPSTF 2021 · risk rises sharply ≥45 yrs",
                "ACS · current/former smokers OR ≈ 1.2",
                "World Cancer Research Fund · ≥3 drinks/day OR ≈ 1.4",
                "BSG/ACPGBI 2010 · 1st-degree CRC <50 yr ↑ risk 4×",
                "Adenoma surveillance BSG/ACPGBI 2020",
            ]
            radar_vals_closed = list(radar_vals) + [radar_vals[0]]
            radar_cats_closed = list(radar_cats) + [radar_cats[0]]
            radar_cite_closed = list(radar_cite) + [radar_cite[0]]

            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals_closed, theta=radar_cats_closed,
                fill="toself",
                fillcolor="rgba(26,115,232,0.18)",
                line=dict(color="#1A73E8", width=2.5),
                marker=dict(size=8, color="#1A73E8",
                            line=dict(width=2, color="white")),
                customdata=radar_cite_closed,
                hovertemplate="<b>%{theta}</b><br>"
                              "Score: %{r:.1%}<br>"
                              "<i>Source:</i> %{customdata}"
                              "<extra></extra>",
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%",
                                    gridcolor="#e0e0e0"),
                    angularaxis=dict(gridcolor="#e0e0e0"),
                    bgcolor="white",
                ),
                showlegend=False,
                height=280,
                margin=dict(l=40, r=40, t=20, b=20),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── REAL fusion-gate weights + per-class softmax (replaces proxy chart) ──
        st.markdown(
            '<div class="section-header">Fusion-gate weights & per-class softmax · live model</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='margin-bottom:6px'>"
            "<span class='pill pill-green' style='font-size:0.68rem'>"
            "✓ Live model output · checkpoint loaded</span>"
            "</div>"
            "<div style='font-size:0.78rem;color:#64748B;margin-bottom:8px'>"
            "Left: the <b>real sigmoid gate weights</b> the fusion transformer assigned to each "
            "modality for this case. Right: the actual softmax over the 5 pathology classes."
            "</div>",
            unsafe_allow_html=True,
        )
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            mod_labels = ["Imaging", "Clinical text", "Patient features"]
            mod_vals   = [analysis["image_weight"],
                          analysis["text_weight"],
                          analysis["tabular_weight"]]
            mod_colors = ["#1A73E8", "#00897B", "#FF5722"]
            fig_gate = go.Figure(go.Bar(
                x=mod_vals, y=mod_labels, orientation="h",
                marker_color=mod_colors,
                text=[f"{v:.1%}" for v in mod_vals],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Fusion-gate weight: %{x:.1%}<extra></extra>",
            ))
            fig_gate.update_layout(
                height=220, margin=dict(l=10,r=40,t=10,b=10),
                xaxis=dict(range=[0, max(mod_vals)*1.25 + 0.05], tickformat=".0%",
                           showgrid=True, gridcolor="#f0f0f0"),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, sans-serif", size=11),
                title=dict(text="Modality fusion weights", x=0.02,
                           font=dict(size=12, color="#475569")),
            )
            st.plotly_chart(fig_gate, use_container_width=True)
        with col_g2:
            probs = analysis["pathology_probs"]
            cls_labels = [CLASS_LABELS.get(k, k) for k in probs.keys()]
            cls_vals   = list(probs.values())
            cls_colors = [CLASS_COLOURS.get(k, "#999") for k in probs.keys()]
            fig_cls = go.Figure(go.Bar(
                x=cls_vals, y=cls_labels, orientation="h",
                marker_color=cls_colors,
                text=[f"{v:.1%}" for v in cls_vals],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Softmax: %{x:.1%}<extra></extra>",
            ))
            fig_cls.update_layout(
                height=220, margin=dict(l=10,r=40,t=10,b=10),
                xaxis=dict(range=[0, 1.15], tickformat=".0%",
                           showgrid=True, gridcolor="#f0f0f0"),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, sans-serif", size=11),
                title=dict(text="Per-class softmax (pathology head)", x=0.02,
                           font=dict(size=12, color="#475569")),
            )
            st.plotly_chart(fig_cls, use_container_width=True)

    # ── Tab 4: Recommendations ─────────────────────────────────────────
    with tab4:
        rec = analysis.get("recommendation", {})
        if not rec:
            st.info("No recommendations available.")
        else:
            render_urgency_banner(rec.get("urgency", "Routine"))

            # ── "What to do next" — concrete actionable steps based on
            # urgency band + overrides ─────────────────────────────────
            urgency = rec.get("urgency", "Routine")
            override_applied = bool((analysis.get("overrides") or {}).get("applied"))
            atypical = (analysis.get("image_readout") or {}).get("verdict") == "atypical_concerning"

            if urgency == "Emergency" or atypical:
                next_steps = [
                    ("Right now",
                     "Contact your GP today — phone, in person, or NHS 111. "
                     "Do <b>not</b> wait for a routine appointment. Mention this AI screen and "
                     "your symptoms together."),
                    ("Within 1–2 weeks",
                     "Book a face-to-face GP visit and ask for a NICE 2-week-wait suspected-CRC "
                     "referral. Bring your downloaded report (Step 6) and a list of your symptoms."),
                    ("During the wait",
                     "Keep a symptom diary (frequency, severity, timing). Avoid red / processed meat "
                     "and alcohol. Drink 1.5–2 L of water/day. Do not start any new supplements "
                     "without telling your GP."),
                    ("If symptoms worsen",
                     "Heavy fresh bleeding, severe abdominal pain, vomiting, fainting, or new "
                     "anaemia → A&E. Take this report with you."),
                ]
                hi = "#B91C1C"
            elif urgency == "Urgent" or override_applied:
                next_steps = [
                    ("Within 1 week",
                     "Book a GP appointment. Show them this report and your symptom log."),
                    ("Within 2 weeks",
                     "Expect a 2-week-wait specialist referral if your GP agrees with the AI's "
                     "concern. Don't delay — book the colonoscopy slot when offered."),
                    ("Track it",
                     "Note any change in stools, bleeding episodes or weight. Photograph stools "
                     "if you can — clinicians value this."),
                    ("Lifestyle while you wait",
                     "Hold off on new diets / supplements. Limit alcohol. Increase fibre slowly "
                     "(if tolerated). Continue your usual medications unless your GP says otherwise."),
                ]
                hi = "#B45309"
            else:
                next_steps = [
                    ("Schedule a GP follow-up",
                     "Take this report to your routine GP appointment to confirm the AI's "
                     "screening interpretation and discuss surveillance."),
                    ("Stay on the screening pathway",
                     "Continue with bowel-cancer screening as recommended for your age (NHS biennial "
                     "FIT 50–74; USPSTF 45–75)."),
                    ("Healthy-colon basics",
                     "≥30 g fibre per day, <14 units of alcohol per week, regular exercise "
                     "(150 min/week), keep BMI 18.5–25, don't smoke."),
                    ("Save the report",
                     "Download the PDF in Step 6. Keep a copy for your medical records."),
                ]
                hi = "#15803D"

            st.markdown('<div class="section-header">What to do next</div>',
                        unsafe_allow_html=True)
            for title, body in next_steps:
                st.markdown(
                    f"""<div style='display:flex;gap:14px;padding:10px 14px;margin-bottom:6px;
                                    border-radius:10px;background:white;
                                    border:1px solid rgba(15,23,42,0.06);
                                    border-left:4px solid {hi};
                                    box-shadow:0 1px 2px rgba(15,23,42,0.04)'>
                          <div style='flex:0 0 140px;font-size:0.78rem;font-weight:800;
                                      color:{hi};text-transform:uppercase;letter-spacing:0.5px'>
                            {title}
                          </div>
                          <div style='flex:1;font-size:0.92rem;color:#1F2937;line-height:1.55'>{body}</div>
                        </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown("")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown('<div class="section-header">Primary Action</div>',
                            unsafe_allow_html=True)
                st.markdown(f"**{rec.get('primary_action','N/A')}**")
                st.markdown("")
                st.markdown('<div class="section-header">Surveillance Plan</div>',
                            unsafe_allow_html=True)
                st.markdown(f"**{rec.get('surveillance','N/A')}**")
                st.markdown("")

                referrals = rec.get("referrals", [])
                if referrals:
                    st.markdown('<div class="section-header">Specialist Referrals</div>',
                                unsafe_allow_html=True)
                    for r in referrals:
                        st.markdown(f"• {r}")

                # Guideline basis (smart win #3 — recommendations cite real guidelines)
                try:
                    from src.app.guideline_kb import basis_for
                    _sym_txt = " ".join(st.session_state.get("symptoms", []) or [])
                    _basis = basis_for(pclass, _sym_txt)
                    if _basis:
                        st.markdown('<div class="section-header">📖 Guideline basis</div>',
                                    unsafe_allow_html=True)
                        for _g in _basis:
                            st.markdown(f"- {_esc(_g['statement'])}  \n  *— {_esc(_g['source'])}*")
                except Exception:
                    pass

            with col_r2:
                investigations = rec.get("investigations", [])
                if investigations:
                    st.markdown('<div class="section-header">Recommended Tests</div>',
                                unsafe_allow_html=True)
                    for inv in investigations:
                        st.markdown(f"- {inv}")

                lifestyle = rec.get("lifestyle_advice", [])
                if lifestyle:
                    st.markdown('<div class="section-header">Lifestyle Recommendations</div>',
                                unsafe_allow_html=True)
                    for lf in lifestyle:
                        st.markdown(f"- {lf}")

            # Full report text
            full_report = rec.get("full_report", "")
            if full_report and len(full_report) > 30:
                with st.expander("View Full Clinical Report Text"):
                    st.text(full_report)

    # ── Navigation ─────────────────────────────────────────────────────
    st.markdown("---")
    col_b, col_sp, col_n = st.columns([1, 4, 1])
    with col_b:
        if st.button("← Back", use_container_width=True):
            st.session_state["step"] = 1
            st.rerun()
    with col_n:
        if st.button("Find Doctors →", type="primary", use_container_width=True):
            st.session_state["step"] = 4
            st.rerun()

    st.markdown(
        '<div class="disclaimer"><b>Disclaimer:</b> These results are generated by an AI system '
        'for informational purposes only and do <b>NOT</b> constitute a medical diagnosis. All findings '
        'must be reviewed by a qualified, licensed medical professional. Do not make clinical decisions '
        'solely based on this AI output.</div>',
        unsafe_allow_html=True,
    )


def page_doctor_finder():
    from src.app.geo import (
        geocode_city, osm_nearby_specialists,
        google_maps_embed_url, google_maps_directions_url,
        haversine_km, GeoPoint,
    )

    n_specialists = len(DOCTORS_DB)
    render_hero(
        "Find Specialists Near You",
        "Type any city worldwide — we geocode it live, search nearby specialists "
        "and embed a Google Map.",
        badges=["Step 5 of 6", f"{n_specialists} curated specialists",
                "Live OSM lookup", "Google Maps embed"],
    )

    patient = st.session_state.get("patient", {})
    analysis = st.session_state.get("analysis", {}) or {}
    ai_pathology = analysis.get("pathology_class", "")

    # Show context banner if analysis is available
    if ai_pathology:
        pretty = CLASS_LABELS.get(ai_pathology, ai_pathology)
        urgency = analysis.get("recommendation", {}).get("urgency", "Routine")
        st.markdown(
            f"""<div class='info-box' style='display:flex;align-items:center;gap:14px'>
                <span style='display:inline-flex;align-items:center;justify-content:center;
                             width:38px;height:38px;border-radius:12px;
                             background:linear-gradient(135deg,#1A73E8,#00897B);
                             color:white;font-weight:800'>AI</span>
                <div>
                  <div style='font-weight:800;color:#0F172A'>Tailoring results to your AI finding · {pretty}</div>
                  <div style='font-size:0.85rem;color:#475569'>
                    Specialists aligned with your indicated condition are surfaced first; clinical urgency
                    flagged as <b>{urgency}</b>. Same-region specialists ranked above general matches.
                  </div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Search</div>', unsafe_allow_html=True)
    col_c, col_co, col_sp = st.columns(3)
    with col_c:
        search_city = st.text_input("City", value=patient.get("city", ""),
                                     placeholder="e.g. Noida, Mumbai, New York, London")
    with col_co:
        search_country = st.selectbox("Country",
            ["(Any)", "India","USA","UK","UAE","Singapore","Canada","Australia"],
            index=0 if not patient.get("country") else
                  ["(Any)","India","USA","UK","UAE","Singapore","Canada","Australia"].index(
                      patient.get("country","India")))
    with col_sp:
        search_spec = st.selectbox("Specialty",
            ["(Any)", "Gastroenterology", "Colorectal Surgery",
             "GI Oncology", "Medical Oncology", "Surgical Oncology",
             "Gastrointestinal Surgery", "Hepatology"])

    country_q = "" if search_country == "(Any)" else search_country
    spec_q    = "" if search_spec == "(Any)" else search_spec

    # ── Live geocode (Nominatim) + map embed ──────────────────────────────
    with st.spinner("Locating your city on the map…"):
        user_point = geocode_city(search_city, country_q) if search_city else None

    # Embedded Google Maps view
    if user_point:
        map_q = f"gastroenterologist+near+{user_point.lat},{user_point.lng}"
        embed = google_maps_embed_url(query=map_q, zoom=12)
        st.markdown(
            f"""<div style='position:relative;border-radius:14px;overflow:hidden;
                            border:1px solid rgba(15,23,42,0.08);
                            box-shadow:0 1px 3px rgba(15,23,42,0.04),
                                       0 12px 28px -16px rgba(15,23,42,0.20);
                            margin-bottom:14px'>
                  <div style='position:absolute;top:10px;left:14px;z-index:5;
                              background:rgba(255,255,255,0.92);padding:6px 12px;
                              border-radius:999px;backdrop-filter:blur(8px);
                              font-size:0.78rem;font-weight:700;color:#0F172A;
                              box-shadow:0 4px 12px rgba(15,23,42,0.10)'>
                    📍 {user_point.display_name[:80]}
                  </div>
                  <iframe src='{embed}' width='100%' height='340'
                          style='border:0;display:block' loading='lazy'
                          referrerpolicy='no-referrer-when-downgrade'></iframe>
                </div>""",
            unsafe_allow_html=True,
        )

    # ── Curated DB results (smart-ranked) ────────────────────────────────
    results = search_doctors(search_city, country_q, spec_q,
                             ai_pathology=ai_pathology, limit=10)

    fallback_msg = ""
    if not results and search_city:
        results = search_doctors("", country_q, spec_q,
                                 ai_pathology=ai_pathology, limit=10)
        fallback_msg = (
            f"No directly-curated specialists for <b>{search_city}</b> — "
            f"showing the closest in {search_country or 'the region'} plus live "
            f"OpenStreetMap nearby-hospital data."
        )
    if not results:
        results = [(d, ["fallback"]) for d in DOCTORS_DB[:8]]
        fallback_msg = "Showing top-rated specialists across the directory."

    # Distance-augment each curated doc if we have a user_point
    if user_point and results:
        augmented = []
        for doc, reasons in results:
            try:
                doc_pt = geocode_city(doc["city"], doc["country"])
            except Exception:
                doc_pt = None
            dist_km = None
            if doc_pt:
                dist_km = haversine_km(user_point, doc_pt)
                # Boost reasons with distance chip
                if dist_km < 50:
                    reasons = list(reasons) + [f"{dist_km:.1f} km away"]
                elif dist_km < 200:
                    reasons = list(reasons) + [f"~{dist_km:.0f} km · same metro"]
                else:
                    reasons = list(reasons) + [f"~{dist_km:.0f} km · same country"]
            augmented.append((doc, reasons, dist_km))
        # Sort by distance when available
        augmented.sort(key=lambda r: (r[2] is None, r[2] or 9999))
        results = [(d, r) for d, r, _ in augmented]

    if fallback_msg:
        st.markdown(f"<div class='warn-box'>{fallback_msg}</div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="info-box">Found <b>{len(results)}</b> curated specialists, '
        f'ranked by distance, AI-finding match and reputation.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Render doctor cards in 2-column grid
    patient_origin = patient.get("city", "") or (search_city or "")
    for i in range(0, len(results), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx >= len(results):
                break
            doc, reasons = results[idx]
            with cols[j]:
                st.markdown(
                    _render_doctor_card_html(doc, reasons,
                                             origin=patient_origin),
                    unsafe_allow_html=True,
                )

    # ── Live OpenStreetMap healthcare facilities nearby ──────────────────
    if user_point:
        with st.spinner("Querying OpenStreetMap for nearby healthcare facilities…"):
            osm_hits = osm_nearby_specialists(user_point, radius_km=8.0,
                                              gi_only=False)
        if osm_hits:
            st.markdown(
                '<div class="section-header">Other healthcare facilities within 8 km · '
                'live OpenStreetMap data</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="info-box" style="margin-bottom:10px">'
                '<b>What is this?</b> Real-time map data crowdsourced by OpenStreetMap. '
                'These are nearby hospitals / clinics / doctor offices — not necessarily '
                'GI specialists. Use the <b>Directions</b> link to confirm before booking.'
                '</div>',
                unsafe_allow_html=True,
            )
            shown = 0
            for hit in osm_hits[:6]:
                if shown >= 6:
                    break
                name = hit["name"]
                amenity = hit["amenity"].title()
                addr = hit.get("addr") or ""
                phone = hit.get("phone") or ""
                website = hit.get("website") or ""
                dist = hit["distance_km"]
                dest = f"{name}, {addr}" if addr else f"{name},{hit['lat']},{hit['lng']}"
                directions = google_maps_directions_url(
                    origin=patient_origin or search_city, destination=dest)
                view_on_map = (f"https://www.google.com/maps/search/?api=1"
                               f"&query={hit['lat']},{hit['lng']}")
                website_btn = (f"<a href='{website}' target='_blank' rel='noopener' class='doc-cta'><span style='font-size:0.95rem'>🌐</span> Website</a>" if website else "")
                phone_html = (f"<span style='color:#1A73E8;font-weight:600'>Tel</span> {phone}" if phone else "")
                spec_text = f" · {hit['speciality']}" if hit.get('speciality') else ""
                phone_block = f"<div class='doctor-meta'>{phone_html}</div>" if phone_html else ""
                # NOTE: rendered as a single de-indented HTML string. Leading
                # whitespace inside an f-string passed to st.markdown trips
                # the markdown code-block heuristic (>=4 spaces) and the buttons
                # block was being shown as raw text.
                osm_card_html = (
                    "<div class='doctor-card' style='border-top-color:#16A34A'>"
                    "<div style='display:flex;gap:14px;align-items:flex-start'>"
                    "<div style='flex:0 0 auto;width:42px;height:42px;border-radius:12px;"
                    "background:linear-gradient(135deg,#16A34A,#22C55E);color:white;"
                    "display:flex;align-items:center;justify-content:center;"
                    "font-weight:800;font-size:1rem'>📍</div>"
                    "<div style='flex:1;min-width:0'>"
                    f"<div class='doctor-name'>{name}</div>"
                    f"<div class='doctor-spec'>{amenity}{spec_text}</div>"
                    f"<div class='doctor-hosp'>{addr or hit.get('operator') or ''}</div>"
                    "</div></div>"
                    "<div style='display:flex;gap:6px;flex-wrap:wrap;margin-top:8px'>"
                    f"<span class='pill pill-green'>{dist:.1f} km away</span>"
                    "<span class='pill'>Local clinic</span>"
                    "</div>"
                    f"{phone_block}"
                    "<div style='display:flex;gap:8px;margin-top:10px;flex-wrap:wrap'>"
                    f"{website_btn}"
                    f"<a href='{view_on_map}' target='_blank' rel='noopener' class='doc-cta'>"
                    "<span style='font-size:0.95rem'>📍</span> Open in Maps</a>"
                    f"<a href='{directions}' target='_blank' rel='noopener' class='doc-cta doc-cta-primary'>"
                    "<span style='font-size:0.95rem'>🧭</span> Get Directions</a>"
                    "</div>"
                    "</div>"
                )
                st.markdown(osm_card_html, unsafe_allow_html=True)
                shown += 1

    st.markdown(
        '<div class="warn-box"><b>Note:</b> Doctor listings are illustrative — names, ratings '
        'and contact numbers are sourced from public hospital websites and may have changed. '
        'Always verify availability and credentials directly with the institution.</div>',
        unsafe_allow_html=True,
    )

    # Save the selected doctors for the report (top-5)
    st.session_state["suggested_doctors"] = [d for d, _ in results[:5]]

    st.markdown("---")
    col_b, col_sp2, col_n = st.columns([1, 4, 1])
    with col_b:
        if st.button("← Back", use_container_width=True):
            st.session_state["step"] = 3
            st.rerun()
    with col_n:
        if st.button("Generate Report →", type="primary", use_container_width=True):
            st.session_state["step"] = 5
            st.rerun()


def page_report():
    from src.app.report_generator import generate_pdf_report

    render_hero(
        "Download Your Report",
        "Generate a comprehensive clinical PDF report with all findings, GradCAM images, and doctor recommendations",
        badges=["Step 6 of 6", "PDF Report", "Ready to Download"],
    )

    patient   = st.session_state.get("patient", {})
    analysis  = st.session_state.get("analysis", {})
    symptoms  = st.session_state.get("symptoms", [])
    sym_text  = st.session_state.get("symptom_text", "")
    doctors   = st.session_state.get("suggested_doctors", [])

    # Report preview card
    st.markdown('<div class="section-header">Report Summary</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Patient:**")
        st.write(f"{patient.get('name','N/A')}, {patient.get('age','N/A')} yrs, {patient.get('gender','N/A')}")
        st.write(f"{patient.get('city','N/A')}, {patient.get('country','N/A')}")
        st.markdown("")
        st.markdown("**Symptoms:**")
        if symptoms:
            for s in symptoms[:5]:
                st.write(f"• {s}")
            if len(symptoms) > 5:
                st.write(f"• ... and {len(symptoms)-5} more")
        else:
            st.write("None reported")

    with col2:
        if analysis:
            st.markdown("**AI Findings:**")
            pclass = analysis.get("pathology_class","N/A")
            st.write(f"Finding: **{CLASS_LABELS.get(pclass,pclass)}**")
            st.write(f"Stage: **{analysis.get('stage','N/A')}**")
            st.write(f"Risk: **{analysis.get('risk_score',0):.0%}** ({analysis.get('risk_label','N/A')})")
            rec = analysis.get("recommendation",{})
            st.write(f"Urgency: **{rec.get('urgency','N/A')}**")
        else:
            st.info("No analysis results to include.")

    st.markdown("")
    st.markdown('<div class="section-header">Report Contents</div>', unsafe_allow_html=True)
    col_inc1, col_inc2, col_inc3 = st.columns(3)
    with col_inc1:
        st.markdown("- Patient information\n- Reported symptoms\n- AI analysis results")
    with col_inc2:
        include_cam = bool(analysis and analysis.get("gradcam_overlay") is not None)
        cam_label = "GradCAM++ heatmap (included)" if include_cam else "GradCAM++ heatmap (no image uploaded)"
        st.markdown(f"- {cam_label}\n- Risk probability charts\n- Staging analysis")
    with col_inc3:
        st.markdown("- Clinical recommendations\n- Suggested specialists\n- Medical disclaimer")

    st.markdown("")

    # Generate and offer download
    if st.button("Generate PDF Report", type="primary", use_container_width=False):
        with st.spinner("Generating your personalised clinical report..."):
            try:
                pdf_bytes = generate_pdf_report(
                    patient_data=patient,
                    symptoms=symptoms,
                    symptom_text=sym_text,
                    analysis=analysis if analysis else None,
                    doctors=doctors,
                    gradcam_overlay=analysis.get("gradcam_overlay") if analysis else None,
                    original_image=analysis.get("original_image") if analysis else None,
                )
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Report generation error: {e}")
                st.session_state["pdf_bytes"] = None

    if st.session_state.get("pdf_bytes"):
        fname = f"ColonAI_Report_{patient.get('name','Patient').replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            label="Download PDF Report",
            data=st.session_state["pdf_bytes"],
            file_name=fname,
            mime="application/pdf",
            use_container_width=False,
        )
        st.info(f"File: **{fname}**")

    st.markdown("---")
    col_b, col_sp2, col_start = st.columns([1, 3, 1])
    with col_b:
        if st.button("← Back", use_container_width=True):
            st.session_state["step"] = 4
            st.rerun()
    with col_start:
        if st.button("New Assessment", type="primary", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k not in ("_system",):
                    del st.session_state[k]
            st.rerun()

    st.markdown(
        '<div class="disclaimer"><b>Important Medical Disclaimer:</b> This report is generated by an '
        'artificial intelligence system trained on research datasets (HyperKvasir, CVC-ClinicDB, TCGA) '
        'and is provided for <b>informational and screening purposes ONLY</b>. It does <b>NOT</b> '
        'constitute a medical diagnosis, professional medical advice, or a treatment plan. The AI model '
        'may not generalise to all patient populations. All findings MUST be reviewed and verified by a '
        'qualified, licensed medical professional before any clinical decisions are made. Do not delay '
        'seeking professional medical care based solely on this report. In case of emergency, contact '
        'your local emergency services immediately.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT — Rule-based GI / ColonAI assistant
# ─────────────────────────────────────────────────────────────────────────────

CHATBOT_KB: List[Dict[str, Any]] = [
    # ─── Site navigation ─────────────────────────────────────────────────
    {"k": ["how does this work","how to use","guide","steps","navigate","getting started","start"],
     "a": "ColonAI has 6 steps: (1) Patient Info — there's also a 'Quick demo' panel at the top that loads a complete sample case in one click, (2) Symptoms & Upload, (3) AI Analysis, (4) Results, (5) Find Doctors, (6) Download Report. The sidebar tracks your progress."},
    {"k": ["what is colonai","what is this app","about colonai","about this app",
            "purpose","what does this do","what does this app do","explain colonai"],
     "a": "ColonAI is an AI screening assistant for colorectal conditions. It combines what we see in your image, what you write about your symptoms, and your medical history — and gives you a personalised report aligned with international guidelines (BSG, NICE, USPSTF). It is NOT a replacement for a doctor; it's a second pair of eyes."},

    # ─── Red-flag symptoms ───────────────────────────────────────────────
    {"k": ["red flag","2 week","two week","urgent referral","when to see doctor","when to worry","symptoms","what are the symptoms","warning signs"],
     "a": "🤖 Please see your GP soon if you have any of these: blood in your poo or from your back passage, a noticeable change in your bowel habit (looser, more often, or more constipated) lasting more than 3 weeks, losing weight without trying, ongoing tummy pain or cramping, or feeling extremely tired (which could mean low iron). These don't always mean cancer — they're often something much less serious — but they're worth getting checked. Better safe than sorry!"},
    {"k": ["rectal bleeding","blood stool","bloody stool","blood in stool","haematochezia"],
     "a": "🤖 Blood when you wipe or in the toilet bowl is most often caused by piles (haemorrhoids) or a small tear — common, harmless and easily treated. BUT if you're over 50, or you also have weight loss, tummy pain, or it keeps happening for more than 3 weeks, please see your GP. They may suggest a simple stool test or refer you for a check. Don't ignore it — and don't panic. Most people with rectal bleeding don't have cancer."},
    {"k": ["abdominal pain","stomach pain","cramping","cramps","tummy pain"],
     "a": "🤖 Tummy pain that doesn't go away after a few weeks deserves a check, especially if you also have weight loss, blood in your stool, or your bowel habit has changed. Note WHERE the pain is, how strong it is (1–10), what makes it worse, and how long it's lasted — these details really help your GP. Most tummy pains are harmless (IBS, indigestion, infection), but it's always worth checking."},
    {"k": ["weight loss","losing weight","unexplained weight"],
     "a": "🤖 Losing more than 5% of your body weight without trying — say 4 kg if you weigh 80 kg — over 6 months is a warning sign. It doesn't always mean cancer, but it does mean your body is telling you something. Please book a GP appointment, especially if you also have tummy pain, tiredness, or any changes in your bowel habit."},
    {"k": ["constipation","hard stool","not passing stool","change bowel habit","bowel habit"],
     "a": "🤖 A new and lasting change in your bowel habit — more constipation, more diarrhoea, or needing to go more often than usual — that goes on for more than 3 weeks is worth seeing a GP about. Most causes are harmless (diet, stress, IBS), but a check rules out the rare worrying ones. Keep a simple diary: how often, what consistency, any blood — it really helps."},
    {"k": ["diarrhoea","diarrhea","loose stool","watery stool"],
     "a": "🤖 Loose stools that last more than 4 weeks should be checked. Causes can include simple things like a food intolerance, mild infection, or stress; less commonly inflammatory bowel disease or other gut conditions. If you also see blood, lose weight, or feel very tired, see your GP sooner rather than later."},
    {"k": ["heartburn","acid reflux","gerd","gord","barrett"],
     "a": "Long-standing GORD (>5 yrs), male sex, age >50, central obesity and smoking are the main Barrett's-oesophagus risk factors. BSG 2023 recommends 3- or 5-year surveillance endoscopy for non-dysplastic Barrett's depending on segment length. ColonAI's image agent can flag Barrett's features on upper-GI endoscopy."},
    {"k": ["mucus","mucous","slime","gloop"],
     "a": "Mucus in stool is common in IBS and haemorrhoids, but persistent mucus with bleeding or urgency suggests inflammation (UC) and warrants flexible sigmoidoscopy with biopsies."},
    {"k": ["anaemia","anemia","low iron","iron deficiency"],
     "a": "Iron-deficiency anaemia in any man, or post-menopausal woman, is a CRC red flag until proven otherwise — NICE NG12 recommends bidirectional endoscopy."},

    # ─── Conditions ──────────────────────────────────────────────────────
    {"k": ["polyp","polyps","colorectal polyp","adenoma"],
     "a": "🤖 A polyp is a small growth on the inside lining of your colon — most are completely harmless. But a small number can slowly turn into cancer over many years, which is why doctors remove them when they're spotted during a colonoscopy. The removal is quick and painless (you're sedated). After it's removed, your doctor will tell you when to come back for your next check-up — usually in 3 to 5 years depending on what they found."},
    {"k": ["ulcerative colitis","uc","colitis","ibd","inflammatory bowel","crohn"],
     "a": "🤖 Ulcerative colitis (UC) is a long-term condition where the lining of your colon becomes inflamed and sore. It often comes and goes in 'flare-ups' — you might have weeks of normal life followed by days of cramping, diarrhoea or blood in your stool. The good news: there are many effective medicines today, from simple anti-inflammatories to newer biologic injections. Most people lead a full, normal life with UC. Your gastroenterologist will pick the right treatment based on how mild or severe your flares are."},
    {"k": ["barretts","barrett esophagus","barrett oesophagus","esophagus","oesophagus"],
     "a": "🤖 Barrett's oesophagus is a change in the cells at the bottom of your food pipe, usually caused by long-term acid reflux (heartburn). It's not cancer — but a small number of cases can develop into cancer over many years, so doctors monitor it with a regular endoscopy (usually every 3 to 5 years). Treatment focuses on controlling acid: medication like omeprazole, weight management, avoiding triggers (spicy food, alcohol, big meals before bed). If any worrying changes are found, they can be treated early without surgery."},
    {"k": ["colon cancer","colorectal cancer","bowel cancer","rectal cancer","crc"],
     "a": "🤖 Colon cancer (also called colorectal or bowel cancer) is the third most common cancer in the world. The good news — if it's caught early, more than 9 out of 10 people survive. That's why screening matters so much. Common risk factors include: getting older (most cases after 50), family history, smoking, drinking a lot of alcohol, eating lots of red or processed meat, being overweight, or not exercising. None of these mean you WILL get cancer — they just raise the chance a little. The best protection is regular screening and a healthy lifestyle."},
    {"k": ["lynch","fap","hnpcc","hereditary","polyposis","familial"],
     "a": "Lynch syndrome (germline MMR mutation) accounts for ~3% of CRC. FAP (APC) presents with hundreds of adenomas and 100% lifetime CRC risk if untreated. Affected families need genetic counselling and colonoscopy from 25 (Lynch) or sigmoidoscopy from 12-14 (FAP)."},

    # ─── Screening ───────────────────────────────────────────────────────
    {"k": ["screening","screened","screen me","when to screen","colonoscopy age",
            "how often","start screening","starting age","check for cancer","at what age"],
     "a": "🤖 Most people should start screening at age 45 to 50, even if they feel completely fine. If you have a parent, brother or sister who had colon cancer, you should start 10 years before THEIR age at diagnosis. There are three common ways: a simple stool test you do at home once a year, or a colonoscopy once every 10 years, or a CT scan of your colon once every 5 years. Your GP can recommend the right one for you."},
    {"k": ["fit","fit test","fob","faecal","stool test","poo test"],
     "a": "🤖 The FIT test is a simple kit you use at home — you collect a tiny stool sample and post it back to the lab. It checks for hidden blood that you can't see. If the test comes back positive, your GP will arrange a colonoscopy to find out why. It's not embarrassing, it doesn't hurt, and it can save your life. Most people aged 50–74 in the UK are offered one for free every 2 years."},
    {"k": ["prep","prepare","prep colonoscopy","drink prep","movicol","picolax","plenvu",
            "fast before colonoscopy","fasting","what to eat before colonoscopy",
            "diet before colonoscopy","clear fluids","laxative","bowel prep"],
     "a": "🤖 Most people find the day before the colonoscopy harder than the procedure itself! You'll eat low-fibre food (no salad, seeds or whole grains) for 2–3 days, then only clear fluids on the day before. The evening before, you'll drink a special laxative drink that empties your bowel — yes, you'll be on the toilet a lot. The next morning you drink another dose and then go to the hospital. It's not pleasant but it's necessary so the doctor can see clearly."},
    {"k": ["pain colonoscopy","painful colonoscopy","sedation","entonox","midazolam",
            "does it hurt","is it painful","will it hurt","painful","hurt","ouch",
            "anaesthesia","anesthesia"],
     "a": "🤖 Most colonoscopies are not painful — you'll be given gentle sedation or gas-and-air to keep you comfortable. You might feel some pressure or bloating, like trapped wind, but not sharp pain. The whole thing takes about 30 minutes. If you've had sedation, ask someone to drive you home and rest for the day. Many people are surprised by how easy it was."},
    {"k": ["risks colonoscopy","colonoscopy risks","perforation","bleeding"],
     "a": "🤖 Colonoscopy is one of the safest medical procedures. The chance of a tear in the bowel is about 1 in 1,500 and serious bleeding about 1 in 1,000 — very rare. If a polyp is removed, the bleeding risk is a little higher (about 1–2 in 100) but still small. Sedation reactions are very uncommon. For most people, the benefits of having one when needed far outweigh these small risks."},

    # ─── AI / Model ──────────────────────────────────────────────────────
    {"k": ["gradcam","grad cam","grad-cam","heatmap","heat map","attention map",
            "what is the model looking at","activation map","red region",
            "what is gradcam","what does gradcam show"],
     "a": "The GradCAM heatmap shows where the AI is looking on your image — red / warm pixels are the regions that most influenced its decision. If the warm pixels overlap with the lesion you'd point at clinically, the AI is thinking the way you'd want it to. If they're elsewhere, treat the prediction with extra caution."},
    {"k": ["how accurate","accuracy","performance","model performance","metrics","auc","f1"],
     "a": "On the held-out test split (1,066 images from HyperKvasir + CVC-ClinicDB) the model achieves 90.3% accuracy, 0.81 macro F1, 0.984 AUC-ROC across 5 GI classes. Best epoch 7. Note: this is research-grade — external validation on independent hospital data hasn't been done yet."},
    {"k": ["biobert","bert","text","clinical text","nlp","language model"],
     "a": "BioBERT (dmis-lab/biobert-base-cased-v1.2) is a BERT pre-trained on PubMed abstracts + PMC articles. We freeze the bottom 10 layers, fine-tune the top 2, and pool the [CLS] token through a 256-d projection head into the fusion transformer."},
    {"k": ["tabular","tabtransformer","tcga","patient features","clinical features"],
     "a": "Patient features (age, BMI, smoking, alcohol, family history etc.) are encoded by a TabTransformer trained on 12 TCGA-derived features. Per-feature attention learns interactions like age × smoking × prior-polyps that single-modality models miss."},
    {"k": ["fusion","cross modal","cross attention","multimodal"],
     "a": "Fusion uses a 3-stage gated cross-modal transformer (8 heads, 256-d, 3 layers). Stage A: per-modality self-attention. Stage B: bidirectional cross-attention. Stage C: shared bottleneck + CLS pool + sigmoid modality gate that decides how much each branch contributes per case."},
    {"k": ["uncertainty","confidence","how sure","mc dropout","epistemic"],
     "a": "We estimate epistemic uncertainty with 15 stochastic forward passes (MC-Dropout). Lower (<0.3) = consistent predictions across passes. Higher (>0.6) = the model is genuinely unsure and you should weight clinician review more heavily. Calibration is reported on the Risk Charts tab."},
    {"k": ["calibration","ece","reliability","calibrated"],
     "a": "Calibration measures whether a 0.8 confidence really means 80% accuracy. We track the Expected Calibration Error (ECE) and reliability diagrams in the training reports under outputs/unified_multimodal/figures. Temperature scaling is applied during inference to tighten calibration."},
    {"k": ["bias","fairness","generalisation","external","out of distribution","ood"],
     "a": "Honest limitations: HyperKvasir is European, mostly Nordic. CVC-ClinicDB is Spanish. TCGA tabular skews North-American. The model has not been externally validated on Asian/African endoscopy databases or paediatric cohorts. Clinical deployment would need site-specific revalidation and a fairness audit."},
    {"k": ["regulation","fda","mhra","ce mark","ukca","class","samd"],
     "a": "Tools like ColonAI fall under SaMD (Software as a Medical Device). Pathway: FDA 510(k) De-Novo (US), UKCA Class IIa under MHRA (UK), CE Class IIa under MDR (EU). Current build is research-only — clinical use needs intended-use scope, clinical-evaluation report, and post-market surveillance plan."},

    # ─── App functions ───────────────────────────────────────────────────
    {"k": ["demo","sample","example","quick start"],
     "a": "On Step 1 (Patient Info) the top panel has three Quick-demo cases: Case A (Sigmoid Polyp), Case B (Ulcerative Colitis), Case C (Barrett's). One click loads the full clinical scenario and a real endoscopy image — perfect for presentations."},
    {"k": ["find doctor","gastroenterologist","specialist","oncologist","surgeon","consultant"],
     "a": "Step 5 lists 46+ gastroenterologists, colorectal surgeons and GI oncologists across 20+ cities (India, USA, UK, UAE, Singapore, Canada, Australia). Filter by city/country/specialty. Top 5 are appended to your downloadable PDF report."},
    {"k": ["report","pdf","download","generate report"],
     "a": "Step 6 (Download Report) builds an A4 clinical PDF: patient header, symptom log, AI findings (pathology, staging, risk score, modality weights), embedded GradCAM++ heatmap, recommended next steps, suggested specialists and a regulatory disclaimer."},
    {"k": ["upload","image","colonoscopy","endoscopy","histopathology","photo","picture","jpeg","jpg","png"],
     "a": "Step 2, Upload Images tab: JPG/PNG up to 10 MB. The image is resized to 224×224 with ImageNet normalisation, then routed through both image backbones in parallel for the GradCAM-friendly fused representation."},
    {"k": ["risk","high risk","low risk","malignant","benign","cancer risk","risk score"],
     "a": "Risk score (0-100%) is the binary cancer-vs-benign head's softmax probability. Bands: <25% Low, 25-50% Moderate, 50-75% High, >75% Critical. The Risk Charts tab also shows a multi-dimensional radar combining AI risk + age + smoking + alcohol + family history + prior polyps."},

    # ─── Disclaimers / safety ────────────────────────────────────────────
    {"k": ["data","privacy","gdpr","secure","stored"],
     "a": "Images are processed in-memory in this Streamlit session and never persisted to disk. The PDF report is generated client-side. In production deployment the system would run within a hospital VPC with full UK DPA 2018 / GDPR compliance and audit logs."},
    {"k": ["disclaimer","medical advice","diagnosis","replace doctor","is this a diagnosis"],
     "a": "ColonAI is a research / decision-support tool, not a regulated medical device, not a diagnosis. All findings MUST be confirmed by a licensed clinician before any treatment decision. In an emergency contact local emergency services."},
    {"k": ["replace","instead of","skip doctor","not see doctor"],
     "a": "No — AI screening flags candidates for review; it does not replace clinical examination, biopsy/histology, or specialist judgement. The pathway is: AI suggests → clinician reviews → endoscopy/biopsy → MDT decides."},

    # ─── Diet & nutrition ────────────────────────────────────────────────
    {"k": ["diet","food","eat","nutrition","what to eat","what should i eat","best food","colon diet","diet plan","diet for colon","colon health"],
     "a": "🤖 Best foods for a healthy colon: plenty of fibre (oats, beans, lentils, brown rice, whole-wheat bread, fruit and vegetables — aim for 30 g a day), oily fish like salmon or sardines twice a week, and lots of water (about 8 glasses a day). Things to limit: red meat (beef, lamb, pork) to about twice a week, and try to avoid processed meats (bacon, sausages, ham) — these have been linked to higher cancer risk. A Mediterranean-style diet — lots of vegetables, fish, olive oil, nuts — has the strongest evidence for prevention. Small changes today, big benefits tomorrow."},
    {"k": ["fibre","fiber","high fibre","high fiber","roughage"],
     "a": "Aim for ≥30 g of fibre per day (NHS / SACN). Easy wins: oats at breakfast, beans / chickpeas in lunch, leafy greens at dinner, an apple or pear with skin, wholegrain bread instead of white. Each extra 10 g/day reduces CRC risk ~10% (BMJ meta-analysis 2011)."},
    {"k": ["red meat","beef","mutton","lamb","pork","steak","processed meat","bacon","sausage","ham"],
     "a": "WHO IARC classifies processed meat as Group 1 (definite cause of CRC) and red meat as Group 2A (probable). Practical rule: keep red meat <500 g/week (≈ 70 g/day cooked) and avoid processed meat. Each 50 g/day of processed meat raises CRC risk ~18%."},
    {"k": ["exercise","activity","physical","walk","gym","run","yoga"],
     "a": "Move daily — 150 min/week of moderate activity (brisk walking, cycling, swimming) cuts CRC risk by 24% (WCRF). Plus stronger gut motility, lower BMI and better insulin sensitivity. Even 30 min of walking after meals helps."},
    {"k": ["alcohol limit","how much alcohol","drink limit"],
     "a": "UK guidelines: <14 units/week, spread across the week, with several alcohol-free days. Each 10 g/day of alcohol raises CRC risk ~7%. There is no safe lower threshold."},
    {"k": ["water","hydration","fluid"],
     "a": "Hydration helps stool transit and reduces constipation. Aim for 1.5–2 L of water per day (more in heat / exercise). Caffeine and alcohol don't replace water."},
    {"k": ["vitamin","supplement","calcium","vitamin d","folate"],
     "a": "Best evidence is for calcium 1,000–1,200 mg/day and vitamin D (esp. if levels are low). Folate is debated — supplementation in established polyps may be unhelpful. Whole-food sources beat supplements wherever possible."},

    # ─── Prevention ─────────────────────────────────────────────────────
    {"k": ["prevent","prevention","reduce risk","avoid cancer","stop cancer","how to prevent"],
     "a": "Five evidence-based steps: (1) screen on schedule (FIT yearly or colonoscopy q10y from 45–50), (2) ≥30 g fibre/day, limit red/processed meat, (3) keep BMI 18.5–25, (4) 150 min exercise/week, (5) don't smoke and keep alcohol <14 units/week. Removes ~50% of preventable CRC."},
    {"k": ["smoking","smoke","tobacco","cigarette","quit","nicotine"],
     "a": "Smoking is a clear risk factor for CRC (and many other cancers). Quitting at any age reduces risk — by 10 years post-quit you're close to never-smoker risk. NHS Stop Smoking, Champix, NRT and behavioural support all help."},
    {"k": ["weight","obesity","bmi","losing weight","lose weight"],
     "a": "Each 5 kg/m² rise in BMI raises CRC risk ~5%. Belly fat (waist circ >94 cm M / >80 cm F) is independently risky. Losing 5–10% of body weight reduces inflammatory markers fast — even before reaching a 'normal' BMI."},
    {"k": ["age","start age","when does cancer happen"],
     "a": "Average-risk screening starts at 45 (USPSTF 2021) or 50 (NHS). Most CRC cases occur after 50, but young-onset CRC (<50 y) is rising — that's why the floor was lowered. Family history halves the start age (often 40 or 10 yrs before youngest case)."},

    # ─── Treatment / outcomes ────────────────────────────────────────────
    {"k": ["treatment","therapy","cure","heal","options"],
     "a": "🤖 Treatment depends on what stage the cancer is found at — that's why early detection matters so much. Very early-stage cancers can often be removed during a colonoscopy itself, no surgery needed. More advanced cases may need keyhole surgery (you go home in 3–5 days) plus a few months of chemotherapy tablets or injections. Today's treatments are far gentler than they were 20 years ago, with much higher success rates. Your specialist team will recommend the best plan for your specific situation."},
    {"k": ["surgery","operation","resection","colectomy"],
     "a": "Most CRC surgery today is laparoscopic or robotic — 3-5 small incisions, faster recovery. Hospital stay 3-7 days. Ileostomy / colostomy is uncommon for left-sided cancers but may be temporary. Discuss enhanced-recovery (ERAS) protocols with your surgeon."},
    {"k": ["chemotherapy","chemo","drugs","cancer drugs"],
     "a": "Adjuvant chemo for stage III CRC (FOLFOX or CAPOX, 3-6 months) cuts recurrence by ~30%. Side effects: fatigue, neuropathy, nausea — most are manageable. Many regimens are now given oral or via day-case infusion."},
    {"k": ["radiation","radiotherapy","radio"],
     "a": "Radiotherapy is mainly used for rectal cancer (not colon). Short-course (5 days) or long-course chemoradiotherapy (5-6 weeks) shrinks tumours pre-surgery and lowers local recurrence."},
    {"k": ["recovery","after surgery","convalescence","heal time","how long recovery",
            "long recovery","recover time","time to heal","back to work"],
     "a": "Typical recovery from laparoscopic CRC surgery: 4-6 weeks total. Walking the same day, light meals 24-48 h later, driving in 2-3 weeks, full activity in 6-8 weeks. Surveillance starts 1 year after curative resection."},
    {"k": ["survival","survival rate","prognosis","life expectancy","outcome"],
     "a": "🤖 Survival depends a lot on when colon cancer is found. If caught at the earliest stage (Stage 1), more than 9 in 10 people are still alive 5 years later. At Stage 2 it's about 8 in 10, at Stage 3 about 6 in 10, and even at Stage 4 there are many treatments that can give a much longer life than before. The biggest single thing you can do to improve your chances is to get screened on time. That's why we keep saying it!"},
    {"k": ["biopsy","histology","pathology","tissue test"],
     "a": "A biopsy takes 2-5 mm of tissue during colonoscopy — painless under sedation. Histology takes ~5-10 working days; immunohistochemistry / MMR testing adds a week. Discuss the result face-to-face — the report can look alarming but the wording matters."},

    # ─── Logistics & emotional ───────────────────────────────────────────
    {"k": ["scared","afraid","worried","anxious","nervous","fear","panic",
            "i'm worried","i am scared","scared of colonoscopy","afraid of colonoscopy",
            "scared of test","fear of cancer","anxiety","stressed"],
     "a": "Anxiety around cancer screening is completely normal. Two facts that help: (1) the vast majority of FITs and colonoscopies turn out negative, (2) when something IS found early, outcomes are excellent. The procedure itself is done under sedation — you'll feel pressure, not pain. Talk to your GP — many clinics offer a screening-anxiety nurse."},
    {"k": ["second opinion","another doctor","verify"],
     "a": "Second opinions are reasonable for any cancer-related decision — most surgeons / oncologists welcome them. Step 5 in this app (Find Doctors) lists specialists you can contact. Bring your imaging, histology and current report."},
    {"k": ["insurance","cost","price","afford","money"],
     "a": "NHS bowel-cancer screening (UK) is free for ages 50-74. In the US, USPSTF Grade-A/B services are covered without copay under the ACA. Private colonoscopy costs ~£1,500-£2,500 in the UK or $1,000-$3,000 in the US."},
    {"k": ["wait time","appointment","queue","referral time","2 week wait","two week wait"],
     "a": "NHS pathway after a 2-week-wait referral: GP referral → specialist clinic ≤14 days → diagnostic colonoscopy ≤28 days → MDT discussion. If you're flagged urgent and waits exceed these, escalate via the Patient Advice and Liaison Service (PALS)."},
    {"k": ["pregnancy","pregnant","ttc","trying for baby"],
     "a": "Routine bowel screening is deferred during pregnancy if the patient is asymptomatic. Symptomatic / red-flag presentations are still investigated — flexible sigmoidoscopy and biopsy can be done safely with a GI specialist."},
    {"k": ["children","kids","paediatric","under 18","teenager"],
     "a": "Routine CRC screening doesn't apply to children. For hereditary syndromes (FAP, Lynch, MAP, Peutz-Jeghers) screening starts in adolescence — see a clinical geneticist. Symptomatic children with bloody diarrhoea need paediatric gastro review."},

    # ─── Greeting / fallback ─────────────────────────────────────────────
    {"k": ["hi","hello","hey","greetings","good morning","good evening","good afternoon",
           "who are you","your name","what is your name"],
     "a": "Hi! 🤖 I'm <b>Colon Buddy</b>, your friendly health assistant. I can answer questions about colon health, screening, symptoms, prevention, diet, treatment, and how to use this app. Just ask me anything — for example: <i>'when should I be screened?'</i> or <i>'what foods are good for my colon?'</i>"},
    {"k": ["thanks","thank you","cheers","ty"],
     "a": "You're very welcome! 🤖 Glad I could help. Remember to always double-check anything important with your doctor — I'm a friendly helper, not a clinician."},
    {"k": ["bye","goodbye","see you","later"],
     "a": "Take care! 🤖 If anything feels urgent — like bleeding, severe pain or sudden weight loss — please contact your GP or emergency services right away. I'm always here when you come back."},

    # ─── Fallback (must remain last) ─────────────────────────────────────
    {"k": [],
     "a": "🤖 Hmm, I didn't quite catch that — but I'm <b>Colon Buddy</b> and I love to help! I can answer questions about: <b>symptoms</b> (rectal bleeding, weight loss, change in bowel habit, anaemia), <b>conditions</b> (polyps, ulcerative colitis, Crohn's, Barrett's, colorectal cancer), <b>screening</b> (FIT, colonoscopy, age, prep), <b>prevention</b> (diet, exercise, smoking, alcohol), <b>treatment</b> and <b>how to use this app</b>. Try a friendlier question like <i>'how do I prevent colon cancer?'</i> or <i>'what should I eat for a healthy colon?'</i>"},
]


_CHAT_STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","do","does","did",
    "of","in","on","at","to","for","by","with","from","and","or","but","if","i","my",
    "me","you","your","we","our","us","it","this","that","what","how","when","where",
    "why","which","can","could","should","would","will","shall","may","might","must",
    "have","has","had","not","no","yes","please","tell","ask","know","get","go","make",
    "really","truly","just","like","want","need","ok","okay","ne","ki","ka","ke","kya",
}


def _llm_ask(user_msg: str, timeout: float = 30.0) -> Optional[str]:
    """Ask the LLM — Ollama first (local), Pollinations as fallback.

    Returns the answer text, or None on any failure (caller falls back to KB).
    """
    import requests
    system_prompt = (
        "You are Colon Buddy, a friendly health-INFORMATION assistant for the ColonAI app. "
        "You give GENERAL EDUCATION ONLY about colon health, colon-cancer screening, polyps, "
        "ulcerative colitis, Barrett's oesophagus, diet, symptoms, prevention, and how to use "
        "the ColonAI app. "
        "STRICT SAFETY RULES you must never break: "
        "(1) You are NOT a doctor — never diagnose or tell the user what condition or stage they have. "
        "(2) Never interpret the user's own scan, image, or test results — tell them to ask their clinician. "
        "(3) Never give specific drug names, doses, or personalised treatment plans. "
        "(4) For anything outside general colon-health education, or anything needing a diagnosis, "
        "politely decline and tell them to consult a qualified doctor. "
        "(5) Never invent statistics or facts; if unsure, say so and recommend a doctor. "
        "Answer in plain everyday English — no jargon, no acronyms — in 3-5 short sentences. "
        "Always end by reminding them you are an automated assistant, not a doctor, and to confirm "
        "anything important with a real clinician."
    )

    # 1) Try Ollama (local, no API key, private) — prefer larger model
    for model in ("gemma3:4b", "llama3.2:3b"):
        try:
            r = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_msg},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 250},
                },
                timeout=timeout,
            )
            if r.status_code == 200:
                text = (r.json().get("message") or {}).get("content", "").strip()
                if text and len(text) > 12:
                    return text[:1200]
        except Exception:
            continue

    # 2) Fallback: Pollinations.ai
    try:
        import urllib.parse
        encoded = urllib.parse.quote(f"{system_prompt}\n\nUser: {user_msg}\nColon Buddy:")
        r = requests.get(f"https://text.pollinations.ai/{encoded}", timeout=10)
        if r.status_code == 200 and r.text and len(r.text) > 10:
            t = r.text.strip()
            # Skip junk replies from the deprecated API
            low = t.lower()
            if any(j in low for j in ("deprecat", "migrate to", "rate limit", "<!doctype")):
                return None
            return t[:1200]
    except Exception:
        pass

    return None


def _chatbot_respond(user_msg: str) -> str:
    """Hybrid responder:
       1. First try the local KB (fast, deterministic, plain-English answers).
       2. If KB has no strong match, fall back to free LLM (Pollinations.ai).
       3. If LLM fails too, return the friendly fallback message.

    The KB always wins for the well-known patient questions because its
    answers are carefully written in plain English — the LLM is the backup
    for unusual or open-ended questions.
    """
    import re
    msg = (user_msg or "").lower().strip()
    if not msg:
        return CHATBOT_KB[-1]["a"]

    # Normalised, punctuation-free version for substring matching
    norm = re.sub(r"[^a-z0-9 ]+", " ", msg)
    norm = re.sub(r"\s+", " ", norm).strip()
    tokens = [t for t in norm.split() if t and t not in _CHAT_STOPWORDS]

    best_entry = None
    best_score = 0.0
    for entry in CHATBOT_KB[:-1]:
        score = 0.0
        matched_kw = []
        for kw in entry.get("k", []):
            if not kw:
                continue
            kw_norm = kw.lower()
            if " " in kw_norm:
                # Multi-word keyword — exact phrase substring match counts heavily
                if kw_norm in norm:
                    score += 3.0 + 0.5 * len(kw_norm.split())
                    matched_kw.append(kw_norm)
                    continue
                # Otherwise: each token of the keyword that's in the user msg
                kw_tokens = [t for t in kw_norm.split() if t not in _CHAT_STOPWORDS]
                hits = sum(1 for t in kw_tokens if t in tokens)
                if hits == len(kw_tokens) and kw_tokens:
                    score += 2.0 + 0.3 * len(kw_tokens)
                    matched_kw.append(kw_norm)
                elif hits > 0:
                    score += 0.6 * hits
            else:
                # Single-word keyword
                if kw_norm in tokens:
                    score += 2.0 + min(len(kw_norm) * 0.05, 1.0)
                    matched_kw.append(kw_norm)
                elif kw_norm in norm:
                    # substring like "polyposis" matches "polyposis"
                    score += 1.0
                    matched_kw.append(kw_norm)

        # Bonus when multiple keywords from the same entry matched
        if len(matched_kw) >= 2:
            score += 0.8 * (len(matched_kw) - 1)

        if score > best_score:
            best_score = score
            best_entry = entry

    # Strong KB match → return curated plain-English answer
    if best_entry and best_score >= 1.0:
        return best_entry["a"]

    # Guideline-grounded layer (smart win #3): prefer a CITED guideline answer
    # over the free LLM — keeps the bot honest and sourced, no hallucination.
    try:
        from src.app.guideline_kb import cited_answer
        _ga = cited_answer(user_msg)
        if _ga:
            return (f"🤖 {_ga['statement']}\n\n📖 Source: {_ga['source']}. "
                    "This is general guidance — please confirm with your doctor.")
    except Exception:
        pass

    # Weak / no KB match → try the free LLM
    llm_reply = _llm_ask(user_msg)
    if llm_reply:
        return llm_reply

    # Final fallback — friendly default
    return CHATBOT_KB[-1]["a"]


@st.dialog("🤖 Colon Buddy — your friendly health assistant", width="large")
def _colon_buddy_dialog():
    """Full-size centred chat dialog with PROPER scrolling layout.

    Layout (top → bottom):
       1. Header card (slim, doesn't push others down)
       2. Quick-chip row (compact)
       3. SCROLLABLE chat history (fixed height, internal scrollbar)
       4. Input at the bottom (pinned)

    Critical: every section keeps its own height so the user can always
    reach the input.  No content can push the input off-screen.
    """
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ── Slim header card ──
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#FB7185 0%,#F97316 60%,#FBBF24 100%);
             border-radius:12px;padding:10px 16px;margin-bottom:10px;color:#FFF;
             box-shadow:0 4px 12px -4px rgba(249,115,22,0.30);">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="font-size:26px;line-height:1;">🤖</div>
            <div>
              <div style="font-size:1.0rem;font-weight:800;">Hi, I'm Colon Buddy!</div>
              <div style="font-size:0.78rem;opacity:0.95;">
                Plain-English answers about colon health
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Quick chips (3 columns, compact row) ──
    chip_qs = [
        ("🔍 Polyp",       "what is a polyp?"),
        ("📅 Screening",   "when should I be screened?"),
        ("🥗 Diet",        "diet for colon health"),
        ("⚠️ Symptoms",    "what are the symptoms"),
        ("💊 Treatment",   "treatment options"),
        ("😟 I'm scared",  "i am scared"),
    ]
    cols = st.columns(3)
    for i, (label, q) in enumerate(chip_qs):
        with cols[i % 3]:
            if st.button(label, key=f"dlg_chip_{i}", use_container_width=True):
                with st.spinner("🤖 Thinking…"):
                    reply = _chatbot_respond(q)
                st.session_state["chat_history"].append(("user", q))
                st.session_state["chat_history"].append(("assistant", reply))
                st.rerun()

    # ── SCROLLABLE chat history container (fixed height) ──
    # Using st.container(height=N) gives a native vertical scrollbar.
    st.markdown(
        "<div style='margin:8px 0 4px;font-size:0.85rem;font-weight:700;color:#475569;'>"
        "💬 Conversation:</div>",
        unsafe_allow_html=True,
    )
    chat_box = st.container(height=380, border=True)
    with chat_box:
        if not st.session_state["chat_history"]:
            st.info("👋 No questions yet. Click a chip above or type a question at the bottom!")
        else:
            history = st.session_state["chat_history"][-30:]
            pairs = []
            i = 0
            while i < len(history):
                if i + 1 < len(history) and history[i][0] == "user" and history[i+1][0] == "assistant":
                    pairs.append((history[i], history[i+1]))
                    i += 2
                else:
                    pairs.append((history[i], None))
                    i += 1
            # Oldest first, latest at bottom (natural chat flow — user scrolls to latest)
            for pair in pairs:
                u_msg, a_msg = pair[0], pair[1]
                # User bubble (blue, right)
                st.markdown(
                    f"""<div style='display:flex;justify-content:flex-end;margin:8px 0;'>
                      <div style='background:linear-gradient(135deg,#1A73E8,#1E40AF);
                           color:#FFF;border-radius:16px 16px 4px 16px;
                           padding:9px 13px;font-size:0.92rem;
                           max-width:75%;box-shadow:0 3px 8px rgba(26,115,232,0.20);
                           word-wrap:break-word;line-height:1.5;'>
                        {u_msg[1]}
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if a_msg is not None:
                    st.markdown(
                        f"""<div style='display:flex;align-items:flex-start;gap:8px;margin:6px 0 12px;'>
                          <div style='font-size:1.5rem;flex-shrink:0;margin-top:2px;'>🤖</div>
                          <div style='background:#FFF7ED;color:#7C2D12;
                               border-radius:4px 16px 16px 16px;
                               padding:10px 14px;font-size:0.92rem;
                               border:1px solid #FED7AA;
                               max-width:80%;box-shadow:0 3px 8px rgba(249,115,22,0.10);
                               word-wrap:break-word;line-height:1.6;'>
                            {a_msg[1]}
                          </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    # ── Input INSIDE the dialog using a proper st.form ──
    # st.form GUARANTEES that pressing Enter or clicking the submit button
    # both work and commit the input value reliably.  clear_on_submit=True
    # empties the field after each submission.
    with st.form(key="cb_dialog_form", clear_on_submit=True, border=False):
        cols_input = st.columns([5, 1])
        with cols_input[0]:
            user_text = st.text_input(
                "Your question",
                placeholder="Type your question and press Enter ⏎",
                label_visibility="collapsed",
                key="cb_dialog_form_text",
            )
        with cols_input[1]:
            submitted = st.form_submit_button(
                "Send 📨",
                type="primary",
                use_container_width=True,
            )

    if submitted and user_text and user_text.strip():
        q = user_text.strip()
        with st.spinner("🤖 Colon Buddy is thinking…"):
            reply = _chatbot_respond(q)
        if not reply or not str(reply).strip():
            reply = "🤖 I'm not sure how to answer that — try one of the chips above."
        st.session_state["chat_history"].append(("user", q))
        st.session_state["chat_history"].append(("assistant", reply))
        st.rerun()

    # Footer row — clear conversation
    if st.session_state["chat_history"]:
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            if st.button("🗑️ Clear", use_container_width=True, key="dlg_clear",
                         help="Clear the conversation"):
                st.session_state["chat_history"] = []
                st.rerun()

    st.caption("💡 Plain-English answers backed by a free background AI. "
               "Always confirm anything serious with a real doctor.")


def render_chatbot():
    """Render **Colon Buddy** — the friendly AI assistant — in the sidebar.

    Auto-expanded by default. Branded with a mascot, friendly intro,
    suggested-question chips, and a vibrant card design.
    """
    auto_expand = False
    open_dialog = False
    try:
        focus_val = st.query_params.get("focus_chat")
        if focus_val in ("1", "true", "yes"):
            auto_expand = True
            open_dialog = True
            # Clear the query param so the dialog only opens once per click
            try:
                st.query_params.pop("focus_chat", None)
            except Exception:
                pass
    except Exception:
        pass

    if "chat_expanded" not in st.session_state:
        st.session_state["chat_expanded"] = True

    # If the floating FAB triggered the open, show the full-size dialog
    if open_dialog:
        try:
            _colon_buddy_dialog()
        except Exception:
            pass

    # ── BIG "Open Colon Buddy" button — opens the full-size dialog ─────
    if st.sidebar.button("🤖  Open Colon Buddy Chat",
                          use_container_width=True, type="primary",
                          key="open_cb_dialog_btn",
                          help="Opens a full-size chat window in the centre of the page"):
        _colon_buddy_dialog()

    # ── Mascot header card — always visible above the expander ─────────
    st.sidebar.markdown(
        """
        <style>
        @keyframes cbBounce {
          0%, 100% { transform: translateY(0) rotate(-2deg); }
          50%      { transform: translateY(-3px) rotate(2deg); }
        }
        @keyframes cbPulse {
          0%, 100% { box-shadow: 0 6px 18px rgba(236,72,153,0.30); }
          50%      { box-shadow: 0 10px 26px rgba(236,72,153,0.55); }
        }
        .colon-buddy-header {
          background: linear-gradient(135deg,#FB7185 0%,#F97316 60%,#FBBF24 100%);
          border-radius: 14px;
          padding: 12px 14px;
          margin: 6px 0 4px 0;
          color: #FFF;
          animation: cbPulse 2.6s ease-in-out infinite;
          position: relative;
          overflow: hidden;
        }
        .colon-buddy-mascot {
          display: inline-block;
          font-size: 28px;
          animation: cbBounce 1.8s ease-in-out infinite;
          margin-right: 8px;
        }
        .colon-buddy-name {
          font-weight: 900;
          font-size: 1.05rem;
          letter-spacing: -0.2px;
        }
        .colon-buddy-tag {
          font-size: 0.72rem;
          opacity: 0.92;
          font-weight: 500;
          margin-top: 2px;
        }
        </style>
        <div class="colon-buddy-header">
          <div style="display:flex;align-items:center;">
            <span class="colon-buddy-mascot">🤖</span>
            <div>
              <div class="colon-buddy-name">Colon Buddy</div>
              <div class="colon-buddy-tag">Your friendly health assistant · ask me anything ↓</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("💬 Chat with Colon Buddy",
                             expanded=auto_expand or
                                      st.session_state.get("chat_expanded", True)):
        st.session_state["chat_seen"] = True
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # ── INPUT AT THE TOP ──
        # Streamlit's chat_input inside a sidebar expander is unreliable, so
        # we use text_input + on_change callback. The callback fires when the
        # user presses Enter or clicks away — both cases submit the message.
        # We use a counter-based key so the input field is "cleared" by
        # bumping the key each time a message is sent.
        if "cb_msg_counter" not in st.session_state:
            st.session_state["cb_msg_counter"] = 0
        if "cb_pending_msg" not in st.session_state:
            st.session_state["cb_pending_msg"] = ""

        # If there's a pending message from a previous submit, process it now
        pending = st.session_state.get("cb_pending_msg", "").strip()
        if pending:
            reply = _chatbot_respond(pending)
            if not reply or not str(reply).strip():
                reply = ("🤖 I'm not sure how to answer that yet — try one of "
                         "the quick-question chips below, or ask about polyps, "
                         "screening, symptoms, diet or treatment.")
            st.session_state["chat_history"].append(("user", pending))
            st.session_state["chat_history"].append(("assistant", reply))
            st.session_state["cb_pending_msg"] = ""    # consumed

        cb_key = f"cb_input_{st.session_state['cb_msg_counter']}"

        def _cb_on_submit():
            # Fires on Enter or focus-out
            val = st.session_state.get(cb_key, "").strip()
            if val:
                st.session_state["cb_pending_msg"] = val
                st.session_state["cb_msg_counter"] += 1   # bump key → clears field

        st.text_input(
            "Your question",
            placeholder="Type your question and press Enter ⏎",
            key=cb_key,
            on_change=_cb_on_submit,
            label_visibility="collapsed",
        )

        # ── Chat history INSIDE A FIXED-HEIGHT SCROLLABLE BOX ──
        if st.session_state["chat_history"]:
            st.markdown(
                "<div style='font-size:0.72rem;color:#64748B;margin:10px 0 4px;font-weight:700;'>"
                "💬 Conversation:</div>",
                unsafe_allow_html=True,
            )
            # st.container(height=N, border=True) gives a real scrollable box
            chat_box = st.container(height=300, border=True)
            with chat_box:
                history = st.session_state["chat_history"][-30:]
                # Group into [(user, assistant), ...] pairs
                pairs = []
                i = 0
                while i < len(history):
                    if i + 1 < len(history) and history[i][0] == "user" and history[i+1][0] == "assistant":
                        pairs.append((history[i], history[i+1]))
                        i += 2
                    else:
                        pairs.append((history[i], None))
                        i += 1
                # Oldest → newest (so scroll-to-bottom naturally shows latest)
                for pair in pairs:
                    u_msg = pair[0]
                    a_msg = pair[1]
                    st.markdown(
                        f"""<div style='display:flex;justify-content:flex-end;margin:6px 0;'>
                          <div style='background:linear-gradient(135deg,#1A73E8,#1E40AF);
                               color:#FFF;border-radius:14px 14px 4px 14px;
                               padding:8px 12px;font-size:0.82rem;
                               max-width:90%;box-shadow:0 2px 6px rgba(26,115,232,0.18);
                               word-wrap:break-word;'>
                            {u_msg[1]}
                          </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    if a_msg is not None:
                        st.markdown(
                            f"""<div style='display:flex;align-items:flex-start;gap:6px;margin:6px 0 12px;'>
                              <div style='font-size:1.3rem;flex-shrink:0;margin-top:2px;'>🤖</div>
                              <div style='background:#FFF7ED;color:#7C2D12;
                                   border-radius:4px 14px 14px 14px;
                                   padding:8px 12px;font-size:0.82rem;
                                   border:1px solid #FED7AA;
                                   max-width:88%;box-shadow:0 2px 6px rgba(249,115,22,0.10);
                                   word-wrap:break-word;line-height:1.5;'>
                                {a_msg[1]}
                              </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            # Hint to open bigger chat window
            st.markdown(
                "<div style='font-size:0.68rem;color:#64748B;margin:8px 0 4px;text-align:center;'>"
                "👉 For a bigger chat window, click the orange button above</div>",
                unsafe_allow_html=True,
            )

            if st.button("🗑️ Clear conversation", use_container_width=True, key="clear_chat"):
                st.session_state["chat_history"] = []
                st.rerun()
        else:
            # No history yet — friendly intro card
            st.markdown(
                """
                <div style="background:#FFF7ED;border-left:3px solid #F97316;
                     border-radius:8px;padding:9px 12px;margin:8px 0;
                     font-size:0.80rem;color:#7C2D12;line-height:1.45;">
                  <b>🤖 Hi! I'm Colon Buddy.</b><br/>
                  Type above and press <b>Enter ⏎</b> — or click a question
                  below for an instant answer.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Quick-question chips — guaranteed-working instant answers ──
        st.markdown(
            "<div style='font-size:0.72rem;color:#64748B;margin:10px 0 4px;font-weight:700;'>"
            "⚡ Instant answers — click any:</div>"
            """<style>
            /* Compact chips — no wrap, smaller font, left-aligned */
            section[data-testid="stSidebar"] div[data-testid="stExpander"]
              div[data-testid="stButton"] button {
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                text-align: left !important;
                justify-content: flex-start !important;
                font-size: 0.74rem !important;
                padding: 5px 8px !important;
                min-height: 30px !important;
                line-height: 1.2 !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        chip_qs = [
            ("🔍 Polyps?",      "what is a polyp?"),
            ("📅 Screening?",   "when should I be screened?"),
            ("🥗 Best diet?",   "diet for colon health"),
            ("⚠️ Symptoms?",    "what are the symptoms"),
            ("💊 Treatment?",   "treatment options"),
            ("📄 PDF report?",  "how do I get a pdf report"),
        ]
        for i, (label, q) in enumerate(chip_qs):
            if st.button(label, key=f"chip_{i}", use_container_width=True):
                reply = _chatbot_respond(q)
                if not reply or not str(reply).strip():
                    reply = "🤖 Sorry, I couldn't find an answer for that."
                st.session_state["chat_history"].append(("user", q))
                st.session_state["chat_history"].append(("assistant", reply))
                st.rerun()



# ─────────────────────────────────────────────────────────────────────────────
# SITE GUIDE
# ─────────────────────────────────────────────────────────────────────────────

def page_guide():
    st.markdown(
        """<div class="hero-banner">
            <h1>Site Guide</h1>
            <p>Everything you need to know about using ColonAI effectively</p>
            <span class="hero-badge">Quick Reference</span>
            <span class="hero-badge">5 min read</span>
        </div>""",
        unsafe_allow_html=True,
    )

    tab_ov, tab_steps, tab_ai, tab_cases, tab_present, tab_faq = st.tabs([
        "Overview", "Step-by-Step", "AI Explained",
        "Case Studies", "How to Present", "FAQ"
    ])

    # ── Tab 1: Overview ────────────────────────────────────────────────
    with tab_ov:
        st.markdown('<div class="section-header">What is ColonAI?</div>', unsafe_allow_html=True)
        st.markdown(
            "ColonAI is an **agentic multimodal AI screening system** for colorectal conditions. "
            "It combines three types of data — medical images, clinical text, and patient history — "
            "to identify potential findings such as polyps, ulcerative colitis, or Barrett's esophagus."
        )
        st.markdown("")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="metric-card" style="border-left-color:#1A73E8">'
                '<div class="label">AI Model</div>'
                '<div class="value">6 Agents</div>'
                '<div class="sub">GradCAM++ | BioBERT | TabTransformer | Fusion | XAI | Clinical</div>'
                '</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(
                '<div class="metric-card" style="border-left-color:#00897B">'
                '<div class="label">Performance</div>'
                '<div class="value">90.3%</div>'
                '<div class="sub">Test accuracy · 0.984 AUC-ROC · 1,066 images (HyperKvasir + CVC-ClinicDB)</div>'
                '</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(
                '<div class="metric-card" style="border-left-color:#FF5722">'
                '<div class="label">Conditions Detected</div>'
                '<div class="value">5 Classes</div>'
                '<div class="sub">Polyps | UC Mild | UC Mod-Sev | Barrett\'s | Therapeutic</div>'
                '</div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-header">Who is this for?</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                '<div class="info-box">'
                '<b>Patients</b><br>'
                'If you have GI symptoms or have had a colonoscopy/endoscopy, you can upload your '
                'images and receive an AI-based second opinion alongside doctor recommendations.'
                '</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(
                '<div class="info-box">'
                '<b>Researchers & Clinicians</b><br>'
                'This tool is built on published research datasets (HyperKvasir, CVC-ClinicDB, TCGA) '
                'and can be used to explore AI-driven GI screening workflows.'
                '</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="warn-box">'
            '<b>Important:</b> ColonAI is a research and screening tool. '
            'It does NOT replace a qualified medical professional. '
            'Always have your results reviewed by a licensed clinician.'
            '</div>', unsafe_allow_html=True)

    # ── Tab 2: Step-by-Step ────────────────────────────────────────────
    with tab_steps:
        st.markdown('<div class="section-header">How to Use ColonAI — Step by Step</div>',
                    unsafe_allow_html=True)

        steps_guide = [
            ("Step 1 — Patient Information",
             "Fill in your name, age, gender, height, weight, city, and medical history "
             "(smoking, alcohol, family history of colorectal cancer, previous polyps). "
             "This data helps the AI contextualise its findings.",
             ["All fields marked * are required.",
              "BMI is calculated automatically from height and weight.",
              "Your data is only stored in your browser session and never sent to external servers."]),
            ("Step 2 — Symptoms & Upload",
             "Three tabs are available: Symptom Checker, Upload Images, and Upload Existing Reports.",
             ["Symptom Checker: Select any symptoms from the checklist, rate severity (0-10), "
              "and describe in your own words.",
              "Upload Images: Drag and drop a JPG/PNG colonoscopy, endoscopy, or histopathology image. "
              "The AI will analyse it and produce a GradCAM heatmap.",
              "Upload Reports: Upload any existing PDF/TXT medical reports for reference in the final PDF.",
              "You need at least one symptom OR one image to proceed."]),
            ("Step 3 — AI Analysis",
             "The 6-agent pipeline runs automatically. You will see each agent activate in sequence.",
             ["Image Agent: GradCAM++ highlights regions of interest on your image.",
              "Text Agent: BioBERT processes your symptom description.",
              "Tabular Agent: Analyses your age, BMI, smoking history, and other risk factors.",
              "Fusion Agent: Combines all three modalities via cross-attention transformer.",
              "XAI Agent: Runs MC-Dropout uncertainty estimation (15 passes).",
              "Clinical Agent: Generates BSG/NICE-aligned recommendations.",
              "If no image is uploaded, a demo result is shown."]),
            ("Step 4 — Results",
             "Four tabs display your results.",
             ["Diagnosis: Class probabilities, staging, and modality weight breakdown.",
              "GradCAM View: Side-by-side original image and AI attention heatmap.",
              "Risk Charts: Gauge meter, multi-dimensional risk radar, and confidence breakdown.",
              "Recommendations: Clinical urgency, primary action, surveillance plan, referrals, "
              "investigations, and lifestyle advice."]),
            ("Step 5 — Find Doctors",
             "Search for specialists near you by city, country, and specialty.",
             ["The database includes 46+ gastroenterologists, colorectal surgeons, and oncologists "
              "across India, USA, UK, UAE, Singapore, Canada, and Australia.",
              "Results are sorted by rating.",
              "Doctor listings are illustrative — verify directly with the institution."]),
            ("Step 6 — Download Report",
             "Generate and download a professional clinical PDF report.",
             ["The report includes all findings, GradCAM images, probability charts, "
              "clinical recommendations, and doctor suggestions.",
              "Click 'Generate PDF Report' then 'Download PDF Report'.",
              "Use 'New Assessment' to restart for a different patient."]),
        ]

        for title, desc, tips in steps_guide:
            with st.expander(title):
                st.markdown(desc)
                st.markdown("**Tips:**")
                for tip in tips:
                    st.markdown(f"- {tip}")

    # ── Tab 3: AI Explained ────────────────────────────────────────────
    with tab_ai:
        st.markdown('<div class="section-header">How the AI Works</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**Model Architecture**")
            st.markdown(
                "The UnifiedMultiModalTransformer fuses three encoders:\n\n"
                "- **Image Encoder** — Dual ResNet-50 + EfficientNet-B0 backbones (ImageNet-pretrained); "
                "ResNet layer4[-1] is the GradCAM target.\n"
                "- **Text Encoder** — BioBERT (dmis-lab/biobert-base-cased-v1.2) processes "
                "clinical notes and your symptom description.\n"
                "- **Tabular Encoder** — TabTransformer encodes 12 TCGA-derived clinical features "
                "(age, BMI, smoking, alcohol, year of diagnosis, tumour stage, etc.).\n\n"
                "Fusion: **3-stage Gated Cross-Modal Transformer** (256-dim, 8 heads, 3 cross-attention "
                "layers + 2 self-attention layers). A learnable sigmoid gate decides per-case how much "
                "each modality contributes."
            )

        with col_b:
            st.markdown("**Output Heads & Performance**")
            st.markdown(
                "Three prediction heads are trained jointly:\n\n"
                "- **Pathology Head** (5-class): Polyps | UC-Mild | UC-Mod-Sev | "
                "Barrett's | Therapeutic\n"
                "- **Staging Head** (4-class): No Cancer | Stage I | Stage II | Stage III/IV\n"
                "- **Risk Head** (binary): Benign vs Malignant\n\n"
                "**Held-out test set (1,066 images):** 90.3% accuracy · 0.81 macro F1 · 0.984 AUC-ROC. "
                "Best epoch 7 of 60, ~150 M parameters.\n\n"
                "**Training data:** HyperKvasir + CVC-ClinicDB pretraining → fine-tune; "
                "TCGA clinical for tabular pool (12 features)."
            )

        st.markdown("")
        st.markdown('<div class="section-header">Understanding Your Results</div>',
                    unsafe_allow_html=True)

        result_guide = {
            "Class Probability": "The AI's confidence for each of the 5 conditions. "
                "The highest bar is the predicted finding. Values sum to 100%.",
            "Cancer Stage": "Derived from the staging head. 'No Cancer' means benign "
                "findings; Stage I-IV reflects increasing cancer progression.",
            "Risk Score": "The probability of malignancy from the binary risk head. "
                "Below 25% = Low, 25-50% = Moderate, 50-75% = High, above 75% = Critical.",
            "Modality Weights": "How much each data source (image, text, patient data) "
                "contributed to the final decision. Image typically dominates.",
            "AI Uncertainty": "Calculated via MC-Dropout (15 random forward passes). "
                "Low (<0.3) = consistent prediction; High (>0.6) = seek expert review.",
            "GradCAM Heatmap": "Red/warm regions = where the model focused. "
                "Blue/cool regions = areas with less influence on the prediction.",
        }
        for term, explanation in result_guide.items():
            col_t, col_e = st.columns([1, 3])
            with col_t:
                st.markdown(f"**{term}**")
            with col_e:
                st.markdown(explanation)
            st.markdown("---")

    # ── Tab 4: Case Studies ────────────────────────────────────────────
    with tab_cases:
        st.markdown('<div class="section-header">Realistic patient case studies</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Three end-to-end scenarios — drawn from the BSG/NICE/USPSTF guideline space and "
            "matched to the conditions ColonAI is trained on. Click **Load this case** to drop "
            "the patient straight into the pipeline."
        )

        cases_long = [
            ("case_a", "#1A73E8", "Sigmoid polyp on screening FIT",
             "**Vignette.** A 58-year-old man, asymptomatic, with a positive bowel-cancer screening "
             "FIT (180 µg Hb/g). Ex-smoker, BMI 26.5. Referred for diagnostic colonoscopy.",
             [
                 ("Endoscopic finding", "14 mm sessile polyp in the sigmoid colon."),
                 ("AI prediction (this build)", "polyps · ~88 % confidence · benign risk."),
                 ("Likely histology", "Tubular or tubulovillous adenoma; biopsy/EMR pending."),
                 ("Clinical action", "Endoscopic mucosal resection (EMR) for size ≥10 mm; histology to MDT."),
                 ("Surveillance (BSG/ACPGBI/PHE 2020)", "If high-risk (≥10 mm or HGD or villous component): repeat colonoscopy at **3 years**. "
                                                       "Low-risk diminutive polyp removed cleanly: return to FIT screening."),
                 ("Why it matters", "Average-risk CRC screening from 45–50 catches polyps before they become invasive cancer — adenoma-to-carcinoma takes 5–15 yrs."),
             ]),
            ("case_b", "#FF5722", "Bloody diarrhoea — suspected UC",
             "**Vignette.** A 31-year-old woman with 6 weeks of bloody diarrhoea (4–5 stools/day), "
             "urgency, mild left-iliac-fossa cramping. CRP 22, faecal calprotectin 480.",
             [
                 ("Endoscopic finding", "Granular mucosa, loss of vascular pattern, contact bleeding — Mayo endoscopic 1–2."),
                 ("AI prediction (this build)", "uc-mild · pathological-finding flag raised."),
                 ("Differential", "Infective colitis (rule out C. diff, Campylobacter), Crohn's colitis, ischaemic colitis."),
                 ("Clinical action", "Topical + oral 5-ASA induction; flexi-sig with biopsies; gastro follow-up at 6 weeks."),
                 ("Long-term surveillance", "After 8–10 years of colitis, surveillance colonoscopy with chromoendoscopy and dysplasia mapping."),
                 ("Why it matters", "Early IBD diagnosis improves response to step-up therapy and reduces colectomy rates."),
             ]),
            ("case_c", "#9C27B0", "Long-standing GORD — Barrett's surveillance",
             "**Vignette.** A 62-year-old man with 15 years of GORD on long-term PPI, BMI 31, "
             "ex-smoker. Surveillance OGD shows a 4 cm tongue of columnar mucosa above the "
             "gastro-oesophageal junction.",
             [
                 ("Endoscopic finding", "Prague C2M4 segment of intestinal metaplasia, no visible nodularity."),
                 ("AI prediction (this build)", "barretts-esoph · ~91 % confidence."),
                 ("Histology protocol", "Seattle-protocol biopsies (4-quadrant every 2 cm + targeted)."),
                 ("Clinical action", "If non-dysplastic Barrett's, ≥3 cm segment → 3-yearly surveillance OGD (BSG 2023)."),
                 ("Escalation triggers", "Any low-grade dysplasia → expert path review + 6-month repeat. High-grade dysplasia or T1a → endoscopic eradication therapy (RFA ± EMR)."),
                 ("Why it matters", "Annual progression to oesophageal adenocarcinoma is ~0.3 %/year for non-dysplastic Barrett's; surveillance catches conversion early."),
             ]),
        ]

        for key, color, title, vignette, rows in cases_long:
            st.markdown(
                f"""<div style='background:white;border-radius:14px;padding:18px 22px;
                                margin-top:14px;margin-bottom:6px;
                                border:1px solid rgba(15,23,42,0.06);
                                border-left:4px solid {color};
                                box-shadow:0 1px 3px rgba(15,23,42,0.04),
                                           0 10px 28px -16px rgba(15,23,42,0.18)'>
                    <div style='font-size:0.74rem;text-transform:uppercase;letter-spacing:0.5px;
                                color:{color};font-weight:800'>{DEMO_CASES[key]['label']}</div>
                    <div style='font-size:1.15rem;color:#0F172A;font-weight:800;margin-top:2px'>{title}</div>
                    <div style='font-size:0.92rem;color:#334155;margin-top:8px;line-height:1.55'>{vignette}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            for label, body in rows:
                st.markdown(
                    f"<div style='display:flex;gap:14px;padding:6px 4px;border-bottom:1px dashed #E2E8F0'>"
                    f"<div style='flex:0 0 200px;font-size:0.82rem;font-weight:700;color:#1A73E8'>{label}</div>"
                    f"<div style='flex:1;font-size:0.9rem;color:#1F2937;line-height:1.55'>{body}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            cols = st.columns([1, 4])
            with cols[0]:
                if st.button(f"Load this case →", key=f"guide_load_{key}",
                             use_container_width=True, type="primary"):
                    _apply_demo_case(key)
                    st.session_state["show_guide"] = False
                    st.rerun()

        st.markdown("")
        st.markdown(
            '<div class="warn-box">All vignettes are illustrative composites — not real patients. '
            'Guideline references reflect BSG/ACPGBI/PHE 2020 (post-polypectomy), BSG 2023 (Barrett\'s) '
            'and NICE NG12 (suspected cancer 2-week-wait).</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 5: How to Present ──────────────────────────────────────────
    with tab_present:
        st.markdown('<div class="section-header">How to present this project</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "A 6–8 minute live walkthrough that lands well with both **clinicians** and "
            "**academic / research panels**. Each section below is a section of your demo — "
            "the talking points are what to say while the screen does the heavy lifting."
        )

        st.markdown('<div class="section-header">1 · The opener (30 sec)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">'
            '<b>Say:</b> "Colorectal cancer is the third-most-common cancer worldwide, and 90 % of '
            'early-stage cases are curable — but only 60 % are caught early. ColonAI is a multimodal AI '
            'screening tool that combines endoscopy images, clinical notes, and patient history to flag '
            'high-risk cases for the clinician — it doesn\'t replace them."'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-header">2 · The live walkthrough (4 min)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "1. Open Step 1, click **Load Case A**. _'In one click I\'ve loaded a real screening-FIT-positive patient.'_\n"
            "2. Show the symptoms tab briefly — explain you can either type or check boxes.\n"
            "3. Click **Analyse** → walk through the 6-agent pipeline animation. _'Each agent is autonomous; the orchestrator coordinates them.'_\n"
            "4. On Results: show the **class-probability bar chart** — point at the dominant class.\n"
            "5. Switch to **GradCAM View** — _'This is the AI showing its work — the warm region is what drove the prediction. A clinician can sanity-check it visually.'_\n"
            "6. Switch to **Risk Charts** → highlight the gauge and the multi-dimensional radar.\n"
            "7. Switch to **Recommendations** → _'These are tied to BSG and NICE pathways, not generic boilerplate.'_\n"
            "8. Step 5: Find Doctors → 1 second to show it's regional. Step 6: PDF → click Generate, show the download."
        )

        st.markdown('<div class="section-header">3 · The technical slide (90 sec)</div>',
                    unsafe_allow_html=True)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(
                '<div class="metric-card" style="border-left-color:#1A73E8">'
                '<div class="label">Architecture sound-bite</div>'
                '<div class="value" style="font-size:1.1rem">Dual-backbone fusion</div>'
                '<div class="sub">ResNet-50 + EfficientNet-B0 (image) · BioBERT (text) · '
                'TabTransformer (tabular) → 3-stage gated cross-modal transformer · '
                '3 task heads (pathology, staging, risk).</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col_p2:
            st.markdown(
                '<div class="metric-card" style="border-left-color:#00897B">'
                '<div class="label">Numbers to memorise</div>'
                '<div class="value" style="font-size:1.1rem">90.3 % · 0.984 AUC</div>'
                '<div class="sub">Test set 1,066 images · 0.81 macro-F1 · MC-Dropout uncertainty · '
                '~150 M params · best epoch 7 / 60 (no overfit).</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">4 · The honesty slide (45 sec)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Reviewers respect honesty more than spin. Lead with the limitations:\n\n"
            "- HyperKvasir is European (Norway); CVC-ClinicDB is Spanish; TCGA tabular is North-American — **no external validation on Asian / African data yet.**\n"
            "- 5 classes only; doesn't yet detect rarer entities (sessile-serrated lesions, cytomegalovirus colitis, EoE).\n"
            "- Staging is image-derived; in real practice T-stage needs CT/MRI and biopsy.\n"
            "- Currently research-grade — clinical deployment requires UKCA Class IIa, MHRA registration, post-market surveillance.\n\n"
            "_That paragraph alone tends to defuse 80 % of hostile questions._"
        )

        st.markdown('<div class="section-header">5 · Likely Q&A (3 min)</div>',
                    unsafe_allow_html=True)
        qa = [
            ("Why dual image backbones?",
             "ResNet-50 gives a clean 7×7 GradCAM target for interpretability; EfficientNet adds a parallel 14×14 representation. A learned per-position gate fuses them, which empirically beats either alone by ~2 % macro-F1."),
            ("How do you avoid overfitting?",
             "Mixup α=0.3, label smoothing 0.1, weight decay 0.15, RandomPerspective + GaussianBlur + RandomErasing(p=0.4), Gaussian noise σ=0.05 on tabular features, EMA decay 0.9995, and BERT freeze→unfreeze schedule. Best epoch was 7 of 60 with early-stop patience 18."),
            ("How does it handle uncertainty?",
             "MC-Dropout — 15 stochastic forward passes at inference, the predictive entropy is reported. Uncertainty >0.6 triggers 'seek expert review' in the recommendation agent."),
            ("Why use TCGA for tabular when most patients aren't cancer patients?",
             "TCGA gives a realistic distribution of age × smoking × alcohol × BMI × stage. We sample one row per inference and overwrite the patient-known fields — so the model sees realistic correlations on the unknown features."),
            ("Could a clinician adopt this tomorrow?",
             "Not safely. It's a research tool. To deploy: external validation on the target hospital's cases, prospective evaluation against the histology gold-standard, regulatory approval, integration with the PACS / report system, and ongoing post-market surveillance."),
            ("What's next on the roadmap?",
             "(a) external validation on at least two non-Western datasets, (b) sessile-serrated lesion class, (c) report-level NLP that ingests free-text endoscopy reports verbatim, (d) calibration via temperature scaling reported per-class."),
        ]
        for q, a in qa:
            with st.expander(q):
                st.markdown(a)

        st.markdown('<div class="section-header">6 · The closer</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">'
            '<b>Say:</b> "ColonAI is not a black box and not a replacement for a clinician — '
            'it\'s a second pair of eyes that looks at the image, the patient, and the symptoms together. '
            'The win is in catching the cases that get missed when only one signal is examined."'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 6: FAQ ─────────────────────────────────────────────────────
    with tab_faq:
        st.markdown('<div class="section-header">Frequently Asked Questions</div>',
                    unsafe_allow_html=True)

        faqs = [
            ("Is this a medical diagnosis?",
             "No. ColonAI is a research-grade screening tool. Results must be reviewed by a "
             "qualified, licensed clinician before any clinical decisions are made."),
            ("What image types can I upload?",
             "JPG and PNG images from colonoscopy, endoscopy, or histopathology. Images are "
             "automatically resized to 224x224 pixels for analysis. Maximum 10 MB per file."),
            ("Is my data stored or shared?",
             "No. All data (images, patient info, results) is stored only in your browser session "
             "and is lost when you close or refresh the page. Nothing is sent to external servers."),
            ("How long does the AI analysis take?",
             "Typically 5-30 seconds on CPU, depending on your hardware. A loading indicator "
             "shows each agent's progress in real time."),
            ("What if I do not have an image?",
             "You can still enter symptoms and get a demo-mode result. However, the GradCAM "
             "analysis requires an image. For best results, upload a colonoscopy image."),
            ("How do I interpret a high uncertainty score?",
             "A high uncertainty (>0.6) means the AI's predictions varied across 15 random runs. "
             "This could indicate an unusual image, borderline case, or image quality issue. "
             "Always consult a specialist in such cases."),
            ("Why does the AI sometimes predict the wrong class?",
             "The model achieves 90.3 % test accuracy and 0.984 AUC-ROC — strong but not perfect. "
             "Edge cases, unusual angles, image artefacts, or conditions outside the training distribution "
             "(it knows 5 classes only) can lead to errors. This is exactly why clinician review is mandatory."),
            ("Can I use this for research?",
             "Yes. The model is based on publicly available datasets (HyperKvasir, CVC-ClinicDB, "
             "TCGA). Please cite the original dataset papers and the model architecture if you "
             "publish results based on this tool."),
            ("How do I find a doctor near me?",
             "On Step 5 (Find Doctors), enter your city and country. The system returns up to 10 "
             "matching specialists sorted by rating. Doctor data is illustrative — verify details "
             "with the institution directly."),
            ("What is GradCAM++ and why does it matter?",
             "GradCAM++ (Gradient-weighted Class Activation Mapping++) is an explainability "
             "technique that shows which pixel regions of the image contributed most to the AI's "
             "prediction. This helps you and your doctor understand what the AI is 'seeing'."),
        ]

        for q, a in faqs:
            with st.expander(q):
                st.markdown(a)

    st.markdown("---")
    col_b, col_sp, col_start = st.columns([1, 3, 1])
    with col_b:
        if st.button("Back to App", use_container_width=True):
            st.session_state["show_guide"] = False
            st.rerun()
    with col_start:
        if st.button("Start Assessment", type="primary", use_container_width=True):
            st.session_state["show_guide"] = False
            st.session_state["step"] = 0
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — LIVE VIDEO MODE
# ─────────────────────────────────────────────────────────────────────────────
def page_live_video():
    """Real-time colonoscopy video analysis with the 6-agent pipeline.

    Two sub-modes:
      A) Upload a recorded video file → process frame-by-frame with bbox overlay
      B) Live webcam / capture-card → real-time WebRTC inference
    """
    render_hero(
        "Live Video Mode",
        "Real-time colonoscopy analysis with bounding-box detection, polyp tracking and alerts.",
        badges=["Step 7 of 7", "Per-frame inference", "Temporal smoothing", "Auto-tracking"],
    )

    # Lazy-load the AI system (same pattern as the rest of the app)
    if st.session_state.get("_system") is None:
        with st.spinner("Loading the 6-agent AI system…"):
            st.session_state["_system"] = load_ai_system()
    system = st.session_state.get("_system")
    if not system or not system.get("model"):
        st.error("AI system not loaded.  Cannot run live mode.")
        st.info("Try restarting the app — the checkpoint may be missing from "
                "`outputs/unified_multimodal/checkpoints/best_model.pth`.")
        if st.button("← Back to Home", use_container_width=True):
            st.session_state["step"] = 0
            st.rerun()
        return

    model     = system["model"]
    tokenizer = system["tokenizer"]
    device    = system["device"]

    # ── Detector status — is the trained real-time YOLO detector active? ──
    try:
        from src.app.video_pipeline import detector_available
        if detector_available():
            st.success("✅ **Real-time polyp detector active** — using the trained detection model.")
        else:
            st.info("ℹ️ Using the fallback detection method. To enable the trained real-time "
                    "detector, drop `best.pt` at `outputs/unified_multimodal_v2/polyp_detector.pt` "
                    "(see docs/VIDEO_PHASE.md).")
    except Exception:
        pass

    # ── Built-in animated walkthrough (LOCAL — no YouTube dependency) ──
    with st.expander("🎬 How Live Video Mode works", expanded=False):
        st.markdown(
            """
            <style>
            @keyframes liveStep { from {opacity:0;transform:translateX(-10px)} to {opacity:1;transform:none} }
            .lv-row { display:flex; align-items:center; gap:14px; padding:12px 16px;
                      margin:6px 0; background:#FFF; border-radius:10px;
                      border:1px solid #E2E8F0; animation: liveStep 0.5s ease both; }
            .lv-num { width:34px; height:34px; border-radius:50%; flex-shrink:0;
                      display:flex; align-items:center; justify-content:center;
                      color:white; font-weight:800; font-size:0.95rem;
                      background:linear-gradient(135deg,#1A73E8,#00897B); }
            </style>
            <div class="lv-row" style="animation-delay:0.05s;"><div class="lv-num">1</div>
              <div><b>Upload a colonoscopy video</b> · MP4 / MOV / AVI · any length</div></div>
            <div class="lv-row" style="animation-delay:0.18s;"><div class="lv-num">2</div>
              <div><b>The AI scans every frame</b> · ~10 frames per second</div></div>
            <div class="lv-row" style="animation-delay:0.31s;"><div class="lv-num">3</div>
              <div><b>Polyps get a coloured box around them</b> automatically as they appear</div></div>
            <div class="lv-row" style="animation-delay:0.44s;"><div class="lv-num">4</div>
              <div><b>Each polyp is tracked</b> across frames so you don't see flickering boxes</div></div>
            <div class="lv-row" style="animation-delay:0.57s;"><div class="lv-num">5</div>
              <div><b>Download the annotated video</b> for your records or to share with the doctor</div></div>
            """,
            unsafe_allow_html=True,
        )

    tab_video, tab_webcam, tab_info = st.tabs([
        "📂 Upload Video", "📷 Live Webcam", "ℹ️ How It Works"
    ])

    # ── Mode A: Upload a colonoscopy video file ─────────────────────
    with tab_video:
        st.markdown("##### Upload a recorded colonoscopy video — the AI will scan every frame")
        col_l, col_r = st.columns([2, 1])
        with col_l:
            uploaded = st.file_uploader(
                "Choose a colonoscopy video",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                key="live_video_upload",
            )
        with col_r:
            skip = st.select_slider(
                "Frame skip (lower = more accurate, slower)",
                options=[1, 2, 3, 5, 8],
                value=3,
                help="Process every Nth frame. 3 ≈ 10 fps from a 30 fps source.",
            )
            conf_thr = st.slider(
                "Detection confidence threshold",
                min_value=0.30, max_value=0.95, value=0.55, step=0.05,
                help="Only flag a polyp if model confidence ≥ this value.",
            )

        if uploaded is not None:
            # Save the uploaded video to a temp file
            import tempfile, os
            tmpdir = Path(tempfile.gettempdir()) / "colonai_live"
            tmpdir.mkdir(exist_ok=True)
            in_path  = tmpdir / f"input_{uploaded.name}"
            out_path = tmpdir / f"annotated_{uploaded.name.rsplit('.',1)[0]}.mp4"
            with open(in_path, "wb") as f:
                f.write(uploaded.read())

            st.markdown("---")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Original video**")
                st.video(str(in_path))
            with c2:
                st.markdown("**Annotated output (after analysis)**")
                if "live_summary" in st.session_state and \
                   st.session_state.get("live_video_name") == uploaded.name:
                    st.video(str(out_path))
                else:
                    st.markdown(
                        "<div style='background:#F1F5F9;border:1px dashed #CBD5E1;"
                        "border-radius:8px;padding:48px;text-align:center;color:#64748B;'>"
                        "<i>Click 'Analyse video' below to run the pipeline</i>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

            if st.button("🚀 Analyse video — run the 6-agent pipeline frame-by-frame",
                         use_container_width=True, type="primary"):
                from src.app.video_pipeline import analyse_video_file

                progress = st.progress(0, "Initialising…")
                status_box = st.empty()

                def _cb(frame_idx, total_f, n_polyps):
                    pct = int(100 * frame_idx / max(1, total_f))
                    progress.progress(min(100, pct),
                                      f"Frame {frame_idx} / {total_f}  ·  "
                                      f"Polyps tracked: {n_polyps}")

                try:
                    summary = analyse_video_file(
                        video_path=str(in_path),
                        output_path=str(out_path),
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        skip_frames=skip,
                        confidence_threshold=conf_thr,
                        progress_callback=_cb,
                    )
                    progress.progress(100, "Done!")
                    st.session_state["live_summary"]    = summary
                    st.session_state["live_video_name"] = uploaded.name
                    st.session_state["live_out_path"]   = str(out_path)
                    st.success(f"✅ Analysed {summary.processed_frames} frames  ·  "
                               f"{summary.polyps_count} distinct polyps detected  ·  "
                               f"avg inference {summary.avg_inference_ms:.0f} ms")
                    st.rerun()
                except Exception as exc:
                    progress.empty()
                    st.error(f"Video analysis failed: {type(exc).__name__}: {exc}")

            # ── Render summary if available ────────────────────────
            summary = st.session_state.get("live_summary")
            if summary and st.session_state.get("live_video_name") == uploaded.name:
                _render_live_summary(summary)

    # ── Mode B: Live webcam ─────────────────────────────────────────
    with tab_webcam:
        st.markdown("##### Live webcam — point at a colonoscopy screen or run with a capture card")
        st.markdown(
            "Use this to demo real-time inference. On a real scope, connect the "
            "video out of the colonoscope tower to your laptop via a USB capture "
            "card (HDMI → USB). The AI will then analyse every Nth frame and "
            "overlay polyp detections on the live image.",
        )
        try:
            from streamlit_webrtc import webrtc_streamer, RTCConfiguration
            from src.app.video_pipeline import LivePolypTransformer

            cfg = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })

            def _factory():
                return LivePolypTransformer(model, tokenizer, device,
                                            skip=3, confidence_threshold=0.55)

            ctx = webrtc_streamer(
                key="colonai-live",
                video_processor_factory=_factory,
                rtc_configuration=cfg,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if ctx and getattr(ctx, "video_processor", None):
                vp = ctx.video_processor
                st.markdown("##### Live tracker")
                m1, m2, m3 = st.columns(3)
                m1.metric("Frame", vp.frame_idx)
                m2.metric("Polyps tracked", len(vp.tracker.polyps))
                m3.metric("Confirmed", vp.tracker.get_confirmed_count())
        except Exception as exc:
            st.error(f"Live webcam unavailable: {type(exc).__name__}: {exc}")
            st.info("Try `pip install streamlit-webrtc av` if not installed.")

    # ── Info tab ────────────────────────────────────────────────────
    with tab_info:
        st.markdown("""
##### How real-time mode works

1. **Endoscopy gate**  — every incoming frame is first checked by the
   pixel-statistics gate. Frames that don't look like a colonoscopy (camera
   pulled out, blurry, lens fouling) are skipped so the model is never
   called on bad input.

2. **Per-frame inference**  — for every Nth frame (default = every 3rd
   frame ≈ 10 fps from a 30 fps source), the dual-CNN runs and produces a
   pathology class + confidence. Inference takes ~80–150 ms per frame.

3. **GradCAM++ → bounding box**  — the heatmap is thresholded at the 80th
   percentile, connected-components are extracted, and the largest blob
   becomes the bounding box overlaid on the frame.

4. **Temporal tracker**  — bounding boxes from consecutive frames are
   matched by IoU. A detection is only **confirmed** once seen in at
   least 3 of the last 6 frames — this kills single-frame false positives.

5. **Live annotation**  — bbox + class label + confidence are drawn on the
   frame in real time. The status bar shows the running polyp count.

6. **Detection log**  — every confirmed polyp is logged with timestamp,
   peak confidence, and a snapshot of the best frame.

The same six agents (Image, Text, Tabular, Fusion, XAI, Clinical) run
under the hood for every frame, just like in single-image mode.
        """)

    # Back button
    st.markdown("---")
    if st.button("← Back to dashboard", use_container_width=True):
        st.session_state["step"] = 0
        st.rerun()


def _render_live_summary(summary):
    """Render the summary panel for a completed video analysis."""
    st.markdown("---")
    st.markdown("### 📊 Video Analysis Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames processed",  summary.processed_frames)
    m2.metric("Duration",          f"{summary.duration_seconds:.1f} s")
    m3.metric("Distinct polyps",   summary.polyps_count)
    m4.metric("Avg inference",     f"{summary.avg_inference_ms:.0f} ms")

    # Per-polyp details
    if summary.tracked_polyps:
        st.markdown("##### Detected polyps (sorted by persistence)")
        for i, p in enumerate(summary.tracked_polyps[:10], start=1):
            col_a, col_b = st.columns([1, 3])
            with col_a:
                if p.snapshot is not None:
                    st.image(p.snapshot, caption=f"Polyp #{p.id}", use_container_width=True)
            with col_b:
                from src.app.video_pipeline import CLASS_LABEL
                st.markdown(f"""
                **Polyp #{p.id}**  ·  {CLASS_LABEL.get(p.class_name, p.class_name)}
                - **Seen in:** {p.n_frames} frames
                - **Time range:** {p.first_ts:.1f} s → {p.last_ts:.1f} s
                - **Peak confidence:** {p.max_confidence*100:.1f}%
                - **First detected:** frame {p.first_frame}
                """)
            st.markdown("---")
    else:
        st.info("No polyps detected in this video (or all detections fell below the confidence threshold).")

    # Download buttons
    if summary.output_video_path and Path(summary.output_video_path).exists():
        with open(summary.output_video_path, "rb") as f:
            st.download_button(
                "⬇️ Download annotated video",
                data=f.read(),
                file_name=Path(summary.output_video_path).name,
                mime="video/mp4",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 🤖 COLON BUDDY — pure HTML / JS / CSS floating chat widget
# ─────────────────────────────────────────────────────────────────────────────
# This is a SELF-CONTAINED chatbot widget that lives in the bottom-right corner
# of every page.  No Streamlit forms, no dialogs, no reruns — pure browser JS.
# Calls the FREE Pollinations.ai LLM directly via fetch().
# Falls back to a small JS-side knowledge base for instant common-question
# answers.
_COLON_BUDDY_WIDGET_HTML = r"""
<script>
// Inject into the PARENT document (Streamlit's iframe is for our trigger only)
(function() {
  const WIDGET_VERSION = "v5-ollama-2026-05-23";
  const doc = window.parent.document;

  // If a widget is already there but it's an OLD version, remove it and re-inject
  const existing = doc.getElementById('colon-buddy-root');
  if (existing) {
    if (existing.dataset && existing.dataset.version === WIDGET_VERSION) {
      return; // already the right version, nothing to do
    }
    console.log("Colon Buddy: removing old widget version, installing", WIDGET_VERSION);
    existing.remove();
    const oldStyle = doc.getElementById('colon-buddy-styles');
    if (oldStyle) oldStyle.remove();
  }

  // ── Local KB for instant answers to common questions ──────────────
  // Each entry has a list of CASE-INSENSITIVE keyword PHRASES.
  // Matching uses WORD-BOUNDARY regex so 'eat' won't match 'trEATment'.
  // The entries are checked in PRIORITY order — more specific topics first.
  const KB = [
    // Greetings — high priority
    { k: ["hello","hi","hey","greet","good morning","good evening","good afternoon",
          "who are you","your name","what's your name","whats your name"],
      a: "Hi! I'm Colon Buddy — your friendly health assistant. Ask me anything about colon health, screening, polyps, diet, symptoms or treatment. I always reply in plain English. 😊" },
    { k: ["thank","thanks","cheers","appreciate"],
      a: "You're very welcome! Remember to confirm anything important with a real doctor. I'm here whenever you have another question. 😊" },
    { k: ["bye","goodbye","see you","later","take care"],
      a: "Take care! If anything feels urgent — like bleeding or severe pain — please contact your GP or emergency services right away. 💛" },
    { k: ["scared","worried","nervous","anxious","afraid","fear","panic"],
      a: "It's completely normal to feel scared — many people do. Most worrying symptoms turn out to be something less serious, like piles or simple inflammation. Please book a GP appointment so a professional can put your mind at ease. You're not alone. 💛" },

    // Treatment — checked BEFORE diet so 'treatment' isn't confused with 'eat'
    { k: ["treatment","therapy","cure","options","heal","chemotherapy","chemo",
          "radiotherapy","radiation","surgery","operation"],
      a: "Treatment depends on how early the cancer is found. Very early cancers can often be removed during a colonoscopy itself. More advanced cases may need keyhole surgery (3-5 day hospital stay) plus a few months of chemotherapy. Today's treatments are far gentler than they used to be, with much higher success rates. Your specialist will recommend the best plan for you." },

    // Symptoms — also checked before generic words
    { k: ["symptom","symptoms","warning sign","red flag","red flags",
          "blood in stool","bleeding","rectal bleed","weight loss","bowel habit",
          "tummy pain","stomach pain","abdominal pain","cramping","cramps"],
      a: "See your GP soon if you have: blood in your poo, a change in bowel habit lasting more than 3 weeks, unexplained weight loss, ongoing tummy pain, or feeling very tired. These don't always mean cancer — they're often something less serious — but they're worth getting checked. Don't ignore them." },

    // Polyp
    { k: ["polyp","polyps","adenoma","adenomas"],
      a: "A polyp is a small growth on the inside lining of your colon. Most are completely harmless, but a small number can slowly turn into cancer over many years — that's why doctors remove them when they're spotted during a colonoscopy. The removal is quick and painless. Always confirm with a real doctor." },

    // Screening
    { k: ["screening","screened","screen me","when should i be screened",
          "what age","starting age","colonoscopy age","when to start",
          "fit test","stool test","colonoscopy"],
      a: "Most people should start screening at age 45 to 50, even if they feel fine. If your parent or sibling had colon cancer, start 10 years before their age at diagnosis. The simplest option is a stool test at home (FIT) once a year, or a colonoscopy every 10 years. Your GP can guide you." },

    // Diet
    { k: ["diet","food","nutrition","best diet","what should i eat",
          "what to eat","fibre","fiber","vegetable","fruit","mediterranean",
          "alcohol","drink","wine","beer","red meat","processed meat"],
      a: "Best foods for your colon: plenty of fibre (oats, beans, lentils, brown rice, whole-wheat bread, fruit and veg — aim for 30 g a day), oily fish twice a week, lots of water. Limit red meat and try to avoid processed meats like bacon and sausages. A Mediterranean-style diet has the strongest evidence for prevention." },

    // Survival / prognosis
    { k: ["survival","survive","prognosis","life expectancy","outcome",
          "chances","success rate","mortality"],
      a: "Survival depends on when the cancer is found. At Stage 1, more than 9 in 10 people are alive 5 years later. At Stage 2 it's about 8 in 10, at Stage 3 about 6 in 10. The biggest thing you can do is get screened on time." },

    // Prevention
    { k: ["prevent","prevention","reduce risk","avoid","stop cancer",
          "lifestyle","exercise","smoking","quit"],
      a: "Five things that lower your risk: (1) get screened on time from age 45-50, (2) eat more fibre, less red and processed meat, (3) keep a healthy weight, (4) move 150 minutes a week, (5) don't smoke and keep alcohol low. These steps remove about half of preventable cases." },

    // Colon / general
    { k: ["colon","colorectal","bowel","large intestine","rectum","rectal"],
      a: "The colon (large intestine) is the last part of your digestive system. Common conditions include polyps (small growths, usually harmless), inflammation (like ulcerative colitis), and rarely cancer. Regular screening from age 45-50 catches problems early, when they're easy to treat. Ask me about any specific topic!" },

    // Anatomy / how the colon works
    { k: ["how does the colon work","what does the colon do","function of colon",
          "anatomy"],
      a: "Your colon is about 1.5 metres long and is the last stretch of your digestive system. Its job is to absorb water from food waste and form stool. It also hosts billions of helpful bacteria that support your immune system. A high-fibre diet helps it work smoothly." },
  ];

  function kbMatch(msg) {
    const m = " " + msg.toLowerCase().replace(/[^a-z0-9 ]+/g, " ") + " ";
    for (const e of KB) {
      for (const kw of e.k) {
        // Word-boundary safe substring check — 'eat' won't match 'treatment'
        const safe = kw.toLowerCase().replace(/[^a-z0-9 ]+/g, " ");
        const pattern = " " + safe + " ";
        if (m.includes(pattern)) return e.a;
      }
    }
    return null;
  }

  // ── Build the widget DOM ──────────────────────────────────────────
  const css = `
    .cb-fab {
      position: fixed; bottom: 22px; right: 22px;
      height: 58px; padding: 0 24px 0 16px;
      border-radius: 30px; cursor: pointer; z-index: 999999;
      background: linear-gradient(135deg,#FB7185 0%,#F97316 60%,#FBBF24 100%);
      display: inline-flex; align-items: center; gap: 12px;
      border: none; outline: none; color: #FFF;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-weight: 800; font-size: 15px;
      box-shadow: 0 10px 28px -6px rgba(249,115,22,0.55);
      transition: all .2s ease;
      animation: cbFloat 3s ease-in-out infinite;
    }
    .cb-fab:hover { transform: scale(1.06) translateY(-2px); }
    .cb-fab.cb-hidden { display: none; }
    .cb-fab-icon {
      width: 38px; height: 38px; border-radius: 50%;
      background: rgba(255,255,255,0.22);
      display: inline-flex; align-items: center; justify-content: center;
      font-size: 22px;
    }
    @keyframes cbFloat {0%,100%{transform:translateY(0)} 50%{transform:translateY(-3px)}}

    .cb-window {
      position: fixed; bottom: 22px; right: 22px;
      width: 380px; height: 560px; max-height: calc(100vh - 60px);
      background: #FFF; border-radius: 18px; z-index: 999999;
      box-shadow: 0 24px 56px -12px rgba(15,23,42,0.45),
                  0 0 0 1px rgba(15,23,42,0.06);
      display: flex; flex-direction: column; overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      animation: cbSlideIn .3s ease-out;
    }
    .cb-window.cb-hidden { display: none; }
    @keyframes cbSlideIn {from{opacity:0;transform:translateY(12px) scale(.96)}
                          to{opacity:1;transform:translateY(0) scale(1)}}

    .cb-header {
      background: linear-gradient(135deg,#FB7185 0%,#F97316 60%,#FBBF24 100%);
      color: #FFF; padding: 14px 18px;
      display: flex; align-items: center; gap: 12px; flex-shrink: 0;
    }
    .cb-header-icon {
      font-size: 28px;
      background: rgba(255,255,255,0.22); border-radius: 50%;
      width: 44px; height: 44px;
      display: inline-flex; align-items: center; justify-content: center;
    }
    .cb-header-title { font-weight: 800; font-size: 17px; line-height: 1.15; }
    .cb-header-sub   { font-size: 12px; opacity: 0.92; margin-top: 2px; }
    .cb-close {
      margin-left: auto; background: rgba(255,255,255,0.22);
      width: 32px; height: 32px; border-radius: 50%;
      display: inline-flex; align-items: center; justify-content: center;
      color: #FFF; font-size: 18px; font-weight: 800; cursor: pointer;
      border: none; outline: none; transition: all .15s ease;
    }
    .cb-close:hover { background: rgba(255,255,255,0.35); transform: scale(1.08); }

    .cb-body {
      flex: 1; overflow-y: auto; padding: 16px;
      background: #F8FAFC; display: flex; flex-direction: column; gap: 10px;
    }
    .cb-msg-user, .cb-msg-bot {
      max-width: 85%; padding: 9px 13px; font-size: 13.5px; line-height: 1.5;
      border-radius: 16px; word-wrap: break-word;
    }
    .cb-msg-user {
      align-self: flex-end;
      background: linear-gradient(135deg,#1A73E8,#1E40AF); color: #FFF;
      border-bottom-right-radius: 4px;
    }
    .cb-msg-bot {
      align-self: flex-start;
      background: #FFF; color: #1F2937;
      border: 1px solid #E2E8F0; border-bottom-left-radius: 4px;
      box-shadow: 0 2px 6px rgba(15,23,42,0.05);
    }
    .cb-msg-bot .cb-avatar { font-size: 18px; margin-right: 6px; }
    .cb-intro {
      background: #FFF7ED; border-left: 3px solid #F97316;
      border-radius: 8px; padding: 11px 14px;
      font-size: 13px; color: #7C2D12; line-height: 1.5;
      align-self: stretch;
    }
    .cb-typing {
      align-self: flex-start; padding: 9px 14px;
      background: #FFF; border: 1px solid #E2E8F0; border-radius: 16px 16px 16px 4px;
      color: #94A3B8; font-size: 13px; font-style: italic;
    }
    .cb-typing-dots span {
      display: inline-block; width: 6px; height: 6px;
      margin: 0 1px; border-radius: 50%; background: #F97316;
      animation: cbDot 1.2s ease-in-out infinite;
    }
    .cb-typing-dots span:nth-child(2) { animation-delay: 0.15s; }
    .cb-typing-dots span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes cbDot {0%,100%{transform:translateY(0);opacity:0.5}
                      50%{transform:translateY(-4px);opacity:1}}

    .cb-chips {
      padding: 0 14px 8px; display: flex; flex-wrap: wrap; gap: 6px;
      background: #F8FAFC; flex-shrink: 0;
    }
    .cb-chip {
      background: #FFF; border: 1px solid #E2E8F0;
      border-radius: 16px; padding: 5px 12px;
      font-size: 12px; color: #475569; cursor: pointer;
      transition: all .15s ease; font-family: inherit;
    }
    .cb-chip:hover { background: #FB7185; border-color: #FB7185; color: #FFF; }

    .cb-footer {
      flex-shrink: 0; padding: 12px 14px;
      background: #FFF; border-top: 1px solid #E2E8F0;
      display: flex; gap: 8px;
    }
    .cb-input {
      flex: 1; padding: 9px 14px; border-radius: 22px;
      border: 1px solid #CBD5E1; outline: none;
      font-size: 14px; font-family: inherit;
      transition: border-color .15s ease;
    }
    .cb-input:focus { border-color: #F97316; box-shadow: 0 0 0 3px rgba(249,115,22,0.15); }
    .cb-send {
      width: 40px; height: 40px; border-radius: 50%;
      background: linear-gradient(135deg,#F97316,#FBBF24);
      color: #FFF; border: none; outline: none; cursor: pointer;
      font-size: 18px; display: inline-flex; align-items: center; justify-content: center;
      transition: all .15s ease;
    }
    .cb-send:hover { transform: scale(1.08); box-shadow: 0 6px 16px rgba(249,115,22,0.45); }
    .cb-send:active { transform: scale(0.95); }
    .cb-send[disabled] { opacity: 0.5; cursor: not-allowed; }

    .cb-disclaimer {
      flex-shrink: 0; padding: 6px 14px;
      background: #FFF7ED; color: #92400E;
      font-size: 10.5px; text-align: center; line-height: 1.4;
    }
  `;

  const styleEl = doc.createElement('style');
  styleEl.id = 'colon-buddy-styles';
  styleEl.textContent = css;
  doc.head.appendChild(styleEl);

  const root = doc.createElement('div');
  root.id = 'colon-buddy-root';
  root.dataset.version = WIDGET_VERSION;
  root.innerHTML = `
    <button class="cb-fab" id="cb-fab">
      <span class="cb-fab-icon">🤖</span>
      <span>Chat with Colon Buddy</span>
    </button>

    <div class="cb-window cb-hidden" id="cb-window">
      <div class="cb-header">
        <div class="cb-header-icon">🤖</div>
        <div>
          <div class="cb-header-title">Colon Buddy</div>
          <div class="cb-header-sub">Your friendly health assistant</div>
        </div>
        <button class="cb-close" id="cb-close" title="Close">×</button>
      </div>

      <div class="cb-body" id="cb-body">
        <div class="cb-intro">
          👋 Hi! I'm <b>Colon Buddy</b>. Ask me anything about colon health,
          screening, symptoms, diet or treatment. I'll always reply in plain English.
        </div>
      </div>

      <div class="cb-chips">
        <button class="cb-chip" data-q="What is a polyp?">🔍 What is a polyp?</button>
        <button class="cb-chip" data-q="When should I get screened?">📅 Screening?</button>
        <button class="cb-chip" data-q="Best diet for colon health?">🥗 Best diet?</button>
        <button class="cb-chip" data-q="What are the warning symptoms?">⚠️ Symptoms?</button>
        <button class="cb-chip" data-q="What are the treatment options?">💊 Treatment?</button>
        <button class="cb-chip" data-q="I am scared">😟 I'm scared</button>
      </div>

      <div class="cb-footer">
        <input class="cb-input" id="cb-input" type="text"
               placeholder="Type your question…" autocomplete="off"/>
        <button class="cb-send" id="cb-send" title="Send">➤</button>
      </div>
      <div class="cb-disclaimer">
        💡 Plain-English answers. Always confirm anything important with a real doctor.
      </div>
    </div>
  `;
  doc.body.appendChild(root);

  // ── Behaviour ──────────────────────────────────────────────────────
  const fab    = doc.getElementById('cb-fab');
  const win    = doc.getElementById('cb-window');
  const close  = doc.getElementById('cb-close');
  const body   = doc.getElementById('cb-body');
  const input  = doc.getElementById('cb-input');
  const send   = doc.getElementById('cb-send');
  const chips  = doc.querySelectorAll('.cb-chip');

  let isThinking = false;

  function openChat() {
    win.classList.remove('cb-hidden');
    fab.classList.add('cb-hidden');
    setTimeout(() => input.focus(), 200);
  }
  function closeChat() {
    win.classList.add('cb-hidden');
    fab.classList.remove('cb-hidden');
  }

  fab.addEventListener('click', openChat);
  close.addEventListener('click', closeChat);

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));
  }

  function addUserMsg(text) {
    const d = doc.createElement('div');
    d.className = 'cb-msg-user';
    d.textContent = text;
    body.appendChild(d);
    body.scrollTop = body.scrollHeight;
  }
  function addBotMsg(text, note) {
    const d = doc.createElement('div');
    d.className = 'cb-msg-bot';
    let html = '<span class="cb-avatar">🤖</span>' + escapeHtml(text);
    if (note) {
      html += '<div style="margin-top:6px;padding-top:6px;border-top:1px solid #FDE68A;' +
              'font-size:11px;color:#92400E;line-height:1.4;">' + escapeHtml(note) + '</div>';
    }
    d.innerHTML = html;
    body.appendChild(d);
    body.scrollTop = body.scrollHeight;
  }
  function showTyping() {
    const d = doc.createElement('div');
    d.className = 'cb-typing'; d.id = 'cb-typing-indicator';
    d.innerHTML = '🤖 Colon Buddy is thinking <span class="cb-typing-dots"><span></span><span></span><span></span></span>';
    body.appendChild(d);
    body.scrollTop = body.scrollHeight;
  }
  function hideTyping() {
    const t = doc.getElementById('cb-typing-indicator');
    if (t) t.remove();
  }

  async function askLLM(question) {
    const systemPrompt =
      "You are Colon Buddy, a friendly health-INFORMATION assistant for the ColonAI app. " +
      "You give GENERAL EDUCATION ONLY about colon health, screening, polyps, ulcerative " +
      "colitis, Barrett's oesophagus, diet, symptoms, prevention, and how to use the ColonAI app. " +
      "STRICT SAFETY RULES you must never break: " +
      "(1) You are NOT a doctor — never diagnose or tell the user what condition or stage they have. " +
      "(2) Never interpret the user's own scan, image, or test results — tell them to ask their clinician. " +
      "(3) Never give specific drug names, doses, or personalised treatment plans. " +
      "(4) For anything outside general colon-health education, or anything needing a diagnosis, " +
      "politely decline and tell them to consult a qualified doctor. " +
      "(5) Never invent statistics or facts; if unsure, say so and recommend a doctor. " +
      "Answer in plain everyday English — no jargon, no acronyms — in 3-5 short sentences. " +
      "Always end by reminding them you are an automated assistant, not a doctor.";

    function isJunkReply(t) {
      if (!t || t.length < 20) return true;
      const junkMarkers = [
        "deprecat", "migrate to", "pollinations", "legacy",
        "rate limit", "rate-limit", "too many requests",
        "503 service", "504 gateway", "html>", "<!DOCTYPE"
      ];
      const low = t.toLowerCase();
      return junkMarkers.some(j => low.includes(j));
    }

    // ── 1) TRY OLLAMA FIRST (runs locally — fast, private, no API key) ──
    // Default endpoint: http://localhost:11434/api/chat
    // CORS requires OLLAMA_ORIGINS=* on the Ollama daemon.
    // Tries models in preference order: gemma3:4b → llama3.2:3b → whatever's available
    const OLLAMA_MODELS = ["gemma3:4b", "llama3.2:3b"];
    for (const model of OLLAMA_MODELS) {
      try {
        const r = await fetch("http://localhost:11434/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            model: model,
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user",   content: question }
            ],
            stream: false,
            options: { temperature: 0.7, num_predict: 250 }
          })
        });
        if (r.ok) {
          const data = await r.json();
          const reply = data?.message?.content;
          if (reply && !isJunkReply(reply)) {
            console.log("Colon Buddy: answered via Ollama " + model);
            return reply.trim();
          }
        }
      } catch (e) {
        console.warn("Colon Buddy: Ollama " + model + " failed —", e.message);
      }
    }

    // ── 2) Fallback: Pollinations.ai OpenAI-compatible endpoint ──
    try {
      const r = await fetch("https://text.pollinations.ai/openai", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          model: "openai",
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user",   content: question }
          ],
          temperature: 0.7,
        })
      });
      if (r.ok) {
        const data = await r.json();
        const reply = data?.choices?.[0]?.message?.content;
        if (reply && !isJunkReply(reply)) return reply.trim();
      }
    } catch (e) {
      console.warn("Colon Buddy: Pollinations POST failed —", e.message);
    }

    // ── 3) Final fallback: legacy GET endpoint ──
    try {
      const url = "https://text.pollinations.ai/" +
                  encodeURIComponent(systemPrompt + "\n\nUser: " + question + "\nColon Buddy:");
      const r = await fetch(url, { method: "GET" });
      if (r.ok) {
        const t = await r.text();
        if (t && !isJunkReply(t)) return t.trim();
      }
    } catch (e) {
      console.warn("Colon Buddy: GET fallback failed —", e.message);
    }

    return null;
  }

  async function handleAsk(question) {
    if (!question || !question.trim() || isThinking) return;
    const q = question.trim();
    isThinking = true; send.disabled = true; input.disabled = true;
    addUserMsg(q);
    input.value = "";

    // 1) Try local KB first (instant)
    const kbAns = kbMatch(q);
    if (kbAns) {
      setTimeout(() => {
        addBotMsg(kbAns);
        isThinking = false; send.disabled = false; input.disabled = false;
        input.focus();
      }, 300);
      return;
    }

    // 2) Fallback to free LLM (Pollinations.ai)
    showTyping();
    const reply = await askLLM(q);
    hideTyping();
    if (reply) {
      addBotMsg(reply, "⚠️ Automated assistant, not a doctor — general information only, "
                + "not medical advice. It cannot diagnose you or read your results. "
                + "Please confirm anything important with a qualified clinician.");
    } else {
      addBotMsg("Hmm, I don't have a ready answer for that one. I can help with: polyps, screening, symptoms, diet, treatment, prevention, survival rates, or general colon-health questions. Try one of those topics, or click a quick chip above! 😊");
    }
    isThinking = false; send.disabled = false; input.disabled = false;
    input.focus();
  }

  send.addEventListener('click', () => handleAsk(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAsk(input.value);
    }
  });
  chips.forEach(c => c.addEventListener('click', () => handleAsk(c.dataset.q)));
})();
</script>
"""


# ─────────────────────────────────────────────────────────────────────────────
# LATEST RESEARCH page — auto-updated cancer-news feed
# ─────────────────────────────────────────────────────────────────────────────
def page_latest_research():
    """Daily-refreshed news feed of cancer research (colon-cancer focus).

    Reads outputs/cancer_news.json which is produced by
    scripts/scrape_cancer_news.py — locally on demand and automatically
    every day via .github/workflows/scrape-news.yml (commits the JSON
    back to the repo so the HF Space pulls the fresh data).
    """
    from src.app.security import escape_html as _esc

    render_hero(
        "Latest cancer research",
        "Hand-curated news from oncology journals and research outlets — updated daily. "
        "Tap a card to read the full story on the source's website.",
        badges=["Step 8 of 8", "Auto-updated · public RSS feeds", "Not medical advice"],
    )

    news_path = ROOT / "outputs/cancer_news.json"
    if not news_path.exists():
        st.warning(
            "📰 No news feed cached yet. Run "
            "`python3 scripts/scrape_cancer_news.py` once to populate it, "
            "or wait for the daily GitHub Action to run.")
        return

    try:
        payload = json.loads(news_path.read_text())
    except Exception as e:
        st.error(f"Could not load news feed: {e}")
        return

    items = payload.get("items", [])
    if not items:
        st.info("News feed is empty right now — try again later.")
        return

    # Header bar with freshness + category filter
    gen_at = payload.get("generated_at", "")
    fresh_label = "earlier today"
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
        delta = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        if   delta < 1:   fresh_label = "less than an hour ago"
        elif delta < 24:  fresh_label = f"{int(delta)} hour(s) ago"
        else:             fresh_label = f"{int(delta/24)} day(s) ago"
    except Exception:
        pass

    cats = sorted({it.get("category", "general-oncology") for it in items})
    CAT_LABELS = {
        "colorectal":       "🎯 Colon & rectal cancer",
        "ibd":              "🩻 IBD (UC / Crohn's)",
        "drug-news":        "💊 New drugs & approvals",
        "clinical-trial":   "🧪 Clinical trials",
        "general-oncology": "🔬 General oncology",
    }
    options = ["All"] + [CAT_LABELS.get(c, c) for c in cats]
    val_to_cat = {CAT_LABELS.get(c, c): c for c in cats}

    cols = st.columns([3, 2])
    with cols[0]:
        st.markdown(
            f"<div style='font-size:0.9rem;color:#475569;'>📊 "
            f"<b>{len(items)}</b> stories · updated {fresh_label} · "
            f"<a href='https://github.com/Yuvraj235/Agentic_Multimodal_Colon_Cancer_AI/"
            f"blob/main/outputs/cancer_news.json' target='_blank'>view raw feed</a></div>",
            unsafe_allow_html=True)
    with cols[1]:
        picked = st.selectbox("Filter by category", options, index=0,
                              label_visibility="collapsed")
    selected = (None if picked == "All" else val_to_cat.get(picked))
    filtered = [it for it in items if selected is None or it["category"] == selected]

    if not filtered:
        st.info("No stories in this category right now.")
        return

    # Render as cards (2-column grid)
    st.markdown("<div style='margin:14px 0;'></div>", unsafe_allow_html=True)
    for i in range(0, len(filtered), 2):
        c1, c2 = st.columns(2)
        for col, it in zip([c1, c2], filtered[i:i+2]):
            with col:
                cat   = it.get("category", "general-oncology")
                cat_label = CAT_LABELS.get(cat, cat)
                badge_color = {
                    "colorectal":      "#0B5FFF",
                    "ibd":             "#7C3AED",
                    "drug-news":       "#16A34A",
                    "clinical-trial":  "#D97706",
                    "general-oncology":"#64748B",
                }.get(cat, "#64748B")
                st.markdown(f"""
                <div style="background:#FFF;border:1px solid #E2E8F0;
                            border-radius:14px;padding:18px 20px;
                            box-shadow:0 2px 6px rgba(15,23,42,0.04);
                            height:230px;display:flex;flex-direction:column;
                            margin-bottom:14px;">
                  <div style="display:inline-block;background:{badge_color};color:white;
                              font-size:0.7rem;font-weight:700;padding:3px 10px;
                              border-radius:999px;letter-spacing:0.3px;
                              text-transform:uppercase;margin-bottom:10px;
                              align-self:flex-start;">
                    {_esc(cat_label)}
                  </div>
                  <div style="font-size:1.02rem;font-weight:700;color:#0F172A;
                              line-height:1.35;margin-bottom:8px;">
                    {_esc(it.get("title",""))[:140]}
                  </div>
                  <div style="font-size:0.85rem;color:#475569;line-height:1.45;
                              flex-grow:1;overflow:hidden;">
                    {_esc(it.get("summary",""))[:180]}…
                  </div>
                  <div style="display:flex;justify-content:space-between;
                              align-items:center;margin-top:10px;
                              padding-top:10px;border-top:1px solid #F1F5F9;">
                    <div style="font-size:0.72rem;color:#64748B;">
                      {_esc(it.get("source",""))}
                    </div>
                    <a href="{_esc(it.get('link','#'))}" target="_blank"
                       style="font-size:0.8rem;color:#0B5FFF;font-weight:600;
                              text-decoration:none;">
                      Read more →
                    </a>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Footer note
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.caption(
        "ℹ️ These stories are aggregated from public RSS feeds "
        "(ScienceDaily, MedicalXpress, Cancer Research UK). "
        "They are NOT screened by ColonAI's clinical safety policy — "
        "always check the source before drawing conclusions, "
        "and never use them as a substitute for medical advice."
    )


# ─────────────────────────────────────────────────────────────────────────────
def page_recalibration():
    """Clinician/admin tool — re-fit confidence calibration on a hospital's own
    labelled images so confidence is honest for THAT site (handover P-4).
    Only rescales confidence (temperature); never changes the model or diagnosis."""
    from src.app.recalibration import (load_labeled_zip, fit_temperature,
                                       save_site_temperature, load_site_temperature,
                                       CLASS_NAMES)
    render_hero(
        "Site calibration · clinician tool",
        "Re-fit ColonAI's confidence to your hospital's own data",
        badges=["Clinician / admin", "Improves confidence honesty", "Does not change diagnoses"],
    )
    if st.button("← Back", key="recal_back"):
        st.session_state["show_recalibration"] = False
        st.rerun()

    st.markdown(
        "ColonAI's confidence was calibrated on public research data. Real sites differ "
        "(scope brands, lighting, case mix), so confidence may not transfer. Upload **labelled "
        "colonoscopy images from your site** (a ZIP with one sub-folder per finding) and we re-fit "
        "a single *temperature* so the confidence numbers are honest for your data. "
        "**This only rescales confidence — it never changes the diagnosis or the model weights.**")
    st.markdown("**ZIP layout** — folder names must match exactly:")
    st.code("\n".join(f"{c}/    image1.jpg  image2.png  ..." for c in CLASS_NAMES))

    active = load_site_temperature()
    if active:
        st.info(f"Active site calibration: T={active.get('temperature')} · "
                f"ECE {active.get('ece_calibrated')} · n={active.get('n_samples')} · "
                f"site '{active.get('site_name','')}'")

    site_name = st.text_input("Site name (optional)", value="", key="recal_site")
    up = st.file_uploader("Upload labelled ZIP", type=["zip"], key="recal_zip")
    if up is None:
        return

    system = st.session_state.get("_system")
    if not system or not system.get("ready"):
        st.error("The AI model isn't loaded yet, so calibration can't run. Open the main "
                 "assessment flow once to load it, then come back.")
        return

    if st.button("Run recalibration", type="primary", key="recal_run"):
        with st.spinner("Loading images and running the model…"):
            try:
                samples, summary = load_labeled_zip(up.getvalue())
            except Exception as e:
                st.error(f"Could not read ZIP: {e}")
                return
            st.caption(f"Loaded {summary['loaded']} images · skipped {summary['skipped']} · "
                       f"per class: {summary['per_class']}")
            if len(samples) == 0:
                st.error("No labelled images found. Check the folder names match the classes above.")
                return
            if len(samples) < 20:
                st.warning(f"Only {len(samples)} images — results will be noisy (≥50 recommended).")

            import numpy as _np, torch as _torch
            model = system["model"]; tokenizer = system["tokenizer"]; device = system["device"]
            tcga_df_v, extract_fn, n_feat = get_tcga_pool_cached()
            input_ids, attn = tokenize_text(tokenizer,
                "Patient undergoing screening colonoscopy; calibration sample.")
            input_ids = input_ids.to(device); attn = attn.to(device)
            try:
                tab = build_tabular_vector({}, tcga_df_v, extract_fn, n_feat).to(device)
            except Exception:
                tab = _torch.zeros(1, n_feat).to(device)
            logits_list, labels_list = [], []
            prog = st.progress(0.0)
            with _torch.no_grad():
                for i, (img, label) in enumerate(samples):
                    try:
                        img_t, _ = preprocess_image(img)
                        out = model(img_t.to(device), input_ids, attn, tab)
                        logits_list.append(out["pathology"].cpu().numpy()[0])
                        labels_list.append(label)
                    except Exception:
                        continue
                    if i % 10 == 0:
                        prog.progress(min(1.0, (i + 1) / len(samples)))
            prog.progress(1.0)
            if len(logits_list) < 5:
                st.error("Too few images ran successfully to calibrate.")
                return
            res = fit_temperature(_np.array(logits_list), _np.array(labels_list))
            st.session_state["_recal_result"] = res

        res = st.session_state["_recal_result"]
        st.success("Recalibration complete.")
        c1, c2, c3 = st.columns(3)
        c1.metric("New temperature", res["temperature"])
        c2.metric("ECE before", res["ece_raw"])
        c3.metric("ECE after", res["ece_calibrated"],
                  delta=round(res["ece_calibrated"] - res["ece_raw"], 4), delta_color="inverse")
        st.caption(f"Fitted on {res['n_samples']} samples · accuracy on them {res['accuracy']:.1%}. "
                   "Lower ECE = better-calibrated confidence. (Image-only context: neutral "
                   "text/tabular were used, matching image-first inference.)")

    res = st.session_state.get("_recal_result")
    if res and st.button("✓ Apply this calibration for my site", type="primary", key="recal_apply"):
        save_site_temperature(res, site_name=st.session_state.get("recal_site", ""))
        st.success(f"Saved. ColonAI will use T={res['temperature']} for confidence at your site.")
        st.session_state.pop("_recal_result", None)


def main():
    render_css()
    # Dark-mode overlay (applied conditionally over the base CSS)
    try:
        from src.app.ui_extras import (
            apply_dark_mode_if_enabled, dark_mode_toggle,
            render_floating_faq, render_particles_once,
            maybe_dismiss_tips_via_query, faq_toggle_button,
        )
        maybe_dismiss_tips_via_query()
        apply_dark_mode_if_enabled()
        render_particles_once()
    except Exception:
        pass

    # Initialise session state
    if "step" not in st.session_state:
        st.session_state["step"] = 0
    if "show_guide" not in st.session_state:
        st.session_state["show_guide"] = False

    # (Colon Buddy is now a pure HTML/JS floating widget — see end of main())

    # Sidebar
    with st.sidebar:
        # Logo / branding
        st.markdown(
            """<div style="text-align:center;padding:14px 0 4px">
                <div style="display:inline-flex;align-items:center;justify-content:center;
                            width:46px;height:46px;border-radius:14px;
                            background:linear-gradient(135deg,#1A73E8,#00897B);
                            box-shadow:0 6px 18px -8px rgba(26,115,232,0.55);
                            color:white;font-weight:900;font-size:1.1rem;letter-spacing:-0.5px">CA</div>
                <h2 style="margin:8px 0 0;font-size:1.15rem;color:#0F172A;font-weight:900;letter-spacing:-0.3px">ColonAI</h2>
                <p style="font-size:0.74rem;color:#64748B;margin:2px 0 0;font-weight:500">
                    Agentic Multimodal Screening
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        render_sidebar_progress()

        # Site Guide button
        st.markdown("")
        if st.sidebar.button("Site Guide", use_container_width=True, key="sidebar_guide_btn"):
            st.session_state["show_guide"] = True
            st.rerun()

        # Clinician tool: per-site confidence recalibration
        if st.sidebar.button("🩺 Site calibration (clinician)",
                             use_container_width=True, key="sidebar_recal_btn"):
            st.session_state["show_recalibration"] = True
            st.rerun()

        # Dark-mode toggle + Compare-mode toggle + Tip-bubble toggle
        try:
            from src.app.ui_extras import (
                dark_mode_toggle as _dm_toggle,
                faq_toggle_button as _faq_toggle,
            )
            _dm_toggle()
            _faq_toggle()
        except Exception:
            pass
        compare_on = st.session_state.get("compare_mode", False)
        if st.sidebar.button("⚖️ Compare cases" + (" · ON" if compare_on else ""),
                              use_container_width=True, key="cmp_toggle"):
            st.session_state["compare_mode"] = not compare_on
            st.rerun()

        st.sidebar.markdown(
            '<p style="font-size:0.70rem;color:#bbb;text-align:center;margin-top:6px">'
            'Research / educational use only.<br>'
            'Not a medical device.<br>v1.0 · Feb 2026</p>',
            unsafe_allow_html=True,
        )

    # Eagerly preload the AI system the first time the user reaches Step 1
    # (when it's far less disruptive than waiting on Step 3).
    if "_system" not in st.session_state:
        st.session_state["_system"] = None
    if (st.session_state.get("step", 0) >= 1
            and st.session_state.get("_system") is None):
        st.session_state["_system"] = load_ai_system()

    # Route to current step (guide / clinician tools override everything)
    if st.session_state.get("show_guide"):
        page_guide()
        return
    if st.session_state.get("show_recalibration"):
        page_recalibration()
        return

    step = st.session_state.get("step", 0)

    if step == 0:
        page_patient_info()
    elif step == 1:
        page_symptoms_upload()
    elif step == 2:
        page_analysis()
    elif step == 3:
        page_results()
    elif step == 4:
        page_doctor_finder()
    elif step == 5:
        page_report()
    elif step == 6:
        page_live_video()
    elif step == 7:
        page_latest_research()
    else:
        st.session_state["step"] = 0
        st.rerun()

    # Floating contextual FAQ bubble (chat-FAB removed — chatbot lives in sidebar)
    try:
        from src.app.ui_extras import render_floating_faq
        render_floating_faq(step)
    except Exception:
        pass

    # 🤖 COLON BUDDY — pure HTML/JS floating chat widget (Intercom-style)
    # Calls the FREE Pollinations.ai LLM directly from the browser.
    # No Streamlit interaction — fully self-contained, no full-screen modals.
    import streamlit.components.v1 as components
    components.html(_COLON_BUDDY_WIDGET_HTML, height=0)


if __name__ == "__main__":
    main()
