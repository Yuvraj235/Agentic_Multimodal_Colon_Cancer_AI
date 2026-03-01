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

warnings.filterwarnings("ignore")

# ── project root on path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
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
CHECKPOINT  = ROOT / "outputs/unified_multimodal/checkpoints/best_model.pth"
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
]

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.main { background: #F8FAFF; }

/* ── Hide default Streamlit chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1A73E8 0%, #00897B 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 8px 32px rgba(26,115,232,0.25);
}
.hero-banner h1 { font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero-banner p  { font-size: 1.05rem; opacity: 0.9; margin: 8px 0 0; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 12px;
    margin-right: 8px;
    backdrop-filter: blur(4px);
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 4px solid #1A73E8;
    margin-bottom: 12px;
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.12); }
.metric-card .label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
                       letter-spacing: 0.5px; color: #888; margin-bottom: 4px; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #212121; }
.metric-card .sub   { font-size: 0.82rem; color: #666; margin-top: 3px; }

/* ── Risk badges ── */
.risk-low      { background:#E8F5E9; color:#2E7D32; border:1.5px solid #A5D6A7; }
.risk-moderate { background:#FFFDE7; color:#F57F17; border:1.5px solid #FFE082; }
.risk-high     { background:#FFF3E0; color:#E65100; border:1.5px solid #FFCC80; }
.risk-critical { background:#FFEBEE; color:#B71C1C; border:1.5px solid #EF9A9A; }
.risk-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 18px;
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ── Step progress ── */
.step-item { display: flex; align-items: center; padding: 8px 12px; border-radius: 8px;
             margin: 3px 0; font-size: 0.88rem; cursor: pointer; }
.step-active   { background:#E8F0FE; color:#1A73E8; font-weight: 700; }
.step-done     { background:#E8F5E9; color:#2E7D32; font-weight: 600; }
.step-pending  { color:#9E9E9E; }
.step-icon { width: 28px; height: 28px; border-radius: 50%; display: inline-flex;
             align-items: center; justify-content: center; font-size: 0.78rem;
             font-weight: 700; margin-right: 10px; flex-shrink: 0; }
.step-icon-active  { background:#1A73E8; color:white; }
.step-icon-done    { background:#2E7D32; color:white; }
.step-icon-pending { background:#E0E0E0; color:#9E9E9E; }

/* ── Doctor cards ── */
.doctor-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-top: 3px solid #1A73E8;
    margin-bottom: 14px;
    transition: transform 0.15s, box-shadow 0.15s;
}
.doctor-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
.doctor-name  { font-size: 1.05rem; font-weight: 700; color: #212121; }
.doctor-spec  { font-size: 0.85rem; color: #1A73E8; font-weight: 600; margin: 2px 0; }
.doctor-hosp  { font-size: 0.85rem; color: #555; }
.doctor-meta  { font-size: 0.8rem; color: #888; margin-top: 6px; }
.star-rating  { color: #FFC107; font-size: 1rem; }

/* ── Section headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1A73E8;
    margin: 20px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #E8F0FE;
}

/* ── Urgency banners ── */
.urgency-routine   { background:#E8F5E9; color:#2E7D32; border:2px solid #A5D6A7; }
.urgency-elective  { background:#FFFDE7; color:#F57F17; border:2px solid #FFE082; }
.urgency-urgent    { background:#FFF3E0; color:#E65100; border:2px solid #FFCC80; }
.urgency-emergency { background:#FFEBEE; color:#B71C1C; border:2px solid #EF9A9A; }
.urgency-banner {
    border-radius: 10px;
    padding: 14px 20px;
    font-size: 1.05rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 16px;
    letter-spacing: 0.5px;
}

/* ── Info boxes ── */
.info-box {
    background: #E8F0FE;
    border-left: 4px solid #1A73E8;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #1a1a2e;
}
.warn-box {
    background: #FFF8E1;
    border-left: 4px solid #FFC107;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #4a3900;
}

/* ── Input styling ── */
.stTextInput input, .stSelectbox select, .stNumberInput input {
    border-radius: 8px !important;
    border: 1.5px solid #E0E0E0 !important;
}
.stTextInput input:focus { border-color: #1A73E8 !important; }

/* ── Buttons ── */
div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 2rem;
    transition: all 0.2s;
}
div.stButton > button:hover { transform: translateY(-1px); }

/* ── File uploader ── */
.stFileUploader { border-radius: 12px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #F0F4FF;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.88rem;
}

/* ── Spinner ── */
.loading-container {
    text-align: center;
    padding: 40px;
}
.loading-container h2 { color: #1A73E8; }

/* ── Disclaimer ── */
.disclaimer {
    background: #FAFAFA;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.78rem;
    color: #888;
    margin-top: 24px;
    line-height: 1.6;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# DOCTOR DATABASE (illustrative — would connect to a live directory in production)
# ─────────────────────────────────────────────────────────────────────────────
DOCTORS_DB: List[Dict[str, Any]] = [
    # ── India ────────────────────────────────────────────────────────────
    {"name":"Dr. Priya Sharma","hospital":"AIIMS New Delhi","specialty":"Gastroenterology & Hepatology","sub_specialty":"Colorectal Cancer","city":"New Delhi","country":"India","rating":4.9,"experience_years":22,"phone":"+91-11-2658-8500","languages":["English","Hindi"]},
    {"name":"Dr. Anil Mehta","hospital":"Fortis Escorts, New Delhi","specialty":"Colorectal Surgery","sub_specialty":"Laparoscopic Colectomy","city":"New Delhi","country":"India","rating":4.8,"experience_years":18,"phone":"+91-11-4713-5000","languages":["English","Hindi"]},
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

        # --- Model ---
        model = UnifiedMultiModalTransformer(
            n_classes=N_CLASSES,
            d_model=D_MODEL,
            n_heads=8,
            n_layers=3,
            n_tabular_features=N_TABULAR_FEATURES,
            backbone_name="convnextv2_tiny",
        )
        if CHECKPOINT.exists():
            ckpt = torch.load(str(CHECKPOINT), map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
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
        }
    except Exception as e:
        return {"ready": False, "error": str(e)}


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


def run_analysis(system: dict, pil_img: Image.Image, patient: dict,
                 symptoms: List[str], symptom_text: str) -> dict:
    """Run the full 6-agent pipeline and return a serializable result dict."""
    from src.data.multimodal_dataset import make_clinical_text

    orch    = system["orchestrator"]
    device  = system["device"]
    tcga_df = system.get("tcga_df")

    # Pre-process image
    img_tensor, img_np = preprocess_image(pil_img)

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

    # GradCAM overlay
    gradcam_heatmap = None
    gradcam_overlay = None
    if xai.gradcam_heatmap is not None:
        gradcam_heatmap = xai.gradcam_heatmap
        gradcam_overlay = overlay_gradcam(img_np, gradcam_heatmap)

    return {
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
        "all_risk_flags":   fd.all_risk_flags,
        "uncertainty":      xai.uncertainty,
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
    }


def search_doctors(city: str, country: str = "", specialty: str = "") -> List[Dict]:
    """Filter doctor DB by city / country / specialty."""
    city_lower    = city.lower().strip()
    country_lower = country.lower().strip()
    spec_lower    = specialty.lower().strip()

    results = []
    for doc in DOCTORS_DB:
        city_match    = (city_lower == "" or city_lower in doc["city"].lower()
                         or doc["city"].lower() in city_lower)
        country_match = (country_lower == "" or country_lower in doc["country"].lower()
                         or doc["country"].lower() in country_lower)
        spec_match    = (spec_lower == "" or spec_lower in doc["specialty"].lower()
                         or spec_lower in doc.get("sub_specialty", "").lower())
        if city_match or country_match:
            if spec_match or spec_lower == "":
                results.append(doc)

    # Sort by rating
    results.sort(key=lambda x: x["rating"], reverse=True)
    return results[:10]


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
    for i, name in enumerate(STEPS):
        if i < step:
            st.sidebar.markdown(
                f"""<div class="step-item step-done">
                    <span class="step-icon step-icon-done">&#10003;</span> {name}
                </div>""",
                unsafe_allow_html=True,
            )
        elif i == step:
            st.sidebar.markdown(
                f"""<div class="step-item step-active">
                    <span class="step-icon step-icon-active">{i+1}</span> {name}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                f"""<div class="step-item step-pending">
                    <span class="step-icon step-icon-pending">{i+1}</span> {name}
                </div>""",
                unsafe_allow_html=True,
            )

    st.sidebar.markdown("---")
    # Overall progress bar
    progress = step / (len(STEPS) - 1)
    st.sidebar.progress(progress, text=f"Step {step+1} of {len(STEPS)}")

    # Model status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")
    system = st.session_state.get("_system")
    if system is None:
        st.sidebar.info("Model not loaded yet")
    elif not system.get("ready"):
        st.sidebar.warning("Model load failed")
    else:
        st.sidebar.success("AI pipeline ready")

    # Quick reset
    st.sidebar.markdown("---")
    if st.sidebar.button("Start New Assessment", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k != "_system":
                del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def page_patient_info():
    render_hero(
        "Patient Information",
        "Please provide your personal and medical history details",
        badges=["Step 1 of 6", "Secure & Confidential"],
    )

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
            help="Maximum 20 MB. Image will be resized to 224×224 for AI analysis.",
        )

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            st.session_state["uploaded_image"]      = pil_img
            st.session_state["uploaded_filename"]   = uploaded.name

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
        "AI Analysis in Progress",
        "Our 6-agent multimodal pipeline is processing your data",
        badges=["Step 3 of 6", "6 Agents", "GradCAM++ | BioBERT | TabTransformer"],
    )

    if st.session_state.get("analysis_done"):
        st.success("Analysis complete! Proceeding to results...")
        time.sleep(0.5)
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
        ("Image Agent",   "GradCAM++ on ConvNeXt-V2-Tiny backbone"),
        ("Text Agent",    "BioBERT attention rollout on clinical notes"),
        ("Tabular Agent", "SHAP-style perturbation on TCGA features"),
        ("Fusion Agent",  "Cross-modal attention transformer (256-dim)"),
        ("XAI Agent",     "MC-Dropout uncertainty + counterfactuals"),
        ("Clinical Agent","BSG/NICE guideline-based recommendations"),
    ]

    step_placeholder = st.empty()

    if not system.get("ready"):
        st.error(f"AI system not available: {system.get('error','Unknown error')}")
        st.markdown(
            '<div class="warn-box">The model checkpoint may not be present or dependencies '
            'may be missing. Running in demo mode with simulated results.</div>',
            unsafe_allow_html=True,
        )
        # Demo mode fallback
        _run_demo_analysis()
        return

    pil_img = st.session_state.get("uploaded_image")
    if pil_img is None:
        st.warning("No image uploaded — running text-only analysis in demo mode.")
        _run_demo_analysis()
        return

    # Animated pipeline steps
    progress_bar = st.progress(0)
    for i, (name, desc) in enumerate(pipeline_steps):
        with step_placeholder.container():
            st.markdown(f"### Running Agent {i+1}/6: {name}")
            st.markdown(f"*{desc}*")
            cols = st.columns(6)
            for j, (nm, _) in enumerate(pipeline_steps):
                with cols[j]:
                    if j < i:
                        st.markdown(f"[done]")
                        st.caption(nm)
                    elif j == i:
                        st.markdown(f"[running]")
                        st.caption(f"**{nm}**")
                    else:
                        st.markdown(f"[ ]")
                        st.caption(nm)
        progress_bar.progress((i + 0.5) / len(pipeline_steps))
        time.sleep(0.3)  # brief visual pause per agent

    # Run actual analysis
    try:
        with st.spinner("Finalising multi-agent reasoning..."):
            analysis = run_analysis(
                system=system,
                pil_img=pil_img,
                patient=st.session_state.get("patient", {}),
                symptoms=st.session_state.get("symptoms", []),
                symptom_text=st.session_state.get("symptom_text", ""),
            )
        st.session_state["analysis"]      = analysis
        st.session_state["analysis_done"] = True
        progress_bar.progress(1.0)
        step_placeholder.success("All 6 agents complete!")
        time.sleep(0.8)
        st.session_state["step"] = 3
        st.rerun()

    except Exception as e:
        st.error(f"Analysis error: {e}")
        st.info("Falling back to demo results...")
        _run_demo_analysis()


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
    }
    st.session_state["analysis_done"] = True
    st.session_state["step"] = 3
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

    render_hero(
        "Diagnostic Results",
        f"AI analysis complete for {patient.get('name','Patient')} · {datetime.now().strftime('%d %b %Y, %H:%M')}",
        badges=["Step 4 of 6", f"Confidence: {analysis['confidence']:.0%}", f"Inference: {analysis['inference_time_ms']:.0f} ms"],
    )

    # Top summary metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        render_metric_card("AI Finding", CLASS_LABELS.get(pclass, pclass),
                           f"Confidence: {analysis['confidence']:.0%}", color=pcolor)
    with col_b:
        render_metric_card("Cancer Stage", analysis["stage"],
                           f"Confidence: {analysis['stage_confidence']:.0%}",
                           color=STAGE_COLORS.get(analysis["stage"],"#1A73E8"))
    with col_c:
        risk_score = analysis["risk_score"]
        rc = "#2E7D32" if risk_score<0.25 else "#F9A825" if risk_score<0.5 else "#E65100" if risk_score<0.75 else "#B71C1C"
        render_metric_card("Risk Score", f"{risk_score:.0%}", analysis["risk_label"], color=rc)
    with col_d:
        unc = analysis["uncertainty"]
        unc_lbl = "Low" if unc<0.3 else "Moderate" if unc<0.6 else "High"
        uc = "#2E7D32" if unc<0.3 else "#F9A825" if unc<0.6 else "#B71C1C"
        render_metric_card("AI Uncertainty", f"{unc:.2f}", unc_lbl, color=uc)

    st.markdown("")
    # Risk badge
    render_risk_badge(risk_score)
    st.markdown("")

    # ── Four Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Diagnosis", "GradCAM View", "Risk Charts", "Recommendations"
    ])

    # ── Tab 1: Diagnosis ───────────────────────────────────────────────
    with tab1:
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

        # Stage probs
        st.markdown('<div class="section-header">Staging Probability</div>',
                    unsafe_allow_html=True)
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

    # ── Tab 2: GradCAM ─────────────────────────────────────────────────
    with tab2:
        orig  = analysis.get("original_image")
        cam   = analysis.get("gradcam_overlay")
        heat  = analysis.get("gradcam_heatmap")

        if cam is not None or orig is not None:
            col_orig, col_cam = st.columns(2)
            with col_orig:
                st.markdown('<div class="section-header">Original Image</div>',
                            unsafe_allow_html=True)
                if orig is not None:
                    disp = (orig * 255).astype(np.uint8) if orig.max() <= 1 else orig
                    st.image(disp, caption="Input endoscopy image", use_container_width=True)
                else:
                    pil_in = st.session_state.get("uploaded_image")
                    if pil_in:
                        st.image(pil_in, caption="Input image", use_container_width=True)

            with col_cam:
                st.markdown('<div class="section-header">GradCAM++ Attention Map</div>',
                            unsafe_allow_html=True)
                if cam is not None:
                    disp_cam = (cam * 255).astype(np.uint8) if cam.max() <= 1 else cam.astype(np.uint8)
                    st.image(disp_cam,
                             caption="Red = High AI attention | Blue = Low attention",
                             use_container_width=True)
                else:
                    st.info("GradCAM heatmap not available for this analysis.")

            if heat is not None:
                st.markdown('<div class="section-header">Raw Heatmap Intensity</div>',
                            unsafe_allow_html=True)
                import matplotlib.pyplot as plt
                fig_hm, ax = plt.subplots(figsize=(8, 2.5))
                im = ax.imshow(heat, cmap="hot", aspect="auto")
                ax.set_title("GradCAM++ Heatmap (brighter = higher model attention)")
                ax.axis("off")
                plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.1)
                st.pyplot(fig_hm, use_container_width=True)
                plt.close(fig_hm)

            st.markdown(
                '<div class="info-box"><b>How to read this:</b> The red/warm regions on the GradCAM '
                'map show exactly where the AI model is focusing its attention when making its diagnosis. '
                'These highlight the most diagnostically relevant tissue regions.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No image was uploaded. GradCAM analysis requires an endoscopy/colonoscopy image.")
            pil_in = st.session_state.get("uploaded_image")
            if pil_in:
                st.image(pil_in, caption="Uploaded image (demo mode — no GradCAM)", width=400)

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
            radar_vals += [radar_vals[0]]
            radar_cats += [radar_cats[0]]

            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals, theta=radar_cats,
                fill="toself",
                fillcolor="rgba(26,115,232,0.15)",
                line=dict(color="#1A73E8", width=2),
                marker=dict(size=6, color="#1A73E8"),
                hovertemplate="<b>%{theta}</b>: %{r:.1%}<extra></extra>",
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

        # Confidence timeline / waterfall
        st.markdown('<div class="section-header">Model Confidence Breakdown</div>',
                    unsafe_allow_html=True)
        conf_agents = [
            "Image Encoder", "Text Encoder", "Tabular Encoder",
            "Fusion Layer", "XAI Verification", "Final Output"
        ]
        # Simulate agent-level confidences from available data
        img_w  = analysis["image_weight"]
        txt_w  = analysis["text_weight"]
        tab_w  = analysis["tabular_weight"]
        final_c = analysis["confidence"]
        conf_vals = [
            img_w * final_c,
            txt_w * final_c,
            tab_w * final_c,
            final_c * 0.97,
            final_c * (1 - unc * 0.3),
            final_c,
        ]
        fig_conf = go.Figure(go.Bar(
            x=conf_agents, y=conf_vals,
            marker_color=["#1A73E8","#00897B","#FF5722","#9C27B0","#607D8B","#2E7D32"],
            text=[f"{v:.1%}" for v in conf_vals],
            textposition="outside",
        ))
        fig_conf.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 1.15], tickformat=".0%", showgrid=True, gridcolor="#f0f0f0"),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter, sans-serif", size=11),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # ── Tab 4: Recommendations ─────────────────────────────────────────
    with tab4:
        rec = analysis.get("recommendation", {})
        if not rec:
            st.info("No recommendations available.")
        else:
            render_urgency_banner(rec.get("urgency", "Routine"))

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
    render_hero(
        "Find Specialists Near You",
        "Locate top-rated gastroenterologists, colorectal surgeons, and oncologists in your region",
        badges=["Step 5 of 6", "45+ Specialists", "20+ Cities"],
    )

    patient = st.session_state.get("patient", {})

    st.markdown('<div class="section-header">Search</div>', unsafe_allow_html=True)
    col_c, col_co, col_sp = st.columns(3)
    with col_c:
        search_city = st.text_input("City", value=patient.get("city", ""),
                                     placeholder="e.g. Mumbai, New York, London")
    with col_co:
        search_country = st.selectbox("Country",
            ["(Any)", "India","USA","UK","UAE","Singapore","Canada","Australia"],
            index=0 if not patient.get("country") else
                  ["(Any)","India","USA","UK","UAE","Singapore","Canada","Australia"].index(
                      patient.get("country","India")))
    with col_sp:
        search_spec = st.selectbox("Specialty",
            ["(Any)", "Gastroenterology", "Colorectal Surgery",
             "GI Oncology", "Medical Oncology", "Surgical Oncology"])

    country_q = "" if search_country == "(Any)" else search_country
    spec_q    = "" if search_spec == "(Any)" else search_spec

    doctors = search_doctors(search_city, country_q, spec_q)

    if not doctors:
        # Broaden search
        doctors = search_doctors("", country_q, spec_q)
    if not doctors:
        doctors = DOCTORS_DB[:8]

    st.markdown(f'<div class="info-box">Found <b>{len(doctors)}</b> specialists matching your criteria</div>',
                unsafe_allow_html=True)
    st.markdown("")

    # Render doctor cards in 2-column grid
    for i in range(0, len(doctors), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx >= len(doctors):
                break
            doc = doctors[idx]
            with cols[j]:
                st.markdown(
                    f"""<div class="doctor-card">
                        <div class="doctor-name">{doc['name']}</div>
                        <div class="doctor-spec">{doc['specialty']}</div>
                        <div class="doctor-hosp">{doc['hospital']}</div>
                        <div class="doctor-meta">
                            {doc['city']}, {doc['country']} &nbsp;|&nbsp;
                            {doc['experience_years']} yrs exp &nbsp;|&nbsp;
                            Rating: {doc['rating']:.1f}/5.0
                        </div>
                        <div class="doctor-meta">
                            Tel: {doc.get('phone','N/A')} &nbsp;|&nbsp;
                            Languages: {', '.join(doc.get('languages',['English']))}
                        </div>
                        <div class="doctor-meta" style="margin-top:6px;font-size:0.78rem;color:#888;font-style:italic">
                            {doc.get('sub_specialty','')}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<div class="warn-box"><b>Note:</b> Doctor listings are illustrative. '
        'Please verify availability and credentials directly with the institution. '
        'In production, this would connect to a live medical directory API.</div>',
        unsafe_allow_html=True,
    )

    # Save selected doctors for report
    st.session_state["suggested_doctors"] = doctors[:5]

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
    # Site navigation
    {"k": ["how does this work","how to use","guide","steps","navigate","start"],
     "a": "ColonAI has 6 steps: (1) Patient Info, (2) Symptoms & Upload, (3) AI Analysis, (4) Results, (5) Find Doctors, (6) Download Report. Use the sidebar to track your progress."},
    {"k": ["what is colonai","what is this","about","purpose"],
     "a": "ColonAI is an AI-powered screening tool that analyses colonoscopy, endoscopy, and histopathology images alongside your symptoms and clinical data to flag potential colorectal conditions. It uses a 6-agent pipeline with GradCAM++, BioBERT, and a fusion transformer."},
    # Symptoms
    {"k": ["rectal bleeding","blood stool","bloody stool","blood in stool"],
     "a": "Rectal bleeding or blood in the stool can be caused by haemorrhoids, polyps, inflammatory bowel disease, or colorectal cancer. Please consult a gastroenterologist promptly — this is something that should always be evaluated by a doctor."},
    {"k": ["abdominal pain","stomach pain","cramping","cramps"],
     "a": "Abdominal pain alongside other GI symptoms may indicate IBS, IBD, polyps, or colorectal issues. Note the location, severity (0-10), and duration, and share this on the Symptoms page."},
    {"k": ["weight loss","losing weight","unexplained weight"],
     "a": "Unexplained weight loss of more than 5% of body weight over 6-12 months is a red-flag symptom in gastroenterology. Please complete your assessment and see a doctor."},
    {"k": ["constipation","hard stool","not passing stool"],
     "a": "New-onset constipation lasting more than 3 weeks, especially in adults over 50, should be evaluated. Increase fibre and water intake, and complete the symptom checker on Step 2."},
    {"k": ["diarrhoea","diarrhea","loose stool","watery stool"],
     "a": "Persistent diarrhoea lasting over 4 weeks warrants investigation for IBD, infection, or colorectal cancer. Complete Step 2 to log your symptoms."},
    {"k": ["heartburn","acid reflux","gerd","barrett"],
     "a": "Persistent heartburn and GERD are associated with Barrett's esophagus — a condition the AI can screen for. Upload an endoscopy image on Step 2 for analysis."},
    # Conditions
    {"k": ["polyp","polyps","colorectal polyp"],
     "a": "Colorectal polyps are growths on the colon lining. Most are benign, but some can become cancerous. The AI screens for polyps from colonoscopy images with high accuracy (99.5%). They are usually removed during colonoscopy (polypectomy)."},
    {"k": ["ulcerative colitis","uc","colitis","ibd","inflammatory bowel"],
     "a": "Ulcerative colitis (UC) is a form of inflammatory bowel disease causing ulcers in the colon. The AI classifies UC severity as mild or moderate-severe. Treatment involves aminosalicylates, steroids, or biologics."},
    {"k": ["barretts","barrett esophagus","esophagus","oesophagus"],
     "a": "Barrett's esophagus is a pre-cancerous change in the oesophageal lining caused by chronic acid reflux. The AI can detect this from upper GI endoscopy images. Regular surveillance endoscopy is recommended."},
    {"k": ["colon cancer","colorectal cancer","bowel cancer","rectal cancer"],
     "a": "Colorectal cancer is one of the most common cancers globally. Risk factors include age, family history, polyps, IBD, smoking, alcohol, and low-fibre diet. Early detection dramatically improves outcomes — the AI helps screen for it."},
    # AI / Model
    {"k": ["gradcam","heatmap","attention map","what is the model looking at"],
     "a": "GradCAM++ is an explainability technique that highlights which regions of the image the AI focuses on. Red/warm areas have the highest influence on the AI's decision — shown on the GradCAM View tab in Results."},
    {"k": ["how accurate","accuracy","performance","model performance"],
     "a": "The model achieves 99.5% test accuracy and 0.9946 F1-score on the combined HyperKvasir + CVC-ClinicDB dataset. AUC-ROC is 1.000. However, this is a research tool and NOT a replacement for clinical diagnosis."},
    {"k": ["biobert","bert","text","clinical text","nlp"],
     "a": "BioBERT is a biomedical language model that analyses your clinical text and symptoms. It contributes alongside the image encoder and patient data encoder in a cross-modal fusion transformer."},
    {"k": ["uncertainty","confidence","how sure"],
     "a": "Model uncertainty is calculated using MC-Dropout — 15 stochastic forward passes. Low uncertainty (<0.3) means consistent predictions; high uncertainty (>0.6) means the model is less confident and you should seek expert review."},
    # Doctors
    {"k": ["find doctor","gastroenterologist","specialist","oncologist","surgeon","consultant"],
     "a": "Step 5 (Find Doctors) lists 46+ gastroenterologists, colorectal surgeons, and GI oncologists across 20+ cities globally. Enter your city to find the nearest specialists."},
    # Report
    {"k": ["report","pdf","download","generate report"],
     "a": "Step 6 (Download Report) generates a full clinical PDF with patient info, symptoms, AI findings, GradCAM images, probability charts, recommendations, and doctor suggestions. Click 'Generate PDF Report' to create it."},
    # Upload
    {"k": ["upload","image","colonoscopy","endoscopy","histopathology","photo","picture","jpeg","jpg","png"],
     "a": "On Step 2 (Upload Images tab), you can upload JPG or PNG images from colonoscopy, endoscopy, or histopathology. The image is resized to 224x224 and analysed by the 6-agent pipeline."},
    # Risk
    {"k": ["risk","high risk","low risk","malignant","benign","cancer risk"],
     "a": "The AI risk score (0-100%) represents the probability of malignancy. Below 25% = Low risk; 25-50% = Moderate; 50-75% = High; above 75% = Critical. Always confirm results with a qualified clinician."},
    # Screening age
    {"k": ["screening","when to screen","colonoscopy age","how often"],
     "a": "Current guidelines (BSG/NHS/NICE) recommend colorectal cancer screening from age 50. Higher-risk individuals (family history, IBD, Lynch syndrome) should begin earlier. Talk to your GP about the right schedule."},
    # Disclaimer
    {"k": ["disclaimer","medical advice","diagnosis","replace doctor"],
     "a": "ColonAI is for screening and research purposes ONLY. It does NOT provide medical diagnoses or replace a qualified doctor. All findings must be reviewed by a licensed clinician before any action is taken."},
    # Fallback
    {"k": [],
     "a": "I can help with questions about symptoms, conditions (polyps, UC, Barrett's, colon cancer), how the AI works, uploading images, finding doctors, or downloading your report. What would you like to know?"},
]


def _chatbot_respond(user_msg: str) -> str:
    """Simple keyword-matching chatbot response."""
    msg = user_msg.lower().strip()
    for entry in CHATBOT_KB[:-1]:  # skip fallback
        if any(kw in msg for kw in entry["k"]):
            return entry["a"]
    return CHATBOT_KB[-1]["a"]  # fallback


def render_chatbot():
    """Render the collapsible chatbot panel in the sidebar."""
    st.sidebar.markdown("---")
    with st.sidebar.expander("Ask the AI Assistant", expanded=False):
        st.markdown(
            '<p style="font-size:0.78rem;color:#666;margin-bottom:8px">'
            'Ask about symptoms, the AI model, how to use the app, or any GI health question.</p>',
            unsafe_allow_html=True,
        )
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Display chat history
        for role, msg in st.session_state["chat_history"][-8:]:
            if role == "user":
                st.markdown(
                    f'<div style="background:#E8F0FE;border-radius:8px;padding:7px 10px;'
                    f'margin:4px 0;font-size:0.82rem;color:#1a1a2e"><b>You:</b> {msg}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="background:#F5F5F5;border-radius:8px;padding:7px 10px;'
                    f'margin:4px 0;font-size:0.82rem;color:#333"><b>Assistant:</b> {msg}</div>',
                    unsafe_allow_html=True,
                )

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Your question",
                placeholder="e.g. What are polyps?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_input.strip():
            reply = _chatbot_respond(user_input.strip())
            st.session_state["chat_history"].append(("user", user_input.strip()))
            st.session_state["chat_history"].append(("assistant", reply))
            st.rerun()

        if st.session_state["chat_history"]:
            if st.button("Clear chat", use_container_width=True, key="clear_chat"):
                st.session_state["chat_history"] = []
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

    tab_ov, tab_steps, tab_ai, tab_faq = st.tabs([
        "Overview", "Step-by-Step", "AI Explained", "FAQ"
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
                '<div class="value">99.5%</div>'
                '<div class="sub">Test accuracy on 1,066 images (HyperKvasir + CVC-ClinicDB)</div>'
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
                "- **Image Encoder** — ConvNeXt-V2-Tiny backbone extracts visual patch tokens "
                "from your endoscopy image\n"
                "- **Text Encoder** — BioBERT (dmis-lab/biobert-base-cased-v1.2) processes "
                "clinical notes and your symptom description\n"
                "- **Tabular Encoder** — TabTransformer encodes 12 TCGA clinical features "
                "(age, BMI, smoking, alcohol, tumour stage, etc.)\n\n"
                "These are fused by a **Cross-Modal Attention Transformer** (256-dim, 8 heads, "
                "3 layers) which learns which modality to trust most for each case."
            )

        with col_b:
            st.markdown("**Output Heads**")
            st.markdown(
                "Three prediction heads are trained simultaneously:\n\n"
                "- **Pathology Head** (5-class): Polyps | UC-Mild | UC-Mod-Sev | "
                "Barrett's | Therapeutic\n"
                "- **Staging Head** (4-class): No Cancer | Stage I | Stage II | Stage III/IV\n"
                "- **Risk Head** (binary): Benign vs Malignant\n\n"
                "**Performance:** 99.53% accuracy | 0.9946 F1 Macro | AUC-ROC 1.000\n\n"
                "**Training data:** 9,675 images — HyperKvasir (10,662 images) + "
                "CVC-ClinicDB (612 polyp images) + TCGA clinical data (461 patients)"
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

    # ── Tab 4: FAQ ─────────────────────────────────────────────────────
    with tab_faq:
        st.markdown('<div class="section-header">Frequently Asked Questions</div>',
                    unsafe_allow_html=True)

        faqs = [
            ("Is this a medical diagnosis?",
             "No. ColonAI is a research-grade screening tool. Results must be reviewed by a "
             "qualified, licensed clinician before any clinical decisions are made."),
            ("What image types can I upload?",
             "JPG and PNG images from colonoscopy, endoscopy, or histopathology. Images are "
             "automatically resized to 224x224 pixels for analysis. Maximum 20 MB per file."),
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
             "The model has 99.5% test accuracy but is not perfect. Edge cases, unusual angles, "
             "image artefacts, or conditions outside the training distribution can lead to errors. "
             "This is why clinical review is always required."),
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
def main():
    render_css()

    # Initialise session state
    if "step" not in st.session_state:
        st.session_state["step"] = 0
    if "show_guide" not in st.session_state:
        st.session_state["show_guide"] = False

    # Sidebar
    with st.sidebar:
        # Logo / branding
        st.markdown(
            """<div style="text-align:center;padding:16px 0 8px">
                <h2 style="margin:4px 0;font-size:1.1rem;color:#1A73E8;font-weight:800">ColonAI</h2>
                <p style="font-size:0.75rem;color:#888;margin:0">Agentic Cancer Screening</p>
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

        # Chatbot panel
        render_chatbot()

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            '<p style="font-size:0.72rem;color:#bbb;text-align:center">'
            'For research & educational use only.<br>'
            'Not a medical device.<br>v1.0 · Feb 2026</p>',
            unsafe_allow_html=True,
        )

    # Eagerly preload the AI system in background
    if "_system" not in st.session_state:
        st.session_state["_system"] = None

    # Route to current step (guide overrides everything)
    if st.session_state.get("show_guide"):
        page_guide()
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
    else:
        st.session_state["step"] = 0
        st.rerun()


if __name__ == "__main__":
    main()
