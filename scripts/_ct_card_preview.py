"""Throwaway preview: render the CT segmentation results card on a real CARE
slice, so the card UI can be eyeballed without fighting Streamlit file-upload.

Run:  streamlit run scripts/_ct_card_preview.py --server.port 8502
(Not part of the app — a local visual check only.)
"""
import sys
from pathlib import Path
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # main() is guarded
from src.app.care_ct_seg import segment_ct

app.render_css()

NPZ = Path.home() / "Downloads" / "case17105001_slice034.npz"
st.title("CT card preview — real CARE held-out slice (slice034)")
if not NPZ.exists():
    st.error(f"Put a CARE .npz at {NPZ} (e.g. case17105001_slice034.npz).")
    st.stop()

z = np.load(str(NPZ))
img = z["image"].astype("float32")
u = ((img - img.min()) / (np.ptp(img) + 1e-6) * 255).astype("uint8")
u3 = np.stack([u, u, u], -1)

out = segment_ct(u3)
analysis = app._build_ct_result(
    out, u3,
    {"reasons": ["near-grayscale image — looks like a radiology scan (CT)"],
     "score": 0.2, "modality": "radiology_grayscale"},
)
app._render_ct_segmentation_report(analysis, {"name": "Demo — CARE slice034"})
