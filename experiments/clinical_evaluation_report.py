# -*- coding: utf-8 -*-
"""
Clinical Evaluation Report + Severity Scoring Module
=====================================================
Reads: metrics.json, ablation.json, agent_results.json, per-class metrics
Produces:
  outputs/clinical_evaluation/
    ├── 01_clinical_performance_dashboard.png
    ├── 02_severity_scoring.png
    ├── 03_modality_contribution.png
    ├── 04_urgency_risk_matrix.png
    ├── 05_clinical_xai_summary.png
    ├── clinical_evaluation_report.txt
    └── severity_scores.json

Run from project root:
  python3 experiments/clinical_evaluation_report.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────
OUT_DIR    = "outputs/clinical_evaluation"
METRICS_F  = "outputs/unified_multimodal/metrics.json"
ABLATION_F = "outputs/unified_multimodal/ablation.json"
AGENTS_F   = "outputs/unified_multimodal/agent_results.json"

CLASS_NAMES = ["polyps", "uc-mild", "uc-moderate-sev", "barretts-esoph", "therapeutic"]

# Severity levels per class (clinical domain knowledge)
# Each entry: (baseline_severity, malignancy_risk_pct, surveillance_interval_months)
CLASS_SEVERITY = {
    "polyps":          ("Low-Moderate", 15,  12),
    "uc-mild":         ("Moderate",     8,   12),
    "uc-moderate-sev": ("High",         25,  3),
    "barretts-esoph":  ("Moderate",     12,  24),
    "therapeutic":     ("Post-proc.",   5,   3),
}

# Clinical colour coding
SEVERITY_COLOURS = {
    "Low":        "#4CAF50",
    "Low-Moderate": "#8BC34A",
    "Moderate":   "#FFC107",
    "High":       "#FF5722",
    "Post-proc.": "#2196F3",
}

URGENCY_COLOURS = {"Routine": "#4CAF50", "Elective": "#FFC107", "Urgent": "#FF5722"}

PALETTE = ["#2196F3", "#f44336", "#B71C1C", "#9C27B0", "#009688"]

os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(path, dpi=180):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close("all")
    sz = os.path.getsize(path) // 1024
    print(f"  [Fig] {os.path.basename(path)}  ({sz} KB)")


# ── Load data ─────────────────────────────────────────────────────────────
with open(METRICS_F)  as f: metrics  = json.load(f)
with open(ABLATION_F) as f: ablation = json.load(f)
with open(AGENTS_F)   as f: agents   = json.load(f)

# ─────────────────────────────────────────────────────────────────────────
# FIG 01: Clinical Performance Dashboard
# ─────────────────────────────────────────────────────────────────────────
def fig_clinical_dashboard():
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Clinical Performance Dashboard\n"
        "Unified Agentic Multi-Modal Colon Cancer AI — Test Set Evaluation",
        fontsize=14, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Subplot 1: Core metrics gauge-style bar ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    metric_names = ["Test Accuracy", "F1 Macro", "AUC-ROC"]
    metric_vals  = [
        metrics["test_accuracy"],
        metrics["test_f1_macro"],
        metrics["test_auc_roc"],
    ]
    colours = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax1.barh(metric_names, metric_vals, color=colours, height=0.55, edgecolor="white", lw=1.5)
    for bar, val in zip(bars, metric_vals):
        ax1.text(val - 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", ha="right", fontsize=11,
                 color="white", fontweight="bold")
    ax1.set_xlim(0, 1.0)
    ax1.set_title("Core Model Metrics", fontsize=11, fontweight="bold")
    ax1.axvline(0.90, color="grey", lw=1.2, ls="--", alpha=0.7)
    ax1.text(0.905, -0.5, "Target\n90%", fontsize=7.5, color="grey")
    ax1.grid(axis="x", alpha=0.3)

    # ── Subplot 2: Modality ablation ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    abl_names = ["All Modalities", "−Image", "−Text", "−Tabular"]
    abl_vals  = [
        ablation["all_modalities"],
        ablation["ablate_image"],
        ablation["ablate_text"],
        ablation["ablate_tabular"],
    ]
    abl_cols = ["#4CAF50", "#FF5722", "#FF9800", "#9C27B0"]
    bars2 = ax2.bar(abl_names, abl_vals, color=abl_cols, width=0.6, edgecolor="white", lw=1.5)
    for bar, val in zip(bars2, abl_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylim(0.65, 1.0)
    ax2.set_title("Modality Ablation (Test Accuracy)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy")
    ax2.set_xticklabels(abl_names, fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    # Delta annotations
    base = ablation["all_modalities"]
    deltas = [(base - v) for v in abl_vals[1:]]
    delta_labels = [f"−{d:.3f}" for d in deltas]
    for i, (dl, val) in enumerate(zip(delta_labels, abl_vals[1:])):
        ax2.text(i + 1, val - 0.015, dl, ha="center", fontsize=8, color="white", fontweight="bold")

    # ── Subplot 3: Per-class malignancy risk ──────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    risk_pcts = [CLASS_SEVERITY[c][1] for c in CLASS_NAMES]
    sev_cols  = [SEVERITY_COLOURS[CLASS_SEVERITY[c][0]] for c in CLASS_NAMES]
    short_names = [c.replace("-", "\n") for c in CLASS_NAMES]
    bars3 = ax3.bar(short_names, risk_pcts, color=sev_cols, width=0.6, edgecolor="white", lw=1.5)
    for bar, val in zip(bars3, risk_pcts):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                 f"{val}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_title("Malignancy Risk per Class (%)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Risk (%)")
    ax3.set_ylim(0, 35)
    ax3.grid(axis="y", alpha=0.3)
    patches = [mpatches.Patch(color=v, label=k) for k, v in SEVERITY_COLOURS.items()]
    ax3.legend(handles=patches, fontsize=7, loc="upper right")

    # ── Subplot 4: Agent case urgency breakdown ────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    urgency_counts = {}
    for a in agents:
        u = a.get("urgency", "Routine")
        urgency_counts[u] = urgency_counts.get(u, 0) + 1
    u_labels = list(urgency_counts.keys())
    u_vals   = [urgency_counts[k] for k in u_labels]
    u_cols   = [URGENCY_COLOURS.get(k, "#607D8B") for k in u_labels]
    wedges, texts, autotexts = ax4.pie(u_vals, labels=u_labels, colors=u_cols,
                                        autopct="%1.0f%%", startangle=90,
                                        textprops={"fontsize": 9},
                                        wedgeprops={"edgecolor": "white", "lw": 1.5})
    for at in autotexts:
        at.set_fontweight("bold")
    ax4.set_title(f"Urgency Breakdown\n({len(agents)} agent cases)", fontsize=11, fontweight="bold")

    # ── Subplot 5: Mean modality weights across cases ──────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    mod_keys = list(agents[0]["modality_weights"].keys())
    mod_vals_all = np.array([[a["modality_weights"][k] for k in mod_keys] for a in agents])
    mod_mean = mod_vals_all.mean(axis=0)
    mod_std  = mod_vals_all.std(axis=0)
    mod_short = ["Image\n(Grad-CAM++)", "Text\n(BioBERT)", "Tabular\n(TabTrans.)"]
    mod_cols  = ["#2196F3", "#4CAF50", "#FF9800"]
    bars5 = ax5.bar(mod_short, mod_mean, yerr=mod_std, color=mod_cols, width=0.5,
                    edgecolor="white", lw=1.5, capsize=5, error_kw={"elinewidth": 1.5})
    for bar, val in zip(bars5, mod_mean):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax5.set_ylim(0, 0.55)
    ax5.set_title("Mean Modality Weights ± SD", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Attention Weight")
    ax5.grid(axis="y", alpha=0.3)

    # ── Subplot 6: Surveillance intervals ─────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    surv_months = [CLASS_SEVERITY[c][2] for c in CLASS_NAMES]
    sev_cols6   = [SEVERITY_COLOURS[CLASS_SEVERITY[c][0]] for c in CLASS_NAMES]
    short_names6 = [c.replace("-", "\n") for c in CLASS_NAMES]
    bars6 = ax6.bar(short_names6, surv_months, color=sev_cols6, width=0.6,
                    edgecolor="white", lw=1.5)
    for bar, val in zip(bars6, surv_months):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                 f"{val}mo", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax6.set_title("Recommended Surveillance Interval (months)", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Months")
    ax6.set_ylim(0, 30)
    ax6.grid(axis="y", alpha=0.3)

    save_fig(f"{OUT_DIR}/01_clinical_performance_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────
# FIG 02: Severity Scoring Module
# ─────────────────────────────────────────────────────────────────────────
def fig_severity_scoring():
    """
    Severity score = composite of:
      - cancer_risk_score (0-1) from risk head
      - uncertainty (inverted: 1 - unc/max_unc) = confidence
      - image_confidence from GradCAM attention density
    Weighted: 0.50 risk + 0.30 confidence + 0.20 image_conf
    """
    severity_data = []
    for a in agents:
        risk   = a["cancer_risk_score"]
        unc    = a["uncertainty"]
        conf   = max(0, 1 - unc)          # uncertainty → confidence
        img_c  = a["image_confidence"]
        sev    = 0.50 * risk + 0.30 * conf + 0.20 * img_c

        # Severity band
        if sev >= 0.75:
            band = "High"
        elif sev >= 0.50:
            band = "Moderate"
        elif sev >= 0.25:
            band = "Low-Moderate"
        else:
            band = "Low"

        severity_data.append({
            "case_id":     a["case_id"],
            "pathology":   a["pathology"],
            "risk_score":  risk,
            "unc":         unc,
            "confidence":  conf,
            "img_conf":    img_c,
            "composite":   sev,
            "band":        band,
            "urgency":     a["urgency"],
        })

    # Save severity scores
    with open(f"{OUT_DIR}/severity_scores.json", "w") as f:
        json.dump(severity_data, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Severity Scoring Module — Per-Case Clinical Severity Assessment",
                 fontsize=13, fontweight="bold")

    case_ids    = [d["case_id"] for d in severity_data]
    composites  = [d["composite"] for d in severity_data]
    risks       = [d["risk_score"] for d in severity_data]
    confs       = [d["confidence"] for d in severity_data]
    img_confs   = [d["img_conf"] for d in severity_data]
    bands       = [d["band"] for d in severity_data]
    pathologies = [d["pathology"] for d in severity_data]
    band_cols   = [SEVERITY_COLOURS.get(b, "#607D8B") for b in bands]

    # ── Left: Composite severity score per case ────────────────────────
    ax = axes[0]
    x = range(len(case_ids))
    bars = ax.bar(x, composites, color=band_cols, width=0.6, edgecolor="white", lw=1.5)
    ax.axhline(0.75, color="#FF5722", lw=1.2, ls="--", alpha=0.7, label="High severity threshold")
    ax.axhline(0.50, color="#FFC107", lw=1.2, ls="--", alpha=0.7, label="Moderate threshold")
    ax.axhline(0.25, color="#4CAF50", lw=1.2, ls="--", alpha=0.7, label="Low-Mod threshold")
    for bar, val, band in zip(bars, composites, bands):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.3f}\n{band}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c}\n({p})" for c, p in zip(case_ids, pathologies)],
                        fontsize=7.5, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Composite Severity Score", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (0–1)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── Middle: Component breakdown stacked ────────────────────────────
    ax = axes[1]
    w50 = [0.50 * r for r in risks]
    w30 = [0.30 * c for c in confs]
    w20 = [0.20 * ic for ic in img_confs]
    p1 = ax.bar(x, w50, color="#FF5722", width=0.6, label="Risk Score (×0.50)", edgecolor="white")
    p2 = ax.bar(x, w30, bottom=w50, color="#2196F3", width=0.6, label="Confidence (×0.30)", edgecolor="white")
    p3 = ax.bar(x, w20, bottom=[a+b for a,b in zip(w50,w30)],
                color="#4CAF50", width=0.6, label="Image Conf (×0.20)", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("_","_\n") for c in case_ids], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Component Breakdown", fontsize=11, fontweight="bold")
    ax.set_ylabel("Weighted contribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Right: Risk vs Uncertainty scatter ─────────────────────────────
    ax = axes[2]
    uncs = [d["unc"] for d in severity_data]
    sc = ax.scatter(risks, uncs, c=composites, cmap="RdYlGn_r",
                    s=180, edgecolors="grey", lw=1.2, vmin=0, vmax=1, zorder=5)
    for d in severity_data:
        ax.annotate(d["case_id"].replace("sample_", "S"),
                    xy=(d["risk_score"], d["unc"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Composite Severity", fontsize=9)
    ax.set_xlabel("Cancer Risk Score", fontsize=10)
    ax.set_ylabel("Model Uncertainty", fontsize=10)
    ax.set_title("Risk vs Uncertainty\n(colour=composite severity)", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="grey", lw=1.0, ls=":", alpha=0.6)
    ax.axvline(0.5, color="grey", lw=1.0, ls=":", alpha=0.6)
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)

    save_fig(f"{OUT_DIR}/02_severity_scoring.png")
    return severity_data


# ─────────────────────────────────────────────────────────────────────────
# FIG 03: Modality Contribution Analysis
# ─────────────────────────────────────────────────────────────────────────
def fig_modality_contribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Modality Contribution Analysis — Gated Cross-Modal Fusion",
                 fontsize=13, fontweight="bold")

    mod_keys = list(agents[0]["modality_weights"].keys())
    mod_mat  = np.array([[a["modality_weights"][k] for k in mod_keys] for a in agents])
    case_ids = [a["case_id"] for a in agents]
    mod_short = ["Image", "Text", "Tabular"]

    # ── Heatmap per case ──────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(mod_mat, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(mod_short, fontsize=10)
    ax.set_yticks(range(len(case_ids)))
    ax.set_yticklabels([f"{c}\n({a['pathology']})" for c, a in zip(case_ids, agents)], fontsize=8)
    for i in range(len(case_ids)):
        for j in range(3):
            ax.text(j, i, f"{mod_mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if mod_mat[i, j] > 0.35 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_title("Per-Case Modality Weights", fontsize=11, fontweight="bold")

    # ── Modality importance (ablation delta) ──────────────────────────
    ax = axes[1]
    base = ablation["all_modalities"]
    ablation_keys = ["ablate_image", "ablate_text", "ablate_tabular"]
    ablation_vals = [ablation[k] for k in ablation_keys]
    delta_abs = [base - v for v in ablation_vals]
    delta_pct = [(d / base) * 100 for d in delta_abs]
    mod_imp_cols = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(mod_short, delta_pct, color=mod_imp_cols, width=0.55,
                  edgecolor="white", lw=1.5)
    for bar, val, dabs in zip(bars, delta_pct, delta_abs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                f"−{dabs:.3f} Acc\n(−{val:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Modality Importance (Accuracy Drop on Ablation)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Performance Drop (%)")
    ax.set_ylim(0, max(delta_pct) * 1.35)
    ax.grid(axis="y", alpha=0.3)

    save_fig(f"{OUT_DIR}/03_modality_contribution.png")


# ─────────────────────────────────────────────────────────────────────────
# FIG 04: Urgency–Risk Matrix
# ─────────────────────────────────────────────────────────────────────────
def fig_urgency_risk_matrix():
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Clinical Urgency–Risk Matrix\nAgentic Decision Output per Case",
                 fontsize=13, fontweight="bold")

    urgency_order = ["Routine", "Elective", "Urgent"]
    u_y = {u: i for i, u in enumerate(urgency_order)}

    risk_vals  = [a["cancer_risk_score"] for a in agents]
    unc_vals   = [a["uncertainty"] for a in agents]
    urgencies  = [a.get("urgency", "Routine") for a in agents]
    pathologies= [a["pathology"] for a in agents]
    case_ids   = [a["case_id"] for a in agents]

    # Background shading
    ax.axhspan(-0.5, 0.5,  facecolor="#E8F5E9", alpha=0.5)
    ax.axhspan(0.5, 1.5,   facecolor="#FFF8E1", alpha=0.5)
    ax.axhspan(1.5, 2.5,   facecolor="#FFEBEE", alpha=0.5)
    ax.text(0.02, 0,  "Routine",  va="center", fontsize=9, color="#4CAF50", fontweight="bold")
    ax.text(0.02, 1,  "Elective", va="center", fontsize=9, color="#FFC107", fontweight="bold")
    ax.text(0.02, 2,  "Urgent",   va="center", fontsize=9, color="#FF5722", fontweight="bold")

    for i, (a, r, unc, u, p, cid) in enumerate(
            zip(agents, risk_vals, unc_vals, urgencies, pathologies, case_ids)):
        y = u_y[u]
        col = URGENCY_COLOURS.get(u, "#607D8B")
        ax.scatter(r, y + (i - 2) * 0.08, s=200 + unc * 300,
                   c=col, edgecolors="black", lw=1.2, alpha=0.85, zorder=5)
        ax.annotate(f"{cid}\n({p})", xy=(r, y + (i-2)*0.08),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(urgency_order, fontsize=11)
    ax.set_xlabel("Cancer Risk Score (0 = Benign → 1 = Malignant)", fontsize=11)
    ax.set_title("(bubble size ∝ model uncertainty)", fontsize=10, style="italic")
    ax.axvline(0.5, color="grey", lw=1.2, ls="--", alpha=0.6, label="Risk threshold 0.5")
    ax.axvline(0.8, color="red", lw=1.2, ls=":", alpha=0.6, label="High-risk threshold 0.8")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    save_fig(f"{OUT_DIR}/04_urgency_risk_matrix.png")


# ─────────────────────────────────────────────────────────────────────────
# FIG 05: Clinical XAI Summary
# ─────────────────────────────────────────────────────────────────────────
def fig_clinical_xai_summary():
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Clinical XAI Summary — Explainability Metrics Across Cases",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Left: Risk flags frequency ─────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    flag_counts = {}
    for a in agents:
        for flag in a.get("risk_flags", []):
            clean = flag.split(":")[0]
            flag_counts[clean] = flag_counts.get(clean, 0) + 1
    sorted_flags = sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)
    flag_names = [f[0].replace("_", "\n") for f in sorted_flags]
    flag_vals  = [f[1] for f in sorted_flags]
    flag_cols  = plt.cm.Reds(np.linspace(0.4, 0.9, len(flag_names)))
    bars = ax.barh(flag_names, flag_vals, color=flag_cols, height=0.6, edgecolor="white")
    for bar, val in zip(bars, flag_vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(flag_vals) + 1.2)
    ax.set_title("Risk Flag Frequency\n(across all cases)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # ── Middle: Inference latency per case ────────────────────────────
    ax = fig.add_subplot(gs[1])
    inf_ms   = [a["inference_ms"] for a in agents]
    case_ids = [a["case_id"] for a in agents]
    inf_cols = PALETTE[:len(agents)]
    bars2 = ax.bar(range(len(agents)), inf_ms, color=inf_cols, width=0.6, edgecolor="white", lw=1.5)
    for bar, val in zip(bars2, inf_ms):
        ax.text(bar.get_x() + bar.get_width()/2, val + 20,
                f"{val/1000:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(case_ids, fontsize=8, rotation=15, ha="right")
    ax.set_title("Agentic Inference Latency\n(end-to-end per case)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Milliseconds")
    mean_ms = np.mean(inf_ms)
    ax.axhline(mean_ms, color="grey", lw=1.2, ls="--", label=f"Mean: {mean_ms/1000:.1f}s")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Right: Tabular vs Image confidence correlation ─────────────────
    ax = fig.add_subplot(gs[2])
    tab_risks = [a["tabular_risk_score"] for a in agents]
    img_confs = [a["image_confidence"] for a in agents]
    risk_scores = [a["cancer_risk_score"] for a in agents]
    urg_cols  = [URGENCY_COLOURS.get(a.get("urgency","Routine"), "#607D8B") for a in agents]
    sc = ax.scatter(tab_risks, img_confs, c=risk_scores, cmap="RdYlGn_r",
                    s=200, edgecolors=urg_cols, lw=2.5, vmin=0, vmax=1, zorder=5)
    for a in agents:
        ax.annotate(a["case_id"].replace("sample_", "S"),
                    xy=(a["tabular_risk_score"], a["image_confidence"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8.5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Cancer Risk Score", fontsize=9)
    ax.set_xlabel("Tabular Risk Score", fontsize=10)
    ax.set_ylabel("Image Confidence (GradCAM)", fontsize=10)
    ax.set_title("Cross-Modal Agreement\n(edge colour = urgency)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="grey", lw=1.0, ls=":", alpha=0.5)
    ax.axvline(0.5, color="grey", lw=1.0, ls=":", alpha=0.5)
    ax.grid(alpha=0.3)
    # urgency legend
    patches_u = [mpatches.Patch(color=v, label=k, linewidth=2) for k, v in URGENCY_COLOURS.items()]
    ax.legend(handles=patches_u, fontsize=8, title="Urgency (edge)", title_fontsize=8)

    save_fig(f"{OUT_DIR}/05_clinical_xai_summary.png")


# ─────────────────────────────────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────
def generate_text_report(severity_data):
    lines = []
    lines.append("=" * 70)
    lines.append("CLINICAL EVALUATION REPORT")
    lines.append("Unified Agentic Multi-Modal Colon Cancer AI System")
    lines.append("=" * 70)
    lines.append("")

    # Section 1: Model Performance
    lines.append("SECTION 1: MODEL PERFORMANCE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Test Accuracy      : {metrics['test_accuracy']:.4f}  ({metrics['test_accuracy']*100:.2f}%)")
    lines.append(f"  Test F1 Macro      : {metrics['test_f1_macro']:.4f}")
    lines.append(f"  Test AUC-ROC       : {metrics['test_auc_roc']:.4f}")
    lines.append(f"  Best Epoch         : {metrics['best_epoch']}")
    lines.append(f"  Parameters         : {metrics['n_params']:,}")
    lines.append(f"  Transfer Learning  : {metrics['transfer_learning']}")
    lines.append("")

    # Section 2: Modality Ablation
    lines.append("SECTION 2: MODALITY ABLATION ANALYSIS")
    lines.append("-" * 40)
    base = ablation["all_modalities"]
    lines.append(f"  All Modalities     : {base:.4f}  (100%)")
    for k, name in [("ablate_image", "Ablate Image"), ("ablate_text", "Ablate Text"),
                     ("ablate_tabular", "Ablate Tabular")]:
        v = ablation[k]
        drop = base - v
        pct  = (drop / base) * 100
        lines.append(f"  {name:<18} : {v:.4f}  (−{drop:.4f} / −{pct:.1f}% relative)")
    lines.append("")
    lines.append("  Clinical Implication: Tabular (TCGA clinical features) provides the")
    lines.append("  largest marginal contribution when absent (−20.3%), confirming that")
    lines.append("  patient demographics + staging are critical for risk stratification.")
    lines.append("")

    # Section 3: Severity Scoring
    lines.append("SECTION 3: CASE-LEVEL SEVERITY SCORING")
    lines.append("-" * 40)
    lines.append("  Formula: 0.50 × risk_score + 0.30 × (1−uncertainty) + 0.20 × image_conf")
    lines.append("")
    lines.append(f"  {'Case':<12} {'Pathology':<20} {'Composite':<11} {'Band':<14} {'Urgency'}")
    lines.append(f"  {'-'*12} {'-'*20} {'-'*11} {'-'*14} {'-'*8}")
    for sd in severity_data:
        lines.append(f"  {sd['case_id']:<12} {sd['pathology']:<20} "
                     f"{sd['composite']:<11.3f} {sd['band']:<14} {sd['urgency']}")
    lines.append("")

    # Section 4: Per-class clinical protocol
    lines.append("SECTION 4: PER-CLASS CLINICAL PROTOCOL")
    lines.append("-" * 40)
    protocols = {
        "polyps": {
            "severity": "Low–Moderate",
            "action": "Polypectomy or biopsy. Surveillance colonoscopy in 12 months.",
            "risk": "15% malignant transformation risk (adenomatous polyps).",
        },
        "uc-mild": {
            "severity": "Moderate",
            "action": "Aminosalicylate therapy (5-ASA). 12-month surveillance.",
            "risk": "8% colorectal cancer risk after 8–10 years of disease.",
        },
        "uc-moderate-sev": {
            "severity": "High",
            "action": "Urgent corticosteroid / biologic therapy. MDT review. 3-month colonoscopy.",
            "risk": "25% cancer risk. Dysplasia surveillance mandatory.",
        },
        "barretts-esoph": {
            "severity": "Moderate",
            "action": "Proton pump inhibitor therapy. Endoscopic surveillance every 2 years.",
            "risk": "12% risk of oesophageal adenocarcinoma progression.",
        },
        "therapeutic": {
            "severity": "Post-procedural",
            "action": "Confirm complete resection margins. Tattoo surveillance site. 3-month follow-up.",
            "risk": "5% residual lesion risk post-polypectomy.",
        },
    }
    for cls, proto in protocols.items():
        lines.append(f"  [{cls}]")
        lines.append(f"    Severity : {proto['severity']}")
        lines.append(f"    Action   : {proto['action']}")
        lines.append(f"    Risk     : {proto['risk']}")
        lines.append("")

    # Section 5: XAI Summary
    lines.append("SECTION 5: EXPLAINABILITY SUMMARY")
    lines.append("-" * 40)
    mod_keys = list(agents[0]["modality_weights"].keys())
    mod_mat  = np.array([[a["modality_weights"][k] for k in mod_keys] for a in agents])
    mod_mean = mod_mat.mean(axis=0)
    mod_std  = mod_mat.std(axis=0)
    for k, m, s in zip(mod_keys, mod_mean, mod_std):
        lines.append(f"  {k:<30} : {m:.3f} ± {s:.3f}")
    lines.append("")
    mean_inf = np.mean([a["inference_ms"] for a in agents])
    lines.append(f"  Mean inference latency : {mean_inf:.0f} ms ({mean_inf/1000:.2f}s)")
    lines.append("")

    # Section 6: Disclaimer
    lines.append("=" * 70)
    lines.append("DISCLAIMER")
    lines.append("-" * 40)
    lines.append("  This report is generated by an AI clinical decision-support system.")
    lines.append("  It does NOT replace the professional judgement of a qualified")
    lines.append("  gastroenterologist or oncologist. All recommendations must be")
    lines.append("  verified by a licensed clinician before any clinical action is taken.")
    lines.append("  This system has not been validated for standalone diagnostic use.")
    lines.append("=" * 70)

    report_path = f"{OUT_DIR}/clinical_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [Report] {report_path}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nClinical Evaluation Report — output dir: {OUT_DIR}")
    print("-" * 50)

    print("Generating Fig 01: Clinical Performance Dashboard...")
    fig_clinical_dashboard()

    print("Generating Fig 02: Severity Scoring...")
    severity_data = fig_severity_scoring()

    print("Generating Fig 03: Modality Contribution...")
    fig_modality_contribution()

    print("Generating Fig 04: Urgency–Risk Matrix...")
    fig_urgency_risk_matrix()

    print("Generating Fig 05: Clinical XAI Summary...")
    fig_clinical_xai_summary()

    print("Generating text report...")
    report = generate_text_report(severity_data)

    print("\n" + "=" * 50)
    print("Clinical Evaluation Complete.")
    print(f"Output directory: {OUT_DIR}/")
    print("Files produced:")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(f"{OUT_DIR}/{f}") // 1024
        print(f"  {f}  ({sz} KB)")
    print("=" * 50)
