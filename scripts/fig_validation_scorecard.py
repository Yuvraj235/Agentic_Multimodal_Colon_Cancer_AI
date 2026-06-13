"""Render the honest validation scorecard as a single figure.

Reads outputs/unified_multimodal_v2/clinical_validation_report.json (produced by
scripts/validate_clinical.py) and draws a colour-coded table: green = Tier A
(truly external), amber = Tier B (held-out split, in-distribution), grey = Tier C
(flagged optimistic / weak baseline). No numbers are typed here — they come from
the report, so the figure can never disagree with the signed JSON.

Output: outputs/unified_multimodal_v2/figures/validation_scorecard.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = Path("outputs/unified_multimodal_v2/clinical_validation_report.json")
OUT = Path("outputs/unified_multimodal_v2/figures/validation_scorecard.png")
TIER_COLOR = {"A": "#1b7f3b", "B": "#b8860b", "C": "#6b6b6b"}
TIER_BG = {"A": "#e7f5ec", "B": "#fbf3df", "C": "#eeeeee"}


def main():
    rep = json.loads(REP.read_text())
    m = rep["metrics"]

    def num(key):
        d = m[key]
        if "mean_iou" in d:
            ci = d.get("iou_95ci", [0, 0])
            return f"IoU {d['mean_iou']:.2f}  (95% CI {ci[0]:.2f}–{ci[1]:.2f})"
        if "auroc" in d:
            return f"AUROC {d['auroc']:.3f}"
        if "macro_f1" in d and d.get("macro_f1") is not None:
            return f"macro-F1 {d['macro_f1']:.3f}"
        if "balanced_acc" in d and d.get("balanced_acc") is not None:
            return f"bal-acc {d['balanced_acc']:.3f}"
        if "accuracy" in d and d.get("accuracy") is not None:
            return f"acc {d['accuracy']:.3f}"
        return ""

    rows = [
        ("Polyp localization (cross-vendor)", "segmentation_localization_ETIS_Pentax"),
        ("Out-of-scope gate (abstain)", "out_of_scope_detection"),
        ("View-quality gate", "view_quality_gate"),
        ("5-class finding", "pathology_5class"),
        ("Polyp characterization (CADx)", "polyp_characterization_CADx"),
        ("Stage from doctor's TNM", "staging_from_TNM"),
        ("Histology tissue type", "histology_tissue_classifier"),
        ("Stage from demographics (rejected)", "staging_from_demographics_REJECTED"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off")
    ax.set_title("ColonAI — honest validation scorecard", fontsize=15, fontweight="bold",
                 loc="left", pad=34)
    ax.text(0, 1.045, f"model {rep['checkpoint_sha16']} · seg {rep['seg_decoder_sha16']} · {rep['generated']}",
            transform=ax.transAxes, fontsize=8, color="#666")

    n = len(rows)
    y0, dy = 0.86, 0.105
    # header
    ax.text(0.005, y0 + 0.06, "Capability", fontsize=10, fontweight="bold")
    ax.text(0.46, y0 + 0.06, "Evidence", fontsize=10, fontweight="bold")
    ax.text(0.60, y0 + 0.06, "Headline number", fontsize=10, fontweight="bold")
    for i, (label, key) in enumerate(rows):
        tier = m[key]["tier"]
        y = y0 - i * dy
        ax.add_patch(plt.Rectangle((0, y - 0.045), 1.0, 0.092, transform=ax.transAxes,
                                   facecolor=TIER_BG[tier], edgecolor="none", zorder=0))
        ax.text(0.005, y, label, fontsize=10, va="center")
        ax.add_patch(plt.Rectangle((0.46, y - 0.028), 0.05, 0.056, transform=ax.transAxes,
                                   facecolor=TIER_COLOR[tier], edgecolor="none"))
        ax.text(0.485, y, tier, fontsize=10, fontweight="bold", color="white", va="center", ha="center")
        ax.text(0.60, y, num(key), fontsize=10, va="center", family="monospace")

    legend = ("A = truly external (never-seen data)    "
              "B = honest held-out split (same population)    "
              "C = flagged optimistic / weak baseline")
    ax.text(0, -0.02, legend, transform=ax.transAxes, fontsize=8.5, color="#444")
    ax.text(0, -0.09, "Numbers read directly from the signed validation report; "
            "cross-vendor IoU is recomputed live each run.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#777")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
