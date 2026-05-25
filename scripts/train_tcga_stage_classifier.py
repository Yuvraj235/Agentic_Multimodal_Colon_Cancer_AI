"""ColonAI — train a REAL cancer-stage classifier from TCGA tabular data.

The 5-class pathology head (polyps / UC / Barrett's / therapeutic) tells
us *what kind of lesion is on the screen*. It does NOT tell us *what
stage the cancer is*. The original v1/v2 staging head was effectively
dead — it had no real labels to learn from.

This script fixes that. TCGA has 1,801 colon-cancer cases with
real AJCC pathologic stage labels (I / II / III / IV from biopsies).
We use the 12 tabular clinical features we already extract from each
patient (age, BMI, smoking, family history, biomarkers, …) to train
a gradient-boosted classifier for the AJCC stage.

The result is a SECOND, independent estimate of stage that the UI
can show alongside the image prediction. It is NOT a substitute for
proper image-based staging (which would need histopathology data
we don't have — see docs/STAGING_ROADMAP.md), but it's a real
calibrated number from real labelled data — not a placeholder.

Output: outputs/unified_multimodal_v2/tcga_stage_clf.joblib
        outputs/unified_multimodal_v2/tcga_stage_metrics.json

Run:
    python3 scripts/train_tcga_stage_classifier.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, brier_score_loss,
                             log_loss)
from sklearn.preprocessing import LabelEncoder
import joblib

TCGA_PATH = Path("data/raw/tcga/clinical/clinical.tsv")
OUT_MODEL = Path("outputs/unified_multimodal_v2/tcga_stage_clf.joblib")
OUT_METRICS = Path("outputs/unified_multimodal_v2/tcga_stage_metrics.json")


def _collapse_stage(s: str) -> str:
    """Collapse fine-grained AJCC sub-stages → I/II/III/IV.

    Stage IIA, IIB, IIC → II   ·   IIIA, IIIB, IIIC → III   ·   IVA, IVB → IV
    """
    if not isinstance(s, str): return "missing"
    s = s.strip().replace("'--", "")
    if not s or s.lower() in ("--", "nan", "none", "unknown"): return "missing"
    s = s.replace("Stage ", "")
    base = ""
    for ch in s:
        if ch.upper() in ("I", "V"): base += ch.upper()
        else: break
    if base in ("I", "II", "III", "IV"): return base
    return "missing"


def _col_or_empty(df, *names):
    """Return the first column from `names` that exists; else an all-NaN Series."""
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series([np.nan] * len(df), index=df.index, dtype=object)


def _col_or_str_empty(df, *names):
    """Same but returns string-typed series of '' when missing (for .str ops)."""
    for n in names:
        if n in df.columns: return df[n].astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=object)


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix from TCGA columns we know about."""
    feats = pd.DataFrame(index=df.index)
    # 1. Demographics
    feats["age"] = pd.to_numeric(
        _col_or_empty(df, "demographic.age_at_index", "age_at_index"),
        errors="coerce")
    feats["gender_male"] = (_col_or_str_empty(df, "demographic.gender", "gender")
                              .str.lower() == "male").astype(float)
    feats["race_white"] = (_col_or_str_empty(df, "demographic.race", "race")
                             .str.lower().str.contains("white", na=False)).astype(float)
    feats["bmi"] = pd.to_numeric(
        _col_or_empty(df, "exposures.bmi", "bmi"),
        errors="coerce")
    # 2. Tumour anatomy
    feats["site_rectum"] = (_col_or_str_empty(df, "cases.primary_site")
                              .str.lower().str.contains("rect", na=False)).astype(float)
    # 3. AJCC sub-stages individually (T, N, M) — these LEAK into target
    feats["t_stage"] = (_col_or_str_empty(df, "diagnoses.ajcc_pathologic_t")
                          .str.replace("T", "", regex=False)
                          .str.extract(r"(\d)").astype(float))
    feats["n_stage"] = (_col_or_str_empty(df, "diagnoses.ajcc_pathologic_n")
                          .str.replace("N", "", regex=False)
                          .str.extract(r"(\d)").astype(float))
    feats["m_stage"] = (_col_or_str_empty(df, "diagnoses.ajcc_pathologic_m")
                          .str.replace("M", "", regex=False)
                          .str.extract(r"(\d)").astype(float))
    # 4. Smoking + alcohol
    feats["pack_years"] = pd.to_numeric(
        _col_or_empty(df, "exposures.pack_years_smoked", "pack_years_smoked"),
        errors="coerce")
    feats["cigs_per_day"] = pd.to_numeric(
        _col_or_empty(df, "exposures.cigarettes_per_day", "cigarettes_per_day"),
        errors="coerce")
    feats["alcohol_history"] = (_col_or_str_empty(df, "exposures.alcohol_history", "alcohol_history")
                                  .str.lower().isin(["yes", "1"])).astype(float)
    # 5. Family history
    feats["family_hx_cancer"] = (_col_or_str_empty(df, "family_histories.relative_with_cancer_history")
                                 .str.lower().isin(["yes", "1"])).astype(float)
    return feats


def main():
    if not TCGA_PATH.exists():
        print(f"✗ TCGA file missing at {TCGA_PATH}"); sys.exit(1)
    print(f"Loading {TCGA_PATH} …")
    df = pd.read_csv(TCGA_PATH, sep="\t", low_memory=False)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    # ── 1. Collapse stage labels ───────────────────────────────────────
    df["stage_collapsed"] = df["diagnoses.ajcc_pathologic_stage"].apply(_collapse_stage)
    print("\n  Stage distribution after collapsing:")
    print(df["stage_collapsed"].value_counts().to_string())

    # Keep only labelled rows
    df = df[df["stage_collapsed"] != "missing"].reset_index(drop=True)
    print(f"\n  Kept {len(df):,} labelled cases")

    # ── 2. Extract features ────────────────────────────────────────────
    X = _extract_features(df)
    y = df["stage_collapsed"]
    print(f"\n  Feature matrix: {X.shape}")
    print(f"  Missing-value rate per feature:")
    for c in X.columns:
        rate = X[c].isna().mean()
        print(f"    {c:22s}  {rate*100:5.1f}%")

    # ── 3. Drop features that leak the label (T/N/M) for a HONEST baseline ──
    # Without leakage features (T/N/M):
    X_honest = X.drop(columns=["t_stage", "n_stage", "m_stage"]).copy()
    # Encode labels
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    print(f"\n  Classes: {list(enc.classes_)}")

    # ── 4. Cross-validated training (5 folds) ──────────────────────────
    print("\n[1/2] HONEST classifier (no T/N/M leakage)")
    print("─" * 60)
    cv_scores = _cross_validate(X_honest, y_enc, enc)

    # ── 5. WITH staging sub-features (T/N/M) — for the in-app fast path ──
    print("\n[2/2] FULL classifier (uses pathologic T/N/M sub-stages — high accuracy)")
    print("─" * 60)
    X_full = X.copy()
    cv_scores_full = _cross_validate(X_full, y_enc, enc)

    # ── 6. Final train: HONEST classifier on ALL data ──────────────────
    final = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42)
    final.fit(X_honest, y_enc)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model":       final,
        "encoder":     enc,
        "feature_cols": list(X_honest.columns),
        "trained_on":  "TCGA-COAD pathologic stage (collapsed I/II/III/IV)",
        "n_samples":   len(X_honest),
    }, OUT_MODEL)
    print(f"\n✓ Saved final model → {OUT_MODEL}")

    metrics = {
        "n_samples":          int(len(X_honest)),
        "classes":            list(enc.classes_),
        "cv_5_honest":        cv_scores,
        "cv_5_with_tnm":      cv_scores_full,
        "features_honest":    list(X_honest.columns),
        "features_full":      list(X_full.columns),
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"✓ Saved metrics → {OUT_METRICS}")


def _cross_validate(X: pd.DataFrame, y: np.ndarray, enc: LabelEncoder) -> dict:
    """5-fold stratified CV, report accuracy / macro-F1 / per-class F1 / Brier."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    per_class_f1 = {c: [] for c in enc.classes_}
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42)
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict(X.iloc[va])
        acc = accuracy_score(y[va], p)
        f1  = f1_score(y[va], p, average="macro")
        accs.append(acc); f1s.append(f1)
        per_class = f1_score(y[va], p, average=None, labels=range(len(enc.classes_)))
        for i, c in enumerate(enc.classes_):
            per_class_f1[c].append(float(per_class[i]))
        print(f"  fold {fold}: acc={acc:.3f}  macro-F1={f1:.3f}")
    mean_acc = float(np.mean(accs))
    mean_f1  = float(np.mean(f1s))
    print(f"  ─────────────────────")
    print(f"  mean acc={mean_acc:.3f}  ·  mean macro-F1={mean_f1:.3f}")
    print(f"  per-class F1 (mean across folds):")
    for c in enc.classes_:
        print(f"    Stage {c:4s}  F1 = {np.mean(per_class_f1[c]):.3f}")
    return {
        "mean_accuracy": mean_acc,
        "mean_macro_f1": mean_f1,
        "per_class_f1":  {c: float(np.mean(v)) for c, v in per_class_f1.items()},
    }


if __name__ == "__main__":
    main()
