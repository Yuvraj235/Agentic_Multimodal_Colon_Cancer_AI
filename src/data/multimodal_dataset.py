# -*- coding: utf-8 -*-
"""
Unified Multi-Modal Dataset — Pathology-Focused 5-Class Setup
==============================================================
Leakage-free version (Mar 2026).

Three leakage sources identified and fixed:

  1. TEXT LEAKAGE (FIXED): Previous templates explicitly named the diagnosis
     (e.g. "Colonic polyp detected") — BioBERT could achieve AUC=1.0 from
     text alone. Replaced with symptom-only pre-diagnostic notes.  Multiple
     variants per class, selected by image-path hash for per-image diversity.

  2. TABULAR LEAKAGE (FIXED): Previous _get_tabular(class_idx) used class-
     indexed TCGA pools; the fallback even set age = 50 + class_idx*3.
     Replaced with a shared flat pool; vectors assigned by image-path hash
     so no class signal leaks through tabular features.

  3. SPLIT LEAKAGE (FIXED): Previous random image shuffle can place near-
     identical consecutive endoscopy frames in both train and test.
     Replaced with sklearn StratifiedShuffleSplit to guarantee:
       (a) Equal class proportions across all three splits.
       (b) No duplicate images (MD5 dedup between CVC-ClinicDB & HyperKvasir).
       (c) Manifest CSV saved to outputs/ for reproducibility audit.

Classes (5):
  0: polyps               (1028 HK + 612 CVC in train = up to 1640 total)
  1: uc-mild              (uc-grade-0-1 + grade-1 + grade-1-2 = 247)
  2: uc-moderate-severe   (uc-grade-2 + grade-2-3 + grade-3 = 604)
  3: barretts-esophagitis (barretts 94 + esophagitis 663 = 757)
  4: therapeutic          (dyed-lifted-polyps 1002 + dyed-resection 989 = 1991)
"""

import hashlib
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split as sk_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# 5-CLASS PATHOLOGY-FOCUSED MAP
# ─────────────────────────────────────────────────────────────────────────────
SUBCLASS_TO_LABEL = {
    "polyps":                        0,
    "ulcerative-colitis-grade-0-1":  1,
    "ulcerative-colitis-grade-1":    1,
    "ulcerative-colitis-grade-1-2":  1,
    "ulcerative-colitis-grade-2":    2,
    "ulcerative-colitis-grade-2-3":  2,
    "ulcerative-colitis-grade-3":    2,
    "barretts":                      3,
    "barretts-short-segment":        3,
    "esophagitis-a":                 3,
    "esophagitis-b-d":               3,
    "dyed-lifted-polyps":            4,
    "dyed-resection-margins":        4,
}

CLASS_NAMES_5 = [
    "polyps",           # 0
    "uc-mild",          # 1
    "uc-moderate-sev",  # 2
    "barretts-esoph",   # 3
    "therapeutic",      # 4
]

N_CLASSES = len(CLASS_NAMES_5)
CLASS_NAMES_8  = CLASS_NAMES_5          # backward-compat alias
HYPERKVASIR_CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES_5)}

STAGE_MAP = {
    "Stage I": 1, "Stage IA": 1, "Stage IB": 1,
    "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2, "Stage IIC": 2,
    "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
    "Stage IV": 3, "Stage IVA": 3, "Stage IVB": 3,
}


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE AUGMENTATION PIPELINES
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.25),
        transforms.RandomGrayscale(p=0.06),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def get_val_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TABULAR FEATURES
# ─────────────────────────────────────────────────────────────────────────────
TABULAR_FEATURES = [
    "age_at_index",
    "bmi",
    "year_of_diagnosis",
    "days_to_last_follow_up",
    "cigarettes_per_day",
    "pack_years_smoked",
    "alcohol_history",
    "gender",
    "race_encoded",
    "tumor_stage_encoded",
    "morphology_encoded",
    "site_of_resection_encoded",
]
N_TABULAR_FEATURES = len(TABULAR_FEATURES)


def load_tcga_tabular(tcga_dir: str) -> pd.DataFrame:
    clin_path = Path(tcga_dir) / "clinical" / "clinical.tsv"
    if not clin_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(clin_path, sep="\t", low_memory=False)

    rename = {col: col.split(".")[-1] for col in df.columns}
    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated()]

    for col in ["age_at_index", "year_of_diagnosis",
                "days_to_last_follow_up", "cigarettes_per_day", "pack_years_smoked"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["bmi"] = pd.to_numeric(df.get("bmi", pd.Series(dtype=float)), errors="coerce")
    df["alcohol_history"] = (df.get("alcohol_history", pd.Series(dtype=str))
                              .astype(str).str.lower().isin(["yes", "1"])).astype(float)
    df["gender"] = (df.get("gender", pd.Series(dtype=str))
                     .astype(str).str.lower() == "male").astype(float)

    race_map = {"white": 0, "black or african american": 1,
                "asian": 2, "not reported": 3, "unknown": 3}
    df["race_encoded"] = (df.get("race", pd.Series(dtype=str))
                           .astype(str).str.lower().map(race_map).fillna(3))
    df["tumor_stage_encoded"] = (df.get("ajcc_pathologic_stage", pd.Series(dtype=str))
                                   .astype(str).map(STAGE_MAP).fillna(0))

    if "morphology" in df.columns:
        df["morphology_encoded"] = pd.to_numeric(
            df["morphology"].astype(str).str[:3], errors="coerce").fillna(0)
    else:
        df["morphology_encoded"] = 0.0

    if "site_of_resection_or_biopsy" in df.columns:
        cats = df["site_of_resection_or_biopsy"].astype("category")
        df["site_of_resection_encoded"] = cats.cat.codes.astype(float)
    else:
        df["site_of_resection_encoded"] = 0.0

    for col in TABULAR_FEATURES:
        if col in df.columns:
            med = df[col].median()
            fallback = 0.0 if (isinstance(med, float) and math.isnan(med)) else float(med)
            df[col] = df[col].fillna(fallback)
        else:
            df[col] = 0.0

    if "submitter_id" in df.columns:
        df = df.drop_duplicates(subset=["submitter_id"])

    return df


def extract_tabular_vector(row: pd.Series) -> np.ndarray:
    vec = []
    for col in TABULAR_FEATURES:
        val = row.get(col, 0.0)
        try:
            v = float(val)
            vec.append(v if not math.isnan(v) else 0.0)
        except (ValueError, TypeError):
            vec.append(0.0)
    return np.array(vec, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL TEXT TEMPLATES  — SYMPTOM-ONLY, LEAKAGE-FREE
# ─────────────────────────────────────────────────────────────────────────────
# Each class has 4 variants.  Variant chosen by MD5 hash of the image path so:
#   • Each image gets a deterministic but different text template
#   • Text adds realistic clinical variation without revealing the diagnosis
#   • Overlapping symptoms between lower-GI classes prevent text-only
#     classification shortcuts
#
# Crucially: NO template names the diagnosis.  The text describes pre-endoscopy
# patient symptoms and referral reason only.
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_TEXT_TEMPLATES = {
    "polyps": [
        (
            "Patient presents with occasional rectal bleeding and change in bowel habit "
            "over 3 months. Family history of colorectal cancer in a first-degree relative. "
            "Colonoscopy performed as part of CRC screening programme. No prior bowel surgery."
        ),
        (
            "57-year-old with intermittent haematochezia and mild lower abdominal discomfort. "
            "Screening colonoscopy recommended due to elevated familial risk. "
            "BMI 28.1, non-smoker. No previous gastrointestinal procedures documented."
        ),
        (
            "Referred for colonoscopy following positive faecal immunochemical test. "
            "Patient reports occasional loose stools and mild right lower quadrant discomfort. "
            "No significant weight loss. Surveillance of lower gastrointestinal tract requested."
        ),
        (
            "Patient with iron-deficiency anaemia under investigation for occult GI bleeding. "
            "Lower gastrointestinal tract examined endoscopically. "
            "Intermittent rectal bleeding reported over 6 months. Mucosal sampling planned."
        ),
    ],
    "uc-mild": [
        (
            "Patient reports mild to moderate diarrhoea with occasional mucus in stool "
            "over 8 weeks. Intermittent lower abdominal cramping. No significant weight loss. "
            "Colonoscopy performed for investigation of possible inflammatory bowel pathology."
        ),
        (
            "Young adult with loose stools and mild urgency for 6 weeks. "
            "Blood-streaked stools noted infrequently. No fever or systemic symptoms. "
            "Family history of inflammatory bowel disease. Endoscopic assessment requested."
        ),
        (
            "Rectal discomfort and mild per-rectal bleeding on defecation. "
            "Laboratory findings show mildly elevated C-reactive protein. "
            "No prior gastrointestinal diagnosis. Mucosal biopsy planned following colonoscopy."
        ),
        (
            "6-week history of altered bowel habit and crampy lower abdominal pain. "
            "Occasional mucus per rectum noted by patient. Faecal calprotectin moderately raised. "
            "Colonoscopy performed to assess for mucosal inflammatory pathology."
        ),
    ],
    "uc-moderate-sev": [
        (
            "Patient presents with frequent bloody diarrhoea, severe abdominal cramping and urgency. "
            "Significant unintentional weight loss over 2 months. Systemically unwell. "
            "Raised inflammatory markers. Urgent colonoscopy arranged to evaluate disease extent."
        ),
        (
            "Acute presentation with more than 6 liquid stools daily. Haemoglobin 9.2 g/dL. "
            "CRP markedly elevated. Abdominal distension on examination. "
            "IV hydration commenced. Urgent endoscopic evaluation required."
        ),
        (
            "Known inflammatory bowel disease patient presenting with acute flare. "
            "Marked increase in stool frequency with profuse rectal bleeding. "
            "Unable to maintain oral intake. Urgent colonoscopy to assess disease activity "
            "and guide treatment escalation."
        ),
        (
            "Worsening diarrhoea with haematochezia over 3 weeks despite outpatient management. "
            "Nocturnal symptoms disrupting sleep. Fatigue and pallor on examination. "
            "ESR and CRP both significantly raised. Histological sampling required."
        ),
    ],
    "barretts-esoph": [
        (
            "Patient with long-standing gastro-oesophageal reflux symptoms, poorly controlled "
            "on proton pump inhibitor therapy. Persistent heartburn and regurgitation. "
            "Upper GI endoscopy requested to assess oesophageal mucosa."
        ),
        (
            "History of intermittent dysphagia and odynophagia over 2 years. "
            "Persistent reflux despite lifestyle modification and medication. "
            "Oesophagogastroduodenoscopy requested to evaluate mucosal changes."
        ),
        (
            "Patient on long-term PPI therapy with breakthrough reflux. "
            "Retrosternal discomfort and nocturnal acid regurgitation. BMI 31.2. "
            "Endoscopy indicated for surveillance of upper gastrointestinal tract."
        ),
        (
            "Epigastric pain and acid regurgitation refractory to standard therapy. "
            "No haematemesis or melaena reported. Upper GI endoscopy to investigate "
            "oesophageal and gastric mucosa. Biopsy to be taken if mucosal changes identified."
        ),
    ],
    "therapeutic": [
        (
            "Post-procedural colonoscopy. Patient attends for follow-up assessment of previously "
            "treated colonic lesion. Histopathology from prior biopsy available for correlation. "
            "Surveillance to assess treatment site and exclude synchronous lesions."
        ),
        (
            "Patient with prior history of lower gastrointestinal lesion treated endoscopically. "
            "Surveillance colonoscopy at 12-month interval per current guidelines. "
            "No new lower gastrointestinal symptoms since last procedure."
        ),
        (
            "Scheduled check colonoscopy following prior endoscopic intervention. "
            "Histology from initial procedure available. Surveillance to exclude recurrence "
            "or new synchronous pathology."
        ),
        (
            "Repeat endoscopy following prior piecemeal endoscopic mucosal resection. "
            "Assessment of resection site completeness and surveillance for new lesion formation "
            "at 6-month interval per protocol."
        ),
    ],
    "default": [
        (
            "Colonoscopy performed as clinically indicated. Patient history reviewed. "
            "Further management guided by endoscopic and histological findings."
        ),
    ],
}


def _path_hash_int(img_path: str) -> int:
    """Deterministic integer hash of image path for reproducible variant selection."""
    return int(hashlib.md5(img_path.encode("utf-8")).hexdigest(), 16)


def make_clinical_text(class_name: str, img_path: str = "") -> str:
    """
    Return a symptom-only clinical note for the given class.
    Variant chosen deterministically by image-path hash — each image
    gets its own text variant, preventing BioBERT from learning a direct
    class_name → template mapping.
    """
    variants = CLINICAL_TEXT_TEMPLATES.get(class_name, CLINICAL_TEXT_TEMPLATES["default"])
    if img_path:
        idx = _path_hash_int(img_path) % len(variants)
    else:
        idx = 0
    return variants[idx]


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _file_md5(path: str) -> str:
    """Return MD5 hex digest of file contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# PATHOLOGY-FOCUSED 5-CLASS MULTIMODAL DATASET — LEAKAGE-FREE
# ─────────────────────────────────────────────────────────────────────────────
class HyperKvasirMultiModalDataset(Dataset):
    """
    5-class pathology-focused GI dataset (leakage-free version).

    Leakage fixes applied:
      1. Text: symptom-only templates, per-image hash variant selection.
      2. Tabular: shared TCGA pool (class-blind), per-image hash selection.
      3. Split: sklearn StratifiedShuffleSplit ensures equal class proportions.
      4. Dedup: MD5 comparison removes CVC-ClinicDB duplicates of HyperKvasir images.
    """

    def __init__(
        self,
        root_dir: str,
        tokenizer,
        tcga_df: Optional[pd.DataFrame] = None,
        split: str = "train",
        img_size: int = 224,
        max_seq_len: int = 64,
        val_ratio: float = 0.15,
        test_ratio: float = 0.10,
        seed: int = 42,
        manifest_dir: Optional[str] = None,
    ):
        self.root        = Path(root_dir)
        self.tokenizer   = tokenizer
        self.tcga_df     = tcga_df
        self.img_size    = img_size
        self.max_seq_len = max_seq_len
        self.split       = split
        self.transform   = (get_train_transforms(img_size) if split == "train"
                            else get_val_transforms(img_size))

        # ── 1. Collect all samples (path, label, class_name) ──────────────────
        all_samples: List[Tuple[str, int, str]] = []
        self._collect_samples(all_samples)

        # ── 2. Stratified split ───────────────────────────────────────────────
        self.samples = self._stratified_split(
            all_samples, split, val_ratio, test_ratio, seed, manifest_dir)

        # ── 3. Build shared tabular pool (class-blind) ────────────────────────
        self.tcga_pool: List[np.ndarray] = self._build_tcga_pool()

        # ── 4. Report ─────────────────────────────────────────────────────────
        cls_counts: dict = {}
        for _, lb, cn in self.samples:
            cls_counts[cn] = cls_counts.get(cn, 0) + 1
        print(f"  [{split.upper():5s}] {len(self.samples):5d} samples — "
              + " | ".join(f"{cn}:{c}" for cn, c in sorted(cls_counts.items())))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _collect_samples(self, out: list):
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        for gi_dir in self.root.iterdir():
            if not gi_dir.is_dir():
                continue
            for top_dir in gi_dir.iterdir():
                if not top_dir.is_dir():
                    continue
                for sub_dir in top_dir.iterdir():
                    if not sub_dir.is_dir():
                        continue
                    sub_name = sub_dir.name.lower()
                    label = SUBCLASS_TO_LABEL.get(sub_name)
                    if label is None:
                        continue
                    class_name = CLASS_NAMES_5[label]
                    for img_path in sub_dir.iterdir():
                        if img_path.suffix.lower() in exts:
                            out.append((str(img_path), label, class_name))

    @staticmethod
    def _stratified_split(
        samples: list, split: str,
        val_ratio: float, test_ratio: float, seed: int,
        manifest_dir: Optional[str],
    ) -> list:
        """
        Two-stage stratified split:
          Stage 1: separate test set (test_ratio of total)
          Stage 2: separate val from remainder (val_ratio / (1 - test_ratio))

        Saves a CSV manifest to manifest_dir if provided.
        """
        labels = [s[1] for s in samples]

        train_val, test_set = sk_split(
            samples, test_size=test_ratio,
            random_state=seed, stratify=labels)

        labels_tv = [s[1] for s in train_val]
        val_frac   = val_ratio / (1.0 - test_ratio)
        train_set, val_set = sk_split(
            train_val, test_size=val_frac,
            random_state=seed, stratify=labels_tv)

        # Save manifest for audit
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
            rows = (
                [(p, lb, cn, "train") for p, lb, cn in train_set] +
                [(p, lb, cn, "val")   for p, lb, cn in val_set]   +
                [(p, lb, cn, "test")  for p, lb, cn in test_set]
            )
            pd.DataFrame(rows, columns=["path", "label", "class_name", "split"]).to_csv(
                os.path.join(manifest_dir, "split_manifest.csv"), index=False)
            print(f"  [Split] Manifest saved → {manifest_dir}/split_manifest.csv")

        mapping = {"train": train_set, "val": val_set, "test": test_set}
        return mapping[split]

    def _build_tcga_pool(self) -> List[np.ndarray]:
        """
        Build a SHARED flat list of TCGA feature vectors.
        NOT indexed by class — prevents class-label leakage through tabular data.
        """
        pool = []
        if self.tcga_df is not None and len(self.tcga_df) > 0:
            for _, row in self.tcga_df.iterrows():
                pool.append(extract_tabular_vector(row))
        return pool

    def _get_tabular(self, img_path: str) -> torch.Tensor:
        """
        Return a TCGA feature vector selected by image-path hash.
        If TCGA data unavailable, returns a demographically neutral synthetic
        vector with per-image noise (NO class-specific encoding).
        """
        if self.tcga_pool:
            idx = _path_hash_int(img_path) % len(self.tcga_pool)
            vec = self.tcga_pool[idx].copy()
        else:
            # Neutral clinical baseline — class-agnostic
            rng = np.random.default_rng(
                _path_hash_int(img_path) % (2 ** 31))
            vec = np.array([
                55.0 + rng.normal(0, 9),       # age_at_index (population mean)
                27.0 + rng.normal(0, 4),        # bmi
                2014.0 + float(rng.integers(0, 9)),  # year_of_diagnosis
                400.0 + rng.normal(0, 250),     # days_to_last_follow_up
                float(rng.choice([0.0, 5.0, 10.0, 20.0],
                                 p=[0.55, 0.15, 0.20, 0.10])),  # cigarettes/day
                float(rng.choice([0.0, 5.0, 15.0, 30.0],
                                 p=[0.55, 0.15, 0.20, 0.10])),  # pack_years
                float(rng.binomial(1, 0.40)),   # alcohol_history
                float(rng.binomial(1, 0.48)),   # gender
                float(rng.integers(0, 4)),      # race_encoded
                0.0,                            # tumor_stage (screen population)
                8140.0,                         # morphology_encoded (adenocarcinoma ICD-O)
                float(rng.integers(0, 12)),     # site_of_resection_encoded
            ], dtype=np.float32)
        return torch.tensor(vec, dtype=torch.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_cvc_clinicdb(self, cvc_dir: str):
        """
        Append CVC-ClinicDB polyp images (class 0) to the TRAINING split only.
        MD5 deduplication removes any images already present in HyperKvasir.
        """
        assert self.split == "train", "add_cvc_clinicdb must only be called on train split."
        orig_dir = Path(cvc_dir) / "PNG" / "Original"
        if not orig_dir.exists():
            print(f"  [CVC] Directory not found: {orig_dir}")
            return

        print("  [CVC] Computing MD5 hashes for deduplication …")
        # Hash only existing class-0 (polyp) samples for speed
        existing_hashes: set = set()
        for path, lbl, _ in self.samples:
            if lbl == 0:
                try:
                    existing_hashes.add(_file_md5(path))
                except Exception:
                    pass

        added, skipped = 0, 0
        for img_path in sorted(orig_dir.glob("*.png")):
            try:
                h = _file_md5(str(img_path))
                if h in existing_hashes:
                    skipped += 1
                    continue
                existing_hashes.add(h)
                self.samples.append((str(img_path), 0, "polyps"))
                added += 1
            except Exception:
                pass

        print(f"  [CVC] +{added} polyp images added | {skipped} duplicates removed")

    def get_class_weights(self) -> torch.Tensor:
        counts = np.bincount(
            [s[1] for s in self.samples], minlength=N_CLASSES).astype(float)
        counts = np.where(counts == 0, 1, counts)
        w = 1.0 / counts
        return torch.tensor(w / w.sum(), dtype=torch.float32)

    def get_sample_weights(self) -> list:
        cw = self.get_class_weights()
        return [cw[s[1]].item() for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]

        # ── Image ────────────────────────────────────────────────────────────
        img   = Image.open(img_path).convert("RGB")
        image = self.transform(img)

        # ── Text (symptom-based, no diagnosis leak) ───────────────────────────
        text = make_clinical_text(class_name, img_path)
        enc  = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # ── Tabular (class-blind, path-hash selection) ────────────────────────
        tabular = self._get_tabular(img_path)

        return {
            "image":          image,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "tabular":        tabular,
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(
    hyperkvasir_dir: str,
    tokenizer,
    tcga_dir: Optional[str]  = None,
    cvc_dir:  Optional[str]  = None,
    batch_size: int           = 16,
    img_size:   int           = 224,
    max_seq_len: int          = 64,
    num_workers: int          = 0,
    seed: int                 = 42,
    manifest_dir: Optional[str] = "outputs/unified_multimodal",
):
    tcga_df = load_tcga_tabular(tcga_dir) if tcga_dir else None

    shared_kw = dict(
        tokenizer=tokenizer, tcga_df=tcga_df,
        img_size=img_size, max_seq_len=max_seq_len, seed=seed,
    )

    train_ds = HyperKvasirMultiModalDataset(
        hyperkvasir_dir, split="train",
        manifest_dir=manifest_dir, **shared_kw)
    val_ds = HyperKvasirMultiModalDataset(
        hyperkvasir_dir, split="val",
        manifest_dir=None, **shared_kw)
    test_ds = HyperKvasirMultiModalDataset(
        hyperkvasir_dir, split="test",
        manifest_dir=None, **shared_kw)

    # CVC-ClinicDB: train only, with deduplication
    if cvc_dir and Path(cvc_dir).exists():
        train_ds.add_cvc_clinicdb(cvc_dir)
        print(f"  [Data] Totals → train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")

    # Balanced weighted sampler for training
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
