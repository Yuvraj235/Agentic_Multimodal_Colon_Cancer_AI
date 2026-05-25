# Roadmap: Real image-based cancer staging

## Why ColonAI v2 cannot stage cancer from images today

The pathology classifier ColonAI uses was trained on three public datasets:

- **HyperKvasir** (10,662 images, 23 GI sub-classes)
- **CVC-ClinicDB** (612 polyp + mask images)
- **TCGA clinical metadata** (1,802 patients — but tabular features only, **no images**)

These contain **screening-stage** colonoscopy frames: polyps, ulcerative colitis,
Barrett's oesophagus, normal mucosa, post-therapy sites. **None of these
datasets include biopsy-confirmed AJCC cancer stages (I / II / III / IV)
linked to images.** Without paired (image, stage) labels, no neural network
can learn image-based staging — only image-based screening.

The original v1 architecture had a 4-class `staging` head wired into the model,
but it had no real labels to learn from and produced essentially "no_cancer"
on every input. v2 honestly anchors it to v1 outputs via KL distillation
(so it doesn't drift) and **replaces it in the UI with the TCGA tabular
stage classifier** (53% accuracy on 4-class, from clinical features alone).

## What's actually needed for real image-based staging

| Step | Data | Effort | Cost |
|---|---|---|---|
| 1. Acquire paired (image, biopsy-confirmed stage) data | TCGA-COAD whole-slide images (free, ~2 TB), PAIP 2022 challenge data, EBHI-SEG | 1-2 weeks data prep | Free dataset, ~$300 storage |
| 2. Train a histopathology-aware encoder | ConvNeXt-V2 large + WSI patching pipeline (CLAM or similar) | 4 weeks training | ~$2,000 GPU rental |
| 3. Cross-modal alignment (endoscopy → histology features) | Need paired (colonoscopy frame, histology slide) for the same patient — *very rare* in public data | 6+ weeks | Requires institutional partnership |
| 4. Clinical validation | 100+ retrospective cases at a hospital partner, sensitivity/specificity per stage | 3-4 months | Hospital IRB + collaboration |

**Realistic timeline: 6-12 months and ~$10-30k in compute + storage**, plus
a university or hospital partnership for paired data. This is not a
weekend's worth of work.

## What we have RIGHT NOW that's clinically useful

| Output | Trained on | Honest accuracy | Where it appears in the UI |
|---|---|---|---|
| **5-class pathology** (polyps / UC mild / UC mod-sev / Barrett's / therapeutic) | HyperKvasir + CVC | Test acc 99.5% on val, but 0.27 → 0.61 IoU on cross-vendor mask alignment (after v2 retrain) | Main result banner |
| **Polyp segmentation mask** | Kvasir-SEG + 5 cross-vendor polyp datasets (2,348 mask-labelled images) | Cross-vendor IoU 0.61 (Pentax = Olympus) | GradCAM overlay & cross-check |
| **TCGA tabular stage estimate** (Stage I/II/III/IV) | TCGA-COAD 1,319 cases | 53% 4-class accuracy (vs 25% random); Stage IV F1 = 0.61 | New "Independent stage estimate" card |
| **Invasive-lesion override** | Pixel statistics (no neural network) | Rule-based — fires on deep ulceration / heavy bleeding / nodular surface / mass reflection | Red alert at top of results page |
| **6-agent cross-check** | All of the above + GradCAM + Integrated Gradients + segmentation | Forces show / abstain / reject decision; abstains when agents disagree | Verdict banner |

## What we explicitly DO NOT do (and a buyer should know it)

- ❌ **We do not stage colon cancer from a single image.** The TCGA tabular
  estimate is a population-level guess from clinical features, not a per-image
  classification.
- ❌ **We do not detect cancers other than colorectal.** Skin / breast / lung /
  cervical / brain cancer detection are entirely different projects, each
  needing their own datasets, architectures, and validation.
- ❌ **We do not classify polyp histology (adenoma vs hyperplastic vs serrated).**
  This is CADx (computer-aided diagnosis) and needs NICE/NICE-NBI labelled
  data we don't have.
- ❌ **We do not replace biopsy.** The system is decision-support for a clinician
  who will perform the biopsy if appropriate.

## Honest mitigations we DID build

1. **Invasive-lesion detector** (`src/app/image_atypicality.detect_advanced_lesion`)
   — rule-based pixel-stats override. When deep ulceration / bleeding /
   nodular mass is detected, the AI's 5-class call is downgraded to *"Atypical
   lesion — urgent endoscopist review"*. This stops the model from
   confidently mis-labelling an invasive carcinoma as `polyps`.

2. **Hierarchical UC hedge** (`src/app/smart_inference.py`)
   — when uc-mild is predicted but uc-mod-sev is also plausible (>30%),
   surface as *"UC of uncertain grade — recommend sigmoidoscopy"*. Stops the
   over-call on UC severity (recall on mod-sev was only 0.15 before).

3. **Per-class abstention thresholds** (`scripts/calibrate_per_class_thresholds.py`)
   — fitted on val data. uc-mild now needs 0.89 confidence (was 0.75)
   before being shown to the user.

4. **TCGA tabular stage classifier** (`scripts/train_tcga_stage_classifier.py`)
   — real labelled stage data, real 53% accuracy. Shown as a SECONDARY
   estimate alongside the image prediction.

## Datasets we would need to add to do this properly

| Capability | Dataset | Size | Access |
|---|---|---|---|
| Image-level cancer staging | TCGA-COAD WSI + clinical | ~2 TB | Free (GDC portal) |
| Polyp histology typing (adenoma/serrated/hyperplastic) | KumarLab Polyp-Histo | ~80 GB | Free (CC-BY) |
| Multi-centre, multi-vendor robustness | PolypGen 2021 | 8,037 images, 6 hospitals | Free (research-use) |
| Capsule endoscopy diversity | Kvasir-Capsule | 4,741 images, 14 classes | Free |
| Narrow-band imaging (NBI) for polyp typing | CVC-ClinicNBI | ~600 images | Free |
| **Real-world clinical pilot** | A hospital partner | 100+ cases | Requires IRB + collaboration |

## Bottom line

ColonAI today is **deployment-ready for colorectal screening assistance**.
It is not, and we do not claim it to be, a replacement for biopsy-based
cancer staging. The four honest mitigations above prevent the system from
giving fake confident stage calls. The 6-month roadmap to true image-based
staging is documented above — it needs data, compute, and clinical
partnership, none of which can be faked.
