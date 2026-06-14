# Phase 2 — MRI/USG data-access spike (result)

**Date:** 2026-06-14 · **Outcome: STOP — no free, usable dataset. MRI/USG stay doctor-entered.**

The plan gated any MRI/USG model build on a free-data check (zero budget → the data
must be free *and* actually downloadable, not application-gated). Findings:

| Source | Modality | Free + downloadable now? | Verdict |
|---|---|---|---|
| **RC4** (657 pts, 51k slices — largest rectal-cancer MRI set) | MRI | ❌ No download link; "available on reasonable request to the authors" | **Gated** (like PLCO/PICCOLO — ruled out) |
| **TCIA — COLORECTAL-LIVER-METASTASES** | CT/MR (liver) | ✅ Free (NBIA) | Wrong target (liver mets, not rectal-primary T-staging) |
| **TCIA — CT COLONOGRAPHY** (825 cases) | CT | ✅ Free (NBIA) | Wrong modality (CT virtual colonoscopy, polyp detection — not MRI) |
| **Kaggle** "colorectal" sets (e.g. EBHI-Seg) | Histology | ✅ Free | Wrong modality (H&E tiles, not MRI) |
| **Colorectal EUS / ultrasound** | USG | ❌ None public | No dataset exists |

**Conclusion:** There is no free, directly-downloadable, T-stage-labelled rectal-MRI
dataset (and no public colorectal USG dataset). Training an honest MRI/USG model is
**not possible right now** without either gated data access or fabricating data — both
off the table.

**What we do instead (honest, already in place):**
- MRI/CT/USG images are still handled correctly: the input gate recognises a
  radiology scan and declines to "read" it (`image_atypicality.py`), pointing the
  user to a radiologist — it does not invent a result.
- MRI/CT findings enter the system the correct way: the **clinician enters T/N/M**
  (from their radiologist's report) and the app computes the exact AJCC stage
  (`structured_report.py` → `staging.py`). This matches real clinical practice
  (MRI is read by a radiologist; the AI applies the staging rulebook).

**If the user wants MRI later, the only realistic paths are:**
1. **Apply to the RC4 authors** for dataset access (gated; needs the user to request it).
2. Use **TCIA CT-COLONOGRAPHY** (free) to build a *CT* polyp tool — a different modality
   from what was asked, and we already have a strong colonoscopy detector.

Revisit if a free, labelled rectal-MRI dataset appears.
