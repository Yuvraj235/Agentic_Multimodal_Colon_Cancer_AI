# ColonAI — Clinical Validation Report (honest)

_Generated 2026-06-11 21:17:05 · model `8057c26cc9aeaa5d` · seg `d1ed738aac1f76a0`_

Every number is tagged by evidence strength: **A** = truly external (never-seen data), **B** = honest held-out split but same population, **C** = flagged optimistic / weak baseline.

| Capability | Tier | Headline number | Data | Honest read |
|---|---|---|---|---|
| Polyp localization (cross-vendor) | A | IoU 0.4547 (95% CI 0.4172–0.4935) | ETIS-Larib (Pentax), held out | Honest different-scanner number; lower than familiar scanners. |
| Out-of-scope gate | A | AUROC 0.9958 | held-out real out-of-scope | Catches ~99% of non-colon views. |
| View-quality gate | A | AUROC 0.9847 | held-out bowel-prep | Flags ~96% of poor views. |
| 5-class finding | B | macro-F1 0.8349 | HyperKvasir+CVC split | In-distribution; UC-mild recall low by safety design. |
| Polyp characterization (CADx) | B | balanced-acc 0.8502 | BKAI split | 'Benign' call defers to histology. |
| Stage from TNM | B | acc 0.997 | TCGA + AJCC rule | Deterministic; the only trusted staging path. |
| Histology tissue | C | macro-F1 0.9957 | CRC-VAL-HE tile split | Over-stated (tile-level split); demonstrator. |
| Stage from demographics | C | acc 0.5345 | TCGA demographics | ~53%; REJECTED — shown only to justify refusing it. |

## Bottom line

The system's strongest evidence is on the safety machinery (out-of-scope and view-quality gates, ~0.97-0.99 on held-out real data) and on honest cross-vendor localization (ETIS IoU ~0.45). The 5-class finding model is ~94% on its own split but unproven on outside hospitals. Cancer STAGE is trusted only when computed from the doctor's TNM (a fixed rule); it is never guessed from an image or demographics. The biggest open gap is true external validation of the finding model and a strict patient-level histology test.
