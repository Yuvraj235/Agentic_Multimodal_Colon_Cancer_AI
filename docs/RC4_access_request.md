# RC4 rectal-MRI dataset — data-access request (ready to send)

The RC4 dataset has no public download — access is "on reasonable request to the
authors." This is the request to send them. **Two placeholders to fill in first.**

## Step 1 — find the corresponding author + email
Open the paper (your library, ResearchGate, or the authors' institutional page —
ScienceDirect blocks automated access, so I couldn't fetch it):

- **Title:** *Towards semi-supervised multi-modal rectal cancer segmentation: a
  large-scale dataset and a multi-teacher uncertainty-aware network*
- **Journal:** Expert Systems with Applications (Elsevier), 2024
- **DOI / id:** S0957417424016014

On the paper's first page look for the **"Corresponding author"** (has the ✉ email),
and the **"Data availability"** statement (it names the contact + any terms). Put
those into the `[...]` slots below.

## Step 2 — email to send

> **Subject:** Request for access to the RC4 rectal-cancer MRI dataset (research use)
>
> Dear Dr. **[Corresponding Author surname]**,
>
> I read your paper *"Towards semi-supervised multi-modal rectal cancer
> segmentation: a large-scale dataset and a multi-teacher uncertainty-aware
> network"* (Expert Systems with Applications, 2024) and would like to request
> access to the **RC4 dataset** (the 657-patient, ~51,000-slice rectal-cancer MRI
> collection with clinical data described in the paper).
>
> I am working on a **non-commercial research / educational project** — an
> open, multimodal colorectal-cancer screening assistant. I would use RC4 only to
> develop and evaluate a **rectal-tumour segmentation and T-stage** model, with
> results reported honestly on held-out data. I will:
> - use the data for **research only**, not clinically or commercially;
> - **not redistribute** it or attempt any patient re-identification;
> - **cite your paper** and acknowledge the dataset in any output;
> - sign any **data-use agreement** you require.
>
> Could you let me know how to obtain the dataset and any terms involved? Thank
> you very much for sharing this resource with the community.
>
> Kind regards,
> **[Your name]**
> **[Your affiliation / "independent researcher"]**
> **[Your email]**

## Step 3 — when (if) you receive it
Tell me, and I'll write the Kaggle-ready training pipeline (segmentation + T-stage,
patient-level split, honest held-out metrics) — exactly like the polyp detector.
Until then, MRI stays clinician-entered (T/N/M → exact AJCC stage), which is honest
and already working.

> Reality check: gated requests like this often take days–weeks and are sometimes
> declined. That's why we're not blocking on it — we proceed with the UI work now.
