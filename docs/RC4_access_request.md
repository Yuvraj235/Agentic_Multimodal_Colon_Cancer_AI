# RC4 rectal-MRI dataset — data-access request (ready to send)

The RC4 dataset has no public download — access is "on reasonable request to the
authors." This is the request to send them. **Two placeholders to fill in first.**

## Step 1 — who to send it to (+ how to get the address)
**Authors** (Expert Systems with Applications, Elsevier, 2024 · id `S0957417424016014`):
**Yu Qiu, Haotian Lu, Jie Mei, Sixu Bao, Jing Xu.**
Corresponding author is most likely **Yu Qiu** (lead + funding) or **Jing Xu**
(supervision / project administration).

Two easy ways to reach them (I couldn't fetch the email — ScienceDirect is paywalled,
and no address is exposed online; do NOT guess one):
1. **ResearchGate (no email needed, easiest):** search the paper title or "Yu Qiu
   rectal cancer segmentation," open an author profile, click **Message**, paste the
   note below.
2. **The paper's first page** (via a library / the authors' university page): the
   corresponding author has a ✉ email, and the **"Data availability"** statement names
   the exact contact + any terms. Use that address.

## Step 2 — email to send

> **Subject:** Request for access to the RC4 rectal-cancer MRI dataset (research use)
>
> Dear Dr. **Qiu** (or Dr. **Xu**),
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
