"""Deploy the current ColonAI build to the Hugging Face Space (one atomic commit).

The Space (Yuvraj2319/colonai, docker SDK) was found to be stale: old checkpoint
(no UC fix), old seg head, and MISSING the CADx / histology / OOD / view-quality
heads entirely. The git-push path errors at the protocol level, so this uses the
authenticated huggingface_hub API instead.

Everything goes in a SINGLE create_commit so the Space rebuilds once and is never
left half-synced. Build files (Dockerfile/requirements/.streamlit) are verified
identical and skipped. Run: python3 scripts/deploy_to_space.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

SPACE = "Yuvraj2319/colonai"
MODEL = "Yuvraj2319/colonai-v2"          # public model repo the Space downloads from
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# ── Architecture: the 600 MB checkpoint lives in the MODEL repo (the Space pulls
# it at runtime via COLONAI_CHECKPOINT_HF_REPO, already configured). The Space
# itself (1 GB cap) holds ONLY code + the small specialist heads. The old
# checkpoint was wrongly baked into the Space — we delete it to free the budget.
CHECKPOINT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"   # → MODEL repo
SPACE_BAKED_CHECKPOINT = CHECKPOINT                                        # delete from Space

# Code: app entrypoint + README (carries the Space YAML header) + full src/ tree
# + scripts/ (reproducibility). requirements.txt IS included because it changed
# (added segmentation-models-pytorch + albumentations for the CT specialist) —
# the docker Space rebuilds and pip-installs it on the next commit.
CODE_SINGLES = ["app.py", "README.md", "requirements.txt"]
CODE_TREES = ["src", "scripts"]

# Small weights + thresholds the specialists load at runtime (all << 1 GB budget).
WEIGHTS = [
    "outputs/unified_multimodal_v2/seg_head.pth",                 # multi-centre seg
    "outputs/unified_multimodal_v2/cadx_head.pth",               # NEW characterization
    "outputs/unified_multimodal_v2/histology_head.pth",          # NEW histology
    "outputs/unified_multimodal_v2/ood_head.pth",                # real OOD
    "outputs/unified_multimodal_v2/view_quality_head.pth",       # view-quality gate
    "outputs/unified_multimodal_v2/temperature.json",
    "outputs/unified_multimodal_v2/per_class_thresholds.json",
    "outputs/unified_multimodal_v2/view_quality_threshold.json",
    "outputs/unified_multimodal_v2/ood_threshold_real.json",
    "outputs/unified_multimodal_v2/ood_threshold.json",
    # honest validation artifacts (small, shareable)
    "outputs/unified_multimodal_v2/clinical_validation_report.json",
    "outputs/unified_multimodal_v2/clinical_validation_report.md",
    "outputs/unified_multimodal_v2/figures/validation_scorecard.png",
    # CT rectal-tumour specialist (CARE): only the tiny metrics JSON goes in the
    # Space. The 93 MB weights live in the MODEL repo (colonai-v2) and are pulled
    # at runtime via COLONAI_CT_SEG_HF_REPO/FILE — keeps the Space under 1 GB.
    "outputs/unified_multimodal_v2/care_ct_seg_metrics.json",
]


def _iter_tree(d: Path):
    for f in sorted(d.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
            yield f


def main():
    api = HfApi()
    print("auth:", api.whoami().get("name"))
    skip_ckpt = "--skip-checkpoint" in sys.argv

    # ── Part 1: uc-fix checkpoint → MODEL repo (overwrite old) ──────────────
    # The 600 MB checkpoint is unchanged across most code deploys, so allow
    # skipping the slow re-upload with --skip-checkpoint.
    if skip_ckpt:
        print("\n[1/2] --skip-checkpoint: leaving the model-repo checkpoint untouched.")
    else:
        ck = ROOT / CHECKPOINT
        print(f"\n[1/2] Uploading uc-fix checkpoint ({ck.stat().st_size/1e6:.0f} MB) → {MODEL}")
        print("      (600 MB LFS upload — this is the slow part) …")
        api.upload_file(
            path_or_fileobj=str(ck), path_in_repo="best_model.pth",
            repo_id=MODEL, repo_type="model",
            commit_message="Promote uc-fix checkpoint (UC-severity recall fix)")
        print("      ✓ checkpoint live in model repo")

    # ── Part 2: lean Space — delete baked-in checkpoint, add code + heads ────
    # Only delete the baked-in checkpoint if it's still on the Space (older
    # deploys already removed it — an unconditional delete 404s).
    try:
        space_files = set(api.list_repo_files(SPACE, repo_type="space"))
    except Exception:
        space_files = set()
    ops, total = [], 0
    if SPACE_BAKED_CHECKPOINT in space_files:
        ops.append(CommitOperationDelete(path_in_repo=SPACE_BAKED_CHECKPOINT))
        print(f"  will delete stale baked checkpoint: {SPACE_BAKED_CHECKPOINT}")
    seen = set()

    def add(local: Path, repo_path: str):
        nonlocal total
        if not local.exists():
            print(f"  ! skip missing {local}"); return
        if repo_path in seen:
            return
        seen.add(repo_path)
        ops.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local)))
        total += local.stat().st_size

    for s in CODE_SINGLES:
        add(ROOT / s, s)
    for t in CODE_TREES:
        for f in _iter_tree(ROOT / t):
            add(f, str(f.relative_to(ROOT)))
    for w in WEIGHTS:
        add(ROOT / w, w)

    n_add = sum(1 for o in ops if isinstance(o, CommitOperationAdd))
    print(f"\n[2/2] Space commit: {n_add} files ({total/1e6:.0f} MB) → {SPACE}")
    big = sorted([o for o in ops if isinstance(o, CommitOperationAdd)],
                 key=lambda o: -os.path.getsize(o.path_or_fileobj))[:5]
    for o in big:
        print(f"   {os.path.getsize(o.path_or_fileobj)/1e6:7.1f} MB  {o.path_in_repo}")
    info = api.create_commit(
        repo_id=SPACE, repo_type="space", operations=ops,
        commit_message="Sync Space to honest build: download uc-fix checkpoint from model repo; "
                        "add multi-centre seg + CADx + histology + OOD + view-quality heads + "
                        "validation scorecard; remove baked-in checkpoint (1 GB budget)")
    print("\n✓ Space committed:", getattr(info, "commit_url", info))
    print("The Space will rebuild (docker) and download the uc-fix checkpoint on boot.")


if __name__ == "__main__":
    main()
