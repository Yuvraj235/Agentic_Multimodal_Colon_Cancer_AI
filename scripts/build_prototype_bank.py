"""ColonAI — Build a prototype bank for case-based explanations.

Run once after every checkpoint update. Loops over the validation set
(NOT the train set, to avoid memorisation noise — val has been split
held-out and labelled), runs the model to extract a fused embedding,
and saves:

    outputs/unified_multimodal_v2/prototype_bank.npz
    outputs/unified_multimodal_v2/prototype_meta.json

At inference time, src/app/prototype_retrieval.retrieve_similar() loads
this bank and finds the K nearest training cases for any new image, so
the UI can show "this case looks like training cases X, Y, Z" alongside
the model's prediction.

Privacy
-------
Only PUBLIC training images (HyperKvasir + CVC-ClinicDB) are stored.
This bank is NOT linked to the privacy-preserving continual-learning
log (src/app/learning_log.py), which uses a separate, irreversible
embedding-only store and never feeds the prototype bank.

Run:
    python3 scripts/build_prototype_bank.py
    python3 scripts/build_prototype_bank.py --limit 500   # quick smoke
    python3 scripts/build_prototype_bank.py --split test  # build from test
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = use all available samples; otherwise cap")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/unified_multimodal_v2/prototype_bank.npz"))
    parser.add_argument("--ckpt", type=Path,
                        default=Path("outputs/unified_multimodal_v2/best.pt"))
    args = parser.parse_args()

    print(f"ColonAI — prototype bank builder")
    print(f"  split        : {args.split}")
    print(f"  limit        : {args.limit or 'all'}")
    print(f"  out          : {args.out}")
    print(f"  checkpoint   : {args.ckpt}")

    # ── 1. Load model
    try:
        import torch
        from src.models.unified_transformer import UnifiedMultiModalTransformer
        from src.data.multimodal_dataset import (MultiModalDataset, CLASS_NAMES_5)
    except Exception as exc:
        print(f"✗ Could not import model/dataset: {exc}")
        sys.exit(1)

    if not args.ckpt.exists():
        print(f"✗ Checkpoint not found at {args.ckpt}.")
        print(f"  Either train first, or pass --ckpt /path/to/best.pt")
        sys.exit(1)

    device = ("cuda" if torch.cuda.is_available() else
              "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
              else "cpu")
    print(f"  device       : {device}")

    print("  loading model …")
    model = UnifiedMultiModalTransformer(num_classes=len(CLASS_NAMES_5))
    try:
        from src.app.security import safe_torch_load
        state = safe_torch_load(args.ckpt, map_location=device)
    except Exception:
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print("  model loaded.")

    # ── 2. Load dataset split
    print(f"  loading {args.split} dataset …")
    try:
        ds = MultiModalDataset(split=args.split)
    except TypeError:
        # Older signature
        ds = MultiModalDataset()
    n_total = len(ds)
    print(f"  {n_total} samples in split.")

    indices = list(range(n_total))
    if args.limit > 0:
        indices = indices[:args.limit]

    # ── 3. Forward pass, extract embeddings
    print(f"  encoding {len(indices)} samples …")
    embeddings: List[np.ndarray] = []
    labels:     List[str] = []
    paths:      List[str] = []
    bad = 0

    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                sample = ds[idx]
                # Try common dataset return formats
                if isinstance(sample, dict):
                    img       = sample.get("image")
                    input_ids = sample.get("input_ids")
                    attn_mask = sample.get("attention_mask")
                    tabular   = sample.get("tabular")
                    label     = sample.get("label", sample.get("pathology"))
                    path      = sample.get("path", sample.get("image_path", ""))
                elif isinstance(sample, (tuple, list)):
                    # (image, input_ids, attention_mask, tabular, label)
                    img, input_ids, attn_mask, tabular = sample[:4]
                    label = sample[4] if len(sample) > 4 else "?"
                    path  = sample[5] if len(sample) > 5 else ""
                else:
                    continue

                # Move + batch
                def _b(t): return t.unsqueeze(0).to(device) if t is not None else None
                out = model(image=_b(img), input_ids=_b(input_ids),
                            attention_mask=_b(attn_mask), tabular=_b(tabular))

                # Prefer fused_embedding, fall back to penultimate
                emb = None
                if isinstance(out, dict):
                    for k in ("fused_embedding", "embedding", "fused", "penultimate"):
                        if k in out and out[k] is not None:
                            emb = out[k].detach().cpu().float().numpy().flatten()
                            break
                if emb is None and hasattr(model, "_last_fused"):
                    emb = model._last_fused.detach().cpu().float().numpy().flatten()
                if emb is None:
                    bad += 1
                    continue

                # Resolve label name
                lbl_name = (CLASS_NAMES_5[int(label)] if isinstance(label, (int, np.integer))
                            and int(label) < len(CLASS_NAMES_5) else str(label))

                embeddings.append(emb)
                labels.append(lbl_name)
                paths.append(str(path))

                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(indices)} ({bad} skipped)")
            except Exception as exc:
                bad += 1
                if bad < 5:
                    print(f"    skip idx={idx}: {type(exc).__name__}: {exc}")
                continue

    if not embeddings:
        print(f"✗ No embeddings extracted. Bank not built.")
        sys.exit(1)

    arr = np.stack(embeddings).astype(np.float16)
    print(f"  ✓ {len(arr)} embeddings extracted (dim={arr.shape[1]}, "
          f"{bad} skipped)")

    # ── 4. Save bank
    from src.app.prototype_retrieval import build_bank
    out = build_bank(
        embeddings=arr,
        labels=labels,
        paths=paths,
        out_npz=args.out,
        out_meta=args.out.with_suffix(".meta.json"),
        extra_meta={
            "built_from_split": args.split,
            "n_skipped":        bad,
            "checkpoint":       str(args.ckpt),
            "device":           device,
        },
    )
    print(f"\n✓ Prototype bank saved → {out}")
    print(f"✓ Metadata saved        → {out.with_suffix('.meta.json')}")

    # Per-class distribution
    from collections import Counter
    counts = Counter(labels)
    print(f"\nClass distribution:")
    for c, n in counts.most_common():
        print(f"  {c:<20}  {n:>4}")


if __name__ == "__main__":
    main()
