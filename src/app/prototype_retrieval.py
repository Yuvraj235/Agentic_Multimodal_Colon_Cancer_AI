"""ColonAI — Prototype Retrieval.

"This case looks like training cases X, Y, Z."  Case-based explanation
is what clinicians actually use ("I've seen this before, it was an
adenoma…"), so we provide the same affordance to the model.

How it works
------------
1. **Build a prototype bank** once, by running the encoder over the
   training set and saving the fused embeddings + their labels.
2. **At inference time**, embed the new case and find the K nearest
   prototypes by cosine similarity in embedding space.
3. **Return**: file paths (or thumbnails), labels, similarities.

The UI shows a small gallery of "most similar training cases" alongside
the model's prediction, so the user can sanity-check the decision against
real, ground-truth-labelled examples.

The bank is a single .npz file in outputs/unified_multimodal_v2/:
    embeddings.npy    (N, D) float16
    labels.npy        (N,)   str
    paths.npy         (N,)   str

If the bank doesn't exist, all retrieval functions return empty results
gracefully (so the UI just hides the section instead of crashing).

PRIVACY: the bank stores public HyperKvasir / CVC training images, not
patient data. The learning_log module is separate and never feeds the
prototype bank (different consent model).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np


# ──────────────────────────────────────────────────────────────────────
#  Default paths
# ──────────────────────────────────────────────────────────────────────
DEFAULT_BANK = Path("outputs/unified_multimodal_v2/prototype_bank.npz")
DEFAULT_META = Path("outputs/unified_multimodal_v2/prototype_meta.json")


# ──────────────────────────────────────────────────────────────────────
#  Build bank (called once after training, offline)
# ──────────────────────────────────────────────────────────────────────
def build_bank(
    *,
    embeddings: np.ndarray,
    labels: List[str],
    paths: List[str],
    out_npz: Path = DEFAULT_BANK,
    out_meta: Path = DEFAULT_META,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a prototype bank for later retrieval.

    Parameters
    ----------
    embeddings   (N, D) float array — fused embedding per sample
    labels       length-N list of class labels
    paths        length-N list of file paths (image only, relative)
    out_npz      where to save the bank
    out_meta     where to save the JSON metadata
    """
    embeddings = np.asarray(embeddings, dtype=np.float16)
    labels_arr = np.asarray(labels, dtype=object)
    paths_arr = np.asarray(paths, dtype=object)
    if len(labels_arr) != len(embeddings):
        raise ValueError(f"len(embeddings)={len(embeddings)} != len(labels)={len(labels_arr)}")
    if len(paths_arr) != len(embeddings):
        raise ValueError(f"len(embeddings)={len(embeddings)} != len(paths)={len(paths_arr)}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, embeddings=embeddings, labels=labels_arr, paths=paths_arr)

    meta: Dict[str, Any] = {
        "n_samples":     int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "classes":       sorted(set(map(str, labels))),
        "schema":        "embeddings: (N,D) float16, labels: (N,) str, paths: (N,) str",
    }
    if extra_meta: meta.update(extra_meta)
    out_meta.write_text(json.dumps(meta, indent=2))
    return out_npz


# ──────────────────────────────────────────────────────────────────────
#  Load bank (cached)
# ──────────────────────────────────────────────────────────────────────
_BANK_CACHE: Dict[str, Any] = {}

def load_bank(path: Path = DEFAULT_BANK) -> Optional[Dict[str, np.ndarray]]:
    """Load a previously-saved bank into memory (cached)."""
    key = str(path)
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        bank = {
            "embeddings": np.asarray(data["embeddings"], dtype=np.float32),
            "labels":     np.asarray(data["labels"], dtype=object),
            "paths":      np.asarray(data["paths"], dtype=object),
        }
        # L2-normalise embeddings once for cosine retrieval
        norms = np.linalg.norm(bank["embeddings"], axis=1, keepdims=True) + 1e-9
        bank["embeddings_norm"] = bank["embeddings"] / norms
        _BANK_CACHE[key] = bank
        return bank
    except Exception:
        return None


def is_bank_available(path: Path = DEFAULT_BANK) -> bool:
    return load_bank(path) is not None


# ──────────────────────────────────────────────────────────────────────
#  Retrieval
# ──────────────────────────────────────────────────────────────────────
def retrieve_similar(
    query_embedding: np.ndarray,
    *,
    k: int = 5,
    bank_path: Path = DEFAULT_BANK,
    filter_class: Optional[str] = None,
    diversify: bool = True,
) -> List[Dict[str, Any]]:
    """Find K most similar training prototypes for this query.

    Parameters
    ----------
    query_embedding   (D,) float array — fused embedding of the new case
    k                 how many neighbours to return
    filter_class      if set, only return prototypes with this label
    diversify         if True, do simple MMR-style diversification so we
                      don't return five near-duplicate frames

    Returns a list of dicts:
        {label, path, similarity, rank}
    """
    bank = load_bank(bank_path)
    if bank is None or len(bank["embeddings"]) == 0:
        return []

    q = np.asarray(query_embedding, dtype=np.float32).flatten()
    if q.shape[0] != bank["embeddings_norm"].shape[1]:
        return []
    q = q / (np.linalg.norm(q) + 1e-9)

    sims = bank["embeddings_norm"] @ q       # (N,)
    labels = bank["labels"]
    paths = bank["paths"]

    # Optional class filter
    mask = np.ones(len(sims), dtype=bool)
    if filter_class is not None:
        mask &= (labels.astype(str) == str(filter_class))
    cand_idx = np.where(mask)[0]
    if len(cand_idx) == 0:
        return []

    # Top-N candidate pool
    pool_size = min(len(cand_idx), max(k * 5, 20))
    cand_sims = sims[cand_idx]
    top_local = np.argpartition(-cand_sims, min(pool_size - 1, len(cand_sims) - 1))[:pool_size]
    pool = cand_idx[top_local]
    pool = pool[np.argsort(-sims[pool])]

    if not diversify or len(pool) <= k:
        chosen = pool[:k].tolist()
    else:
        # Greedy MMR with λ=0.7 (favour similarity, mild diversity penalty)
        chosen: List[int] = [int(pool[0])]
        remaining = pool[1:].tolist()
        lam = 0.7
        while len(chosen) < k and remaining:
            best, best_score = None, -1e9
            chosen_embs = bank["embeddings_norm"][chosen]
            for idx in remaining:
                sim_q = sims[idx]
                sim_chosen = float(np.max(chosen_embs @ bank["embeddings_norm"][idx]))
                score = lam * sim_q - (1 - lam) * sim_chosen
                if score > best_score:
                    best_score, best = score, idx
            if best is None: break
            chosen.append(int(best))
            remaining.remove(best)

    out: List[Dict[str, Any]] = []
    for rank, idx in enumerate(chosen, start=1):
        out.append({
            "rank":       rank,
            "label":      str(labels[idx]),
            "path":       str(paths[idx]),
            "similarity": float(sims[idx]),
        })
    return out


# ──────────────────────────────────────────────────────────────────────
#  Concordance check
# ──────────────────────────────────────────────────────────────────────
def neighbour_concordance(
    neighbours: List[Dict[str, Any]],
    predicted_class: str,
) -> Dict[str, Any]:
    """How many of the K nearest neighbours agree with the model?

    Useful as an independent sanity check:
       "Model says polyps. 5/5 nearest training images are also polyps." ✓
       "Model says polyps. 2/5 nearest training images are polyps."  ⚠ — review.
    """
    if not neighbours:
        return {"concordance": 0.0, "majority": None, "k": 0}
    labels = [n["label"] for n in neighbours]
    matches = sum(1 for L in labels if L == predicted_class)
    from collections import Counter
    counts = Counter(labels)
    majority, maj_count = counts.most_common(1)[0]
    return {
        "concordance":   matches / len(labels),
        "majority":      majority,
        "majority_pct":  maj_count / len(labels),
        "k":             len(labels),
        "all_labels":    labels,
        "agrees":        (majority == predicted_class),
    }


# ──────────────────────────────────────────────────────────────────────
#  Embedding extraction helper (called from app.py)
# ──────────────────────────────────────────────────────────────────────
def extract_fused_embedding(model, *, image, text=None, tabular=None,
                            attention_mask=None) -> Optional[np.ndarray]:
    """Run a forward pass and return the fused [CLS]-equivalent embedding.

    Tries several common attribute names on the model so we don't have
    to hard-code the architecture. Returns None if no embedding could be
    extracted.
    """
    try:
        import torch
    except ImportError:
        return None
    model.eval()
    try:
        with torch.no_grad():
            out = model(
                images=image,
                text_input_ids=text,
                text_attention_mask=attention_mask,
                tabular=tabular,
                return_embedding=True,        # most builds support this kwarg
            )
            if isinstance(out, dict):
                emb = out.get("embedding", out.get("fused", out.get("cls", None)))
                if emb is not None:
                    return emb.detach().cpu().float().numpy().flatten()
            if isinstance(out, torch.Tensor):
                return out.detach().cpu().float().numpy().flatten()
    except TypeError:
        # Model doesn't accept return_embedding kwarg — try forward, then
        # look for cached embedding on the model.
        try:
            with torch.no_grad():
                _ = model(images=image, text_input_ids=text,
                          text_attention_mask=attention_mask, tabular=tabular)
            for attr in ("last_fused", "_last_fused", "fused_embedding",
                         "_cached_embedding", "_cls_embedding"):
                if hasattr(model, attr):
                    val = getattr(model, attr)
                    if val is not None:
                        return val.detach().cpu().float().numpy().flatten()
        except Exception:
            pass
    except Exception:
        pass
    return None
