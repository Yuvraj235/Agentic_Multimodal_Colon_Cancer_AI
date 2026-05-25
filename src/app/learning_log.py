"""ColonAI — privacy-preserving learning log.

Goal
────
Let the model keep improving from real cases without ever storing
patient-identifying data.

What we DO store (per case, append-only):
   • timestamp + opaque case UUID (random, not derived from anything)
   • SHA-256 hash of the image bytes (irreversible)
   • the model's 256-dimensional fused embedding (a float vector;
     enough to fine-tune downstream, NOT enough to reconstruct the image)
   • the predicted pathology class + confidence + uncertainty
   • optional one-click clinician feedback:
       - "correct"  → reinforces the predicted label
       - "wrong"    → with a corrected label (4 alternatives)
       - "unsure"   → no signal but recorded for audit volume
   • a coarse safety-policy flag (show / abstain / reject)

What we DO NOT store (ever):
   • patient name / DOB / address / contact / NHS or hospital ID
   • the raw image bytes
   • the raw symptom text
   • patient age / BMI / smoking / family-history fields
       (those live in the browser session only, used at inference, then dropped)
   • IP address, browser user-agent

Files
─────
   outputs/learning_log/<YYYY-MM>.jsonl     ← append-only learning records
   outputs/learning_log/embeddings_<YYYY-MM>.npz  ← compressed embedding tensors
                                                    (separate so the small JSONL
                                                     stays human-readable)

File perms are chmod 0o600 — owner-only read/write.
"""
from __future__ import annotations
import os, json, uuid, hashlib, time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np


LOG_DIR = Path("outputs/learning_log")


def _month_key() -> str:
    """Year-month bucket so files don't grow indefinitely."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _ensure_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try: os.chmod(LOG_DIR, 0o700)
    except Exception: pass


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Public API — call these from app.py / scripts/serve_api.py
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LearningRecord:
    case_uuid:        str
    iso_ts:           str
    image_sha256:     str
    predicted_class:  str
    confidence:       float
    uncertainty:      float
    safety_action:    str               # "show" | "abstain" | "reject"
    embedding_idx:    Optional[int]     # index into the .npz file (None if no embedding)
    feedback:         Optional[str] = None   # "correct" | "wrong" | "unsure"
    correct_label:    Optional[str] = None   # only set when feedback == "wrong"
    extras:           Dict             = field(default_factory=dict)


def record_case(*,
                image_bytes:     Optional[bytes],
                fused_embedding: Optional[np.ndarray],
                predicted_class: str,
                confidence:      float,
                uncertainty:     float,
                safety_action:   str,
                extras:          Optional[Dict] = None) -> str:
    """Append a new case to today's learning log. Returns the case_uuid
    (random opaque ID — pass it back when you call record_feedback later).

    `image_bytes` and `fused_embedding` are both optional:
       • Without image_bytes, image_sha256 is "".
       • Without fused_embedding, the case is logged but cannot be used
         for retraining (which is fine — abstain / reject cases don't
         need embeddings).
    """
    _ensure_dir()
    case_uuid = uuid.uuid4().hex
    img_hash  = _hash_bytes(image_bytes) if image_bytes else ""

    emb_idx: Optional[int] = None
    if fused_embedding is not None:
        emb_idx = _append_embedding(case_uuid, fused_embedding)

    rec = LearningRecord(
        case_uuid       = case_uuid,
        iso_ts          = datetime.now(timezone.utc).isoformat(),
        image_sha256    = img_hash,
        predicted_class = predicted_class,
        confidence      = float(confidence),
        uncertainty     = float(uncertainty),
        safety_action   = safety_action,
        embedding_idx   = emb_idx,
        extras          = extras or {},
    )
    _append_jsonl(rec)
    return case_uuid


def record_feedback(case_uuid: str,
                    feedback: str,
                    correct_label: Optional[str] = None) -> bool:
    """Append a feedback row tied to a previously-recorded case_uuid.

    feedback ∈ {"correct", "wrong", "unsure"}
    correct_label: only set when feedback == "wrong"

    Returns True on success. We append a separate row rather than mutating
    the original record so the log stays append-only (auditable).
    """
    if feedback not in {"correct", "wrong", "unsure"}:
        raise ValueError(f"feedback must be correct|wrong|unsure, got {feedback!r}")
    _ensure_dir()
    fb = {
        "case_uuid":     case_uuid,
        "iso_ts":        datetime.now(timezone.utc).isoformat(),
        "feedback":      feedback,
        "correct_label": correct_label,
        "_type":         "feedback",
    }
    path = LOG_DIR / f"{_month_key()}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(fb) + "\n")
    try: os.chmod(path, 0o600)
    except Exception: pass
    return True


def load_training_set() -> Dict:
    """Reduce every (case + its latest feedback) into a labelled tensor set.

    Used by scripts/retrain_from_feedback.py. Returns:
        {"X": np.ndarray (N, 256),
         "y": list[str],            # the corrected label, or predicted if "correct"
         "weights": np.ndarray (N,) # confidence-weighted importance}

    Only cases with feedback="correct" OR feedback="wrong"+correct_label are
    kept. "unsure" and unlabelled cases are ignored.
    """
    _ensure_dir()
    cases:   Dict[str, dict] = {}
    feedback: Dict[str, dict] = {}
    embs:    Dict[str, np.ndarray] = {}

    for jsonl in sorted(LOG_DIR.glob("*.jsonl")):
        for line in jsonl.read_text().splitlines():
            if not line.strip(): continue
            try:
                r = json.loads(line)
            except Exception: continue
            if r.get("_type") == "feedback":
                feedback[r["case_uuid"]] = r
            else:
                cases[r["case_uuid"]] = r

    # Load embeddings (lazily — only the months we have cases for)
    months_needed = {datetime.fromisoformat(c["iso_ts"]).strftime("%Y-%m")
                     for c in cases.values()}
    for m in months_needed:
        p = LOG_DIR / f"embeddings_{m}.npz"
        if p.exists():
            data = np.load(p)
            for k in data.files: embs[k] = data[k]

    X, y, w = [], [], []
    for cu, c in cases.items():
        fb = feedback.get(cu)
        if fb is None: continue
        if fb["feedback"] == "correct":
            label = c["predicted_class"]
        elif fb["feedback"] == "wrong" and fb.get("correct_label"):
            label = fb["correct_label"]
        else:
            continue
        emb = embs.get(cu)
        if emb is None or c.get("embedding_idx") is None: continue
        X.append(emb)
        y.append(label)
        # Higher confidence + correct → higher weight; wrong → constant 1.0
        w.append(1.5 if fb["feedback"] == "wrong"
                 else max(0.5, float(c.get("confidence", 0.5))))
    return {"X": (np.vstack(X) if X else np.zeros((0, 256))),
            "y": y,
            "weights": np.asarray(w, dtype=np.float32)}


def stats() -> Dict:
    """Quick summary for the UI: total cases, with-feedback, per-class counts."""
    _ensure_dir()
    total = with_feedback = 0
    per_class: Dict[str, int] = {}
    for jsonl in sorted(LOG_DIR.glob("*.jsonl")):
        for line in jsonl.read_text().splitlines():
            if not line.strip(): continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("_type") == "feedback":
                with_feedback += 1
            else:
                total += 1
                cls = r.get("predicted_class", "?")
                per_class[cls] = per_class.get(cls, 0) + 1
    return {"total_cases": total,
            "with_feedback": with_feedback,
            "feedback_rate": (with_feedback / total) if total else 0.0,
            "per_class": per_class}


# ─────────────────────────────────────────────────────────────────────────────
# Internal — embedding storage uses .npz so per-record .jsonl stays small
# ─────────────────────────────────────────────────────────────────────────────
def _append_embedding(case_uuid: str, emb: np.ndarray) -> int:
    """Append the embedding to this month's .npz. Returns position index."""
    if emb.ndim > 1:
        emb = emb.reshape(-1)
    path = LOG_DIR / f"embeddings_{_month_key()}.npz"
    if path.exists():
        data = dict(np.load(path))
    else:
        data = {}
    data[case_uuid] = emb.astype(np.float32)
    # Atomic-ish write — numpy.savez_compressed auto-appends .npz to the path
    # if missing, so we use a filename that already ends in .npz to avoid surprise.
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(path)
    try: os.chmod(path, 0o600)
    except Exception: pass
    return len(data) - 1


def _append_jsonl(rec: LearningRecord) -> None:
    path = LOG_DIR / f"{_month_key()}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")
    try: os.chmod(path, 0o600)
    except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─ learning_log self-test ─")
    fake_emb = np.random.randn(256).astype(np.float32)
    fake_img = b"\x89PNG\x0d\x0a\x1a\x0a" + b"\x00" * 100   # PNG magic + filler
    cu = record_case(
        image_bytes=fake_img, fused_embedding=fake_emb,
        predicted_class="polyps", confidence=0.87, uncertainty=0.12,
        safety_action="show")
    print(f"  ✓ recorded case {cu[:12]}…")
    record_feedback(cu, "correct")
    print(f"  ✓ feedback ‘correct’ written")
    cu2 = record_case(
        image_bytes=b"different", fused_embedding=np.random.randn(256).astype(np.float32),
        predicted_class="uc-mild", confidence=0.62, uncertainty=0.25,
        safety_action="show")
    record_feedback(cu2, "wrong", correct_label="uc-moderate-sev")
    print(f"  ✓ feedback ‘wrong → uc-moderate-sev’ written")
    print("\n─ training set ─")
    s = load_training_set()
    print(f"  X.shape  = {s['X'].shape}")
    print(f"  labels   = {s['y']}")
    print(f"  weights  = {s['weights']}")
    print("\n─ stats ─")
    for k, v in stats().items(): print(f"  {k}: {v}")
