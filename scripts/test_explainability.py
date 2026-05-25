"""ColonAI — self-contained smoke tests for the explainability pipeline.

Tests the 5 new modules end-to-end with synthetic inputs:
  • src/app/decision_trace.py        — build_trace, detect_disagreements
  • src/app/modality_attribution.py  — fused_attribution, silencing_attribution
  • src/app/counterfactual.py        — silence_modalities, stability_score
  • src/app/prototype_retrieval.py   — build_bank, retrieve_similar
  • src/app/explanation_engine.py    — narrative, clinician_report

Run:
    python3 scripts/test_explainability.py

Exit code is 0 if every assertion passes, 1 otherwise. Used by the
daily auto-bug-check (.github/workflows/auto-bug-check.yml).
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

# Make 'src' importable when run from the project root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


# ──────────────────────────────────────────────────────────────────────
def _ok(name: str):  print(f"  ✓ {name}")
def _fail(name: str, exc): print(f"  ✗ {name}: {exc}"); sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
def test_decision_trace():
    from src.app.decision_trace import (
        build_trace, detect_disagreements, render_narrative, trace_to_json
    )

    # 1. Empty trace
    trace = build_trace(final_verdict="polyps", final_confidence=0.5)
    assert len(trace) == 1
    assert trace[0]["agent"] == "orchestrator"
    _ok("build_trace with minimal inputs")

    # 2. Full trace with all agents
    trace = build_trace(
        smart_pred={
            "predicted_class": "polyps", "confidence": 0.87,
            "uncertainty": 0.22, "mutual_info": 0.05,
            "is_hedged": False, "differential": [("polyps", 0.87)],
        },
        atypicality_finding={"override": True, "reasons": ["deep ulceration"],
                             "label": "Atypical lesion", "confidence": 0.9},
        polyp_typing_finding={"paris": {"class": "0-Is", "confidence": 0.7}},
        fusion_finding={"finding": "Fused: polyps", "confidence": 0.87,
                        "modality_weights": {"image": 0.6, "text": 0.1, "tabular": 0.3}},
        xai_finding={"finding": "low uncertainty", "confidence": 0.78,
                     "epistemic_uncertainty": 0.22},
        safety_finding={"passed": True, "confidence": 1.0, "evidence": []},
        clinical_finding={"recommendation": "EMR", "confidence": 0.85,
                          "evidence": []},
        tcga_stage_finding={"stage": "II", "confidence": 0.45, "evidence": {}},
        llm_refined=True,
        final_verdict="polyps", final_confidence=0.87,
    )
    assert len(trace) >= 8, f"expected ≥8 trace steps, got {len(trace)}"
    agents = [s["agent"] for s in trace]
    for must in ("smart_inference", "atypicality", "polyp_typing", "fusion",
                 "xai", "safety", "clinical", "tcga_stage", "llm",
                 "orchestrator"):
        assert must in agents, f"missing agent {must} in trace"
    _ok("build_trace with all agents produces full trace")

    # 3. Disagreement detection
    d = detect_disagreements(trace)
    # Atypicality fired override → should NOT be unanimous
    assert not d["unanimous"], "atypicality override should register as disagreement"
    assert "atypicality" in d["disagreed"]
    _ok("detect_disagreements catches atypicality override")

    # 4. Narrative rendering
    text = render_narrative(trace)
    assert "FINAL VERDICT: polyps" in text
    assert "Atypicality" in text
    _ok("render_narrative produces a non-empty story")

    # 5. JSON round-trip
    js = trace_to_json(trace)
    import json
    parsed = json.loads(js)
    assert len(parsed) == len(trace)
    _ok("trace_to_json round-trips through json.loads")


# ──────────────────────────────────────────────────────────────────────
def test_modality_attribution():
    from src.app.modality_attribution import (
        silencing_attribution, fused_attribution,
        render_text_bar, attribution_summary,
    )

    # 1. fused_attribution falls back when no silencing result
    attr = fused_attribution(image_confidence=0.6, text_confidence=0.1,
                              tabular_confidence=0.3)
    assert attr["interpretable"]
    assert "image" in attr["contributions"]
    assert sum(attr["contributions"].values()) > 99.9
    _ok("fused_attribution sums to 100%")

    # 2. silencing_attribution with mock predict_fn
    def fake_predict(image=None, text=None, tabular=None):
        # Baseline polyps=0.9 — silencing image drops it the most
        if image is None: return {"polyps": 0.4, "uc": 0.6}
        if text is None or text == "": return {"polyps": 0.85, "uc": 0.15}
        if tabular is None: return {"polyps": 0.80, "uc": 0.20}
        return {"polyps": 0.90, "uc": 0.10}

    sil = silencing_attribution(
        predict_fn=fake_predict, image="img", text="text", tabular={"x": 1},
        silenced_image=None, silenced_tabular=None,
        predicted_class="polyps",
    )
    assert sil["interpretable"]
    assert sil["contributions"]["image"] > sil["contributions"]["text"]
    _ok("silencing_attribution ranks image > text when image drop is bigger")

    # 3. Rendering
    bar = render_text_bar(attr["contributions"])
    assert "%" in bar and "█" in bar
    _ok("render_text_bar produces bar with %s and █s")
    summary = attribution_summary(attr)
    assert "drove" in summary
    _ok("attribution_summary produces sentence")


# ──────────────────────────────────────────────────────────────────────
def test_counterfactual():
    from src.app.counterfactual import (
        silence_modalities, perturb_tabular, stability_score,
    )

    def fake_predict(image=None, text=None, tabular=None):
        base = {"polyps": 0.7, "uc": 0.2, "barretts": 0.1}
        # Silencing image flips to uc
        if image is None: return {"polyps": 0.3, "uc": 0.5, "barretts": 0.2}
        return base

    # 1. Silence modalities
    cfs = silence_modalities(
        predict_fn=fake_predict, image="img", text="text", tabular={"x": 1},
        silenced_image=None, silenced_tabular=None,
        original_class="polyps", original_confidence=0.7,
    )
    assert any(cf["flipped"] for cf in cfs), "silencing image should flip"
    _ok("silence_modalities flips when image silenced")

    # 2. Stability score
    stab = stability_score(cfs)
    assert "stability" in stab and 0.0 <= stab["stability"] <= 1.0
    assert "verdict" in stab
    _ok("stability_score returns valid stability + verdict")

    # 3. Perturb tabular (no flip expected with fake_predict)
    cfs2 = perturb_tabular(
        predict_fn=fake_predict, image="img", text="text",
        tabular={"age": 50, "bmi": 25, "pack_years": 0,
                 "gender_male": 1, "site_rectum": 0,
                 "alcohol_history": 0, "family_hx_cancer": 0,
                 "cigs_per_day": 0},
        original_class="polyps", original_confidence=0.7,
    )
    assert isinstance(cfs2, list)
    _ok("perturb_tabular runs without crashing")


# ──────────────────────────────────────────────────────────────────────
def test_prototype_retrieval():
    from src.app.prototype_retrieval import (
        build_bank, load_bank, is_bank_available,
        retrieve_similar, neighbour_concordance,
    )

    tmpdir = Path(tempfile.mkdtemp())
    bank_path = tmpdir / "bank.npz"
    meta_path = tmpdir / "meta.json"

    np.random.seed(0)
    N, D = 30, 32
    embs = np.random.randn(N, D).astype(np.float32)
    labels = (["polyps"] * 10) + (["uc-mild"] * 10) + (["barretts"] * 10)
    paths = [f"fake/{i:03d}.jpg" for i in range(N)]

    # 1. Build bank
    out = build_bank(embeddings=embs, labels=labels, paths=paths,
                     out_npz=bank_path, out_meta=meta_path)
    assert out.exists()
    _ok("build_bank writes a non-empty bank file")

    # 2. Load bank
    bank = load_bank(bank_path)
    assert bank is not None
    assert bank["embeddings"].shape == (N, D)
    _ok("load_bank reads the bank back into memory")

    # 3. is_bank_available
    assert is_bank_available(bank_path)
    assert not is_bank_available(tmpdir / "nonexistent.npz")
    _ok("is_bank_available correctly reports presence/absence")

    # 4. Retrieve similar — query close to sample 5 (polyps)
    q = embs[5] + np.random.randn(D) * 0.01
    out = retrieve_similar(q, k=5, bank_path=bank_path, diversify=False)
    assert len(out) == 5
    assert out[0]["label"] == "polyps", "closest match should be polyps"
    assert out[0]["similarity"] > 0.95
    _ok("retrieve_similar finds the closest training case")

    # 5. Class filter
    out2 = retrieve_similar(q, k=3, bank_path=bank_path,
                             filter_class="uc-mild")
    assert all(n["label"] == "uc-mild" for n in out2)
    _ok("retrieve_similar honours filter_class")

    # 6. Neighbour concordance
    conc = neighbour_concordance(out, "polyps")
    assert conc["concordance"] > 0.0
    assert conc["k"] == 5
    _ok("neighbour_concordance reports correct k and ratio")


# ──────────────────────────────────────────────────────────────────────
def test_explanation_engine():
    from src.app.decision_trace import build_trace, detect_disagreements
    from src.app.modality_attribution import fused_attribution
    from src.app.explanation_engine import (
        narrative, clinician_report, report_to_markdown,
    )

    attr = fused_attribution(image_confidence=0.6, text_confidence=0.1,
                              tabular_confidence=0.3)
    trace = build_trace(
        smart_pred={"predicted_class": "polyps", "confidence": 0.87,
                    "uncertainty": 0.22, "is_hedged": False},
        fusion_finding={"finding": "Fused: polyps", "confidence": 0.87,
                        "modality_weights": {"image": 0.6}},
        final_verdict="polyps", final_confidence=0.87,
    )
    d = detect_disagreements(trace)

    # 1. Narrative
    text = narrative(final_class="polyps", final_confidence=0.87,
                      attribution=attr, disagreement=d)
    assert "Polyps" in text
    assert 10 < len(text.split()) < 200, f"narrative wrong length: {len(text.split())} words"
    _ok("narrative produces a non-trivial paragraph")

    # 2. Clinician report
    rep = clinician_report(
        final_class="polyps", final_confidence=0.87,
        trace=trace, attribution=attr,
    )
    assert "sections" in rep
    assert len(rep["sections"]) >= 3
    section_kinds = [s["kind"] for s in rep["sections"]]
    assert "headline" in section_kinds
    assert "caveats" in section_kinds
    _ok("clinician_report produces multi-section dossier with headline + caveats")

    # 3. Markdown export
    md = report_to_markdown(rep)
    assert md.startswith("# ColonAI explanation report")
    assert "Polyps" in md
    assert "##" in md
    _ok("report_to_markdown produces valid markdown")


# ──────────────────────────────────────────────────────────────────────
def main():
    print("ColonAI explainability self-test")
    print("─" * 50)
    print("1) decision_trace.py")
    try: test_decision_trace()
    except Exception as e: _fail("decision_trace", e)
    print("2) modality_attribution.py")
    try: test_modality_attribution()
    except Exception as e: _fail("modality_attribution", e)
    print("3) counterfactual.py")
    try: test_counterfactual()
    except Exception as e: _fail("counterfactual", e)
    print("4) prototype_retrieval.py")
    try: test_prototype_retrieval()
    except Exception as e: _fail("prototype_retrieval", e)
    print("5) explanation_engine.py")
    try: test_explanation_engine()
    except Exception as e: _fail("explanation_engine", e)
    print("─" * 50)
    print("All explainability self-tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
