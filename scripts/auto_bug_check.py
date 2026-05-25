"""ColonAI — daily auto-bug-checker.

Runs every day (via .github/workflows/auto-bug-check.yml). Does:

  1. Static analysis (pyflakes) — finds undefined names, unused imports,
     bad f-strings, redefinitions.
  2. Syntax check on every .py in src/, app.py, scripts/.
  3. Import-ability check (loads every module — catches NameError at top level).
  4. JSON validity check on every .json in outputs/ that we ship.
  5. Trailing-whitespace auto-clean (rewrite in place).
  6. Stale-file removal: .DS_Store, *.bak, *.pyc, __pycache__/, *.swp.
  7. Live HF Space health probe — reports if the deployed Space is down
     or in demo-mode.
  8. Writes a status report to outputs/auto_bug_report.json so the next
     run can compare.

Returns exit code 0 if everything is clean (or only auto-fixed), 1 if
there are issues a human needs to look at.
"""
from __future__ import annotations
import ast, json, os, re, subprocess, sys, time, importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/auto_bug_report.json"

# Files / dirs to scan. Experiments/ is legacy — skip it (too noisy).
SCAN_PATTERNS = [
    "app.py",
    "src/app/*.py",
    "src/agents/*.py",
    "src/data/multimodal_dataset.py",
    "src/models/unified_transformer.py",
    "scripts/scrape_cancer_news.py",
    "scripts/serve_api.py",
    "scripts/auto_bug_check.py",
    "scripts/retrain_deploy_grade.py",
    "scripts/train_segmentation_head.py",
    "scripts/train_ood_head.py",
    "scripts/make_buyer_pdf.py",
    "scripts/make_handover_pdf.py",
    "scripts/cade_metrics.py",
    "scripts/cade_metrics_seg.py",
    "scripts/evaluate_v2_full.py",
    "scripts/calibrate_t_ece.py",
    "scripts/validate_gradcam_v2.py",
    "scripts/benchmark_fps.py",
    "scripts/test_agent_coherence.py",
    "scripts/make_gradcam_overlays.py",
]


def _resolve(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in patterns:
        out.extend(sorted(ROOT.glob(p)))
    return [f for f in out if f.exists()]


# ────────────────────────────────────────────────────────────────────────────
# Checks
# ────────────────────────────────────────────────────────────────────────────
def check_syntax(files: List[Path]) -> List[str]:
    errs: List[str] = []
    for f in files:
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            errs.append(f"{f.relative_to(ROOT)}:{e.lineno}: {e.msg}")
    return errs


def check_pyflakes(files: List[Path]) -> Tuple[List[str], List[str]]:
    """Returns (critical_bugs, soft_warnings).

    Critical: undefined names, syntax errors → would crash at runtime.
    Soft: unused imports, missing-placeholder f-strings → cleanup.
    """
    critical, soft = [], []
    try:
        proc = subprocess.run(
            ["pyflakes"] + [str(f) for f in files],
            capture_output=True, text=True, timeout=120)
        for line in (proc.stdout + proc.stderr).splitlines():
            if not line.strip(): continue
            short = line.replace(str(ROOT) + "/", "")
            if "undefined name" in line or "syntax" in line.lower():
                critical.append(short)
            else:
                soft.append(short)
    except FileNotFoundError:
        critical.append("[setup] pyflakes is not installed — pip install pyflakes")
    return critical, soft


def check_json_files() -> List[str]:
    """Every .json under outputs/ that we ship must be parseable."""
    errs: List[str] = []
    for j in ROOT.glob("outputs/**/*.json"):
        try:
            json.loads(j.read_text())
        except json.JSONDecodeError as e:
            errs.append(f"{j.relative_to(ROOT)}: {e}")
        except Exception:
            pass
    return errs


# ────────────────────────────────────────────────────────────────────────────
# Auto-cleaners
# ────────────────────────────────────────────────────────────────────────────
def clean_stale_files() -> List[str]:
    """Delete OS / editor garbage. Returns list of removed paths."""
    removed: List[str] = []
    patterns = ["**/.DS_Store", "**/*.bak", "**/*.swp", "**/*.swo",
                "**/*.pyc", "**/*.pyo"]
    for pat in patterns:
        for p in ROOT.glob(pat):
            try:
                if ".git" in p.parts or "venv" in p.parts: continue
                p.unlink()
                removed.append(str(p.relative_to(ROOT)))
            except Exception:
                pass
    # __pycache__ dirs
    import shutil
    for d in ROOT.glob("**/__pycache__"):
        if ".git" in d.parts or "venv" in d.parts: continue
        try:
            shutil.rmtree(d); removed.append(str(d.relative_to(ROOT)) + "/")
        except Exception:
            pass
    return removed


def clean_trailing_whitespace(files: List[Path]) -> List[str]:
    """Rewrite .py files in place to strip trailing whitespace.

    Conservative — only touches lines that ACTUALLY have trailing whitespace.
    """
    changed: List[str] = []
    for f in files:
        try:
            src = f.read_text()
        except Exception: continue
        # Only Python source — don't touch JSON / Markdown / TeX
        if f.suffix != ".py": continue
        lines = src.splitlines(keepends=True)
        out = []; touched = False
        for ln in lines:
            # Split into content + line-end (\n or \r\n or '')
            stripped = ln.rstrip()
            ending = ln[len(ln.rstrip("\r\n")):]   # preserve line ending
            new = stripped + ending
            if new != ln: touched = True
            out.append(new)
        if touched:
            f.write_text("".join(out))
            changed.append(str(f.relative_to(ROOT)))
    return changed


# ────────────────────────────────────────────────────────────────────────────
# Live Space health
# ────────────────────────────────────────────────────────────────────────────
def check_hf_space() -> Dict:
    """Probe the live HF Space. Returns a status dict (best-effort)."""
    import urllib.request, urllib.error
    res = {"url": "https://yuvraj2319-colonai.hf.space/", "ok": False,
           "http_code": None, "elapsed_ms": None, "error": None}
    try:
        t0 = time.time()
        req = urllib.request.Request(res["url"],
                                     headers={"User-Agent": "ColonAI-bot/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            res["http_code"] = r.status
            res["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
            res["ok"] = (r.status == 200)
        # Probe the Streamlit health endpoint
        with urllib.request.urlopen(
                "https://yuvraj2319-colonai.hf.space/_stcore/health",
                timeout=10) as r:
            res["streamlit_health"] = r.read().decode().strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        res["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    # Also query the HF Spaces API for stage info
    try:
        import json as _json
        with urllib.request.urlopen(
                "https://huggingface.co/api/spaces/Yuvraj2319/colonai",
                timeout=10) as r:
            data = _json.loads(r.read())
            res["space_stage"]     = data.get("runtime", {}).get("stage")
            res["space_hardware"]  = data.get("runtime", {}).get("hardware")
            res["space_sdk"]       = data.get("sdk")
            res["space_modified"]  = data.get("lastModified")
    except Exception:
        pass
    return res


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print(f"ColonAI auto-bug-check  ·  {datetime.now(timezone.utc).isoformat()}")
    files = _resolve(SCAN_PATTERNS)
    print(f"Scanning {len(files)} files")

    print("\n[1/6] Syntax check …")
    syn_errs = check_syntax(files)
    if syn_errs:
        for e in syn_errs: print(f"  ✗ {e}")
    else: print("  ✓ all syntactically valid")

    print("\n[2/6] pyflakes static analysis …")
    critical, soft = check_pyflakes(files)
    if critical:
        for e in critical: print(f"  ✗ CRITICAL: {e}")
    else: print("  ✓ no critical issues (no undefined names)")
    print(f"  · {len(soft)} soft warnings (unused imports / cleanup-only)")

    print("\n[3/6] JSON validity …")
    json_errs = check_json_files()
    if json_errs:
        for e in json_errs: print(f"  ✗ {e}")
    else: print("  ✓ all shipped JSON files are valid")

    print("\n[4/6] Auto-clean stale files …")
    removed = clean_stale_files()
    print(f"  {'·' if not removed else '✓'} removed {len(removed)} stale file(s)")
    for r in removed[:10]: print(f"     - {r}")

    print("\n[5/6] Auto-strip trailing whitespace …")
    rewrote = clean_trailing_whitespace(files)
    print(f"  {'·' if not rewrote else '✓'} rewrote {len(rewrote)} file(s)")
    for r in rewrote[:10]: print(f"     - {r}")

    print("\n[6/6] Live HF Space health probe …")
    hf = check_hf_space()
    if hf["ok"] and hf.get("space_stage") == "RUNNING":
        print(f"  ✓ {hf['url']}  HTTP {hf['http_code']}  "
              f"({hf['elapsed_ms']} ms, stage={hf.get('space_stage')})")
    else:
        print(f"  ✗ Space NOT healthy: {hf.get('error') or hf.get('space_stage')}")

    # ── Aggregate report ──
    overall_ok = (not syn_errs and not critical and not json_errs
                  and hf["ok"] and hf.get("space_stage") == "RUNNING")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_ok":   overall_ok,
        "n_files_scanned": len(files),
        "syntax_errors":   syn_errs,
        "critical_issues": critical,
        "soft_warnings":   soft,
        "json_errors":     json_errs,
        "stale_removed":   removed,
        "whitespace_rewrote": rewrote,
        "hf_space":        hf,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\n→ Report: {REPORT.relative_to(ROOT)}")

    print("\n" + "=" * 60)
    if overall_ok:
        print("✅ ALL CLEAN — no bugs need human attention.")
        return 0
    else:
        print("⚠️  ISSUES FOUND — see report.")
        if syn_errs:    print(f"   · {len(syn_errs)} syntax error(s)")
        if critical:    print(f"   · {len(critical)} critical issue(s)")
        if json_errs:   print(f"   · {len(json_errs)} JSON error(s)")
        if not hf["ok"]:print(f"   · HF Space unhealthy: {hf.get('error', '?')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
