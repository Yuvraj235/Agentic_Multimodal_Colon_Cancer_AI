"""Real, exact colorectal cancer staging from clinician-entered findings.

This is the honest path to accurate staging: a stage CANNOT be read off a
colonoscopy surface image (it needs tumour depth from histology + spread from
CT/MRI). But once a clinician has those findings (the T, N, M categories), the
final stage is a FIXED rulebook — and this module applies it exactly.

Reference: AJCC Cancer Staging Manual, 8th edition (colon & rectum). The output
is deterministic — same T/N/M always give the same stage.

This module is pure logic (no model, no I/O) so it is unit-testable and exact.
"""
from __future__ import annotations
from typing import Dict

# Categories offered to the clinician (the ones that change the stage grouping).
T_OPTIONS = ["Tis", "T1", "T2", "T3", "T4a", "T4b"]
N_OPTIONS = ["N0", "N1", "N1c", "N2a", "N2b"]   # N1 covers N1a/N1b
M_OPTIONS = ["M0", "M1a", "M1b", "M1c"]

T_HELP = {
    "Tis": "Carcinoma in situ — confined to the innermost lining",
    "T1":  "Invades the submucosa",
    "T2":  "Invades the muscularis propria",
    "T3":  "Through muscularis into pericolorectal tissue",
    "T4a": "Penetrates the surface of the visceral peritoneum",
    "T4b": "Directly invades / adheres to other organs or structures",
}
N_HELP = {
    "N0":  "No regional lymph-node spread",
    "N1":  "1–3 regional nodes involved",
    "N1c": "Tumour deposits, no positive nodes",
    "N2a": "4–6 regional nodes involved",
    "N2b": "7+ regional nodes involved",
}
M_HELP = {
    "M0":  "No distant spread",
    "M1a": "Spread to one distant site/organ",
    "M1b": "Spread to two or more distant sites/organs",
    "M1c": "Spread to the peritoneal surface",
}

_BUCKET = {  # coarse bucket for the rest of the app's UI
    "0": "Stage 0", "I": "Stage I",
    "IIA": "Stage II", "IIB": "Stage II", "IIC": "Stage II",
    "IIIA": "Stage III", "IIIB": "Stage III", "IIIC": "Stage III",
    "IVA": "Stage IV", "IVB": "Stage IV", "IVC": "Stage IV",
}


def ajcc_colorectal_stage(t: str, n: str, m: str) -> Dict:
    """Return the exact AJCC 8th-edition stage for the given T, N, M.

    Returns {stage_group, bucket, exact, rationale}. `stage_group` is e.g. 'IIIB';
    `bucket` is the coarse 'Stage III'; `exact` is True when the inputs map to a
    defined group, False if the combination is unusual and needs expert review.
    """
    t = (t or "").strip(); n = (n or "").strip(); m = (m or "").strip()

    def out(group, why):
        return {"stage_group": group, "bucket": _BUCKET.get(group, "Needs review"),
                "exact": group not in ("?",), "rationale": why}

    # M1 dominates everything → Stage IV
    if m == "M1a":
        return out("IVA", "Distant spread to one site (M1a) → Stage IVA, regardless of T/N.")
    if m == "M1b":
        return out("IVB", "Distant spread to ≥2 sites (M1b) → Stage IVB, regardless of T/N.")
    if m == "M1c":
        return out("IVC", "Peritoneal spread (M1c) → Stage IVC, regardless of T/N.")
    if m != "M0":
        return out("?", "Unrecognised M category.")

    # ── M0 ────────────────────────────────────────────────────────────────────
    if n == "N0":
        if t == "Tis":  return out("0",   "Tis N0 M0 → Stage 0 (in situ).")
        if t in ("T1", "T2"): return out("I",  f"{t} N0 M0 → Stage I.")
        if t == "T3":   return out("IIA", "T3 N0 M0 → Stage IIA.")
        if t == "T4a":  return out("IIB", "T4a N0 M0 → Stage IIB.")
        if t == "T4b":  return out("IIC", "T4b N0 M0 → Stage IIC.")
        return out("?", "Unrecognised T category.")

    # Node-positive, M0
    if n in ("N1", "N1c"):
        if t in ("T1", "T2"): return out("IIIA", f"{t} {n} M0 → Stage IIIA.")
        if t in ("T3", "T4a"): return out("IIIB", f"{t} {n} M0 → Stage IIIB.")
        if t == "T4b":         return out("IIIC", f"{t} {n} M0 → Stage IIIC.")
    elif n == "N2a":
        if t == "T1":            return out("IIIA", "T1 N2a M0 → Stage IIIA.")
        if t in ("T2", "T3"):    return out("IIIB", f"{t} N2a M0 → Stage IIIB.")
        if t in ("T4a", "T4b"):  return out("IIIC", f"{t} N2a M0 → Stage IIIC.")
    elif n == "N2b":
        if t in ("T1", "T2"):           return out("IIIB", f"{t} N2b M0 → Stage IIIB.")
        if t in ("T3", "T4a", "T4b"):   return out("IIIC", f"{t} N2b M0 → Stage IIIC.")

    return out("?", "Unusual T/N/M combination — please review against the AJCC manual.")


if __name__ == "__main__":
    # Self-test against known AJCC 8th-edition groupings.
    cases = {
        ("Tis", "N0", "M0"): "0",
        ("T1", "N0", "M0"): "I",
        ("T2", "N0", "M0"): "I",
        ("T3", "N0", "M0"): "IIA",
        ("T4a", "N0", "M0"): "IIB",
        ("T4b", "N0", "M0"): "IIC",
        ("T1", "N1", "M0"): "IIIA",
        ("T2", "N1c", "M0"): "IIIA",
        ("T3", "N1", "M0"): "IIIB",
        ("T4a", "N1", "M0"): "IIIB",
        ("T4b", "N1", "M0"): "IIIC",
        ("T1", "N2a", "M0"): "IIIA",
        ("T2", "N2a", "M0"): "IIIB",
        ("T3", "N2a", "M0"): "IIIB",
        ("T4a", "N2a", "M0"): "IIIC",
        ("T1", "N2b", "M0"): "IIIB",
        ("T2", "N2b", "M0"): "IIIB",
        ("T3", "N2b", "M0"): "IIIC",
        ("T4b", "N2b", "M0"): "IIIC",
        ("T2", "N0", "M1a"): "IVA",
        ("T3", "N1", "M1b"): "IVB",
        ("T4b", "N2b", "M1c"): "IVC",
    }
    ok = True
    for (t, n, m), expect in cases.items():
        got = ajcc_colorectal_stage(t, n, m)["stage_group"]
        flag = "" if got == expect else "  <-- MISMATCH"
        if got != expect:
            ok = False
        print(f"  {t:4s} {n:4s} {m:4s} -> {got:4s} (expect {expect}){flag}")
    print("ALL OK" if ok else "FAILURES PRESENT")
