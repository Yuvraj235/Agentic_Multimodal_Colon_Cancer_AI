"""Guideline-grounded knowledge base (smart win #3).

Turns ColonAI's already-curated, guideline-aligned content (the clinical
recommendation rules + chatbot FAQ) into a small, CITED, retrievable knowledge
base. The report and chatbot answer FROM this — every statement carries its source
guideline — instead of free-form LLM text. Nothing here is invented: statements are
reused from the existing codebase content and attributed to the guideline body each
was already aligned with (NICE NG12, BSG, ESGE, USPSTF, NHS BCSP).

Retrieval is transparent keyword-overlap (no external deps, no embeddings).
"""
from __future__ import annotations
from typing import List, Dict
import re

# Each entry: id, topic, statement (plain English), source (citation), keywords.
GUIDELINES: List[Dict] = [
    {"id": "screening_age", "topic": "When to start screening",
     "statement": "Average-risk adults should begin colorectal cancer screening at age 45 and continue to 75 (individualised 76-85).",
     "source": "USPSTF 2021", "keywords": ["screening", "age", "start", "begin", "when", "fit", "colonoscopy", "screen"]},
    {"id": "red_flags", "topic": "Urgent referral red flags",
     "statement": "Rectal bleeding, unexplained weight loss, a persistent change in bowel habit (>3-6 weeks), or iron-deficiency anaemia warrant urgent assessment (suspected-cancer / 2-week-wait pathway).",
     "source": "NICE NG12", "keywords": ["bleeding", "blood", "weight", "loss", "bowel", "habit", "anaemia", "anemia", "red", "flag", "urgent", "refer", "2-week", "symptom"]},
    {"id": "ida", "topic": "Iron-deficiency anaemia",
     "statement": "Iron-deficiency anaemia in any man, or in a post-menopausal woman, is a colorectal-cancer red flag until proven otherwise — bidirectional endoscopy is recommended.",
     "source": "NICE NG12", "keywords": ["anaemia", "anemia", "iron", "tired", "fatigue", "pale", "deficiency"]},
    {"id": "fit", "topic": "FIT stool test",
     "statement": "A faecal immunochemical test (FIT) is recommended to triage symptomatic patients and as the primary screening test in national bowel-screening programmes.",
     "source": "NICE DG56 / NHS BCSP", "keywords": ["fit", "stool", "faecal", "fecal", "poo", "test", "occult", "screening"]},
    {"id": "surv_lowrisk", "topic": "Post-polypectomy surveillance (low risk)",
     "statement": "1-2 small (<10 mm) low-risk adenomas: return to routine screening (no early surveillance needed in most cases).",
     "source": "BSG/ESGE post-polypectomy surveillance", "keywords": ["polyp", "adenoma", "surveillance", "interval", "follow-up", "followup", "low", "risk", "after", "removed"]},
    {"id": "surv_highrisk", "topic": "Post-polypectomy surveillance (high risk)",
     "statement": "High-risk findings (e.g. >=2 premalignant polyps including >=1 advanced, or >=5 polyps, or a serrated polyposis pattern): 3-year surveillance colonoscopy; consider genetic counselling if Lynch syndrome is suspected.",
     "source": "BSG/ESGE post-polypectomy surveillance", "keywords": ["polyp", "adenoma", "surveillance", "high", "risk", "advanced", "serrated", "lynch", "3-year", "interval"]},
    {"id": "barretts", "topic": "Barrett's oesophagus surveillance",
     "statement": "Non-dysplastic Barrett's oesophagus: surveillance endoscopy every 3-5 years depending on segment length; main risk factors are long-standing reflux (>5 yr), male sex, age >50, central obesity and smoking.",
     "source": "BSG 2023 Barrett's guideline", "keywords": ["barrett", "barretts", "oesophagus", "esophagus", "reflux", "gord", "gerd", "heartburn", "surveillance"]},
    {"id": "uc", "topic": "Ulcerative colitis",
     "statement": "Ulcerative-colitis severity (endoscopic Mayo score) guides treatment; moderate-to-severe disease needs prompt gastroenterology review, and longstanding colitis requires surveillance colonoscopy for dysplasia.",
     "source": "BSG / ECCO IBD guidelines", "keywords": ["colitis", "uc", "ulcerative", "ibd", "inflammation", "inflammatory", "bowel", "disease", "mayo", "flare"]},
    {"id": "polyp_what", "topic": "What a polyp is / removal",
     "statement": "A polyp is a growth on the bowel lining; most are benign but some are precancerous, so they are removed at colonoscopy (polypectomy) and sent for histology.",
     "source": "BSG / general", "keywords": ["polyp", "what", "growth", "remove", "removal", "polypectomy", "benign", "precancer"]},
    {"id": "staging_real", "topic": "Cancer staging",
     "statement": "Colorectal cancer stage (AJCC TNM) is determined from tumour depth (biopsy/histology), nodes and metastasis (imaging) - it cannot be read from a colonoscopy surface image alone.",
     "source": "AJCC 8th edition / ICCR", "keywords": ["stage", "staging", "tnm", "ajcc", "cancer", "spread", "metastasis"]},
    {"id": "disclaimer", "topic": "Scope of AI advice",
     "statement": "AI screening output is decision support only; a qualified clinician must confirm every finding before any clinical decision.",
     "source": "ColonAI safety policy", "keywords": ["ai", "doctor", "diagnosis", "advice", "trust", "reliable", "accurate"]},
]

_STOP = {"the", "a", "an", "is", "are", "do", "i", "my", "me", "to", "of", "for", "and",
         "what", "how", "when", "should", "can", "you", "it", "this", "that", "have", "has",
         "in", "on", "with", "be", "or", "if", "about", "tell"}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOP and len(w) > 1}


def _kw_matches(q_tokens: set, kw: str) -> bool:
    """A keyword matches if a query token equals it, or shares a >=4-char prefix
    (handles variants like screened/screening/screen, anaemia/anaemic)."""
    for t in q_tokens:
        if t == kw:
            return True
        if min(len(t), len(kw)) >= 4 and (t.startswith(kw) or kw.startswith(t)):
            return True
    return False


def retrieve(query: str, k: int = 2, min_score: int = 1) -> List[Dict]:
    """Return up to k guideline entries best matching the query (keyword overlap).
    Each result includes a `score`. Empty list if nothing meets min_score."""
    q = _tokens(query)
    if not q:
        return []
    scored = []
    for g in GUIDELINES:
        score = sum(1 for kw in g["keywords"] if _kw_matches(q, kw))
        if score >= min_score:
            scored.append((score, g))
    scored.sort(key=lambda x: -x[0])
    return [{**g, "score": s} for s, g in scored[:k]]


def cited_answer(query: str) -> Dict | None:
    """Best single guideline-grounded answer with citation, or None."""
    hits = retrieve(query, k=1)
    if not hits:
        return None
    g = hits[0]
    return {"statement": g["statement"], "source": g["source"], "topic": g["topic"]}


def basis_for(pathology_class: str, symptoms_text: str = "") -> List[Dict]:
    """Guideline statements that back a recommendation for this case."""
    key = {
        "polyps": "polyp surveillance adenoma removed",
        "uc-mild": "ulcerative colitis inflammation",
        "uc-moderate-sev": "ulcerative colitis moderate severe",
        "barretts-esoph": "barretts oesophagus reflux surveillance",
        "therapeutic": "polyp surveillance after removed",
    }.get(pathology_class, "")
    q = (key + " " + (symptoms_text or "")).strip()
    return retrieve(q, k=2)


if __name__ == "__main__":
    for query in ["when should I get screened?", "blood in my stool and weight loss",
                  "how often for barrett's surveillance?", "what is a polyp",
                  "best diet for my dog"]:
        a = cited_answer(query)
        print(f"Q: {query}")
        print(f"  -> {a['statement'][:80]+'...' if a else '(no guideline match - safe fallback)'}"
              + (f"  [{a['source']}]" if a else ""))
    print("\nbasis_for('polyps'):", [g["source"] for g in basis_for("polyps")])
    assert cited_answer("when should I get screened?")["source"] == "USPSTF 2021"
    assert cited_answer("best diet for my dog") is None  # out-of-scope -> no fake answer
    print("OK")
