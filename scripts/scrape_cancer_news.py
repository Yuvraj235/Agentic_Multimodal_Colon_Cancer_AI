"""ColonAI — daily cancer-news scraper.

Pulls the latest cancer-related research and clinical news from free RSS
feeds. Focuses on colon / colorectal / polyp / colonoscopy topics. No
API keys, no payment, no scraping rate-limit dodging — just public RSS.

Sources (all public, no auth):
  • PubMed search RSS — multiple colon-cancer-related queries
  • NCI (National Cancer Institute) cancer news
  • ASCO Daily News — ASCO Annual Meeting + journal alerts
  • Cancer Research UK news feed
  • Medical Xpress oncology RSS

Output: outputs/cancer_news.json — consumed by the Streamlit news page.

Run:
    python3 scripts/scrape_cancer_news.py

For automation: see .github/workflows/scrape-news.yml (runs daily).
"""
from __future__ import annotations
import json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import quote_plus

import feedparser   # robust RSS/Atom parser, stdlib not good enough for PubMed CDATA

OUT = Path("outputs/cancer_news.json")
USER_AGENT = "ColonAI-NewsScraper/1.0 (+https://huggingface.co/spaces/Yuvraj2319/colonai)"
TIMEOUT = 20  # seconds per feed


# ─────────────────────────────────────────────────────────────────────────────
# Feed sources
# ─────────────────────────────────────────────────────────────────────────────
def _pubmed_rss(query: str, max_items: int = 15) -> str:
    """Build a PubMed RSS URL from a free-text query (most relevant + recent)."""
    return (
        "https://pubmed.ncbi.nlm.nih.gov/rss/search.xml?"
        f"term={quote_plus(query)}"
        f"&filter=simsearch1.fha&filter=years.2025-2026"   # last 2 years
        f"&size={max_items}&sort=date"
    )


# Topic → (feed URL, priority weight).  Higher priority shows first.
# Note: PubMed deprecated their RSS endpoint in 2024 — use eUtils API later
# if academic abstracts are needed. For now we rely on consumer-facing
# RSS feeds which are stable, well-formed, and updated daily.
FEEDS = [
    ("ScienceDaily — colorectal cancer",  "https://www.sciencedaily.com/rss/health_medicine/colon_cancer.xml",  1.0),
    ("Medical Xpress — gastroenterology", "https://medicalxpress.com/rss-feed/gastroenterology-news/",          0.95),
    ("ScienceDaily — cancer",             "https://www.sciencedaily.com/rss/health_medicine/cancer.xml",         0.7),
    ("Cancer Research UK news",           "https://news.cancerresearchuk.org/feed/",                            0.7),
    ("ScienceDaily — immune system",      "https://www.sciencedaily.com/rss/health_medicine/immune_system.xml",  0.55),
    ("ScienceDaily — pharmacology",       "https://www.sciencedaily.com/rss/health_medicine/pharmacology.xml",   0.55),
]

# Keywords we boost OR drop
PRIORITY_KEYWORDS = [
    "colon", "colorectal", "polyp", "colonoscop", "rectal",
    "adenoma", "ulcerative", "crohn", "bowel", "intestin", "ibd",
    "fecal", "stool",
]
NICE_TO_HAVE_KEYWORDS = [
    "immunotherapy", "checkpoint inhibitor", "ctdna", "biomarker",
    "early detection", "screening", "ai ", "artificial intelligence",
    "deep learning", "drug approval", "fda", "ema", "clinical trial",
    "phase iii", "phase 3",
]
DROP_KEYWORDS = [
    "veterinary", "pet ",
]


# ─────────────────────────────────────────────────────────────────────────────
# RSS / Atom minimal parser (stdlib only — no feedparser dep)
# ─────────────────────────────────────────────────────────────────────────────
def _strip_html(html: str) -> str:
    if not html: return ""
    html = (html.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"')
                .replace("&#39;", "'").replace("&nbsp;", " "))
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html).strip()


def _truncate(text: str, max_chars: int = 280) -> str:
    text = text.strip()
    if len(text) <= max_chars: return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _parse_feed(url: str, source_name: str) -> List[Dict]:
    """Parse any RSS / Atom feed via feedparser (handles CDATA, namespaces, etc)."""
    try:
        d = feedparser.parse(url, request_headers={"User-Agent": USER_AGENT})
    except Exception as e:
        print(f"  ! parse error for {source_name}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return []
    if getattr(d, "bozo", 0) and not d.entries:
        print(f"  ! feedparser bozo (no entries) for {source_name}: "
              f"{getattr(d, 'bozo_exception', '?')}", file=sys.stderr)
        return []
    items: List[Dict] = []
    for e in d.entries[:30]:
        title = (getattr(e, "title", "") or "").strip()
        link  = (getattr(e, "link",  "") or "").strip()
        desc  = (getattr(e, "summary", "") or
                 (getattr(e, "content", [{}])[0].get("value", "")
                  if getattr(e, "content", None) else "") or "")
        pub = (getattr(e, "published", "") or getattr(e, "updated", "") or "")
        items.append({
            "title":   title,
            "link":    link,
            "summary": _truncate(_strip_html(desc)),
            "date":    pub,
            "source":  source_name,
        })
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Ranking + filtering
# ─────────────────────────────────────────────────────────────────────────────
def _score(item: Dict, source_weight: float) -> float:
    text = (item["title"] + " " + item["summary"]).lower()
    if any(k in text for k in DROP_KEYWORDS):
        return -1
    s = source_weight
    s += sum(2.0 for k in PRIORITY_KEYWORDS if k in text)
    s += sum(0.5 for k in NICE_TO_HAVE_KEYWORDS if k in text)
    return s


def _categorise(item: Dict) -> str:
    text = (item["title"] + " " + item["summary"]).lower()
    if any(k in text for k in ("colon", "colorectal", "polyp", "colonoscop", "rectal",
                                "bowel", "intestin")):
        return "colorectal"
    if any(k in text for k in ("ulcerative", "crohn", "ibd")):
        return "ibd"
    if any(k in text for k in ("drug", "approval", "fda", "ema")):
        return "drug-news"
    if any(k in text for k in ("trial", "phase iii", "phase 3", "clinical trial")):
        return "clinical-trial"
    return "general-oncology"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Scraping {len(FEEDS)} feeds at {datetime.now(timezone.utc).isoformat()}")
    all_items: List[Dict] = []
    feed_status: List[Dict] = []

    for name, url, weight in FEEDS:
        print(f"  · {name:38s} {url[:60]}")
        t0 = time.time()
        items = _parse_feed(url, name)
        if not items:
            feed_status.append({"name": name, "ok": False, "n": 0})
            continue
        # Score + drop negatives + filter low-relevance
        scored = []
        for it in items:
            s = _score(it, weight)
            if s < 0: continue   # dropped
            it["score"]    = round(s, 2)
            it["category"] = _categorise(it)
            scored.append(it)
        all_items.extend(scored)
        elapsed = time.time() - t0
        print(f"    → {len(items)} raw, {len(scored)} kept ({elapsed:.1f}s)")
        feed_status.append({"name": name, "ok": True, "n": len(scored)})

    # Dedup by title (case-insensitive)
    seen = set(); dedup = []
    for it in all_items:
        key = re.sub(r"\W+", "", it["title"].lower())[:80]
        if not key or key in seen: continue
        seen.add(key); dedup.append(it)

    # Sort by score desc, then by source diversity
    dedup.sort(key=lambda x: -x["score"])

    # Final cap — keep top 40 most relevant
    final = dedup[:40]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "n_items":         len(final),
        "feed_status":     feed_status,
        "items":           final,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n✓ wrote {OUT} — {len(final)} stories")

    # Quick per-category breakdown
    cats: Dict[str, int] = {}
    for it in final: cats[it["category"]] = cats.get(it["category"], 0) + 1
    print("  by category:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {c:18s} {n}")


if __name__ == "__main__":
    main()
