#!/usr/bin/env bash
# ColonAI — Download additional reliable colonoscopy datasets
# Run this once.  Total size ~38 GB.  Estimated download time: 2–6 hours
# depending on your connection.
#
# After download, run:   python3 scripts/retrain_unified_v2.py
#
# Datasets included (all CC-BY or research-use):
#   1. HyperKvasir Labelled    (already in repo) — 10,662 images, 23 classes
#   2. Kvasir-SEG              — 1,000 polyp images with pixel-level masks
#   3. Kvasir-Capsule          — 4,741 capsule-endoscopy frames, 14 classes
#   4. CVC-ClinicDB            (already in repo) — 612 polyps with masks
#   5. CVC-ColonDB             — 380 polyps with masks (different patients)
#   6. ETIS-LaribPolypDB       — 196 polyps from a different scope (Pentax)
#   7. PolypGen                — 8,037 polyp images from 6 centres (multi-vendor!)
#   8. SUN-SEG                 — 158k video frames with frame-level masks
#   9. EndoSLAM (HD subset)    — high-resolution multi-vendor video
#  10. PICCOLO                 — Hospital Universitario Donostia, multi-vendor

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATA="$ROOT/data/raw"
mkdir -p "$DATA"

echo "ColonAI Dataset Downloader"
echo "=========================="
echo "Target directory: $DATA"
echo ""
echo "This will download ~38 GB.  Continue?  (y/N)"
read -r ANS
[ "$ANS" = "y" ] || [ "$ANS" = "Y" ] || { echo "Cancelled."; exit 0; }

# Helper to download with resume + retries
dl() {
    local url="$1"; local out="$2"
    if [ -f "$out" ]; then echo "  ✓ exists  $out"; return; fi
    echo "  → $url"
    curl -L -C - --retry 6 --retry-delay 10 -o "$out" "$url"
}

# ───────────────────────────────────────────────────────
# 2. Kvasir-SEG (polyp segmentation, 1,000 images)
# ───────────────────────────────────────────────────────
echo ""
echo "[2/10] Kvasir-SEG (46 MB)"
mkdir -p "$DATA/kvasir-seg"
dl "https://datasets.simula.no/downloads/kvasir-seg.zip" "$DATA/kvasir-seg.zip"
unzip -oq "$DATA/kvasir-seg.zip" -d "$DATA/kvasir-seg/"

# ───────────────────────────────────────────────────────
# 3. Kvasir-Capsule (capsule endoscopy, 4,741 images)
# ───────────────────────────────────────────────────────
echo ""
echo "[3/10] Kvasir-Capsule (3.5 GB — labelled subset only)"
mkdir -p "$DATA/kvasir-capsule"
dl "https://datasets.simula.no/downloads/kvasir-capsule/labelled-images.zip" \
   "$DATA/kvasir-capsule/labelled.zip"
unzip -oq "$DATA/kvasir-capsule/labelled.zip" -d "$DATA/kvasir-capsule/"

# ───────────────────────────────────────────────────────
# 5. CVC-ColonDB (380 polyps)
# ───────────────────────────────────────────────────────
echo ""
echo "[5/10] CVC-ColonDB"
mkdir -p "$DATA/cvc-colondb"
echo "  → Manual download required from:"
echo "    https://polyp.grand-challenge.org/CVCColonDB/"
echo "  Place the extracted folder at: $DATA/cvc-colondb/"

# ───────────────────────────────────────────────────────
# 6. ETIS-LaribPolypDB (196 polyps, Pentax scope)
# ───────────────────────────────────────────────────────
echo ""
echo "[6/10] ETIS-LaribPolypDB"
mkdir -p "$DATA/etis-larib"
echo "  → Manual download required from:"
echo "    https://polyp.grand-challenge.org/ETISLarib/"
echo "  Place the extracted folder at: $DATA/etis-larib/"

# ───────────────────────────────────────────────────────
# 7. PolypGen (8,037 multi-centre polyps)
# ───────────────────────────────────────────────────────
echo ""
echo "[7/10] PolypGen (1.2 GB, multi-vendor)"
mkdir -p "$DATA/polypgen"
echo "  → Apply for access at:"
echo "    https://github.com/sharibox/PolypGen-Benchmark"
echo "  Place the data at: $DATA/polypgen/"

# ───────────────────────────────────────────────────────
# 8. SUN-SEG (158k video frames)
# ───────────────────────────────────────────────────────
echo ""
echo "[8/10] SUN-SEG (28 GB video frames)"
mkdir -p "$DATA/sun-seg"
echo "  → Request access at:"
echo "    http://amed8k.sundatabase.org/"
echo "  Place the data at: $DATA/sun-seg/"

# ───────────────────────────────────────────────────────
# 9. EndoSLAM HD subset (multi-vendor video)
# ───────────────────────────────────────────────────────
echo ""
echo "[9/10] EndoSLAM HD subset (2.5 GB)"
mkdir -p "$DATA/endoslam"
echo "  → Download from:"
echo "    https://github.com/CapsuleEndoscope/EndoSLAM"

# ───────────────────────────────────────────────────────
# 10. PICCOLO (Hospital Donostia)
# ───────────────────────────────────────────────────────
echo ""
echo "[10/10] PICCOLO"
mkdir -p "$DATA/piccolo"
echo "  → Request access at:"
echo "    https://www.biobancovasco.org/en/Sample-and-data-catalog/Databases/PD178-PICCOLO-EN.html"

echo ""
echo "=========================="
echo "  Auto-downloadable datasets complete."
echo "  Manual datasets:  follow the URLs above to request access."
echo "  Then run:  python3 scripts/retrain_unified_v2.py"
echo "=========================="
