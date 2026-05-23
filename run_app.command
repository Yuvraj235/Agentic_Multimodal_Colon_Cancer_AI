#!/usr/bin/env bash
# ColonAI · double-clickable launcher (macOS)
# -----------------------------------------------------------------------
# Double-click this file in Finder. It will:
#   1. cd into the project directory (works from any location)
#   2. verify Python and streamlit are available
#   3. install missing requirements once on first run
#   4. find a free port (8501, 8502, …)
#   5. start streamlit and open the browser
# Press Ctrl-C in the terminal window to stop.
# -----------------------------------------------------------------------

set -e

# Resolve the directory of this script — works whether double-clicked or run from CLI.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Pretty banner -----------------------------------------------------------
clear
printf "\033[1;34m"
printf "════════════════════════════════════════════════════════════\n"
printf "  ColonAI · Agentic Multimodal Colon Cancer Screening\n"
printf "  Starting up — this window must stay open while you demo.\n"
printf "════════════════════════════════════════════════════════════\n"
printf "\033[0m\n"

# Pick a Python interpreter -----------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    printf "\033[1;31m✗ Python is not installed.\033[0m\n"
    printf "Install Python 3.10+ from https://www.python.org/downloads/ then re-run.\n"
    read -p "Press enter to close…" _
    exit 1
fi
echo "→ Python:  $($PY --version 2>&1)"

# Make sure streamlit (and other deps) are installed ----------------------
if ! $PY -c "import streamlit" >/dev/null 2>&1; then
    printf "\033[1;33m  Streamlit not found — installing dependencies (one-time)…\033[0m\n"
    $PY -m pip install --quiet --upgrade pip
    $PY -m pip install --quiet -r requirements.txt
fi

# Pick the first free port between 8501 and 8520 --------------------------
PORT=8501
while lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do
    PORT=$((PORT + 1))
    if [ $PORT -gt 8520 ]; then
        echo "✗ Could not find a free port between 8501 and 8520."
        read -p "Press enter to close…" _
        exit 1
    fi
done
URL="http://localhost:${PORT}"
echo "→ Port:    ${PORT}"
echo "→ URL:     ${URL}"
echo

# Open the URL in the default browser after a short delay -----------------
( sleep 3 && open "${URL}" >/dev/null 2>&1 ) &

printf "\033[1;32m  Launching Streamlit — keep this window open.\033[0m\n"
printf "\033[2m  Press Ctrl-C to stop the server.\033[0m\n"
echo

# Hand off to streamlit ---------------------------------------------------
exec $PY -m streamlit run app.py \
    --server.port "${PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false
