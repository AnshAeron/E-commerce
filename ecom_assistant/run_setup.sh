#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_setup.sh  —  One-shot setup for the E-Commerce CRS Assistant
#
# Requirements:
#   1. Python venv activated  (source ../.venv/bin/activate)
#   2. Groq API key for the chat UI  (export GROQ_API_KEY='YOUR_GROQ_KEY')
#      Get free key: https://console.groq.com
#      NOTE: Steps 1-3 (CSV, descriptions, FAISS) need NO API key.
#
# Usage:
#   chmod +x run_setup.sh
#   ./run_setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "════════════════════════════════════════════════"
echo "  E-Commerce CRS — Setup Pipeline"
echo "════════════════════════════════════════════════"
echo ""

# No API key check needed — embeddings now run locally via sentence-transformers
# (Groq key is only needed when launching app.py)

echo "▶  Step 1/4  Process BigBasket CSV → rich catalog + synthetic purchase sequences"
python3 setup/01_process_bigbasket.py
echo ""

echo "▶  Step 2/4  Build retrieval descriptions from rich metadata"
python3 setup/02_generate_descriptions.py
echo ""

echo "▶  Step 3/4  Build FAISS vector store (local sentence-transformers — no API key)"
python3 setup/03_build_vector_store.py
echo ""

echo "▶  Step 4/4  Train SASRec on purchase sequences"
python3 setup/04_train_sasrec.py
echo ""

echo "════════════════════════════════════════════════"
echo "  ✅  Setup complete!"
echo ""
echo "  Launch the assistant:"
echo "    python3 app.py"
echo "════════════════════════════════════════════════"
