#!/bin/bash
# Launch the E-Commerce AI Shopping Assistant
set -e

cd "$(dirname "$0")"
source ../.venv/bin/activate

if [ -z "$GROQ_API_KEY" ]; then
    echo "❌  GROQ_API_KEY not set. Usage:"
    echo "    export GROQ_API_KEY='YOUR_GROQ_KEY'"
    echo "    ./launch.sh"
    exit 1
fi

# macOS: prevents segfaults with PyTorch/sentence-transformers threading
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "🚀 Starting assistant..."
python3 app.py
