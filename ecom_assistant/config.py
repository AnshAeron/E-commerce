"""
Central configuration for the e-commerce CRS assistant.
"""
import os
from pathlib import Path

# Project root
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Input ──────────────────────────────────────────────────────────────────────
CSV_PATH = ROOT.parent / "BigBasket Products.csv"
CSV_SAMPLE_SIZE = None             # None => read full file

# ── Generated data files ───────────────────────────────────────────────────────
CATALOG_PATH      = DATA_DIR / "product_catalog.json"       # {pid: rich product metadata}
SEQUENCES_PATH    = DATA_DIR / "purchase_sequences.json"    # {session_id: [pid, ...]}
DESCRIPTIONS_PATH = DATA_DIR / "product_descriptions.json"  # {pid: "text description"}
FAISS_INDEX_PATH  = DATA_DIR / "faiss.index"
FAISS_ID_MAP_PATH = DATA_DIR / "faiss_id_map.json"          # {faiss_int_idx: pid}
SASREC_MODEL_PATH = DATA_DIR / "sasrec_model.pt"
ITEM_INDEX_PATH   = DATA_DIR / "item_id2index.json"         # {pid: int_index}

# ── Groq API (free tier — chat/LLM only) ──────────────────────────────────────
# Create free account: https://console.groq.com  → API Keys → Create key
# Set via:  export GROQ_API_KEY="YOUR_GROQ_KEY"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CHAT_MODEL   = "llama-3.3-70b-versatile"   # free on Groq

# ── Local embeddings (sentence-transformers — fully offline, free) ─────────────
# Downloaded once (~90MB) to ~/.cache/huggingface/
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast on CPU/MPS

# ── SASRec hyper-parameters ────────────────────────────────────────────────────
SASREC_MAX_SEQ_LEN = 50
SASREC_EMBED_DIM   = 64
SASREC_NUM_HEADS   = 2
SASREC_NUM_LAYERS  = 2
SASREC_DROPOUT     = 0.2
SASREC_EPOCHS      = 30
SASREC_BATCH_SIZE  = 256
SASREC_LR          = 1e-3
SASREC_EARLY_STOP_PATIENCE = 4
SASREC_EARLY_STOP_MIN_DELTA = 0.005

# ── Synthetic sequence generation (BigBasket) ─────────────────────────────────
SYNTHETIC_NUM_SESSIONS = 1500
SYNTHETIC_MIN_SEQ_LEN  = 3
SYNTHETIC_MAX_SEQ_LEN  = 7

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE = 20    # FAISS fetches this many candidates
TOP_K_SHOW     = 5     # final products shown to the user

# ── Conversation ───────────────────────────────────────────────────────────────
MAX_ELICIT_TURNS = 2   # max clarifying questions per conversation
