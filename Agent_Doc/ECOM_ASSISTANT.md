# Session 2 — E-Commerce CRS Assistant (Flask + Groq + FAISS + SASRec)

> **New agent: read this entire file before touching anything.**  
> This documents everything built in Session 2. The original `AGENT_CONTEXT.md` covers the original Chinese UniMIND/U-NEED codebase — this file covers the NEW English CRS system built from scratch.

---

## TL;DR — Current State

**The system is FULLY BUILT and WORKING.**  
All 4 setup steps are complete. The Flask server runs at `http://localhost:7860`.  
The only command needed to restart it:

```bash
cd /Users/aryandubeytopg/Downloads/e-com/ecom_assistant
export GROQ_API_KEY="YOUR_GROQ_KEY"
./launch.sh
# → Open http://localhost:7860
```

---

## 1. What Was Built

A fully working **English e-commerce conversational recommender system** implemented from scratch at `/Users/aryandubeytopg/Downloads/e-com/ecom_assistant/`.

**Stack:**
| Component | Technology | Notes |
|---|---|---|
| Chat LLM | Groq API — `llama-3.3-70b-versatile` | Free tier, 14,400 req/day |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, free, 384-dim |
| Vector store | FAISS `IndexFlatIP` | Cosine similarity via L2-normalised inner product |
| Sequential recommender | SASRec (custom PyTorch) | Trained on purchase sequences |
| Web UI | Flask + vanilla HTML/CSS/JS | Replaced Gradio due to macOS segfaults |
| Data source | `2020-Mar.csv` (56M rows, 500K sampled) | E-commerce event log |
| Python | 3.9 | venv at `/Users/aryandubeytopg/Downloads/e-com/.venv` |

---

## 2. Pipeline Architecture

```
User message
     │
     ▼
[Understand]  Groq LLM extracts structured attributes:
              {category, brand, price_min, price_max, keywords[]}
     │
     ▼ (if attributes incomplete, ≤2 turns)
[Elicit]      Groq LLM generates clarifying question
              e.g. "What's your budget?" / "Any brand preference?"
     │
     ▼
[Retrieve]    FAISS semantic search over 58,332 product embeddings
              + hard filters: brand substring, category substring, price range
     │
     ▼
[Re-rank]     SASRec scores candidates using user's purchase history
              (cold-start: returns FAISS order unchanged)
     │
     ▼
[Respond]     Groq LLM generates natural language reply
              grounded strictly in retrieved products
     │
     ▼
Flask JSON response → browser renders chat bubble + product cards
```

---

## 3. Complete File Map

```
ecom_assistant/
├── app.py                        ← Flask server (NOT Gradio — see section 6)
├── config.py                     ← All paths, API keys, hyperparameters
├── launch.sh                     ← One-shot launch script (use this!)
├── run_setup.sh                  ← Re-runs all 4 setup steps from scratch
├── README.md                     ← Updated to reflect free stack
│
├── setup/
│   ├── 01_process_csv.py         ← CSV → product_catalog.json + purchase_sequences.json
│   ├── 02_generate_descriptions.py ← Template + synonym expansion → product_descriptions.json
│   ├── 03_build_vector_store.py  ← sentence-transformers → faiss.index + faiss_id_map.json
│   └── 04_train_sasrec.py        ← Trains SASRec on purchase sequences → sasrec_model.pt
│
├── pipeline/
│   ├── crs.py                    ← CRSPipeline orchestrator (main entry point)
│   ├── llm.py                    ← Groq API wrapper: understand / elicit / respond
│   ├── retriever.py              ← FAISS search with sentence-transformers embeddings
│   ├── recommender.py            ← SASRec re-ranker (SASRecReranker class)
│   └── sasrec_model.py           ← SASRec model architecture + load_model()
│
└── data/                         ← ALL GENERATED — do not delete!
    ├── product_catalog.json      ← 58,332 unique products {product_id, brand, category_code, price}
    ├── purchase_sequences.json   ← 1,087 multi-purchase sessions
    ├── item_id2index.json        ← product_id → integer index
    ├── product_descriptions.json ← Template descriptions for all 58,332 products
    ├── faiss.index               ← FAISS IndexFlatIP (58,332 × 384 vectors)
    ├── faiss_id_map.json         ← FAISS position integer → product_id
    └── sasrec_model.pt           ← Trained SASRec (30 epochs, loss 6.45→2.97)
```

---

## 4. Key Config (`config.py`)

```python
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
CHAT_MODEL    = "llama-3.3-70b-versatile"
EMBED_MODEL   = "all-MiniLM-L6-v2"          # local sentence-transformers
CSV_PATH      = "../2020-Mar.csv"
CSV_SAMPLE_SIZE = 500_000
TOP_K_RETRIEVE  = 20
TOP_K_SHOW      = 5
MAX_ELICIT_TURNS = 2
# SASRec
SASREC_EMBED_DIM = 64
SASREC_NUM_HEADS = 2
SASREC_NUM_LAYERS = 2
SASREC_MAX_SEQ_LEN = 50
SASREC_EPOCHS = 30
```

---

## 5. Data Details

**Source CSV:** `/Users/aryandubeytopg/Downloads/e-com/2020-Mar.csv`

- 56M rows total, 500K sampled
- Columns: `event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session`
- `event_type` values: `view`, `cart`, `purchase`
- Only `purchase` events used for SASRec training sequences

**Generated data stats:**

- 58,332 unique products
- 1,087 purchase sessions with ≥2 items (used for SASRec training)
- 10,289 purchase events total
- FAISS: 58,332 vectors, 384-dim, cosine similarity

---

## 6. Critical Bug Fixes & Decisions

### Why Flask, not Gradio

Gradio was replaced because of persistent macOS segfaults (exit 139).  
Root cause: PyTorch + Rust tokenizers (`sentence_transformers`) crash when called from Gradio's ASGI/uvicorn worker threads on macOS Apple Silicon.

**Fix:** Flask with `threaded=False` (single-threaded, requests handled in main Python thread where PyTorch is safe) + model warm-up before server starts.

### Required env vars (ALL must be set before launch)

```bash
export GROQ_API_KEY="YOUR_GROQ_KEY"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES   # macOS ObjC runtime fork safety
export TOKENIZERS_PARALLELISM=false               # prevents Rust tokenizer deadlock
export OMP_NUM_THREADS=1                          # prevents numpy/torch thread contention
export MKL_NUM_THREADS=1
```

All of these are set inside `launch.sh` — just use that.

### Why SASRec was broken (and fix)

PyTorch `MultiheadAttention` produces NaN when `key_padding_mask` (bool) and `attn_mask` (float -inf) are combined — known issue on MPS and CPU with certain PyTorch versions.  
**Fix:** Removed `key_padding_mask` entirely; changed causal mask from bool triu → float `-inf` triu; added `clip_grad_norm_(5.0)`; force `device=cpu`.

### Why OpenAI was replaced

OpenAI key `YOUR_OPENAI_KEYj-xQWEb19...` returned `429 insufficient_quota`.  
**Fix:**

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, completely free)
- Chat LLM: Groq API free tier (14,400 req/day, OpenAI-compatible interface)

---

## 7. Flask App Endpoints

| Route    | Method | Description                                                    |
| -------- | ------ | -------------------------------------------------------------- |
| `/`      | GET    | Serves the full HTML chat UI (self-contained, no static files) |
| `/chat`  | POST   | `{"message": "..."}` → `{"reply": "...", "products": [...]}`   |
| `/reset` | POST   | Clears conversation state → `{"ok": true}`                     |

**Product object shape** (in `/chat` response):

```json
{
  "product_id": "5809910",
  "brand": "samsung",
  "category_code": "electronics.audio.headphone",
  "price": 45.99,
  "score": 0.712
}
```

---

## 8. Pipeline Classes

### `CRSPipeline` (`pipeline/crs.py`)

Main entry point. Stateful per-conversation.

```python
pipeline = CRSPipeline()
reply = pipeline.chat("wireless earbuds under $50")  # returns str
products = pipeline.last_products                     # list of product dicts
pipeline.reset()                                      # clear conversation
pipeline.set_user_history(["product_id_1", ...])      # personalise with purchase history
```

### `LLMPipeline` (`pipeline/llm.py`)

Groq API wrapper.

- `understand(query, history)` → dict via function calling: `{category, brand, price_min, price_max, keywords}`
- `elicit(query, attributes, history)` → clarifying question string
- `respond(query, products, history, attributes)` → natural language reply

### `ProductRetriever` (`pipeline/retriever.py`)

FAISS search with sentence-transformers.

- `search(query, filters={}, top_k=5)` → list of product dicts
- filters: `brand`, `category`, `price_min`, `price_max`

### `SASRecReranker` (`pipeline/recommender.py`)

SASRec-based re-ranker.

- `rerank(candidates, user_history)` → reordered candidates list
- Cold-start (no history): returns FAISS order unchanged

### `SASRec` (`pipeline/sasrec_model.py`)

- 64-dim embedding, 2 attention heads, 2 transformer layers, max_seq_len=50
- `load_model(device)` — convenience loader

---

## 9. Installed Packages (`.venv`)

```
torch               (CPU, MPS disabled for SASRec)
faiss-cpu
flask
gradio==4.44.1      (installed but NOT used anymore)
sentence-transformers==5.1.2
groq==1.0.0
openai              (installed but NOT used)
pandas
numpy
tqdm
```

---

## 10. What Works, What Doesn't

### Works ✅

- Full chat pipeline end-to-end
- FAISS semantic search over 58K products
- Hard filters (brand, category, price)
- Groq LLM understand / elicit / respond
- SASRec re-ranking (cold-start gracefully falls back to FAISS order)
- Flask server stable (single-threaded, no segfaults)
- Conversation state management
- Reset conversation
- Product cards in the sidebar UI

### Known Limitations

- `threaded=False` means requests are sequential (fine for demo / single user; won't scale to concurrent users without swapping in gunicorn)
- SASRec only trained on 1,087 sessions (small), so personalisation signal is weak — cold-start is the common case
- Category codes in the CSV are raw dot-notation strings (`electronics.audio.headphone`) — no category taxonomy/hierarchy
- No product images

### For Production Scaling

Replace Flask dev server with gunicorn + single worker:

```bash
gunicorn -w 1 -b 0.0.0.0:7860 app:app
```

`-w 1` keeps single-process model to preserve PyTorch thread safety.

---

## 11. How to Re-run Setup From Scratch

Only needed if `data/` is deleted. Steps 1–4 already complete.

```bash
cd /Users/aryandubeytopg/Downloads/e-com/ecom_assistant
source ../.venv/bin/activate
python3 setup/01_process_csv.py        # ~2 min, samples 500K rows
python3 setup/02_generate_descriptions.py  # ~30 sec
python3 setup/03_build_vector_store.py # ~30 sec, local embeddings
python3 setup/04_train_sasrec.py       # ~5-10 min on CPU, 30 epochs
```

---

## 12. Relationship to Original UniMIND Codebase

The new `ecom_assistant/` system is **completely independent** of `UniMIND/`.

- `UniMIND/` still exists untouched and still runs (Chinese CRS with CPT-base backbone)
- `ecom_assistant/` is a parallel implementation for English data
- They share no code, no data, no models
- UniMIND context is in `Agent_Doc/AGENT_CONTEXT.md`

---

## 13. API Keys

| Service | Key           | Status                       |
| ------- | ------------- | ---------------------------- |
| Groq    | `YOUR_GROQ_KEY`     | Active, free tier            |
| OpenAI  | `YOUR_OPENAI_KEYj-...` | Exhausted quota — do NOT use |

---

## 14. Next Steps (Possible Future Work)

1. **Add user login / session persistence** — currently state is in-memory, lost on server restart
2. **Product image integration** — scrape/link product images using product_id
3. **Richer product data** — join with additional metadata (ratings, descriptions)
4. **Better SASRec training** — train on more data (all 56M rows, not just 500K sample)
5. **Category taxonomy** — build a hierarchy from `category_code` dot-notation
6. **Streaming responses** — use Groq streaming API + server-sent events for faster UX
7. **Deploy to cloud** — gunicorn + Railway/Render/Fly.io (single worker to preserve thread safety)
