# AI Agent Context Document — E-Commerce Conversational Recommender System
> Read this fully before doing anything. This document gives complete context on the project, decisions made, work done, and where to go next.
Original Dataset(but in chinese, not wanted)- https://github.com/LeeeeoLiu/U-NEED/blob/main/examples.md

> Session update: For the BigBasket migration and architecture pivot, read `SESSION_3_BIGBASKET_PIVOT.md` first.
---

## 1. Project Goal (What We Are Building)

An **English-language, interactive e-commerce product recommendation assistant** where:
- A user types a natural language query (vague or specific), e.g. *"I need wireless earbuds under $50, something like Sony"*
- The AI asks intelligent follow-up questions to clarify needs
- The AI searches the user's **own product catalog CSV** (the source of truth) via RAG
- The AI returns grounded product recommendations (only products that exist in the catalog)
- Responses are personalized using past purchase patterns (sequential recommendation)

This is NOT a generic chatbot. It is a **4-task pipeline**: Understand → Elicit → Recommend → Response.

---

## 2. Source Codebase

**Repository location:** `/Users/aryandubeytopg/Downloads/e-com`

**Paper:** [Conversational Recommender System and Large Language Model Are Made for Each Other in E-commerce Pre-sales Dialogue](https://aclanthology.org/2023.findings-emnlp.643.pdf) — EMNLP 2023 findings

**Authors:** Yuanxing Liu et al.

### Directory Structure
```
e-com/
├── UniMIND/                      ← Main CRS training/eval code (PRIMARY)
│   ├── pretrain_crs.py           ← Entry point for training
│   ├── base_models/
│   │   ├── base.py               ← BARTCRSModel, CPTCRSModel (text + rec fusion)
│   │   ├── sasrec.py             ← SASRec sequential recommendation model
│   │   ├── modeling_cpt.py       ← CPT model architecture
│   │   └── module.py
│   ├── utils/
│   │   ├── train.py              ← Training loop (already patched — see section 5)
│   │   └── eval.py               ← Evaluation metrics
│   └── available_resources/
│       ├── datasets/u_need/
│       │   ├── u_need.py         ← Dataset class, data processing pipeline
│       │   ├── item_kg.txt       ← Product knowledge graph (SAMPLE ONLY — 1 item)
│       │   ├── train_dialogue.json, valid_dialogue.json, test_dialogue.json ← Chinese dialogues
│       │   ├── allsid2seq.json   ← Session → item purchase sequence map (stub created)
│       │   ├── item2vec-300d.pkl ← 300-dim item embeddings (stub created, zeros)
│       │   ├── prompt4LLM/       ← LLM-annotated samples (train/valid/test)
│       │   └── saved_data/       ← Generated pkl/json cache files
│       ├── metrics/              ← acc.py, gen.py, rec.py
│       └── models/
│           ├── cpt-base/         ← fnlp/cpt-base weights DOWNLOADED (580MB)
│           ├── cpt-large/        ← config only, no weights
│           ├── bart-base-chinese/← config only, no weights
│           └── bart-large-chinese/ ← config only, no weights
├── data_processing/              ← Scripts to convert data formats
│   ├── rec_data_process.py       ← Add candidate sets to raw data
│   ├── unified2ft.py             ← Convert CRS data → LLM fine-tune format (WORKING)
│   ├── crs_s2_data_construct.py  ← Stage 2 CRS training data
│   └── llm_s2_data_construct.py  ← Stage 2 LLM training data
└── llm_tuning/                   ← LLM fine-tuning code (Alpaca, ChatGLM)
    ├── lora_ALpaca/
    └── lora_ChatGLM/
```

---

## 3. Original System Architecture (What the Paper Built)

### 4-Task Pipeline
```
User utterance
      │
      ▼
[Understand] → Extract product attributes from user/system input
               e.g. "品类：化妆水" (category: toner)
      │
      ▼
[Elicit]     → Generate clarifying question to get more user needs
               e.g. "What skin type / price range?"
      │
      ▼
[Recommend]  → Rank items from catalog using:
               - SASRec (sequential rec over user purchase history)
               - CPT/BART encoder hidden state (dialogue context)
               - Fusion layer combining both signals
               - Candidate re-ranking over 20 items per session
      │
      ▼
[Response]   → Generate natural language system reply
```

### Model Architecture
- **Text backbone:** CPT (Chinese Pretrained Transformer, `fnlp/cpt-base`) or BART-Chinese
- **Sequential recommender:** SASRec operating on item ID sequences
- **Fusion:** Custom `Fusion` layer combining SASRec output (300d) + BART/CPT hidden state
- **LLM integration slot:** Optional `[LLM]` prefix token — LLM output injected as context hint before recommendation. Supports `alpaca` and `chatglm` as injection sources.

### Key Data Files Required by the Original System
| File | Description |
|---|---|
| `item_kg.txt` | CSV: `item_id, seller_id, attribute_name, value` — the product knowledge graph |
| `allsid2seq.json` | `{session_id: [item_id, item_id, ...]}` — user purchase sequences |
| `item2vec-300d.pkl` | `{item_id: np.array(300)}` — word2vec item embeddings |
| `saved_data/item_id2index.pkl` | `{item_id: integer_index}` — item ID → embedding matrix index |
| `{train,valid,test}_dialogue.json` | Raw dialogue sessions with act tags, attributes, rec items |

---

## 4. Why the Original System Cannot Be Used As-Is

| Blocker | Detail |
|---|---|
| **Language** | Entire dataset, model vocabulary, prompts are Chinese (Mandarin). User wants English. |
| **Dataset** | Only 2–3 sample dialogues included. Full U-NEED dataset requires author access request. |
| **No interactive interface** | Batch train/eval only. No chat loop, no CLI, no web UI. |
| **No product descriptions** | The incoming user CSV has no free-text descriptions, only structured attributes. |
| **GPU required** | Training the CPT/BART model requires CUDA. User is on macOS (Apple Silicon, no CUDA). |

---

## 5. Code Patches Already Applied

The following bugs were fixed during the session. Do NOT revert these:

### `UniMIND/utils/train.py`
1. **`pytorch_transformers.WarmupLinearSchedule`** → replaced with `transformers.get_linear_schedule_with_warmup` (package removed)
2. **`transformers.AdamW`** → replaced with `torch.optim.AdamW` (removed from transformers in v4.x)
3. **`best_valid_score = 0`** → changed to `-float('inf')` in both `finetune_model` and `train_model` (model checkpoint was never saved because score never exceeded 0)

### All `.py` files in `UniMIND/`
4. **Hardcoded `torch.device("cuda:0")`** → replaced with `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` in:
   - `utils/train.py`
   - `utils/eval.py`
   - `base_models/base.py`
   - `base_models/sasrec.py`

### Stub data files created (minimal, for testing only)
- `available_resources/datasets/u_need/allsid2seq.json` — stubs with 3 session IDs, empty histories
- `available_resources/datasets/u_need/item2vec-300d.pkl` — zero vectors for 61 item IDs
- `available_resources/datasets/u_need/saved_data/item_id2index.pkl` — 61 item IDs from sample data

### `data_processing/unified2ft.py`
- Successfully runs with `--kg_file item_kg.txt` against the sample data
- Output goes to `/tmp/ecom_output/` (train/valid/test_sample.json with LLM fine-tune format)

---

## 6. Planned Architecture — Approach A (AGREED DIRECTION)

**Discard the Chinese CPT/BART backbone. Keep the architectural pattern and SASRec.**

```
ORIGINAL COMPONENT          →  REPLACEMENT
─────────────────────────────────────────────────────
CPT/BART text model         →  GPT-4o mini (OpenAI API)
Chinese dialogue dataset    →  Not needed (LLM is zero-shot)
Chinese item_kg.txt         →  User's product CSV (reformatted)
item2vec embeddings         →  OpenAI text-embedding-3-small or similar
allsid2seq.json sequences   →  Derived from user_session + event_time in CSV
SASRec model                →  Keep, retrain on user's purchase sequences
Fusion layer                →  Keep or simplify to weighted sum
[LLM] hint injection        →  GPT-4o mini output injected here
```

### Full New Pipeline
```
User query (English)
      │
      ▼
GPT-4o mini [Understand]
  → Structured attribute extraction: {brand, category, price_max, ...}
      │
      ▼
GPT-4o mini [Elicit] (if attributes incomplete)
  → "What is your budget?" / "Which brand do you prefer?"
      │
      ▼
Vector Store [RAG Retrieval]
  → Embed user query + extracted attributes
  → Similarity search over product catalog embeddings
  → Filter by hard constraints (price ≤ X, category = Y)
  → Return top-K candidate products grounded in the catalog
      │
      ▼
SASRec [Re-ranking]
  → "Users with similar purchase history also bought..."
  → Re-rank candidates using sequential behavior patterns
      │
      ▼
GPT-4o mini [Response]
  → Given: user query + retrieved product details
  → Generate natural language recommendation
  → ONLY references products from retrieval step (no hallucination)
      │
      ▼
"Based on your needs, I recommend [Product X] at $45 —
 it's a Sony-style wireless earbud in electronics.audio,
 frequently bought alongside [Product Y]."
```

---

## 7. Incoming User Dataset (CSV — NOT YET PROVIDED)

### Schema
| Column | Type | Role in System |
|---|---|---|
| `event_time` | datetime UTC | Sort order for session sequences → `allsid2seq.json` |
| `event_type` | string (always "purchase") | Filters data (no browse/view events) |
| `product_id` | string/int | Becomes item ID — primary key for catalog |
| `category_id` | int | Fallback if `category_code` missing |
| `category_code` | string e.g. `electronics.smartphone` | Searchable attribute in vector store |
| `brand` | string (lowercase) | Searchable attribute, used for "similar to brand X" queries |
| `price` | float | Hard filter constraint in retrieval |
| `user_id` | string/int | User identity for SASRec personalization |
| `user_session` | string | Maps to `sid` — groups purchases into sessions |

### Known Limitations of This CSV
- **No product descriptions** — only structured attributes. RAG quality will be limited for spec-based queries ("long battery life", "gaming headset").
  - Mitigation options: enrich via product API lookup, or GPT-4o mini generates synthetic descriptions from brand + category_code
- **Purchase-only events** — SASRec works best with click/view sequences. Purchase-only is sparser but still valid.
- **brand can be null** — handle missing values in preprocessing

### CSV → System Data Files Mapping
```
CSV columns                         → System file
────────────────────────────────────────────────────────
product_id + brand + category_code  → item_kg.txt (reformatted)
  + price + category_id

user_session + product_id           → allsid2seq.json
  (sorted by event_time)              {session_id: [product_id, ...]}

product_id (all unique)             → item_id2index.pkl
                                      {product_id: integer_index}

product_id + text fields            → Vector store embeddings
  (brand + category_code + price)     (FAISS or ChromaDB)
```

---

## 8. LLM Decision

**Chosen: GPT-4o mini (OpenAI API)**

Reasons:
- User dataset is English; model is English-first
- No GPU available (macOS, Apple Silicon)
- Handles all 4 tasks zero-shot via prompting
- Cheapest capable OpenAI model
- OpenAI function calling enables structured attribute extraction (Understand task)
- Easy integration: `pip install openai`, single API key

The project already has an `[LLM]` token injection slot in `u_need.py` designed for this — LLM output maps to the `alpaca_output` field in the data pipeline.

---

## 9. Environment

- **OS:** macOS (Apple Silicon)
- **Python:** 3.9.6 (system Python at `/Library/Developer/CommandLineTools/usr/bin/python3`)
- **No virtual environment set up** — packages installed via `pip3` to user site-packages
- **No CUDA** — CPU-only torch

### Installed Packages (relevant)
```
torch, transformers, sentencepiece, sacremoses, scikit-learn,
nltk, gensim, sentence-transformers, pytorch-transformers,
pandas, numpy, jsonlines, wandb, huggingface-hub
```

### Downloaded Models
- `fnlp/cpt-base` → `UniMIND/available_resources/models/cpt-base/` (full weights, 580MB)
- All other model directories have configs only (no weights)

---

## 10. What Has Been Run Successfully

| Script | Command | Result |
|---|---|---|
| `unified2ft.py` | `python3 unified2ft.py --dataset_path ../UniMIND/.../saved_data/json --save_path /tmp/ecom_output --kg_file item_kg.txt` | ✅ Outputs train/valid/test LLM fine-tune JSONs |
| `pretrain_crs.py` (data gen) | Data generation block uncommented, ran once to generate pkl files | ✅ Created `train/valid/test_samples.pkl` |
| `pretrain_crs.py` (training) | `python3 pretrain_crs.py --base_name cpt-base --sample_random --no_wandb` | ❌ Fails at recommendation training due to insufficient sample data (only 2-3 items) — not fixable without full U-NEED dataset |

---

## 11. Immediate Next Steps (Waiting On)

1. **User to provide product CSV** — will trigger the main implementation work
2. Once CSV is received:
   - Parse and validate schema
   - Build `item_kg.txt` from CSV columns
   - Build `allsid2seq.json` from `user_session` + `event_time` + `product_id`
   - Build `item_id2index.pkl` from all unique `product_id` values
   - Optionally generate synthetic product descriptions via GPT-4o mini for richer RAG
   - Set up vector store (FAISS recommended for local, no server needed)
   - Train SASRec on purchase sequences
   - Build conversation loop: GPT-4o mini ↔ vector store ↔ SASRec
   - Build simple CLI chat interface

---

## 12. Key Design Decisions (Settled — Do Not Re-debate)

| Decision | Choice | Reason |
|---|---|---|
| LLM | GPT-4o mini | English, no GPU, zero-shot capable, cheap |
| CRS backbone | Discard CPT/BART | Chinese-only, requires training data we don't have |
| Architecture | Keep 4-task pipeline pattern | Maps perfectly to user's requirements |
| SASRec | Keep, retrain on CSV | Language-independent, trains on item IDs only |
| RAG store | FAISS (local) | No external server, works on macOS CPU |
| Dialogue training data | Not needed | LLM handles Understand/Elicit/Response zero-shot |
| Chinese U-NEED dataset | Discard | Language mismatch, domain mismatch (beauty/Chinese) |

---

## 13. Open Questions (Unresolved)

1. **Product descriptions** — Will user enrich the CSV with descriptions, or should GPT generate synthetic ones from brand + category_code?
2. **Scale of CSV** — How many products and how many purchase events? This determines whether FAISS in-memory works or needs an index on disk.
3. **OpenAI API key** — User needs to provide this for GPT-4o mini integration.
4. **Conversation memory** — Should the system remember context across sessions (returning user), or only within a single conversation?
5. **Output format** — CLI chat loop, or will this eventually need a web API / frontend?
