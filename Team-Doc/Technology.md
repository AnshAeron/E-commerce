## SASRec in Our E-Commerce Assistant — Team Briefing

---

### What Problem Does SASRec Solve?

When a user is chatting with our assistant and asks *"what should I buy next?"*, we need to give a **personalized recommendation based on what they've purchased before** — not just a generic popular item. SASRec is the model that does this. It looks at a user's purchase history (a sequence of products) and predicts what they're most likely to want next.

---

### The Core Idea (Without the Jargon)

Think of it like this: if someone bought a phone case, then a screen protector, then a charger — SASRec learns the *pattern* of that sequence and predicts "this person probably wants earphones next." It pays more **attention to recent purchases** than older ones, and it understands the *order* items were bought in — not just which items.

The "self-attention" mechanism is borrowed from the same technology behind ChatGPT. It lets the model figure out which past purchases are relevant to predicting the next one.

---

### How We Implemented It

Our starting point is 2020-Mar.csv — a real-world e-commerce event log with hundreds of thousands of purchase records. This is the only external data source SASRec relies on.

From this CSV, we extract **per-session purchase sequences** — every product a user bought in a single session, in the order they bought it. These sequences are the raw material for training.

The model itself is built entirely with **PyTorch** — no external recommendation library, no black box. We wrote the architecture from scratch in sasrec_model.py and use it for both training and live inference. All tunable parameters (embedding size, layers, learning rate, etc.) sit in a single config.py so they're easy to find and change without touching model code.

Training runs on CPU by default (with CUDA support if available). We deliberately avoided MPS (Apple Silicon GPU) because of known PyTorch bugs with masked multi-head attention — it would silently produce wrong results.

---

### How Training Works — Step by Step

**Step 1 — Preparing the data**

We start with raw purchase event logs (2020-Mar.csv). After processing, we get sequences like:
```
session_42: [phone_case, screen_protector, charger, earphones]
```
Each session is one user's purchase history in order.

**Step 2 — Creating training samples**

We don't just train on the full sequence. For a sequence of length 4, we create **all sub-sequences**:
- Input: `[phone_case]` → Predict: `screen_protector`
- Input: `[phone_case, screen_protector]` → Predict: `charger`
- Input: `[phone_case, screen_protector, charger]` → Predict: `earphones`

This maximizes how much we learn from each session. Sequences shorter than 2 items are thrown away. Inputs are **left-padded** to a fixed length of 50 — padding is index `0`, which the model learns to ignore.

**Step 3 — The model reads a sequence, produces a context vector**

The model converts each item to a 64-dimensional vector (embedding), adds position information (so it knows *order*), then runs it through 2 Transformer-style attention layers. Each layer does two things: first, multi-head self-attention with a **causal mask** (each item can only look at itself and items before it, not future ones), then a small feed-forward network. The output is a single 64-dim vector representing "what this user wants next" — extracted from the last real (non-padded) position.

**Step 4 — Training signal: positive vs. negative**

For each prediction, we give the model:
- A **positive item** — the actual next purchase (correct answer)
- A **negative item** — a random product the user did *not* buy (wrong answer)

The model scores both by taking a dot product between the user context vector and each item's embedding. It's trained to push the positive score toward 1 and the negative score toward 0 using **binary cross-entropy loss**:

$$\mathcal{L} = \text{BCE}(\text{score}_{\text{pos}},\ 1) + \text{BCE}(\text{score}_{\text{neg}},\ 0)$$

This is called **sampled negative training** and is very common in recommendation systems — it's efficient because you don't need to score every item in the catalog on every step, just one negative sample per positive.

We also apply **gradient clipping** (max norm = 5.0) to prevent training instability, which can be an issue with attention-based models on sparse data.

**Step 5 — Saving**

After 30 epochs, the model weights + all architecture config are saved together into `sasrec_model.pt`. This means you can load and use the model without needing to know the hyperparameters separately — they're bundled in the checkpoint.

---

### Hyperparameters (All Tunable in config.py)

These are sensible defaults for our dataset size. If the dataset grows significantly, bumping `SASREC_EMBED_DIM` to 128 and `SASREC_NUM_LAYERS` to 4 would be the first things to try.

| What | Value | Why |
|---|---|---|
| Max sequence length | 50 | Only the last 50 purchases matter |
| Embedding size | 64 | Compact, fast on CPU |
| Attention heads | 2 | Captures 2 independent patterns simultaneously |
| Layers | 2 | Enough depth without overfitting |
| Dropout | 0.2 | Regularization — prevents memorizing |
| Epochs | 30 | Enough passes for convergence |
| Batch size | 256 | Standard, fits in RAM |
| Learning rate | 1e-3 | Adam default, works well here |

---

### How It Fits in the Bigger Picture

```
CSV Data
   ↓ (setup/01)
purchase_sequences.json + item_id2index.json
   ↓ (setup/04)
sasrec_model.pt  ←── trained SASRec
   ↓ (pipeline/recommender.py)
Live recommendations during conversation
   ↓ (pipeline/crs.py + llm.py)
Natural language response to the user
```

SASRec is one component of a larger **Conversational Recommendation System (CRS)**. It handles the *behavioral/sequential* signal. The FAISS vector store handles *semantic/content-based* retrieval. The LLM (Llama via Groq) handles *natural language understanding and generation*. They all work together.


------------------------------------------------------------------------------------


# FAISS in Our E-Commerce Assistant — Team Briefing

---

### What Problem Does FAISS Solve?

When a user types something like *"I need a budget gaming phone with good battery"*, we need to find the most relevant products from our entire catalog — **fast, and by meaning, not just by keyword match**. A simple keyword search would miss products whose descriptions don't contain those exact words. FAISS solves this by letting us search by *concept similarity* across the entire product catalog in milliseconds.

---

### The Core Idea

FAISS (Facebook AI Similarity Search) is essentially **a search engine for vectors**. The idea is:

1. Convert every product into a list of numbers (a vector) that captures the *meaning* of that product
2. Store all those vectors in an index
3. When a user query arrives, convert that query into a vector the same way
4. Ask FAISS: *"which stored vectors are closest to this query vector?"*

The closeness between vectors is the closeness in *meaning*. A query for "wireless earbuds for gym" will land geometrically close to a product described as "Bluetooth sport earphones" even though no word matched exactly. This is called **semantic similarity search** — and it's fundamentally different from a keyword/SQL search.

Think of it like a map. Every product has a coordinate. Similar products cluster together. When a query arrives, FAISS finds the nearest coordinates in that map.

---

### How We Implemented It

**Data source**: 2020-Mar.csv. The CSV has structured fields — brand, category code, price — but no natural language descriptions. Raw structured fields embed poorly, so we first generate rich text descriptions for every product.

**Description generation** (Step 2 of setup): We synthesize a natural language description per product by combining the brand, category, price tier, and — crucially — a **category synonym map** we hand-crafted. For example, `electronics.smartphone` expands to include *"smartphone mobile phone Android iOS gaming calling camera selfie battery"*. This is deliberate vocabulary enrichment so that when a user says "gaming phone", the embedding space connects it to the right products even though the CSV never contained those words.

**Embedding model**: We use `all-MiniLM-L6-v2` from `sentence-transformers` — a locally-running (~90MB) model that produces 384-dimensional vectors. **No API key, no internet dependency at runtime.** This was a conscious decision to keep inference fully offline and free.

**Index type**: We use `faiss.IndexFlatIP` — Flat Inner Product index. "Flat" means it does an exact search (no approximation). Because we L2-normalize all embeddings before storing, inner product is mathematically equivalent to cosine similarity. For our catalog size, exact search is fast enough; if the catalog grew to millions, we'd switch to an approximate index (`IndexIVFFlat`) for speed.

---

### How It Works in Detail

**Step 1 — Building the index (offline, once)**

Every product description is fed through `all-MiniLM-L6-v2` in batches of 512. Each description comes out as a 384-dimensional float32 vector. All vectors are L2-normalized before indexing — this ensures that dot product = cosine similarity during search. The full matrix (all products × 384 dims) is added to a `faiss.IndexFlatIP` and written to disk as `faiss.index`. A companion `faiss_id_map.json` maps each integer index position (0, 1, 2…) back to the actual product ID, since FAISS itself only knows about integer positions.

**Step 2 — Query time (live, every search)**

When a search arrives:
1. The user's query text (e.g. *"budget wireless earphones for running"*) is embedded using the same `SentenceTransformer` model — same 384 dimensions, same L2 normalization
2. `faiss.index.search(query_vector, k)` is called — FAISS scans all stored vectors and returns the top-k closest by inner product (= cosine similarity), along with their scores
3. We deliberately **over-fetch** — we retrieve `TOP_K_RETRIEVE × 4 = 80` candidates instead of just the final 5 we'll show. This is because the next step applies hard filters

**Step 3 — Hard filtering on top of semantic results**

FAISS can only rank by semantic similarity — it knows nothing about price or brand constraints. So after retrieval, we apply structured filters in Python: brand substring match, category substring match, price min/max. Products that don't pass are dropped. We over-fetch precisely to absorb this drop-off and still have enough results after filtering. Final output is `TOP_K_SHOW = 5` products.

**Step 4 — Score interpretation**

The `score` returned by FAISS here is a cosine similarity value between 0 and 1 (since vectors are normalized). A score of 1.0 would be a perfect match. In practice, scores around 0.4–0.7 are typical good matches; anything below 0.2 is likely semantically unrelated.

---

### Important Parameters (All in config.py)

| Parameter | Value | When to Change |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` (384-dim) | Upgrade to `all-mpnet-base-v2` (768-dim) for better accuracy at the cost of ~2× memory and index size |
| `TOP_K_RETRIEVE` | 20 | Increase if filter drop-off is too high and you're regularly getting fewer than 5 results |
| `TOP_K_SHOW` | 5 | The number of products shown to the user — tune based on UX preference |
| `BATCH_SIZE` (index build) | 512 | Lower if you hit memory issues during the build step |
| Index type | `IndexFlatIP` (exact) | Switch to `IndexIVFFlat` if catalog grows past ~500K products and search latency becomes noticeable |

---

### How It Fits in the Bigger Picture

```
2020-Mar.csv
   ↓ (setup/01) extract catalog
product_catalog.json
   ↓ (setup/02) generate rich text descriptions
product_descriptions.json
   ↓ (setup/03) embed + index
faiss.index + faiss_id_map.json
   ↓ (pipeline/retriever.py) live semantic search
Top-K candidate products
   ↓ (pipeline/crs.py + llm.py)
Natural language response to the user
```

FAISS handles the **content/semantic signal** — it finds products relevant to what the user is *saying*. SASRec handles the **behavioral signal** — it finds products relevant to what the user has *bought before*. The LLM then synthesizes both into a natural conversation. All three are needed: FAISS alone ignores purchase history; SASRec alone ignores what the user is explicitly asking for; the LLM alone has no knowledge of the product catalog.
