# How This Project Works — Complete Interview Guide

> Written for the team. No jargon without explanation. Read top-to-bottom once and you'll be able to answer any question about this system.

---

## The Big Picture — What Are We Building?

We built an **AI Shopping Assistant** — you type "I want wireless earbuds under $50 for the gym" and it replies in plain English with the best matching products.

The technical name for this is a **CRS — Conversational Recommender System**.

> **Jargon buster — CRS:** A system that recommends things through a back-and-forth conversation, not just a one-shot search bar.

It combines four technologies working in a chain:

1. A **language model (LLM)** that understands what you want
2. A **vector search engine (FAISS)** that finds similar products
3. A **sequential recommendation model (SASRec)** that personalises results using purchase history
4. A **web server (Flask)** that glues everything into a chat UI

---

## Part 1 — The Raw Data

**File:** `2020-Mar.csv`  
**Size:** 56 million rows (about 4 GB on disk)  
**Source:** A Russian e-commerce platform's event log for March 2020

Each row records one **event** — something a user did:

| Column          | Example                      | Meaning                      |
| --------------- | ---------------------------- | ---------------------------- |
| `event_type`    | `purchase` / `view` / `cart` | What the user did            |
| `product_id`    | `18001424`                   | Anonymous product identifier |
| `brand`         | `samsung`                    | Brand name                   |
| `category_code` | `electronics.smartphone`     | Dot-notation category        |
| `price`         | `25.31`                      | Price in USD                 |
| `user_id`       | `512345678`                  | Anonymous user identifier    |
| `user_session`  | `abc-123-xyz`                | One browsing session         |

**Critical limitation:** There are **no product names** anywhere in the CSV. Every product is only identified by an anonymous numeric ID with brand, category, and price. This is a real-world data quality problem.

---

## Part 2 — Data Processing (Setup Step 1)

**File:** `ecom_assistant/setup/01_process_csv.py`

**Problem:** 56 million rows is too much to work with directly. We only read the first **500,000 rows** using `pandas nrows=500_000`.

**What comes out of 500,000 rows:**

- **58,332 unique products** (after deduplication on `product_id`)
- **1,087 purchase sessions** that have 2 or more items bought (used later for training)

> **Why 500K rows?** RAM and speed. Loading all 56M rows needs ~4 GB RAM and takes 15–20 minutes. 500K rows takes ~2 minutes and still gives us 58K unique products which is a solid product catalog.

**Output files:**

- `product_catalog.json` — one entry per unique product: `{product_id → brand, category, price}`
- `purchase_sequences.json` — one entry per multi-buy session: `{session_id → [product_id, product_id, ...]}`
- `item_id2index.json` — maps each product_id string to an integer (needed by the neural model)

---

## Part 3 — Generating Product Descriptions (Setup Step 2)

**File:** `ecom_assistant/setup/02_generate_descriptions.py`

**Problem:** We have `brand=samsung, category=electronics.smartphone, price=25.31`. How do we match a user query like "cheap gaming phone with good battery"? The words "gaming", "battery", "cheap" appear nowhere in the raw data.

**Solution — Template + Synonym Expansion:**

We write a text description for every product using a template:

```
Samsung Electronics Smartphone. Price: $25.31 — budget value under $50.
Brand: Samsung. Category: Electronics Smartphone.
Keywords: smartphone mobile phone Android iOS gaming calling camera selfie battery
```

The keywords at the end are added from a **synonym map** — a hand-crafted dictionary like:

```python
"electronics.smartphone" → "smartphone mobile phone Android iOS gaming calling camera"
"electronics.audio.headphone" → "headphones earphones wireless Bluetooth noise cancelling"
```

> **Why this matters:** When a user searches "gym earphones", the word "earphones" and "gym" might not be in the raw data, but because we added "earphones" to all headphone descriptions, FAISS can now match them.

**Output:** `product_descriptions.json` — 58,332 rich text descriptions

---

## Part 4 — Building the Vector Search Index (Setup Step 3)

**File:** `ecom_assistant/setup/03_build_vector_store.py`

This is the core of the search system. Here's what happens:

### Step 4a — Text → Numbers (Embedding)

**Tool used:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free, no API needed)

Every product description gets converted into a **vector** — a list of 384 numbers.

> **Jargon buster — Vector/Embedding:** Imagine plotting every product on a map. Products that are semantically similar end up close together on the map. A "Samsung gaming phone" and a "budget Android gaming smartphone" would be near each other. A "kitchen mixer" would be far away. A vector is just the coordinates of a point on this map, written as a list of numbers.

For example:

```
"Samsung Electronics Smartphone..." → [0.23, -0.11, 0.67, 0.04, ..., 0.88]  ← 384 numbers
"Kitfort Appliances Kitchen Mixer..." → [-0.44, 0.09, -0.31, ..., 0.12]  ← 384 different numbers
```

### Step 4b — Storing the Vectors (FAISS Index)

**Tool used:** FAISS (Facebook AI Similarity Search) — an open-source library by Meta

All 58,332 vectors (one per product) are stored in a **FAISS IndexFlatIP** index.

> **Jargon buster — FAISS:** A very fast database for number lists (vectors). Instead of searching text, it compares vectors using dot-product similarity. The product whose vector is closest to your search query's vector is the best match.

> **Jargon buster — IndexFlatIP:** "IP" = Inner Product. Each vector is first L2-normalised (scaled to length 1), so Inner Product becomes Cosine Similarity — the standard measure of how "similar" two directions are in high-dimensional space. "Flat" means it checks every single vector (brute-force), which is accurate but fine at 58K products.

**When a user searches:**

1. Their query text is converted to a vector using the same sentence-transformer model
2. FAISS compares it against all 58,332 stored vectors
3. The 20 most similar vectors are returned as candidates

**Output files:**

- `faiss.index` — the searchable vector database (58,332 × 384 numbers)
- `faiss_id_map.json` — maps FAISS position number → product_id string

---

## Part 5 — Training the Sequential Recommender (Setup Step 4)

**File:** `ecom_assistant/setup/04_train_sasrec.py`  
**Model:** SASRec (Self-Attentive Sequential Recommendation)

### The idea

FAISS finds semantically similar products. But it knows nothing about _you_.

SASRec asks a different question: **"Given that this user previously bought Product A, then Product B, what would they buy next?"**

> **Real-world analogy:** Netflix doesn't just recommend popular shows — it also looks at what you watched last, and in what order, to predict what you'll want next. SASRec does the same for products.

### The training data

From the 500K rows we extracted **1,087 purchase sessions** where a user bought 2+ items in one session. Example:

```
Session X: [kitchen_mixer_id, kitchen_blender_id, coffee_machine_id]
Session Y: [samsung_phone_id, phone_case_id]
```

The model learns: after buying kitchen items → likely to buy more kitchen items. After buying a phone → likely to buy phone accessories.

### How SASRec works internally

> **Jargon buster — Transformer / Self-Attention:** Originally invented for language translation (Google's BERT, GPT). The key idea: every item in a sequence looks at every other item and decides which ones are relevant to it. "I bought a PHONE — the PHONE CASE I bought next is very relevant; the BLENDER I bought 3 items ago is not relevant." Self-attention learns these relationships automatically.

SASRec architecture:

- **Item Embedding layer:** Every product ID is converted to a 64-number vector (learnable)
- **Positional Embedding:** The position of each item in the sequence gets its own vector (so order matters — item bought 1st vs item bought 5th are treated differently)
- **2× Transformer layers:** Each layer runs Multi-Head Self-Attention + Feed-Forward Network
- **Output:** A single "user state" vector representing the user right now

**Hyperparameters used:**

```
Embedding dimension : 64
Attention heads     : 2
Transformer layers  : 2
Max sequence length : 50 (last 50 purchases considered)
Training epochs     : 30
Final training loss : 6.45 → 2.97
```

> **Jargon buster — Epoch:** One full pass through all training data. 30 epochs = the model saw all 1,087 sessions 30 times, adjusting itself each time to get better.

### Cold Start problem

We only have 1,087 training sessions, so most users in a real deployment would have no purchase history. In that case SASRec is skipped and FAISS order is used directly.

> **Jargon buster — Cold Start:** When a recommender system has no prior data about a new user. It's a fundamental challenge in recommendation systems.

---

## Part 6 — The Live Pipeline (How a Chat Message is Processed)

**File:** `ecom_assistant/pipeline/crs.py` — the orchestrator

Every user message goes through **5 stages** in order:

```
User types: "I want wireless earbuds under $50 for the gym"
                            │
                    ┌───────▼────────┐
                    │  1. UNDERSTAND │  LLM reads the message and extracts:
                    │   (Groq LLM)   │  category="electronics.audio.headphone"
                    └───────┬────────┘  price_max=50, keywords=["wireless","gym"]
                            │
                    ┌───────▼────────┐
                    │  2. ELICIT     │  Do we know enough to search?
                    │   (Groq LLM)   │  If not → ask a clarifying question.
                    └───────┬────────┘  Max 2 clarifying questions per conversation.
                            │
                    ┌───────▼────────┐
                    │  3. RETRIEVE   │  Convert query to vector → FAISS search
                    │    (FAISS)     │  Hard filters: brand, category, price range
                    └───────┬────────┘  Returns top 20 product candidates
                            │
                    ┌───────▼────────┐
                    │   4. RERANK    │  If user has purchase history → SASRec
                    │   (SASRec)     │  re-scores candidates by personal fit.
                    └───────┬────────┘  Top 5 products kept.
                            │
                    ┌───────▼────────┐
                    │  5. RESPOND    │  LLM writes a natural reply grounded in
                    │   (Groq LLM)   │  the top 5 actual products.
                    └───────┬────────┘
                            │
             "I recommend the JBL headphones ($29) —
              excellent for workouts, fully wireless..."
```

---

## Part 7 — The LLM Tasks in Detail

**File:** `ecom_assistant/pipeline/llm.py`  
**Model:** `llama-3.3-70b-versatile` via Groq API

The LLM is called three separate times per user message (three different "tasks"):

### Task 1 — Understand (Function Calling)

The LLM is given the user message and a **JSON schema** defining what fields to extract:

> **Jargon buster — Function Calling:** Instead of asking the LLM to write free text, you tell it "fill in this form". You define a schema with fields like `category`, `brand`, `price_max`, `keywords`. The LLM reads the user message and returns a filled JSON object. This is far more reliable than parsing free text.

Input:

```
User said: "wireless earbuds under $50 for the gym"
```

Output:

```json
{
  "category": "electronics.audio.headphone",
  "price_max": 50,
  "keywords": ["wireless", "gym", "sport"]
}
```

These attributes **accumulate across the conversation** — if the user later says "make it Samsung", `brand=samsung` is added to the existing attributes.

### Task 2 — Elicit

If the LLM's Understand step returns very few attributes (e.g. user just said "something nice"), the system asks a clarifying question before searching.

```
User: "I need something for my kid"
→ Elicit: "How old is your child, and are you looking for electronics, toys, or clothing?"
```

Maximum 2 elicitation turns per conversation (configurable in `config.py`).

### Task 3 — Respond

Once we have products from FAISS + SASRec, the LLM is given:

- The user's original message
- Full conversation history
- The top 5 products (brand, category, price, product_id)

It then writes a conversational recommendation grounded strictly in those products — it is **not allowed to invent products** that aren't in the list.

---

## Part 8 — The Web Server

**File:** `ecom_assistant/app.py`  
**Technology:** Flask (Python web framework)

### Why Flask and not Gradio?

Gradio was originally used but caused **segfault (exit code 139)** on macOS Apple Silicon.

> **Jargon buster — Segfault:** A "segmentation fault" — the program tried to read/write memory it doesn't own. The OS immediately kills it. This happened because Gradio uses multi-threaded async workers, and PyTorch's sentence-transformers Rust tokenizer is not thread-safe on macOS.

Fix: Flask with `threaded=False` — all requests are handled one at a time in the main Python thread where PyTorch is safe.

### API Endpoints

| URL      | Method | What it does                                                               |
| -------- | ------ | -------------------------------------------------------------------------- |
| `/`      | GET    | Serves the full HTML chat UI page                                          |
| `/chat`  | POST   | Takes `{"message": "..."}` → returns `{"reply": "...", "products": [...]}` |
| `/reset` | POST   | Clears conversation memory → `{"ok": true}`                                |

---

## Part 9 — All Technologies Used

| Technology                | What it is                               | Why we use it                              |
| ------------------------- | ---------------------------------------- | ------------------------------------------ |
| **Python 3.9**            | Programming language                     | Main language for all code                 |
| **Flask**                 | Lightweight Python web framework         | Serves the chat UI and API                 |
| **Groq API**              | Cloud API to run LLaMA models            | Free tier, fast, 14,400 req/day            |
| **LLaMA 3.3 70B**         | Large language model by Meta             | Understands user intent, writes replies    |
| **sentence-transformers** | Python library for text embeddings       | Converts text to vectors (local, free)     |
| **all-MiniLM-L6-v2**      | Specific embedding model (22MB)          | Fast 384-dim embeddings, good quality      |
| **FAISS**                 | Vector search library by Meta/Facebook   | Search 58K products in milliseconds        |
| **SASRec**                | Transformer-based sequential recommender | Personalises results with purchase history |
| **PyTorch**               | Deep learning framework by Meta          | Powers SASRec model                        |
| **pandas**                | Data analysis library                    | Reading and processing the CSV             |
| **numpy**                 | Numerical computing library              | Vector math operations                     |

---

## Part 10 — Key Design Decisions (Common Interview Questions)

**Q: Why use a vector search instead of keyword search (like SQL LIKE)?**

> Keyword search only matches exact words. "Earphones" won't match a product tagged "headphones". Vector search understands meaning — both map to similar regions in vector space, so they match even without sharing words.

**Q: Why use a separate LLM for understanding vs responding?**

> They're the same model (LLaMA 3.3 70B) but called with different prompts and tools. Separating "understand" (function calling → structured JSON) from "respond" (free-text generation) makes each step reliable and debuggable independently.

**Q: Why does the pipeline call the LLM 3 times per message? Isn't that slow?**

> Yes, it takes 2–5 seconds per message. The tradeoff is quality — each step has a focused prompt with a clear output format. An alternative is to do it in one mega-prompt, but that hurts reliability significantly.

**Q: What is cosine similarity and why is it used in FAISS?**

> It measures the angle between two vectors. Two vectors pointing in the same direction = score 1.0 (identical meaning). Perpendicular = 0.0 (unrelated). Opposite = -1.0. We L2-normalise all vectors so dot-product equals cosine similarity, which is the standard for semantic search because the length of the vector (influenced by text length) doesn't matter, only the direction (meaning) does.

**Q: What's the difference between FAISS retrieval and SASRec reranking?**

> - FAISS answers: "Which products are semantically similar to this query text?"
> - SASRec answers: "Given _this specific user's_ purchase history, which of these candidates do they prefer?"  
>   Both signals together are stronger than either alone. FAISS handles relevance; SASRec handles personalisation.

**Q: What is the cold start problem in this system?**

> SASRec needs purchase history to personalise. New users have no history. In that case, SASRec is skipped entirely and the FAISS-ranked order is used directly. This is the common case right now since our training data only has 1,087 sessions.

**Q: Why not use OpenAI for embeddings?**

> The OpenAI API key hit its quota limit (`429 insufficient_quota`). We replaced it with `sentence-transformers/all-MiniLM-L6-v2`, which runs locally for free. It has slightly lower quality than `text-embedding-ada-002` but no cost, no rate limits, and no internet dependency.

**Q: How does the conversation remember context?**

> The `ConversationState` object stores:
>
> 1. Full `history` list (all user + assistant messages)
> 2. Accumulated `attributes` (category, brand, price_max etc. extracted over multiple turns)
> 3. `elicit_count` (how many clarifying questions asked so far)
>
> Each turn, the LLM gets the full history + the newly extracted attributes are merged into the growing attributes dict.

---

## Part 11 — Current Limitations

| Limitation                               | Cause                                                                        | Fix                                                      |
| ---------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------- |
| Products have no real names              | Source CSV had no names                                                      | Batch Groq LLM generation to create names                |
| Only 500K of 56M rows processed          | RAM / time constraint                                                        | Set `CSV_SAMPLE_SIZE = None` in config.py + re-run setup |
| SASRec rarely activates                  | Only 1,087 training sessions                                                 | Train on full 56M rows for more sessions                 |
| Single-user only (threaded=False)        | macOS PyTorch thread safety                                                  | Switch to gunicorn with 1 worker for production          |
| Prices look too low (e.g. $3 smartphone) | Source data is from 2020 Russia (ruble-denominated, converted to USD poorly) | Normalise prices or label them as relative, not absolute |

---

## Part 12 — How to Restart the System

```bash
# One-time setup of environment variables + start
export GROQ_API_KEY="YOUR_GROQ_KEY"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES   # prevents macOS ObjC fork crash
export TOKENIZERS_PARALLELISM=false               # prevents Rust tokenizer deadlock
export OMP_NUM_THREADS=1                          # prevents numpy thread contention
export MKL_NUM_THREADS=1

source /Users/aryandubeytopg/Downloads/e-com/.venv/bin/activate
cd /Users/aryandubeytopg/Downloads/e-com/ecom_assistant
python3 app.py
```

---

## Summary in One Paragraph

We took a 56-million-row e-commerce event log, sampled 500K rows, extracted 58,332 unique products, wrote rich text descriptions for each using brand + category + synonym expansion, then converted all descriptions to 384-dimensional vectors using a local sentence-transformer model and stored them in a FAISS vector index. We also trained a SASRec transformer model on 1,087 purchase sessions to learn sequential buying patterns. At runtime, a user's chat message is parsed by LLaMA 3.3 (70B) via Groq to extract structured attributes (category, brand, budget), the query is embedded and searched against FAISS with hard filters applied, the top 20 candidates are re-ranked by SASRec using the user's purchase history, and finally LLaMA 3.3 writes a natural-language recommendation grounded in the top 5 products. The whole thing runs as a Flask web server at `localhost:7860`.

---

## From a CSV File to Answering User Queries — Full Pipeline

---

### The Big Picture First

The CSV is not used at query time. It is used **once, offline, during setup** to produce 3 things: a product knowledge base, a trained search index, and a trained recommendation model. At query time, only those outputs are used. Think of setup as "cooking the food" and query-time as "serving it."

---

## PHASE 1 — SETUP (Run Once)

### Step 1 — Extract Everything From the CSV

The CSV (2020-Mar.csv) contains raw e-commerce event logs — every `view`, `cart`, `purchase` event with columns: `event_time`, `event_type`, `product_id`, `session_id`, `user_session`, `brand`, `category_code`, `price`.

Three things are extracted from it:

**A. Product Catalog** — Every unique product with its brand, category, and price. This is purely the structured metadata. ~58K unique products become one JSON lookup table.

**B. Purchase Sequences** — Only `purchase` events are kept. They're grouped by `user_session` and sorted by time. So a session becomes:

```
session_XYZ → [phone_case, screen_protector, charger, earphones]
```

Only sessions with ≥2 purchases are kept — single-item sessions are useless for sequential training. These sequences are what SASRec will learn from.

**C. Item Index Map** — Every product ID gets a unique integer (0, 1, 2…). Models work with numbers, not string IDs.

---

### Step 2 — Generate Product Descriptions (Turning Structured Data → Natural Language)

This is a crucial step that's easy to overlook. The catalog only has: `brand=samsung`, `category=electronics.smartphone`, `price=299`. You can't do meaningful semantic search on that alone.

So we synthetically generate a rich text description for every product using a template + a hand-crafted **category synonym map**:

```
"Samsung electronics smartphone mobile phone Android iOS gaming
 calling camera selfie battery — mid-range value under $300"
```

The synonym map is why _"gaming phone"_ from a user query will actually match a product categorized as `electronics.smartphone` — the vocabulary bridge is built here, not at search time.

---

### Step 3 — Build the FAISS Vector Index

Every one of those text descriptions is fed through `all-MiniLM-L6-v2` (a local sentence embedding model) which converts each description into a 384-number vector capturing its _meaning_. All ~58K vectors are stored in a FAISS index on disk.

Result: `faiss.index` — a searchable map of semantic space, where similar products sit close to each other numerically.

---

### Step 4 — Train SASRec

The purchase sequences from Step 1 are used to train SASRec for 30 epochs. After training, the model has learned: _"given this sequence of purchased products, predict what this user wants next."_

The trained model is saved as `sasrec_model.pt`.

---

**After these 4 steps, the CSV is never touched again.** Everything the assistant needs is in:

- `product_catalog.json` — metadata for filtering
- `faiss.index` — semantic search
- `sasrec_model.pt` — behavioral recommendation

---

## PHASE 2 — QUERY TIME (Every User Message)

Let's say the user types: _"I want wireless earphones for running, budget around $50"_

---

### Step 1 — Understand (LLM extracts structure)

The message is sent to **Llama-3.3-70b** (via Groq API) with a function-calling schema. The LLM responds not with text but with a structured JSON:

```json
{
  "category": "electronics.audio.earphone",
  "price_max": 50,
  "keywords": ["wireless", "running", "sport"]
}
```

This structured output is stored in the `ConversationState` for this session and accumulates across turns.

---

### Step 2 — Elicit (optional clarifying question)

If the extracted attributes are too sparse (no category found, no price, no keywords at all), the LLM generates **one short clarifying question** — e.g. _"Are you looking for over-ear headphones or in-ear earbuds?"_

This only happens up to `MAX_ELICIT_TURNS = 2` times per conversation so it doesn't become annoying. Once enough is known, it's skipped entirely.

---

### Step 3 — Build a Rich Query + Retrieve via FAISS

The extracted attributes are combined back into a rich query string:

```
"I want wireless earphones for running, budget around $50
 audio earphone wireless running sport"
```

This query is embedded using the same `all-MiniLM-L6-v2` model → 384-dim vector.

FAISS searches its index and returns the **80 most semantically similar products**. We over-fetch 80 to account for the next step.

Then **hard filters** are applied from the structured attributes:

- `category` contains _"earphone"_ → drop anything that isn't
- `price` ≤ 50 → drop anything more expensive

After filtering, we have ~10 semantically relevant AND structurally valid candidates.

---

### Step 4 — Re-rank with SASRec

If this user has a purchase history (known from `purchase_sequences.json`), SASRec steps in:

1. Their purchase history is converted to an integer sequence (padded to length 50)
2. SASRec produces a 64-dim user vector — _"what this user tends to buy next"_
3. That user vector is dot-producted against each candidate product's embedding
4. Candidates are re-sorted: products that fit both the query **and** the user's behavioral pattern come first

If the user has no history (new/anonymous user), FAISS order is kept unchanged — this is the cold-start fallback.

Top 5 results are selected.

---

### Step 5 — Respond (LLM generates natural language)

The top 5 products + the full conversation history are passed to the LLM with a prompt like:

```
"You are a helpful shopping assistant. The user asked: '...'
Here are the best matching products: [product list]
Write a natural, helpful recommendation."
```

The LLM generates a conversational response explaining the products, why they match, and any relevant comparisons.

---

## The Complete Flow — One Diagram

````
2020-Mar.csv```

     │
     ├──[setup/01]──► product_catalog.json   ──────────────────────────► hard filters at query time
     │                purchase_sequences.json ──[setup/04 train SASRec]──► sasrec_model.pt
     │                item_id2index.json      ──────────────────────────► integer encoding
     │
     └──[setup/01]──► product_catalog.json
                           │
                      [setup/02]
                           │
                      product_descriptions.json
                           │
                      [setup/03 embed + index]
                           │
                      faiss.index

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ SETUP DONE — CSV NEVER USED AGAIN ━━━

User message: "wireless earphones under $50 for running"
     │
     ├──[LLM understand]──► structured attributes {category, price_max, keywords}
     │
     ├──[LLM elicit]──► clarifying question (if attributes too sparse)
     │
     ├──[FAISS search]──► 80 semantic candidates → hard filter → ~10 candidates
     │
     ├──[SASRec rerank]──► re-order by user purchase history
     │
     └──[LLM respond]──► natural language reply to user
````

---

### Key Insight

The CSV gives us two completely different things that feed two completely different systems:

| What we extract from CSV                      | What it powers                                                                              |
| --------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Product metadata (brand, category, price)     | FAISS (after converting to descriptions) — answers _"what is relevant to what I'm asking?"_ |
| Purchase sequences (who bought what in order) | SASRec — answers _"what does this user personally tend to buy next?"_                       |

The LLM contributes no product knowledge of its own — it only understands the user's language and composes the final response. All product intelligence comes from the CSV-derived data.
