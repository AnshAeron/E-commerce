# SESSION 3: BigBasket Pivot

## Why 2020-Mar.csv Was Deprecated

The previous pipeline failed on turn-2+ interactions because the source data was weak for semantic grounding:

- No real product titles or descriptions in the working catalog representation
- Category signal depended on sparse dot-notation instead of human-friendly taxonomy
- Generated synthetic descriptions were generic and repetitive
- Retrieval had low semantic precision for intent refinements (brand/quality/scent)
- SASRec was trained on weak event dynamics and produced low practical lift for conversational follow-ups

Consequence: turn-1 could often return something plausible, but follow-up constraints like "something branded" or "same thing but cheaper" were brittle.

## Why BigBasket Is Better

Actual audit (from the attached CSV in this workspace):

- Rows: 27,555
- Unique brands: 2,312
- Top-level categories: 11
- Sub-categories: 90
- Types: 426
- Missing descriptions: 115 rows
- Missing ratings: 8,626 rows
- Rating distribution: min 1.0, max 5.0, avg 3.943 (18,929 rated products)

### Richness Comparison

| Dimension | Old 2020-Mar Working Representation | BigBasket Pivot |
|---|---|---|
| Product identity | Numeric IDs | Real product names |
| Brand quality | Sparse/lower signal in chat layer | 2,312 real brands |
| Category UX | Dot-notation raw code | Human-readable category + sub-category + type |
| Text grounding | Synthetic templates | Real product descriptions |
| Price realism | Single field with weaker context | Sale + market price |
| Quality signal | None/indirect | Explicit ratings |
| Turn-2 retrieval | Weak | Strong via metadata + reranking |

## Breaking Changes

1. Default CSV source switched from `2020-Mar.csv` to `BigBasket Products.csv`.
2. Setup stage-1 script replaced:
   - Removed: `ecom_assistant/setup/01_process_csv.py`
   - Added: `ecom_assistant/setup/01_process_bigbasket.py`
3. Catalog schema changed from `{brand, category_code, price}` to rich metadata:
   - `product_name`, `brand`, `category`, `sub_category`, `type`
   - `sale_price`, `market_price`, `rating`, `description`, `search_text`
4. Retriever output now returns grounded product payload (name/category/rating/description), not just IDs.
5. CRS orchestration now includes query parsing + rating reranking + follow-up handling.

## Migration Guide

### Kept and Adapted

- FAISS indexing pipeline: kept (`setup/03_build_vector_store.py`) and now embeds richer descriptions
- Flask app: kept (`ecom_assistant/app.py`), updated product cards for real names/ratings
- Groq LLM flow: kept (`pipeline/llm.py`), updated schema prompts for BigBasket categories

### Replaced

- Legacy 2020 schema processor: removed
- Description generator: rewritten to use real metadata rather than synthetic-only category synonyms
- CRS retrieval stack: now combines parser + brand normalization + rating-aware reranking

### New Modules

- `ecom_assistant/pipeline/query_parser.py`
- `ecom_assistant/pipeline/brand_mapper.py`
- `ecom_assistant/pipeline/rating_reranker.py`

## New Capability Envelope

1. Multi-turn robustness
   - Follow-up detail questions can answer from previously surfaced product description.
2. Brand-aware retrieval
   - User brand mentions are normalized before hard filtering.
3. Price-constrained alternatives
   - "same thing but cheaper" now triggers cheaper-variant reranking logic.
4. Quality-aware ranking
   - Ratings influence rank order when semantic scores are close.
5. Grounded responses
   - LLM receives real product names, brand, category, price, rating, and description context.

## Training Strategy Reset

### Old (failed)

- Large weak event sample
- Sparse semantics
- SASRec loss reduction did not translate to robust conversational utility

### New (implemented baseline)

- Full BigBasket catalog as item universe
- Synthetic purchase sessions generated from category/brand affinity + rating-weighted sampling
- Sequence length: 3-7
- Session count: configurable (default 240)
- SASRec use in CRS gated by data sufficiency:
  - Enabled only when sequence count >= 500
  - Otherwise fallback to FAISS + rating reranker

## Recommended Next Work (Session 4)

1. Increase synthetic sessions to 800-1500 and retrain SASRec.
2. Add benchmark harness for 100 adversarial multi-turn queries.
3. Add explicit description evidence snippets in final answer templates.
4. Add taxonomy dictionary for better category extraction from colloquial language.

## Operational Commands

```bash
cd /Users/aryandubeytopg/Downloads/e-com/ecom_assistant
./run_setup.sh
python3 app.py
```
