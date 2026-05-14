# E-Commerce AI Shopping Assistant (BigBasket Pivot)

Conversational recommender powered by Groq + FAISS + optional SASRec.

## Architecture

User query
-> LLM understand (brand/category/price/keywords)
-> Query parser + brand normalization
-> FAISS retrieval over real product descriptions
-> Rating-aware reranker
-> Optional SASRec rerank (enabled only when synthetic sessions >= 500)
-> LLM grounded response

## Project Structure

```
ecom_assistant/
├── app.py
├── config.py
├── run_setup.sh
├── setup/
│   ├── 01_process_bigbasket.py
│   ├── 02_generate_descriptions.py
│   ├── 03_build_vector_store.py
│   └── 04_train_sasrec.py
├── pipeline/
│   ├── brand_mapper.py
│   ├── query_parser.py
│   ├── rating_reranker.py
│   ├── retriever.py
│   ├── recommender.py
│   ├── llm.py
│   ├── sasrec_model.py
│   └── crs.py
└── data/
```

## Setup

```bash
cd /Users/aryandubeytopg/Downloads/e-com
source .venv/bin/activate
export GROQ_API_KEY="YOUR_GROQ_KEY"
cd ecom_assistant
./run_setup.sh
python3 app.py
```

Open: http://127.0.0.1:7860

## Example Queries

- Good after-shave with nice smell
- What's in it? Any alcohol?
- Same thing but cheaper
- Organic skin care under $200
- Anti-bacterial hand wash under $100

## Notes

- Catalog path defaults to `BigBasket Products.csv`.
- Retrieval is grounded in product name, category, brand, description, price, and rating.
- Synthetic purchase sessions are generated for sequence training bootstrapping.
- If synthetic sequences are too few (<500), CRS falls back to FAISS + rating reranking.
