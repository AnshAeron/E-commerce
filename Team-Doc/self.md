# CRS

User: "I need wireless earbuds under $50, something like Sony"
│
▼
┌─────────────────────┐
│ GPT-4o mini │ Understand task
│ "Extract: brand=Sony,│ → Structured query
│ category=earbuds, │
│ price_max=50" │
└─────────────────────┘
│
▼
┌─────────────────────┐
│ Vector Search │ RAG retrieval
│ over YOUR CSV │ → Filter by price ≤ 50
│ (product_id,brand, │ → Similarity search on
│ category, price) │ brand + category embeddings
└─────────────────────┘
│
▼
[product_A, product_B, product_C] ← grounded in YOUR data
│
▼
┌─────────────────────┐
│ SASRec │ Re-rank by purchase patterns
│ (trained on your │ "users who bought X also bought Y"
│ user_session data) │
└─────────────────────┘
│
▼
┌─────────────────────┐
│ GPT-4o mini │ Response task
│ + retrieved products│ RAG-grounded generation
│ as context │ LLM only says things provable
└─────────────────────┘ from your catalog
│
▼
"Based on your budget, I'd recommend [product_B]
— it's a Sony-style earbuds at $45 in the
electronics.audio category, frequently bought
alongside [product_C]."

# Flow Chart

User message
│
▼
┌──────────────┐ "I need a moisturiser for sensitive skin"
│ Understand │──► Extract: {属性:肤质, 值:敏感肌} (attribute extraction)
└──────────────┘
│
▼
┌──────────────┐ "What price range are you looking for?"
│ Elicit │──► Ask follow-up question (need clarification)
└──────────────┘
│
▼
┌──────────────┐ item_id: 586711180132, 653561578767...
│ Recommend │──► Rank items from YOUR catalog (retrieval + ranking)
└──────────────┘ (SASRec history + CPT/BART context fusion)
│
▼
┌──────────────┐ "I'd recommend this moisturising toner..."
│ Response │──► Generate natural language reply (generation)
└──────────────┘

openai api -
YOUR_OPENAI_KEYj-...

grok api-
gsk\_...

# Logs 26-02

- No product name is provided, only brand
  Map the products with somewhat realistic name with genrative enrichment
- Fails at high level specification
  'does smartphone has s-pen'
  '8gb ram and 128 gb storage will be adorable for a gaming smartphone. Any matches?'
- 500k not sufficient, change to 2.5M rows

# During full csv training

- Resetup with at least 50 epocs
- At least 25 words synthetic data for richer description for every product
- data should be smart: apple smartphone have lightning/usb c charging 5g, vivo phones some still have 4g type c charging, some have 5g type c gaming phone. Apple laptops are office work not gaming magsafe charging. Asus laptops are gaming laptops some having 16gb RAM 512gb sdd harddisk....etc

# Run

http://localhost:7860

GROQ_API_KEY=your_api_key

cd /Users/ansh/Desktop/e-com
pip install flask faiss-cpu sentence-transformers groq
source .venv/bin/activate
export GROQ_API_KEY=your_api_key
python3 ecom_assistant/app.py
