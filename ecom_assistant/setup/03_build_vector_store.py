"""
Step 3 — Build FAISS vector store using local sentence-transformers.

Embeds all product descriptions with all-MiniLM-L6-v2 (runs locally, 100% free).
Model is ~90MB and downloads once to ~/.cache/huggingface/

No API key required.

Outputs:
  data/faiss.index         – FAISS binary index
  data/faiss_id_map.json   – { "0": product_id, "1": product_id, … }
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DESCRIPTIONS_PATH, FAISS_INDEX_PATH, FAISS_ID_MAP_PATH, EMBED_MODEL,
)

try:
    import faiss
except ImportError:
    sys.exit("faiss-cpu not installed. Run:  pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("sentence-transformers not installed. Run:  pip install sentence-transformers")


BATCH_SIZE = 512


def main():
    print(f"Loading descriptions from {DESCRIPTIONS_PATH} …")
    descriptions: dict = json.loads(DESCRIPTIONS_PATH.read_text())
    pids  = list(descriptions.keys())
    texts = [descriptions[p] for p in pids]
    total = len(texts)
    print(f"  {total:,} products to embed")

    print(f"\nLoading local embedding model: {EMBED_MODEL}")
    print("  (downloads ~90 MB on first run, cached afterwards — no API key needed)")
    model = SentenceTransformer(EMBED_MODEL)
    dim   = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}")

    print(f"\nEmbedding {total:,} descriptions …")
    matrix = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → cosine via inner product
    ).astype("float32")

    print(f"  ✓ Embeddings shape: {matrix.shape}")

    # ── Build FAISS index ──────────────────────────────────────────────────────
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    id_map = {str(i): pid for i, pid in enumerate(pids)}
    FAISS_ID_MAP_PATH.write_text(json.dumps(id_map))

    print(f"\n✅  FAISS index  → {FAISS_INDEX_PATH}  ({index.ntotal:,} vectors, dim={dim})")
    print(f"✅  ID map       → {FAISS_ID_MAP_PATH}")


if __name__ == "__main__":
    main()
