"""FAISS-based product retriever for the BigBasket catalog."""
import json
import re
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CATALOG_PATH,
    EMBED_MODEL,
    FAISS_ID_MAP_PATH,
    FAISS_INDEX_PATH,
    TOP_K_RETRIEVE,
    TOP_K_SHOW,
)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}. Run: pip install faiss-cpu sentence-transformers")


class ProductRetriever:
    def __init__(self):
        self._st_model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.id_map: dict[str, str] = json.loads(FAISS_ID_MAP_PATH.read_text())
        self.catalog: dict[str, dict] = json.loads(CATALOG_PATH.read_text())

    def _embed(self, text: str) -> np.ndarray:
        vec = self._st_model.encode([text], normalize_embeddings=True)
        return vec.astype("float32")

    @staticmethod
    def _matches(meta: dict, filters: dict) -> bool:
        brand = (filters.get("brand") or "").lower().strip()
        category = (filters.get("category") or "").lower().strip()
        sub_category = (filters.get("sub_category") or "").lower().strip()
        product_type = (filters.get("type") or "").lower().strip()

        price_max = filters.get("price_max")
        price_min = filters.get("price_min")

        haystack_brand = (meta.get("brand") or "").lower()
        haystack_category = (meta.get("category") or "").lower()
        haystack_sub_category = (meta.get("sub_category") or "").lower()
        haystack_type = (meta.get("type") or "").lower()

        if brand and brand not in haystack_brand:
            return False
        if category and category not in haystack_category:
            return False
        if sub_category and sub_category not in haystack_sub_category:
            return False
        if product_type and product_type not in haystack_type:
            return False

        price = float(meta.get("sale_price") or 0.0)
        if isinstance(price_max, (int, float)) and price > float(price_max):
            return False
        if isinstance(price_min, (int, float)) and price < float(price_min):
            return False

        min_rating = filters.get("min_rating")
        rating = meta.get("rating")
        if isinstance(min_rating, (int, float)):
            if not isinstance(rating, (int, float)):
                return False
            if float(rating) < float(min_rating):
                return False

        return True

    def search(self, query: str, filters: Optional[dict] = None, top_k: int = TOP_K_SHOW) -> list[dict]:
        filters = filters or {}
        vec = self._embed(query)

        k_fetch = min(TOP_K_RETRIEVE * 8, self.index.ntotal)
        scores, indices = self.index.search(vec, k_fetch)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            pid = self.id_map.get(str(idx))
            if not pid:
                continue

            meta = self.catalog.get(pid, {})
            if not self._matches(meta, filters):
                continue

            results.append(self._build_result(pid, meta, float(score)))
            if len(results) >= top_k:
                break

        return results

    def keyword_fallback_search(self, query: str, filters: Optional[dict] = None, top_k: int = TOP_K_SHOW) -> list[dict]:
        filters = filters or {}
        tokens = [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", query.lower()) if t not in {"the", "and", "for", "with", "some", "give", "show"}]
        if not tokens:
            return []

        scored = []
        for pid, meta in self.catalog.items():
            if not self._matches(meta, filters):
                continue

            text = " ".join(
                [
                    str(meta.get("product_name", "")),
                    str(meta.get("brand", "")),
                    str(meta.get("category", "")),
                    str(meta.get("sub_category", "")),
                    str(meta.get("type", "")),
                    str(meta.get("description", "")),
                ]
            ).lower()

            overlap = sum(1 for t in tokens if t in text)
            if overlap <= 0:
                continue

            rating = meta.get("rating")
            rating_bonus = (float(rating) / 5.0) if isinstance(rating, (int, float)) else 0.55
            lexical_score = overlap + rating_bonus
            scored.append((lexical_score, pid, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._build_result(pid, meta, score) for score, pid, meta in scored[:top_k]]

    @staticmethod
    def _build_result(pid: str, meta: dict, score: float) -> dict:
        return {
            "product_id": pid,
            "product_name": meta.get("product_name", "N/A"),
            "brand": meta.get("brand", ""),
            "category": meta.get("category", ""),
            "sub_category": meta.get("sub_category", ""),
            "type": meta.get("type", ""),
            "sale_price": float(meta.get("sale_price") or 0.0),
            "market_price": float(meta.get("market_price") or 0.0),
            "rating": meta.get("rating"),
            "description": meta.get("description", ""),
            "score": float(score),
        }
