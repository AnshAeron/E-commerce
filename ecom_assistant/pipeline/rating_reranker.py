"""Rating-aware re-ranker for retrieval candidates."""
from __future__ import annotations


class RatingReranker:
    def rerank(self, candidates: list[dict], context: dict | None = None) -> list[dict]:
        context = context or {}
        if not candidates:
            return []

        max_semantic = max((float(c.get("score", 0.0)) for c in candidates), default=1.0)
        max_semantic = max(max_semantic, 1e-6)

        target_brand = (context.get("brand") or "").lower().strip()
        target_category = (context.get("category") or "").lower().strip()
        price_max = context.get("price_max")
        cheaper_variant = bool(context.get("cheaper_variant"))
        anchor_price = context.get("anchor_price")

        reranked = []
        for c in candidates:
            rating = c.get("rating")
            rating_score = (float(rating) / 5.0) if isinstance(rating, (int, float)) else 0.58

            semantic = float(c.get("score", 0.0)) / max_semantic
            brand_score = 1.0 if target_brand and target_brand in (c.get("brand") or "") else 0.0
            category_score = 1.0 if target_category and target_category in (c.get("category") or "").lower() else 0.0

            price = float(c.get("sale_price") or 0.0)
            price_score = 0.0
            if isinstance(price_max, (int, float)) and price > 0:
                price_score = 1.0 if price <= float(price_max) else -0.5

            cheaper_bonus = 0.0
            if cheaper_variant and isinstance(anchor_price, (int, float)) and anchor_price > 0 and price > 0:
                if price < anchor_price:
                    cheaper_bonus = min((anchor_price - price) / anchor_price, 1.0)
                else:
                    cheaper_bonus = -0.4

            final = (
                0.40 * semantic +
                0.28 * rating_score +
                0.12 * brand_score +
                0.10 * category_score +
                0.06 * price_score +
                0.04 * cheaper_bonus
            )

            reranked.append({**c, "rank_score": round(final, 6)})

        reranked.sort(key=lambda x: x["rank_score"], reverse=True)
        return reranked
