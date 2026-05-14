"""
Step 2 — Build retrieval descriptions from rich BigBasket metadata.

Each product already has name/category/brand/type/price/rating/description.
This step composes those fields into a compact retrieval text optimized for
semantic embedding and follow-up grounding.

Outputs:
  data/product_descriptions.json  – {pid: "retrieval text"}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CATALOG_PATH, DESCRIPTIONS_PATH


def _price_band(price: float) -> str:
    if price <= 0:
        return "unknown"
    if price <= 100:
        return "budget"
    if price <= 300:
        return "value"
    if price <= 800:
        return "premium"
    return "luxury"


def build_description(pid: str, meta: dict) -> str:
    name = (meta.get("product_name") or "").strip()
    brand = (meta.get("brand") or "").strip()
    category = (meta.get("category") or "").strip()
    sub_category = (meta.get("sub_category") or "").strip()
    product_type = (meta.get("type") or "").strip()
    raw_description = (meta.get("description") or "").strip()

    sale_price = float(meta.get("sale_price") or 0.0)
    market_price = float(meta.get("market_price") or sale_price)
    rating = meta.get("rating")

    price_sentence = (
        f"Sale price {sale_price:.2f}; market price {market_price:.2f}; "
        f"price band {_price_band(sale_price)}."
    )
    rating_sentence = (
        f"Rating {float(rating):.1f}/5." if isinstance(rating, (float, int)) else "Rating unavailable."
    )

    parts = [
        f"Product ID {pid}.",
        f"Name: {name}.",
        f"Brand: {brand or 'unknown'}.",
        f"Category: {category}.",
        f"Sub-category: {sub_category or 'unknown'}.",
        f"Type: {product_type or 'unknown'}.",
        price_sentence,
        rating_sentence,
    ]
    if raw_description:
        parts.append(f"Details: {raw_description}")
    return " ".join(parts)


def main():
    print(f"Loading catalog from {CATALOG_PATH} ...")
    catalog: dict = json.loads(CATALOG_PATH.read_text())
    print(f"  {len(catalog):,} products to describe")

    descriptions = {}
    for pid, meta in catalog.items():
        descriptions[pid] = build_description(pid, meta)

    DESCRIPTIONS_PATH.write_text(json.dumps(descriptions, indent=2))
    print(f"\nSaved {len(descriptions):,} descriptions -> {DESCRIPTIONS_PATH}")


if __name__ == "__main__":
    main()
