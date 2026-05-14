"""
Step 1 — Process BigBasket Products.csv
Outputs:
  data/product_catalog.json    – { product_id: rich metadata }
  data/purchase_sequences.json – synthetic purchase sessions for SASRec
  data/item_id2index.json      – { product_id: int_index }
"""
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CSV_PATH,
    CATALOG_PATH,
    ITEM_INDEX_PATH,
    SEQUENCES_PATH,
    SYNTHETIC_MAX_SEQ_LEN,
    SYNTHETIC_MIN_SEQ_LEN,
    SYNTHETIC_NUM_SESSIONS,
)


def _to_float(value: str, default: float = 0.0) -> float:
    raw = (value or "").strip()
    if not raw:
        return default
    cleaned = re.sub(r"[^0-9.+-]", "", raw)
    if not cleaned:
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def _norm(value: str) -> str:
    return " ".join((value or "").strip().split())


def _read_rows() -> list[dict]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"BigBasket CSV not found at: {CSV_PATH}")

    with CSV_PATH.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    return rows


def _build_catalog(rows: list[dict]) -> tuple[dict, dict, dict]:
    catalog: dict[str, dict] = {}
    by_category: dict[str, list[str]] = defaultdict(list)
    by_brand: dict[str, list[str]] = defaultdict(list)

    for i, row in enumerate(rows, start=1):
        product_name = _norm(row.get("product", ""))
        if not product_name:
            continue

        raw_pid = _norm(row.get("index", ""))
        pid = raw_pid if raw_pid else f"bb_{i}"

        brand = _norm(row.get("brand", "")).lower()
        category = _norm(row.get("category", ""))
        sub_category = _norm(row.get("sub_category", ""))
        product_type = _norm(row.get("type", ""))
        description = _norm(row.get("description", ""))

        sale_price = _to_float(row.get("sale_price", ""), 0.0)
        market_price = _to_float(row.get("market_price", ""), sale_price)
        rating_raw = _to_float(row.get("rating", ""), -1.0)
        rating = round(rating_raw, 2) if rating_raw >= 0 else None

        keywords = [
            product_name,
            brand,
            category,
            sub_category,
            product_type,
            description,
        ]
        search_text = " | ".join([k for k in keywords if k])

        meta = {
            "product_name": product_name,
            "brand": brand,
            "category": category,
            "sub_category": sub_category,
            "type": product_type,
            "sale_price": sale_price,
            "market_price": market_price,
            "rating": rating,
            "description": description,
            "search_text": search_text,
        }
        catalog[pid] = meta

        if category:
            by_category[category].append(pid)
        if brand:
            by_brand[brand].append(pid)

    return catalog, by_category, by_brand


def _weighted_pick(candidates: list[str], catalog: dict[str, dict]) -> str:
    weights = []
    for pid in candidates:
        rating = catalog[pid].get("rating")
        rating_boost = rating if isinstance(rating, (int, float)) else 3.5
        weights.append(max(rating_boost, 1.0))
    return random.choices(candidates, weights=weights, k=1)[0]


def _generate_synthetic_sequences(
    catalog: dict[str, dict],
    by_category: dict[str, list[str]],
    by_brand: dict[str, list[str]],
) -> dict[str, list[str]]:
    random.seed(42)
    pids = list(catalog.keys())
    if not pids:
        return {}

    categories = [c for c in by_category.keys() if by_category[c]]
    sequences: dict[str, list[str]] = {}

    for s in range(SYNTHETIC_NUM_SESSIONS):
        seq_len = random.randint(SYNTHETIC_MIN_SEQ_LEN, SYNTHETIC_MAX_SEQ_LEN)

        start_pid = random.choice(pids)
        base_category = catalog[start_pid].get("category", "")
        base_brand = catalog[start_pid].get("brand", "")

        session_items = [start_pid]
        for _ in range(seq_len - 1):
            pool = []

            # Strong category affinity to simulate coherent shopping missions.
            if base_category and random.random() < 0.72:
                pool.extend(by_category.get(base_category, []))

            # Moderate brand affinity for users loyal to familiar brands.
            if base_brand and random.random() < 0.38:
                pool.extend(by_brand.get(base_brand, []))

            # Occasional cross-category exploration.
            if random.random() < 0.22 and categories:
                alt_cat = random.choice(categories)
                pool.extend(by_category.get(alt_cat, []))

            if not pool:
                pool = pids

            next_pid = _weighted_pick(pool, catalog)
            if next_pid not in session_items:
                session_items.append(next_pid)

        # Ensure a minimum useful sequence length after de-duplication.
        if len(session_items) < 2:
            alt = random.choice(pids)
            if alt != session_items[0]:
                session_items.append(alt)

        sequences[f"synthetic_session_{s + 1:04d}"] = session_items

    return sequences


def main():
    print(f"Reading BigBasket catalog from {CSV_PATH} ...")
    rows = _read_rows()
    print(f"  Loaded {len(rows):,} raw rows")

    catalog, by_category, by_brand = _build_catalog(rows)
    print(f"  Catalog products kept: {len(catalog):,}")
    print(f"  Categories discovered: {len(by_category):,}")
    print(f"  Brands discovered: {len(by_brand):,}")

    sequences = _generate_synthetic_sequences(catalog, by_category, by_brand)
    total_events = sum(len(v) for v in sequences.values())
    print(
        f"  Synthetic sessions: {len(sequences):,} "
        f"| synthetic purchases: {total_events:,}"
    )

    item_id2index = {pid: idx for idx, pid in enumerate(sorted(catalog.keys()))}

    CATALOG_PATH.write_text(json.dumps(catalog, indent=2))
    SEQUENCES_PATH.write_text(json.dumps(sequences, indent=2))
    ITEM_INDEX_PATH.write_text(json.dumps(item_id2index, indent=2))

    print("\nSaved:")
    print(f"  {CATALOG_PATH}")
    print(f"  {SEQUENCES_PATH}")
    print(f"  {ITEM_INDEX_PATH}")


if __name__ == "__main__":
    main()
