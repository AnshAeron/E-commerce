"""Rule-based query parsing for price/category/quality cues and follow-up intents."""
from __future__ import annotations

import re


class QueryParser:
    def __init__(self, categories: set[str] | None = None, sub_categories: set[str] | None = None):
        self.categories = {c.lower() for c in (categories or set()) if c}
        self.sub_categories = {c.lower() for c in (sub_categories or set()) if c}
        self.intent_phrase_map = {
            "after shave": {"sub_category": "men's grooming", "keywords": ["after", "shave", "fragrance"]},
            "after-shave": {"sub_category": "men's grooming", "keywords": ["after", "shave", "fragrance"]},
            "face wash": {"sub_category": "skin care", "keywords": ["face", "wash", "cleanser"]},
            "hand wash": {"sub_category": "bath & hand wash", "keywords": ["hand", "wash"]},
            "deodorant": {"sub_category": "fragrances & deos", "keywords": ["deodorant", "fragrance"]},
            "shampoo": {"sub_category": "hair care", "keywords": ["shampoo", "hair"]},
            "conditioner": {"sub_category": "hair care", "keywords": ["conditioner", "hair"]},
            "toothpaste": {"sub_category": "oral care", "keywords": ["toothpaste", "oral"]},
        }

        self.quality_keywords = {
            "good": "good",
            "best": "premium",
            "premium": "premium",
            "organic": "organic",
            "branded": "branded",
            "nice smell": "fragrant",
            "fragrance": "fragrant",
            "long lasting": "long_lasting",
            "anti bacterial": "anti_bacterial",
            "antibacterial": "anti_bacterial",
        }

    def parse(self, text: str) -> dict:
        q = (text or "").strip().lower()
        out = {
            "intent": "search",
            "price_max": None,
            "price_min": None,
            "category": "",
            "sub_category": "",
            "quality_signal": [],
            "keywords": [],
            "followup_details": False,
            "cheaper_variant": False,
        }

        if not q:
            return out

        if re.search(r"\b(what'?s in it|ingredients|any alcohol|is it alcohol free)\b", q):
            out["intent"] = "product_detail"
            out["followup_details"] = True

        if re.search(r"\b(same thing but cheaper|cheaper|lower price|budget version)\b", q):
            out["intent"] = "refine"
            out["cheaper_variant"] = True

        m_under = re.search(r"(?:under|below|less than)\s*(?:\$|₹)?\s*(\d+(?:\.\d+)?)", q)
        if m_under:
            out["price_max"] = float(m_under.group(1))

        m_over = re.search(r"(?:above|over|more than)\s*(?:\$|₹)?\s*(\d+(?:\.\d+)?)", q)
        if m_over:
            out["price_min"] = float(m_over.group(1))

        for key, signal in self.quality_keywords.items():
            if key in q:
                out["quality_signal"].append(signal)

        for phrase, mapped in self.intent_phrase_map.items():
            if phrase in q:
                if not out["sub_category"]:
                    out["sub_category"] = mapped.get("sub_category", "")
                out["keywords"].extend(mapped.get("keywords", []))

        # Category and sub-category hints from known taxonomies.
        for category in self.categories:
            if category in q:
                out["category"] = category
                break

        for sub_category in self.sub_categories:
            if sub_category in q:
                out["sub_category"] = sub_category
                break

        # Generic fallback keywords to enrich retrieval when taxonomy does not match.
        raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", q)
        stop = {
            "what", "with", "from", "that", "this", "have", "need", "want",
            "show", "find", "same", "thing", "cheaper", "price", "under",
            "good", "best", "nice", "any", "for", "and", "the",
        }
        token_keywords = [t for t in raw_tokens if t not in stop][:8]
        out["keywords"] = list(dict.fromkeys(out["keywords"] + token_keywords))[:10]

        return out
