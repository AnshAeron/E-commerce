"""CRS Orchestrator for BigBasket-based conversational recommendations."""
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CATALOG_PATH, MAX_ELICIT_TURNS, SEQUENCES_PATH, TOP_K_SHOW

from pipeline.brand_mapper import BrandMapper
from pipeline.llm import LLMPipeline
from pipeline.query_parser import QueryParser
from pipeline.rating_reranker import RatingReranker
from pipeline.recommender import SASRecReranker
from pipeline.retriever import ProductRetriever


@dataclass
class ConversationState:
    history: list[dict] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    elicit_count: int = 0
    user_history: list[str] = field(default_factory=list)
    last_products: list[dict] = field(default_factory=list)
    selected_product_id: str = ""


class CRSPipeline:
    def __init__(self):
        print("[CRS] Loading pipeline ...")
        self.llm = LLMPipeline()
        self.retriever = ProductRetriever()
        self.rating_reranker = RatingReranker()
        self.sasrec = SASRecReranker()
        self.state = ConversationState()

        catalog = json.loads(CATALOG_PATH.read_text()) if CATALOG_PATH.exists() else {}
        categories = {str(v.get("category", "")).strip() for v in catalog.values()}
        sub_categories = {str(v.get("sub_category", "")).strip() for v in catalog.values()}
        brands = {str(v.get("brand", "")).strip() for v in catalog.values()}

        self.query_parser = QueryParser(categories=categories, sub_categories=sub_categories)
        self.brand_mapper = BrandMapper(known_brands=brands)
        self.known_brands = {b.lower() for b in brands if b}

        self.use_sasrec = self._is_sasrec_ready()
        print(f"[CRS] Ready. SASRec enabled: {self.use_sasrec}")

    def _is_sasrec_ready(self) -> bool:
        if not SEQUENCES_PATH.exists():
            return False
        try:
            seqs = json.loads(SEQUENCES_PATH.read_text())
        except json.JSONDecodeError:
            return False
        return len(seqs) >= 500

    def chat(self, user_message: str) -> str:
        state = self.state
        state.history.append({"role": "user", "content": user_message})

        parsed = self.query_parser.parse(user_message)
        if self._looks_like_new_product_query(user_message):
            state.selected_product_id = ""

        if self._is_selected_product_followup(user_message):
            selected = self._get_selected_product()
            if selected:
                reply = self._respond_product_followup(user_message, selected)
                state.history.append({"role": "assistant", "content": reply})
                return reply

        if parsed.get("followup_details") and state.last_products:
            reply = self._respond_product_followup(user_message, state.last_products[0])
            state.history.append({"role": "assistant", "content": reply})
            return reply

        llm_attrs = self.llm.understand(user_message, state.history[:-1])
        merged = self._merge_attributes(state.attributes, llm_attrs, parsed)
        merged["brand"] = self.brand_mapper.normalize(merged.get("brand"))
        state.attributes = merged

        if state.elicit_count < MAX_ELICIT_TURNS:
            question = self.llm.elicit(user_message, state.attributes, state.history[:-1])
            if question:
                if not state.attributes.get("category") and not state.attributes.get("sub_category"):
                    state.elicit_count += 1
                    state.history.append({"role": "assistant", "content": question})
                    return question

        query_text = self._build_query_text(user_message, state.attributes)
        filters = self._build_filters(state.attributes)

        contextual = []
        if self._should_use_shortlist_context(user_message, parsed):
            contextual = self._refine_from_last_products(query_text)
        if contextual:
            top = contextual[:TOP_K_SHOW]
            state.last_products = top
            reply = self.llm.respond(user_message, top, state.history[:-1], state.attributes)
            state.history.append({"role": "assistant", "content": reply})
            return reply

        candidates = self._robust_retrieve(query_text, filters)

        anchor_price = None
        if state.last_products:
            anchor_price = state.last_products[0].get("sale_price")

        ranked = self.rating_reranker.rerank(
            candidates,
            {
                "brand": state.attributes.get("brand"),
                "category": state.attributes.get("category"),
                "price_max": state.attributes.get("price_max"),
                "cheaper_variant": bool(parsed.get("cheaper_variant")),
                "anchor_price": anchor_price,
            },
        )

        if self.use_sasrec:
            ranked = self.sasrec.rerank(ranked, state.user_history or None)

        top = ranked[:TOP_K_SHOW]
        state.last_products = top

        reply = self.llm.respond(user_message, top, state.history[:-1], state.attributes)
        state.history.append({"role": "assistant", "content": reply})
        return reply

    def select_product(self, product_id: str):
        self.state.selected_product_id = (product_id or "").strip()

    def _respond_product_followup(self, user_message: str, product: dict) -> str:
        description = (product.get("description") or "").strip()
        name = product.get("product_name", "this product")
        text = (user_message or "").lower()
        desc_l = description.lower()

        if "alcohol" in text:
            if "alcohol free" in desc_l:
                alcohol_line = "It is described as alcohol free."
            elif "alcohol" in desc_l:
                alcohol_line = "The description references alcohol-related content."
            else:
                alcohol_line = "I do not see an explicit alcohol statement in the description."
        else:
            alcohol_line = ""

        feature_line = ""
        feature_terms = [
            t for t in ["ginger", "fragrance", "scent", "organic", "antibacterial", "tea tree", "caffeine"]
            if t in text
        ]
        if feature_terms:
            present = [t for t in feature_terms if t in desc_l]
            missing = [t for t in feature_terms if t not in desc_l]
            chunks = []
            if present:
                chunks.append("Found in description: " + ", ".join(present) + ".")
            if missing:
                chunks.append("Not clearly mentioned: " + ", ".join(missing) + ".")
            feature_line = " ".join(chunks)

        snippet = _compact_description(description)
        parts = [
            f"Here are the details for {name}.",
            f"{alcohol_line}" if alcohol_line else "",
            f"{feature_line}" if feature_line else "",
            f"Description: {snippet}" if snippet else "Description is limited in the catalog.",
        ]
        return " ".join([p for p in parts if p])

    def _merge_attributes(self, current: dict, llm_attrs: dict, parsed_attrs: dict) -> dict:
        merged = dict(current)
        for source in (parsed_attrs, llm_attrs):
            for key, value in (source or {}).items():
                if value in (None, "", []):
                    continue
                if key == "keywords":
                    existing = merged.get("keywords", [])
                    merged["keywords"] = list(dict.fromkeys(existing + value))
                elif key == "quality_signal":
                    existing = merged.get("quality_signal", [])
                    merged["quality_signal"] = list(dict.fromkeys(existing + value))
                else:
                    merged[key] = value
        return merged

    def _build_query_text(self, user_message: str, attrs: dict) -> str:
        parts = [user_message]
        for k in ("category", "sub_category", "brand", "use_case"):
            v = attrs.get(k)
            if v:
                parts.append(str(v))
        for k in ("keywords", "quality_signal"):
            if attrs.get(k):
                parts.extend(attrs[k])
        return " ".join(parts)

    def _build_filters(self, attrs: dict) -> dict:
        f = {}
        for key in ("category", "sub_category", "price_max", "price_min"):
            if attrs.get(key) is not None and attrs.get(key) != "":
                f[key] = attrs[key]
        brand = (attrs.get("brand") or "").strip().lower()
        if brand and brand in self.known_brands:
            f["brand"] = brand
        if attrs.get("quality_signal"):
            if "premium" in attrs["quality_signal"]:
                f["min_rating"] = 4.0
            if "good" in attrs["quality_signal"] and "min_rating" not in f:
                f["min_rating"] = 3.5
        return f

    def _robust_retrieve(self, query_text: str, filters: dict) -> list[dict]:
        target_k = TOP_K_SHOW * 4
        candidates = self.retriever.search(query_text, filters=filters, top_k=target_k)
        if candidates:
            return candidates

        # If topic filters exist, try lexical fallback inside current topic before dropping it.
        if filters.get("category") or filters.get("sub_category"):
            scoped = self.retriever.keyword_fallback_search(query_text, filters=filters, top_k=target_k)
            if scoped:
                return scoped

        relaxed = dict(filters)
        for key in ("brand", "sub_category", "category", "min_rating"):
            if key in relaxed:
                relaxed.pop(key)
                candidates = self.retriever.search(query_text, filters=relaxed, top_k=target_k)
                if candidates:
                    return candidates

        candidates = self.retriever.search(query_text, filters={}, top_k=target_k)
        if candidates:
            return candidates

        return self.retriever.keyword_fallback_search(query_text, filters=filters, top_k=target_k)

    def reset(self):
        user_hist = self.state.user_history
        self.state = ConversationState(user_history=user_hist)

    def set_user_history(self, product_ids: list[str]):
        self.state.user_history = product_ids

    @property
    def last_products(self) -> list[dict]:
        return self.state.last_products

    def _get_selected_product(self) -> Optional[dict]:
        pid = self.state.selected_product_id
        if not pid:
            return None
        for p in self.state.last_products:
            if str(p.get("product_id", "")) == pid:
                return p
        return self.retriever.catalog.get(pid)

    @staticmethod
    def _is_selected_product_followup(user_message: str) -> bool:
        text = (user_message or "").lower().strip()
        phrase_markers = ["this one", "that one", "is it", "does it", "any alcohol"]
        if any(p in text for p in phrase_markers):
            return True

        word_markers = ["it", "this", "that", "selected", "ingredients", "feature", "contains", "benefits", "fragrance"]
        return any(re.search(rf"\b{re.escape(w)}\b", text) for w in word_markers)

    @staticmethod
    def _should_use_shortlist_context(user_message: str, parsed: dict) -> bool:
        text = (user_message or "").strip().lower()
        if parsed.get("cheaper_variant") or parsed.get("followup_details"):
            return True
        cue_words = ["same", "this", "that", "it", "similar", "instead"]
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in cue_words):
            return True
        return False

    @staticmethod
    def _looks_like_new_product_query(user_message: str) -> bool:
        text = (user_message or "").lower().strip()
        explicit_followup_phrases = ["this one", "that one", "is it", "does it", "same thing", "what's in it"]
        if any(p in text for p in explicit_followup_phrases):
            return False

        # A query mentioning a concrete noun phrase (e.g. "pearl facial kit")
        # should be treated as a fresh retrieval turn, not a selected-item follow-up.
        fresh_cues = [
            "facial", "kit", "face wash", "soap", "tea", "shampoo", "conditioner",
            "toothpaste", "cream", "serum", "oil", "lotion", "deodorant",
        ]
        return any(c in text for c in fresh_cues)

    def _refine_from_last_products(self, query_text: str) -> list[dict]:
        if not self.state.last_products:
            return []
        tokens = [t for t in query_text.lower().split() if len(t) > 2]
        if not tokens:
            return []

        scored = []
        for p in self.state.last_products:
            text = " ".join(
                [
                    str(p.get("product_name", "")),
                    str(p.get("brand", "")),
                    str(p.get("category", "")),
                    str(p.get("sub_category", "")),
                    str(p.get("description", "")),
                ]
            ).lower()
            overlap = sum(1 for t in tokens if t in text)
            if overlap > 0:
                scored.append((overlap, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]


def _compact_description(description: str) -> str:
    text = (description or "").strip()
    if not text:
        return ""
    if len(text) <= 900:
        return text

    cut = text[:900]
    period = cut.rfind(".")
    if period > 200:
        return cut[:period + 1]
    return cut
