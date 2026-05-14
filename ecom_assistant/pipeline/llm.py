"""
Groq API wrapper (free tier) — handles the 3 language tasks:

  understand(query, history) → extracted attributes dict
  elicit(query, attributes, history) → clarifying question string
  respond(query, products, history) → natural language recommendation

Free account: https://console.groq.com  → API Keys → Create key
Set via:  export GROQ_API_KEY="YOUR_GROQ_KEY"
"""
import os
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, CHAT_MODEL

try:
    from groq import Groq
except ImportError:
    raise ImportError("Run: pip install groq")

# ── Attribute schema used for function calling (Understand task) ───────────────
EXTRACT_FUNCTION = {
    "name": "extract_product_attributes",
    "description": (
        "Extract structured product search attributes from the user's message. "
        "Return only the fields that are explicitly or implicitly mentioned."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": (
                    "Top-level product category matching catalog values "
                    "(e.g. 'Beauty & Hygiene', 'Kitchen, Garden & Pets')."
                ),
            },
            "sub_category": {
                "type": "string",
                "description": "Sub-category if the user expresses a specific need.",
            },
            "brand": {
                "type": "string",
                "description": "Brand name mentioned by user (lowercase).",
            },
            "price_max": {
                "type": ["number", "string"],
                "description": "Maximum acceptable price in INR rupees.",
            },
            "price_min": {
                "type": ["number", "string"],
                "description": "Minimum acceptable price in INR rupees.",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Quality/feature keywords the user cares about "
                    "e.g. ['organic', 'pleasant fragrance', 'anti-bacterial', 'premium']."
                ),
            },
            "use_case": {
                "type": "string",
                "description": "Intended use or occasion, e.g. 'gaming', 'travel', 'office work'.",
            },
        },
        "required": [],
    },
}


class LLMPipeline:
    def __init__(self):
        api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set.\n"
                "1. Create free account: https://console.groq.com\n"
                "2. Go to API Keys → Create key\n"
                "3. Run:  export GROQ_API_KEY='YOUR_GROQ_KEY'"
            )
        self.client = Groq(api_key=api_key)

    def _chat(self, messages: list[dict], **kw) -> str:
        # Groq doesn't support tool_choice with type="function" the same way;
        # strip unsupported kwargs gracefully
        resp = self.client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, **kw
        )
        return resp.choices[0].message

    # ── Task 1: Understand ─────────────────────────────────────────────────────
    def understand(
        self,
        query: str,
        history: Optional[list[dict]] = None,
    ) -> dict:
        """
        Extract structured attributes from the user's latest message.
        Returns a dict with keys: category, brand, price_max, price_min,
                                   keywords, use_case  (all optional).
        """
        messages = _build_history(history) + [{"role": "user", "content": query}]

        try:
            msg = self._chat(
                messages,
                tools=[{"type": "function", "function": EXTRACT_FUNCTION}],
                tool_choice={"type": "function", "function": {"name": "extract_product_attributes"}},
            )
        except Exception:
            return _heuristic_extract(query)

        if msg.tool_calls:
            try:
                parsed = json.loads(msg.tool_calls[0].function.arguments)
                return _sanitize_attrs(parsed)
            except (json.JSONDecodeError, IndexError):
                pass
        # Fallback: parse from text content if tool call failed
        return _heuristic_extract(query)

    # ── Task 2: Elicit ─────────────────────────────────────────────────────────
    def elicit(
        self,
        query: str,
        attributes: dict,
        history: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate ONE concise clarifying question to fill the most important
        missing attribute.  Returns empty string if nothing is needed.
        """
        missing = []
        if not attributes.get("category"):
            missing.append("product category")
        if not attributes.get("price_max") and not attributes.get("price_min"):
            missing.append("budget / price range")
        if not attributes.get("brand") and not attributes.get("keywords"):
            missing.append("specific requirements or brand preferences")

        if not missing:
            return ""

        system = (
            "You are a helpful e-commerce shopping assistant. "
            "Ask ONE short, natural clarifying question to gather the single most "
            "important missing piece of information to make a good recommendation. "
            "Do NOT ask multiple questions at once. Keep it under 25 words."
        )
        context = (
            f"User said: \"{query}\"\n"
            f"Already known: {json.dumps(attributes) if attributes else 'nothing yet'}\n"
            f"Most important missing info: {missing[0]}"
        )
        messages = [
            {"role": "system", "content": system},
            *_build_history(history),
            {"role": "user",   "content": context},
        ]
        return self._chat(messages, max_tokens=80).content.strip()

    # ── Task 3: Respond ────────────────────────────────────────────────────────
    def respond(
        self,
        query: str,
        products: list[dict],
        history: Optional[list[dict]] = None,
        attributes: Optional[dict] = None,
    ) -> str:
        """
        Generate a natural recommendation reply grounded in *products*.
        Only references products from the provided list (no hallucination).
        """
        if not products:
            return (
                "I'm sorry — I couldn't find products matching your requirements "
                "in the current catalog.  Could you try broadening your search "
                "or adjusting the price range?"
            )

        # Format product list for the prompt
        product_lines = []
        for i, p in enumerate(products, 1):
            name     = p.get("product_name", "Unknown product")
            brand    = (p.get("brand", "") or "N/A").title()
            cat      = p.get("category", "N/A")
            sub_cat  = p.get("sub_category", "N/A")
            price    = float(p.get("sale_price", 0))
            rating   = p.get("rating")
            pid      = p.get("product_id", "N/A")
            product_lines.append(
                f"{i}. [{pid}] {name} | Brand: {brand} | Category: {cat}/{sub_cat} | Price: INR ₹{price:.2f} | Rating: {rating}"
            )
        product_block = "\n".join(product_lines)

        system = (
            "You are a knowledgeable, friendly e-commerce shopping assistant. "
            "Recommend products from the CATALOG LIST ONLY — never invent products. "
            "When mentioning price, always use INR rupees with the ₹ symbol. "
            "Explain briefly why each product fits the user's needs. "
            "Keep the response concise (≤120 words). Use a warm, helpful tone."
        )
        user_msg = (
            f"User request: \"{query}\"\n"
            f"Known attributes: {json.dumps(attributes or {})}\n\n"
            f"CATALOG RESULTS (recommend from these only):\n{product_block}"
        )
        messages = [
            {"role": "system", "content": system},
            *_build_history(history),
            {"role": "user",   "content": user_msg},
        ]
        return self._chat(messages, max_tokens=300).content.strip()


# ── Helper ─────────────────────────────────────────────────────────────────────

def _build_history(history: Optional[list[dict]]) -> list[dict]:
    """Convert conversation history to OpenAI message format."""
    if not history:
        return []
    return [
        {"role": h["role"], "content": h["content"]}
        for h in history[-6:]   # keep last 3 exchanges for context window efficiency
    ]


def _to_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        cleaned = "".join(ch for ch in v if ch.isdigit() or ch in ".-")
        if cleaned:
            try:
                return float(cleaned)
            except ValueError:
                return None
    return None


def _sanitize_attrs(attrs: dict) -> dict:
    out = dict(attrs or {})
    if "price_min" in out:
        out["price_min"] = _to_float(out.get("price_min"))
    if "price_max" in out:
        out["price_max"] = _to_float(out.get("price_max"))
    if out.get("price_min") is None:
        out.pop("price_min", None)
    if out.get("price_max") is None:
        out.pop("price_max", None)
    return out


def _heuristic_extract(query: str) -> dict:
    q = (query or "").lower()
    out = {}
    import re
    m_under = re.search(r"(?:under|below|less than)\s*(?:\$|₹)?\s*(\d+(?:\.\d+)?)", q)
    if m_under:
        out["price_max"] = float(m_under.group(1))
    m_over = re.search(r"(?:above|over|more than)\s*(?:\$|₹)?\s*(\d+(?:\.\d+)?)", q)
    if m_over:
        out["price_min"] = float(m_over.group(1))
    return out
