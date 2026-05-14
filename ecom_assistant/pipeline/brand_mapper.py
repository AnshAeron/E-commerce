"""Brand normalization utilities for noisy user brand mentions."""
from __future__ import annotations

from difflib import get_close_matches


class BrandMapper:
    def __init__(self, known_brands: set[str] | None = None):
        self.known_brands = {b.lower().strip() for b in (known_brands or set()) if b}
        self.aliases = {
            "p&g": "procter & gamble",
            "bb": "bb home",
            "h&s": "head & shoulders",
            "st botanica": "stbotanica",
        }

    def normalize(self, raw_brand: str | None) -> str:
        if not raw_brand:
            return ""

        cleaned = raw_brand.lower().strip()
        if not cleaned:
            return ""

        if cleaned in self.known_brands:
            return cleaned

        if cleaned in self.aliases:
            mapped = self.aliases[cleaned]
            if mapped in self.known_brands:
                return mapped
            return cleaned

        # Substring fallback for phrases like "from gillette".
        for brand in self.known_brands:
            if brand in cleaned or cleaned in brand:
                return brand

        # Soft fuzzy fallback for typos.
        match = get_close_matches(cleaned, list(self.known_brands), n=1, cutoff=0.82)
        return match[0] if match else cleaned
