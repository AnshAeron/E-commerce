"""
SASRec-based re-ranker.

Given a list of candidate products from FAISS, re-ranks them using the user's
purchase history (sequential recommendation signal).

If no history is available (cold-start) the FAISS order is returned unchanged.
"""
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SASREC_MODEL_PATH, ITEM_INDEX_PATH, SASREC_MAX_SEQ_LEN
from pipeline.sasrec_model import load_model


class SASRecReranker:
    def __init__(self):
        self.device   = torch.device("cpu")
        self.model    = None
        self.item2idx : dict[str, int] = {}
        self.idx2item : dict[int, str] = {}
        self._loaded  = False

    def _lazy_load(self):
        if self._loaded:
            return
        if not SASREC_MODEL_PATH.exists():
            print("[SASRec] model not found — will skip reranking.")
            return

        self.model, self.item2idx, self.device = load_model(self.device)
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        self._loaded  = True

    def rerank(
        self,
        candidates: list[dict],
        user_history: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        candidates    – list of dicts from ProductRetriever.search()
        user_history  – list of product_id strings (most recent last)

        Returns candidates sorted by SASRec score (desc), or unchanged if
        no model / no history.
        """
        self._lazy_load()

        if not self._loaded or not user_history:
            return candidates    # cold-start: keep FAISS order

        # Build integer sequence
        int_seq = [self.item2idx[pid] + 1
                   for pid in user_history
                   if pid in self.item2idx]
        if not int_seq:
            return candidates

        # Pad / trim to max_len
        int_seq = int_seq[-SASREC_MAX_SEQ_LEN:]
        padded  = [0] * (SASREC_MAX_SEQ_LEN - len(int_seq)) + int_seq
        seq_t   = torch.tensor([padded], dtype=torch.long, device=self.device)

        # User representation
        with torch.no_grad():
            user_vec = self.model.predict(seq_t)          # (1, d)

        # Score each candidate
        candidate_pids = [c["product_id"] for c in candidates]
        indices = [
            self.item2idx.get(pid) for pid in candidate_pids
        ]

        scored = []
        for cand, idx in zip(candidates, indices):
            if idx is None:
                scored.append({**cand, "sasrec_score": 0.0})
                continue
            item_t = torch.tensor([idx + 1], dtype=torch.long, device=self.device)
            with torch.no_grad():
                item_vec = self.model.item_emb(item_t)    # (1, d)
            sas_score = float((user_vec * item_vec).sum())
            scored.append({**cand, "sasrec_score": sas_score})

        scored.sort(key=lambda x: x["sasrec_score"], reverse=True)
        return scored
