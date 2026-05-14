"""
SASRec model definition and load helper.
Shared by setup/04_train_sasrec.py (training) and pipeline/recommender.py (inference).
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SASREC_MODEL_PATH, ITEM_INDEX_PATH,
    SASREC_MAX_SEQ_LEN, SASREC_EMBED_DIM,
    SASREC_NUM_HEADS, SASREC_NUM_LAYERS, SASREC_DROPOUT,
)


class PointWiseFeedForward(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 4, d), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x + self.net(x))


class SASRec(nn.Module):
    def __init__(self, num_items: int, max_len: int, d: int,
                 n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, d, padding_idx=0)  # 0 = pad
        self.pos_emb  = nn.Embedding(max_len, d)
        self.emb_drop = nn.Dropout(dropout)

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.attn_norms  = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ff_layers   = nn.ModuleList([
            PointWiseFeedForward(d, dropout) for _ in range(n_layers)
        ])
        self.out_norm    = nn.LayerNorm(d)
        self.max_len     = max_len
        self.d           = d

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        item_seq : (B, L) int64  item indices (0 = pad)
        returns  : (B, L, d)    contextual representations
        """
        B, L = item_seq.shape
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0)

        x = self.emb_drop(self.item_emb(item_seq) + self.pos_emb(positions))

        # Causal float mask — do NOT pass key_padding_mask to avoid NaN when
        # entire rows are masked (padding positions have zero embeddings anyway).
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=item_seq.device), diagonal=1
        )

        for attn, norm, ff in zip(self.attn_layers, self.attn_norms, self.ff_layers):
            residual = x
            attn_out, _ = attn(x, x, x, attn_mask=causal_mask)
            x = norm(residual + attn_out)
            x = ff(x)

        return self.out_norm(x)

    def predict(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Returns (B, d) representation of the last real (non-pad) item."""
        h = self.forward(item_seq)
        lengths = (item_seq != 0).sum(dim=1).clamp(min=1) - 1
        idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d)
        return h.gather(1, idx).squeeze(1)

    def score(self, item_seq: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        h = self.predict(item_seq)
        w = self.item_emb(item_ids)
        return (h * w).sum(dim=-1)


def load_model(device=None) -> tuple["SASRec", dict, torch.device]:
    """Load a trained SASRec checkpoint.  Returns (model, item2idx, device)."""
    if device is None:
        device = torch.device("cpu")
    ckpt  = torch.load(SASREC_MODEL_PATH, map_location=device, weights_only=False)
    model = SASRec(
        ckpt["num_items"], ckpt["max_len"], ckpt["embed_dim"],
        ckpt["n_heads"],   ckpt["n_layers"], ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    item2idx: dict = json.loads(ITEM_INDEX_PATH.read_text())
    return model, item2idx, device
