"""
Step 4 — Train SASRec (Self-Attentive Sequential Recommendation).

Trains a lightweight SASRec model on the purchase sequences extracted from the CSV.
Runs entirely on CPU (MPS optional on Apple Silicon).

Architecture:
  item embedding  →  positional encoding
  →  N × (multi-head self-attention  +  point-wise feed-forward)
  →  dot-product score with item embeddings

Training:
  For each sequence [i1, i2, … iT]:
    - Input  : [i1, … i_{T-1}]
    - Target : [i2, … i_T]
    - Loss   : binary cross-entropy with 1 random negative item per position

Outputs:
  data/sasrec_model.pt    – full model state dict + metadata
"""
import sys
import json
import random
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEQUENCES_PATH, ITEM_INDEX_PATH, SASREC_MODEL_PATH,
    SASREC_MAX_SEQ_LEN, SASREC_EMBED_DIM,
    SASREC_NUM_HEADS, SASREC_NUM_LAYERS,
    SASREC_DROPOUT, SASREC_EPOCHS,
    SASREC_BATCH_SIZE, SASREC_LR,
    SASREC_EARLY_STOP_PATIENCE, SASREC_EARLY_STOP_MIN_DELTA,
)
from pipeline.sasrec_model import SASRec


# ── Dataset ────────────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, sequences: list[list[int]], max_len: int, num_items: int):
        self.samples   = []
        self.max_len   = max_len
        self.num_items = num_items

        for seq in sequences:
            if len(seq) < 2:
                continue
            # All sub-sequences of length 2..len
            for end in range(2, len(seq) + 1):
                sub = seq[:end]
                self.samples.append(sub)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq      = self.samples[idx]
        target   = seq[-1]
        inp      = seq[:-1][-self.max_len:]        # trim to max_len

        # Pad left
        padded   = [0] * (self.max_len - len(inp)) + inp
        neg      = random.randint(1, self.num_items)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(neg,    dtype=torch.long),
        )


# ── Training ───────────────────────────────────────────────────────────────────

def train():
    # Use CPU for training — MPS has known issues with masked MultiheadAttention
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    sequences_raw: dict = json.loads(SEQUENCES_PATH.read_text())
    item2idx:       dict = json.loads(ITEM_INDEX_PATH.read_text())

    # Convert string product_ids to integer indices (1-based, 0=pad)
    int_sequences = []
    for seq in sequences_raw.values():
        int_seq = [item2idx[pid] + 1 for pid in seq if pid in item2idx]
        if len(int_seq) >= 2:
            int_sequences.append(int_seq)

    num_items = len(item2idx)
    print(f"Items: {num_items:,} | Training sequences: {len(int_sequences):,}")

    dataset    = SeqDataset(int_sequences, SASREC_MAX_SEQ_LEN, num_items)
    loader     = DataLoader(dataset, batch_size=SASREC_BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    model = SASRec(
        num_items, SASREC_MAX_SEQ_LEN, SASREC_EMBED_DIM,
        SASREC_NUM_HEADS, SASREC_NUM_LAYERS, SASREC_DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=SASREC_LR)
    bce       = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    print(f"\nTraining SASRec for {SASREC_EPOCHS} epochs …")
    for epoch in range(1, SASREC_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        t0         = time.time()

        for inp_seq, pos_item, neg_item in loader:
            inp_seq  = inp_seq.to(device)
            pos_item = pos_item.to(device)
            neg_item = neg_item.to(device)

            pos_score = model.score(inp_seq, pos_item)
            neg_score = model.score(inp_seq, neg_item)

            loss = bce(pos_score, torch.ones_like(pos_score)) + \
                   bce(neg_score, torch.zeros_like(neg_score))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # prevent NaN exploding gradients
            optimizer.step()
            total_loss += loss.item()

        avg  = total_loss / max(len(loader), 1)
        secs = time.time() - t0
        print(f"  Epoch {epoch:3d}/{SASREC_EPOCHS} | loss={avg:.4f} | {secs:.1f}s")

        if avg < (best_loss - SASREC_EARLY_STOP_MIN_DELTA):
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= SASREC_EARLY_STOP_PATIENCE:
            print(
                f"  Early stopping at epoch {epoch} "
                f"(best_loss={best_loss:.4f}, patience={SASREC_EARLY_STOP_PATIENCE})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save
    torch.save({
        "state_dict": model.state_dict(),
        "num_items":  num_items,
        "max_len":    SASREC_MAX_SEQ_LEN,
        "embed_dim":  SASREC_EMBED_DIM,
        "n_heads":    SASREC_NUM_HEADS,
        "n_layers":   SASREC_NUM_LAYERS,
        "dropout":    SASREC_DROPOUT,
    }, SASREC_MODEL_PATH)

    print(f"\n✅  Model saved → {SASREC_MODEL_PATH}")


if __name__ == "__main__":
    train()
