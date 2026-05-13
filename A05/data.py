"""
data.py — dataset preparation for character-level language modelling.

Downloads tinyshakespeare.txt (~1MB), builds a character vocabulary,
encodes the full text as integer indices, and returns DataLoaders that
yield (input_chunk, target_chunk) pairs for truncated BPTT.

Usage:
    python data.py          # downloads data, prints dataset statistics
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path("data/tinyshakespeare.txt")

# Train / validation / test split ratios
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# TEST_FRAC  = 0.10  (implicit remainder)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_data(url: str = DATA_URL, path: Path = DATA_PATH) -> None:
    """
    Download tinyshakespeare.txt if it does not already exist.

    Args:
        url:  source URL
        path: local destination path
    """
    if path.exists():
        print(f"Data already exists at {path}  ({path.stat().st_size / 1e6:.2f} MB)")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}  ({path.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """
    Character-level vocabulary built from a text string.

    Attributes:
        chars:    sorted list of unique characters
        size:     vocabulary size (number of unique characters)
        char2idx: dict mapping character → integer index
        idx2char: dict mapping integer index → character
    """

    def __init__(self, text: str) -> None:
        self.chars    = sorted(set(text))
        self.size     = len(self.chars)
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer indices."""
        return [self.char2idx[c] for c in text]

    def decode(self, indices: list[int] | torch.Tensor) -> str:
        """Convert a list (or 1-D tensor) of integer indices back to a string."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return "".join(self.idx2char[i] for i in indices)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Sliding-window character dataset for language modelling.

    Each item is a pair (input_chunk, target_chunk) where:
        input_chunk:  integer tensor of length seq_len
        target_chunk: integer tensor of length seq_len, shifted one step ahead

    So target_chunk[t] is the character that follows input_chunk[t].

    Args:
        encoded: 1-D tensor of integer character indices (the full split)
        seq_len: length of each training chunk (truncated BPTT window)
        stride:  step between consecutive windows (default = seq_len, no overlap)
    """

    def __init__(
        self,
        encoded: torch.Tensor,
        seq_len: int = 100,
        stride:  int | None = None,
    ) -> None:
        self.encoded = encoded
        self.seq_len = seq_len
        self.stride  = stride if stride is not None else seq_len

        # Number of complete windows that fit in the data
        # Each window needs seq_len+1 characters (input + target)
        self.n_windows = (len(encoded) - seq_len - 1) // self.stride + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end   = start + self.seq_len
        x = self.encoded[start : end]
        y = self.encoded[start + 1 : end + 1]
        return x, y


# ---------------------------------------------------------------------------
# Build DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    seq_len:    int = 100,
    batch_size: int = 64,
    stride:     int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    Download data, build vocabulary, split, and return DataLoaders.

    Split sizes (of total ~1M characters):
        train: 80%   (~800K chars)
        val:   10%   (~100K chars)
        test:  10%   (~100K chars)

    Args:
        seq_len:    TBPTT chunk length (number of characters per training step)
        batch_size: mini-batch size
        stride:     step between windows; defaults to seq_len (non-overlapping)

    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    download_data()

    text = DATA_PATH.read_text(encoding="utf-8")
    vocab = Vocabulary(text)
    encoded = torch.tensor(vocab.encode(text), dtype=torch.long)

    # Split indices
    n      = len(encoded)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_enc = encoded[:n_train]
    val_enc   = encoded[n_train : n_train + n_val]
    test_enc  = encoded[n_train + n_val :]

    train_ds = CharDataset(train_enc, seq_len, stride)
    val_ds   = CharDataset(val_enc,   seq_len, stride)
    test_ds  = CharDataset(test_enc,  seq_len, stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, vocab


# ---------------------------------------------------------------------------
# Main — dataset statistics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        seq_len=100, batch_size=64
    )

    print(f"\nVocabulary size : {vocab.size}")
    print(f"Characters      : {''.join(repr(c) for c in vocab.chars[:20])} ...")
    print(f"\nDataset splits  :")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    # Peek at one batch
    x_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch shapes    : x={x_batch.shape}, y={y_batch.shape}")
    print(f"  Sample input  : {repr(vocab.decode(x_batch[0][:40].tolist()))}")
    print(f"  Sample target : {repr(vocab.decode(y_batch[0][:40].tolist()))}")

    # Sanity: target is input shifted by 1
    assert torch.equal(x_batch[0][1:], y_batch[0][:-1]), \
        "Target should be input shifted one step ahead"
    print("\nSanity check PASSED — target is input shifted by 1")