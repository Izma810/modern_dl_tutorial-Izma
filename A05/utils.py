"""Utility functions for A05."""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_perplexity(avg_cross_entropy: float) -> float:
    """Compute perplexity from average cross-entropy."""
    return math.exp(avg_cross_entropy)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str,
) -> None:
    """Save model and optimizer state to a checkpoint file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )
    print(f"  Saved checkpoint → {path}  (epoch {epoch}, val_loss={val_loss:.4f})")


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load model (and optional optimizer) state from checkpoint."""
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Loaded checkpoint from {path}  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return ckpt


def plot_curves(
    train_losses: list[float],
    val_metrics: list[float],
    save_path: str,
    title: str,
    val_label: str,
) -> None:
    """Plot training loss and validation metric curves."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(train_losses, label="train_loss")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(val_metrics, label=val_label)
    axes[1].set_title(title)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(val_label)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
