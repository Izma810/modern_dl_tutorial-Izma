"""Evaluation script for A05."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import get_dataloaders
from lstm import CharLSTM
from rnn import CharRNN
from utils import compute_perplexity, get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate A05 checkpoints")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    return parser.parse_args()


def eval_model(model: nn.Module, loader, device: torch.device) -> float:
    """Compute average cross-entropy loss over dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    total = 0
    h_state = None
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if isinstance(model, CharLSTM):
                logits, h_state = model(x, h_state)
                h, c = h_state
                h_state = (h.detach(), c.detach())
            else:
                logits, h_state = model(x, h_state)
                h_state = h_state.detach()

            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss_sum += loss.item() * y.numel()
            total += y.numel()
    return loss_sum / max(total, 1)


def main(args: argparse.Namespace) -> None:
    """Evaluate RNN and LSTM checkpoints on the test split."""
    device = get_device()
    _, _, test_loader, vocab = get_dataloaders(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    rnn = CharRNN(vocab.size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    lstm = CharLSTM(vocab.size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)

    rnn_path = Path(args.save_dir) / "checkpoints" / "rnn_last.pt"
    lstm_path = Path(args.save_dir) / "checkpoints" / "lstm_last.pt"
    load_checkpoint(rnn, str(rnn_path), optimizer=None, device=device)
    load_checkpoint(lstm, str(lstm_path), optimizer=None, device=device)

    rnn_loss = eval_model(rnn, test_loader, device)
    lstm_loss = eval_model(lstm, test_loader, device)

    print("=" * 60)
    print("Evaluation report")
    print("=" * 60)
    print(f"RNN  test loss: {rnn_loss:.4f}  PPL: {compute_perplexity(rnn_loss):.2f}")
    print(f"LSTM test loss: {lstm_loss:.4f}  PPL: {compute_perplexity(lstm_loss):.2f}")


if __name__ == "__main__":
    main(parse_args())
