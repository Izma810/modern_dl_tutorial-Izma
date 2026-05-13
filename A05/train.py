"""Training scaffold for A05."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import get_dataloaders
from lstm import CharLSTM
from rnn import CharRNN
from utils import compute_perplexity, get_device, plot_curves, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train A05 RNN/LSTM on tinyshakespeare")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, choices=["rnn", "lstm"], default="rnn")
    return parser.parse_args()


def run_eval(model: nn.Module, loader, device: torch.device) -> float:
    """Compute validation loss."""
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
    """Run training loop scaffold."""
    set_seed(args.seed)
    device = get_device()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, vocab = get_dataloaders(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    if args.model == "lstm":
        model: nn.Module = CharLSTM(vocab.size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    else:
        model = CharRNN(vocab.size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    val_ppls: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        h_state = None

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # TODO: zero gradients with optimizer.zero_grad()
            # TODO: forward pass to get logits, manage hidden state detachment
            # TODO: compute loss with criterion
            # TODO: loss.backward() and optimizer.step()
            raise NotImplementedError(
                "Fill in the core training step inside the loop."
            )

        train_loss = running_loss / max(seen, 1)
        val_loss = run_eval(model, val_loader, device)
        val_ppl = compute_perplexity(val_loss)

        train_losses.append(train_loss)
        val_ppls.append(val_ppl)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Val PPL: {val_ppl:.2f}"
        )

        ckpt_path = Path(args.save_dir) / "checkpoints"
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            str(ckpt_path / f"{args.model}_last.pt"),
        )

    plot_curves(
        train_losses,
        val_ppls,
        str(Path(args.save_dir) / "plots" / f"{args.model}_curves.png"),
        title="Validation Perplexity",
        val_label="PPL",
    )


if __name__ == "__main__":
    main(parse_args())
