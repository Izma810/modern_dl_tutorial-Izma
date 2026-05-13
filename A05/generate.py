"""Generate samples for A05."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import get_dataloaders
from lstm import CharLSTM
from rnn import CharRNN
from utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate text from A05 models")
    parser.add_argument("--model", choices=["rnn", "lstm"], default="lstm")
    parser.add_argument("--seed", type=str, default="To be")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--seed_value", type=int, default=42)
    return parser.parse_args()


def sample_logits(logits: torch.Tensor, temperature: float) -> int:
    """Sample an index from logits with temperature."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_rnn(model: CharRNN, vocab, seed: str, length: int, temperature: float, device: torch.device) -> str:
    """Generate text from a trained RNN."""
    model.eval()
    indices = vocab.encode(seed)
    h = None

    with torch.no_grad():
        for idx in indices[:-1]:
            x = torch.tensor([[idx]], device=device)
            _, h = model(x, h)
            h = h.detach()

        current = indices[-1]
        out = indices.copy()
        for _ in range(length):
            x = torch.tensor([[current]], device=device)
            logits, h = model(x, h)
            h = h.detach()
            next_idx = sample_logits(logits[0, -1], temperature)
            out.append(next_idx)
            current = next_idx

    return vocab.decode(out)


def generate_lstm(model: CharLSTM, vocab, seed: str, length: int, temperature: float, device: torch.device) -> str:
    """Generate text from a trained LSTM."""
    model.eval()
    indices = vocab.encode(seed)
    hc = None

    with torch.no_grad():
        for idx in indices[:-1]:
            x = torch.tensor([[idx]], device=device)
            _, hc = model(x, hc)
            h, c = hc
            hc = (h.detach(), c.detach())

        current = indices[-1]
        out = indices.copy()
        for _ in range(length):
            x = torch.tensor([[current]], device=device)
            logits, hc = model(x, hc)
            h, c = hc
            hc = (h.detach(), c.detach())
            next_idx = sample_logits(logits[0, -1], temperature)
            out.append(next_idx)
            current = next_idx

    return vocab.decode(out)


def main(args: argparse.Namespace) -> None:
    """Load checkpoint and generate samples."""
    set_seed(args.seed_value)
    device = get_device()
    Path(args.save_dir, "samples").mkdir(parents=True, exist_ok=True)

    _, _, _, vocab = get_dataloaders()

    if args.model == "rnn":
        model = CharRNN(vocab.size).to(device)
        ckpt_path = Path(args.save_dir) / "checkpoints" / "rnn_last.pt"
        load_checkpoint(model, str(ckpt_path), optimizer=None, device=device)
        text = generate_rnn(model, vocab, args.seed, args.length, args.temperature, device)
    else:
        model = CharLSTM(vocab.size).to(device)
        ckpt_path = Path(args.save_dir) / "checkpoints" / "lstm_last.pt"
        load_checkpoint(model, str(ckpt_path), optimizer=None, device=device)
        text = generate_lstm(model, vocab, args.seed, args.length, args.temperature, device)

    suffix = f"{args.model}_T{args.temperature:.2f}.txt"
    out_path = Path(args.save_dir) / "samples" / suffix
    out_path.write_text(text, encoding="utf-8")

    print("=" * 60)
    print(f"Model: {args.model.upper()}  |  Temperature: {args.temperature}")
    print(f"Saved sample to: {out_path}")
    print("=" * 60)
    print(text)


if __name__ == "__main__":
    main(parse_args())
