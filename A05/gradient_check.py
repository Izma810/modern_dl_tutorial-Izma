"""
gradient_check.py — verify manual RNN cell gradients against autograd.

Run:
    python gradient_check.py
"""

from __future__ import annotations

import torch


def manual_rnn_step(x: torch.Tensor, h: torch.Tensor, W_xh: torch.Tensor, W_hh: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    a_t = x @ W_xh.T + h @ W_hh.T + b_h
    return torch.tanh(a_t)


def main() -> None:
    torch.manual_seed(0)
    batch = 2
    embed_dim = 4
    hidden_dim = 3

    x = torch.randn(batch, embed_dim, requires_grad=False)
    h = torch.randn(batch, hidden_dim, requires_grad=False)

    W_xh = torch.randn(hidden_dim, embed_dim, requires_grad=True)
    W_hh = torch.randn(hidden_dim, hidden_dim, requires_grad=True)
    b_h = torch.randn(hidden_dim, requires_grad=True)

    out = manual_rnn_step(x, h, W_xh, W_hh, b_h)
    loss = out.sum()
    loss.backward()

    def finite_diff(param: torch.Tensor, idx: tuple[int, ...], eps: float = 1e-5) -> float:
        with torch.no_grad():
            orig = param[idx].item()
            param[idx] = orig + eps
            plus = manual_rnn_step(x, h, W_xh, W_hh, b_h).sum().item()
            param[idx] = orig - eps
            minus = manual_rnn_step(x, h, W_xh, W_hh, b_h).sum().item()
            param[idx] = orig
        return (plus - minus) / (2 * eps)

    max_rel = 0.0
    for name, param in [("W_xh", W_xh), ("W_hh", W_hh), ("b_h", b_h)]:
        grad = param.grad
        for idx in [(0, 0), (min(1, param.shape[0] - 1), 0)]:
            if param.ndim == 1:
                idx = (idx[0],)
            num = finite_diff(param, idx)
            ana = grad[idx].item()
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            max_rel = max(max_rel, rel)
            print(f"{name}{idx}: analytic={ana:.6f}, numeric={num:.6f}, rel={rel:.2e}")

    print(f"\nMax relative error: {max_rel:.2e}")
    if max_rel < 1e-4:
        print("PASSED")
    else:
        print("FAILED")


if __name__ == "__main__":
    main()
