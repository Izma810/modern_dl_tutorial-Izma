"""
lstm.py — Character-level LSTM implemented from scratch using PyTorch.

You implement all four LSTM gates manually inside forward().
nn.RNN, nn.LSTM, nn.GRU are NOT allowed anywhere in this file.
You may use: nn.Embedding, nn.Linear, torch operations, autograd.

Architecture:
    Embedding → manual LSTM cell (loop over time) → Linear → logits

LSTM cell equations at each time step t:
    Combined input: z_t = x_t @ W_x.T + h_{t-1} @ W_h.T + b

    where W_x ∈ R^{4H × E} and W_h ∈ R^{4H × H} stack all four gate
    weight matrices into a single operation for efficiency:

        [f_pre, i_pre, g_pre, o_pre] = z_t.chunk(4, dim=-1)

    Forget gate:     f_t = sigmoid(f_pre)
    Input gate:      i_t = sigmoid(i_pre)
    Candidate mem:   g_t = tanh(g_pre)
    Output gate:     o_t = sigmoid(o_pre)
    Cell state:      c_t = f_t * c_{t-1} + i_t * g_t
    Hidden state:    h_t = o_t * tanh(c_t)

Using a single fused linear (4H × E and 4H × H) is the standard PyTorch
pattern — it is one matrix multiply instead of four, and it means you only
define two weight matrices (W_x, W_h) and one bias (b) rather than twelve.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    """
    Character-level language model using a manually implemented LSTM cell.

    The LSTM has two state vectors:
        h_t  — hidden state (short-term memory), shape (batch, hidden_dim)
        c_t  — cell state  (long-term memory),  shape (batch, hidden_dim)

    Both are passed between chunks during truncated BPTT. Both must be
    detached before the next chunk to cut the gradient graph.

    Args:
        vocab_size: number of unique characters (V)
        embed_dim:  character embedding dimension
        hidden_dim: LSTM hidden/cell state dimension (H)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Fused LSTM cell weights — do NOT use nn.LSTM
        # W_x maps the input embedding to all four gate pre-activations at once
        # W_h maps the previous hidden state to all four gate pre-activations
        # b   is the shared bias for all four gates
        # TODO: define self.W_x as nn.Linear(embed_dim,  4 * hidden_dim, bias=False)
        #              self.W_h as nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        #              self.b   as nn.Parameter(torch.zeros(4 * hidden_dim))
        # Initialise W_x and W_h with nn.init.xavier_uniform_.
        # Initialise the forget-gate slice of b to 1.0:
        #     self.b.data[hidden_dim : 2 * hidden_dim] = 1.0
        # This biases f_t to start near σ(1)≈0.73 — the standard trick for
        # initialising forget gates to preserve memory by default.
        raise NotImplementedError

        # Output projection
        # TODO: define self.fc as nn.Linear(hidden_dim, vocab_size)
        raise NotImplementedError

    def forward(
        self,
        x:  torch.Tensor,
        hc: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass over a sequence chunk.

        Args:
            x:  integer tensor of shape (batch, seq_len) — character indices
            hc: tuple (h, c) of initial hidden and cell states,
                each shape (batch, hidden_dim).
                If None, both are initialised to zeros.

        Returns:
            logits: float tensor of shape (batch, seq_len, vocab_size)
            (h, c): final hidden and cell states, shape (batch, hidden_dim) each
                    — detach both before the next chunk

        Implementation notes:
            1. Embed x → (batch, seq_len, embed_dim)
            2. Initialise h, c to zeros if hc is None
            3. Loop over t in range(seq_len):
                   x_t  = embeds[:, t, :]
                   z_t  = self.W_x(x_t) + self.W_h(h) + self.b
                   f_pre, i_pre, g_pre, o_pre = z_t.chunk(4, dim=-1)
                   f_t  = torch.sigmoid(f_pre)
                   i_t  = torch.sigmoid(i_pre)
                   g_t  = torch.tanh(g_pre)
                   o_t  = torch.sigmoid(o_pre)
                   c    = f_t * c + i_t * g_t
                   h    = o_t * torch.tanh(c)
                   collect h into a list
            4. Stack → (batch, seq_len, hidden_dim) → apply fc → logits
        """
        # TODO: implement the forward pass following the notes above.
        # Do NOT use nn.LSTM. The loop in step 3 is mandatory.
        raise NotImplementedError

    def init_hidden(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero hidden and cell states of the correct shape."""
        zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
        return zeros, zeros.clone()