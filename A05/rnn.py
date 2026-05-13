"""
rnn.py — Character-level RNN implemented from scratch using PyTorch.

You implement the Elman RNN cell manually inside forward().
nn.RNN, nn.LSTM, nn.GRU are NOT allowed anywhere in this file.
You may use: nn.Embedding, nn.Linear, torch operations, autograd.

Architecture:
    Embedding → manual RNN cell (loop over time) → Linear → logits

The RNN cell at each time step t:
    a_t = x_t @ W_xh.T + h_{t-1} @ W_hh.T + b_h
    h_t = tanh(a_t)

where x_t is the embedded input at step t (shape: batch × embed_dim).
The output layer maps each hidden state to vocabulary logits:
    logits_t = h_t @ W_hy.T + b_y
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """
    Character-level language model using a manually implemented RNN cell.

    The forward pass processes a full sequence chunk and returns logits
    for every time step. Cross-entropy loss is computed outside this class
    (in train.py) so that the same model can be used for generation.

    Args:
        vocab_size: number of unique characters (V)
        embed_dim:  character embedding dimension
        hidden_dim: RNN hidden state dimension (m)
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

        # Input embedding: integer index → dense vector
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN cell weights — do NOT use nn.RNN
        # W_xh: maps input embedding to hidden pre-activation
        # W_hh: maps previous hidden state to hidden pre-activation
        # b_h:  hidden bias
        # TODO: define self.W_xh, self.W_hh, self.b_h as nn.Parameter
        # Shapes:
        #   W_xh: (hidden_dim, embed_dim)
        #   W_hh: (hidden_dim, hidden_dim)
        #   b_h:  (hidden_dim,)
        # Initialise with nn.init.xavier_uniform_ for weight matrices,
        # nn.init.zeros_ for biases.
        raise NotImplementedError

        # Output projection: hidden state → vocabulary logits
        # TODO: define self.fc as nn.Linear(hidden_dim, vocab_size)
        raise NotImplementedError

    def forward(
        self,
        x:    torch.Tensor,
        h:    torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over a sequence chunk.

        Args:
            x: integer tensor of shape (batch, seq_len) — character indices
            h: initial hidden state, shape (batch, hidden_dim).
               If None, initialise to zeros.

        Returns:
            logits: float tensor of shape (batch, seq_len, vocab_size)
                    — unnormalised scores over vocabulary at each time step
            h:      final hidden state, shape (batch, hidden_dim)
                    — detach before passing to the next chunk (truncated BPTT)

        Implementation notes:
            1. Embed x → (batch, seq_len, embed_dim)
            2. Initialise h to zeros if None
            3. Loop over t in range(seq_len):
                   x_t = embeds[:, t, :]                 shape (batch, embed_dim)
                   a_t = x_t @ W_xh.T + h @ W_hh.T + b_h  shape (batch, hidden_dim)
                   h   = torch.tanh(a_t)                  shape (batch, hidden_dim)
                   collect h into a list
            4. Stack hidden states → (batch, seq_len, hidden_dim)
            5. Apply self.fc → logits (batch, seq_len, vocab_size)
        """
        # TODO: implement the forward pass following the notes above.
        # Do NOT use nn.RNN. The loop in step 3 is mandatory.
        raise NotImplementedError

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero hidden state of the correct shape."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)