# A05 — Recurrent Neural Networks from Scratch (PyTorch)

## Overview

This assignment builds a character-level language model using manual RNN and
LSTM cells in PyTorch. You may **not** use `nn.RNN`, `nn.LSTM`, or `nn.GRU`.
You must implement the recurrence manually inside your own `nn.Module`.

The task is language modeling on **tinyshakespeare** (Karpathy). Given a
sequence of characters, predict the next character at each time step. You will
train both a vanilla RNN and an LSTM, compare perplexity, and generate text to
see the difference in long-range coherence.

This assignment bridges into A06 (Attention from Scratch): you will see how
long-range dependency failures motivate attention.

---

## Learning objectives

- Implement an Elman RNN cell manually in PyTorch
- Implement an LSTM cell manually in PyTorch
- Understand truncated BPTT and hidden-state detachment
- Use cross-entropy loss and perplexity for sequence modeling
- Generate text with temperature sampling

---

## Dataset

We use the standard tinyshakespeare corpus from Karpathy’s char-rnn repo
(~1.1MB, ~1M characters). `data.py` downloads and prepares the data.

- URL: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Vocabulary size: ~65 characters
- Split: 80% train / 10% val / 10% test

---

## Task

**Part 1 — Vanilla RNN (manual cell)**

Implement the Elman RNN recurrence and train a character-level model.
Observe failure modes on long-range structure.

**Part 2 — LSTM (manual cell)**

Implement a full LSTM cell and train on the same data. Compare perplexities
and generated samples.

---

## Metrics

- **Cross-entropy loss** (per character)
- **Perplexity**: `exp(avg_cross_entropy)`

Perplexity equal to the vocabulary size means the model is no better than
random guessing. Good character models typically reach PPL in low single digits.

---

## File structure

```
A05/
├── README.md
├── data.py            # download + vocab + dataloaders
├── rnn.py             # CharRNN: manual RNN cell (TODOs)
├── lstm.py            # CharLSTM: manual LSTM cell (TODOs)
├── train.py           # training loop scaffold (TODOs)
├── evaluate.py        # compute test perplexity
├── generate.py        # sample text to ./outputs/samples
├── utils.py           # helpers (seed, plots, save/load)
└── gradient_check.py  # verify manual RNN cell gradients
```

---

## What to implement (TODOs only here)

- `rnn.py`: RNN parameters + forward pass
- `lstm.py`: LSTM parameters + forward pass
- `train.py`: core training step inside the batch loop

All other files are fully implemented and should not require edits.

---

## Training details

- **Truncated BPTT**: sequences are chunked into windows (e.g. 100 chars)
- **Hidden state detachment**: detach hidden state between chunks
- **Teacher forcing**: always predict next character from true input

---

## Expected outputs

All outputs save under `./outputs` by default:

- `./outputs/checkpoints/rnn_last.pt`
- `./outputs/checkpoints/lstm_last.pt`
- `./outputs/plots/rnn_curves.png`
- `./outputs/plots/lstm_curves.png`
- `./outputs/samples/rnn_T0.80.txt`
- `./outputs/samples/lstm_T0.80.txt`

---

## Recommended run order

1. `python gradient_check.py`
2. `python train.py --model rnn`
3. `python train.py --model lstm`
4. `python evaluate.py`
5. `python generate.py --model lstm --seed "To be" --temperature 0.8`

---

## Notes

1. **No high-level RNNs allowed.** Using `nn.RNN`, `nn.LSTM`, or `nn.GRU`
   anywhere in this assignment is incorrect.
2. **RNN vs LSTM:** LSTM gating should yield lower perplexity and more
   coherent samples at higher temperatures.
3. **Bridge to attention:** even LSTMs struggle to retain very long-range
   dependencies, motivating attention in A06.
