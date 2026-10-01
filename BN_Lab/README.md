# AI Laboratory — Bayesian Networks and Autoregressive Language Models

**Student:** Pranay Ghumatkar | **Roll No:** 2024A3PS0328G
**Worksheet:** [bn_lab.pdf](bn_lab.pdf)

First-order (bigram) and second-order (trigram) autoregressive language models
estimated from transition counts — no ML library, no pretrained model — used to
explain how an autoregressive language model is a Bayesian network.

## Files

| File | Purpose |
|---|---|
| `first_order_lm.py` | First-order model `P(X_t \| X_{t-1})`: counts, CPT, greedy/sampling generation, normalisation |
| `second_order_lm.py` | Second-order model `P(X_t \| X_{t-2}, X_{t-1})`: triple counts and first-order back-off |
| `ANSWERS.md` | Answers to Questions 1–14 and the reflection |
| `results.txt` | Output of both models (CPTs, normalisation, generated text) |
| `bn_lab.pdf` | Worksheet |

## How to run

```bash
python first_order_lm.py
python second_order_lm.py
```

Requires only the Python standard library.

See [ANSWERS.md](ANSWERS.md) for the full answers.
