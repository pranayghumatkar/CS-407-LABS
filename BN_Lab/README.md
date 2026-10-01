# AI Laboratory — Bayesian Networks and Autoregressive Language Models

**Student:** Pranay Ghumatkar
**Roll No:** 2024A3PS0328G
**Worksheet:** [bn_lab.pdf](bn_lab.pdf)

Implements first-order (bigram) and second-order (trigram) autoregressive
language models from transition counts — no ML library, no pretrained model —
and uses them to explain how an autoregressive language model is a Bayesian
network.

**Central idea.** The chain rule factorises the joint probability of a sentence:

```
P(x1, …, xT) = P(x1) · Π_{t=2..T} P(xt | x1, …, x_{t-1})
```

A first-order model assumes the Markov property
`P(X_t | X_1..X_{t-1}) = P(X_t | X_{t-1})`, giving the Bayesian network
`X1 → X2 → X3 → …`. A second-order model uses `P(X_t | X_{t-2}, X_{t-1})` with
structure `X_{t-2} → X_t ← X_{t-1}`.

## Files

| File | Purpose |
|---|---|
| `first_order_lm.py` | First-order model: `P(X_t \| X_{t-1})`, counts, CPT, greedy/sampling generation, normalisation check |
| `second_order_lm.py` | Second-order model: `P(X_t \| X_{t-2}, X_{t-1})`, counts of triples, first-order back-off |
| `run_all.py` | Runs both models and writes every file in `results/` |
| `tests/test_normalisation.py` | Probabilistic-invariant tests (`Σ_v P(v\|w) = 1`, ranges, counts, determinism) |
| `ANSWERS.md` | Answers to Questions 1–14 |
| `REFLECTION.md` | How the LLM was used and how its output was validated |
| `results/` | CPTs, normalisation tests, generated text and the first- vs second-order comparison |
| `bn_lab.pdf` | Worksheet |

## How to run

```bash
cd BN_Lab

# Run both models and regenerate every file in results/
python run_all.py

# Run the probabilistic-invariant tests
python tests/test_normalisation.py      # or: python -m pytest tests/ -v
```

Requires only the Python standard library (developed with Python 3.14).

## Headline results

- Training data: six sentences, all lower-cased, tokenised on whitespace, wrapped
  in `<START>` / `<END>`.
- **First-order model:** 17 non-zero parameters over 11 context rows. Row for
  `the`: `cat 0.25, dog 0.25, mat 0.1667, rug 0.1667, park 0.1667`.
- **Second-order model:** 19 non-zero parameters over 15 observed context rows,
  out of 121 possible contexts (106 unseen) — the data-sparsity cost of more
  context.
- **Normalisation test:** every CPT row of both models sums to `1.000000`.
- **Greedy vs sampling:** greedy generation is deterministic and gets stuck in
  the cycle `the cat sat on the cat sat on …`; sampling produces varied,
  well-formed sentences.
- **First vs second order:** in 1000 samples the first-order model produces 175
  distinct sentences (169 novel), while the second-order model produces just 6
  (0 novel) and reproduces a training sentence verbatim 100% of the time.

## A note on the dataset

The worksheet's illustrative example uses a five-transition toy corpus giving
`P(cat|the)=3/5`, `P(dog|the)=2/5`. The actual six-sentence training set has
`the` as a context 12 times, so on the real data
`P(cat|the) = P(dog|the) = 3/12 = 0.25`. Both are correct for their respective
corpora; the difference is noted in `REFLECTION.md`.
