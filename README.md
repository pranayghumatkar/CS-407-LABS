# CS-407 AI Laboratory — Labs & Worksheet Submissions

**Student:** Pranay Ghumatkar
**Roll No:** 2024A3PS0328G

This repository contains my submissions for the CS-407 AI Laboratory.
Each laboratory lives in its own folder with the code, results and written
answers required by its worksheet.

## Labs

| Folder | Topic | Worksheet |
|---|---|---|
| [`Lab_Bayesian_Networks/`](Lab_Bayesian_Networks/) | Bayesian Networks and Autoregressive Language Models | Parts I–XV, Questions 1–14 |

---

## Lab: Bayesian Networks and Autoregressive Language Models

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

### Deliverables and where to find them

| Worksheet deliverable | File |
|---|---|
| First-order implementation | [`Lab_Bayesian_Networks/first_order_lm.py`](Lab_Bayesian_Networks/first_order_lm.py) |
| Second-order implementation | [`Lab_Bayesian_Networks/second_order_lm.py`](Lab_Bayesian_Networks/second_order_lm.py) |
| Conditional probability tables for selected contexts | [`results/first_order_report.txt`](Lab_Bayesian_Networks/results/first_order_report.txt), [`results/second_order_report.txt`](Lab_Bayesian_Networks/results/second_order_report.txt) |
| Examples of generated text | [`results/generated_first_order_sampling.txt`](Lab_Bayesian_Networks/results/generated_first_order_sampling.txt), [`results/generated_first_order_modes.txt`](Lab_Bayesian_Networks/results/generated_first_order_modes.txt), [`results/generated_second_order_sampling.txt`](Lab_Bayesian_Networks/results/generated_second_order_sampling.txt) |
| Probability-normalisation tests | [`results/normalisation_first_order.txt`](Lab_Bayesian_Networks/results/normalisation_first_order.txt), [`results/normalisation_second_order.txt`](Lab_Bayesian_Networks/results/normalisation_second_order.txt), [`tests/test_normalisation.py`](Lab_Bayesian_Networks/tests/test_normalisation.py) |
| Answers to Questions 1–14 | [`Lab_Bayesian_Networks/ANSWERS.md`](Lab_Bayesian_Networks/ANSWERS.md) |
| Reflection on LLM use and validation | [`Lab_Bayesian_Networks/REFLECTION.md`](Lab_Bayesian_Networks/REFLECTION.md) |
| First- vs second-order comparison | [`results/comparison.txt`](Lab_Bayesian_Networks/results/comparison.txt) |

### How to run

```bash
cd Lab_Bayesian_Networks

# Run both models and regenerate every file in results/
python run_all.py

# Run the probabilistic-invariant tests
python tests/test_normalisation.py      # or: python -m pytest tests/ -v
```

Requires only the Python standard library (developed with Python 3.14).

### Headline results

- Training data: six sentences, all lower-cased, tokenised on whitespace, wrapped
  in `<START>` / `<END>`.
- **First-order model:** 17 non-zero parameters over 11 context rows. Row for
  `the`: `cat 0.25, dog 0.25, mat 0.1667, rug 0.1667, park 0.1667`.
- **Second-order model:** 19 non-zero parameters over 15 observed context rows,
  out of 144 possible `|V|²` contexts (129 unseen) — showing the data-sparsity
  cost of more context.
- **Normalisation test:** every CPT row of both models sums to `1.000000`.
- **Greedy vs sampling:** greedy generation is deterministic and gets stuck in
  the cycle `the cat sat on the cat sat on …`; sampling produces varied,
  well-formed sentences.
- **First vs second order:** the second-order model reproduces training-like
  sentences almost exactly, whereas the first-order model produces some
  ungrammatical ones (e.g. `the dog sat on the park`), illustrating the
  bias–variance trade-off.

### A note on the dataset

The worksheet's illustrative example uses a five-transition toy corpus giving
`P(cat|the)=3/5`, `P(dog|the)=2/5`. The actual six-sentence training set has
`the` as a context 12 times, so on the real data
`P(cat|the) = P(dog|the) = 3/12 = 0.25`. Both are correct for their respective
corpora; the difference is noted in `REFLECTION.md`.
