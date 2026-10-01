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
| [`Lab_Neural_Models/`](Lab_Neural_Models/) | Neural Models: Learning, Depth, Activations, and Output Layers | Tasks 1–5, 7 Reflection Questions |

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
  out of 121 possible contexts (106 unseen) — showing the data-sparsity cost of
  more context.
- **Normalisation test:** every CPT row of both models sums to `1.000000`.
- **Greedy vs sampling:** greedy generation is deterministic and gets stuck in
  the cycle `the cat sat on the cat sat on …`; sampling produces varied,
  well-formed sentences.
- **First vs second order:** in 1000 samples the first-order model produces 175
distinct sentences (169 novel), while the second-order model produces just 6
(0 novel) and reproduces a training sentence verbatim 100% of the time. The
first-order model also produces some ungrammatical sentences (e.g. `the dog sat
on the park`), illustrating the bias–variance trade-off.
- The worksheet `BN_lab.pdf` is included in `Lab_Bayesian_Networks/` for
  reference.

### A note on the dataset

The worksheet's illustrative example uses a five-transition toy corpus giving
`P(cat|the)=3/5`, `P(dog|the)=2/5`. The actual six-sentence training set has
`the` as a context 12 times, so on the real data
`P(cat|the) = P(dog|the) = 3/12 = 0.25`. Both are correct for their respective
corpora; the difference is noted in `REFLECTION.md`.

---

## Lab: Neural Models — Learning, Depth, Activations, and Output Layers

Implements a 2→2→1 neural network from scratch in PyTorch — no pretrained
model — for the XOR problem, and uses it to show why a **nonlinear hidden
layer** is necessary, how the **backward pass** is exactly the chain rule, why
**symmetric initialisation** stalls learning, and how the **output layer must
match the task** (binary sigmoid+BCE vs multiclass softmax+cross-entropy).

**Central idea.** A stack of affine layers with no nonlinearity collapses into a
single affine map `W'x + b'`, so XOR — which is not linearly separable — is
unrepresentable no matter how many linear layers are added:

```
a(ℓ) = W(ℓ)h(ℓ−1) + b(ℓ),   h(ℓ) = f(a(ℓ)),   h(0) = x
```
The nonlinearity `f` is what gives the composition real expressive power; the
task then determines the output activation/loss pairing (`softmax`+cross-entropy
has logit gradient `p − y`, `sigmoid`+BCE has `σ(z) − y`).

### Deliverables and where to find them

| Worksheet deliverable | File |
|---|---|
| Binary XOR experiment (Task 3/4 A/B) | [`Lab_Neural_Models/xor_net.py`](Lab_Neural_Models/xor_net.py) |
| Symmetry experiment, all-zero init (Task 4 C) | [`Lab_Neural_Models/symmetry_experiment.py`](Lab_Neural_Models/symmetry_experiment.py) |
| Activation experiment, sigmoid/tanh/ReLU (Task 4 D) | [`Lab_Neural_Models/activation_experiment.py`](Lab_Neural_Models/activation_experiment.py) |
| Three-class softmax extension (Task 5) | [`Lab_Neural_Models/three_class_net.py`](Lab_Neural_Models/three_class_net.py) |
| Autograd vs finite-difference gradient check | [`Lab_Neural_Models/gradient_check.py`](Lab_Neural_Models/gradient_check.py) |
| Answers to Tasks 1–5, Think-About-Its, Reflections 1–7 | [`Lab_Neural_Models/ANSWERS.md`](Lab_Neural_Models/ANSWERS.md) |
| Reflection on LLM use and validation | [`Lab_Neural_Models/REFLECTION.md`](Lab_Neural_Models/REFLECTION.md) |
| Program output (results) | [`Lab_Neural_Models/results/`](Lab_Neural_Models/results/) |
| Invariant tests | [`Lab_Neural_Models/tests/test_neural_models.py`](Lab_Neural_Models/tests/test_neural_models.py) |
| Worksheet PDF | [`Lab_Neural_Models/neur_models_lab_ex.pdf`](Lab_Neural_Models/neur_models_lab_ex.pdf) |

### How to run

```bash
cd Lab_Neural_Models
pip install -r requirements.txt          # CPU-only PyTorch is enough

# Run every experiment and regenerate results/
python run_all.py

# Run the invariants (also works with pytest)
python tests/test_neural_models.py
```

Requires Python 3 and PyTorch (developed with Python 3.14, `torch 2.9.1+cpu`).

### Headline results

- **Binary XOR (2-2-1, sigmoid hidden, `BCEWithLogitsLoss`, Adam lr 0.1, 6000
  steps, seed 2):** loss falls `0.698454 → 0.000016`; probabilities
  `[0.000013, 0.999983, 0.999983, 0.000017]` threshold to `[0,1,1,0]` — **all four
  correct**.
- **Backprop check:** at initialisation `‖∂L/∂W⁽¹⁾‖ = 0.005453`; autograd agrees
  with a `float64` central finite-difference estimate to **`3.3e-12`**. The
  gradient shrinks to `2.6e-7` at convergence.
- **Activation experiment (seed 2):** all three activations classify 4/4;
  final losses `0.000016 / 0.000006 / 0.000003` and early `‖∂L/∂W⁽¹⁾‖`
  `0.005453 / 0.005077 / 0.004868` for sigmoid / tanh / ReLU. The initial mean
  hidden derivative differs much more (`0.245 / 0.920 / 1.000`), so the
  composite gradient norm alone does not rank them.
- **Symmetry (all-zero init):** the two rows of `W⁽¹⁾` stay identical for all
  6000 steps (`‖row0 − row1‖ = 0`), the units never specialise, and the loss is
  stuck at `ln 2 = 0.693147` — the network behaves like one linear unit.
- **Three-class extension (2-2-3, softmax + cross-entropy):** loss
  `1.069200 → 0.000008`, all four inputs classified correctly, every softmax row
  sums to 1. Adding `+100` to all logits leaves the stable softmax unchanged
  (`2.6e-11`) but makes the *naive* softmax `NaN` — why implementations subtract
  the max logit. Verified that `∂L/∂logits = p − y`.
- **Tests:** `8/8` pass (`tests/test_neural_models.py`), including the
  finite-difference agreement, `p − y`, softmax normalisation/shift-invariance,
  zero-init symmetry, and 4/4 accuracy for all three activations.

### A note on the environment

The lab needs PyTorch. On the machine used to produce `results/`,
`torch 2.14.1+cpu` failed to import because a Windows Application Control policy
blocked its `_C` extension (`WinError 4551`); `torch 2.9.1+cpu` imports and runs
correctly, and is what `requirements.txt` targets. The finite-difference check
also runs in `float64`: in `float32` the `1e-7` loss resolution divided by the
step size produces `~1e-4` of roundoff that looks like a gradient disagreement.

