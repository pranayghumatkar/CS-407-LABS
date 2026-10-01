# AI Laboratory — Neural Models: Learning, Depth, Activations, and Output Layers

**Student:** Pranay Ghumatkar
**Roll No:** 2024A3PS0328G
**Worksheet:** [neur_models_lab_ex.pdf](neur_models_lab_ex.pdf)

Implements a 2→2→1 neural network from scratch in PyTorch — no pretrained model
— for the XOR problem, and uses it to show why a **nonlinear hidden layer** is
necessary, how the **backward pass** is exactly the chain rule, why **symmetric
initialisation** stalls learning, and how the **output layer must match the
task** (binary sigmoid+BCE vs multiclass softmax+cross-entropy).

**Central idea.** A stack of affine layers with no nonlinearity collapses into a
single affine map `W'x + b'`, so XOR — which is not linearly separable — is
unrepresentable no matter how many linear layers are added:

```
a(ℓ) = W(ℓ)h(ℓ−1) + b(ℓ),   h(ℓ) = f(a(ℓ)),   h(0) = x
```

The nonlinearity `f` gives the composition real expressive power; the task then
determines the output activation/loss pairing (`softmax`+cross-entropy has logit
gradient `p − y`, `sigmoid`+BCE has `σ(z) − y`).

## Files

| File | Purpose |
|---|---|
| `xor_net.py` | Binary XOR 2-2-1 network, logits + `BCEWithLogitsLoss`, greedy/sampling, gradient inspection (Tasks 3/4 A–B) |
| `symmetry_experiment.py` | Zero-initialisation symmetry experiment (Task 4 C) |
| `activation_experiment.py` | Sigmoid vs tanh vs ReLU hidden activation comparison (Task 4 D) |
| `three_class_net.py` | Three-class 2-2-3 extension with softmax + cross-entropy (Task 5) |
| `gradient_check.py` | Autograd vs central finite-difference gradient check (float64) |
| `run_all.py` | Runs every experiment and regenerates `results/` |
| `tests/test_neural_models.py` | Invariant tests (accuracy, softmax normalisation, `p − y`, symmetry, finite differences) |
| `ANSWERS.md` | Answers to Tasks 1–5, the Think-About-Its and Reflection Questions 1–7 |
| `REFLECTION.md` | How the LLM was used and how its output was validated |
| `results/` | Raw program output for every experiment |
| `requirements.txt` | Dependencies (PyTorch) |
| `neur_models_lab_ex.pdf` | Worksheet |

## How to run

```bash
cd Neural_Models_Lab
pip install -r requirements.txt          # CPU-only PyTorch is enough

# Run every experiment and regenerate results/
python run_all.py

# Run the invariants (also works with pytest)
python tests/test_neural_models.py
```

Requires Python 3 and PyTorch (developed with Python 3.14, `torch 2.9.1+cpu`).

## Headline results

- **Binary XOR (2-2-1, sigmoid hidden, `BCEWithLogitsLoss`, Adam lr 0.1, 6000
  steps, seed 2):** loss falls `0.698454 → 0.000016`; probabilities
  `[0.000013, 0.999983, 0.999983, 0.000017]` threshold to `[0,1,1,0]` — **all four
  correct**.
- **Backprop check:** at initialisation `‖∂L/∂W⁽¹⁾‖ = 0.005453`; autograd agrees
  with a `float64` central finite-difference estimate to **`3.3e-12`**. The
  gradient shrinks to `2.6e-7` at convergence.
- **Activation experiment (seed 2):** all three activations classify 4/4; final
  losses `0.000016 / 0.000006 / 0.000003` and early `‖∂L/∂W⁽¹⁾‖`
  `0.005453 / 0.005077 / 0.004868` for sigmoid / tanh / ReLU.
- **Symmetry (all-zero init):** the two rows of `W⁽¹⁾` stay identical for all
  6000 steps (`‖row0 − row1‖ = 0`), the units never specialise, and the loss is
  stuck at `ln 2 = 0.693147` — the network behaves like one linear unit.
- **Three-class extension (2-2-3, softmax + cross-entropy):** loss
  `1.069200 → 0.000008`, all four inputs classified correctly, every softmax row
  sums to 1. Adding `+100` to all logits leaves the stable softmax unchanged
  (`2.6e-11`) but makes the *naive* softmax `NaN` — why implementations subtract
  the max logit. Verified that `∂L/∂logits = p − y`.
- **Tests:** `8/8` pass (`tests/test_neural_models.py`).

## A note on the environment

`torch 2.14.1+cpu` failed to import on the machine used to produce `results/`
because a Windows Application Control policy blocked its `_C` extension
(`WinError 4551`); `torch 2.9.1+cpu` imports and runs correctly. The
finite-difference check runs in `float64` — in `float32` the `1e-7` loss
resolution divided by the step size produces `~1e-4` of roundoff that looks like
a gradient disagreement.
