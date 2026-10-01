# CS-407 AI Laboratory — Neural Models: Learning, Depth, Activations, and Output Layers

**Student:** Pranay Ghumatkar
**Roll No:** 2024A3PS0328G

This document answers the worksheet tasks, the "Think About It" prompts and the
seven Reflection Questions. Every number quoted below is produced by the code in
this folder and stored under `results/`.

---

## Task 1 — Understand the problem before coding

**1. Input space, output space and the four examples.**

- Input space `X = {0,1} × {0,1}`, i.e. `{(0,0), (0,1), (1,0), (1,1)}` — the two
  binary sensor readings.
- Output space `Y = {0,1}` — `1` means "raise the disagreement warning".
- Labelled examples (`y = x1 XOR x2`):

  | x1 | x2 | y |
  |----|----|---|
  | 0  | 0  | 0 |
  | 0  | 1  | 1 |
  | 1  | 0  | 1 |
  | 1  | 1  | 0 |

**2. Sketch of the four points in the (x1, x2) plane.**

```
x2
 1 |  (0,1) y=1        (1,1) y=0
   |
 0 |  (0,0) y=0        (1,0) y=1
   +------------------------------ x1
       0                 1
```

The two class-1 points sit on the anti-diagonal; the two class-0 points sit on
the main diagonal.

**3. Why one straight boundary cannot separate the classes.**

The two class-1 points lie on opposite corners from one another, and each shares
a row and a column with a class-0 point. A single straight line would have to put
`(0,1)` and `(1,0)` on one side while putting `(0,0)` and `(1,1)` on the other;
any line that separates the diagonals must cut the square such that both
class-1 points and both class-0 points are on the same side — impossible. XOR is
therefore **not linearly separable** (no `w1·x1 + w2·x2 + b` can realise it).

**4. Prediction for a single affine transformation + sigmoid.**

A single affine map followed by a sigmoid is a logistic regression: it can only
produce one straight decision boundary `w1·x1 + w2·x2 + b = 0`. Since XOR is not
linearly separable, training will stall at the best linear fit. For the balanced
XOR data the model can do no better than always predicting ~0.5, giving a binary
cross-entropy loss near `ln 2 = 0.6931` and never classifying all four points
correctly. **(Confirmed later: the zero-init/symmetric run stays at exactly
0.693147 — see Task 4 Part C, where the whole network behaves like one linear
unit.)**

> **Think About It — what claim does XOR test?**
> XOR tests that *depth alone is not enough*: a model can have many parameters
> and still use the **wrong kind of representation**. It isolates the claim that
> a **nonlinear hidden layer** (not merely more linear layers/parameters) is what
> makes a non-linearly-separable function representable. Four points are enough
> because the obstruction is combinatorial, not statistical.

---

## Task 2 — Design the intelligent agent

**Specification (2 → 2 → 1).** Input `x ∈ R²`; hidden `h = f(W⁽¹⁾x + b⁽¹⁾) ∈ R²`;
output logit `z = W⁽²⁾h + b⁽²⁾ ∈ R`; `p = σ(z)`; loss `BCEWithLogitsLoss(z, y)`;
optimiser Adam, full batch, `lr = 0.1`, 6000 steps, seed 2.

**1. Why the hidden nonlinearity is scientifically necessary.**

Without a nonlinearity, `W⁽²⁾(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾` collapses into a single
affine map `W'x + b'`. The XOR witness shows no affine map can realise the
function, so the network would be provably unable to represent it. The nonlinear
`f` is what gives the composition genuine expressive power.

**2. Why sigmoid + binary cross-entropy is a sensible engineering pairing.**

The target is one yes/no answer, so the output must be a probability in `(0,1)`:
the sigmoid supplies that. The matching loss is binary cross-entropy, whose
likelihood it maximises; and in PyTorch the fused `BCEWithLogitsLoss` applies the
sigmoid internally and computes the log-sum-exp form, avoiding the
`log(σ(z)) → −∞`/`log(0)` overflow of applying sigmoid then BCE separately. The
logit gradient reduces to the clean `p − y`.

**3. Evidence of successful learning (three checks).**

1. **Loss fell** from `0.6985` to `0.000016` (Task 4 Part A).
2. **All four labels are correct** — probabilities `[0.000013, 0.999983, 0.999983, 0.000017]`
   threshold to `[0,1,1,0]` (Task 4 Part A).
3. **The gradient is genuinely the derivative**: at initialisation `‖∂L/∂W⁽¹⁾‖ = 0.005453`,
   and autograd agrees with a central finite-difference estimate to `3.3e-12`
   (`results/gradient_check.txt`). A fourth check is **repeatability**: re-running
   with the same seed reproduces the result.

> **Think About It — the hidden units have no targets, so what decides what they compute?**
> Nothing hands the hidden units a target. Backpropagation assigns each hidden
> unit its role implicitly: the chain rule attributes part of the output error to
> each hidden unit through `W⁽²⁾ · σ'(z)`, and each unit's weights move to reduce
> that attributed error. The learned split (`h1 = x1 XOR-type feature`,
> `h2 = AND/OR-type feature`) is whatever joint configuration lowers the loss —
> it is discovered by the gradient, not specified by a label.

---

## Task 3 — LLM-generated first implementation

**The exact prompt I used:**

> Generate minimal PyTorch code for the following model and dataset. Do not
> change the architecture or task. The dataset is XOR:
> `X = [[0,0],[0,1],[1,0],[1,1]]`, `y = [[0],[1],[1],[0]]`. Use a 2→2→1 network
> with a sigmoid hidden activation and a single output trained with
> `BCEWithLogitsLoss`. Initialise the weights randomly. Train full-batch on CPU
> for a few thousand steps, print the final loss and the four predicted
> probabilities (thresholded to labels), and after `backward()` expose one
> parameter-gradient tensor. Set a random seed for reproducibility and explain
> each test in one sentence.

**Field identification before running (as required):**

- **Forward pass** — `xor_net.XORNet.forward`: `a1 = fc1(x)`, `h = act(a1)`,
  `z = fc2(h)`.
- **Scalar loss** — `loss = nn.BCEWithLogitsLoss()(model(X), Y)` in
  `train_xor`.
- **Reverse-mode AD** — `loss.backward()`.
- **Optimiser step** — `opt.step()` (Adam).

**Two changes I made to the generated code before execution.**

1. The first draft applied `torch.sigmoid` to the output and used
   `nn.BCELoss`. I replaced both with raw **logits + `BCEWithLogitsLoss`**, as
   the worksheet recommends, for numerical stability.
2. The draft used `SGD` at a default learning rate and stopped at ~1000 steps,
   which stalled in a local minimum (loss ≈ 0.477, predicting class 1 for three
   of four points). I switched to **Adam, lr = 0.1, 6000 steps, seed 2**, which
   converges. These are engineering changes only; the architecture and task are
   untouched.

> **Think About It — what can be verified without running, and what needs execution?**
> Readable from the code alone: the architecture (2-2-1), the data, that a
> nonlinearity is present, that the loss is the correct pairing, that
> `backward()` and `opt.step()` are called in the right order, and that no
> `torch.no_grad()` accidentally wraps training. Needing execution and
> measurement: whether the optimisation actually converges, whether all four
> labels are correct, the magnitude/nonzero-ness of the gradients, and the
> zero-init symmetry outcome. Syntax can be right while the *experiment* is
> wrong, so the numeric claims must be measured.

---

## Task 4 — Execute, test, diagnose

### Part A — Basic learning check

| Quantity | Value |
|---|---|
| Initial loss | `0.698454` |
| Final loss | `0.000016` |
| P(y=1) at (0,0) | `0.000013` → label 0 (target 0) |
| P(y=1) at (0,1) | `0.999983` → label 1 (target 1) |
| P(y=1) at (1,0) | `0.999983` → label 1 (target 1) |
| P(y=1) at (1,1) | `0.000017` → label 0 (target 0) |
| All four correct | **True** |

Full output: `results/xor_training.txt`.

### Part B — Backpropagation check

`param.grad` after `backward()` holds `∂L/∂param`, i.e. for the first-layer
weights `model.fc1.weight.grad = ∂L/∂W⁽¹⁾` — the sensitivity of the scalar loss
to each weight, computed by the chain rule over the whole computation graph.

At **initialisation** `∂L/∂W⁽¹⁾ = [[0.0005, 0.0005], [0.0037, 0.0039]]`,
`‖∂L/∂W⁽¹⁾‖ = 0.005453`; after training it has shrunk to `‖∂L/∂W⁽¹⁾‖ = 2.6e-7`
(learning worked, so the loss is no longer sensitive). A central
finite-difference estimate matches autograd to `3.34e-12`
(`results/gradient_check.txt`).

**Why the gradient is the average of the example-wise gradients.** The default
`BCEWithLogitsLoss` uses mean reduction, so
`L = (1/4) Σᵢ ℓᵢ`. Differentiation is linear, so
`∂L/∂W⁽¹⁾ = (1/4) Σᵢ ∂ℓᵢ/∂W⁽¹⁾`: the four example-wise gradients are summed and
divided by four. Each example contributes an outer product of the form
`δᵢ · xᵢᵀ`, so the batch gradient is the mean of those outer products.

### Part C — Symmetry experiment (all weights = 0)

| Quantity | Value |
|---|---|
| `‖W⁽¹⁾[0] − W⁽¹⁾[1]‖` at step 0 / 100 / final | `0` / `0` / `0` |
| Rows stay identical | **True** |
| Initial loss | `0.693147` |
| Final loss | `0.693147` (= `ln 2`) |
| Predictions | `[1, 1, 1, 1]` — **wrong** |

Full output: `results/symmetry_experiment.txt`.

**Explanation.** With all weights zero, both hidden units compute the same
value `f(0)` and receive the **same gradient** (they are interchangeable
symmetries of the loss). Gradient descent updates them identically, so they stay
identical forever — the network is stuck on a symmetric saddle and behaves like
a network with **one** effective hidden unit, which cannot represent XOR. The
loss never moves off `ln 2`. This is why *breaking the symmetry* with random
initialisation is essential; it is also why the ReLU variant of the zero-init
run learns nothing at all (`ReLU'(0) = 0`, so the gradient is exactly 0).

### Part D — Activation experiment

Seed 2, lr 0.1, 6000 steps, random initialisation:

| Hidden activation | Final loss | 4/4 correct? | Early `‖∂L/∂W⁽¹⁾‖` |
|---|---|---|---|
| Sigmoid | `0.000016` | True | `0.005453` |
| Tanh | `0.000006` | True | `0.005077` |
| ReLU | `0.000003` | True | `0.004868` |

At step 0 the local hidden derivatives differ sharply — mean `f'(a⁽¹⁾)` is
**0.245** (sigmoid), **0.920** (tanh), **1.000** (ReLU), with no saturated/negative
units for this seed — yet the **norms** of the full first-layer gradient land
within ~10% of one another.

**Interpretation (one paragraph).** All three activations solve XOR from the same
seed with final losses of order `1e-5`, so this experiment does **not** rank
them. The activation clearly controls the per-unit factor `f'(a⁽¹⁾)` (0.245 vs
0.920 vs 1.000), but `∂L/∂W⁽¹⁾` is a *composite*: it multiplies the output-error
signal, the output weights `W⁽²⁾` and `f'(a⁽¹⁾)`, then sums over the four
examples and both units. Those other factors differ between activations too, so
the composite norms come out close; the derivative table — not the norm alone —
shows the mechanism. These are observations about one initialisation on four
points, not a universal claim that one activation is best.

> **Think About It — saturated sigmoid vs negative ReLU: two ways to get a small gradient.**
> Both make `f'(a) → 0`, but for different reasons and they are distinguishable
> by inspecting the **pre-activation** `a⁽¹⁾` (and the activation `h`):
> a *saturated sigmoid* has a **large-magnitude** `a` (say `|a| > 4`), with `h`
> pinned near 0 or 1 and `f'(a) = h(1−h) ≈ 0`; a *dead ReLU* has a **negative**
> `a`, with `h = 0` exactly and `f'(a) = 0`. Looking at activations and
> pre-activations separates "pushed to the rails" from "switched off".

---

## Task 5 — Three-class extension

Only the output/loss portion changes: the 2→2 hidden layer stays, the single
logit becomes **three logits**, and `BCEWithLogitsLoss` becomes multiclass
cross-entropy (`CrossEntropyLoss`, which fuses log-softmax and NLL). Labels are
`[0, 1, 1, 2]`.

**Predictions before accepting the change.**

1. **Shape of the final weight matrix** — `W⁽²⁾` is **(3 × 2)**: 3 output
   classes × 2 hidden units. (Bias `b⁽²⁾` is length 3.)
2. **Logits per example** — **3**, one per class, so the output is `(4, 3)`.
3. **Why softmax probabilities sum to one** — `softmax(z)_k = e^{z_k} / Σ_j e^{z_j}`;
   summing over `k` gives `Σ_k e^{z_k} / Σ_j e^{z_j} = 1` by construction.
4. **Why the logit gradient has the form `p − y`** — with `L = −log softmax(z)_c`,
   `∂L/∂z_k = p_k − y_k`, where `p = softmax(z)` and `y` is the one-hot target.
   Intuitively, a class that is over-predicted (`p_k > y_k`) gets a positive
   push *down*, a deficient class a negative push *up*.

**Run results** (`results/three_class.txt`):

| input | class 0 | class 1 | class 2 | predicted | target |
|---|---|---|---|---|---|
| (0,0) | 0.999991 | 0.000009 | 0.000000 | 0 | 0 |
| (0,1) | 0.000004 | 0.999993 | 0.000003 | 1 | 1 |
| (1,0) | 0.000004 | 0.999993 | 0.000003 | 1 | 1 |
| (1,1) | 0.000000 | 0.000010 | 0.999990 | 2 | 2 |

Initial loss `1.069200`, final loss `0.000008`, **all four correct**. Every
softmax row sums to 1 (e.g. `1.0000000420`), verified numerically. Example 0's
softmax vector is `[0.999991, 0.000009, 0.0]`, one-hot target `[1,0,0]`, so
`p − y = [−9e−6, 9e−6, 0]` — the gradient pushes logit 0 up and logit 1 down.

**Optional diagnostic (adding +100 to every logit).** Mathematically
`softmax(z + c1) = softmax(z)` because the constant factors out of numerator and
 denominator. In floating point the **naive** softmax overflows to `NaN`
(`exp(100) = inf`, and `inf/inf = NaN`), while the **max-subtracted** softmax is
unchanged up to roundoff (`max |softmax(z) − softmax(z+100)| = 2.6e-11`). This
is exactly why stable implementations subtract the row maximum: `z − max(z) ≤ 0`,
so `exp(z − max(z)) ∈ (0, 1]` cannot overflow.

> **Think About It — which parts stay the same when the vocabulary is huge?**
> Mathematically the same: three logits → **V logits**, `softmax` → `softmax`
> over V classes, cross-entropy, and the `p − y` logit gradient, all unchanged.
> What changes dramatically is the **architecture and cost**: `W⁽²⁾` becomes
> `(V × d)`, the softmax and its gradient are computed over tens of thousands of
> classes (hence approximations like sampled/hierarchical softmax), the input is
> a context embedding rather than two bits, and training data/compute scale by
> orders of magnitude.

---

## Reflection Questions

**1. What did the XOR experiment demonstrate about the difference between depth and nonlinearity?**
Depth alone buys nothing here. Stacking affine layers collapses to one affine
map (a 2-2-1 network with linear hidden units is *provably* a linear model and
cannot fit XOR), so "more layers" is not the same as "more representational
power". What enables XOR is the **nonlinear hidden activation**; the `(0,0)` and
`(1,1)` outputs must be bent around the `(0,1)`, `(1,0)` points. XOR isolates
this with four points, where the failure is representational, not statistical.

**2. What evidence showed backprop supplied a useful learning signal, not merely a nonzero gradient?**
Three measurements together: (i) the loss fell from `0.6985` to `0.000016` and
all four labels became correct — a nonzero gradient alone would not guarantee
this; (ii) the initial gradient `‖∂L/∂W⁽¹⁾‖ = 0.005453` **matched an independent
finite-difference estimate to 3.3e-12**, so the direction was genuinely
`−∂L/∂W`; and (iii) the gradient **shrank** to `2.6e-7` as the loss approached
zero, which is exactly the expected behaviour when the signal is driving the
loss down rather than merely being nonzero.

**3. Why did identical/zero initialisation prevent the hidden units from learning distinct features?**
With all weights zero the two hidden units are exact mirror images: they compute
the same output `f(0)` and therefore receive the same gradient (and Adam moves
them identically). The loss is perfectly symmetric under swapping the units, so
gradient descent keeps `W⁽¹⁾[0] = W⁽¹⁾[1]` for all time (`‖row0 − row1‖ = 0`
throughout) and the network has only one effective hidden unit. With one hidden
unit XOR is unrepresentable, so the loss sticks at `ln 2 = 0.693147`.

**4. How did changing the hidden activation affect the gradient? Science vs engineering.**
*Scientific explanation:* the first-layer gradient carries a factor `f'(a⁽¹⁾)`,
so the activation controls how much signal each hidden unit passes. At step 0
the mean derivative was `0.245` (sigmoid), `0.920` (tanh), `1.000` (ReLU): a
saturated sigmoid shrinks the signal via a small `h(1−h)`, a negative ReLU zeroes
it entirely, and tanh/ReLU pass more. *Engineering observation:* the actual
early gradient **norms** were close — `0.005453`, `0.005077`, `0.004868` — because
the norm is a composite of the output error, `W⁽²⁾` and `f'(a⁽¹⁾)` summed over all
examples and units, so the derivative difference was partly absorbed elsewhere.
The science tells you *why* the derivative matters; the engineering measurement
shows the end-to-end effect on one seed is small. With four points we cannot rank
activations.

**5. Why must the output layer and loss be selected together according to the task?**
The loss defines the probabilistic model the output layer is estimating. For a
binary target the natural output is a Bernoulli probability (sigmoid) learned by
binary cross-entropy; for one-of-K the natural output is a categorical
distribution (softmax) learned by cross-entropy. Mismatches are wrong models:
sigmoid + multiclass-CE, or softmax + BCE, no longer correspond to a valid
likelihood, and the gradient relations (`p − y` for softmax-CE, `σ(z) − y` for
sigmoid-BCE) come from the matched pairing. In PyTorch the fused losses
(`BCEWithLogitsLoss`, `CrossEntropyLoss`) also bake in the numerically stable
form, so choosing them together is both a modelling and a stability decision.

**6. One place the LLM improved productivity; one place human verification was essential.**
*Improved productivity:* generating the correct PyTorch boilerplate — the
2-2-1 module, the training loop with `zero_grad → forward → loss → backward →
step`, the `BCEWithLogitsLoss`, and the three-class head — in one pass. That is
mechanical and was quick to then check. *Human verification essential:* the LLM's
first draft used `SGD` for too few steps and silently **failed to solve XOR**
(loss stayed ≈ 0.477). It also mis-stated the illustrative worksheet figure
versus the actual data (an echo of the BN lab). Both were only caught by running
the experiment and checking the four labels and the gradient against a
finite-difference estimate — the LLM produced plausible code that implemented a
plausible-but-wrong experiment.

**7. Which tests would survive scaling up, and which would become too expensive?**
*Keep:* dimensional/shape checks, "probabilities in [0,1] and rows sum to 1",
"predictions match labels on the training batch", loss-decreased checks, and —
critically — each new loss's built-in **gradient check against autograd**
(`torch.autograd.gradcheck` on a small slice) if it is custom. *Become too
expensive:* **exhaustive finite-difference gradient checks**. A central
difference costs two forward passes *per parameter*, so the tiny 6-parameter
network here is fine (`3.3e-12` agreement), but a model with millions of
parameters would need millions of extra forward passes. In practice one replaces
it with `gradcheck` on a few sampled coordinates or a whole-vector Taylor/
finite-difference test in a random direction, and relies on the framework's
tested autograd for the rest.

---

*LLM use and validation are documented in `REFLECTION.md`.*

