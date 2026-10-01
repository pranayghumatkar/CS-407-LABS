# Reflection — How the LLM Was Used and How Its Output Was Validated

**Student:** Pranay Ghumatkar
**Roll No:** 2024A3PS0328G

## 1. The workflow I followed

I followed the laboratory workflow: **specify → design → ask the LLM →
implement → test → reflect.** Before asking for any code I wrote down, in words
(Task 1 and Task 2):

- **Problem:** `X = {0,1}²`, `Y = {0,1}`, four examples `[0,1,1,0]`; XOR is not
  linearly separable, so a single affine+sigmoid will stall at `ln 2`.
- **Model:** `2 → 2 → 1`, nonlinear hidden activation, sigmoid output, binary
  cross-entropy, gradient descent.
- **Validation criteria:** final loss, all four labels correct, a
  nonzero-and-correct gradient, and repeatability.

Only then did I request an implementation with the constraints from Task 3. The
specification is what I tested the generated code against.

## 2. How I validated the output

Three independent checks, not one number:

1. **Read the code against the specification** — located the forward pass
   (`XORNet.forward`), the scalar loss (`BCEWithLogitsLoss`), the AD call
   (`loss.backward()`), and the parameter update (`opt.step()`).
2. **Tested invariants** in `tests/test_neural_models.py`: all three activations
   solve XOR; probabilities lie in `[0,1]`; softmax rows sum to 1; stable
   softmax is shift-invariant while the naive one overflows; zero-init rows stay
   identical; and — the strongest check — **autograd matches a finite-difference
   gradient**.
3. **Ran the model and measured** every number quoted in `ANSWERS.md`, rather
   than trusting a single final loss.

## 3. Concrete errors I found and corrected

### (a) The LLM's first implementation did not solve XOR

The generated code used plain `SGD` at a low learning rate and stopped after
~1000 steps. It ran without errors, but the loss was stuck at `0.477` and it
predicted class 1 for three of four points. **Diagnosis:** XOR from random init
has poor local minima; SGD at that rate had not escaped. **Correction:**
`Adam`, `lr = 0.1`, 6000 steps, `seed = 2` (chosen after sweeping seeds).
Architecture and task unchanged — only justified engineering settings.

### (b) The finite-difference check "disagreed" because of float32

My first gradient check computed the numerical gradient in `float32` with
`eps = 1e-4` and reported a maximum difference of `3.5e-4`, which looked like
autograd being wrong. **Diagnosis:** the loss resolution in float32 is ~`1e-7`,
and dividing that by `2·eps = 2e-4` yields `~5e-4` of pure roundoff noise — the
disagreement was numerical, not algorithmic. **Correction:** run the check in
`float64` with `eps = 1e-5`; the two estimates then agree to `3.3e-12`. The
lesson: a failing test can indict the *test*, not the code.

### (c) The `p − y` test failed by a factor of 4

`∂L/∂logits = p − y` holds for the **sum** reduction. `F.cross_entropy` defaults
to **mean** reduction, so the true gradient is `(p − y)/4`; my assertion failed
with a maximum error of `0.606`. **Correction:** compute the reference gradient
with `reduction="sum"` and compare there. This is exactly the kind of silent
factor an LLM will gloss over — it wrote the formula correctly but not the
reduction that makes it true for this code path.

### (d) An over-claimed interpretation that the data contradicted

I initially wrote that ReLU had the *largest* early gradient because its
derivative is 1. Measurement showed the opposite ordering
(`0.0055 / 0.0051 / 0.0049` for sigmoid / tanh / ReLU) even though the mean
derivatives were `0.245 / 0.920 / 1.000`. **Correction:** I removed the
mechanistic claim and replaced it with the honest observation that the gradient
norm is a composite of several factors, so the derivative table — not the norm —
is what shows the mechanism. Writing a plausible story is not the same as
measuring one.

## 4. What the LLM was good at, and what it was not

- **Good at:** boilerplate — the module definitions, the training loop ordering
  (`zero_grad → forward → loss → backward → step`), the fused losses, the
  softmax/shift-invariance scaffolding, and translating my written design into a
  first draft quickly.
- **Not good at:** judging whether the *experiment* worked. It produced code that
  ran but did not converge (a); it stated `p − y` without noticing the reduction
  (c); and it is happy to assert a mechanism the data does not support (d). It
  also cannot know the numeric-precision issues in (b).

The pattern, as in the Bayesian-networks lab, is that the LLM is a tool for
*construction*, not for *understanding*. The understanding — what the model
estimates, which invariants it must satisfy, where the boundary/numerical
pitfalls are — is what made the errors above detectable. Implementation and model
stay separate; keeping them separate is what makes validation possible.
