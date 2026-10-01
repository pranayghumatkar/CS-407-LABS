# CS-407 AI Laboratory — Bayesian Networks and Autoregressive Language Models

**Student:** Pranay Ghumatkar
**Roll No:** 2024A3PS0328G

This document answers Questions 1–14 of the worksheet. All numbers quoted
below are produced by the programs in this folder and recorded in `results.txt`.

---

## Question 1 — Why is the chain-rule decomposition useful for generating text?

The chain rule rewrites the joint probability of a whole sentence as a product
of conditionals:

```
P(X1,...,X6) = P(X1)·P(X2|X1)·P(X3|X1,X2)·P(X4|X1,X2,X3)·P(X5|X1..X4)·P(X6|X1..X5)
```

This is exactly what an autoregressive model needs, for three reasons:

1. **It factorises an intractable joint into tractable small pieces.** Directly
   assigning a probability to every possible sentence is impossible; each
   conditional above only ranges over one token given a fixed prefix.
2. **It gives a left-to-right generation procedure.** `P(X1)` picks the first
   token, `P(X2|X1)` picks the second, and so on. Generation becomes a loop of
   "sample the next token given what I have so far", which is precisely how
   modern LLMs emit text.
3. **It is exact, not an approximation.** The product of the conditionals equals
   the true joint probability of the sequence; the approximations (e.g. Markov
   assumptions) are introduced later, to make the factors learnable from data.

---

## Question 2 — What independence assumption is being made by the first-order network?

For the chain `X1 → X2 → X3 → X4` the model asserts the **first-order Markov**
(first-order Markov) property:

```
X_t  ⊥  {X_1, ..., X_{t-2}}  |  X_{t-1}
```

In words: once the immediately preceding word is known, all earlier words give
no additional information about the next word. Probability notation:

```
P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1})
```

This is what collapses the full chain-rule product into
`P(X1)·P(X2|X1)·P(X3|X2)·P(X4|X3)`.

---

## Question 3 — Conditional probability table P(next word | current word)

Built from the six-sentence dataset (each sentence wrapped in `<START>`/`<END>`):

| current word | next word | P(next \| current) |
|---|---|---|
| the   | cat  | 3/12 = 0.2500 |
| the   | dog  | 3/12 = 0.2500 |
| the   | mat  | 2/12 = 0.1667 |
| the   | rug  | 2/12 = 0.1667 |
| the   | park | 2/12 = 0.1667 |
| cat   | sat  | 2/3 = 0.6667 |
| cat   | ran  | 1/3 = 0.3333 |
| dog   | sat  | 2/3 = 0.6667 |
| dog   | ran  | 1/3 = 0.3333 |
| sat   | on   | 1.0000 |
| ran   | to   | 1.0000 |
| on    | the  | 1.0000 |
| to    | the  | 1.0000 |
| mat   | `<END>` | 1.0000 |
| rug   | `<END>` | 1.0000 |
| park  | `<END>` | 1.0000 |
| `<START>` | the | 1.0000 |

(Full machine-generated tables are in `results.txt`.)

**Zero-probability transitions.** With maximum-likelihood counting, every row is
sparse. For example:

- `P(<END> | the) = 0`, `P(sat | the) = 0`, `P(on | the) = 0`, … — "the" is
  followed only by cat/dog/mat/rug/park in the corpus.
- `P(mat | cat) = 0`, `P(dog | cat) = 0`, `P(cat | dog) = 0`, …
- Nothing can follow `<END>`, and `<START>` only precedes "the".

There are 12 vocabulary words and 17 non-zero transitions out of a possible
12 × 12 = 144, so the CPT is extremely sparse. Any transition not seen in
training is assigned probability zero — the classic **data-sparsity** problem of
n-gram models, and the reason smoothing/back-off is needed.

---

## Question 4 — Where in the program are the transition counts stored?

In `first_order_lm.py`, inside `FirstOrderLanguageModel`:

```python
self.transition_counts = defaultdict(Counter)   # C(w_i, w_j)
```

`.train()` fills it: for each adjacent pair it does
`self.transition_counts[current_word][next_word] += 1`. So
`transition_counts['the']` is a `Counter` such as `{'cat': 3, 'dog': 3, 'mat': 2, 'rug': 2, 'park': 2}`.
This is the raw count `C(w_i, w_j)` from the worksheet formula.

---

## Question 5 — Where is P(X_t | X_{t-1}) computed?

Also in `first_order_lm.py`, in `_compute_probabilities()`, which is called at
the end of `train()`:

```python
for current_word, next_counts in self.transition_counts.items():
    total = sum(next_counts.values())
    self.probabilities[current_word] = {
        next_word: count / total for next_word, count in next_counts.items()
    }
```

Each count is divided by the row total, i.e.
`P(w_j | w_i) = C(w_i, w_j) / Σ_k C(w_i, w_k)`. The result is stored in
`self.probabilities` and returned by the `distribution()` method.

---

## Question 6 — How does the program choose the next word?

The program supports **both**, selected by the `mode` argument of `generate()`:

1. **Greedy** (`mode="greedy"`) — `predict_next()` returns
   `argmax_w P(w | current)`, always the single most probable word.
2. **Sampling** (`mode="sample"`, the default) — `sample_next()` calls
   `random.choices(words, weights=probabilities, k=1)`, which draws a word with
   probability exactly equal to `P(w | current)`.

The difference is behavioural and important:

- Greedy is **deterministic**. Re-running it gives identical output. Because the
  argmax chain in this corpus is `the → cat → sat → on → the → …`, greedy never
  reaches `<END>` and gets trapped in the cycle
  `"the cat sat on the cat sat on …"` (see `results.txt`).
- Sampling is **stochastic**. Different samples follow different plausible paths,
  so `<END>` is eventually reached and a variety of sentences is produced.

Neither is "the model"; the model is the distribution. Greedy and sampling are
two different *decoders* operating on the same `P(X_t | X_{t-1})`.

---

## Question 7 — What happens for a word with no observed transition?

At query time `distribution(word)` returns an empty dict, so `predict_next()`
and `sample_next()` both return `None`. `generate()` then breaks out of its loop
and stops the sentence. In effect the model treats that continuation as
**impossible (probability zero)** rather than guessing.

This is a defect of pure maximum-likelihood n-grams: a single unseen bigram
kills the sentence. Standard remedies, which the worksheet's second-order model
uses, are:

- **Back-off**: if the higher-order context is unseen, fall back to a lower-order
  distribution (`second_order_lm._row()` backs off to `P(w | w_{t-1})`).
- **Smoothing** (Laplace/add-k): add a small constant to every count so no
  probability is exactly zero.

---

## Question 8 — If one of the totals is 0.87, what does this tell you?

It means that CPT row is **not a valid probability distribution** — its values
sum to 0.87, so 0.13 of the probability mass is missing. Something in the
implementation has gone wrong, for example:

- counts were divided by the wrong denominator (e.g. the total number of tokens
  instead of the row total `Σ_k C(w_i, w_k)`);
- the row was built from raw counts and probabilities mixed together;
- some transitions were dropped/filtered before normalising;
- a "smoothing" path added new words to the numerator without updating the
  denominator.

Since every proper conditional distribution must satisfy `Σ_v P(v | w) = 1`,
the correct program's rows all print `1.000000` (see
`results.txt`).
A 0.87 row is a concrete, detectable bug — this is why the invariant test is a
useful part of validating an LLM-generated implementation.

---

## Question 9 — Are the most probable predictions the same as human expectation?

Partly. For many contexts the model agrees with intuition because the corpus is
so regular:

```
arg max P(w | sat)  = on      (matches "sat on ...")
arg max P(w | ran)  = to      (matches "ran to ...")
arg max P(w | on)   = the     (matches "on the ...")
arg max P(w | mat)  = <END>   (matches "the mat." ending a sentence)
```

But for `the` the model is **tied**: `P(cat|the) = P(dog|the) = 0.25`, and an
English speaker has strong expectations about which is more likely given wider
context. The model has no notion of meaning, world knowledge, or long context —
it only counts co-occurrence in six sentences. So:

- A **probability model** answers "how often did this follow that in the data",
  i.e. it encodes frequency, not semantics.
- **Human linguistic expectations** come from meaning, grammar and world
  knowledge and can differ sharply from corpus frequencies, especially with tiny
  data.

The lesson: agreement on easy, high-frequency patterns does not imply the model
"understands" language.

---

## Question 10 — Greedy vs sampling: which produces more variation? Why?

**Sampling produces far more variation.** In `results.txt`:

- **Mode A (greedy, 5 sentences):** all five outputs are the identical cycle
  `the cat sat on the cat sat on …` (it never emits `<END>`, so it runs to the
  token cap). Greedy always takes the highest-probability edge, so once it enters
  the cycle `the → cat → sat → on → the` it can never leave.
- **Mode B (sampling, 5 sentences):** five *different* sentences, e.g.
  `the cat sat on the mat`, `the rug`, `the park`,
  `the dog sat on the mat`.

**Why:** greedy is a deterministic `argmax` — one fixed path per starting
context, so variation is impossible. Sampling draws from the full distribution,
so lower-probability continuations (e.g. `ran` instead of `sat`, or `<END>`
instead of `the`) do occur, producing diverse outputs. Variation is a property
of decoding, not of the trained distribution.

---

## Question 11 — How does the second-order model differ?

| Aspect | First-order | Second-order |
|---|---|---|
| **1. Graph structure** | chain `X_{t-1} → X_t` | `X_{t-2} → X_t ← X_{t-1}` (each token has two parents) |
| **2. CPT** | one row per word: `P(X_t\|X_{t-1})`, 17 non-zero params | one row per *pair*: `P(X_t\|X_{t-2},X_{t-1})`, 19 non-zero params over 15 observed contexts |
| **3. Context available** | 1 previous token | 2 previous tokens (more disambiguating information) |
| **4. Data needed** | ~`\|V\|²` parameters to estimate | ~`\|V\|³` rows in the worst case; far more data, sparse, needs back-off |

The second-order model is *less* of an approximation of
`P(X_t | X_1..X_{t-1})` than the first-order one, but it pays for that with a
much larger, sparser table.

---

## Question 12 — Why can more context help *and* hurt?

**Helps:** extra context disambiguates. `P(w | the)` is a 5-way split, but
`P(w | the, cat)` is nearly deterministic (sat 0.667 / ran 0.333), and
`P(w | the, dog)` is the same but about a different subject. In general, the more
of the prefix the model conditions on, the closer it gets to the true
`P(X_t | X_1..X_{t-1})`.

**Hurts (data sparsity):** the CPT is indexed by the context, so its size grows
like `|V|^order`. For this corpus the context vocabulary has 11 tokens
(`<END>` can never precede anything), so there are `11² = 121` possible
second-order contexts, of which only **15** were observed and **106 were never
seen**. Each unseen context has no reliable estimate, so probabilities must be
backed off or smoothed; the estimates that *are* observed are based on one or two
examples. Doubling the context length roughly squares the number of parameters,
so with limited data the model becomes brittle even as its theoretical
expressive power rises. This is the **bias–variance / sparsity trade-off** at the
heart of n-gram models.

---

## Question 13 — Why is Approach B preferable?

> A: "Write a Python language model for me."
> B: "Implement the following probabilistic model: P(X_t | X_{t-1}), estimated
> from transition counts, with sampling-based generation."

Approach B is preferable because it treats the LLM as a *tool for implementing a
specified system* rather than as an oracle:

- **Specifying the intended behaviour** — B states the exact distribution, the
  estimator (counts), and the decoder (sampling). This removes ambiguity and makes
  the result checkable against the specification.
- **Understanding the representation** — because the model is explicit
  (`P(X_t|X_{t-1})` from a CPT), you know what the code *should* contain: a count
  table and a normalisation step. You can therefore read the generated code and
  verify those pieces really exist.
- **Validating the generated implementation** — with a stated spec you can test
  properties such as `Σ_v P(v|w) = 1`, which would silently pass unnoticed if you
  had just asked for "a language model".
- **Testing probabilistic invariants** — invariants are derived from the model,
  not the code, so they catch bugs the code's author (LLM) did not consider.
- **Distinguishing implementation from model** — the model is the conditional
  distribution; the implementation is one program that estimates it. Keeping them
  separate means a buggy implementation can be corrected without changing what is
  being modelled.

In short, Approach A asks the LLM to *define* the system; Approach B asks it to
*implement* a system you already understand and can test.

---

## Question 14 — What did thinking of the language model as a Bayesian network give you?

Thinking of the model as a Bayesian network gave, among others:

1. **A representation of dependencies** — the graph `X_{t-1} → X_t` makes the
   conditional-independence assumptions explicit and readable, and shows exactly
   which variables influence which. Extending it to
   `X_{t-2} → X_t ← X_{t-1}` made the added context visible at a glance.
2. **A factorisation of the joint distribution** — the chain rule turns
   `P(X_1,...,X_T)` into a product of local conditionals, which is what makes both
   estimation (counting) and generation (sampling one factor at a time) possible.
3. **A principled method for generation** — each generated token is a sample from
   one factor of the factorisation; `P(X_1)` is `P(X_1 | <START>)`, and stopping at
   `<END>` is just another conditional outcome. Generation is not an ad-hoc loop,
   it is sampling from the joint.
4. **A way to reason about assumptions and context** — Question 12's
   help-vs-hurt trade-off is a statement about the size of the conditional tables,
   which the BN view makes precise.
5. **A way to test whether an implementation matches its specification** — the BN
   tells you what must be true (`Σ_v P(v|w) = 1`, all probabilities in [0,1],
   argmax equals the row maximum), turning correctness into checkable invariants.

The Bayesian-network view separates the *probabilistic question*
`P(next token | previous tokens)` from the *engineering machinery* that estimates
it — a distinction that carries straight over to modern neural language models.

---

## Reflection — how the LLM was used and how its output was validated

**Workflow.** Understand → design → ask the LLM → implement → test → reflect. I
wrote down the variables, dependencies and the distribution to be estimated
before prompting, and used that specification as the test oracle.

**Validation.** (1) Read the generated code against the spec (where counts are
stored, where `P(X_t|·)` is computed, how the next word is chosen). (2) Tested
probabilistic invariants — `Σ_v P(v|w) = 1` for every row, all probabilities in
`[0,1]`, greedy determinism. (3) Ran the model end-to-end and inspected the
generated sentences rather than trusting one number.

**A concrete bug the LLM omitted.** Generation seeds the second-order model with
`(<START>, <START>)`, but `<START>` never occurred as the *second* element of a
triple, so the back-off row was missing and every generation stopped at the first
step. The fix was to prepend **two** `<START>` tokens in the tokeniser. The
normalisation test passed throughout — only running the model exposed the
boundary condition.

**A second correction.** The worksheet's illustrative `P(cat|the)=3/5` comes from
a five-transition toy corpus, not the six-sentence dataset; on the real data
`the` is a context 12 times, so `P(cat|the)=3/12`. Tests must be written against
the actual data, not a figure quoted in the prompt.

**LLM strengths and limits.** Good at boilerplate — tokenisation, `Counter`
counting, normalisation, weighted sampling and the greedy/sampling switch. Not
good at boundary conditions or at judging whether the *model* is appropriate.
The LLM constructs the system; understanding — what distribution is estimated,
which invariants it must satisfy, where the edge cases are — is what makes
validation possible.
