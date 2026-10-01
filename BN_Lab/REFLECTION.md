# Reflection — How the LLM Was Used and How Its Output Was Validated

**Student:** Pranay Ghumatkar
**Roll No:** 2024A3PS0328G

## 1. The workflow I followed

I followed the laboratory workflow: **Understand → Design → Ask the LLM →
Implement → Test → Reflect.** Before asking for code I wrote down, in words:

- **Variables:** a sequence of tokens `X_1, …, X_T` over the vocabulary
  `{<START>, the, cat, dog, sat, ran, on, to, the, mat, rug, park, <END>}`.
- **Dependencies:** first-order model `P(X_t | X_{t-1})`; second-order model
  `P(X_t | X_{t-2}, X_{t-1})`.
- **Distribution to estimate:** the conditional distribution obtained by
  normalising transition counts.
- **What the program should compute:** count transitions, normalise into a CPT,
  print `P(· | w)`, compute `argmax`, sample sentences and stop at `<END>`.

Only then did I ask the LLM for the implementation, using the behavioural
specification from Part V (and, for the second-order model, the prompt from
Part XII). This ordering matters: the spec is what I tested the code against.

## 2. How I validated the output

Rather than trusting the generated code, I checked three things:

1. **Read the code against the spec.** I located where counts are stored
   (`transition_counts` / `trigram_counts`), where `P(X_t|·)` is computed
   (`_compute_probabilities`), and how the next word is chosen
   (`predict_next` = greedy, `sample_next` = sampling).
2. **Tested probabilistic invariants.** `tests/test_normalisation.py` asserts
   `Σ_v P(v|w) = 1` for every row of both models, that all probabilities lie in
   `[0,1]`, that greedy generation is deterministic, and that the counts match a
   hand computation. This caught a real error — see below.
3. **Ran the model end-to-end.** I generated 20 sampling sentences and 5 greedy
   sentences per model and inspected them for coherence, rather than relying on a
   single number.

## 3. A concrete bug I found and corrected

**Symptom.** The first version of `second_order_lm.py` trained fine and passed
normalisation, but *every* generated sentence was empty:

```
 1.
 2.
...
```

**Diagnosis.** Generation seeds the model with the context `(<START>, <START>)`.
That context was never observed, so the code fell back to the first-order row
`bigram_counts["<START>"]`. But at training time `<START>` only ever occurs as
the **first** element of a triple — it is never the *second* element `w_j`, so
`bigram_counts` never contained a `<START>` key. The back-off lookup therefore
returned `None`, and generation stopped on the first step.

**Correction.** I changed the second-order tokeniser to prepend *two* `<START>`
tokens:

```python
return [START, START] + words + [END]
```

Now the triples `(<START>, <START>, w_1)` and `(<START>, w_1, w_2)` are observed
during training, so `P(X_1 | <START>, <START>)` is well defined and generation
starts correctly. After the fix the second-order model produces coherent
sentences such as `the dog ran to the park` and `the cat sat on the mat`.

This is exactly the kind of edge case an LLM omits: the function "looks right",
normalisation passes, but the *boundary condition* at sequence start is wrong.
The invariant test did not catch it; running the model end-to-end did.

**Second correction.** My first hand-written test asserted the worksheet's
illustrative numbers `P(cat|the) = 3/5` and `P(dog|the) = 2/5`. Those come from
the *five-example toy corpus in Part IV*, not from the six-sentence dataset used
for the lab. On the real dataset, `the` occurs as a context 12 times
(cat×3, dog×3, mat×2, rug×2, park×2), so `P(cat|the) = P(dog|the) = 3/12`. The
test failed and forced me to re-derive the counts by hand — a good reminder to
test against the actual data, not against a figure from the prompt.

## 4. What the LLM was good at, and what it was not

- **Good at:** writing the boilerplate — tokenisation, `defaultdict(Counter)`
  counting, normalisation, weighted sampling with `random.choices`, and the
  greedy-vs-sampling switch. It turned a clear specification into working code
  quickly.
- **Not good at:** boundary conditions (the `<START>` context above) and judgement
  about whether the *model* was appropriate. It also has no way to know that the
  worksheet's `3/5` example and our dataset's `3/12` are different things.

The lesson is that the LLM is a tool for *constructing* the system, not a
replacement for understanding it. The understanding — what distribution is being
estimated, what invariants it must satisfy, where the boundary cases are — is
what let me catch both errors. Model and implementation are different things;
keeping them separate is what makes validation possible.
