"""
CS-407 AI Laboratory
Bayesian Networks and Autoregressive Language Models

Part XI & XII - Second-order autoregressive language model.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G

The model estimates

        P(X_t | X_{t-2}, X_{t-1})

from counts of observed triples (X_{t-2}, X_{t-1}, X_t):

        P(w_k | w_i, w_j) = C(w_i, w_j, w_k) / sum_m C(w_i, w_j, w_m)

While generating, if a bigram context (w_i, w_j) was never observed, the
model backs off to the first-order distribution P(w_k | w_j).  This keeps
generation alive under sparse data and is reported as the number of
zero-probability second-order contexts.

No machine-learning library and no pretrained model are used.
"""

from collections import Counter, defaultdict
import random

START = "<START>"
END = "<END>"


SENTENCES = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenise(sentence):
    # Two <START> tokens are prepended so that the second-order context
    # (<START>, <START>) and (<START>, w_1) are both observed during
    # training.  This makes P(X_1 | <START>, <START>) = P(X_1 | <START>)
    # well defined and lets generation start without needing a backoff.
    words = sentence.lower().split()
    return [START, START] + words + [END]


def tokenised_dataset(sentences=SENTENCES):
    return [tokenise(s) for s in sentences]


class SecondOrderLanguageModel:
    """A second-order Markov / trigram language model."""

    def __init__(self):
        self.trigram_counts = defaultdict(Counter)   # C(w_i, w_j, w_k)
        self.bigram_counts = defaultdict(Counter)    # C(w_j, w_k)  (backoff)
        self.probabilities = {}                      # P(w_k | w_i, w_j)
        self.vocabulary = set()

    # -- training -----------------------------------------------------------
    def train(self, tokenised_sentences):
        """Count triples.  This is where the transition counts are stored:
        `self.trigram_counts[(w_i, w_j)]` maps each observed next token w_k
        to the number of times the triple (w_i, w_j, w_k) occurred."""
        self.trigram_counts = defaultdict(Counter)
        self.bigram_counts = defaultdict(Counter)
        for tokens in tokenised_sentences:
            self.vocabulary.update(tokens)
            for i in range(len(tokens) - 2):
                w_i, w_j, w_k = tokens[i], tokens[i + 1], tokens[i + 2]
                self.trigram_counts[(w_i, w_j)][w_k] += 1
                self.bigram_counts[w_j][w_k] += 1
        self._compute_probabilities()

    def _compute_probabilities(self):
        """Normalise the triple counts into P(X_t | X_{t-2}, X_{t-1})."""
        self.probabilities = {}
        for context, next_counts in self.trigram_counts.items():
            total = sum(next_counts.values())
            self.probabilities[context] = {
                next_word: count / total
                for next_word, count in next_counts.items()
            }

    # -- queries ------------------------------------------------------------
    def distribution(self, context):
        """P(next | (w_i, w_j)); empty dict if the context is unseen."""
        return self.probabilities.get(tuple(context), {})

    def _row(self, context):
        """Return the distribution, backing off to first order if needed."""
        dist = self.distribution(context)
        if dist:
            return dist, "second-order"
        backoff = self.bigram_counts.get(context[1])
        if backoff:
            total = sum(backoff.values())
            return {w: c / total for w, c in backoff.items()}, "backed-off"
        return {}, "unseen"

    def predict_next(self, context):
        dist, _ = self._row(context)
        if not dist:
            return None
        return max(sorted(dist), key=lambda w: dist[w])

    def sample_next(self, context, rng=None):
        dist, _ = self._row(context)
        if not dist:
            return None
        rng = rng or random
        words, weights = list(dist.keys()), list(dist.values())
        return rng.choices(words, weights=weights, k=1)[0]

    # -- generation ---------------------------------------------------------
    def generate(self, mode="sample", max_tokens=50, rng=None):
        """Generate by repeatedly sampling/argmax from
        P(w_k | w_{k-2}, w_{k-1}); stop at <END>."""
        rng = rng or random
        history = [START, START]     # two <START> tokens seed the model
        tokens = []
        for _ in range(max_tokens):
            if mode == "greedy":
                nxt = self.predict_next(history[-2:])
            else:
                nxt = self.sample_next(history[-2:], rng=rng)
            if nxt is None:
                break
            if nxt == END:
                break
            tokens.append(nxt)
            history.append(nxt)
        return " ".join(tokens)

    def generate_many(self, n, mode="sample", seed=1234):
        rng = random.Random(seed)
        return [self.generate(mode=mode, rng=rng) for _ in range(n)]

    # -- diagnostics --------------------------------------------------------
    def normalisation_check(self):
        rows = []
        for context in sorted(self.probabilities):
            rows.append((context, sum(self.probabilities[context].values())))
        return rows

    def n_parameters(self):
        return sum(len(row) for row in self.probabilities.values())

    def n_observed_contexts(self):
        return len(self.probabilities)

    def n_possible_contexts(self):
        """All bigram contexts that could in principle occur."""
        return len(self.vocabulary) ** 2

    def zero_probability_contexts(self):
        """Observed contexts whose CPT row contains at least one zero."""
        vocab = sorted(self.vocabulary)
        zeros = []
        for context in sorted(self.probabilities):
            missing = [v for v in vocab if self.probabilities[context].get(v, 0.0) == 0.0]
            if missing:
                zeros.append((context, missing))
        return zeros

    def n_unseen_possible_contexts(self):
        """Number of bigram contexts possible in principle but never observed."""
        observed = {c for c in self.probabilities}
        count = 0
        for a in self.vocabulary:
            for b in self.vocabulary:
                if (a, b) not in observed and (a, b) not in self.trigram_counts:
                    count += 1
        return count


def main():
    model = SecondOrderLanguageModel()
    model.train(tokenised_dataset())

    print("=" * 70)
    print("SECOND-ORDER AUTOREGRESSIVE LANGUAGE MODEL  (P(X_t | X_{t-2}, X_{t-1}))")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    print("\n--- Conditional probability tables for selected contexts ---")
    for ctx in [("the", "cat"), ("the", "dog"), ("cat", "sat"),
                ("dog", "sat"), ("cat", "ran"), ("dog", "ran"),
                ("on", "the"), ("sat", "on")]:
        dist = model.distribution(ctx)
        if not dist:
            print(f"\nP(next | {ctx}) = (context never observed)")
            continue
        print(f"\nP(next | {ctx}):")
        for nxt, p in sorted(dist.items(), key=lambda kv: -kv[1]):
            print(f"    P({nxt!r} | {ctx}) = {p:.4f}")

    print("\n--- Normalisation test (rows must sum to 1.0) ---")
    for context, total in model.normalisation_check():
        flag = "OK" if abs(total - 1.0) < 1e-9 else "FAIL"
        print(f"    {str(context):<22} sum = {total:.6f}   [{flag}]")

    print("\n--- Argmax predictions for selected contexts ---")
    for ctx in [("the", "cat"), ("the", "dog"), ("cat", "sat"),
                ("dog", "sat"), ("cat", "ran"), ("dog", "ran"),
                ("on", "the"), ("sat", "on")]:
        print(f"    arg max P(w | {ctx}) = {model.predict_next(ctx)!r}")

    print("\n--- 20 generated sentences (sampling, seed=1234) ---")
    for i, s in enumerate(model.generate_many(20, mode="sample", seed=1234), 1):
        print(f"    {i:>2}. {s}")

    print("\n--- Greedy generation (5 sentences) ---")
    for i, s in enumerate(model.generate_many(5, mode="greedy"), 1):
        print(f"    {i}. {s}")

    print("\n--- Model statistics ---")
    print(f"    distinct non-zero parameters     : {model.n_parameters()}")
    print(f"    observed second-order contexts   : {model.n_observed_contexts()}")
    print(f"    possible contexts (|V|^2)        : {model.n_possible_contexts()}")
    print(f"    unseen possible contexts         : {model.n_unseen_possible_contexts()}")
    print(f"    observed rows with a zero entry  : {len(model.zero_probability_contexts())}")


if __name__ == "__main__":
    main()
