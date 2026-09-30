"""
CS-407 AI Laboratory
Bayesian Networks and Autoregressive Language Models

Part V & VI  -  First-order (bigram) autoregressive language model.
Part VIII/X  -  Next-word prediction, greedy vs sampling generation.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G

The model estimates the conditional distribution

        P(X_t | X_{t-1})

from counts of adjacent token pairs in the training data:

        P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)

No machine-learning library and no pretrained model are used, as required
by the worksheet.
"""

from collections import Counter, defaultdict
import random

START = "<START>"
END = "<END>"


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
SENTENCES = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenise(sentence):
    """Lower-case a sentence, split on whitespace and add <START>/<END>."""
    words = sentence.lower().split()
    return [START] + words + [END]


def tokenised_dataset(sentences=SENTENCES):
    return [tokenise(s) for s in sentences]


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class FirstOrderLanguageModel:
    """A first-order Markov / bigram language model."""

    def __init__(self):
        self.transition_counts = defaultdict(Counter)   # C(w_i, w_j)
        self.probabilities = {}                          # P(w_j | w_i)
        self.vocabulary = set()

    # -- training -----------------------------------------------------------
    def train(self, tokenised_sentences):
        """Count transitions C(w_i, w_j) over the training corpus.

        This is where the transition counts are stored:
        `self.transition_counts[previous_word]` is a Counter mapping each
        observed next word to the number of times it followed previous_word.
        """
        self.transition_counts = defaultdict(Counter)
        for tokens in tokenised_sentences:
            self.vocabulary.update(tokens)
            for current_word, next_word in zip(tokens, tokens[1:]):
                self.transition_counts[current_word][next_word] += 1
        self._compute_probabilities()

    # -- P(X_t | X_{t-1}) ---------------------------------------------------
    def _compute_probabilities(self):
        """Normalise the counts into P(X_t | X_{t-1}).

        This is where P(X_t | X_{t-1}) is computed:
        each next-word count is divided by the row total for that context.
        """
        self.probabilities = {}
        for current_word, next_counts in self.transition_counts.items():
            total = sum(next_counts.values())
            self.probabilities[current_word] = {
                next_word: count / total
                for next_word, count in next_counts.items()
            }

    # -- queries ------------------------------------------------------------
    def distribution(self, current_word):
        """Return P(next word | current_word) as a dict."""
        return self.probabilities.get(current_word, {})

    def predict_next(self, current_word):
        """Greedy: return argmax_w P(w | current_word)."""
        dist = self.distribution(current_word)
        if not dist:
            return None
        # Tie-break alphabetically so results are deterministic.
        return max(sorted(dist), key=lambda w: dist[w])

    def sample_next(self, current_word, rng=None):
        """Sample a next word from P(next word | current_word).

        random.choices performs inverse-transform sampling over the
        probability weights, so tokens are drawn with their true probability.
        """
        dist = self.distribution(current_word)
        if not dist:
            return None
        rng = rng or random
        words = list(dist.keys())
        weights = list(dist.values())
        return rng.choices(words, weights=weights, k=1)[0]

    # -- generation ---------------------------------------------------------
    def generate(self, mode="sample", max_tokens=50, rng=None):
        """Generate one sentence.

        mode = "greedy"  -> always pick argmax P(w | previous)
        mode = "sample"  -> sample from P(w | previous)

        Generation starts at P(X_1) = P(w | <START>) and stops when the
        <END> token is produced (or max_tokens is reached).
        """
        rng = rng or random
        current = START
        tokens = []
        for _ in range(max_tokens):
            if mode == "greedy":
                nxt = self.predict_next(current)
            else:
                nxt = self.sample_next(current, rng=rng)
            if nxt is None:          # unobserved context -> cannot continue
                break
            if nxt == END:
                break
            tokens.append(nxt)
            current = nxt
        return " ".join(tokens)

    def generate_many(self, n, mode="sample", seed=1234):
        rng = random.Random(seed)
        return [self.generate(mode=mode, rng=rng) for _ in range(n)]

    # -- diagnostics --------------------------------------------------------
    def normalisation_check(self):
        """Return (word, row_total) for every context. Property: total == 1."""
        rows = []
        for word in sorted(self.probabilities):
            rows.append((word, sum(self.probabilities[word].values())))
        return rows

    def n_parameters(self):
        """Number of non-zero parameters in the CPT."""
        return sum(len(row) for row in self.probabilities.values())

    def zero_probability_contexts(self):
        """Contexts (rows) that contain at least one structurally absent
        transition, i.e. some vocabulary word with count zero."""
        vocab = sorted(self.vocabulary)
        zeros = []
        for word in sorted(self.probabilities):
            missing = [v for v in vocab if self.probabilities[word].get(v, 0.0) == 0.0]
            # a row is a "zero-probability context" if at least one possible
            # continuation has probability zero
            if missing:
                zeros.append((word, missing))
        return zeros


# ---------------------------------------------------------------------------
# 3. Demonstration
# ---------------------------------------------------------------------------
def main():
    model = FirstOrderLanguageModel()
    model.train(tokenised_dataset())

    print("=" * 70)
    print("FIRST-ORDER AUTOREGRESSIVE LANGUAGE MODEL  (P(X_t | X_{t-1}))")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    print("\n--- Conditional probability tables for selected contexts ---")
    for word in ["the", "cat", "dog", "sat", "ran", "on", "mat", "rug"]:
        dist = model.distribution(word)
        if not dist:
            print(f"\nP(next | {word!r}) = (not observed as a context)")
            continue
        print(f"\nP(next | {word!r}):")
        for nxt, p in sorted(dist.items(), key=lambda kv: -kv[1]):
            print(f"    P({nxt!r} | {word!r}) = {p:.4f}")

    print("\n--- Normalisation test (rows must sum to 1.0) ---")
    for word, total in model.normalisation_check():
        flag = "OK" if abs(total - 1.0) < 1e-9 else "FAIL"
        print(f"    {word:<8} sum = {total:.6f}   [{flag}]")

    print("\n--- Argmax predictions for selected contexts ---")
    for word in ["the", "cat", "dog", "sat", "ran", "on", "mat", "rug"]:
        print(f"    arg max P(w | {word!r}) = {model.predict_next(word)!r}")

    print("\n--- 20 generated sentences (sampling, seed=1234) ---")
    for i, s in enumerate(model.generate_many(20, mode="sample", seed=1234), 1):
        print(f"    {i:>2}. {s}")

    print("\n--- Mode A: greedy generation (5 sentences) ---")
    for i, s in enumerate(model.generate_many(5, mode="greedy"), 1):
        print(f"    {i}. {s}")

    print("\n--- Mode B: sampling generation (5 sentences) ---")
    for i, s in enumerate(model.generate_many(5, mode="sample", seed=99), 1):
        print(f"    {i}. {s}")

    print("\n--- Model statistics ---")
    print(f"    distinct non-zero parameters : {model.n_parameters()}")
    print(f"    observed context rows        : {len(model.probabilities)}")
    print(f"    vocabulary size              : {len(model.vocabulary)}")


if __name__ == "__main__":
    main()
