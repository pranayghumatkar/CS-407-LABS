"""
CS-407 AI Laboratory - Bayesian Networks and Autoregressive Language Models

Part XIII - Driver that runs both models, writes every result file used in the
submission into results/, and prints the first-order vs second-order
comparison.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

import contextlib
import io
import os

import first_order_lm as fo
import second_order_lm as so

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

WORDS = ["the", "cat", "dog", "sat", "ran", "on", "mat", "rug"]


def capture(func):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        func()
    return buf.getvalue()


def write(name, text):
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {os.path.relpath(path, HERE)}")


def main():
    # ---- full textual reports -------------------------------------------
    write("first_order_report.txt", capture(fo.main))
    write("second_order_report.txt", capture(so.main))

    # ---- normalisation tests --------------------------------------------
    m1 = fo.FirstOrderLanguageModel()
    m1.train(fo.tokenised_dataset())
    lines = ["CS-407 Lab - Probability normalisation test (first-order)",
             "Property: for every current word w, sum_v P(v | w) = 1",
             "Student: Pranay Ghumatkar | 2024A3PS0328G", ""]
    for word, total in m1.normalisation_check():
        lines.append(f"  P(v | {word:<8}) totals to {total:.6f}")
    write("normalisation_first_order.txt", "\n".join(lines) + "\n")

    m2 = so.SecondOrderLanguageModel()
    m2.train(so.tokenised_dataset())
    lines = ["CS-407 Lab - Probability normalisation test (second-order)",
             "Property: for every context (w1,w2), sum_v P(v | w1,w2) = 1", ""]
    for context, total in m2.normalisation_check():
        lines.append(f"  P(v | {str(context):<20}) totals to {total:.6f}")
    write("normalisation_second_order.txt", "\n".join(lines) + "\n")

    # ---- generated text --------------------------------------------------
    lines = ["CS-407 Lab - Generated sentences (first-order)",
             "Mode: sampling from P(X_t | X_{t-1}), seed = 1234", ""]
    for i, s in enumerate(m1.generate_many(20, "sample", seed=1234), 1):
        lines.append(f"{i:>2}. {s}")
    write("generated_first_order_sampling.txt", "\n".join(lines) + "\n")

    lines = ["CS-407 Lab - Generated sentences (first-order, Mode A greedy)", ""]
    for i, s in enumerate(m1.generate_many(5, "greedy"), 1):
        lines.append(f"{i}. {s}")
    lines += ["", "Mode B: sampling from P(X_t | X_{t-1}), seed = 99", ""]
    for i, s in enumerate(m1.generate_many(5, "sample", seed=99), 1):
        lines.append(f"{i}. {s}")
    write("generated_first_order_modes.txt", "\n".join(lines) + "\n")

    lines = ["CS-407 Lab - Generated sentences (second-order)",
             "Mode: sampling from P(X_t | X_{t-2}, X_{t-1}) with first-order backoff",
             "seed = 1234", ""]
    for i, s in enumerate(m2.generate_many(20, "sample", seed=1234), 1):
        lines.append(f"{i:>2}. {s}")
    write("generated_second_order_sampling.txt", "\n".join(lines) + "\n")

    # ---- comparison ------------------------------------------------------
    # Part XIII also asks for a measure of the DIVERSITY of generated text.
    # For each model we draw 1000 sentences and count (a) how many are
    # distinct and (b) how many are novel (not one of the six training
    # sentences).  A model that just memorises the corpus has low diversity.
    training = set(fo.SENTENCES)

    def diversity(model, n=1000, seed=7):
        sents = model.generate_many(n, "sample", seed=seed)
        distinct = set(sents)
        novel = {s for s in distinct if s not in training}
        reproduced = sum(s in training for s in sents) / len(sents)
        return len(distinct), len(novel), reproduced

    div1 = diversity(m1)
    div2 = diversity(m2)

    rows = [
        ("Distinct non-zero parameters", f"{m1.n_parameters()}", f"{m2.n_parameters()}"),
        ("Observed context rows", f"{len(m1.probabilities)}", f"{m2.n_observed_contexts()}"),
        ("Possible contexts (|V|-1) / (|V|-1)^2", f"{len(m1.context_vocabulary())}", f"{m2.n_possible_contexts()}"),
        ("Contexts never observed", f"{len(m1.context_vocabulary()) - len(m1.probabilities)}", f"{m2.n_unseen_possible_contexts()}"),
        ("Observed rows containing a zero", f"{len(m1.zero_probability_contexts())}", f"{len(m2.zero_probability_contexts())}"),
        ("Vocabulary size (incl. START/END)", f"{len(m1.vocabulary)}", f"{len(m2.vocabulary)}"),
        ("Distinct sentences in 1000 samples", f"{div1[0]}", f"{div2[0]}"),
        ("Novel sentences (not in training)", f"{div1[1]}", f"{div2[1]}"),
        ("Training sentences reproduced verbatim", f"{div1[2]*100:.1f}%", f"{div2[2]*100:.1f}%"),
    ]
    lines = ["CS-407 Lab - First-order vs second-order comparison",
             "Student: Pranay Ghumatkar | 2024A3PS0328G", "",
             f"{'Metric':<42}{'1st order':>14}{'2nd order':>14}",
             "-" * 70]
    lines += [f"{name:<42}{a:>14}{b:>14}" for name, a, b in rows]
    write("comparison.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
