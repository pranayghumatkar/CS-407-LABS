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
    lines = ["CS-407 Lab - First-order vs second-order comparison",
             "Student: Pranay Ghumatkar | 2024A3PS0328G", "",
             f"{'Metric':<40}{'1st order':>14}{'2nd order':>14}",
             "-" * 68,
             f"{'Distinct non-zero parameters':<40}{m1.n_parameters():>14}{m2.n_parameters():>14}",
             f"{'Observed context rows':<40}{len(m1.probabilities):>14}{m2.n_observed_contexts():>14}",
             f"{'Possible contexts (|V|^2)':<40}{'-':>14}{m2.n_possible_contexts():>14}",
             f"{'Unseen possible contexts':<40}{'-':>14}{m2.n_unseen_possible_contexts():>14}",
             f"{'Observed rows containing a zero':<40}{len(m1.zero_probability_contexts()):>14}{len(m2.zero_probability_contexts()):>14}",
             f"{'Vocabulary size':<40}{len(m1.vocabulary):>14}{len(m2.vocabulary):>14}",
             ]
    write("comparison.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
