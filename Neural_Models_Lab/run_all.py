"""
CS-407 AI Laboratory
Neural Models: Learning, Depth, Activations, and Output Layers

Driver: runs every experiment in the lab and writes the result files used in
the submission into results/.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

import contextlib
import io
import os

import activation_experiment as ae
import gradient_check as gc
import symmetry_experiment as se
import three_class_net as tc
import xor_net as xor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


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
    write("xor_training.txt", capture(xor.main))
    write("symmetry_experiment.txt", capture(se.main))
    write("activation_experiment.txt", capture(ae.main))
    write("three_class.txt", capture(tc.main))
    write("gradient_check.txt", capture(gc.main))

    # ---- compact summary table -------------------------------------------
    rows = ae.run_all_activations()
    headers = ["Hidden activation", "Final loss", "4/4 correct?",
               "Early ||dL/dW(1)||"]
    lines = [
        "CS-407 Lab - Neural Models summary",
        "Student: Pranay Ghumatkar | 2024A3PS0328G",
        "",
        f"{headers[0]:<20}{headers[1]:>14}{headers[2]:>16}{headers[3]:>20}",
        "-" * 70,
    ]
    for r in rows:
        lines.append(f"{r['activation']:<20}{r['final_loss']:>14.6f}"
                     f"{str(r['correct']):>16}{r['early_grad_norm']:>20.6f}")

    xor_result = xor.train_xor(hidden="sigmoid")
    tc_result = tc.train_three_class(hidden="sigmoid")
    zero = se.run_zero_init(hidden="sigmoid")
    lines += [
        "",
        "--- binary XOR (2-2-1, sigmoid hidden, BCEWithLogits) ---",
        f"    final loss              : {xor_result['losses'][-1]:.6f}",
        f"    predictions             : {xor.thresholded_labels(xor_result['model'])}",
        f"    all four correct        : {xor.thresholded_labels(xor_result['model']) == [0, 1, 1, 0]}",
        "",
        "--- three-class extension (2-2-3, softmax + cross-entropy) ---",
        f"    final loss              : {tc_result['losses'][-1]:.6f}",
        f"    predicted classes       : {tc_result['model'].predicted_classes()}",
        f"    all four correct        : {tc_result['model'].predicted_classes() == [0, 1, 1, 2]}",
        "",
        "--- symmetry experiment (all-zero init) ---",
        f"    final loss              : {zero['losses'][-1]:.6f}  (ln 2 = 0.693147)",
        f"    max row gap ||W1[0]-W1[1]|| : {max(zero['row_gap']):.3e}",
        f"    rows stay identical     : {max(zero['row_gap']) < 1e-12}",
    ]
    write("summary.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
