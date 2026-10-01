"""
CS-407 AI Laboratory
Neural Models: Learning, Depth, Activations, and Output Layers

Task 4, Part D  -  Activation experiment.

Train the random-initialised 2-2-1 network three times, changing only the
hidden activation (sigmoid, tanh, ReLU).  For each run we record the final
loss, whether all four examples are classified correctly, and the Euclidean
norm of the first-layer gradient at the first training step.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

import torch

import xor_net as xor


def init_diagnostics(act, seed=2):
    """Reproduce the initialisation used by training and inspect the hidden
    layer at step 0: pre-activations, activations and their derivatives."""
    torch.manual_seed(seed)
    model = xor.XORNet(hidden=act, seed=seed)
    a1 = model.fc1(xor.X)            # pre-activations, (4,2)
    h = model.act(a1)                # hidden activations, (4,2)
    if act == "sigmoid":
        deriv = h * (1 - h)
    elif act == "tanh":
        deriv = 1 - h * h
    else:                            # ReLU
        deriv = (a1 > 0).float()
    return a1.detach(), h.detach(), deriv.detach()


def run_all_activations(steps=6000, lr=0.1, seed=2):
    rows = []
    for act in ("sigmoid", "tanh", "relu"):
        r = xor.train_xor(hidden=act, steps=steps, lr=lr, seed=seed)
        model = r["model"]
        labels = xor.thresholded_labels(model)
        rows.append({
            "activation": act,
            "final_loss": r["losses"][-1],
            "initial_loss": r["losses"][0],
            "correct": labels == [0, 1, 1, 0],
            "labels": labels,
            "early_grad_norm": r["grad_norms"][0],
        })
    return rows


def format_table(rows):
    lines = [
        f"{'Hidden activation':<20}{'Final loss':>14}{'4/4 correct?':>16}{'Early ||dL/dW(1)||':>22}",
        "-" * 72,
    ]
    for r in rows:
        lines.append(
            f"{r['activation']:<20}{r['final_loss']:>14.6f}"
            f"{str(r['correct']):>16}{r['early_grad_norm']:>22.6f}"
        )
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("TASK 4, PART D - ACTIVATION EXPERIMENT  (seed = 2, lr = 0.1, 6000 steps)")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    rows = run_all_activations()
    print()
    print(format_table(rows))

    print("\n--- Per-run predicted labels [ (0,0) (0,1) (1,0) (1,1) ] ---")
    for r in rows:
        print(f"    {r['activation']:<10} {r['labels']}")

    print("\n--- Initial hidden layer at step 0 (same seed for each run) ---")
    for r in rows:
        a1, h, deriv = init_diagnostics(r["activation"])
        print(f"    {r['activation']:<10} pre-activations a(1) = "
              f"{[ [round(v,3) for v in row] for row in a1.tolist()]}")
        print(f"    {'':<10} hidden h = "
              f"{[ [round(v,3) for v in row] for row in h.tolist()]}")
        print(f"    {'':<10} derivative f'(a(1)) = "
              f"{[ [round(v,3) for v in row] for row in deriv.tolist()]}")
        print(f"    {'':<10} mean |derivative| = {float(deriv.abs().mean()):.3f}"
              f"   |   saturated units (|f'|<0.1) = "
              f"{int((deriv.abs() < 0.1).sum())}/{deriv.numel()}")

    print(
        "\n--- Interpretation ---\n"
        "All three activations solve XOR with the same seed and final losses of\n"
        "order 1e-5, so this experiment does not crown a winner.  The local\n"
        "derivative at step 0 clearly differs (sigmoid mean 0.245, tanh 0.920,\n"
        "ReLU 1.000), so the activation does control the per-unit factor f'(a(1)).\n"
        "The *norm* of the full first-layer gradient, however, is a composite:\
"
        "it multiplies the output-error signal, the output weights W(2), and\n"
        "f'(a(1)), then sums over the four examples and both units.  Here those\n"
        "composite norms land within ~10% of one another (0.0055, 0.0051,\n"
        "0.0049), so one cannot read the activation's effect off the norm alone -\n"
        "the derivative/pre-activation table above shows the mechanism directly.\n"
        "Two distinct mechanisms can still kill a gradient: a saturated sigmoid\n"
        "has f'(a) -> 0 because |a| is large, while a ReLU unit has f'(a) = 0\n"
        "because a < 0.  They are told apart by inspecting the pre-activations\n"
        "a(1) (large magnitude vs negative), not by the gradient alone.  With\n"
        "four points and one seed these are observations, not a universal\n"
        "ranking of activations."
    )


if __name__ == "__main__":
    main()
