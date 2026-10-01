"""
CS-407 AI Laboratory
Neural Models: Learning, Depth, Activations, and Output Layers

Task 4, Part C  -  Symmetry experiment.

Every weight (and bias) is set to zero before training, architecture
unchanged.  We then watch the two rows of the hidden-layer weight matrix
W(1) over training.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

import torch
import torch.nn as nn

import xor_net as xor


def run_zero_init(hidden="sigmoid", steps=6000, lr=0.1, seed=0):
    """Train from an all-zero initialisation, tracking the W(1) row gap."""
    torch.manual_seed(seed)
    model = xor.XORNet(hidden=hidden, zero_init=True)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    row_gap, losses, snapshots = [], [], []
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(xor.X), xor.Y)
        loss.backward()
        # ||W(1)[0, :] - W(1)[1, :]|| : how far apart the two hidden units are.
        gap = float((model.fc1.weight[0] - model.fc1.weight[1]).detach().norm())
        row_gap.append(gap)
        opt.step()
        losses.append(loss.item())
        if step < 5:
            snapshots.append((step, model.fc1.weight.detach().clone()))
    return {"model": model, "losses": losses, "row_gap": row_gap,
            "snapshots": snapshots}


def main():
    print("=" * 70)
    print("TASK 4, PART C - SYMMETRY EXPERIMENT  (all weights initialised to 0)")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    result = run_zero_init(hidden="sigmoid")
    model, losses, gaps = result["model"], result["losses"], result["row_gap"]

    print("\n--- W(1) rows over the first steps ---")
    for step, w in result["snapshots"]:
        print(f"    step {step}:  row0 = {w[0].tolist()}   row1 = {w[1].tolist()}")
        print(f"              ||row0 - row1|| = {float((w[0]-w[1]).norm()):.3e}")

    print("\n--- Row gap over training ---")
    print(f"    ||W(1)[0]-W(1)[1]|| at step 0     : {gaps[0]:.3e}")
    print(f"    ||W(1)[0]-W(1)[1]|| at step 100   : {gaps[100]:.3e}")
    print(f"    ||W(1)[0]-W(1)[1]|| at final step : {gaps[-1]:.3e}")
    print(f"    rows remain identical throughout  : {max(gaps) < 1e-12}")

    print("\n--- Learning outcome from the symmetric start ---")
    print(f"    initial loss : {losses[0]:.6f}")
    print(f"    final loss   : {losses[-1]:.6f}  (ln 2 = {torch.log(torch.tensor(2.0)):.6f})")
    print(f"    predictions  : {xor.thresholded_labels(model)}")
    print(f"    all four correct : {xor.thresholded_labels(model) == [0, 1, 1, 0]}")

    print(
        "\nBecause both hidden units start identical and receive the identical\n"
        "gradient, gradient descent keeps them identical forever.  The network\n"
        "behaves as if it had a single hidden unit, which cannot represent XOR:\n"
        "the loss stays at ln(2) and the four predictions never separate."
    )


if __name__ == "__main__":
    main()
