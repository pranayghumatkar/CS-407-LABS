"""
CS-407 AI Laboratory
Neural Models: Learning, Depth, Activations, and Output Layers

Task 5  -  Three-class extension.

The same two binary sensor inputs as XOR, but the task is now three-way:
    0  both sensors inactive  (0,0)
    1  sensors disagree       (0,1) or (1,0)
    2  both sensors active    (1,1)

Only the output/loss portion changes: one logit becomes three logits and
binary cross-entropy becomes multiclass cross-entropy.  The hidden layer
stays 2 -> 2.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import xor_net as xor

X = xor.X                                   # (4,2)
Y_CLASS = torch.tensor([0, 1, 1, 2])        # (4,) class indices
CLASS_MEANING = ["both inactive", "disagreement", "both active"]

HIDDEN_ACTIVATIONS = xor.HIDDEN_ACTIVATIONS


class ThreeClassNet(nn.Module):
    """2 -> 2 -> 3 network.  The output is three logits, one per class."""

    def __init__(self, hidden="sigmoid", seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.hidden = hidden
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 3)      # W(2) is (3,2), b(2) is (3,)
        self.act = HIDDEN_ACTIVATIONS[hidden]()

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)              # logits, shape (4,3)

    def logits(self, x=X):
        with torch.no_grad():
            return self.forward(x)

    def probabilities(self, x=X):
        """Softmax over the three logits (row-stochastic)."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    def predicted_classes(self, x=X):
        return [int(c) for c in self.probabilities(x).argmax(dim=-1).tolist()]


def train_three_class(hidden="sigmoid", steps=6000, lr=0.1, seed=2):
    if seed is not None:
        torch.manual_seed(seed)
    model = ThreeClassNet(hidden=hidden, seed=seed)
    loss_fn = nn.CrossEntropyLoss()     # softmax + NLL, fused and stable
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(X), Y_CLASS)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return {"model": model, "losses": losses}


def softmax_naive(logits):
    """Textbook softmax *without* max-subtraction (for the stability check)."""
    exps = torch.exp(logits)
    return exps / exps.sum(dim=-1, keepdim=True)


def softmax_stable(logits):
    """Numerically stable softmax: subtract the row maximum before exp."""
    z = logits - logits.max(dim=-1, keepdim=True).values
    exps = torch.exp(z)
    return exps / exps.sum(dim=-1, keepdim=True)


def main():
    print("=" * 70)
    print("TASK 5 - THREE-CLASS EXTENSION  (2-2-3, softmax + cross-entropy)")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    result = train_three_class(hidden="sigmoid")
    model, losses = result["model"], result["losses"]

    print("\n--- Training ---")
    print(f"    initial loss : {losses[0]:.6f}")
    print(f"    final loss   : {losses[-1]:.6f}")
    print(f"    W(2) shape   : {tuple(model.fc2.weight.shape)}  (3 classes x 2 hidden units)")
    print(f"    logits per example : {tuple(model.logits().shape)}")

    probs = model.probabilities()
    preds = model.predicted_classes()
    print("\n--- Class probabilities for each input ---")
    print("    input     class 0     class 1     class 2    predicted  target")
    for (x1, x2), row, pred, tgt in zip(X.tolist(), probs.tolist(), preds, Y_CLASS.tolist()):
        print(f"    ({x1:.0f},{x2:.0f})    {row[0]:.6f}   {row[1]:.6f}   {row[2]:.6f}"
              f"      {pred}         {tgt}")
    print(f"\n    all four correct : {preds == Y_CLASS.tolist()}")

    print("\n--- Softmax sanity checks ---")
    for i, row in enumerate(probs.tolist()):
        print(f"    example {i}: sum = {sum(row):.10f}")
    print(f"    all rows sum to 1 : {all(abs(sum(r) - 1.0) < 1e-6 for r in probs.tolist())}")

    # Optional diagnostic: add +100 to every logit and check shift-invariance.
    # Mathematically softmax(z) == softmax(z + c); numerically the *naive*
    # version overflows, which is exactly why we subtract the row maximum.
    logits = model.logits()
    print("\n--- Shift-invariance diagnostic (add +100 to all logits) ---")
    naive = softmax_naive(logits)
    naive_100 = softmax_naive(logits + 100.0)
    stable = softmax_stable(logits)
    stable_100 = softmax_stable(logits + 100.0)
    print(f"    naive softmax at z        : {[round(v, 6) for v in naive[0].tolist()]}")
    print(f"    naive softmax at z+100    : {[None if v != v else round(float(v), 6) for v in naive_100[0].tolist()]}")
    print(f"    any NaN in naive(z+100)   : {bool(torch.isnan(naive_100).any())}  <- exp(100) overflows")
    print(f"    max |stable(z) - stable(z+100)| = {float((stable - stable_100).abs().max()):.3e}")
    print(f"    stable matches F.softmax       : {bool(torch.allclose(stable, probs, atol=1e-6))}")
    print(
        "    Adding a constant c multiplies numerator and denominator by exp(c),\n"
        "    so the ratio is unchanged mathematically.  In floating point,\n"
        "    exp(large) overflows to +inf (and inf/inf = NaN), so implementations\n"
        "    subtract the row maximum first: exp(z - max z) is in (0, 1] and safe."
    )

    print("\n--- What p - y means ---")
    row = probs[0].tolist()
    onehot = [1.0 if i == Y_CLASS[0].item() else 0.0 for i in range(3)]
    print(f"    example 0 softmax p   : {[round(v, 6) for v in row]}")
    print(f"    example 0 one-hot y   : {onehot}")
    print(f"    p - y                 : {[round(a - b, 6) for a, b in zip(row, onehot)]}")
    print(
        "    With softmax plus cross-entropy the gradient of the loss with respect\n"
        "    to the logits is exactly p - y, so a confident-but-wrong class pushes\n"
        "    that logit down and the correct logit up in proportion to the error."
    )


if __name__ == "__main__":
    main()
