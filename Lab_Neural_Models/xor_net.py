"""
CS-407 AI Laboratory
Neural Models: Learning, Depth, Activations, and Output Layers

Task 3 & 4  -  Binary XOR experiment.

A 2 -> 2 -> 1 network trained on the four XOR examples.  The hidden
activation is configurable (sigmoid / tanh / relu); the output is a single
logit trained with BCEWithLogitsLoss (sigmoid + binary cross-entropy fused
into one numerically stable op).  The class also exposes the first-layer
weight gradient after backward(), which Task 4 Part B asks us to inspect.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Task 1 - the problem specification, in code
# ---------------------------------------------------------------------------
# Input space  X = {0,1}^2, output space Y = {0,1} (yes/no warning).
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]], dtype=torch.float32)

Y = torch.tensor([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]], dtype=torch.float32)

HIDDEN_ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


class XORNet(nn.Module):
    """2 -> 2 -> 1 feed-forward network for XOR.

    The forward pass is a(`1`) = W(`1`)x + b(`1`), h = f(a(`1`)),
    z = W(`2`)h + b(`2`).  We return the *logit* z and let
    BCEWithLogitsLoss apply the sigmoid, which is the numerically stable
    pairing described in the worksheet.
    """

    def __init__(self, hidden="sigmoid", zero_init=False, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        if hidden not in HIDDEN_ACTIVATIONS:
            raise ValueError(f"unknown hidden activation {hidden!r}")
        self.hidden = hidden
        self.fc1 = nn.Linear(2, 2)     # W(1) is (2,2), b(1) is (2,)
        self.fc2 = nn.Linear(2, 1)     # W(2) is (1,2), b(2) is (1,)
        self.act = HIDDEN_ACTIVATIONS[hidden]()

        if zero_init:
            # Task 4 Part C - symmetric initialisation.
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        a1 = self.fc1(x)          # pre-activation of the hidden layer
        h = self.act(a1)          # hidden representation
        z = self.fc2(h)           # output logit
        return z

    # -- convenience --------------------------------------------------------
    def hidden_representation(self, x=X):
        with torch.no_grad():
            return self.act(self.fc1(x))

    def probabilities(self, x=X):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def predictions(self, x=X):
        return (self.probabilities(x) >= 0.5).float()


def train_xor(hidden="sigmoid", steps=6000, lr=0.1, seed=2,
              zero_init=False, optimizer="adam", record_grad=True):
    """Full-batch training on the four XOR examples.

    Returns a dict with the trained model, the loss trace, and (optionally)
    the per-step Euclidean norm of the first-layer gradient.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = XORNet(hidden=hidden, zero_init=zero_init, seed=seed)
    loss_fn = nn.BCEWithLogitsLoss()          # sigmoid + BCE, fused

    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    losses, grad_norms = [], []
    for step in range(steps):
        opt.zero_grad()
        logits = model(X)                     # forward pass
        loss = loss_fn(logits, Y)             # scalar loss
        loss.backward()                       # reverse-mode autodiff
        if record_grad:
            grad_norms.append(float(model.fc1.weight.grad.norm()))
        opt.step()                            # parameter update
        losses.append(loss.item())

    return {"model": model, "losses": losses, "grad_norms": grad_norms}


def thresholded_labels(model):
    return [int(v) for v in model.predictions().flatten().tolist()]


def main():
    print("=" * 70)
    print("TASK 3/4 - BINARY XOR, 2-2-1 NETWORK  (sigmoid hidden, BCEWithLogits)")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    result = train_xor(hidden="sigmoid")
    model, losses = result["model"], result["losses"]

    print("\n--- Part A: basic learning check ---")
    print(f"    initial loss              : {losses[0]:.6f}")
    print(f"    final loss                : {losses[-1]:.6f}")
    probs = model.probabilities().flatten().tolist()
    labels = thresholded_labels(model)
    print("    input          P(y=1)     label   target")
    for (x1, x2), p, lab, t in zip(X.tolist(), probs, labels, Y.flatten().tolist()):
        print(f"    ({x1:.0f},{x2:.0f})          {p:.6f}   {lab}       {int(t)}")
    print(f"    all four correct          : {labels == [int(v) for v in Y.flatten()]}")

    print("\n--- Part B: backpropagation check ---")
    loss_fn = nn.BCEWithLogitsLoss()          # mean reduction over 4 examples

    # (i) the gradient at initialisation: this is the learning signal.
    torch.manual_seed(2)
    fresh = XORNet(hidden="sigmoid", seed=2)
    fresh.zero_grad()
    loss0 = loss_fn(fresh(X), Y)
    loss0.backward()
    print("    at initialisation:")
    print(f"      loss                    : {loss0.item():.6f}")
    print(f"      dL/dW(1) (2x2)          :\n{fresh.fc1.weight.grad}")
    print(f"      ||dL/dW(1)||            : {float(fresh.fc1.weight.grad.norm()):.6f}")
    print(f"      dL/dW(2)                : {fresh.fc2.weight.grad.tolist()}")

    # (ii) the gradient after training: it has shrunk because learning worked.
    model.zero_grad()
    loss = loss_fn(model(X), Y)
    loss.backward()
    print("    after training:")
    print(f"      loss                    : {loss.item():.6f}")
    print(f"      dL/dW(1) (2x2)          :\n{model.fc1.weight.grad}")
    print(f"      ||dL/dW(1)||            : {float(model.fc1.weight.grad.norm()):.8f}")
    print(
        "    The loss uses mean reduction, so dL/dW(1) is the average of the four\n"
        "    example-wise gradients: each of the four rows of X contributes its own\n"
        "    outer-product term and the sum is divided by 4."
    )


if __name__ == "__main__":
    main()
