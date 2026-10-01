"""
CS-407 AI Laboratory
Neural Models: Learning, Depth, Activations, and Output Layers

The worksheet asks to submit "one notebook or Python file plus a short
report", so this single file contains every program required:

  * binary XOR 2-2-1 network + gradient inspection   (Tasks 3 / 4 A-B)
  * zero-initialisation symmetry experiment          (Task 4 C)
  * sigmoid / tanh / ReLU comparison                 (Task 4 D)
  * three-class softmax extension                    (Task 5)
  * autograd vs finite-difference gradient check

Run:  python neural_models_lab.py

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Data (Task 1)
# ---------------------------------------------------------------------------
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
Y_CLASS = torch.tensor([0, 1, 1, 2])
XOR_LABELS = [0, 1, 1, 0]

HIDDEN_ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}
STEPS, LR, SEED = 6000, 0.1, 2


# ---------------------------------------------------------------------------
# Binary XOR network (Tasks 2-4)
# ---------------------------------------------------------------------------
class XORNet(nn.Module):
    """2 -> 2 -> 1 network.  Returns a logit; BCEWithLogitsLoss applies sigmoid."""

    def __init__(self, hidden="sigmoid", zero_init=False, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.act = HIDDEN_ACTIVATIONS[hidden]()
        if zero_init:
            for p in self.parameters():
                nn.init.zeros_(p)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def probabilities(self, x=X):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def predictions(self, x=X):
        return (self.probabilities(x) >= 0.5).float()


def train_xor(hidden="sigmoid", steps=STEPS, lr=LR, seed=SEED, zero_init=False):
    torch.manual_seed(seed)
    model = XORNet(hidden=hidden, zero_init=zero_init, seed=seed)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


def labels_of(model):
    return [int(v) for v in model.predictions().flatten().tolist()]


# ---------------------------------------------------------------------------
# Three-class network (Task 5)
# ---------------------------------------------------------------------------
class ThreeClassNet(nn.Module):
    """2 -> 2 -> 3 network; output is three logits trained with cross-entropy."""

    def __init__(self, hidden="sigmoid", seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 3)
        self.act = HIDDEN_ACTIVATIONS[hidden]()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def probabilities(self, x=X):
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    def predicted_classes(self, x=X):
        return [int(c) for c in self.probabilities(x).argmax(dim=-1).tolist()]


def train_three_class(hidden="sigmoid", steps=STEPS, lr=LR, seed=SEED):
    torch.manual_seed(seed)
    model = ThreeClassNet(hidden=hidden, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), Y_CLASS)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


def softmax_naive(logits):
    exps = torch.exp(logits)
    return exps / exps.sum(dim=-1, keepdim=True)


def softmax_stable(logits):
    z = logits - logits.max(dim=-1, keepdim=True).values
    exps = torch.exp(z)
    return exps / exps.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Gradient check and experiments
# ---------------------------------------------------------------------------
def analytic_gradient(model, loss_fn, inputs=X, targets=Y):
    model.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    return loss.item(), model.fc1.weight.grad.clone()


def finite_difference_gradient(model, loss_fn, inputs=X, targets=Y, eps=1e-5):
    grad = torch.zeros_like(model.fc1.weight)
    with torch.no_grad():
        for i in range(model.fc1.weight.shape[0]):
            for j in range(model.fc1.weight.shape[1]):
                original = model.fc1.weight[i, j].item()
                model.fc1.weight[i, j] = original + eps
                lp = loss_fn(model(inputs), targets).item()
                model.fc1.weight[i, j] = original - eps
                lm = loss_fn(model(inputs), targets).item()
                model.fc1.weight[i, j] = original
                grad[i, j] = (lp - lm) / (2 * eps)
    return grad


def gradient_check():
    """Run in float64: float32 roundoff would masquerade as a mismatch."""
    torch.manual_seed(SEED)
    model = XORNet(hidden="sigmoid", seed=SEED).double()
    xd, yd = X.double(), Y.double()
    loss_fn = nn.BCEWithLogitsLoss()
    loss, analytic = analytic_gradient(model, loss_fn, xd, yd)
    numeric = finite_difference_gradient(model, loss_fn, xd, yd, eps=1e-5)
    return loss, analytic, numeric, float((analytic - numeric).abs().max())


def symmetry_experiment(hidden="sigmoid", steps=STEPS, lr=LR, seed=0):
    torch.manual_seed(seed)
    model = XORNet(hidden=hidden, zero_init=True)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gaps, losses = [], []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        gaps.append(float((model.fc1.weight[0] - model.fc1.weight[1]).detach().norm()))
        opt.step()
        losses.append(loss.item())
    return model, losses, gaps


def activation_experiment():
    rows = []
    for act in ("sigmoid", "tanh", "relu"):
        model, losses = train_xor(hidden=act)
        torch.manual_seed(SEED)
        fresh = XORNet(hidden=act, seed=SEED)
        fresh.zero_grad()
        nn.BCEWithLogitsLoss()(fresh(X), Y).backward()
        rows.append({"activation": act, "final_loss": losses[-1],
                     "correct": labels_of(model) == XOR_LABELS,
                     "labels": labels_of(model),
                     "early_grad": float(fresh.fc1.weight.grad.norm())})
    return rows


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("NEURAL MODELS - single-file submission")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    # ---- Task 4 Part A/B: XOR + backprop --------------------------------
    model, losses = train_xor(hidden="sigmoid")
    print("\n--- Task 4A: binary XOR (2-2-1, sigmoid, BCEWithLogits) ---")
    print(f"    initial loss : {losses[0]:.6f}")
    print(f"    final loss   : {losses[-1]:.6f}")
    print("    input     P(y=1)      label  target")
    for (x1, x2), p, lab, t in zip(X.tolist(), model.probabilities().flatten().tolist(),
                                   labels_of(model), Y.flatten().tolist()):
        print(f"    ({x1:.0f},{x2:.0f})     {p:.6f}    {lab}      {int(t)}")
    print(f"    all four correct : {labels_of(model) == XOR_LABELS}")

    print("\n--- Task 4B: backpropagation check ---")
    loss0, analytic, numeric, max_err = gradient_check()
    print(f"    dL/dW(1) from autograd : {analytic.tolist()}")
    print(f"    dL/dW(1) from finite differences : {numeric.tolist()}")
    print(f"    max |autograd - finite difference| : {max_err:.3e}  (float64)")
    print("    loss uses mean reduction, so dL/dW(1) is the average of the")
    print("    four example-wise gradients.")

    # ---- Task 4C: symmetry ----------------------------------------------
    _, sym_losses, gaps = symmetry_experiment()
    print("\n--- Task 4C: symmetry experiment (all weights = 0) ---")
    print(f"    ||W1[0]-W1[1]|| at step 0 / 100 / final : {gaps[0]:.1e} / {gaps[100]:.1e} / {gaps[-1]:.1e}")
    print(f"    rows remain identical : {max(gaps) < 1e-12}")
    print(f"    initial / final loss  : {sym_losses[0]:.6f} / {sym_losses[-1]:.6f}  (ln 2 = 0.693147)")
    print("    => the two hidden units never separate, so the network behaves")
    print("       like one linear unit and cannot represent XOR.")

    # ---- Task 4D: activations -------------------------------------------
    print("\n--- Task 4D: activation experiment (seed 2) ---")
    print(f"    {'activation':<12}{'final loss':>14}{'4/4?':>8}{'early ||dL/dW1||':>20}")
    for r in activation_experiment():
        print(f"    {r['activation']:<12}{r['final_loss']:>14.6f}{str(r['correct']):>8}{r['early_grad']:>20.6f}")

    # ---- Task 5: three-class --------------------------------------------
    model3, losses3 = train_three_class()
    probs = model3.probabilities()
    print("\n--- Task 5: three-class extension (2-2-3, softmax + cross-entropy) ---")
    print(f"    initial / final loss : {losses3[0]:.6f} / {losses3[-1]:.6f}")
    print(f"    W(2) shape {tuple(model3.fc2.weight.shape)}   logits per example {tuple(model3(X).shape)}")
    print("    input     class0     class1     class2   predicted target")
    for (x1, x2), row, pred, tgt in zip(X.tolist(), probs.tolist(),
                                        model3.predicted_classes(), Y_CLASS.tolist()):
        print(f"    ({x1:.0f},{x2:.0f})   {row[0]:.6f}   {row[1]:.6f}   {row[2]:.6f}     {pred}       {tgt}")
    print(f"    all four correct : {model3.predicted_classes() == Y_CLASS.tolist()}")
    print(f"    softmax rows sum to 1 : {all(abs(sum(r) - 1.0) < 1e-6 for r in probs.tolist())}")
    logits = model3(X).detach()
    naive100 = softmax_naive(logits + 100.0)
    stable = softmax_stable(logits)
    print(f"    naive softmax(z+100) is NaN : {bool(torch.isnan(naive100).any())}  (exp(100) overflows)")
    print(f"    max |stable(z) - stable(z+100)| : {float((stable - softmax_stable(logits + 100.0)).abs().max()):.3e}")
    print("    => stable softmax subtracts the row max so exp stays in (0,1].")


if __name__ == "__main__":
    main()
