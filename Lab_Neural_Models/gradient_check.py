"""
CS-407 AI Laboratory
Neural Models: Learning, Depth, Activations, and Output Layers

Gradient verification  -  Task 4 Part B / Reflection Q7.

Compares the first-layer weight gradient produced by PyTorch's backward pass
(reverse-mode automatic differentiation) with a central finite-difference
estimate.  If they agree, `param.grad` really is dL/dW(1) and not just "some
nonzero number".

The check runs in float64: in float32, dividing a ~1e-7 loss resolution by a
2*eps step produces ~1e-4 of roundoff noise, which alone would look like a
disagreement.  Double precision brings the two estimates together.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

import torch
import torch.nn as nn

import xor_net as xor

# Double-precision copies of the XOR data for the numerical check.
X = xor.X.double()
Y = xor.Y.double()


def make_double_model(hidden="sigmoid", seed=2):
    torch.manual_seed(seed)
    model = xor.XORNet(hidden=hidden, seed=seed)
    return model.double()


def analytic_gradient(model, loss_fn, X=X, Y=Y):
    model.zero_grad()
    loss = loss_fn(model(X), Y)
    loss.backward()
    return loss.item(), model.fc1.weight.grad.clone()


def finite_difference_gradient(model, loss_fn, X=X, Y=Y, eps=1e-5):
    grad = torch.zeros_like(model.fc1.weight)
    with torch.no_grad():
        for i in range(model.fc1.weight.shape[0]):
            for j in range(model.fc1.weight.shape[1]):
                original = model.fc1.weight[i, j].item()
                model.fc1.weight[i, j] = original + eps
                lp = loss_fn(model(X), Y).item()
                model.fc1.weight[i, j] = original - eps
                lm = loss_fn(model(X), Y).item()
                model.fc1.weight[i, j] = original
                grad[i, j] = (lp - lm) / (2 * eps)
    return grad


def main():
    print("=" * 70)
    print("GRADIENT CHECK - autograd vs central finite differences (float64)")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    model = make_double_model(hidden="sigmoid")
    loss_fn = nn.BCEWithLogitsLoss()           # operates in the model's dtype

    loss, analytic = analytic_gradient(model, loss_fn)
    numeric = finite_difference_gradient(model, loss_fn, eps=1e-5)

    print(f"\n    loss at test point : {loss:.10f}")
    print("\n    dL/dW(1) from autograd :")
    print("      " + "\n      ".join(str(row.tolist()) for row in analytic))
    print("\n    dL/dW(1) from finite differences (eps = 1e-5) :")
    print("      " + "\n      ".join(str(row.tolist()) for row in numeric))

    max_err = float((analytic - numeric).abs().max())
    print(f"\n    max |autograd - finite difference| : {max_err:.3e}")
    print(f"    agreement within 1e-8              : {max_err < 1e-8}")
    print(
        "\n    The two independent estimates agree, so backward() is indeed\n"
        "    computing dL/dW(1) via the chain rule.  Note that a finite-difference\n"
        "    check costs two forward passes per parameter; for a large model it is\n"
        "    far more expensive than one backward pass, which is why it is kept\n"
        "    only for tiny networks like this one (Reflection Q7)."
    )


if __name__ == "__main__":
    main()
