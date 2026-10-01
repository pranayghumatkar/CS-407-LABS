"""
CS-407 AI Laboratory - automated tests for the neural-model experiments.

Run with:  python tests/test_neural_models.py
or:        python -m pytest tests/test_neural_models.py -v

Student: Pranay Ghumatkar | 2024A3PS0328G
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import gradient_check as gc
import symmetry_experiment as se
import three_class_net as tc
import xor_net as xor

XOR_LABELS = [0, 1, 1, 0]
STEPS = 6000
LR = 0.1
SEED = 2


# -- Task 4 Part A -----------------------------------------------------------
def test_xor_learns_for_every_activation():
    for act in ("sigmoid", "tanh", "relu"):
        model = xor.train_xor(hidden=act, steps=STEPS, lr=LR, seed=SEED)["model"]
        labels = xor.thresholded_labels(model)
        assert labels == XOR_LABELS, f"{act} predicted {labels}"


def test_probabilities_are_valid():
    model = xor.train_xor(hidden="sigmoid", steps=STEPS, lr=LR, seed=SEED)["model"]
    for p in model.probabilities().flatten().tolist():
        assert 0.0 <= p <= 1.0, f"probability {p} out of range"


# -- Task 4 Part B -----------------------------------------------------------
def test_autograd_matches_finite_difference():
    model = gc.make_double_model(hidden="sigmoid", seed=SEED)
    loss_fn = nn.BCEWithLogitsLoss()
    _, analytic = gc.analytic_gradient(model, loss_fn)
    numeric = gc.finite_difference_gradient(model, loss_fn, eps=1e-5)
    assert float((analytic - numeric).abs().max()) < 1e-8


def test_logit_gradient_is_p_minus_y():
    torch.manual_seed(SEED)
    model = tc.ThreeClassNet(hidden="sigmoid", seed=SEED)
    logits = model(tc.X).detach().requires_grad_(True)   # leaf tensor
    # reduction='sum' makes dLoss/dLogit exactly p - y (not the batch mean).
    loss = F.cross_entropy(logits, tc.Y_CLASS, reduction="sum")
    grad = torch.autograd.grad(loss, logits)[0]
    p = F.softmax(logits.detach(), dim=-1)
    y = F.one_hot(tc.Y_CLASS, 3).float()
    assert torch.allclose(grad, p - y, atol=1e-6)


# -- Task 4 Part C -----------------------------------------------------------
def test_zero_init_keeps_hidden_rows_identical():
    result = se.run_zero_init(hidden="sigmoid", steps=500)
    assert max(result["row_gap"]) < 1e-12
    # and it fails to learn XOR, as the symmetric network has one effective unit
    assert xor.thresholded_labels(result["model"]) != XOR_LABELS


# -- Task 5 ------------------------------------------------------------------
def test_three_class_learns():
    model = tc.train_three_class(hidden="sigmoid", steps=STEPS, lr=LR, seed=SEED)["model"]
    assert model.predicted_classes() == [0, 1, 1, 2]


def test_softmax_rows_sum_to_one():
    model = tc.train_three_class(hidden="sigmoid", steps=STEPS, lr=LR, seed=SEED)["model"]
    for row in model.probabilities().tolist():
        assert abs(sum(row) - 1.0) < 1e-6


def test_softmax_is_shift_invariant():
    torch.manual_seed(SEED)
    logits = tc.ThreeClassNet(hidden="sigmoid", seed=SEED)(tc.X).detach()
    base = tc.softmax_stable(logits)
    shifted = tc.softmax_stable(logits + 100.0)
    assert torch.allclose(base, shifted, atol=1e-6)
    # the naive version overflows, which is *why* we subtract the max
    assert bool(torch.isnan(tc.softmax_naive(logits + 100.0)).any())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    sys.exit(1 if failures else 0)
