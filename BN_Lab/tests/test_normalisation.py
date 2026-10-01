"""
CS-407 AI Laboratory - automated tests for the probabilistic invariant.

Part VII: for every context w, sum_v P(v | w) = 1.

Run with:  python -m pytest tests/test_normalisation.py -v
or simply: python tests/test_normalisation.py

Student: Pranay Ghumatkar | 2024A3PS0328G
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import first_order_lm as fo
import second_order_lm as so


def test_first_order_rows_sum_to_one():
    model = fo.FirstOrderLanguageModel()
    model.train(fo.tokenised_dataset())
    for word, total in model.normalisation_check():
        assert abs(total - 1.0) < 1e-9, f"row for {word!r} sums to {total}"


def test_second_order_rows_sum_to_one():
    model = so.SecondOrderLanguageModel()
    model.train(so.tokenised_dataset())
    for context, total in model.normalisation_check():
        assert abs(total - 1.0) < 1e-9, f"row for {context} sums to {total}"


def test_probabilities_are_valid():
    model = fo.FirstOrderLanguageModel()
    model.train(fo.tokenised_dataset())
    for word, row in model.probabilities.items():
        for nxt, p in row.items():
            assert 0.0 <= p <= 1.0, f"P({nxt!r}|{word!r}) = {p} out of range"


def test_transition_counts_match_dataset():
    """Check hand-computed counts on the worksheet dataset.

    'the' occurs as a context 12 times: cat x3, dog x3, mat x2, rug x2,
    park x2.  Hence P(cat|the) = 3/12 and P(dog|the) = 3/12.
    (The worksheet's 3/5 and 2/5 come from the smaller illustrative
    corpus described in Part IV, not from this six-sentence dataset.)
    """
    model = fo.FirstOrderLanguageModel()
    model.train(fo.tokenised_dataset())
    the = model.distribution("the")
    assert abs(the["cat"] - 3 / 12) < 1e-12
    assert abs(the["dog"] - 3 / 12) < 1e-12
    assert abs(the["mat"] - 2 / 12) < 1e-12
    assert abs(the["rug"] - 2 / 12) < 1e-12
    assert abs(the["park"] - 2 / 12) < 1e-12


def test_greedy_is_deterministic():
    model = fo.FirstOrderLanguageModel()
    model.train(fo.tokenised_dataset())
    assert model.generate_many(5, "greedy") == model.generate_many(5, "greedy")


if __name__ == "__main__":
    test_first_order_rows_sum_to_one()
    test_second_order_rows_sum_to_one()
    test_probabilities_are_valid()
    test_transition_counts_match_dataset()
    test_greedy_is_deterministic()
    print("All tests passed.")
