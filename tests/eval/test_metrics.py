import pytest

from classeval.eval.metrics import confusion, exact_and_within_one, kendall_tau


def test_exact_and_within_one_standard_cases():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]
    exact, within_one = exact_and_within_one(y_true, y_pred)
    assert exact == 1.0
    assert within_one == 1.0

    y_pred_off_one = [1, 2, 3, 2]
    exact, within_one = exact_and_within_one(y_true, y_pred_off_one)
    assert exact == 0.0
    assert within_one == 1.0

    y_pred_mixed = [0, 2, 2, 0]  # diffs: 0, 1, 0, 3 -> exact=2/4=0.5, within_one=3/4=0.75
    exact, within_one = exact_and_within_one(y_true, y_pred_mixed)
    assert exact == 0.5
    assert within_one == 0.75


def test_exact_and_within_one_empty():
    exact, within_one = exact_and_within_one([], [])
    assert exact == 0.0
    assert within_one == 0.0


def test_kendall_perfect_and_reversed():
    # Perfectly concordant rankings
    a = [10.0, 20.0, 30.0, 40.0]
    b = [0.0, 1.0, 2.0, 3.0]
    tau = kendall_tau(a, b)
    assert pytest.approx(tau, abs=1e-3) == 1.0

    # Perfectly discordant rankings
    b_rev = [3.0, 2.0, 1.0, 0.0]
    tau_rev = kendall_tau(a, b_rev)
    assert pytest.approx(tau_rev, abs=1e-3) == -1.0

    # Tie case
    a_ties = [10.0, 10.0, 30.0, 40.0]
    b_ties = [0.0, 1.0, 2.0, 3.0]
    tau_ties = kendall_tau(a_ties, b_ties)
    assert 0.0 < tau_ties < 1.0


def test_confusion_matrix():
    y_true = [0, 1, 2, 3, 1]
    y_pred = [0, 1, 1, 3, 2]
    matrix = confusion(y_true, y_pred, k=4)

    assert len(matrix) == 4
    for row in matrix:
        assert len(row) == 4

    # Class 0: 1 sample predicted 0
    assert matrix[0][0] == 1
    # Class 1: 1 sample predicted 1, 1 sample predicted 2
    assert matrix[1][1] == 1
    assert matrix[1][2] == 1
    # Class 2: 1 sample predicted 1
    assert matrix[2][1] == 1
    # Class 3: 1 sample predicted 3
    assert matrix[3][3] == 1
