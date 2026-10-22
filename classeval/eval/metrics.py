import math
from collections.abc import Sequence

import scipy.stats


def exact_and_within_one(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> tuple[float, float]:
    """Calculate exact match and within-one fraction between true and predicted labels."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0, 0.0

    n = len(y_true)
    exact_count = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == yp)
    within_one_count = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if abs(yt - yp) <= 1)

    return round(float(exact_count) / float(n), 4), round(float(within_one_count) / float(n), 4)


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Calculate Kendall's tau-b rank correlation coefficient between two score/rank arrays."""
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return 0.0

    res = scipy.stats.kendalltau(a, b).statistic
    if res is None or math.isnan(res):
        return 0.0
    return float(res)


def confusion(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    k: int = 4,
) -> list[list[int]]:
    """Compute k x k confusion matrix where matrix[i][j] is count of true=i, pred=j."""
    matrix = [[0 for _ in range(k)] for _ in range(k)]
    for yt, yp in zip(y_true, y_pred, strict=False):
        if 0 <= yt < k and 0 <= yp < k:
            matrix[yt][yp] += 1
    return matrix
