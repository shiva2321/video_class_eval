from collections.abc import Sequence


def grade_of(score: float, cutoffs: tuple[float, float, float]) -> int:
    """Map numeric score to grade index 0..3 (Poor=0, Average=1, Good=2, Excellent=3)."""
    c1, c2, c3 = cutoffs
    if score < c1:
        return 0
    if score < c2:
        return 1
    if score < c3:
        return 2
    return 3


def fit_cutoffs(
    scores: Sequence[float],
    labels: Sequence[int],
    soft_le_average: Sequence[float],
) -> tuple[float, float, float]:
    """Fit boundaries c1 < c2 < c3 over grid 0..100 to maximise exact matches under soft constraint."""
    # Hard constraint: every score in soft_le_average must map to <= Average (index <= 1), meaning score < c2
    min_c2 = int(max(soft_le_average)) + 1 if soft_le_average else 1

    best_key = (-1, -1, float("-inf"))
    best_cands: list[tuple[float, float, float]] = []

    for c1 in range(99):
        c2_start = max(c1 + 1, min_c2)
        for c2 in range(c2_start, 100):
            for c3 in range(c2 + 1, 101):
                cutoffs = (float(c1), float(c2), float(c3))
                exact = 0
                within_one = 0
                total_err = 0

                for s, y in zip(scores, labels, strict=False):
                    g = grade_of(s, cutoffs)
                    diff = abs(g - y)
                    if diff == 0:
                        exact += 1
                    if diff <= 1:
                        within_one += 1
                    total_err += diff

                key = (exact, within_one, -total_err)
                if key > best_key:
                    best_key = key
                    best_cands = [cutoffs]
                elif key == best_key:
                    best_cands.append(cutoffs)

    if not best_cands:
        return (45.0, 62.0, 78.0)

    # Tie break 3: pick the middle of the best plateau
    return best_cands[len(best_cands) // 2]


def loo_predict(
    scores: Sequence[float],
    labels: Sequence[int],
    soft_le_average: Sequence[float],
) -> list[int]:
    """Leave-one-out predictions for hard-labelled classes without label leakage."""
    n = len(scores)
    predictions: list[int] = []

    for i in range(n):
        scores_train = [s for j, s in enumerate(scores) if j != i]
        labels_train = [y for j, y in enumerate(labels) if j != i]

        cutoffs_i = fit_cutoffs(scores_train, labels_train, soft_le_average)
        pred_i = grade_of(scores[i], cutoffs_i)
        predictions.append(pred_i)

    return predictions
