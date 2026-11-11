"""Time to target (METHODS 8, 9.4): how long a run trains before its validation NDCG@10
first reaches the dataset's target quality."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

TARGET_SHARE = 0.95
NOISE = 0.0684  # spread of long timing windows on the registered host (D046, METHODS 9.4)


def target_quality(best_validation: Mapping[str, Sequence[float]]) -> float:
    """0.95 times the highest mean, over an arm's runs, of the best validation NDCG@10."""
    return TARGET_SHARE * max(statistics.fmean(values) for values in best_validation.values())


def time_to_target(train_s: Sequence[float], validation: Sequence[float], target: float) -> float:
    """The cumulative training time of the first epoch whose validation NDCG@10 is at least
    the target; +inf for a run that never reaches it (censored)."""
    if len(train_s) != len(validation):
        raise ValueError("a curve needs one training time per validation value")
    return next((t for t, v in zip(train_s, validation, strict=True) if v >= target), math.inf)


def median_ratio(times: Sequence[float], reference: Sequence[float]) -> float:
    """Median time of an arm over the median of the reference arm. A median is +inf when half
    of its runs or more are censored; with both censored the ratio is NaN."""
    a, b = statistics.median(times), statistics.median(reference)
    return math.nan if math.isinf(a) and math.isinf(b) else a / b


def reading(ratio: float) -> str:
    """How a ratio of medians is reported (METHODS 9.4)."""
    if math.isnan(ratio):
        return "not measured"
    return "no difference measured" if 1 - NOISE <= ratio <= 1 + NOISE else "difference"
