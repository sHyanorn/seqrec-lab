"""Time to target (METHODS 8, 9.4) on hand-made curves."""

import math

import pytest

from seqrec.analysis import timing


def test_the_target_is_95_percent_of_the_best_arm_mean():
    """Arm means 0.10, 0.20 and 0.15: the target is 0.95 x 0.20."""
    best = {"a": [0.09, 0.11], "b": [0.18, 0.22], "c": [0.15, 0.15]}
    assert timing.target_quality(best) == pytest.approx(0.19)


def test_time_to_target_is_the_first_cumulative_time_at_or_above_the_target():
    train_s = [10.0, 20.0, 30.0, 40.0]
    assert timing.time_to_target(train_s, [0.1, 0.3, 0.2, 0.5], 0.3) == 20.0  # reached exactly
    assert timing.time_to_target(train_s, [0.1, 0.3, 0.2, 0.5], 0.31) == 40.0  # not the dip after
    assert timing.time_to_target(train_s, [0.4, 0.1, 0.1, 0.1], 0.3) == 10.0
    assert timing.time_to_target(train_s, [0.1, 0.2, 0.2, 0.29], 0.3) == math.inf  # censored
    assert timing.time_to_target([], [], 0.3) == math.inf


def test_curves_of_unequal_length_are_refused():
    with pytest.raises(ValueError):
        timing.time_to_target([1.0, 2.0], [0.1], 0.3)


def test_the_ratio_of_medians_and_its_reading():
    """METHODS 9.4: a ratio within 6.84% of 1 is no difference measured."""
    slow, fast = [100.0, 110, 120, 130, 140], [50.0, 55, 60, 65, 70]
    assert timing.median_ratio(slow, fast) == pytest.approx(2.0)
    assert timing.reading(2.0) == "difference"
    assert timing.reading(1.0684) == timing.reading(0.9316) == "no difference measured"
    assert timing.reading(1.07) == timing.reading(0.93) == "difference"


def test_medians_of_censored_groups():
    """Two of five censored: the median is finite. Three of five: the median itself is
    censored, and a ratio with it has no reading."""
    inf = math.inf
    assert timing.median_ratio([10.0, 20, 30, inf, inf], [10.0, 10, 10, 10, 10]) == 3.0
    assert timing.median_ratio([10.0, 20, inf, inf, inf], [10.0, 10, 10, 10, 10]) == inf
    assert timing.median_ratio([10.0, 10, 10, 10, 10], [10.0, 20, inf, inf, inf]) == 0.0
    assert math.isnan(timing.median_ratio([1.0, inf, inf], [1.0, inf, inf]))
    assert timing.reading(inf) == timing.reading(0.0) == "difference"
    assert timing.reading(math.nan) == "not measured"
