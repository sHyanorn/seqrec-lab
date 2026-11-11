"""Known answers for the tests of METHODS 7.3, 7.4 and 8. Expected values are computed by hand
or by an independent scipy path, never by the function under test."""

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from seqrec.analysis import inference
from seqrec.analysis.inference import NOT_REJECTED, PENDING, REJECTED
from seqrec.analysis.power import DELTA

A = np.log([0.180, 0.184, 0.179, 0.186, 0.182])
B = np.log([0.171, 0.176, 0.169, 0.174, 0.180, 0.172])


# --- Welch (7.3) --------------------------------------------------------------------------


def test_welch_on_a_case_computed_by_hand():
    """a = 1, 2, 3 (mean 2, variance 1); b = 2, 4, 6, 8 (mean 5, variance 20/3).
    SE^2 = 1/3 + 5/3 = 2; df = 2^2 / ((1/3)^2 / 2 + (5/3)^2 / 3) = 4 / (53/54) = 216/53."""
    w = inference.welch([1, 2, 3], [2, 4, 6, 8])
    assert w.diff == pytest.approx(-3)
    assert w.se == pytest.approx(math.sqrt(2))
    assert w.df == pytest.approx(216 / 53)


def test_the_directional_test_is_the_two_sided_welch_test():
    expected = stats.ttest_ind(A, B, equal_var=False).pvalue
    assert inference.p_two_sided(A, B) == pytest.approx(expected, rel=1e-12)
    assert inference.p_two_sided(B, A) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("margin", [-DELTA, 0.0, DELTA])
def test_the_one_sided_test_shifts_the_null_by_its_margin(margin):
    """H0: mu_a - mu_b <= margin is H0: mean(a - margin) <= mean(b)."""
    expected = stats.ttest_ind(A - margin, B, equal_var=False, alternative="greater").pvalue
    assert inference.p_above(A, B, margin) == pytest.approx(expected, rel=1e-12)


def test_non_inferiority_and_minimum_effect_differ_only_by_the_sign_of_the_margin():
    """With a difference of exactly delta, the non-inferiority statistic is 2 delta / SE and
    the minimum-effect statistic 0, so its p is one half."""
    b = np.array([0.0, 0.01, -0.01, 0.02, -0.02])
    a = b + DELTA
    w = inference.welch(a, b)
    assert inference.p_above(a, b, DELTA) == pytest.approx(0.5)
    assert inference.p_above(a, b, -DELTA) == pytest.approx(stats.t.sf(2 * DELTA / w.se, w.df))
    assert inference.p_above(a, b, -DELTA) < 0.05 < inference.p_above(a, b, DELTA)


def test_tost_is_the_larger_of_its_two_one_sided_p_values():
    lower = stats.ttest_ind(A + DELTA, B, equal_var=False, alternative="greater").pvalue
    upper = stats.ttest_ind(A - DELTA, B, equal_var=False, alternative="less").pvalue
    assert inference.p_tost(A, B) == pytest.approx(max(lower, upper), rel=1e-12)
    assert inference.p_tost(A, B) == pytest.approx(inference.p_tost(B, A), rel=1e-12)


def test_tost_at_the_edges():
    """Equal means: both one-sided tests have the statistic delta / SE. A difference of
    exactly delta: the upper test has statistic 0, so the TOST p is one half."""
    b = np.array([0.0, 0.001, -0.001, 0.002, -0.002])
    w = inference.welch(b, b)
    assert inference.p_tost(b, b) == pytest.approx(stats.t.sf(DELTA / w.se, w.df))
    assert inference.p_tost(b, b) < 0.05
    assert inference.p_tost(b + DELTA, b) == pytest.approx(0.5)
    assert inference.p_tost(b - DELTA, b) == pytest.approx(0.5)


def test_a_comparison_without_variance_or_with_one_run_is_refused():
    with pytest.raises(ValueError, match="two runs"):
        inference.welch([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="variance"):
        inference.welch([1.0, 1.0], [2.0, 2.0])


# --- Holm (7.4) ---------------------------------------------------------------------------


def holm_by_adjusted_p(pvalues, alpha=0.05):
    """An independent path: reject where the adjusted p, the running maximum of
    (m - rank + 1) p over the sorted p-values, is at most alpha."""
    names = sorted(pvalues, key=pvalues.get)
    m, worst, out = len(names), 0.0, {}
    for rank, name in enumerate(names, start=1):
        worst = max(worst, min(1.0, (m - rank + 1) * pvalues[name]))
        out[name] = REJECTED if worst <= alpha else NOT_REJECTED
    return out


def test_holm_on_a_complete_family_computed_by_hand():
    """m = 4, thresholds 0.0125, 0.0167, 0.025, 0.05: 0.005 and 0.01 pass, 0.03 stops the
    procedure, so 0.04 is not rejected although it is below 0.05."""
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    assert inference.holm(p) == {
        "a": REJECTED, "b": NOT_REJECTED, "c": NOT_REJECTED, "d": REJECTED,
    }  # fmt: skip


def test_holm_at_its_thresholds():
    """A p-value equal to its threshold is rejected; the next float above it is not."""
    edge = 0.05 / 3
    assert inference.holm({"a": edge, "b": 1.0, "c": 1.0})["a"] == REJECTED
    assert inference.holm({"a": math.nextafter(edge, 1), "b": 1.0, "c": 1.0})["a"] == NOT_REJECTED
    assert inference.holm({"a": 0.0, "b": 0.025, "c": 0.05}) == dict.fromkeys("abc", REJECTED)
    tied = inference.holm({"a": 0.02, "b": 0.02, "c": 1.0})  # 0.02 > 0.05 / 3
    assert tied == dict.fromkeys("abc", NOT_REJECTED)


def test_holm_matches_adjusted_p_values_on_random_complete_families():
    rng = np.random.default_rng(0)
    for _ in range(300):
        m = int(rng.integers(1, 13))
        p = {f"t{i}": float(v) for i, v in enumerate(rng.beta(0.3, 2.0, m))}
        assert inference.holm(p) == holm_by_adjusted_p(p)


def test_an_incomplete_family_decides_only_what_no_missing_value_can_change():
    """A family of 24 with 6 known p-values (one dataset of F1). The smallest passes 0.05/24
    and the next 0.05/23: rejected whatever comes. 0.0023 is above 0.05/22 and at most
    0.05/4, its threshold if all 18 missing tests rejected first: pending. 0.04 is above
    0.05/3, so it stops the procedure in every completion, and what follows it is never
    rejected."""
    p = {"a": 0.0001, "b": 0.0021, "c": 0.0023, "d": 0.04, "e": 0.045, "f": 0.6}
    family = p | {f"other{i}": None for i in range(18)}
    assert 0.05 / 24 < p["b"] <= 0.05 / 23 and 0.05 / 22 < p["c"]
    status = inference.holm(family)
    assert [status[k] for k in "abcdef"] == [
        REJECTED, REJECTED, PENDING, NOT_REJECTED, NOT_REJECTED, NOT_REJECTED,
    ]  # fmt: skip
    assert all(status[f"other{i}"] == PENDING for i in range(18))


def test_a_decided_status_survives_every_completion_of_the_family():
    """Random families with missing values: whatever the missing p-values turn out to be,
    ordinary Holm on the completed family agrees with every status decided early; and each
    pending test is rejected in one completion and not rejected in another."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        m = int(rng.integers(2, 10))
        known = {f"t{i}": float(v) for i, v in enumerate(rng.beta(0.3, 3.0, m))}
        missing = [name for name in known if rng.random() < 0.5]
        early = inference.holm({k: None if k in missing else v for k, v in known.items()})
        completions = [dict.fromkeys(missing, 0.0), dict.fromkeys(missing, 1.0)]
        completions += [{k: float(rng.random() ** 4) for k in missing} for _ in range(20)]
        finals = [holm_by_adjusted_p(known | c) for c in completions]
        for name in known:
            if name in missing:
                assert early[name] == PENDING
            elif early[name] == PENDING:
                assert finals[0][name] == REJECTED and finals[1][name] == NOT_REJECTED
            else:
                assert {f[name] for f in finals} == {early[name]}


def test_holm_refuses_values_that_are_not_probabilities():
    with pytest.raises(ValueError):
        inference.holm({"a": float("nan"), "b": 0.5})
    with pytest.raises(ValueError):
        inference.holm({"a": 1.2})


# --- Mann-Whitney with censoring (8) ------------------------------------------------------


def test_complete_separation_of_five_against_five_gives_one_in_252():
    fast, slow = [10.0, 11, 12, 13, 14], [15.0, 16, 17, 18, 19]
    assert inference.mann_whitney_less(fast, slow) == pytest.approx(1 / 252)
    assert inference.mann_whitney_less(slow, fast) == pytest.approx(1.0)
    one_overlap = [10.0, 11, 12, 13, 15.5]
    assert inference.mann_whitney_less(one_overlap, slow) == pytest.approx(2 / 252)


def test_without_ties_it_is_scipys_exact_test():
    rng = np.random.default_rng(2)
    for _ in range(50):
        x, y = rng.random(int(rng.integers(2, 7))), rng.random(int(rng.integers(2, 7)))
        expected = stats.mannwhitneyu(x, y, alternative="less", method="exact").pvalue
        assert inference.mann_whitney_less(x, y) == pytest.approx(expected, rel=1e-12)


def test_censored_runs_rank_last_and_share_a_mid_rank():
    """x = 1, inf and y = inf, inf: ranks 1, 3, 3, 3. The rank sum of x is 4; of the 6 ways to
    choose two of the four values, the 3 that hold the finite one have rank sum 4 and the
    others 6, so p = 3/6."""
    inf = math.inf
    assert inference.mann_whitney_less([1.0, inf], [inf, inf]) == pytest.approx(0.5)
    assert inference.mann_whitney_less([inf, inf], [inf, inf]) == pytest.approx(1.0)
    # the censored value ranks behind every finite one, however large
    assert inference.mann_whitney_less([1.0, 2, 3], [1e12, inf, inf]) == pytest.approx(1 / 20)
    assert inference.mann_whitney_less([1.0, 2, inf], [3.0, 4, 5]) == pytest.approx(
        inference.mann_whitney_less([1.0, 2, 6], [3.0, 4, 5])
    )


def test_with_ties_and_censoring_it_is_the_exact_permutation_test_of_the_rank_sum():
    """Against a direct enumeration of all splits of the pooled mid-ranks."""
    inf = math.inf
    cases = [
        ([1.0, 2, 2, inf, inf], [2.0, 3, inf, inf, inf]),
        ([5.0, 5, 5, 7, inf], [5.0, 6, 7, 7, 8]),
        ([3.0, inf, inf, inf, inf], [1.0, 2, inf, inf, inf]),
    ]
    for x, y in cases:
        ranks = stats.rankdata(np.concatenate([x, y]))
        observed = ranks[: len(x)].sum()
        sums = [sum(c) for c in itertools.combinations(ranks, len(x))]
        expected = sum(s <= observed + 1e-9 for s in sums) / len(sums)
        assert inference.mann_whitney_less(x, y) == pytest.approx(expected, rel=1e-12)


def test_mann_whitney_refuses_values_it_cannot_rank():
    with pytest.raises(ValueError):
        inference.mann_whitney_less([1.0, math.nan], [2.0, 3.0])
    with pytest.raises(ValueError):
        inference.mann_whitney_less([], [2.0, 3.0])
