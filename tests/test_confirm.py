"""The confirmatory analysis on hand-made final runs: which runs feed which registered
comparison, the size of the family each is corrected in, and what is concluded from them."""

import csv
import json
import math

import numpy as np
import pytest

from seqrec.analysis import confirm, inference
from seqrec.analysis.confirm import NOT_SUPPORTED, SUPPORTED, Outcome
from seqrec.analysis.families import A1, A2, A3, A4, A5, Comparison
from seqrec.analysis.inference import NOT_REJECTED, PENDING, REJECTED
from seqrec.analysis.power import DELTA, directional_power, equivalence_power
from seqrec.cli import main as cli_main
from seqrec.data.stats import write_csv

OFFSETS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
# Test NDCG@10 per arm on leave-one-out: A1 far below the rest, A3 and A4 within 2% of A2,
# A5 below A2 by more than 2%.
LOO = {A1: 0.100, A2: 0.150, A3: 0.151, A4: 0.152, A5: 0.130}
GTS = {A1: 0.050, A2: 0.060, A3: 0.061, A4: 0.062, A5: 0.070}  # A5 ahead here
SIGMA = 0.004


REACH = {A2: 10.0, A1: 100.0, A3: 50.0, A4: 30.0}


def final_rows(
    dataset="ml-1m", loo=LOO, gts=GTS, spread=1e-4, pod="pod1", shuffled=0.05, reach=REACH
):
    """Five seeds per arm and split; y spreads symmetrically around log(mean) in steps of
    `spread`. HR@10 is twice NDCG@10 except for A4, where it is 1.5 times. Validation climbs
    0.05, 0.12, 0.20 in three epochs of `reach[arm] + seed` seconds; an arm whose reach is
    None stays at 0.05."""
    rows = []

    def add(arm, split, shuffle, mean):
        for seed, k in enumerate(OFFSETS, start=1):
            ndcg = mean * math.exp(k * spread)
            step = reach.get(arm, 5.0)
            epoch_s = (step or 1.0) + seed
            values = [0.05, 0.12, 0.20] if step else [0.05]
            records = [[(i + 1) * epoch_s, v] for i, v in enumerate(values)]
            rows.append({
                "run_id": f"{arm}-{dataset}-{split}-s{seed}" + ("-sh" if shuffle != "none" else ""),
                "dataset": dataset, "split": split, "arm": arm, "seed": str(seed),
                "phase": "final", "shuffle": shuffle, "tag": "", "status": "tested", "pod": pod,
                "test_ndcg@10": repr(ndcg),
                "test_hr@10": repr(ndcg * (1.5 if arm == A4 else 2.0)),
                "validation_records": json.dumps(records),
                "hardware_slowdown": "False", "slow_epochs": "", "sm_mhz_median": "2000.0",
            })  # fmt: skip

    for arm, mean in loo.items():
        add(arm, "loo", "none", mean)
    for arm, mean in gts.items():
        add(arm, "gts", "none", mean)
    for arm in (A2, A5):
        add(arm, "loo", "train_eval", shuffled)
    return rows


def by_name(outcomes, variant="primary"):
    return {o.comparison.name: o for o in outcomes if o.variant == variant}


@pytest.fixture(scope="module")
def outcomes():
    return confirm.analyze(final_rows(), {"ml-1m": SIGMA})


def test_each_comparison_reads_the_runs_of_its_two_arms(outcomes):
    primary = by_name(outcomes)
    o = primary[f"F1 ml-1m {A2} vs {A5}"]
    assert (o.n_a, o.n_b) == (5, 5)
    assert o.estimate == pytest.approx(math.log(0.150 / 0.130))
    assert o.low < o.estimate < o.high
    y = {arm: np.log(mean) + 1e-4 * OFFSETS for arm, mean in LOO.items()}
    assert o.p == pytest.approx(inference.p_two_sided(y[A2], y[A5]))
    w = inference.welch(y[A2], y[A5])
    from scipy import stats

    assert o.high - o.estimate == pytest.approx(stats.t.ppf(0.975, w.df) * w.se)
    assert primary[f"F1-NI ml-1m {A2} vs {A5}"].p == pytest.approx(
        inference.p_above(y[A2], y[A5], -DELTA)
    )
    assert primary[f"F1-ME ml-1m {A5} vs {A1}"].p == pytest.approx(
        inference.p_above(y[A5], y[A1], DELTA)
    )
    assert primary[f"F2 ml-1m {A3} vs {A2}"].p == pytest.approx(inference.p_tost(y[A3], y[A2]))
    shuffled = np.log(0.05) + 1e-4 * OFFSETS
    assert primary[f"F4 ml-1m {A5} vs C1-{A5}"].p == pytest.approx(
        inference.p_above(y[A5], shuffled, DELTA)
    )


def test_one_dataset_decides_what_it_can_and_leaves_the_rest_pending(outcomes):
    """Differences far beyond the noise: every MovieLens-1M test with a tiny p is rejected in
    its whole family; tests with p near 1 are not rejected; other datasets are pending."""
    primary = by_name(outcomes)
    assert len(primary) == 24 + 2 + 12 + 8 + 8 + 12 + 6
    for name, o in primary.items():
        if " ml-1m " not in name:
            assert (o.p, o.status, o.estimate) == (None, PENDING, None), name
    assert {o.status for n, o in primary.items() if n.startswith("F1 ml-1m")} == {REJECTED}
    assert primary[f"F1-NI ml-1m {A2} vs {A5}"].status == REJECTED
    assert {o.status for n, o in primary.items() if n.startswith("F1-ME ml-1m")} == {REJECTED}
    assert {o.status for n, o in primary.items() if n.startswith("F2 ml-1m")} == {REJECTED}
    assert {o.status for n, o in primary.items() if n.startswith("F4 ml-1m")} == {REJECTED}
    # gaps of 15% to 30%: equivalence is out of reach whatever the other datasets give
    assert {o.status for n, o in primary.items() if n.startswith("F5 ml-1m")} == {NOT_REJECTED}


def test_a_p_value_between_the_strictest_and_the_loosest_step_is_pending():
    """A4 against A2 with enough spread that its directional p lies above 0.05/24 and below
    0.05/6: with the other 18 tests of F1 still to come it is neither rejected nor safe."""
    rows = final_rows(loo=LOO | {A4: 0.1508}, spread=1.5e-3)
    y2, y4 = (np.log(m) + 1.5e-3 * OFFSETS for m in (LOO[A2], 0.1508))
    p = inference.p_two_sided(y4, y2)
    assert 0.05 / 19 < p < 0.05 / 6
    o = by_name(confirm.analyze(rows, {"ml-1m": SIGMA}))
    assert o[f"F1 ml-1m {A4} vs {A2}"].p == pytest.approx(p)
    assert o[f"F1 ml-1m {A4} vs {A2}"].status == PENDING
    assert o[f"F1 ml-1m {A2} vs {A1}"].status == REJECTED


def test_time_to_target_uses_the_group_of_twenty_on_one_pod():
    """Target 0.95 x 0.20 = 0.19, reached in the third epoch: 3 x (reach + seed) seconds. A4
    never reaches it. A sixth seed with a far higher validation value is outside the group,
    and so is BERT4Rec."""
    rows = final_rows(reach=REACH | {A4: None})
    extra = dict(rows[5], seed="6", run_id="extra", validation_records="[[1.0, 0.9]]")  # A2
    target, times, _ = confirm.time_group([*rows, extra], "ml-1m")
    assert target == pytest.approx(0.19)
    assert times[A2] == [33.0, 36.0, 39.0, 42.0, 45.0]
    assert times[A1] == [303.0, 306.0, 309.0, 312.0, 315.0]
    assert times[A4] == [math.inf] * 5
    assert sorted(times) == sorted([A1, A2, A3, A4])
    o = by_name(confirm.analyze(rows, {"ml-1m": SIGMA}))
    slower = o[f"F6 ml-1m {A2} vs {A1}"]
    assert slower.p == pytest.approx(1 / 252) and slower.status == REJECTED
    assert slower.estimate == pytest.approx(309 / 39)  # median of A1 over median of A2
    assert o[f"F6 ml-1m {A2} vs {A4}"].estimate == math.inf
    assert o[f"F6 steam {A2} vs {A1}"].status == PENDING


def test_a_target_between_two_records_is_reached_at_the_later_one():
    """With every A2 run ending at 0.30 the target is 0.95 x 0.30 = 0.285. The record at 0.24
    is below it, so a run reaches the target at its next record, and runs that end at 0.20
    are censored."""
    rows = final_rows()
    for r in rows:
        if r["arm"] == A2 and r["split"] == "loo" and r["shuffle"] == "none":
            r["validation_records"] = "[[4.0, 0.1], [8.0, 0.24], [20.0, 0.3]]"
        if r["arm"] == A3 and r["split"] == "loo" and r["seed"] in ("1", "2", "3"):
            r["validation_records"] = "[[4.0, 0.1], [8.0, 0.24], [20.0, 0.3]]"
    target, times, _ = confirm.time_group(rows, "ml-1m")
    assert target == pytest.approx(0.285)
    assert times[A2] == [20.0] * 5
    assert times[A3] == [20.0, 20.0, 20.0, math.inf, math.inf]
    assert times[A1] == [math.inf] * 5


def test_the_timing_repeat_leaves_out_flagged_runs():
    """METHODS 7.6 with the flags of METHODS 8 and D038. A1's fastest run overlaps A2 and is
    flagged for a hardware event, an A2 run for slow epochs and an A3 run for a clock more
    than 5% below the median of 2,000 MHz: 1,899 is flagged, 1,900 is not."""
    rows = final_rows()
    runs = {
        (r["arm"], r["seed"]): r for r in rows if r["split"] == "loo" and r["shuffle"] == "none"
    }
    runs[A1, "1"] |= {"validation_records": "[[34.0, 0.2]]", "hardware_slowdown": "True"}
    runs[A2, "5"]["slow_epochs"] = "True"
    runs[A3, "2"]["sm_mhz_median"] = "1899.0"
    runs[A3, "3"]["sm_mhz_median"] = "1900.0"
    _, times, flags = confirm.time_group(rows, "ml-1m", clock=2000.0)
    assert times[A1][0] == 34.0
    assert flags == {
        A1: [True, False, False, False, False], A2: [False, False, False, False, True],
        A3: [False, True, False, False, False], A4: [False] * 5,
    }  # fmt: skip
    assert confirm.time_group(rows, "ml-1m")[2][A3] == [False] * 5  # no clock to compare with
    outcomes = confirm.analyze(rows, {"ml-1m": SIGMA}, clock=2000.0)
    name = f"F6 ml-1m {A2} vs {A1}"
    primary, repeat = by_name(outcomes)[name], by_name(outcomes, confirm.UNFLAGGED)[name]
    # A2 is 33, 36, 39, 42, 45 and A1 34, 306, 309, 312, 315: A2 has ranks 1, 3, 4, 5, 6 with
    # sum 19, and 12 of the 252 splits have a rank sum of 19 or less (1, 1, 2, 3 and 5 splits
    # with sums 15 to 19)
    assert primary.p == pytest.approx(12 / 252) and (primary.n_a, primary.n_b) == (5, 5)
    assert (repeat.n_a, repeat.n_b) == (4, 4) and repeat.p == pytest.approx(1 / 70)
    assert len(by_name(outcomes, confirm.UNFLAGGED)) == 12


def test_a_group_spread_over_two_pods_is_not_compared():
    rows = final_rows()
    rows[7]["pod"] = "pod2"  # one SASRec-CE run
    with pytest.raises(ValueError, match="Pods"):
        confirm.analyze(rows, {"ml-1m": SIGMA})
    rows = [r for r in final_rows() if not (r["arm"] == A3 and r["seed"] == "4")]
    with pytest.raises(ValueError, match="lacks"):
        confirm.time_group(rows, "ml-1m")


def test_the_temporal_split_and_hr_variants_read_their_own_values(outcomes):
    gts = by_name(outcomes, "temporal split")
    assert len(gts) == 24 + 2 + 12 + 8 + 8
    assert gts[f"F1 ml-1m {A2} vs {A5}"].estimate == pytest.approx(math.log(0.060 / 0.070))
    assert gts[f"F1-NI ml-1m {A2} vs {A5}"].status == NOT_REJECTED  # decided: p is near 1
    hr = by_name(outcomes, "HR@10")
    assert len(hr) == 24 + 8
    assert hr[f"F1 ml-1m {A4} vs {A2}"].estimate == pytest.approx(math.log(0.152 * 1.5 / 0.300))
    assert hr[f"F1 ml-1m {A2} vs {A1}"].estimate == pytest.approx(math.log(1.5))


def test_power_comes_from_the_registered_functions_at_the_datasets_sigma(outcomes):
    primary = by_name(outcomes)
    assert primary[f"F1 ml-1m {A2} vs {A5}"].power == pytest.approx(directional_power(SIGMA, 5))
    assert primary[f"F2 ml-1m {A3} vs {A2}"].power == pytest.approx(equivalence_power(SIGMA, 5))
    assert primary[f"F2 ml-1m {A3} vs {A2}"].underpowered is False
    assert primary[f"F1 steam {A2} vs {A5}"].power is None


def test_a_dataset_with_final_runs_must_have_every_group():
    rows = [r for r in final_rows() if r["shuffle"] == "none"]
    with pytest.raises(ValueError, match="F4"):
        confirm.analyze(rows, {"ml-1m": SIGMA})
    rows = [r for r in final_rows() if r["split"] == "loo"]
    with pytest.raises(ValueError, match="temporal split"):
        confirm.analyze(rows, {"ml-1m": SIGMA})
    twice = final_rows() + final_rows()[:1]
    with pytest.raises(ValueError, match="second run"):
        confirm.analyze(twice, {"ml-1m": SIGMA})


def test_conclusions_of_one_dataset(outcomes):
    result = {(c["question"], c["dataset"]): c["result"] for c in confirm.conclusions(outcomes)}
    for question in ("H1a", "H1b", "H2a", "H2b sampled softmax", "H2b gBCE", "H2b BCE"):
        assert result[question, "ml-1m"] == SUPPORTED, question
    assert result[f"shuffle control {A2}", "ml-1m"] == SUPPORTED
    assert result[f"shuffle control {A5}", "ml-1m"] == SUPPORTED
    assert (f"shuffle control {A5}", "steam") not in result and ("H1a", "steam") not in result
    assert result["H1b", "steam"] == result["H1a", "ml-20m"] == PENDING
    # A5 ahead of A1, A4 and A2 ahead of A5, as published, by far more than 2%
    assert result["G1", "ml-1m"] == result["G2", "ml-1m"] == result["G3", "ml-1m"] == "confirmed"
    assert result["G2", "steam"].startswith("pending: reversed below SESOI or reversed or")
    assert ("G2", "ml-20m") not in result


def test_a_hypothesis_needs_all_its_tests():
    assert confirm.all_rejected([REJECTED, REJECTED, REJECTED]) == SUPPORTED
    assert confirm.all_rejected([REJECTED, PENDING, REJECTED]) == PENDING
    assert confirm.all_rejected([REJECTED, PENDING, NOT_REJECTED]) == NOT_SUPPORTED


def gap(sign, estimate, directional, tost):
    c = Comparison("F1", "ml-1m", "directional", A5, A1)
    return confirm.gap_class(
        sign,
        Outcome("primary", c, status=directional, estimate=estimate),
        Outcome("primary", c, status=tost),
    )


@pytest.mark.parametrize(
    "sign, estimate, directional, tost, expected",
    [
        (+1, -0.01, REJECTED, REJECTED, ["reversed below SESOI"]),
        (+1, -0.10, REJECTED, NOT_REJECTED, ["reversed"]),
        (+1, +0.01, REJECTED, REJECTED, ["confirmed below SESOI"]),
        (+1, +0.10, REJECTED, NOT_REJECTED, ["confirmed"]),
        (-1, -0.10, REJECTED, NOT_REJECTED, ["confirmed"]),
        (-1, +0.10, REJECTED, NOT_REJECTED, ["reversed"]),
        (+1, +0.001, NOT_REJECTED, REJECTED, ["equivalent"]),
        (+1, -0.03, NOT_REJECTED, NOT_REJECTED, ["inconclusive"]),
        (+1, +0.10, PENDING, NOT_REJECTED, ["confirmed", "inconclusive"]),
        (+1, +0.01, REJECTED, PENDING, ["confirmed below SESOI", "confirmed"]),
        (+1, -0.01, PENDING, REJECTED, ["reversed below SESOI", "reversed", "equivalent"]),
        (+1, -0.01, NOT_REJECTED, PENDING, ["equivalent", "inconclusive"]),
        (+1, None, PENDING, PENDING, list(confirm.GAP_CLASSES)),
    ],
)
def test_the_class_of_a_published_gap_is_the_first_that_applies(
    sign, estimate, directional, tost, expected
):
    assert gap(sign, estimate, directional, tost) == expected


def test_rank_agreement_between_the_two_splits():
    """Leave-one-out orders the arms A1 < A5 < A2 < A3 < A4; the temporal split moves A5 to
    the top: three of the ten pairs change order, so tau = (7 - 3) / 10."""
    assert confirm.rank_agreement(final_rows(), "ml-1m") == pytest.approx(0.4)
    assert confirm.rank_agreement(final_rows(gts=LOO), "ml-1m") == pytest.approx(1.0)


def test_the_report_writes_its_tables_from_exported_batch_files(tmp_path):
    """Through the command: the four tables, the sigma of the first five seeds and the
    conclusions as text."""
    rows = final_rows()
    for i, r in enumerate(rows):
        r |= {
            "batch": "m3-final-ml-1m", "validation_ndcg@10": repr(0.2 + 0.001 * (i % 5)),
            "test_ndcg@20": r["test_ndcg@10"], "test_hr@20": r["test_hr@10"],
            "test_tie_rate": "0.01", "truncated": "False", "gpu_hours": "0.02",
        }  # fmt: skip
    rows[0]["truncated"] = "True"  # one of five SASRec-BCE runs on leave-one-out
    trial = dict(rows[1], phase="tune", tag="t03", run_id="a-trial")  # not a final run
    path = tmp_path / "m3-final-ml-1m.csv"
    write_csv(path, [*rows, trial])
    out = tmp_path / "analysis"
    text = confirm.report([path], out)
    assert ["H1a", "ml-1m", SUPPORTED] in [line.split() for line in text.splitlines()]
    assert "sigma of ml-1m" in text
    assert cli_main(["analyze", str(path), "--out-dir", str(out)]) == 0

    def table(name):
        with (out / f"{name}.csv").open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    tests = table("tests")
    assert len(tests) == 72 + 54 + 32 + 12
    ni = next(t for t in tests if t["variant"] == "primary" and t["family"] == "F1-NI"
              and t["dataset"] == "ml-1m")  # fmt: skip
    assert (ni["a"], ni["b"], ni["n_a"], ni["status"]) == (A2, A5, "5", REJECTED)
    assert float(ni["relative"]) == pytest.approx(0.150 / 0.130 - 1)
    timed = next(t for t in tests if t["family"] == "F6" and t["dataset"] == "ml-1m"
                 and t["b"] == A1)  # fmt: skip
    assert float(timed["estimate"]) == pytest.approx(309 / 39) and timed["reading"] == "difference"
    control = [t for t in tests if t["family"] == "F4" and t["dataset"] == "ml-1m"]
    assert {t["b"] for t in control} == {f"C1-{A2}", f"C1-{A5}"}
    arms = {(a["split"], a["shuffle"], a["arm"]): a for a in table("arms")}
    assert len(arms) == 12
    bce = arms["loo", "none", A1]
    assert (bce["runs"], bce["tested"], bce["truncated_share"]) == ("5", "5", "0.2")
    assert float(bce["test_ndcg@10_mean"]) == pytest.approx(0.1, rel=1e-6)
    assert float(bce["gpu_hours"]) == pytest.approx(0.1)
    times = table("times")
    assert len(times) == 20 and {t["flagged"] for t in times} == {"False"}
    assert all(float(t["target_ndcg@10"]) == pytest.approx(0.19) for t in times)
    assert {t["result"] for t in table("conclusions") if t["dataset"] == "steam"} == {
        PENDING, "pending: " + " or ".join(confirm.GAP_CLASSES),
    }  # fmt: skip
