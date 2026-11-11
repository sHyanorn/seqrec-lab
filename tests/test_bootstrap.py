"""The two-level bootstrap of METHODS 7.6 on hand-made ranks whose intervals are known."""

import csv
import math

import numpy as np
import pytest

from seqrec.analysis import bootstrap
from seqrec.analysis.families import A1, A2, A3, A4, A5
from seqrec.cli import main as cli_main
from seqrec.eval.ranking import metrics_from_ranks
from seqrec.runner.config import RunConfig

NDCG_2 = 1 / math.log2(3)  # NDCG@10 at rank 2
ARMS = (A1, A2, A3, A4, A5)


def interval(a, b, resamples=1000):
    """The single interval of arm a against arm b, given each arm's ranks, one row per run."""
    arms = {"a": bootstrap.ndcg(np.array(a)), "b": bootstrap.ndcg(np.array(b))}
    ((estimate, low, high),) = bootstrap.intervals(arms, [("a", "b")], resamples)
    return estimate, low, high


def test_ndcg_per_user_is_the_metric_of_the_evaluation():
    ranks = np.array([1, 2, 3, 10, 11, 3416])
    assert bootstrap.ndcg(ranks).tolist() == pytest.approx(
        [1, NDCG_2, 0.5, 1 / math.log2(11), 0, 0]
    )
    assert bootstrap.ndcg(ranks).mean() == pytest.approx(metrics_from_ranks(ranks)["ndcg@10"])


def test_arms_are_paired_by_user():
    """B's NDCG@10 is half of A's for both users (ranks 1 and 2 against 3 and 8), so every
    resample that draws the same users for both arms gives log 2. Users drawn for each arm
    apart would spread the interval."""
    assert interval([[1, 2]], [[3, 8]]) == pytest.approx((math.log(2),) * 3)


def test_with_one_seed_per_arm_only_users_are_drawn():
    """Two users: A has NDCG@10 1 and 0.5, B 0.63 and 1. The draws (u1, u1), one of each and
    (u2, u2) come with chances 1/4, 1/2 and 1/4, so the interval runs from the difference on
    u2 alone to the one on u1 alone, and the estimate is the difference on both."""
    estimate, low, high = interval([[1, 3]], [[2, 1]])
    assert estimate == pytest.approx(math.log(0.75 / ((NDCG_2 + 1) / 2)))
    assert (low, high) == pytest.approx((math.log(0.5), math.log(1 / NDCG_2)))


def test_with_identical_users_only_seeds_are_drawn():
    """Every user of a run has the same rank, so drawing users changes nothing. A's two seeds
    have y = 0 and log 0.5; two draws give 0, log(0.5) / 2 or log 0.5 with chances 1/4, 1/2
    and 1/4; B has one seed."""
    estimate, low, high = interval([[1] * 4, [3] * 4], [[2] * 4])
    b = math.log(NDCG_2)
    assert estimate == pytest.approx(math.log(0.5) / 2 - b)
    assert (low, high) == pytest.approx((math.log(0.5) - b, -b))


def test_the_interval_covers_the_population_difference_nearly_95_times_in_100():
    """Synthetic ranks with both sources of variation: users drawn from a population, shared
    by every run as a test set is, and five seeds per arm whose hit rates scatter around 0.30
    and 0.24 with sd 0.1 on the log scale. The population difference is log 1.25. Five seeds
    understate the variation between runs, so the intervals cover it a little less often than
    95 times in 100 (182 of 200 here). Drawing users alone covers it about 2 times in 3, and
    drawing users apart for each arm every time."""
    rng = np.random.default_rng(7)
    replicates, covered = 200, 0
    for _ in range(replicates):
        ease = rng.random(300)  # a user hits when ease falls below the run's hit rate
        arms = {}
        for arm, rate in (("a", 0.30), ("b", 0.24)):
            rates = rate * np.exp(rng.normal(0, 0.1, size=5))
            arms[arm] = bootstrap.ndcg(np.where(ease < rates[:, None], 1, 100))
        ((_, low, high),) = bootstrap.intervals(arms, [("a", "b")], resamples=400, seed=rng)
        covered += low <= math.log(1.25) <= high
    assert 0.85 <= covered / replicates <= 0.98


def write_run(batch_dir, arm, seed, ranks, split="loo", shuffle="none", state="tested"):
    """A final run directory as training and the test read leave it. The ranks of the read
    with shuffled inputs are one place worse."""
    cfg = RunConfig("b", "ml-1m", split, arm, seed, "final", shuffle=shuffle)
    run_dir = batch_dir / cfg.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(cfg.to_json(), encoding="utf-8")
    if state == "failed":
        (run_dir / "error.txt").write_text("out of memory", encoding="utf-8")
        return
    (run_dir / "summary.json").write_text('{"status": "trained"}', encoding="utf-8")
    if state == "tested":
        ranks = np.array(ranks)
        np.savez(run_dir / "test_ranks.npz", test=ranks, test_inputs_shuffled=ranks + 1)
        (run_dir / "test.json").write_text(f'{{"run_id": "{cfg.run_id}"}}', encoding="utf-8")


def batch_of_tested_runs(root):
    """Arm i ranks the three users 1, s + i and 4 in seed s, for seeds 1 and 2."""
    batch = root / "runs" / "b"
    for i, arm in enumerate(ARMS):
        for seed in (1, 2):
            write_run(batch, arm, seed, [1, seed + i, 4])
    return batch


def test_the_report_reads_the_tested_leave_one_out_runs_of_a_batch(tmp_path):
    """Every F1 and F2 difference of the dataset gets a row. A failed run and the runs on the
    temporal split or with shuffled order stay out."""
    batch = batch_of_tested_runs(tmp_path)
    write_run(batch, A2, 3, [], state="failed")
    write_run(batch, A2, 1, [5, 5, 5], split="gts")
    write_run(batch, A2, 1, [5, 5, 5], shuffle="train_eval")
    out = tmp_path / "bootstrap.csv"
    assert cli_main(["bootstrap", str(batch), "--out", str(out)]) == 0
    with out.open(encoding="utf-8", newline="") as f:
        rows = {(r["family"], r["a"], r["b"]): r for r in csv.DictReader(f)}
    assert sorted(rows) == sorted(
        [("F1", A2, A1), ("F1", A4, A1), ("F1", A5, A1), ("F1", A4, A2), ("F1", A2, A5),
         ("F1", A4, A5), ("F2", A3, A2), ("F2", A4, A2)]
    )  # fmt: skip
    row = rows["F1", A2, A1]
    fields = ("dataset", "test", "n_a", "n_b", "users", "resamples")
    assert [row[k] for k in fields] == ["ml-1m", "directional", "2", "2", "3", "10000"]
    y = {arm: [math.log(bootstrap.ndcg(np.array([1, s + i, 4])).mean()) for s in (1, 2)]
         for i, arm in enumerate(ARMS)}  # fmt: skip
    assert float(row["estimate"]) == pytest.approx(np.mean(y[A2]) - np.mean(y[A1]))
    assert float(row["low"]) <= float(row["estimate"]) <= float(row["high"])
    assert rows["F2", A3, A2]["test"] == "equivalence"


def test_a_trained_run_waiting_for_its_read_stops_the_bootstrap(tmp_path):
    """As in the analysis (D051): queued behind a test job that ended early, the bootstrap
    must not pass fewer seeds off as the arm."""
    batch = batch_of_tested_runs(tmp_path)
    write_run(batch, A5, 3, [], state="trained")
    with pytest.raises(ValueError, match=f"{A5}-ml-1m-loo-s3 is trained, not tested"):
        bootstrap.report(batch, tmp_path / "bootstrap.csv", resamples=10)
    assert not (tmp_path / "bootstrap.csv").exists()


def test_a_batch_without_tested_runs_writes_nothing(tmp_path):
    for batch in (tmp_path / "missing", tmp_path / "runs"):
        (tmp_path / "runs").mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="no tested final run"):
            bootstrap.report(batch, tmp_path / "bootstrap.csv", resamples=10)
    assert not (tmp_path / "bootstrap.csv").exists()
