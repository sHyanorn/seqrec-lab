"""The final-run plan of a dataset, generated from its tuning studies (METHODS 5.5, 9.4)."""

import json

import pytest

from seqrec.cli import main as cli_main
from seqrec.data.stats import write_csv
from seqrec.runner.config import load_plan
from seqrec.runner.final_plan import final_plan

NEURAL = ("sasrec-bce", "sasrec-ce", "sasrec-ss", "gsasrec", "bert4rec")
H2A = NEURAL[:4]
PARAMS = {
    "lr": 0.0007447144904083078, "dim": 128, "blocks": 1, "heads": 1,
    "dropout": 0.23341363755143332, "weight_decay": 0.0001, "shared_output": True,
}  # fmt: skip


def best(dataset="ml-1m", batch="tune", shares=None, **replace):
    """Rows of tuning_best.csv as export writes them: one per study."""
    rows = []
    for i, arm in enumerate(NEURAL):
        params = PARAMS | {"dim": 64 * 2 ** (i % 3)}
        if arm == "bert4rec":
            params = {k: v for k, v in params.items() if k != "shared_output"} | {"mask_prob": 0.2}
        row = {
            "batch": batch, "dataset": dataset, "arm": arm, "trials": 20,
            "truncated_share": (shares or {}).get(arm, 0.0),
            "best_run_id": f"{arm}-{dataset}-loo-s0-t07",
            "best_params": json.dumps(params, sort_keys=True),
        }  # fmt: skip
        rows.append(row | replace.get(arm, {}))
    return rows


def runs(text, tmp_path, name="m3-final-ml-1m"):
    path = tmp_path / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return load_plan(path).runs


def key(r):
    return (r.arm, r.split, r.shuffle, r.seed)


def test_the_first_twenty_runs_are_the_h2a_group_in_seed_rounds(tmp_path):
    """METHODS 9.4: A1 to A4 with seeds 1 to 5 on leave-one-out, one after another, seeds as
    rounds and the four arms in an order drawn at random within each seed."""
    plan = runs(final_plan(best(), "ml-1m", "tune", 5, "m3-final-ml-1m"), tmp_path)
    group = plan[:20]
    assert {(r.split, r.shuffle, r.phase) for r in group} == {("loo", "none", "final")}
    rounds = [group[i : i + 4] for i in range(0, 20, 4)]
    assert [{r.seed for r in rnd} for rnd in rounds] == [{s} for s in range(1, 6)]
    assert all(sorted(r.arm for r in rnd) == sorted(H2A) for rnd in rounds)
    assert len({tuple(r.arm for r in rnd) for rnd in rounds}) > 1  # not one fixed order
    assert not any(key(r) in {key(g) for g in group} for r in plan[20:])


def test_the_plan_holds_every_registered_final_run_once(tmp_path):
    """METHODS 5.5 and 3.3 on MovieLens-1M with n = 5: five arms on both splits, the shuffle
    control for SASRec-CE and BERT4Rec, and the two floor baselines once per split."""
    plan = runs(final_plan(best(), "ml-1m", "tune", 5, "m3-final-ml-1m"), tmp_path)
    seeds = range(1, 6)
    expected = {(a, split, "none", s) for a in NEURAL for split in ("loo", "gts") for s in seeds}
    expected |= {(a, "loo", "train_eval", s) for a in ("sasrec-ce", "bert4rec") for s in seeds}
    expected |= {
        (a, split, "none", 0) for a in ("mostpop", "transition") for split in ("loo", "gts")
    }
    assert sorted(map(key, plan)) == sorted(expected)
    assert {(r.batch, r.dataset, r.phase, r.tag) for r in plan} == {
        ("m3-final-ml-1m", "ml-1m", "final", "")
    }
    rest = [key(r) for r in plan[20:]]
    assert rest != sorted(rest) and rest != sorted(rest, key=lambda k: (k[3], k))  # interleaved


def test_runs_take_the_best_trial_of_their_study_exactly(tmp_path):
    rows = best()
    plan = runs(final_plan(rows, "ml-1m", "tune", 5, "m3-final-ml-1m"), tmp_path)
    by_arm = {r["arm"]: json.loads(r["best_params"]) for r in rows}
    for r in plan:
        assert r.params == by_arm.get(r.arm, {})  # floats survive the plan file bit for bit
        assert type(r.params.get("dim", 0)) is int
    assert {r.max_epochs for r in plan if r.arm == "bert4rec"} == {1000}
    assert {(r.max_epochs, r.patience, r.start_limit, r.guard_hours) for r in plan[:20]} == {
        (300, 30, 300, 8.0)
    }


def test_only_the_named_tuning_batch_and_dataset_are_read(tmp_path):
    other = best(batch="m2-tune-smoke", **{"sasrec-ce": {"best_params": '{"dim": 7}'}})
    steam = best(dataset="steam", **{"sasrec-ce": {"best_params": '{"dim": 9}'}})
    plan = runs(final_plan(other + best() + steam, "ml-1m", "tune", 5, "m3-final-ml-1m"), tmp_path)
    assert {r.params["dim"] for r in plan if r.arm == "sasrec-ce"} == {128}


def test_seeds_added_later_extend_the_plan_without_touching_its_first_runs(tmp_path):
    """METHODS 7.5: extra seeds go into the plan before the test read, on leave-one-out only;
    the shuffle control keeps the seeds of the unshuffled runs (3.3)."""
    first = runs(final_plan(best(), "ml-1m", "tune", 5, "m3-final-ml-1m"), tmp_path)
    more = runs(final_plan(best(), "ml-1m", "tune", 8, "m3-final-ml-1m"), tmp_path)
    assert more[: len(first)] == first
    added = {(a, "loo", "none", s) for a in NEURAL for s in (6, 7, 8)}
    added |= {(a, "loo", "train_eval", s) for a in ("sasrec-ce", "bert4rec") for s in (6, 7, 8)}
    assert sorted(map(key, more[len(first) :])) == sorted(added)


@pytest.mark.parametrize("seeds", [4, 11])
def test_the_seed_count_stays_inside_the_rule(seeds):
    with pytest.raises(ValueError, match="5 to 10"):
        final_plan(best(), "ml-1m", "tune", seeds, "m3-final-ml-1m")


def test_an_arm_with_more_than_a_fifth_of_its_trials_truncated_gets_doubled_limits(tmp_path):
    """METHODS 5.2 and D030: strictly more than 20%; all three limits, and the start limit of
    D035 with them, in every final run of that arm on the dataset."""
    rows = best(shares={"gsasrec": 0.25, "sasrec-ce": 0.2, "bert4rec": 0.21})
    plan = runs(final_plan(rows, "ml-1m", "tune", 5, "m3-final-ml-1m"), tmp_path)
    limits = {
        arm: {
            (r.max_epochs, r.patience, r.start_limit, r.guard_hours) for r in plan if r.arm == arm
        }
        for arm in NEURAL
    }
    assert limits["gsasrec"] == {(600, 60, 600, 16.0)}
    assert limits["bert4rec"] == {(2000, 200, 600, 16.0)}
    assert limits["sasrec-ce"] == limits["sasrec-bce"] == {(300, 30, 300, 8.0)}


def test_the_bert4rec_shuffle_control_runs_on_the_movielens_datasets_only(tmp_path):
    plan = runs(
        final_plan(best(dataset="steam"), "steam", "tune", 5, "m3-final-steam"),
        tmp_path,
        "m3-final-steam",
    )
    assert {r.arm for r in plan if r.shuffle == "train_eval"} == {"sasrec-ce"}
    assert len(plan) == 50 + 5 + 4


@pytest.mark.parametrize(
    "broken, message",
    [
        ({"gsasrec": {"dataset": "steam"}}, "gsasrec"),
        ({"bert4rec": {"best_params": ""}}, "bert4rec"),
        ({"sasrec-ss": {"truncated_share": ""}}, "sasrec-ss"),
    ],
)
def test_a_study_that_is_missing_or_has_no_best_trial_stops_the_plan(broken, message):
    with pytest.raises(ValueError, match=message):
        final_plan(best(**broken), "ml-1m", "tune", 5, "m3-final-ml-1m")


def test_the_command_writes_a_plan_named_after_its_batch(tmp_path):
    write_csv(tmp_path / "tuning_best.csv", best(batch="m3-tune-ml-1m"))
    out = tmp_path / "m3-final-ml-1m.toml"
    args = ["final-plan", str(tmp_path / "tuning_best.csv"), "--tuning-batch", "m3-tune-ml-1m"]
    assert cli_main([*args, "--dataset", "ml-1m", "--seeds", "5", "--out", str(out)]) == 0
    plan = load_plan(out)
    assert plan.batch == "m3-final-ml-1m" and plan.warmup_s == 180 and len(plan.runs) == 64
    assert 'batch = "m3-final-ml-1m"\n' in out.read_text()  # the form the workflow checks
