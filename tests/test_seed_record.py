"""The seed count checked before a final test read (METHODS 5.5, 7.5; D064)."""

import json
from itertools import product

import pytest
from conftest import random_frame

import seqrec.models  # noqa: F401  registers the arms
from seqrec.analysis import power
from seqrec.cli import main as cli_main
from seqrec.data.preprocess import build_log
from seqrec.data.store import load_splits, log_path
from seqrec.runner import batch as batch_mod
from seqrec.runner.batch import SeedCountRefused
from seqrec.runner.config import RunConfig

QUIET = {"log": lambda _: None}
ARMS = power.NEURAL_ARMS


def _value(a: int, seed: int) -> float:
    return 0.15 + 0.01 * a + 0.001 * seed


def _final_export(path, seeds=range(1, 8)):
    """Final runs of five arms on leave-one-out, as the export writes them."""
    lines = ["run_id,arm,seed,tag,dataset,split,shuffle,status,validation_ndcg@10"]
    for (a, arm), seed in product(enumerate(ARMS), seeds):
        run_id = RunConfig("b", "ml-1m", "loo", arm, seed, "final").run_id
        lines.append(f"{run_id},{arm},{seed},,ml-1m,loo,none,trained,{_value(a, seed)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_record_holds_the_estimate_and_the_runs_it_rests_on(tmp_path):
    path = _final_export(tmp_path / "b.csv")
    record = power.seed_record(path, "ml-1m", range(1, 6))
    sigma, _ = power.sigma_from_export(path, "", "ml-1m", seeds=range(1, 6))
    n = power.seed_count(sigma)
    assert (record["dataset"], record["seeds"], record["sigma"], record["n"]) == (
        "ml-1m", [1, 2, 3, 4, 5], sigma, n
    )  # fmt: skip
    assert record["power_directional"] == power.directional_power(sigma, n)
    assert record["power_equivalence"] == power.equivalence_power(sigma, n)
    runs = [(r["arm"], r["seed"], r["validation_ndcg@10"]) for r in record["runs"]]
    assert runs == [(arm, s, _value(a, s)) for (a, arm), s in product(enumerate(ARMS), range(1, 6))]
    assert all(r["run_id"].endswith(f"-s{r['seed']}") for r in record["runs"])


def test_seeds_writes_the_record_it_prints(tmp_path, capsys):
    path = _final_export(tmp_path / "b.csv")
    out = tmp_path / "seeds-ml-1m.json"
    args = ["seeds", str(path), "--tag", "", "--dataset", "ml-1m", "--seeds", "1", "2", "3", "4"]
    assert cli_main([*args, "5", "--write", str(out)]) == 0
    assert "pooled sigma" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8")) == power.seed_record(
        path, "ml-1m", range(1, 6)
    )


@pytest.mark.parametrize("args", [["--tag", "sigma"], []])
def test_a_record_is_written_only_for_the_first_final_seeds(tmp_path, args):
    """The record is the estimate after the first five final seeds (METHODS 7.5)."""
    path = _final_export(tmp_path / "b.csv")
    out = tmp_path / "seeds.json"
    seeds = ["--seeds", "1", "2", "3", "4", "5"] if args else []
    with pytest.raises(SystemExit):
        cli_main(["seeds", str(path), *(args or ["--tag", ""]), *seeds, "--write", str(out)])
    assert not out.exists()


# --- the check before a test read ----------------------------------------------------------


@pytest.fixture
def volume(tmp_path, monkeypatch):
    """Runs of a final plan on a volume, as training leaves them, and the directory of
    committed batch files. Reads are stubbed: the check comes before any of them."""
    data = tmp_path / "data"
    build_log(random_frame()).save(log_path(data, "ml-1m"))
    load_splits.cache_clear()
    monkeypatch.chdir(tmp_path)
    tested = []
    monkeypatch.setattr(batch_mod, "_test", lambda cfg, *args: tested.append(cfg.run_id) or True)
    committed = tmp_path / "results" / "runs"
    committed.mkdir(parents=True)
    yield tmp_path / "runs", data, committed, tested
    load_splits.cache_clear()


def _plan(root, seeds=6, controls=6, phase="final"):
    """Five neural arms with `seeds` leave-one-out seeds, the shuffle control with
    `controls`, and the temporal split with five."""
    text = f'batch = "b"\n[defaults]\ndataset = "ml-1m"\nsplit = "loo"\nphase = "{phase}"\n'
    for arm in ARMS:
        text += f'[[runs]]\narm = "{arm}"\nseeds = {list(range(1, seeds + 1))}\n'
        text += f'[[runs]]\narm = "{arm}"\nsplit = "gts"\nseeds = [1, 2, 3, 4, 5]\n'
    shuffled = list(range(1, controls + 1))
    text += f'[[runs]]\narm = "sasrec-ce"\nshuffle = "train_eval"\nseeds = {shuffled}\n'
    plan = root / "plan.toml"
    plan.write_text(text, encoding="utf-8")
    return plan


def _train(runs, plan, value=_value):
    """Summaries of trained runs; leave-one-out runs score `value(arm index, seed)`."""
    from seqrec.runner.config import load_plan
    from seqrec.runner.run import run_dir_of

    for cfg in load_plan(plan).runs:
        run_dir = run_dir_of(cfg, runs)
        run_dir.mkdir(parents=True, exist_ok=True)
        score = value(ARMS.index(cfg.arm), cfg.seed) if cfg.split == "loo" else 0.1
        summary = {"status": "trained", "validation": {"ndcg@10": score}}
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _record(committed, root, n=6, **change):
    record = power.seed_record(_final_export(root / "b.csv"), "ml-1m", range(1, 6))
    record |= {"n": n} | change
    (committed / "seeds-ml-1m.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def _test(plan, runs, data, committed):
    return batch_mod.test_batch(plan, runs, data, None, committed=committed, **QUIET)


def test_a_final_plan_that_matches_its_record_is_tested(volume):
    runs, data, committed, tested = volume
    plan = _plan(runs.parent)
    _train(runs, plan)
    _record(committed, runs.parent)
    assert _test(plan, runs, data, committed) == []
    assert len(tested) == 5 * 6 + 5 * 5 + 6


def test_a_final_plan_without_its_record_is_refused_before_any_read(volume):
    runs, data, committed, tested = volume
    plan = _plan(runs.parent)
    _train(runs, plan)
    with pytest.raises(SeedCountRefused, match="seeds-ml-1m.json"):
        _test(plan, runs, data, committed)
    assert tested == [] and not any(runs.rglob("test_access.jsonl"))


@pytest.mark.parametrize(
    ("seeds", "controls"), [(5, 5), (7, 7), (6, 5)], ids=["fewer", "more", "control"]
)
def test_a_plan_with_other_seeds_than_its_record_is_refused(volume, seeds, controls):
    """Every neural arm and the shuffle control need seeds 1 to n on leave-one-out."""
    runs, data, committed, tested = volume
    plan = _plan(runs.parent, seeds, controls)
    _train(runs, plan)
    _record(committed, runs.parent, n=6)
    with pytest.raises(SeedCountRefused, match="seeds 1 to 6"):
        _test(plan, runs, data, committed)
    assert tested == []


def test_a_record_whose_values_differ_from_the_volume_is_refused(volume):
    """A record written from another export than the runs on this volume."""
    runs, data, committed, tested = volume
    plan = _plan(runs.parent)
    _train(runs, plan, value=lambda a, seed: _value(a, seed) + (seed == 3) * 1e-12)
    _record(committed, runs.parent)
    with pytest.raises(SeedCountRefused, match="validation NDCG@10"):
        _test(plan, runs, data, committed)
    assert tested == []


def test_a_record_of_another_dataset_or_other_runs_is_refused(volume):
    runs, data, committed, tested = volume
    plan = _plan(runs.parent)
    _train(runs, plan)
    record = _record(committed, runs.parent, dataset="steam")
    with pytest.raises(SeedCountRefused, match="steam"):
        _test(plan, runs, data, committed)
    _record(committed, runs.parent, runs=record["runs"][:-1])
    with pytest.raises(SeedCountRefused, match="seeds 1 to 5 of every neural arm"):
        _test(plan, runs, data, committed)
    assert tested == []


def _point(entries, which, arm, seed):
    """Entries `which` name the run of `arm` and `seed`, with that run's value on the volume,
    and keep their own arm and seed."""
    run_id = RunConfig("b", "ml-1m", "loo", arm, seed, "final").run_id
    value = _value(ARMS.index(arm), seed)
    return [
        e | {"run_id": run_id, "validation_ndcg@10": value} if i in which else e
        for i, e in enumerate(entries)
    ]


@pytest.mark.parametrize(
    "point",
    [
        lambda entries: _point(entries, {0}, "sasrec-bce", 6),
        lambda entries: _point(entries, {0}, "bert4rec", 1),
        lambda entries: _point(entries, set(range(25)), "sasrec-ce", 2),
    ],
    ids=["another seed of the arm", "another arm", "one run for all entries"],
)
def test_a_record_whose_entries_name_other_runs_is_refused(volume, point):
    """Each entry's arm and seed are those of the run it names, and no run is named twice;
    every value matches the volume, so only these comparisons see it."""
    runs, data, committed, tested = volume
    plan = _plan(runs.parent)
    _train(runs, plan)
    record = _record(committed, runs.parent)
    _record(committed, runs.parent, runs=point(record["runs"]))
    with pytest.raises(SeedCountRefused, match="is not a run of this plan|named twice"):
        _test(plan, runs, data, committed)
    assert tested == []


def test_a_final_plan_needs_the_directory_of_committed_batch_files(volume):
    runs, data, _, tested = volume
    plan = _plan(runs.parent)
    _train(runs, plan)
    with pytest.raises(SeedCountRefused):
        batch_mod.test_batch(plan, runs, data, None, **QUIET)
    assert tested == []


def test_an_anchor_plan_needs_no_record(volume):
    runs, data, committed, tested = volume
    plan = _plan(runs.parent, phase="anchor")
    _train(runs, plan)
    assert _test(plan, runs, data, committed) == []
    assert len(tested) == 5 * 6 + 5 * 5 + 6
