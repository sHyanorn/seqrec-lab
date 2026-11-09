import csv
import json
import math
from pathlib import Path

import pytest

from seqrec.analysis import anchors, budget, power
from seqrec.cli import main as cli_main
from seqrec.runner.config import load_plan

ARMS = ("sasrec-bce", "sasrec-ce", "sasrec-ss", "gsasrec", "bert4rec")
ANCHOR_PLAN = Path(__file__).resolve().parents[1] / "configs" / "plans" / "m2-anchor.toml"
PLANNED = {r.arm: r for r in load_plan(ANCHOR_PLAN).runs}


def write(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_row(arm, tag, val, phase="tune", dataset="ml-1m", **kw):
    return {
        "dataset": dataset, "split": "loo", "arm": arm, "phase": phase, "shuffle": "none",
        "tag": tag, "status": "trained", "validation_ndcg@10": val, "epochs": 40,
        "best_epoch": 9, "truncated": False, "stop_reason": "patience",
    } | kw  # fmt: skip


def anchor_row(arm, test, val, **kw):
    planned = PLANNED[arm]
    fixed = {
        "test_ndcg@10": test, "reads_anchor": 1, "reads_completed": 1, "reads_final": 0,
        "run_id": planned.run_id, "params": json.dumps(planned.params, sort_keys=True),
        "max_epochs": planned.max_epochs, "patience": planned.patience, "git_commit": "70b2b0f",
    }  # fmt: skip
    tag = kw.pop("tag", "r1")
    return run_row(arm, tag, val, phase="anchor", status="tested", **(fixed | kw))


def test_band_edges_follow_the_methods_table():
    ce = anchors.ANCHORS["sasrec-ce"]
    assert [anchors.position(v, ce) for v in (0.172, 0.192, 0.1719, 0.1921, None)] == [
        "inside", "inside", "below", "above", "missing",
    ]  # fmt: skip


def test_gap_margin_is_three_bounded_standard_errors():
    expected = 3 * math.sqrt(2 * 0.18 * 0.82 / 6040)
    assert anchors.gap_margin(0.18, 0.18) == pytest.approx(expected)
    assert 0.020 < expected < 0.022


def test_anchor_table_bands_gaps_and_stop_conditions(tmp_path):
    path = write(
        tmp_path / "anchor.csv",
        [
            anchor_row("sasrec-ce", 0.182, 0.180, best_epoch=62),
            anchor_row("sasrec-bce", 0.160, 0.130),  # above band, test far above validation
            anchor_row("bert4rec", 0.150, 0.180, reads_completed=0),  # test far below
            run_row("sasrec-ce", "sigma", 0.18),  # not an anchor run
        ],
    )
    table = {r["arm"]: r for r in anchors.anchor_table(path)}
    assert list(table) == ["sasrec-ce", "sasrec-bce", "bert4rec"]
    assert table["sasrec-ce"]["position"] == "inside" and table["sasrec-ce"]["best_epoch"] == 63
    assert table["sasrec-ce"]["gap_flag"] == ""
    assert table["sasrec-bce"]["position"] == "above"
    assert table["sasrec-bce"]["gap_flag"] == "test above validation"
    assert table["bert4rec"]["gap_flag"] == "test below validation"
    issues = anchors.stop_conditions(list(table.values()), load_plan(ANCHOR_PLAN))
    assert issues[0] == "gsasrec: no anchor run"
    assert any(i.startswith("sasrec-bce r1: test NDCG@10 above the band") for i in issues)
    assert any(i.startswith("sasrec-bce r1: test above validation") for i in issues)
    assert "bert4rec r1: 1 anchor reads, 0 done" in issues
    assert not any(i.startswith("sasrec-ce") for i in issues)


def test_parameters_limits_and_commits_are_checked_against_the_plan(tmp_path):
    rows = [anchor_row(a, 0.17 if a == "gsasrec" else 0.18, 0.17) for a in PLANNED]
    rows[0]["params"] = rows[0]["params"].replace("0.001", "0.002")
    rows[1]["patience"] = 30
    rows[2]["git_commit"] = "abc1234"
    table = anchors.anchor_table(write(tmp_path / "anchor.csv", rows))
    issues = anchors.stop_conditions(table, load_plan(ANCHOR_PLAN))
    names = [r["arm"] for r in rows]
    assert f"{names[0]} r1: parameters differ from the anchor plan" in issues
    assert f"{names[1]} r1: patience differs from the anchor plan" in issues
    assert any(i.startswith("anchor runs come from several commits") for i in issues)


def test_cap_cost_compares_seed_runs_with_the_anchor(tmp_path):
    anchor = write(
        tmp_path / "anchor.csv",
        [anchor_row("sasrec-ce", 0.18, 0.180), anchor_row("gsasrec", 0.17, 0.170)],
    )
    pilot = write(
        tmp_path / "pilot.csv",
        [run_row("sasrec-ce", "sigma", v) for v in (0.181, 0.179, 0.180)]
        + [run_row("gsasrec", "sigma", v, truncated=True) for v in (0.150, 0.152, 0.149)]
        + [run_row("sasrec-ss", "sigma", v) for v in (0.170, 0.171, 0.169)],
    )
    sigma, rows = anchors.cap_cost(anchor, pilot)
    assert sigma == pytest.approx(power.sigma_from_export(pilot, "sigma", "ml-1m")[0])
    by_arm = {r["arm"]: r for r in rows}
    logs = [math.log(v) for v in (0.150, 0.152, 0.149)]
    diff = sum(logs) / 3 - math.log(0.17)
    assert by_arm["gsasrec"]["log_diff"] == pytest.approx(diff)
    assert by_arm["gsasrec"]["z"] == pytest.approx(diff / (sigma * math.sqrt(4 / 3)))
    assert by_arm["gsasrec"]["raise_cap"] and by_arm["gsasrec"]["truncated_runs"] == 3
    assert not by_arm["sasrec-ce"]["raise_cap"]
    assert "z" not in by_arm["sasrec-ss"] and by_arm["sasrec-ss"]["anchor_validation"] is None


def pilot_rows(train_small=1.0, best=None, checkpoint_s=0.3):
    best = best or {"sasrec-bce": 53, "sasrec-ce": 63, "sasrec-ss": 63, "gsasrec": 53,
                    "bert4rec": 197}  # fmt: skip
    rows = []
    for d in budget.DATASETS:
        for a in ARMS:
            for size, scale in (("small", 1.0), ("large", 3.0)):
                rows.append(
                    run_row(a, f"time-{size}", 0.1, dataset=d, epochs=3,
                            epoch_train_s_median=train_small * scale - 0.1,
                            epoch_eval_s_median=0.2 * scale, checkpoint_s=checkpoint_s)
                )  # fmt: skip
    for a in ARMS:
        rows += [run_row(a, "sigma", 0.1 + s / 1000, best_epoch=best[a] - 1) for s in range(3)]
    return rows


@pytest.fixture
def pilot(tmp_path):
    path = write(tmp_path / "m2-pilot.csv", pilot_rows())
    write(
        tmp_path / "overview.csv",
        [{"batch": "m2-anchor", "gpu_hours": 2.5}, {"batch": "tune-ml-1m", "gpu_hours": 30.0}],
    )
    return budget.load_pilot(path, tmp_path / "overview.csv")


def test_spent_hours_are_the_m2_batches_only(pilot):
    assert pilot.spent_hours == 2.5


def test_epoch_ratios_come_from_the_reference_table():
    assert budget.epoch_ratio("steam", "sasrec-ce") == pytest.approx(16 / 63)
    assert budget.epoch_ratio("ml-20m", "gsasrec") == pytest.approx(30 / 53)
    assert budget.epoch_ratio("ml-20m", "bert4rec") == pytest.approx(68 / 197)
    assert budget.epoch_ratio("video-games", "bert4rec") == 1.0


def test_run_hours_scale_epochs_add_patience_and_respect_the_cap(pilot, tmp_path):
    central, ml1m, upper = budget.SCENARIOS
    epochs, hours = budget.run_hours(pilot, "steam", "sasrec-ce", central)
    assert epochs == pytest.approx(16 + 30)
    # mean of small and large; training time includes 0.1 s of checkpointing per epoch
    assert hours == pytest.approx((46 * (2.0 + 0.4) + 0.4) / 3600)
    assert budget.run_hours(pilot, "steam", "sasrec-ce", ml1m)[0] == pytest.approx(63 + 30)
    assert budget.run_hours(pilot, "ml-1m", "bert4rec", upper)[1] == pytest.approx(
        (297 * 3.6 + 0.6) / 3600
    )
    capped = write(tmp_path / "capped.csv", pilot_rows(best={a: 990 for a in ARMS}))
    pilot2 = budget.load_pilot(capped, tmp_path / "overview.csv")
    assert budget.run_hours(pilot2, "ml-1m", "sasrec-ce", central)[0] == 300
    assert budget.run_hours(pilot2, "ml-1m", "bert4rec", central)[0] == 1000


def test_projection_adds_up_every_block(pilot):
    scenario, scope = budget.SCENARIOS[0], budget.Scope()
    h = {(d, a): budget.run_hours(pilot, d, a, scenario)[1] for d in budget.DATASETS for a in ARMS}
    blocks = budget.project(pilot, scope, scenario)
    assert blocks["spent"] == 2.5
    assert blocks["tuning"] == pytest.approx(sum(budget.TRIALS[d] * v for (d, _), v in h.items()))
    assert blocks["final_loo"] == pytest.approx(5 * sum(h.values()))  # METHODS 12: 5 seeds
    assert blocks["final_temporal"] == pytest.approx(5 * sum(h.values()))
    shuffled = sum(h[(d, "sasrec-ce")] for d in budget.DATASETS)
    shuffled += h[("ml-1m", "bert4rec")] + h[("ml-20m", "bert4rec")]
    assert blocks["shuffle_control"] == pytest.approx(5 * shuffled)
    extra = budget.extra_seed_hours(pilot, scope, scenario)
    assert extra["ml-20m"] == pytest.approx(
        sum(h[("ml-20m", a)] for a in ARMS) + h[("ml-20m", "sasrec-ce")] + h[("ml-20m", "bert4rec")]
    )
    assert blocks["warmup"] == pytest.approx(13 * 180 / 3600) and blocks["rq3"] == 15
    parts = ("spent", "tuning", "final_loo", "final_temporal", "shuffle_control", "warmup", "rq3")
    assert blocks["core"] == pytest.approx(sum(blocks[k] for k in parts))
    assert blocks["reserve"] == pytest.approx(500 - blocks["core"])  # D034 amendment


def test_cuts_follow_the_order_of_d012(tmp_path):
    write(tmp_path / "overview.csv", [{"batch": "m2-anchor", "gpu_hours": 2.5}])
    cheap = budget.load_pilot(write(tmp_path / "a.csv", pilot_rows()), tmp_path / "overview.csv")
    assert budget.cuts(cheap, budget.Scope())[1] == []
    costly = budget.load_pilot(
        write(tmp_path / "b.csv", pilot_rows(train_small=60.0)), tmp_path / "overview.csv"
    )
    scope, steps = budget.cuts(costly, budget.Scope())
    assert [name for name, _ in steps] == [
        "halve MovieLens-20M tuning trials",
        "drop the MovieLens-20M temporal split",
        "drop Steam",
    ]
    cores = [core for _, core in steps]
    assert cores == sorted(cores, reverse=True)
    assert scope.datasets == ("ml-1m", "video-games", "ml-20m") and scope.trials["ml-20m"] == 5
    assert budget.project(costly, scope, budget.SCENARIOS[0])["warmup"] == pytest.approx(
        (3 * 2 + 2 + 1) * 180 / 3600
    )


def test_extra_seeds_follow_the_priority_order():
    costs = {"ml-1m": 1.0, "video-games": 2.0, "steam": 3.0, "ml-20m": 4.0}
    assert budget.allocate_seeds(10.0, costs, 7) == {
        "ml-1m": 7, "video-games": 7, "steam": 6, "ml-20m": 5,
    }  # fmt: skip
    assert budget.allocate_seeds(100.0, costs, 5) == dict.fromkeys(costs, 5)
    assert budget.allocate_seeds(-3.0, costs, 8) == dict.fromkeys(costs, 5)


def test_missing_pilot_rows_are_named(tmp_path):
    rows = [r for r in pilot_rows() if not (r["tag"] == "time-large" and r["dataset"] == "steam"
                                            and r["arm"] == "gsasrec")]  # fmt: skip
    write(tmp_path / "overview.csv", [{"batch": "m2-pilot", "gpu_hours": 1}])
    with pytest.raises(ValueError, match="steam/gsasrec/large"):
        budget.load_pilot(write(tmp_path / "p.csv", rows), tmp_path / "overview.csv")


def test_commands_print_their_tables(tmp_path, capsys):
    anchor = write(tmp_path / "m2-anchor.csv", [anchor_row(a, 0.17, 0.17) for a in ARMS[:2]])
    pilot = write(tmp_path / "m2-pilot.csv", pilot_rows())
    write(tmp_path / "overview.csv", [{"batch": "m2-pilot", "gpu_hours": 1}])
    args = ["anchors", str(anchor), "--plan", str(ANCHOR_PLAN), "--pilot", str(pilot)]
    assert cli_main(args) == 0
    out = capsys.readouterr().out
    assert "STOP bert4rec: no anchor run" in out and "raise_cap" in out
    assert cli_main(["budget", str(pilot), "--seeds", "6"]) == 0
    out = capsys.readouterr().out
    assert "seeds from the METHODS 7.5 rule: 6" in out and "no cut" in out


def test_only_the_latest_anchor_run_of_an_arm_is_checked(tmp_path):
    plan = tmp_path / "anchor.toml"
    bce = PLANNED["sasrec-bce"]
    params = ", ".join(f"{k} = {str(v).lower()}" for k, v in bce.params.items())
    plan.write_text(
        'batch = "m2-anchor"\n[defaults]\ndataset = "ml-1m"\nsplit = "loo"\n'
        'phase = "anchor"\nseeds = [0]\n'
        + "".join(
            f'[[runs]]\narm = "sasrec-bce"\ntag = "{tag}"\nmax_epochs = 100\n'
            f"patience = 10\nparams = {{ {params} }}\n"
            for tag in ("r1", "r2")
        ),
        encoding="utf-8",
    )
    r2 = load_plan(plan).runs[1]
    rows = [
        anchor_row("sasrec-bce", 0.160, 0.155),  # r1: above the band, an older commit
        anchor_row("sasrec-bce", 0.134, 0.135, tag="r2", run_id=r2.run_id, git_commit="def5678"),
    ]
    table = anchors.anchor_table(write(tmp_path / "anchor.csv", rows))
    assert [r["tag"] for r in table] == ["r1", "r2"]
    issues = anchors.stop_conditions(table, load_plan(plan))
    assert not any(i.startswith("sasrec-bce") or "commits" in i for i in issues)
    assert len(issues) == 3  # the three arms without an anchor run


def test_a_later_export_replaces_the_runs_it_repeats(tmp_path, capsys):
    """D035: the gSASRec rerun stands in for the pilot's gSASRec runs of the same tag."""
    stuck = {"best_epoch": 13, "epochs": 44}
    rows = [r for r in pilot_rows() if not (r["arm"] == "gsasrec" and r["tag"] == "sigma")]
    rows += [run_row("gsasrec", "sigma", v, **stuck) for v in (0.0216, 0.1841, 0.0216)]
    pilot = write(tmp_path / "m2-pilot.csv", rows)
    rerun_rows = [run_row("gsasrec", "sigma", v, best_epoch=220) for v in (0.183, 0.185, 0.184)]
    rerun = write(tmp_path / "m2-gsasrec.csv", rerun_rows)
    write(tmp_path / "overview.csv", [{"batch": "m2-pilot", "gpu_hours": 1}])

    sigma, groups = power.sigma_from_export([pilot, rerun], "sigma", "ml-1m")
    assert groups["gsasrec"] == pytest.approx([math.log(v) for v in (0.183, 0.185, 0.184)])
    assert len(groups["sasrec-ce"]) == 3 and sigma < 0.01
    anchor = write(tmp_path / "m2-anchor.csv", [anchor_row("gsasrec", 0.17, 0.1828)])
    gsasrec = {r["arm"]: r for r in anchors.cap_cost(anchor, [pilot, rerun])[1]}["gsasrec"]
    assert not gsasrec["raise_cap"]
    loaded = budget.load_pilot([pilot, rerun], tmp_path / "overview.csv")
    assert loaded.best_epoch["gsasrec"] == 221
    assert loaded.epoch_s == budget.load_pilot(pilot, tmp_path / "overview.csv").epoch_s

    assert cli_main(["seeds", str(pilot), str(rerun), "--tag", "sigma"]) == 0
    assert "gsasrec: 3 runs, sd 0.00543" in capsys.readouterr().out
    args = ["anchors", str(anchor), "--plan", str(ANCHOR_PLAN), "--pilot", str(pilot), str(rerun)]
    assert cli_main(args) == 0
    assert cli_main(["budget", str(pilot), str(rerun), "--seeds", "6"]) == 0


def test_a_rerun_with_an_unfinished_run_is_refused(tmp_path, capsys):
    """A later export replaces whole groups, so a failed seed in it would shrink n and sigma
    without notice; the analysis stops instead (D040)."""
    pilot = write(tmp_path / "m2-pilot.csv", pilot_rows())
    rows = [run_row("gsasrec", "sigma", v, run_id=f"g{i}") for i, v in enumerate((0.18, 0.19))]
    rows.append(run_row("gsasrec", "sigma", "", run_id="g2", status="failed"))
    rerun = write(tmp_path / "m2-gsasrec.csv", rows)
    with pytest.raises(power.IncompleteRerun, match="g2 is failed"):
        power.read_exports([pilot, rerun])
    assert cli_main(["seeds", str(pilot), str(rerun)]) == 1
    assert "g2 is failed" in capsys.readouterr().err
    assert len(power.read_exports(pilot)) == len(pilot_rows())  # the first file is not checked


def test_a_rerun_missing_a_seed_is_refused(tmp_path):
    """A seed whose run never started has no row; the replaced group must keep its seeds."""
    rows = [r for r in pilot_rows() if not (r["arm"] == "gsasrec" and r["tag"] == "sigma")]
    rows += [run_row("gsasrec", "sigma", 0.0216, seed=s) for s in (1, 2, 3)]
    pilot = write(tmp_path / "m2-pilot.csv", rows)
    rerun_rows = [run_row("gsasrec", "sigma", v, seed=s) for s, v in ((1, 0.18), (2, 0.19))]
    rerun = write(tmp_path / "m2-gsasrec.csv", rerun_rows)
    with pytest.raises(power.IncompleteRerun, match="seeds 1, 2, 3 with 1, 2"):
        power.read_exports([pilot, rerun])
