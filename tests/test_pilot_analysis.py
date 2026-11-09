import csv
import math

import pytest

from seqrec.analysis import anchors, budget, power
from seqrec.cli import main as cli_main

ARMS = ("sasrec-bce", "sasrec-ce", "sasrec-ss", "gsasrec", "bert4rec")


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
    reads = {"reads_anchor": 1, "reads_completed": 1, "reads_final": 0}
    return run_row(
        arm, "r1", val, phase="anchor", status="tested", **({"test_ndcg@10": test} | reads | kw)
    )


def test_band_edges_follow_the_methods_table():
    ce = anchors.ANCHORS["sasrec-ce"]
    assert [anchors.position(v, ce) for v in (0.172, 0.192, 0.1719, 0.1921, None)] == [
        "inside", "inside", "below", "above", "missing",
    ]  # fmt: skip


def test_gap_margin_is_three_bounded_standard_errors():
    expected = 3 * math.sqrt(2 * 0.18 * 0.82 / 6040)
    assert anchors.gap_margin(0.18, 0.18) == pytest.approx(expected)
    assert 0.020 < expected < 0.022


def test_anchor_table_bands_gaps_and_access(tmp_path):
    path = write(
        tmp_path / "anchor.csv",
        [
            anchor_row("sasrec-ce", 0.182, 0.180, best_epoch=62),
            anchor_row("sasrec-bce", 0.160, 0.130),  # above band, test far above validation
            anchor_row("bert4rec", 0.150, 0.180, reads_completed=0),  # test far below
            anchor_row("gsasrec", 0.170, 0.171),
            run_row("sasrec-ce", "sigma", 0.18),  # not an anchor run
        ],
    )
    table = {r["arm"]: r for r in anchors.anchor_table(path)}
    assert list(table) == ["sasrec-ce", "sasrec-bce", "bert4rec", "gsasrec"]
    assert table["sasrec-ce"]["position"] == "inside" and table["sasrec-ce"]["best_epoch"] == 63
    assert table["sasrec-ce"]["gap_flag"] == ""
    assert table["sasrec-bce"]["position"] == "above"
    assert table["sasrec-bce"]["gap_flag"] == "test above validation"
    assert table["bert4rec"]["gap_flag"] == "test below validation"
    assert anchors.access_issues(list(table.values())) == [
        "bert4rec r1: an anchor read did not complete"
    ]


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


def pilot_rows(train_small=1.0, best=None):
    best = best or {"sasrec-bce": 53, "sasrec-ce": 63, "sasrec-ss": 63, "gsasrec": 53,
                    "bert4rec": 197}  # fmt: skip
    rows = []
    for d in budget.DATASETS:
        for a in ARMS:
            for size, scale in (("small", 1.0), ("large", 3.0)):
                rows.append(
                    run_row(a, f"time-{size}", 0.1, dataset=d, epochs=3,
                            epoch_train_s_median=train_small * scale,
                            epoch_eval_s_median=0.2 * scale)
                )  # fmt: skip
    for a in ARMS:
        rows += [run_row(a, "sigma", 0.1 + s / 1000, best_epoch=best[a] - 1) for s in range(3)]
    return rows


@pytest.fixture
def pilot(tmp_path):
    path = write(tmp_path / "m2-pilot.csv", pilot_rows())
    write(tmp_path / "overview.csv", [{"batch": "m2-anchor", "gpu_hours": 2.5}])
    return budget.load_pilot(path, tmp_path / "overview.csv")


def test_epoch_ratios_come_from_the_reference_table():
    assert budget.epoch_ratio("steam", "sasrec-ce") == pytest.approx(16 / 63)
    assert budget.epoch_ratio("ml-20m", "gsasrec") == pytest.approx(30 / 53)
    assert budget.epoch_ratio("ml-20m", "bert4rec") == pytest.approx(68 / 197)
    assert budget.epoch_ratio("video-games", "bert4rec") == 1.0


def test_run_hours_scale_epochs_add_patience_and_respect_the_cap(pilot, tmp_path):
    central, ml1m, upper = budget.SCENARIOS
    epochs, hours = budget.run_hours(pilot, "steam", "sasrec-ce", central)
    assert epochs == pytest.approx(16 + 30)
    assert hours == pytest.approx((46 * (2.0 + 0.4) + 0.4) / 3600)  # mean of small and large
    assert budget.run_hours(pilot, "steam", "sasrec-ce", ml1m)[0] == pytest.approx(63 + 30)
    assert budget.run_hours(pilot, "ml-1m", "bert4rec", upper)[1] == pytest.approx(
        (297 * 3.6 + 0.6) / 3600
    )
    capped = write(tmp_path / "capped.csv", pilot_rows(best={a: 990 for a in ARMS}))
    pilot2 = budget.load_pilot(capped, tmp_path / "overview.csv")
    assert budget.run_hours(pilot2, "ml-1m", "sasrec-ce", central)[0] == 300
    assert budget.run_hours(pilot2, "ml-1m", "bert4rec", central)[0] == 1000


def test_projection_adds_up_every_block(pilot):
    scenario, scope = budget.SCENARIOS[0], budget.Scope(seeds=6)
    h = {(d, a): budget.run_hours(pilot, d, a, scenario)[1] for d in budget.DATASETS for a in ARMS}
    blocks = budget.project(pilot, scope, scenario)
    assert blocks["spent"] == 2.5
    assert blocks["tuning"] == pytest.approx(sum(budget.TRIALS[d] * v for (d, _), v in h.items()))
    assert blocks["final_loo"] == pytest.approx(6 * sum(h.values()))
    assert blocks["final_temporal"] == pytest.approx(5 * sum(h.values()))
    shuffled = sum(h[(d, "sasrec-ce")] for d in budget.DATASETS)
    shuffled += h[("ml-1m", "bert4rec")] + h[("ml-20m", "bert4rec")]
    assert blocks["shuffle_control"] == pytest.approx(6 * shuffled)
    assert blocks["warmup"] == pytest.approx(13 * 180 / 3600) and blocks["rq3"] == 15
    parts = ("spent", "tuning", "final_loo", "final_temporal", "shuffle_control", "warmup", "rq3")
    assert blocks["core"] == pytest.approx(sum(blocks[k] for k in parts))
    assert blocks["reserve"] == pytest.approx(200 - blocks["core"])


def test_cuts_follow_the_order_of_d012(tmp_path):
    write(tmp_path / "overview.csv", [{"batch": "m2-anchor", "gpu_hours": 2.5}])
    cheap = budget.load_pilot(write(tmp_path / "a.csv", pilot_rows()), tmp_path / "overview.csv")
    assert budget.cuts(cheap, budget.Scope(seeds=6)) == []
    costly = budget.load_pilot(
        write(tmp_path / "b.csv", pilot_rows(train_small=60.0)), tmp_path / "overview.csv"
    )
    steps = budget.cuts(costly, budget.Scope(seeds=6))
    assert [name for name, _ in steps] == [
        "halve MovieLens-20M tuning trials",
        "drop the MovieLens-20M temporal split",
        "drop Steam",
    ]
    cores = [core for _, core in steps]
    assert cores == sorted(cores, reverse=True)


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
    assert cli_main(["anchors", str(anchor), "--pilot", str(pilot)]) == 0
    out = capsys.readouterr().out
    assert "test access: as expected" in out and "raise_cap" in out
    assert cli_main(["budget", str(pilot), "--seeds", "6"]) == 0
    out = capsys.readouterr().out
    assert "seeds per arm (leave-one-out): 6" in out and "no cut" in out
