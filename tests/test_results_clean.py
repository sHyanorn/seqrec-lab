"""Committed results must hold aggregates only, never row-level data."""

from pathlib import Path

import pytest

from seqrec.data.stats import results_are_aggregate

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.mark.parametrize(
    "path", sorted(RESULTS.rglob("*.csv")), ids=lambda p: str(p.relative_to(RESULTS))
)
def test_committed_csv_is_aggregate(path):
    results_are_aggregate(path)


def test_guard_rejects_row_level_files(tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("dataset,user_id,score\nml-1m,7,0.5\n")
    with pytest.raises(ValueError):
        results_are_aggregate(bad)
