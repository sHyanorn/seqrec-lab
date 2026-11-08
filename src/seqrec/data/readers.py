"""Readers that turn each raw file into a table of (user, item, ts, row).

`row` is the line order in the source file and breaks timestamp ties (METHODS 2.3).
Users and items are kept as strings so that every dataset is handled the same way.
"""

from __future__ import annotations

import ast
import datetime as dt
import gzip
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["user", "item", "ts", "row"]
_EPOCH = dt.date(1970, 1, 1)


def _finish(user, item, ts) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": pd.Series(user, dtype=str),
            "item": pd.Series(item, dtype=str),
            "ts": np.asarray(ts, dtype=np.int64),
            "row": np.arange(len(ts), dtype=np.int64),
        }
    )


def read_movielens_1m(path: Path) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(path) as z, z.open("ml-1m/ratings.dat") as f:
        text = f.read().decode("latin-1")
    df = pd.read_csv(
        io.StringIO(text.replace("::", "\t")),
        sep="\t",
        header=None,
        names=["user", "item", "rating", "ts"],
        dtype={"user": str, "item": str},
    )
    return _finish(df["user"], df["item"], df["ts"]), {"rows_read": len(df), "rows_skipped": 0}


def read_movielens_20m(path: Path) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(path) as z, z.open("ml-20m/ratings.csv") as f:
        df = pd.read_csv(f, usecols=["userId", "movieId", "timestamp"], dtype=str)
    out = _finish(df["userId"], df["movieId"], df["timestamp"].astype(np.int64))
    return out, {"rows_read": len(df), "rows_skipped": 0}


def read_steam(path: Path, log=print) -> tuple[pd.DataFrame, dict]:
    """Steam reviews are Python dict literals, one per line; the date has day resolution."""
    users, items, days = [], [], []
    skipped = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            record = ast.literal_eval(line)
            user, item, date = record.get("username"), record.get("product_id"), record.get("date")
            if not user or not item or not date:
                skipped += 1
                continue
            users.append(user)
            items.append(item)
            days.append((dt.date.fromisoformat(date) - _EPOCH).days)
            if n % 1_000_000 == 0:
                log(f"[data] steam: {n:,} lines")
    return _finish(users, items, days), {"rows_read": len(days) + skipped, "rows_skipped": skipped}


def read_amazon_2023(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, usecols=["user_id", "parent_asin", "timestamp"], dtype=str)
    out = _finish(df["user_id"], df["parent_asin"], df["timestamp"].astype(np.int64))
    return out, {"rows_read": len(df), "rows_skipped": 0}


READERS = {
    "ml-1m": read_movielens_1m,
    "ml-20m": read_movielens_20m,
    "steam": read_steam,
    "video-games": read_amazon_2023,
}
