import numpy as np
import pandas as pd
import pytest


def frame(rows):
    """rows: (user, item, ts) tuples in file order."""
    users, items, ts = zip(*rows, strict=True)
    return pd.DataFrame(
        {
            "user": [str(u) for u in users],
            "item": [str(i) for i in items],
            "ts": np.asarray(ts, dtype=np.int64),
            "row": np.arange(len(rows), dtype=np.int64),
        }
    )


def random_frame(seed=0, users=60, items=40, per_user=(6, 15), t_max=1000):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(users):
        n = rng.integers(*per_user)
        for i, t in zip(
            rng.choice(items, n, replace=False), np.sort(rng.integers(0, t_max, n)), strict=True
        ):
            rows.append((f"u{u:03d}", f"i{i:03d}", int(t)))
    rng.shuffle(rows)
    return frame(rows)


@pytest.fixture
def toy_frame():
    return random_frame()
