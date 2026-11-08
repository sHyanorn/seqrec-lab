"""Deterministic preprocessing (METHODS section 2).

`Log` holds every deduplicated interaction in stable order; both splits start from it.
Sequences are stored in CSR form: user u owns positions indptr[u]:indptr[u+1].
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(str((a.dtype.str, a.shape)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Log:
    """Deduplicated interactions sorted by (user, timestamp, source row).

    user and item are codes into the sorted key arrays; ts is in source units.
    """

    user: np.ndarray
    item: np.ndarray
    ts: np.ndarray
    user_keys: np.ndarray
    item_keys: np.ndarray

    def __len__(self) -> int:
        return len(self.user)

    def fingerprint(self) -> str:
        return fingerprint(self.user, self.item, self.ts)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, **{k: getattr(self, k) for k in ("user", "item", "ts", "user_keys", "item_keys")}
        )

    @classmethod
    def load(cls, path: Path, keys: bool = True) -> Log:
        """Load a saved log; runs skip the string keys, which only traceability needs."""
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in ("user", "item", "ts")}
            for k in ("user_keys", "item_keys"):
                arrays[k] = z[k] if keys else np.empty(0, dtype=str)
        return cls(**arrays)


def build_log(df: pd.DataFrame) -> Log:
    """Stable sort by (user, ts, row), then keep the earliest of repeated (user, item) pairs."""
    df = df.sort_values(["user", "ts", "row"], kind="stable")
    df = df.drop_duplicates(["user", "item"], keep="first")
    user_codes, user_keys = pd.factorize(df["user"], sort=True)
    item_codes, item_keys = pd.factorize(df["item"], sort=True)
    if (user_codes < 0).any() or (item_codes < 0).any() or (np.diff(user_codes) < 0).any():
        raise ValueError("empty user or item keys, or rows not grouped by user")
    return Log(
        user=user_codes.astype(np.int64),
        item=item_codes.astype(np.int64),
        ts=df["ts"].to_numpy(np.int64),
        user_keys=np.asarray(user_keys, dtype=str),
        item_keys=np.asarray(item_keys, dtype=str),
    )


def kcore_mask(user: np.ndarray, item: np.ndarray, k: int) -> np.ndarray:
    """Iterative k-core: drop users and items with fewer than k interactions until stable."""
    keep = np.ones(len(user), dtype=bool)
    while True:
        user_count = np.bincount(user[keep], minlength=user.max(initial=-1) + 1)
        item_count = np.bincount(item[keep], minlength=item.max(initial=-1) + 1)
        new = keep & (user_count[user] >= k) & (item_count[item] >= k)
        if new.sum() == keep.sum():
            return keep
        keep = new


def remap(codes: np.ndarray, kept: np.ndarray, start: int) -> np.ndarray:
    """Dense ids for `kept` codes in sorted order starting at `start`; others become -1."""
    table = np.full(codes.max(initial=-1) + 1 if len(codes) else 0, -1, dtype=np.int64)
    table[kept] = np.arange(start, start + len(kept))
    return table[codes]


@dataclass(frozen=True)
class Interactions:
    """User sequences in CSR form with items numbered 1..n_items (-1 marks items outside it)."""

    indptr: np.ndarray
    items: np.ndarray
    ts: np.ndarray
    n_items: int

    @property
    def n_users(self) -> int:
        return len(self.indptr) - 1

    def lengths(self) -> np.ndarray:
        return np.diff(self.indptr)

    def fingerprint(self) -> str:
        return fingerprint(self.indptr, self.items, self.ts)


def csr_from_sorted(user: np.ndarray, items: np.ndarray, ts: np.ndarray, n_items: int):
    """CSR for rows already sorted by user; users are renumbered densely in order."""
    _, counts = np.unique(user, return_counts=True)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return Interactions(indptr, items.astype(np.int64), ts.astype(np.int64), int(n_items))


def leave_one_out_interactions(log: Log, k: int = 5) -> Interactions:
    """Iterative k-core on the whole log, items renumbered 1..N in key order (METHODS 2.5-2.6)."""
    keep = kcore_mask(log.user, log.item, k)
    kept_items = np.unique(log.item[keep])
    items = remap(log.item[keep], kept_items, start=1)
    return csr_from_sorted(log.user[keep], items, log.ts[keep], len(kept_items))
