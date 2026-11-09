"""Synthetic data with a planted next-item rule (METHODS 11.2, D027)."""

import numpy as np
from conftest import frame


def planted_frame(n_items=200, users=1500, length=(30, 50), p=0.2, seed=0):
    """Next item = succ(current), with succ one cycle over all items. Training transitions
    follow the rule with probability p, otherwise a random unseen item; the last two
    transitions (validation and test targets) always follow it. Users whose rule-given
    target was already seen are dropped. Items within a user are distinct."""
    rng = np.random.default_rng(seed)
    cycle = rng.permutation(n_items)
    succ = np.empty(n_items, dtype=np.int64)
    succ[cycle] = np.roll(cycle, -1)
    rows = []
    for u in range(users):
        n = int(rng.integers(length[0], length[1] + 1))
        seq = [int(rng.integers(n_items))]
        seen = set(seq)
        while len(seq) < n:
            forced = len(seq) >= n - 2
            nxt = int(succ[seq[-1]]) if forced or rng.random() < p else -1
            if nxt in seen and forced:
                break
            while nxt < 0 or nxt in seen:
                nxt = int(rng.integers(n_items))
            seq.append(nxt)
            seen.add(nxt)
        if len(seq) == n:
            rows += [(f"u{u:04d}", f"i{item:03d}", t) for t, item in enumerate(seq)]
    return frame(rows)
