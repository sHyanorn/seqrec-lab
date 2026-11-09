"""Static guard for METHODS 11.6: only approved code opens the sealed test split."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "seqrec"
ALLOWED = {
    ("eval/test_access.py", "read"),
    ("data/invariants.py", "check"),
    ("data/splits.py", "transform"),
}


def test_only_approved_modules_unseal_test_data():
    found = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(SRC).as_posix()
        assert "_TestHandle__split" not in text, rel
        for purpose in re.findall(r"\.unseal\(\s*\"(\w+)\"\s*\)", text):
            found.add((rel, purpose))
        assert not re.search(r"\.unseal\(\s*[^\"\s)]", text), f"{rel}: unseal with a variable"
    assert found == ALLOWED


# `SplitSet.full` holds every complete sequence, test targets included, so reading it opens
# test data for structure (METHODS 11.6). Each allowed use is counted; a new one fails here.
FULL_USES = {
    "data/invariants.py": 2,  # invariant checks
    "data/stats.py": 1,  # dataset statistics
    "data/store.py": 1,  # preprocessing checks
    "runner/run.py": 2,  # data fingerprint in the manifest; item count for the test shuffle
}


def test_only_approved_code_reads_the_full_sequences():
    found = {}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel == "data/splits.py":  # defines SplitSet and the shuffle transform
            continue
        uses = len(re.findall(r"\.full\b(?!\s*\()", path.read_text(encoding="utf-8")))
        if uses:
            found[rel] = uses
    assert found == FULL_USES
