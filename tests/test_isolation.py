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
