"""The registered comparisons and their families (METHODS 7.3, 7.4, 8).

Every comparison of a family is listed for every dataset it is registered on, whether or not
that dataset has been run: Holm's correction needs the size of the whole family. Family F3
(bf16 against strict fp32, H3c) belongs to the efficiency measurements and is not listed.
"""

from __future__ import annotations

from dataclasses import dataclass

A1, A2, A3, A4, A5 = "sasrec-bce", "sasrec-ce", "sasrec-ss", "gsasrec", "bert4rec"
DATASETS = ("ml-1m", "steam", "ml-20m", "video-games")
MOVIELENS = ("ml-1m", "ml-20m")

DIRECTIONAL, NON_INFERIORITY, MINIMUM_EFFECT, TOST, TIME = (
    "directional", "non-inferiority", "minimum effect", "equivalence", "time to target",
)  # fmt: skip

# Published gaps A - B and the sign their source reports per dataset (METHODS 7.3).
GAPS = {
    "G1": ((A5, A1), {"ml-1m": +1, "steam": +1, "ml-20m": +1}),
    "G2": ((A4, A5), {"ml-1m": +1, "steam": -1}),
    "G3": ((A2, A5), {"ml-1m": +1, "steam": -1, "ml-20m": +1}),
}
# The six pairs among A1, A2, A4 and A5, oriented as the gaps and the minimum-effect tests.
F1_PAIRS = ((A2, A1), (A4, A1), (A5, A1), (A4, A2), (A2, A5), (A4, A5))


@dataclass(frozen=True)
class Comparison:
    family: str
    dataset: str
    kind: str
    a: str
    b: str
    shuffled_b: bool = False  # b is arm `b` trained and evaluated on shuffled order (C1)

    @property
    def name(self) -> str:
        return f"{self.family} {self.dataset} {self.a} vs {'C1-' * self.shuffled_b}{self.b}"


def registered() -> list[Comparison]:
    out = [Comparison("F1", d, DIRECTIONAL, a, b) for d in DATASETS for a, b in F1_PAIRS]
    out += [Comparison("F1-NI", d, NON_INFERIORITY, A2, A5) for d in MOVIELENS]
    out += [Comparison("F1-ME", d, MINIMUM_EFFECT, a, A1) for d in DATASETS for a in (A2, A4, A5)]
    out += [Comparison("F5", d, TOST, a, b) for (a, b), signs in GAPS.values() for d in signs]
    out += [Comparison("F2", d, TOST, a, A2) for d in DATASETS for a in (A3, A4)]
    # H2a: A2's times to target are smaller than those of A1, A3 and A4.
    out += [Comparison("F6", d, TIME, A2, b) for d in DATASETS for b in (A1, A3, A4)]
    out += [Comparison("F4", d, MINIMUM_EFFECT, A2, A2, shuffled_b=True) for d in DATASETS]
    out += [Comparison("F4", d, MINIMUM_EFFECT, A5, A5, shuffled_b=True) for d in MOVIELENS]
    return out


def family(name: str) -> list[Comparison]:
    return [c for c in registered() if c.family == name]
