"""The registered families against the table of METHODS 7.4, typed in here by hand."""

from collections import Counter

from seqrec.analysis import families
from seqrec.analysis.families import A1, A2, A3, A4, A5


def test_family_sizes_are_those_of_the_methods_table():
    sizes = Counter(c.family for c in families.registered())
    assert sizes == {"F1": 24, "F1-NI": 2, "F1-ME": 12, "F5": 8, "F2": 8, "F6": 12, "F4": 6}
    assert len({c.name for c in families.registered()}) == sum(sizes.values())


def on(family, dataset):
    return {
        (c.kind, c.a, c.b, c.shuffled_b) for c in families.family(family) if c.dataset == dataset
    }


def test_movielens_1m_holds_every_family_as_registered():
    assert {frozenset((a, b)) for _, a, b, _ in on("F1", "ml-1m")} == {
        frozenset(p) for p in [(A1, A2), (A1, A4), (A1, A5), (A2, A4), (A2, A5), (A4, A5)]
    }
    assert {k for k, *_ in on("F1", "ml-1m")} == {"directional"}
    assert on("F1-NI", "ml-1m") == {("non-inferiority", A2, A5, False)}
    assert on("F1-ME", "ml-1m") == {("minimum effect", a, A1, False) for a in (A2, A4, A5)}
    assert on("F5", "ml-1m") == {
        ("equivalence", a, b, False) for a, b in [(A5, A1), (A4, A5), (A2, A5)]
    }
    assert on("F2", "ml-1m") == {("equivalence", A3, A2, False), ("equivalence", A4, A2, False)}
    assert on("F6", "ml-1m") == {("time to target", A2, b, False) for b in (A1, A3, A4)}
    assert on("F4", "ml-1m") == {("minimum effect", A2, A2, True), ("minimum effect", A5, A5, True)}


def test_families_that_do_not_cover_every_dataset():
    """H1a on the MovieLens datasets; the BERT4Rec shuffle control there too; published signs
    as in the table of METHODS 7.3 (none on Video_Games, none for G2 on MovieLens-20M)."""
    assert {c.dataset for c in families.family("F1-NI")} == {"ml-1m", "ml-20m"}
    assert on("F4", "steam") == on("F4", "video-games") == {("minimum effect", A2, A2, True)}
    assert on("F5", "video-games") == set()
    assert on("F5", "ml-20m") == {("equivalence", A5, A1, False), ("equivalence", A2, A5, False)}
    assert on("F5", "steam") == on("F5", "ml-1m")
    signs = {gap: s for gap, (_, s) in families.GAPS.items()}
    assert signs == {
        "G1": {"ml-1m": 1, "steam": 1, "ml-20m": 1},
        "G2": {"ml-1m": 1, "steam": -1},
        "G3": {"ml-1m": 1, "steam": -1, "ml-20m": 1},
    }
    assert {gap: pair for gap, (pair, _) in families.GAPS.items()} == {
        "G1": (A5, A1), "G2": (A4, A5), "G3": (A2, A5),
    }  # fmt: skip


def test_every_gap_and_minimum_effect_pair_is_an_f1_pair_in_the_same_orientation():
    """The classification of a published gap reads the F1 test of the same pair, so its sign
    must mean the same thing in both."""
    f1 = {(c.dataset, c.a, c.b) for c in families.family("F1")}
    for other in ("F5", "F1-ME", "F1-NI"):
        assert {(c.dataset, c.a, c.b) for c in families.family(other)} <= f1
