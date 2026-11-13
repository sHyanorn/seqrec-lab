"""Inject known silent bugs; the listed check must fail (METHODS 11.3, docs/MUTATIONS.md)."""

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import frame, random_frame
from test_losses import (
    check_full_ce_matches_naive_softmax_over_items_only,
    check_sampled_softmax_drops_negatives_equal_to_the_positive,
)
from test_sasrec import assert_causal, net

from seqrec.data import invariants as invariants_mod
from seqrec.data import preprocess as preprocess_mod
from seqrec.data import splits as splits_mod
from seqrec.data.invariants import (
    InvariantError,
    check_split,
    item_popularity,
    require_from_train,
)
from seqrec.data.preprocess import build_log
from seqrec.data.splits import Sequences, TrainSplit
from seqrec.data.store import build_splits
from seqrec.eval import ranking
from seqrec.models import batches, losses, sasrec
from seqrec.runner.config import RunConfig
from seqrec.runner.run import ARMS


@pytest.fixture
def log():
    return build_log(random_frame())


def _caught(log, match, kind="loo"):
    with pytest.raises(InvariantError, match=match):
        build_splits(log, kind)


def test_validation_item_in_training(log, monkeypatch):
    monkeypatch.setattr(splits_mod, "_loo_train_len", lambda lengths: lengths - 1)
    _caught(log, "train: no evaluation targets")


def test_test_item_in_training(log, monkeypatch):
    monkeypatch.setattr(splits_mod, "_loo_train_len", lambda lengths: lengths)
    _caught(log, "train: no evaluation targets")


def test_targets_off_by_one(log, monkeypatch):
    monkeypatch.setattr(splits_mod, "_loo_target_offsets", lambda n: (n - 3, n - 2))
    _caught(log, "train: no evaluation targets")


def test_target_inside_its_input(log, monkeypatch):
    """Each evaluation input ends with its own target; the training split is left as it is,
    so the checks on the training rows pass and those on the evaluation inputs must fail."""
    original = splits_mod.EvalSplit

    def leaky(phase, n_items, inputs, targets, target_ts, target_pos, shuffled=False):
        ends = inputs.indptr[1:]
        inputs = Sequences(
            inputs.indptr + np.arange(len(inputs.indptr)),
            np.insert(inputs.items, ends, targets),
            np.insert(inputs.ts, ends, target_ts),
            np.insert(inputs.pos, ends, target_pos),
        )
        return original(phase, n_items, inputs, targets, target_ts, target_pos, shuffled)

    monkeypatch.setattr(splits_mod, "EvalSplit", leaky)
    _caught(log, "validation: inputs precede target")


def test_evaluation_input_drops_an_item(log, monkeypatch):
    original = splits_mod.select

    def lossy(full, owner, keep, n_rows):  # evaluation inputs lose their first item
        first = np.zeros_like(keep)
        idx = np.flatnonzero(keep)
        if len(idx):
            starts = idx[np.r_[True, owner[idx][1:] != owner[idx][:-1]]]
            first[starts] = True
        return original(full, owner, keep & ~first, n_rows)

    monkeypatch.setattr(splits_mod, "select", lossy)
    _caught(log, "earlier in-vocabulary item|validation input equals training")


@pytest.mark.parametrize("part", ["train", "validation", "test"])
def test_items_detached_from_their_log_positions(log, part):
    """Two items of one sequence change places while their positions stay: every check made
    through the positions still passes."""
    splits, _ = build_splits(log, "loo")

    def swapped(seqs):
        items = seqs.items.copy()
        items[[0, 1]] = items[[1, 0]]
        return replace(seqs, items=items)

    if part == "train":
        mutated = replace(splits, train=replace(splits.train, seqs=swapped(splits.train.seqs)))
    elif part == "validation":
        inputs = swapped(splits.validation.inputs)
        mutated = replace(splits, validation=replace(splits.validation, inputs=inputs))
    else:
        test = splits.test.unseal("check")
        mutated = replace(
            splits, test=splits_mod.TestHandle(replace(test, inputs=swapped(test.inputs)))
        )
    with pytest.raises(InvariantError, match=f"{part}: items match log"):
        check_split(mutated)


def test_statistic_from_all_data(log):
    splits, _ = build_splits(log, "loo")
    full = splits.full
    everything = Sequences(full.indptr, full.items, full.ts, np.arange(len(full.items)))
    stat = item_popularity(TrainSplit(full.n_items, everything))
    with pytest.raises(InvariantError, match="not computed from this training split"):
        require_from_train(stat, splits.train)


def _tied_rows():
    rows = [(f"u{u}", f"i{(u + j) % 15}", 10 * j) for u in range(40) for j in range(10)]
    return rows + [("late", "i1", 75), ("late", "new", 90)]


def _tied_log():
    return build_log(frame(_tied_rows()))  # every timestamp is shared by many interactions


def test_temporal_cutoff_off_by_one_at_ties(monkeypatch):
    monkeypatch.setattr(splits_mod, "_gts_in_period", lambda ts, cutoff: ts < cutoff)
    _caught(_tied_log(), "gts: validation targets inside validation period", "gts")


def test_temporal_vocabulary_from_whole_log(monkeypatch):
    def whole_log_vocab(log, in_train_period, k):  # post-cutoff items enter the vocabulary
        return np.unique(log.item)

    monkeypatch.setattr(splits_mod, "_gts_vocabulary", whole_log_vocab)
    _caught(_tied_log(), "vocabulary is the 5-core of the training period", "gts")


def _single_pass(user, item, k):
    """One filter on the raw counts, as in the code of [1] (METHODS 2.5)."""
    return (np.bincount(user)[user] >= k) & (np.bincount(item)[item] >= k)


def _chain_log():
    """A dense core, and item x with five users in the training period, one of whom (p) has
    only four interactions: the 5-core drops p and then x, a single pass keeps x."""
    rows = [(f"u{u}", f"c{(u + j) % 12}", 10 * j + u % 10) for u in range(30) for j in range(8)]
    rows += [(f"u{u}", "x", 5 + u) for u in range(4)]
    rows += [("p", f"c{j}", 1 + j) for j in range(3)] + [("p", "x", 4)]
    rows += [(f"u{u}", f"c{(u + 9) % 12}", 1000 + u) for u in range(30)]  # T falls among these
    return build_log(frame(rows))


def test_temporal_vocabulary_from_a_single_pass_filter(monkeypatch):
    """The split builder and every module that could lend the check its filter get the same
    faulty one, as a defect in the shared k-core code would give them."""
    log = _chain_log()
    correct = splits_mod.global_temporal(log)
    for module in (splits_mod, preprocess_mod, invariants_mod):
        monkeypatch.setattr(module, "kcore_mask", _single_pass, raising=False)
    assert splits_mod.global_temporal(log).n_items == correct.n_items + 1  # x stays in
    _caught(log, "vocabulary is the 5-core of the training period", "gts")


def _train_rows(splits, pos):
    """The split with training rows at the log positions `pos`, items and times from the log."""
    full, seqs = splits.full, splits.train.seqs
    train = Sequences(seqs.indptr, full.items[pos], full.ts[pos], pos)
    return replace(splits, train=TrainSplit(splits.n_items, train))


def _reversed_training_rows(splits):
    seqs = splits.train.seqs
    owner = np.repeat(np.arange(len(seqs)), seqs.lengths())
    return _train_rows(splits, seqs.pos[np.lexsort((-np.arange(len(seqs.pos)), owner))])


def _training_rows_of_two_users_swapped(splits):
    seqs = splits.train.seqs
    lengths = seqs.lengths()
    a = int(np.flatnonzero(lengths == lengths[0])[1])  # the next user as long as user 0
    rows = [seqs.pos[seqs.indptr[u] : seqs.indptr[u + 1]] for u in range(len(seqs))]
    rows[0], rows[a] = rows[a], rows[0]
    return _train_rows(splits, np.concatenate(rows))


def _training_rows_shuffled_with_other_keys(splits):
    shuffled = splits_mod.shuffle_order(splits, seed=1)
    return replace(shuffled, train=splits_mod.shuffle_order(splits, seed=2).train)


@pytest.mark.parametrize(
    "mutate",
    [
        _reversed_training_rows,
        _training_rows_of_two_users_swapped,
        _training_rows_shuffled_with_other_keys,
    ],
    ids=["reversed", "two-users-swapped", "shuffled-with-other-keys"],
)
def test_training_rows_out_of_step_with_the_validation_input(log, mutate):
    """Each user's training rows keep their set of log positions and length, so only an
    order-sensitive comparison with the validation input sees the change (METHODS 3.3, 11.1)."""
    mutated = mutate(build_splits(log, "loo")[0])
    with pytest.raises(InvariantError, match="loo: validation input equals training sequence"):
        check_split(mutated)


def test_temporal_validation_target_first_not_last(monkeypatch):
    original = splits_mod._last_target

    def first_target(full, owner, candidate, phase):
        if phase != "validation":
            return original(full, owner, candidate, phase)
        firsts = np.zeros_like(candidate)
        rows = np.flatnonzero(candidate)
        if len(rows):
            firsts[rows[np.r_[True, owner[rows][1:] != owner[rows][:-1]]]] = True
        return original(full, owner, firsts, phase)

    monkeypatch.setattr(splits_mod, "_last_target", first_target)
    dense = [(f"u{u}", f"i{(u + j) % 25}", 40 * j + u) for u in range(40) for j in range(20)]
    _caught(build_log(frame(dense)), "last item up to the cutoff", "gts")


TARGET_FIELDS = ("targets", "target_ts", "target_pos")


def _half_the_users(original, phase):
    def build(full, owner, candidate, p):  # every other user loses its candidate rows
        return original(full, owner, candidate & (owner % 2 == 0) if p == phase else candidate, p)

    return build


def _a_user_twice(original, phase):
    def build(full, owner, candidate, p):  # the first user's input and target appear again
        split = original(full, owner, candidate, p)
        if p != phase:
            return split
        q, n = split.inputs, split.inputs.indptr[1]
        inputs = Sequences(
            np.append(q.indptr, q.indptr[-1] + n),
            *(np.concatenate([a, a[:n]]) for a in (q.items, q.ts, q.pos)),
        )
        again = {k: np.append(getattr(split, k), getattr(split, k)[0]) for k in TARGET_FIELDS}
        return replace(split, inputs=inputs, **again)

    return build


@pytest.mark.parametrize("phase", ["validation", "test"])
@pytest.mark.parametrize("mutation", [_half_the_users, _a_user_twice], ids=["half", "twice"])
def test_temporal_evaluation_users_differ_from_the_registered_ones(monkeypatch, mutation, phase):
    """METHODS 3.2 and 11.1: the users of each temporal evaluation set follow from the source
    log. Each kept user's target and input stay correct, so only that comparison sees it."""
    monkeypatch.setattr(splits_mod, "_last_target", mutation(splits_mod._last_target, phase))
    dense = [(f"u{u}", f"i{(u + j) % 25}", 40 * j + u) for u in range(40) for j in range(20)]
    _caught(build_log(frame(dense)), "validation target is the last item up to the cutoff", "gts")


def test_temporal_split_times_one_unit_early_in_the_training_period(monkeypatch):
    """METHODS 11.1, "every training interaction has t ≤ T_val": the split's own times put an
    interaction of the validation period into training, and every check that reads them
    agrees with them."""
    log = build_log(
        frame([(f"u{u}", f"i{(u + j) % 25}", 40 * j + u) for u in range(40) for j in range(20)])
    )
    cutoff = splits_mod.time_quantile(log.ts, 0.9)
    original = splits_mod.Interactions

    def early(indptr, items, ts, n_items):
        return original(indptr, items, ts - (ts <= cutoff), n_items)

    monkeypatch.setattr(splits_mod, "Interactions", early)
    mutated = splits_mod.global_temporal(log)
    assert (log.ts[mutated.train.seqs.pos] > mutated.meta["validation_cutoff"]).any()
    _caught(log, "vocabulary is the 5-core of the training period", "gts")


def test_temporal_split_rows_moved_between_two_training_users(monkeypatch):
    """No two neighbouring users share an item, and each user's activity ends before the next
    one's begins, so the user boundary can move by one row without breaking any check made on
    a user's own rows. Users 4 and 5 are in neither evaluation set."""
    rows = [
        (f"u{u:02d}", f"i{(u % 2) * 25 + (u // 2 + j) % 25:02d}", 20 * u + j)
        for u in range(40)
        for j in range(10)
    ]
    log = build_log(frame(rows))
    original = splits_mod.Interactions

    def moved(indptr, items, ts, n_items):
        indptr = indptr.copy()
        indptr[5] += 1  # user 4's last row becomes user 5's first
        return original(indptr, items, ts, n_items)

    monkeypatch.setattr(splits_mod, "Interactions", moved)
    _caught(log, "vocabulary is the 5-core of the training period", "gts")


def _training_log():
    """Dense users, and user "solo" with one vocabulary interaction and one outside the
    vocabulary, both before T_val and nothing later: no evaluation set holds solo."""
    rows = [(f"u{u}", f"i{(u + j) % 25}", 40 * j + u) for u in range(40) for j in range(20)]
    return build_log(frame(rows + [("solo", "i1", 3), ("solo", "unknown", 4)]))


def _half_the_training_users(splits, log):
    seqs = splits.train.seqs
    keep = np.repeat(np.arange(len(seqs)) % 2 == 0, seqs.lengths())
    return _train_rows_from_log(splits, log, seqs.pos[keep])


def _a_user_with_one_vocabulary_item(splits, log):
    solo = np.flatnonzero((log.user_keys[log.user] == "solo") & (splits.full.items > 0))
    assert len(solo) == 1
    return _train_rows_from_log(splits, log, np.sort(np.append(splits.train.seqs.pos, solo)))


def _train_rows_from_log(splits, log, pos):
    """Training sequences at the log rows `pos`, one per user, in the log's order."""
    _, lengths = np.unique(log.user[pos], return_counts=True)
    full = splits.full
    seqs = Sequences(np.r_[0, np.cumsum(lengths)], full.items[pos], full.ts[pos], pos)
    return replace(splits, train=TrainSplit(splits.n_items, seqs))


@pytest.mark.parametrize(
    "mutate",
    [_half_the_training_users, _a_user_with_one_vocabulary_item],
    ids=["half-the-users", "one-vocabulary-item"],
)
def test_temporal_training_users_differ_from_the_registered_ones(mutate):
    """METHODS 3.2: training sequences are each user's vocabulary interactions with
    t ≤ T_val, for users with at least two of them. Every row kept or added is a training row
    of its user before T_val, so only a comparison of the users with the log sees it."""
    log = _training_log()
    splits, _ = build_splits(log, "gts")
    with pytest.raises(InvariantError, match="validation target is the last item up to the cutoff"):
        check_split(mutate(splits, log), log, 5)


def _validation(log):
    splits, _ = build_splits(log, "loo")
    return splits.validation


def _uniform_scorer(n_items):
    weights = torch.linspace(0, 1, n_items + 1)

    def scorer(inputs):
        return torch.ones(len(inputs), 1) @ weights[None, :]  # a matmul: autocast changes dtype

    return scorer


def test_history_masking_removed(log, monkeypatch):
    split = _validation(log)
    monkeypatch.setattr(ranking, "mask_history", lambda scores, rows, items: None)
    with pytest.raises(InvariantError, match="candidate count"):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=50)


def test_history_masking_limited_to_truncated_input(log, monkeypatch):
    split = _validation(log)

    def truncated(ds, a, b):
        rows, items = ranking.model_inputs(ds, a, b, 2).nonzero(as_tuple=True)
        return rows, ranking.model_inputs(ds, a, b, 2)[rows, items]

    monkeypatch.setattr(ranking, "history_entries", truncated)
    with pytest.raises(InvariantError, match="candidate count"):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=2)


def test_history_masking_applied_to_target(log, monkeypatch):
    split = _validation(log)
    original = ranking.history_entries

    def with_target(ds, a, b):
        rows, items = original(ds, a, b)
        extra = torch.arange(b - a)
        return torch.cat([rows, extra]), torch.cat([items, ds.targets[a:b]])

    monkeypatch.setattr(ranking, "history_entries", with_target)
    with pytest.raises(InvariantError, match="target score must stay finite"):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=50)


def test_scores_in_bf16(log, monkeypatch):
    split = _validation(log)
    monkeypatch.setattr(
        ranking, "eval_autocast", lambda device: torch.autocast("cpu", dtype=torch.bfloat16)
    )
    with pytest.raises(InvariantError, match="fp32"):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=50)


# Added in M2, with the models.


def _sasrec(arm, train):
    torch.manual_seed(0)
    cfg = RunConfig("b", "ml-1m", "loo", arm, 1, "tune", params={"dim": 16}, max_len=8)
    return ARMS[arm](cfg, train, torch.device("cpu"))


def test_sasrec_labels_not_shifted(log, monkeypatch):
    splits, _ = build_splits(log, "loo")
    original = batches.sasrec_batch

    def unshifted(data, rows, length):
        b = original(data, rows, length)
        return batches.Batch(b.rows, b.inputs, b.inputs, b.window)

    monkeypatch.setattr(batches, "sasrec_batch", unshifted)
    with pytest.raises(InvariantError, match="shifted by one"):
        _sasrec("sasrec-ce", splits.train).train_epoch(splits.train, 0)


def test_bce_negatives_exclude_the_full_history(log, monkeypatch):
    splits, _ = build_splits(log, "loo")
    full = splits.full
    everything = TrainSplit(
        full.n_items, Sequences(full.indptr, full.items, full.ts, np.arange(len(full.items)))
    )  # includes the validation and test items
    original = batches.HistoryExcludingSampler
    monkeypatch.setattr(
        batches, "HistoryExcludingSampler", lambda train, device: original(everything, device)
    )
    model = _sasrec("sasrec-bce", splits.train)
    with pytest.raises(InvariantError, match="not computed from this training split"):
        model.train_epoch(splits.train, 0)


def test_sasrec_causal_mask_removed(monkeypatch):
    def bidirectional(real):
        n = real.shape[1]
        return (real[:, None, :] | torch.eye(n, dtype=torch.bool))[:, None].expand(-1, 1, n, n)

    monkeypatch.setattr(sasrec, "attention_mask", bidirectional)
    with pytest.raises(AssertionError):
        assert_causal(net(), torch.randint(1, 31, (4, 8)), 3)


def test_softmax_includes_the_padding_id(monkeypatch):
    def with_padding(h, w, y, bias=None, n_items=None, chunk=0):
        logits = h @ w[: n_items + 1].T + (0 if bias is None else bias[: n_items + 1])
        return F.cross_entropy(logits, y)

    monkeypatch.setattr(losses, "full_ce", with_padding)
    with pytest.raises(AssertionError):
        check_full_ce_matches_naive_softmax_over_items_only()


def test_sampled_softmax_keeps_collisions(monkeypatch):
    monkeypatch.setattr(losses, "mask_collisions", lambda neg_logits, neg, y: neg_logits)
    with pytest.raises(AssertionError):
        check_sampled_softmax_drops_negatives_equal_to_the_positive()
