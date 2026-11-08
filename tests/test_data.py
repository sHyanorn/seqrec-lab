import csv
import gzip
import zipfile

import numpy as np
import pytest
from conftest import frame, random_frame

from seqrec.data import prepare as prepare_mod
from seqrec.data import readers
from seqrec.data.invariants import InvariantError, item_popularity, require_from_train
from seqrec.data.preprocess import Log, build_log, kcore_mask, leave_one_out_interactions
from seqrec.data.splits import shuffle_eval_inputs, shuffle_order
from seqrec.data.store import build_splits


def test_build_log_sorts_stably_and_keeps_earliest_repeat():
    log = build_log(frame([("b", "x", 5), ("a", "y", 3), ("a", "x", 3), ("a", "y", 9)]))
    # user a: tie at ts 3 keeps file order (y before x); the repeat of y at ts 9 is dropped
    assert list(log.user_keys[log.user]) == ["a", "a", "b"]
    assert list(log.item_keys[log.item]) == ["y", "x", "x"]
    assert list(log.ts) == [3, 3, 5]


def test_kcore_is_iterative():
    # item z has 2 users; removing it drops user c below 2, which drops item w below 2
    user = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    item = np.array([0, 1, 0, 1, 2, 3, 3, 2])
    keep = kcore_mask(user, item, 2)
    assert keep.all()
    keep = kcore_mask(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 0, 1, 1]), 2)
    assert list(keep) == [True, True, True, True, False]


def test_leave_one_out_passes_invariants(toy_frame):
    splits, checks = build_splits(build_log(toy_frame), "loo")
    assert "loo: test input equals training sequence plus validation item" in checks
    full = splits.full
    assert len(splits.validation) == len(splits.test.unseal("check")) == full.n_users
    assert splits.train.seqs.lengths().sum() == len(full.items) - 2 * full.n_users


def test_global_temporal_split_handles_ties_and_new_users():
    rows = []
    for u in range(30):
        rows += [(f"u{u}", f"i{(u + j) % 12}", 10 * j) for j in range(8)]
    rows += [("late", "i1", 75), ("late", "i2", 80), ("late", "new-item", 90)]
    log = build_log(frame(rows))
    splits, checks = build_splits(log, "gts")
    cutoff = splits.meta["cutoff"]
    test = splits.test.unseal("check")
    assert (test.target_ts > cutoff).all()
    assert "gts: test targets after cutoff" in checks
    late = int(np.flatnonzero(log.user_keys == "late")[0])
    assert splits.full.indptr[late + 1] - 1 in test.target_pos  # user first seen after the cutoff
    assert splits.meta["test_oov_target_share"] > 0  # "new-item" is outside the vocabulary


def test_shuffled_splits_keep_contents_and_pass_invariants(toy_frame):
    splits, _ = build_splits(build_log(toy_frame), "loo")
    mixed = shuffle_order(splits, seed=3)
    assert np.array_equal(mixed.train.seqs.items, mixed.validation.inputs.items)
    assert not np.array_equal(mixed.train.seqs.items, splits.train.seqs.items)
    from seqrec.data.invariants import check_split

    assert check_split(mixed)
    test = splits.test.unseal("check")
    only_test = shuffle_eval_inputs(test, 3, len(splits.full.items))
    assert np.array_equal(np.sort(only_test.inputs.pos), np.sort(test.inputs.pos))


def test_popularity_only_from_training_split(toy_frame):
    splits, _ = build_splits(build_log(toy_frame), "loo")
    stat = item_popularity(splits.train)
    require_from_train(stat, splits.train)
    with pytest.raises(TypeError):
        item_popularity(splits.full)
    other, _ = build_splits(build_log(random_frame(seed=1)), "loo")
    with pytest.raises(InvariantError):
        require_from_train(stat, other.train)


def test_log_roundtrip(tmp_path, toy_frame):
    log = build_log(toy_frame)
    log.save(tmp_path / "log.npz")
    again = Log.load(tmp_path / "log.npz")
    assert again.fingerprint() == log.fingerprint()
    assert (
        leave_one_out_interactions(again).fingerprint()
        == leave_one_out_interactions(log).fingerprint()
    )


def _write_raw_files(root):
    rows = [(u, (u + j) % 10, 1_000_000 + 3600 * (u + j)) for u in range(1, 13) for j in range(8)]
    ml1m = root / "ml-1m" / "ml-1m.zip"
    ml1m.parent.mkdir(parents=True)
    with zipfile.ZipFile(ml1m, "w") as z:
        z.writestr("ml-1m/ratings.dat", "".join(f"{u}::{i}::4::{t}\n" for u, i, t in rows))
    ml20m = root / "ml-20m" / "ml-20m.zip"
    ml20m.parent.mkdir(parents=True)
    with zipfile.ZipFile(ml20m, "w") as z:
        body = "userId,movieId,rating,timestamp\n" + "".join(
            f"{u},{i},4.0,{t}\n" for u, i, t in rows
        )
        z.writestr("ml-20m/ratings.csv", body)
    steam = root / "steam" / "steam_reviews.json.gz"
    steam.parent.mkdir(parents=True)
    with gzip.open(steam, "wt", encoding="utf-8") as f:
        for u, i, t in rows:
            day = f"2017-01-{1 + (t // 3600) % 28:02d}"
            f.write(f"{{u'username': u'user {u}', u'product_id': u'{i}', u'date': u'{day}'}}\n")
        f.write("{u'username': u'no date', u'product_id': u'1'}\n")
    amazon = root / "video-games" / "Video_Games.csv.gz"
    amazon.parent.mkdir(parents=True)
    with gzip.open(amazon, "wt", encoding="utf-8") as f:
        f.write("user_id,parent_asin,rating,timestamp\n")
        f.writelines(f"U{u},B{i},5.0,{t * 1000}\n" for u, i, t in rows)


def test_readers_and_prepare_end_to_end(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    _write_raw_files(raw)
    df, info = readers.read_steam(raw / "steam" / "steam_reviews.json.gz", log=lambda _: None)
    assert info["rows_skipped"] == 1 and len(df) == 96

    def offline(spec, raw_dir, log):
        path = raw_dir / spec.name / spec.filename
        return {"dataset": spec.name, "url": spec.url, "file": spec.filename,
                "bytes": path.stat().st_size, "sha256": "test"}  # fmt: skip

    monkeypatch.setattr(prepare_mod, "download", offline)
    results = tmp_path / "results"
    prepare_mod.prepare(list(readers.READERS), tmp_path / "data", results, log=lambda _: None)
    with (results / "stats.csv").open() as f:
        stats = list(csv.DictReader(f))
    assert {r["dataset"] for r in stats} == set(readers.READERS)
    assert {r["split"] for r in stats} == {"log", "loo", "gts"}
    with (results / "published_comparison.csv").open() as f:
        assert list(csv.DictReader(f))


def test_prepare_twice_with_different_datasets(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    _write_raw_files(raw)

    def offline(spec, raw_dir, log):
        return {"dataset": spec.name, "url": spec.url, "file": spec.filename, "bytes": 1,
                "sha256": "test"}  # fmt: skip

    monkeypatch.setattr(prepare_mod, "download", offline)
    results = tmp_path / "results"
    quiet = {"log": lambda _: None}
    prepare_mod.prepare(["ml-1m", "steam"], tmp_path / "data", results, **quiet)
    prepare_mod.prepare(["video-games"], tmp_path / "data", results, **quiet)
    with (results / "stats.csv").open() as f:
        assert {r["dataset"] for r in csv.DictReader(f)} == {"ml-1m", "steam", "video-games"}
    with (results / "published_comparison.csv").open() as f:
        assert {r["dataset"] for r in csv.DictReader(f)} >= {"ml-1m", "video-games"}


def test_download_adopts_placed_files_and_checks_pins(tmp_path):
    from dataclasses import replace

    from seqrec.data.download import download, sha256_file
    from seqrec.data.registry import DATASETS

    source = tmp_path / "remote.bin"
    source.write_bytes(b"payload" * 1000)
    spec = replace(DATASETS["ml-1m"], url=source.as_uri(), sha256=None)
    meta = download(spec, tmp_path / "raw", log=lambda _: None)
    assert meta["sha256"] == sha256_file(source)

    placed = tmp_path / "raw2" / spec.name / spec.filename
    placed.parent.mkdir(parents=True)
    placed.write_bytes(b"by hand")
    assert download(spec, tmp_path / "raw2", log=lambda _: None)["bytes"] == 7

    pinned = replace(spec, sha256="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        download(pinned, tmp_path / "raw3", log=lambda _: None)


def test_certificate_errors_fall_back_to_curl_then_to_instructions(tmp_path, monkeypatch):
    import ssl
    import subprocess
    import urllib.error
    from dataclasses import replace

    from seqrec.data import download as dl
    from seqrec.data.registry import DATASETS

    spec = replace(DATASETS["ml-1m"], sha256=None)

    def broken_chain(url, part, log):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("unable to get local issuer"))

    monkeypatch.setattr(dl, "_fetch", broken_chain)
    monkeypatch.setattr(dl, "_fetch_with_curl", lambda url, part: part.write_bytes(b"zip"))
    assert dl.download(spec, tmp_path / "a", log=lambda _: None)["bytes"] == 3

    def curl_fails(url, part):
        raise subprocess.CalledProcessError(60, "curl")

    monkeypatch.setattr(dl, "_fetch_with_curl", curl_fails)
    with pytest.raises(RuntimeError, match="by hand"):
        dl.download(spec, tmp_path / "b", log=lambda _: None)
