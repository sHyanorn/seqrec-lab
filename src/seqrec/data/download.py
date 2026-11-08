"""Download raw files with resume support and record their SHA-256."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from seqrec.data.registry import DatasetSpec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(spec: DatasetSpec, raw_dir: Path, log=print) -> dict:
    """Fetch `spec` into raw_dir/<name>/ unless a verified copy is already there."""
    target_dir = raw_dir / spec.name
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / spec.filename
    meta_path = target_dir / "source.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["bytes"] == path.stat().st_size:
            return meta

    part = path.with_name(path.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    log(f"[data] downloading {spec.url}" + (f" (resuming at {start} bytes)" if start else ""))
    with urllib.request.urlopen(urllib.request.Request(spec.url, headers=headers), timeout=60) as r:
        mode = "ab" if start and r.status == 206 else "wb"
        with part.open(mode) as f:
            shutil.copyfileobj(r, f, 1 << 20)
    part.replace(path)

    meta = {
        "dataset": spec.name,
        "url": spec.url,
        "file": spec.filename,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "downloaded_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta
