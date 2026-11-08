"""Download raw files with resume and retry, and verify or record their SHA-256."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from seqrec.data.registry import DatasetSpec

CHUNK = 1 << 20
REPORT_EVERY = 100 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, part: Path, log) -> None:
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as err:
        if err.code == 416 and start:  # the partial file is already complete
            return
        raise
    with response:
        resumed = start and getattr(response, "status", 200) == 206
        written = start if resumed else 0
        with part.open("ab" if resumed else "wb") as f:
            next_report = written + REPORT_EVERY
            while block := response.read(CHUNK):
                f.write(block)
                written += len(block)
                if written >= next_report:
                    log(f"[data] {part.name}: {written >> 20} MiB")
                    next_report += REPORT_EVERY


def _is_certificate_error(err: Exception) -> bool:
    return isinstance(getattr(err, "reason", err), ssl.SSLCertVerificationError)


def _fetch_with_curl(url: str, part: Path) -> None:
    """Some servers omit intermediate certificates; curl (shipped with Windows 10 and later)
    completes the chain from the system store where Python's ssl module cannot."""
    curl = shutil.which("curl")
    if curl is None:
        raise FileNotFoundError("curl not found")
    subprocess.run(
        [curl, "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(part), url], check=True
    )


def download(spec: DatasetSpec, raw_dir: Path, log=print, attempts: int = 4) -> dict:
    """Fetch `spec` into raw_dir/<name>/ unless a verified copy is already there.

    A file placed there by hand (for example when a proxy blocks downloads) is adopted.
    """
    target_dir = raw_dir / spec.name
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / spec.filename
    meta_path = target_dir / "source.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["bytes"] == path.stat().st_size:
            return meta

    if not path.exists():
        part = path.with_name(path.name + ".part")
        log(f"[data] downloading {spec.url}")
        for attempt in range(1, attempts + 1):
            try:
                _fetch(spec.url, part, log)
                break
            except (urllib.error.URLError, OSError) as err:
                if _is_certificate_error(err):
                    log("[data] certificate chain incomplete for Python; retrying with curl")
                    try:
                        _fetch_with_curl(spec.url, part)
                        break
                    except (OSError, subprocess.CalledProcessError) as curl_err:
                        raise RuntimeError(
                            f"Download {spec.url} by hand, for example\n"
                            f'  curl.exe -L -o "{path}" {spec.url}\n'
                            f"then rerun: uv run seqrec data prepare --datasets {spec.name}"
                        ) from curl_err
                if attempt == attempts:
                    raise
                log(f"[data] attempt {attempt} failed ({err}); retrying")
                time.sleep(2**attempt)
        part.replace(path)

    digest = sha256_file(path)
    if spec.sha256 and digest != spec.sha256:
        bad = path.with_name(path.name + ".bad")
        path.replace(bad)
        raise ValueError(f"{spec.filename}: SHA-256 {digest} does not match the pinned value")
    meta = {
        "dataset": spec.name,
        "url": spec.url,
        "file": spec.filename,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "recorded_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta
