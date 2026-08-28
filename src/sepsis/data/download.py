"""Fetch the PhysioNet/CinC 2019 training data.

Two acquisition paths:

* ``--full``  -- stream the official archive with ``wget`` mirroring
  (``https://physionet.org/files/challenge-2019/1.0.0/``). ~2.6 GB uncompressed,
  ~40k stays across ``training_setA`` (Beth Israel) and ``training_setB``
  (Emory). This is what real experiments should use.
* default     -- download only the first ``--limit`` stays of each set over
  HTTPS. Enough for smoke tests and to refresh ``data/sample/``.

The data are released under the PhysioNet Credentialed Health Data License; see
``docs/methodology.md`` for the citation requirement.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests

from sepsis.utils.logging import get_logger

log = get_logger("data.download")

BASE = "https://physionet.org/files/challenge-2019/1.0.0"
# training_setA (Beth Israel) is p000001..; training_setB (Emory) is p100001..
SETS = {"training_setA": "setA", "training_setB": "setB"}
SET_START_IDX = {"training_setA": 1, "training_setB": 100001}


def _download_one(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except requests.RequestException as exc:  # pragma: no cover - network
            log.warning("retry %d/%d for %s (%s)", attempt, retries, url, exc)
            time.sleep(2 * attempt)
    return False


def download_sample(root: str | Path, limit: int = 400) -> int:
    """Grab up to ``limit`` stays from each training set. Returns file count."""
    root = Path(root)
    total = 0
    for set_dir in SETS:
        out = root / set_dir
        out.mkdir(parents=True, exist_ok=True)
        got = 0
        idx = SET_START_IDX[set_dir]
        misses = 0
        while got < limit and misses < 25:
            pid = f"p{idx:06d}.psv"
            idx += 1
            dest = out / pid
            if dest.exists():
                got += 1
                continue
            if _download_one(f"{BASE}/training/{set_dir}/{pid}", dest):
                got += 1
                misses = 0
            else:
                misses += 1
        log.info("%s: %d files under %s", set_dir, got, out)
        total += got
    return total


def download_full(root: str | Path) -> None:  # pragma: no cover - network heavy
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "-r",
        "-N",
        "-c",
        "-np",
        "-nH",
        "--cut-dirs=4",
        "-R",
        "index.html*",
        "-P",
        str(root),
        f"{BASE}/training/",
    ]
    log.info("running: %s", " ".join(cmd))
    if subprocess.run(cmd, check=False).returncode != 0:
        raise RuntimeError(
            "wget mirror failed. Install wget, or download the ZIP manually "
            f"from {BASE}/ and unzip into {root}/"
        )


def ensure_physionet(root: str | Path, min_files: int = 50) -> Path:
    """Return ``root`` if it already holds >= ``min_files`` PSV files, else fetch
    a sample. Used by pipelines that must not silently fall back to synthetic."""
    root = Path(root)
    n = sum(1 for _ in root.rglob("*.psv"))
    if n >= min_files:
        return root
    log.info("only %d PSV files under %s; downloading a sample", n, root)
    download_sample(root)
    return root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw")
    ap.add_argument(
        "--limit", type=int, default=400, help="stays per training set for the sample path"
    )
    ap.add_argument(
        "--full", action="store_true", help="mirror the entire 1.0.0 training tree with wget"
    )
    args = ap.parse_args(argv)

    if args.full:
        download_full(args.root)
    else:
        n = download_sample(args.root, args.limit)
        log.info("done: %d PSV files under %s", n, args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
