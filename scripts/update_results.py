#!/usr/bin/env python
"""Rewrite the RESULTS blocks in README.md and docs/paper.md from run artifacts.

    python scripts/update_results.py [artifacts_dir]

Scans ``<artifacts_dir>/<model>/results.json`` for the five known model names
and writes a markdown leaderboard between the RESULTS:START / RESULTS:END
markers in place.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MODELS = ["lightgbm", "lstm", "gru", "tcn", "transformer"]
TARGETS = ["README.md", "docs/paper.md"]


def _row(name: str, r: dict) -> str:
    d = r["discrimination"]
    s = d["summary_at_operating_point"]
    a = d["auroc"]
    p = d["auprc"]
    return (
        f"| {name} | {a['point']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}] "
        f"| {p['point']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}] "
        f"| {r['utility']['test_utility']['normalized']:.3f} "
        f"| {s['sensitivity_at_spec85']:.3f} "
        f"| {r['calibration']['ece_uncalibrated']['ece']:.3f} → "
        f"{r['calibration']['ece_calibrated']['ece']:.3f} "
        f"| {r['conformal']['empirical_coverage']:.3f} "
        f"| {r['model_extra'].get('n_parameters', '—')} |"
    )


def build_table(art: Path) -> str:
    header = (
        "| model | AUROC (95% CI) | AUPRC (95% CI) | utility | sens@spec85 "
        "| ECE (raw → cal) | conf. cov. (α=0.1) | params |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = []
    for m in MODELS:
        f = art / m / "results.json"
        if f.exists():
            rows.append(_row(m, json.loads(f.read_text())))
    if not rows:
        return "_No run artifacts found. Run `bash scripts/reproduce_all.sh`._"
    return header + "\n" + "\n".join(rows)


def main() -> int:
    art = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
    table = build_table(art)
    pat = re.compile(
        r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.DOTALL
    )
    for t in TARGETS:
        p = Path(t)
        if not p.exists():
            continue
        new = pat.sub(rf"\1\n{table}\n\2", p.read_text())
        p.write_text(new)
        print(f"updated {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
