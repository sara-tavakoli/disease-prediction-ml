"""``sepsis`` command-line entry point.

    sepsis train      --config configs/model_transformer.yaml [a.b=c ...]
    sepsis synth      --n 4000 --prevalence 0.08 --out data/raw
    sepsis download   --limit 400 [--full]
    sepsis serve      --run-dir artifacts/transformer --port 8000

``train`` runs the full pipeline in :func:`sepsis.training.experiment.run_experiment`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sepsis.config import ExperimentConfig
from sepsis.utils.logging import get_logger

log = get_logger("cli")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        nargs="*",
        default=["configs/base.yaml"],
        help="one or more YAML files, merged left-to-right",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="a.b=c",
        help="dotted config overrides",
    )
    p.add_argument("--output-dir", default=None)


def cmd_train(args: argparse.Namespace) -> int:
    from sepsis.training.experiment import run_experiment

    cfg = ExperimentConfig.load(*args.config, overrides=args.overrides)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    log.info("resolved config:\n%s", json.dumps(cfg.to_dict(), indent=2))
    results = run_experiment(cfg)
    disc = results["discrimination"]
    print(
        json.dumps(
            {
                "output_dir": cfg.output_dir,
                "auroc": disc["auroc"]["point"],
                "auprc": disc["auprc"]["point"],
                "utility": results["utility"]["test_utility"]["normalized"],
                "ece_calibrated": results["calibration"]["ece_calibrated"]["ece"],
            },
            indent=2,
        )
    )
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from sepsis.data.synthetic import generate_cohort, write_cohort

    recs = generate_cohort(args.n, args.prevalence, args.seed)
    write_cohort(recs, args.out)
    n_sep = sum(r.is_septic for r in recs)
    log.info(
        "wrote %d synthetic stays (%d septic) to %s/training_setSYN", len(recs), n_sep, args.out
    )
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from sepsis.data.download import download_full, download_sample

    if args.full:
        download_full(args.root)
    else:
        n = download_sample(args.root, args.limit)
        log.info("downloaded %d PSV files to %s", n, args.root)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    os.environ["SEPSIS_RUN_DIR"] = args.run_dir
    if not Path(args.run_dir, "config.json").exists():
        log.error("no trained bundle at %s (run `sepsis train` first)", args.run_dir)
        return 2
    uvicorn.run("sepsis.serve.api:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sepsis", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="run the full experiment pipeline")
    _add_common(t)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("synth", help="write a synthetic cohort to disk")
    s.add_argument("--n", type=int, default=4000)
    s.add_argument("--prevalence", type=float, default=0.08)
    s.add_argument("--seed", type=int, default=20190804)
    s.add_argument("--out", default="data/raw")
    s.set_defaults(func=cmd_synth)

    d = sub.add_parser("download", help="fetch PhysioNet/CinC 2019 data")
    d.add_argument("--root", default="data/raw")
    d.add_argument("--limit", type=int, default=400)
    d.add_argument("--full", action="store_true")
    d.set_defaults(func=cmd_download)

    v = sub.add_parser("serve", help="start the FastAPI inference service")
    v.add_argument("--run-dir", default="artifacts/run")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
