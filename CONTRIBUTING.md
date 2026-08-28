# Contributing

Thanks for your interest. This is a research/education codebase; contributions
that improve rigor, reproducibility or clarity are very welcome.

## Setup

```bash
make setup          # venv + editable install + pre-commit hook
source .venv/bin/activate
make test           # 48 tests, ~15 s (synthetic data only)
```

On macOS, LightGBM needs `brew install libomp` (the `Makefile` wires
`DYLD_LIBRARY_PATH` for you).

## Workflow

1. Branch from `main`.
2. Keep changes focused. New behaviour needs a test in `tests/`.
3. `make format` (ruff) before committing; `pre-commit` enforces it.
4. `make test` must pass. The end-to-end test (`-m slow`) trains a 1-epoch
   model and exercises the API — keep it under a minute.
5. Open a PR. CI runs lint + tests on Python 3.10–3.12 and a train-and-serve
   smoke job.

## Conventions

* **No data leakage.** Any preprocessing that learns parameters must `fit` on
  the training split only, and there must be a test proving it.
* **Causality.** Sequence models and features must not see the future; add a
  perturbation test if you touch them.
* Public functions get a docstring that says *what* and *why*, with a
  literature pointer where one exists (`docs/references.bib`).
* Metrics that aggregate over hours must support / document patient clustering.

## What to work on

* GRU-D / SAITS-style learned imputation as an alternative preprocessing head.
* Proper adaptive conformal (ACI) for the streaming setting.
* A real MIMIC-IV / eICU loader behind the same `PatientRecord` interface.
* Group-DRO or post-hoc equalized-odds adjustment in `sepsis.audit`.
