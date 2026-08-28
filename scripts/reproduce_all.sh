#!/usr/bin/env bash
# Train every model on the configured data source and print a leaderboard.
#
#   scripts/reproduce_all.sh [SOURCE] [N_PATIENTS] [EPOCHS]
#
# SOURCE = synthetic (default) | physionet
# With physionet, run `sepsis download --full` first for real numbers; otherwise
# a 240-stay sample is used and metrics will be weak.
set -euo pipefail

SOURCE="${1:-synthetic}"
N="${2:-5000}"
EPOCHS="${3:-25}"
MODELS=(lightgbm lstm gru tcn transformer)

if [[ "$(uname)" == "Darwin" && -d "$(brew --prefix libomp 2>/dev/null)/lib" ]]; then
  export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:${DYLD_LIBRARY_PATH:-}"
fi
export MLFLOW_ALLOW_FILE_STORE=true

common=(--set "data.source=${SOURCE}" "train.epochs=${EPOCHS}" "train.seed=20190804")
if [[ "$SOURCE" == "synthetic" ]]; then
  common+=("data.synthetic_n_patients=${N}" "data.synthetic_prevalence=0.08"
           "data.group_by_hospital=false")
else
  common+=("data.root=data/sample" "data.group_by_hospital=true")
fi

for m in "${MODELS[@]}"; do
  echo "=================  training ${m}  ================="
  sepsis train --config configs/base.yaml "configs/model_${m}.yaml" "${common[@]}"
done

python - <<'PY'
import json, pathlib, pandas as pd
rows = []
for m in ["lightgbm", "lstm", "gru", "tcn", "transformer"]:
    p = pathlib.Path("artifacts") / m / "results.json"
    if not p.exists():
        continue
    r = json.loads(p.read_text()); d = r["discrimination"]
    rows.append(dict(
        model=m,
        AUROC=round(d["auroc"]["point"], 4),
        AUPRC=round(d["auprc"]["point"], 4),
        utility=round(r["utility"]["test_utility"]["normalized"], 4),
        ECE_cal=round(r["calibration"]["ece_calibrated"]["ece"], 4),
        coverage=round(r["conformal"]["empirical_coverage"], 3),
        params=r["model_extra"].get("n_parameters", "-"),
    ))
print("\n" + pd.DataFrame(rows).set_index("model").sort_values("AUPRC", ascending=False).to_string())
PY
