# Data

| path | tracked | contents |
| --- | --- | --- |
| `data/sample/` | **yes** | 240 real PhysioNet/CinC 2019 stays (120 from `training_setA`, 120 from `training_setB`). Enough to exercise the real-data code path and CI; **far too small for meaningful metrics.** |
| `data/raw/` | no (`.gitkeep` only) | full corpus goes here via `sepsis download --full` (~2.6 GB, ~40 000 stays), or a synthetic cohort via `sepsis synth` |
| `data/processed/` | no | reserved for cached tensors |

## Getting the full dataset

```bash
sepsis download --full          # wget mirror of the 1.0.0 training tree
# or a bigger HTTPS sample:
sepsis download --limit 2000
```

The data are distributed by PhysioNet under the Credentialed Health Data
License 1.5.0. Cite Reyna et al., *Critical Care Medicine* 2020 (see
`../CITATION.cff`).

## File format

Pipe-separated, one file per ICU stay, one row per hour: 34 physiological
channels + 6 context columns (`Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS`)
+ `SepsisLabel`. Missing measurements are `NaN`. `SepsisLabel` is already
shifted +6 h for septic patients.
