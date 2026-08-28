"""End-to-end experiment: data -> model -> calibration/conformal -> evaluation
-> fairness/robustness -> explainability -> serialisable bundle + report.

``run_experiment(cfg)`` is deterministic given ``cfg`` and writes everything a
reviewer needs into ``cfg.output_dir``:

    config.json  preprocess.json  model.pt|model.txt
    calibrator.joblib  conformal.joblib  operating_point.json
    results.json  figures/*.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from sepsis.audit.fairness import subgroup_report
from sepsis.audit.robustness import missingness_stress_test, noise_robustness_curve
from sepsis.config import ExperimentConfig
from sepsis.data.preprocess import Preprocessor, TensorDataset
from sepsis.data.psv import load_dataset
from sepsis.data.splits import make_splits
from sepsis.data.synthetic import generate_cohort
from sepsis.evaluation.bootstrap import bootstrap_ci
from sepsis.evaluation.decision_curve import decision_curve
from sepsis.evaluation.metrics import auprc, auroc, classification_summary
from sepsis.evaluation.utility_score import best_threshold_by_utility, normalized_utility
from sepsis.reporting import figures as F
from sepsis.uncertainty.calibration import (
    expected_calibration_error,
    fit_calibrator,
    reliability_curve,
)
from sepsis.uncertainty.conformal import ConformalRiskClassifier
from sepsis.utils.logging import get_logger
from sepsis.utils.seeding import seed_everything

log = get_logger("training.experiment")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _load_records(cfg: ExperimentConfig):
    d = cfg.data
    if d.source == "synthetic":
        log.info(
            "generating synthetic cohort: n=%d prevalence=%.3f",
            d.synthetic_n_patients,
            d.synthetic_prevalence,
        )
        return generate_cohort(d.synthetic_n_patients, d.synthetic_prevalence, d.seed)

    root = Path(d.root)
    n_psv = sum(1 for _ in root.rglob("*.psv")) if root.exists() else 0
    if n_psv < 20:
        from sepsis.data.download import download_sample

        log.info("only %d PSV files under %s -- fetching a sample", n_psv, root)
        download_sample(root, limit=400)
    return load_dataset(root)


# --------------------------------------------------------------------------- #
# model branch: sequence
# --------------------------------------------------------------------------- #
def _run_sequence(cfg, train_td, val_td, test_td, out_dir):
    from sepsis.training.trainer import SequenceTrainer, predict_sequence_scores

    trainer = SequenceTrainer(cfg)
    fit = trainer.fit(train_td, val_td)
    fit.save(out_dir / "model.pt")
    F.plot_training_history(fit.history, out_dir / "figures" / "training_history.png")

    dev = fit.device
    val_scores, val_labels = predict_sequence_scores(fit.model, val_td, dev)
    test_scores, test_labels = predict_sequence_scores(fit.model, test_td, dev)

    def score_fn(td: TensorDataset) -> np.ndarray:
        s, _ = predict_sequence_scores(fit.model, td, dev)
        return np.concatenate(s)

    extra = {
        "best_epoch": fit.best_epoch,
        "history": fit.history,
        "n_parameters": int(sum(p.numel() for p in fit.model.parameters())),
    }
    return fit.model, val_scores, val_labels, test_scores, test_labels, score_fn, extra


# --------------------------------------------------------------------------- #
# model branch: gradient boosting
# --------------------------------------------------------------------------- #
def _run_gbm(cfg, train_td, val_td, test_td, out_dir):
    from sepsis.features.tabular import WindowFeatureExtractor
    from sepsis.models.gbm import GBMRiskModel

    fe = WindowFeatureExtractor(cfg.model.gbm_window)
    tr_tab, va_tab, te_tab = fe.transform(train_td), fe.transform(val_td), fe.transform(test_td)
    model = GBMRiskModel(cfg.model, seed=cfg.train.seed)
    res = model.fit(tr_tab, va_tab)
    model.save(str(out_dir / "model.txt"))

    def _seqs(tab, td):
        sc, lb = model.predict_sequences(tab, len(td), td.X.shape[1])
        out_s, out_l = [], []
        for i in range(len(td)):
            n = int(td.lengths[i])
            out_s.append(sc[i, :n])
            out_l.append(td.y[i, :n].astype(np.int8))
        return out_s, out_l

    val_scores, val_labels = _seqs(va_tab, val_td)
    test_scores, test_labels = _seqs(te_tab, test_td)

    def score_fn(td: TensorDataset) -> np.ndarray:
        tab = fe.transform(td)
        sc, _ = model.predict_sequences(tab, len(td), td.X.shape[1])
        flat = []
        for i in range(len(td)):
            flat.append(sc[i, : int(td.lengths[i])])
        return np.concatenate(flat)

    extra = {
        "gbm_best_iteration": res.best_iteration,
        "gbm_val_auprc": res.val_auprc,
        "top_gain_features": list(res.feature_importance.items())[:25],
        "_tables": (tr_tab, va_tab, te_tab),
    }
    return model, val_scores, val_labels, test_scores, test_labels, score_fn, extra


# --------------------------------------------------------------------------- #
# explainability
# --------------------------------------------------------------------------- #
def _explain_gbm(model, extra, out_dir) -> dict:
    from sepsis.explain.attributions import tree_shap_summary
    from sepsis.explain.pdp_ale import accumulated_local_effects, partial_dependence
    from sepsis.explain.surrogate import GlobalSurrogateTree

    tr_tab, _, te_tab = extra["_tables"]
    shap_summary = tree_shap_summary(model, te_tab, max_samples=4000)
    F.plot_shap_bar(shap_summary["ranking"], out_dir / "figures" / "shap_global.png")

    probs = model.predict(te_tab)
    surro = GlobalSurrogateTree(max_depth=4).fit(te_tab.X, probs, te_tab.feature_names)
    (out_dir / "surrogate_rules.txt").write_text(surro.rules)

    def _predict_matrix(X):
        return model.booster.predict(X, num_iteration=model.booster.best_iteration)

    top_idx = [te_tab.feature_names.index(n) for n, _ in shap_summary["ranking"][:4]]
    pdp = {
        te_tab.feature_names[i]: {
            "pdp": {
                k: v.tolist() for k, v in partial_dependence(_predict_matrix, te_tab.X, i).items()
            },
            "ale": {
                k: v.tolist()
                for k, v in accumulated_local_effects(_predict_matrix, te_tab.X, i).items()
            },
        }
        for i in top_idx
    }
    return {
        "method": "TreeSHAP + PDP/ALE + global surrogate tree",
        "global_ranking": [[n, v] for n, v in shap_summary["ranking"][:25]],
        "surrogate": {
            "fidelity_r2": surro.fidelity_r2,
            "alarm_agreement": surro.alarm_agreement,
            "max_depth": surro.max_depth,
        },
        "pdp_ale": pdp,
    }


def _explain_sequence(model, cfg, test_td, test_scores, out_dir) -> dict:
    from sepsis.explain.attributions import group_attributions, integrated_gradients
    from sepsis.explain.surrogate import GlobalSurrogateTree

    rng = np.random.default_rng(cfg.train.seed)
    # prefer septic stays with signal; fall back to any
    septic = [i for i in range(len(test_td)) if test_td.y[i].max() > 0]
    pool = septic or list(range(len(test_td)))
    pick = rng.choice(pool, size=min(24, len(pool)), replace=False)

    agg: dict[str, float] = {}
    completeness = []
    for i in pick:
        n = int(test_td.lengths[i])
        ig = integrated_gradients(model, test_td.X[i, :n], target_t=n - 1, steps=48)
        completeness.append(abs(ig["completeness_gap"]))
        for name, val in group_attributions(ig["per_feature"], test_td.feature_names):
            agg[name] = agg.get(name, 0.0) + abs(val)
    ranking = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    ranking = [[k, v / len(pick)] for k, v in ranking]
    F.plot_shap_bar(ranking, out_dir / "figures" / "ig_global.png")

    # surrogate on last-hour features vs. model risk
    last_rows, last_p = [], []
    for i in range(len(test_td)):
        n = int(test_td.lengths[i])
        last_rows.append(test_td.X[i, n - 1])
        last_p.append(test_scores[i][-1])
    surro = GlobalSurrogateTree(max_depth=4).fit(
        np.vstack(last_rows), np.asarray(last_p), test_td.feature_names
    )
    (out_dir / "surrogate_rules.txt").write_text(surro.rules)

    out = {
        "method": "Integrated Gradients (grouped) + global surrogate tree",
        "global_ranking": ranking[:25],
        "mean_completeness_gap": float(np.mean(completeness)),
        "surrogate": {
            "fidelity_r2": surro.fidelity_r2,
            "alarm_agreement": surro.alarm_agreement,
            "max_depth": surro.max_depth,
        },
    }
    if cfg.model.name == "transformer":
        from sepsis.explain.attention import temporal_attention_profile

        prof = []
        for i in pick[:12]:
            n = int(test_td.lengths[i])
            p = temporal_attention_profile(model, test_td.X[i, :n])
            prof.append(p["mean_lookback_hours"])
        out["attention_mean_lookback_hours"] = float(np.mean(prof))
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_experiment(cfg: ExperimentConfig) -> dict:
    t0 = time.time()
    seed_everything(cfg.train.seed)
    out_dir = Path(cfg.output_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    records = _load_records(cfg)
    split = make_splits(
        records,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.data.seed,
        group_by_hospital=cfg.data.group_by_hospital,
    )
    log.info("split summary: %s", json.dumps(split.summary()))

    pre = Preprocessor(max_seq_len=cfg.data.max_seq_len).fit(split.train)
    pre.artifacts.save(out_dir / "preprocess.json")
    train_td = pre.transform(split.train)
    val_td = pre.transform(split.val)
    test_td = pre.transform(split.test)

    branch = _run_gbm if cfg.model.name == "lightgbm" else _run_sequence
    (model, val_scores, val_labels, test_scores, test_labels, score_fn, extra) = branch(
        cfg, train_td, val_td, test_td, out_dir
    )

    val_s = np.concatenate(val_scores)
    val_y = np.concatenate(val_labels)
    test_s = np.concatenate(test_scores)
    test_y = np.concatenate(test_labels)

    # --- calibration -----------------------------------------------------
    calibrator = fit_calibrator(val_y, val_s, cfg.uncertainty.calibration)
    joblib.dump(calibrator, out_dir / "calibrator.joblib")
    test_s_cal = calibrator.transform(test_s)
    test_scores_cal = [calibrator.transform(s) for s in test_scores]

    ece_raw = expected_calibration_error(test_y, test_s)
    ece_cal = expected_calibration_error(test_y, test_s_cal)
    F.plot_reliability(
        {
            "uncalibrated": reliability_curve(test_y, test_s, strategy="quantile"),
            cfg.uncertainty.calibration: reliability_curve(test_y, test_s_cal, strategy="quantile"),
        },
        out_dir / "figures" / "reliability.png",
    )

    # --- conformal -----------------------------------------------------------
    val_s_cal = calibrator.transform(val_s)
    conformal = ConformalRiskClassifier(
        alpha=cfg.uncertainty.conformal_alpha, mondrian=cfg.uncertainty.conformal_mondrian
    ).fit(val_y, val_s_cal)
    joblib.dump(conformal, out_dir / "conformal.joblib")
    conformal_eval = conformal.evaluate(test_y, test_s_cal)

    # --- operating point via utility --------------------------------------
    grid = np.round(np.linspace(0.02, 0.8, 40), 4)
    thr, util_break = best_threshold_by_utility(
        val_labels, [calibrator.transform(s) for s in val_scores], grid
    )
    util_curve = [
        normalized_utility(
            val_labels, [(calibrator.transform(s) >= g).astype(int) for s in val_scores]
        ).normalized
        for g in grid
    ]
    F.plot_utility_threshold(grid, util_curve, thr, out_dir / "figures" / "utility_threshold.png")
    test_util = normalized_utility(
        test_labels, [(s >= thr).astype(int) for s in test_scores_cal], threshold=thr
    )
    (out_dir / "operating_point.json").write_text(
        json.dumps(
            {"alarm_threshold": float(thr), "conformal_alpha": cfg.uncertainty.conformal_alpha},
            indent=2,
        )
    )

    # --- headline metrics + clustered CIs -------------------------------
    stay_groups = np.concatenate([np.full(len(s), i) for i, s in enumerate(test_scores)])
    ci_auroc = bootstrap_ci(
        auroc, test_y, test_s_cal, stay_groups, n_resamples=500, seed=cfg.train.seed
    )
    ci_auprc = bootstrap_ci(
        auprc, test_y, test_s_cal, stay_groups, n_resamples=500, seed=cfg.train.seed
    )
    summary = classification_summary(test_y, test_s_cal, threshold=thr)

    dc = decision_curve(test_y, test_s_cal)
    F.plot_decision_curve(dc, out_dir / "figures" / "decision_curve.png")

    # --- fairness -------------------------------------------------------
    fairness = subgroup_report(
        test_y, test_s_cal, test_td.static_raw, test_td.lengths, threshold=thr
    )
    F.plot_subgroup_gaps(fairness, out_dir / "figures" / "subgroup_gaps.png")

    # --- robustness ---------------------------------------------------------
    noise_rows = noise_robustness_curve(test_td, score_fn, seed=cfg.train.seed)
    miss_rows = missingness_stress_test(test_td, score_fn, seed=cfg.train.seed)
    F.plot_robustness(noise_rows, miss_rows, out_dir / "figures" / "robustness.png")

    # --- explainability -------------------------------------------------
    try:
        explain = (
            _explain_gbm(model, extra, out_dir)
            if cfg.model.name == "lightgbm"
            else _explain_sequence(model, cfg, test_td, test_scores_cal, out_dir)
        )
    except Exception as exc:  # pragma: no cover - keep the report resilient
        log.warning("explainability step failed: %s", exc)
        explain = {"error": str(exc)}

    # --- example trajectory -------------------------------------------------
    septic_idx = next((i for i in range(len(test_td)) if test_td.y[i].max() > 0), 0)
    n = int(test_td.lengths[septic_idx])
    onset = next((t for t in range(n) if test_td.y[septic_idx, t] > 0), None)
    F.plot_risk_trajectory(
        np.arange(n),
        test_scores_cal[septic_idx],
        onset,
        thr,
        out_dir / "figures" / "example_trajectory.png",
    )

    results = {
        "config": cfg.to_dict(),
        "runtime_seconds": round(time.time() - t0, 1),
        "data": {"n_records": len(records), "split": split.summary()},
        "model_extra": {k: v for k, v in extra.items() if not k.startswith("_")},
        "discrimination": {
            "auroc": ci_auroc.__dict__,
            "auprc": ci_auprc.__dict__,
            "summary_at_operating_point": summary,
        },
        "calibration": {
            "method": cfg.uncertainty.calibration,
            "ece_uncalibrated": ece_raw,
            "ece_calibrated": ece_cal,
        },
        "conformal": conformal_eval,
        "utility": {
            "alarm_threshold": float(thr),
            "val_utility": util_break.__dict__,
            "test_utility": test_util.__dict__,
        },
        "decision_curve": dc.to_dict(),
        "fairness": fairness,
        "robustness": {"noise": noise_rows, "missingness": miss_rows},
        "explainability": explain,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    log.info(
        "DONE in %.1fs | test AUROC %s | AUPRC %s | utility %.3f | ECE %.3f->%.3f",
        results["runtime_seconds"],
        str(ci_auroc),
        str(ci_auprc),
        test_util.normalized,
        ece_raw["ece"],
        ece_cal["ece"],
    )
    return results
