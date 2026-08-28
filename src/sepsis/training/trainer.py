"""Training loop for the causal sequence models.

Features that matter for a defensible clinical benchmark:

* checkpoint on the **validation AUPRC** (prevalence ~2%, so AUROC saturates),
* imbalance-aware loss with ``pos_weight`` derived from the *training* split,
* gradient clipping and cosine LR decay for stable Transformer/TCN training,
* early stopping with best-weight restore,
* MLflow logging that degrades gracefully to a local ``./mlruns`` store.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sepsis.config import ExperimentConfig
from sepsis.data.preprocess import TensorDataset
from sepsis.data.torch_dataset import SequenceDataset, collate_padded
from sepsis.evaluation.metrics import auprc, auroc
from sepsis.models.losses import build_loss
from sepsis.models.registry import build_sequence_model
from sepsis.utils.logging import get_logger
from sepsis.utils.seeding import seed_everything

log = get_logger("training.trainer")


def resolve_device(name: str) -> torch.device:
    """``auto`` picks CUDA when present, else CPU. Apple MPS is only used when
    requested explicitly (``train.device=mps``) because several ops used here
    (masked TransformerEncoder, deterministic RNG) still fall back or error."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclasses.dataclass
class TrainOutput:
    model: torch.nn.Module
    history: list[dict[str, float]]
    best_epoch: int
    best_metric: float
    best_state: dict
    device: str

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.best_state,
                "history": self.history,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
            },
            path,
        )


def _loaders(train_td, val_td, batch_size, num_workers):
    tr = DataLoader(
        SequenceDataset(train_td),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_padded,
        num_workers=num_workers,
        drop_last=False,
    )
    va = DataLoader(
        SequenceDataset(val_td),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_padded,
        num_workers=num_workers,
    )
    return tr, va


class SequenceTrainer:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.train.device)

    def _pos_weight(self, td: TensorDataset) -> float:
        if self.cfg.train.pos_weight is not None:
            return float(self.cfg.train.pos_weight)
        _, y = td.flat_valid()
        pos = float(y.sum())
        neg = float(y.size - pos)
        return max(neg / max(pos, 1.0), 1.0)

    def fit(self, train_td: TensorDataset, val_td: TensorDataset) -> TrainOutput:
        cfg = self.cfg
        seed_everything(cfg.train.seed)
        model = build_sequence_model(cfg.model, input_size=train_td.n_features)
        model.to(self.device)

        loss_fn = build_loss(cfg.train.loss, cfg.train.focal_gamma, self._pos_weight(train_td)).to(
            self.device
        )
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        tr_loader, va_loader = _loaders(
            train_td, val_td, cfg.train.batch_size, cfg.train.num_workers
        )
        steps = max(1, len(tr_loader)) * cfg.train.epochs
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
        )
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and self.device.type == "cuda")

        run = _mlflow_start(cfg)
        history: list[dict[str, float]] = []
        best_metric = -math.inf
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = 0
        patience = 0

        for epoch in range(1, cfg.train.epochs + 1):
            model.train()
            t0 = time.time()
            running = 0.0
            for x, y, pad_mask, _ in tr_loader:
                x, y, pad_mask = x.to(self.device), y.to(self.device), pad_mask.to(self.device)
                opt.zero_grad(set_to_none=True)
                autocast = (
                    torch.cuda.amp.autocast(enabled=scaler.is_enabled())
                    if self.device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast:
                    logits = model(x, pad_mask)
                    loss = loss_fn(logits, y, pad_mask)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += float(loss.detach()) * x.size(0)

            val = self.evaluate(model, va_loader)
            row = {
                "epoch": epoch,
                "train_loss": running / len(tr_loader.dataset),
                "val_auroc": val["auroc"],
                "val_auprc": val["auprc"],
                "val_loss": val["loss"],
                "lr": sched.get_last_lr()[0],
                "seconds": round(time.time() - t0, 1),
            }
            history.append(row)
            _mlflow_log(run, row, step=epoch)
            log.info(
                "epoch %02d | loss %.4f | val AUROC %.4f | val AUPRC %.4f | %.1fs",
                epoch,
                row["train_loss"],
                row["val_auroc"],
                row["val_auprc"],
                row["seconds"],
            )

            monitored = val["auprc"] if cfg.train.monitor == "val_auprc" else val["auroc"]
            if monitored > best_metric + 1e-5:
                best_metric = monitored
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.train.early_stopping_patience:
                    log.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                    break

        model.load_state_dict(best_state)
        _mlflow_end(run, {"best_val_metric": best_metric, "best_epoch": best_epoch})
        return TrainOutput(
            model=model,
            history=history,
            best_epoch=best_epoch,
            best_metric=float(best_metric),
            best_state=best_state,
            device=str(self.device),
        )

    @torch.no_grad()
    def evaluate(self, model, loader) -> dict[str, float]:
        model.eval()
        loss_fn = build_loss(self.cfg.train.loss, self.cfg.train.focal_gamma, None).to(self.device)
        scores, labels = [], []
        total = 0.0
        for x, y, pad_mask, _ in loader:
            x, y, pad_mask = x.to(self.device), y.to(self.device), pad_mask.to(self.device)
            logits = model(x, pad_mask)
            total += float(loss_fn(logits, y, pad_mask)) * x.size(0)
            m = pad_mask.bool()
            scores.append(torch.sigmoid(logits)[m].cpu().numpy())
            labels.append(y[m].cpu().numpy())
        s = np.concatenate(scores)
        y_all = np.concatenate(labels)
        return {
            "loss": total / len(loader.dataset),
            "auroc": auroc(y_all, s),
            "auprc": auprc(y_all, s),
        }


@torch.no_grad()
def predict_sequence_scores(
    model, td: TensorDataset, device: str = "cpu", batch_size: int = 128
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-stay ``(risk_scores, labels)`` lists, ordered like ``td``."""
    model.eval()
    model.to(device)
    loader = DataLoader(
        SequenceDataset(td),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_padded,
    )
    out_scores: list[np.ndarray] = []
    out_labels: list[np.ndarray] = []
    for x, y, pad_mask, lengths in loader:
        probs = torch.sigmoid(model(x.to(device), pad_mask.to(device))).cpu().numpy()
        for i, n in enumerate(lengths.tolist()):
            out_scores.append(probs[i, :n].astype(np.float64))
            out_labels.append(y[i, :n].numpy().astype(np.int8))
    return out_scores, out_labels


# --------------------------------------------------------------------------- #
# MLflow: optional, never fatal
# --------------------------------------------------------------------------- #
def _mlflow_start(cfg: ExperimentConfig):
    try:
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        mlflow.set_experiment(cfg.train.mlflow_experiment)
        run = mlflow.start_run(run_name=cfg.train.run_name)
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "hidden_size": cfg.model.hidden_size,
                "num_layers": cfg.model.num_layers,
                "loss": cfg.train.loss,
                "lr": cfg.train.lr,
                "batch_size": cfg.train.batch_size,
                "data_source": cfg.data.source,
            }
        )
        return run
    except Exception as exc:  # pragma: no cover
        log.warning("MLflow disabled (%s)", exc)
        return None


def _mlflow_log(run, row: dict[str, float], step: int) -> None:
    if run is None:
        return
    try:
        import mlflow

        mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "epoch"}, step=step)
    except Exception:  # pragma: no cover
        pass


def _mlflow_end(run, metrics: dict[str, float]) -> None:
    if run is None:
        return
    try:
        import mlflow

        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.end_run()
    except Exception:  # pragma: no cover
        pass
