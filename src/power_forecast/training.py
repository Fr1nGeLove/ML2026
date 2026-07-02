from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import Standardizer, WindowArrays


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    min_delta: float = 1e-5
    val_fraction: float = 0.2
    device: str = "auto"
    num_workers: int = 0
    grad_clip: float = 1.0


@dataclass(frozen=True)
class TrainResult:
    history: list[dict[str, float]]
    best_val_loss: float
    best_epoch: int
    checkpoint_path: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))


def split_train_validation(windows: WindowArrays, val_fraction: float) -> tuple[WindowArrays, WindowArrays]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5.")
    n = len(windows.x)
    val_size = max(1, int(round(n * val_fraction)))
    train_size = n - val_size
    if train_size <= 0:
        raise ValueError("Not enough windows for train/validation split.")

    train = WindowArrays(
        x=windows.x[:train_size],
        y=windows.y[:train_size],
        input_dates=windows.input_dates[:train_size],
        forecast_dates=windows.forecast_dates[:train_size],
    )
    val = WindowArrays(
        x=windows.x[train_size:],
        y=windows.y[train_size:],
        input_dates=windows.input_dates[train_size:],
        forecast_dates=windows.forecast_dates[train_size:],
    )
    return train, val


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
    }


def train_model(
    model: nn.Module,
    train_windows: WindowArrays,
    val_windows: WindowArrays,
    config: TrainConfig,
    seed: int,
    checkpoint_path: str | Path | None = None,
) -> TrainResult:
    set_seed(seed)
    device = resolve_device(config.device)
    model = model.to(device)

    train_loader = DataLoader(
        make_tensor_dataset(train_windows.x, train_windows.y),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        make_tensor_dataset(val_windows.x, val_windows.y),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - config.min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    saved_path: str | None = None
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": asdict(config),
                "history": history,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
            },
            path,
        )
        saved_path = str(path)

    return TrainResult(history=history, best_val_loss=best_val, best_epoch=best_epoch, checkpoint_path=saved_path)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x)
            losses.append(float(criterion(prediction, batch_y).detach().cpu()))
    return float(np.mean(losses))


def predict(model: nn.Module, windows: WindowArrays, batch_size: int, device: str = "auto") -> np.ndarray:
    resolved = resolve_device(device)
    model = model.to(resolved)
    model.eval()
    loader = DataLoader(make_tensor_dataset(windows.x, windows.y), batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            prediction = model(batch_x.to(resolved)).detach().cpu().numpy()
            predictions.append(prediction)
    return np.concatenate(predictions, axis=0)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
