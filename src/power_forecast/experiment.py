from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .baselines import BASELINE_NAMES, baseline_predict, is_baseline
from .data import (
    DEFAULT_FEATURE_COLUMNS,
    TARGET_COLUMN,
    chronological_split,
    fit_standardizer,
    load_raw_power,
    aggregate_daily,
    make_windows,
    scale_daily,
)
from .models import build_model
from .training import TrainConfig, compute_metrics, predict, set_seed, split_train_validation, train_model


@dataclass(frozen=True)
class Preset:
    epochs: int
    d_model: int
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    learning_rate: float
    patience: int


PRESETS = {
    "smoke": Preset(epochs=1, d_model=16, hidden_size=16, num_layers=1, num_heads=4, dropout=0.0, learning_rate=1e-3, patience=1),
    "quick": Preset(epochs=3, d_model=24, hidden_size=24, num_layers=1, num_heads=4, dropout=0.05, learning_rate=1e-3, patience=2),
    "course": Preset(epochs=16, d_model=48, hidden_size=48, num_layers=2, num_heads=4, dropout=0.1, learning_rate=8e-4, patience=4),
    "full": Preset(epochs=40, d_model=64, hidden_size=64, num_layers=2, num_heads=4, dropout=0.1, learning_rate=7e-4, patience=6),
}


MODEL_NAMES = [
    "last_value",
    "moving_average",
    "weekly_seasonal_naive",
    "lstm",
    "transformer",
    "patch_channel_mixer",
]
HORIZONS = [90, 365]


def prepare_or_load_splits(
    raw_path: Path,
    data_dir: Path,
    test_days: int,
    input_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_path = data_dir / "daily_power.csv"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if daily_path.exists() and train_path.exists() and test_path.exists():
        daily = pd.read_csv(daily_path, parse_dates=["date"])
        train = pd.read_csv(train_path, parse_dates=["date"])
        test = pd.read_csv(test_path, parse_dates=["date"])
        return daily, train, test

    data_dir.mkdir(parents=True, exist_ok=True)
    daily = aggregate_daily(load_raw_power(raw_path))
    train, test = chronological_split(daily, test_days=test_days, input_days=input_days)
    daily.to_csv(daily_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return daily, train, test


def available_feature_columns(daily: pd.DataFrame) -> list[str]:
    return [column for column in DEFAULT_FEATURE_COLUMNS if column in daily.columns]


def flatten_predictions(
    model_name: str,
    horizon: int,
    seed: int,
    forecast_dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_idx in range(y_true.shape[0]):
        for step in range(y_true.shape[1]):
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "seed": seed,
                    "sample_index": sample_idx,
                    "step": step + 1,
                    "forecast_date": pd.Timestamp(forecast_dates[sample_idx, step]).strftime("%Y-%m-%d"),
                    "ground_truth": float(y_true[sample_idx, step]),
                    "prediction": float(y_pred[sample_idx, step]),
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["model", "horizon"], as_index=False).agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        best_epoch_mean=("best_epoch", "mean"),
        seconds_mean=("seconds", "mean"),
    )
    return grouped.sort_values(["horizon", "model"]).reset_index(drop=True)


def run_suite(
    raw_path: Path,
    data_dir: Path,
    reports_dir: Path,
    models_dir: Path,
    model_names: Iterable[str],
    horizons: Iterable[int],
    seeds: Iterable[int],
    preset_name: str,
    input_days: int = 90,
    test_days: int = 365,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name}. Choose from {sorted(PRESETS)}")

    preset = PRESETS[preset_name]
    daily, train_daily, test_daily = prepare_or_load_splits(raw_path, data_dir, test_days=test_days, input_days=input_days)
    feature_columns = available_feature_columns(daily)
    if TARGET_COLUMN not in feature_columns:
        raise ValueError(f"{TARGET_COLUMN} must be included in feature columns.")

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for horizon in horizons:
        scaler = fit_standardizer(train_daily, feature_columns, TARGET_COLUMN)
        train_scaled = scale_daily(train_daily, feature_columns, TARGET_COLUMN, scaler)
        test_scaled = scale_daily(test_daily, feature_columns, TARGET_COLUMN, scaler)
        train_windows_all = make_windows(train_scaled, feature_columns, TARGET_COLUMN, input_days=input_days, horizon=horizon)
        test_windows = make_windows(test_scaled, feature_columns, TARGET_COLUMN, input_days=input_days, horizon=horizon)
        train_windows, val_windows = split_train_validation(train_windows_all, val_fraction=0.2)

        for model_name in model_names:
            for seed in seeds:
                if is_baseline(model_name):
                    start = time.perf_counter()
                    pred_scaled = baseline_predict(model_name, test_windows.x, horizon=horizon)
                    y_pred = scaler.inverse_target(pred_scaled)
                    y_true = scaler.inverse_target(test_windows.y)
                    metrics = compute_metrics(y_true, y_pred)
                    seconds = time.perf_counter() - start
                    metrics_rows.append(
                        {
                            "model": model_name,
                            "horizon": horizon,
                            "seed": seed,
                            "mse": metrics["mse"],
                            "mae": metrics["mae"],
                            "best_val_loss": np.nan,
                            "best_epoch": 0,
                            "epochs_run": 0,
                            "seconds": seconds,
                            "preset": preset_name,
                            "checkpoint": "",
                        }
                    )
                    prediction_frames.append(
                        flatten_predictions(model_name, horizon, seed, test_windows.forecast_dates, y_true, y_pred)
                    )
                    print(
                        f"{model_name:24s} horizon={horizon:3d} seed={seed} "
                        f"MSE={metrics['mse']:.3f} MAE={metrics['mae']:.3f} "
                        f"epoch=0 time={seconds:.1f}s",
                        flush=True,
                    )
                    continue

                batch_size = 16 if horizon >= 365 else 32
                config = TrainConfig(
                    epochs=preset.epochs,
                    batch_size=batch_size,
                    learning_rate=preset.learning_rate,
                    weight_decay=1e-4,
                    patience=preset.patience,
                    val_fraction=0.2,
                    device=device,
                )
                set_seed(seed)
                model = build_model(
                    model_name=model_name,
                    input_dim=len(feature_columns),
                    horizon=horizon,
                    d_model=preset.d_model,
                    hidden_size=preset.hidden_size,
                    num_layers=preset.num_layers,
                    num_heads=preset.num_heads,
                    dropout=preset.dropout,
                )
                checkpoint = models_dir / f"{model_name}_h{horizon}_seed{seed}.pt"
                start = time.perf_counter()
                result = train_model(model, train_windows, val_windows, config, seed=seed, checkpoint_path=checkpoint)
                seconds = time.perf_counter() - start

                pred_scaled = predict(model, test_windows, batch_size=batch_size, device=device)
                y_pred = scaler.inverse_target(pred_scaled)
                y_true = scaler.inverse_target(test_windows.y)
                metrics = compute_metrics(y_true, y_pred)
                metrics_rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "seed": seed,
                        "mse": metrics["mse"],
                        "mae": metrics["mae"],
                        "best_val_loss": result.best_val_loss,
                        "best_epoch": result.best_epoch,
                        "epochs_run": len(result.history),
                        "seconds": seconds,
                        "preset": preset_name,
                        "checkpoint": str(checkpoint),
                    }
                )
                prediction_frames.append(
                    flatten_predictions(model_name, horizon, seed, test_windows.forecast_dates, y_true, y_pred)
                )
                print(
                    f"{model_name:24s} horizon={horizon:3d} seed={seed} "
                    f"MSE={metrics['mse']:.3f} MAE={metrics['mae']:.3f} "
                    f"epoch={result.best_epoch} time={seconds:.1f}s",
                    flush=True,
                )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = summarize_metrics(metrics_df)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    metrics_df.to_csv(reports_dir / "metrics_runs.csv", index=False)
    summary_df.to_csv(reports_dir / "metrics_summary.csv", index=False)
    predictions_df.to_csv(reports_dir / "predictions.csv", index=False)
    return metrics_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run household power forecasting experiments.")
    parser.add_argument("--raw-path", type=Path, default=Path("household_power_consumption.txt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", "--reports-dir", dest="reports_dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--input-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=365)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = run_suite(
        raw_path=args.raw_path,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        model_names=args.models,
        horizons=args.horizons,
        seeds=args.seeds,
        preset_name=args.preset,
        input_days=args.input_days,
        test_days=args.test_days,
        device=args.device,
    )
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
