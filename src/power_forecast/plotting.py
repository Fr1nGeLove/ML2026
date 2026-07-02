from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "last_value": "Last Value",
    "moving_average": "Moving Average",
    "weekly_seasonal_naive": "Weekly Seasonal",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "trend_conv_transformer": "PatchChannelMixer",
    "patch_channel_mixer": "PatchChannelMixer",
}

MODEL_ORDER = [
    "last_value",
    "moving_average",
    "weekly_seasonal_naive",
    "lstm",
    "transformer",
    "patch_channel_mixer",
]

BASELINE_MODELS = {"last_value", "moving_average", "weekly_seasonal_naive"}


def plot_metric_summary(summary_csv: str | Path, output_path: str | Path) -> Path:
    summary = pd.read_csv(summary_csv)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)
    colors = ["#9ecae9", "#ffbf79", "#a1d99b", "#4c78a8", "#f58518", "#54a24b"]

    for axis, metric, ylabel in zip(axes, ["mse", "mae"], ["MSE", "MAE"]):
        pivot = summary.pivot(index="horizon", columns="model", values=f"{metric}_mean")
        pivot = pivot[[model for model in MODEL_ORDER if model in pivot.columns]]
        err = summary.pivot(index="horizon", columns="model", values=f"{metric}_std")
        err = err[pivot.columns]
        labels = [MODEL_LABELS.get(column, column) for column in pivot.columns]
        pivot.plot(kind="bar", yerr=err, ax=axis, color=colors[: len(pivot.columns)], capsize=3)
        axis.set_title(f"{ylabel} by horizon")
        axis.set_xlabel("Forecast horizon (days)")
        axis.set_ylabel(ylabel)
        axis.legend(labels, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=0)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_prediction_curves(
    predictions_csv: str | Path,
    output_dir: str | Path,
    sample_index: int = 0,
) -> list[Path]:
    predictions = pd.read_csv(predictions_csv, parse_dates=["forecast_date"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for horizon in sorted(predictions["horizon"].unique()):
        subset = predictions[(predictions["horizon"] == horizon) & (predictions["sample_index"] == sample_index)]
        grouped = (
            subset.groupby(["model", "forecast_date"], as_index=False)
            .agg(ground_truth=("ground_truth", "mean"), prediction=("prediction", "mean"))
            .sort_values("forecast_date")
        )
        truth = grouped.groupby("forecast_date", as_index=False)["ground_truth"].mean()

        fig, axis = plt.subplots(figsize=(11, 4.8), dpi=160)
        axis.plot(truth["forecast_date"], truth["ground_truth"], color="#222222", linewidth=2.0, label="Ground Truth")
        for model_name in MODEL_ORDER:
            model_frame = grouped[grouped["model"] == model_name]
            if model_frame.empty:
                continue
            is_base = model_name in BASELINE_MODELS
            axis.plot(
                model_frame["forecast_date"],
                model_frame["prediction"],
                linewidth=1.2 if is_base else 1.8,
                linestyle="--" if is_base else "-",
                alpha=0.55 if is_base else 0.95,
                label=MODEL_LABELS.get(model_name, model_name),
            )
        axis.set_title(f"Prediction vs Ground Truth, horizon={horizon} days")
        axis.set_xlabel("Date")
        axis.set_ylabel("Daily global active power")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        path = output / f"prediction_h{horizon}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def make_all_plots(reports_dir: str | Path = "results", figures_dir: str | Path = "figures") -> list[Path]:
    reports = Path(reports_dir)
    figures = Path(figures_dir)
    paths = [plot_metric_summary(reports / "metrics_summary.csv", figures / "metrics_summary.png")]
    paths.extend(plot_prediction_curves(reports / "predictions.csv", figures))
    return paths
