from __future__ import annotations

import numpy as np


BASELINE_NAMES = ["last_value", "moving_average", "weekly_seasonal_naive"]


def is_baseline(model_name: str) -> bool:
    return model_name.lower().replace("-", "_") in BASELINE_NAMES


def baseline_predict(model_name: str, x: np.ndarray, horizon: int, target_index: int = 0) -> np.ndarray:
    normalized = model_name.lower().replace("-", "_")
    target_history = np.asarray(x, dtype=np.float32)[:, :, target_index]

    if normalized == "last_value":
        return np.repeat(target_history[:, -1:], horizon, axis=1).astype(np.float32)

    if normalized == "moving_average":
        mean = target_history.mean(axis=1, keepdims=True)
        return np.repeat(mean, horizon, axis=1).astype(np.float32)

    if normalized == "weekly_seasonal_naive":
        period = min(7, target_history.shape[1])
        pattern = target_history[:, -period:]
        repeats = int(np.ceil(horizon / period))
        return np.tile(pattern, (1, repeats))[:, :horizon].astype(np.float32)

    raise ValueError(f"Unknown baseline model: {model_name}")
