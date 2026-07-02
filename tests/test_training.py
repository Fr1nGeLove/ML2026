import numpy as np

from power_forecast.training import compute_metrics, make_tensor_dataset


def test_compute_metrics_returns_mse_and_mae() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[2.0, 2.0], [1.0, 4.0]])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mse"] == 1.25
    assert metrics["mae"] == 0.75


def test_make_tensor_dataset_preserves_window_shapes() -> None:
    x = np.zeros((3, 4, 2), dtype=np.float32)
    y = np.ones((3, 5), dtype=np.float32)

    dataset = make_tensor_dataset(x, y)

    assert len(dataset) == 3
    first_x, first_y = dataset[0]
    assert tuple(first_x.shape) == (4, 2)
    assert tuple(first_y.shape) == (5,)
