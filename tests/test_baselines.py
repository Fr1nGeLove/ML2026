import numpy as np

from power_forecast.baselines import baseline_predict


def test_last_value_repeats_most_recent_target_feature() -> None:
    x = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)

    pred = baseline_predict("last_value", x, horizon=4)

    np.testing.assert_array_equal(pred, np.array([[3.0, 3.0, 3.0, 3.0]], dtype=np.float32))


def test_moving_average_repeats_input_mean() -> None:
    x = np.array([[[1.0], [2.0], [6.0]]], dtype=np.float32)

    pred = baseline_predict("moving_average", x, horizon=2)

    np.testing.assert_array_equal(pred, np.array([[3.0, 3.0]], dtype=np.float32))


def test_weekly_seasonal_naive_repeats_recent_week_pattern() -> None:
    x = np.array([np.arange(1, 9, dtype=np.float32).reshape(8, 1)])

    pred = baseline_predict("weekly_seasonal_naive", x, horizon=10)

    np.testing.assert_array_equal(
        pred,
        np.array([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    )
