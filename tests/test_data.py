import numpy as np
import pandas as pd

from power_forecast.data import (
    TARGET_COLUMN,
    add_calendar_features,
    add_monthly_weather_features,
    aggregate_daily,
    chronological_split,
    fit_standardizer,
    load_monthly_weather,
    make_windows,
    scale_daily,
)


def sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["01/01/2010", "01/01/2010", "02/01/2010", "02/01/2010"],
            "Time": ["00:00:00", "00:01:00", "00:00:00", "00:01:00"],
            "Global_active_power": [1.0, 2.0, 3.0, 4.0],
            "Global_reactive_power": [0.1, 0.2, 0.3, 0.4],
            "Voltage": [230.0, 232.0, 234.0, 236.0],
            "Global_intensity": [4.0, 5.0, 6.0, 7.0],
            "Sub_metering_1": [1.0, 2.0, 3.0, 4.0],
            "Sub_metering_2": [2.0, 3.0, 4.0, 5.0],
            "Sub_metering_3": [3.0, 4.0, 5.0, 6.0],
        }
    )


def test_aggregate_daily_matches_pdf_rules() -> None:
    daily = aggregate_daily(sample_raw_frame())

    assert list(daily["date"].astype(str)) == ["2010-01-01", "2010-01-02"]
    assert daily.loc[0, "global_active_power"] == 3.0
    assert np.isclose(daily.loc[0, "global_reactive_power"], 0.3)
    assert daily.loc[0, "sub_metering_1"] == 3.0
    assert daily.loc[0, "sub_metering_2"] == 5.0
    assert daily.loc[0, "voltage"] == 231.0
    assert daily.loc[0, "global_intensity"] == 4.5
    expected_remainder = (1.0 * 1000 / 60 - 1 - 2 - 3) + (2.0 * 1000 / 60 - 2 - 3 - 4)
    assert np.isclose(daily.loc[0, "sub_metering_remainder"], expected_remainder)


def test_chronological_split_keeps_test_warmup() -> None:
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    daily = pd.DataFrame({"date": dates, TARGET_COLUMN: np.arange(20)})

    train, test = chronological_split(daily, test_days=5, input_days=3)

    assert train["date"].iloc[0] == pd.Timestamp("2020-01-01")
    assert train["date"].iloc[-1] == pd.Timestamp("2020-01-15")
    assert test["date"].iloc[0] == pd.Timestamp("2020-01-13")
    assert test["date"].iloc[-1] == pd.Timestamp("2020-01-20")
    assert len(test) == 8


def test_make_windows_returns_expected_shapes_and_dates() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    daily = pd.DataFrame(
        {
            "date": dates,
            TARGET_COLUMN: np.arange(8, dtype=float),
            "feature": np.arange(100, 108, dtype=float),
        }
    )

    windows = make_windows(
        daily,
        feature_columns=[TARGET_COLUMN, "feature"],
        target_column=TARGET_COLUMN,
        input_days=3,
        horizon=2,
    )

    assert windows.x.shape == (4, 3, 2)
    assert windows.y.shape == (4, 2)
    np.testing.assert_array_equal(windows.y[0], np.array([3.0, 4.0]))
    assert list(pd.to_datetime(windows.forecast_dates[0]).date.astype(str)) == ["2020-01-04", "2020-01-05"]


def test_standardizer_uses_train_statistics_only() -> None:
    train = pd.DataFrame(
        {
            TARGET_COLUMN: [1.0, 2.0, 3.0],
            "feature": [10.0, 20.0, 30.0],
        }
    )
    test = pd.DataFrame(
        {
            TARGET_COLUMN: [100.0],
            "feature": [1000.0],
        }
    )

    scaler = fit_standardizer(train, [TARGET_COLUMN, "feature"], TARGET_COLUMN)
    train_scaled = scaler.transform_features(train[[TARGET_COLUMN, "feature"]].to_numpy(float))
    test_scaled = scaler.transform_features(test[[TARGET_COLUMN, "feature"]].to_numpy(float))

    assert np.isclose(train_scaled[:, 0].mean(), 0.0)
    assert test_scaled[0, 0] > 50.0


def test_scale_daily_does_not_double_scale_target_feature() -> None:
    train = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            TARGET_COLUMN: [1.0, 2.0, 3.0],
            "feature": [10.0, 20.0, 30.0],
        }
    )
    scaler = fit_standardizer(train, [TARGET_COLUMN, "feature"], TARGET_COLUMN)

    scaled = scale_daily(train, [TARGET_COLUMN, "feature"], TARGET_COLUMN, scaler)

    assert np.isclose(scaled[TARGET_COLUMN].iloc[1], 0.0)


def test_calendar_features_are_deterministic() -> None:
    daily = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2, freq="D")})
    enriched = add_calendar_features(daily)

    assert {"dayofweek_sin", "dayofweek_cos", "dayofyear_sin", "dayofyear_cos"}.issubset(enriched.columns)
    assert enriched["dayofweek_sin"].between(-1, 1).all()


def test_monthly_weather_is_distance_weighted(tmp_path) -> None:
    weather_path = tmp_path / "weather.csv.gz"
    frame = pd.DataFrame(
        {
            "NUM_POSTE": ["near", "far"],
            "NOM_USUEL": ["near", "far"],
            "LAT": [48.7786, 48.8786],
            "LON": [2.2906, 2.3906],
            "AAAAMM": ["201001", "201001"],
            "RR": [10.0, 100.0],
            "TX": [8.0, 18.0],
            "TN": [2.0, 12.0],
            "TM": [5.0, 15.0],
            "TAMPLIM": [6.0, 6.0],
            "NBJRR1": [4.0, 14.0],
            "NBJGELEE": [8.0, 0.0],
            "NBJTX25": [0.0, 3.0],
        }
    )
    frame.to_csv(weather_path, sep=";", index=False, compression="gzip")

    monthly = load_monthly_weather(weather_path)

    assert monthly.loc[0, "month"] == "201001"
    assert monthly.loc[0, "weather_rain_mm"] < 20.0
    assert monthly.loc[0, "weather_tmax"] < 10.0


def test_add_monthly_weather_uses_previous_month_by_default() -> None:
    daily = pd.DataFrame({"date": pd.date_range("2010-02-01", periods=2, freq="D"), TARGET_COLUMN: [1.0, 2.0]})
    monthly = pd.DataFrame(
        {
            "month": ["201001", "201002"],
            "weather_rain_mm": [10.0, 20.0],
            "weather_tmax": [8.0, 9.0],
            "weather_tmin": [2.0, 3.0],
            "weather_tmean": [5.0, 6.0],
            "weather_temp_amp": [6.0, 6.0],
            "weather_rain_days": [4.0, 5.0],
            "weather_frost_days": [8.0, 2.0],
            "weather_hot_days": [0.0, 0.0],
        }
    )

    enriched = add_monthly_weather_features(daily, monthly, lag_months=1)

    assert enriched["weather_rain_mm"].tolist() == [10.0, 10.0]
    assert enriched["weather_tmean"].tolist() == [5.0, 5.0]
