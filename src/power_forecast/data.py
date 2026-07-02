from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

TARGET_COLUMN = "global_active_power"

RAW_TO_CANONICAL = {
    "Date": "date_raw",
    "Time": "time_raw",
    "Global_active_power": "global_active_power",
    "Global_reactive_power": "global_reactive_power",
    "Voltage": "voltage",
    "Global_intensity": "global_intensity",
    "Sub_metering_1": "sub_metering_1",
    "Sub_metering_2": "sub_metering_2",
    "Sub_metering_3": "sub_metering_3",
}

NUMERIC_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]

SUM_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
]

MEAN_COLUMNS = ["voltage", "global_intensity"]

CALENDAR_COLUMNS = [
    "dayofweek_sin",
    "dayofweek_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "month_sin",
    "month_cos",
]

SCEAUX_LATITUDE = 48.7786
SCEAUX_LONGITUDE = 2.2906
METEO_FRANCE_MONTHLY_URL_TEMPLATE = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/MENS/"
    "MENSQ_{department}_previous-1950-2024.csv.gz"
)

WEATHER_FIELD_MAP = {
    "weather_rain_mm": "RR",
    "weather_tmax": "TX",
    "weather_tmin": "TN",
    "weather_tmean": "TM",
    "weather_temp_amp": "TAMPLIM",
    "weather_rain_days": "NBJRR1",
    "weather_frost_days": "NBJGELEE",
    "weather_hot_days": "NBJTX25",
}

WEATHER_COLUMNS = list(WEATHER_FIELD_MAP)

DEFAULT_FEATURE_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
    *CALENDAR_COLUMNS,
    *WEATHER_COLUMNS,
]


@dataclass(frozen=True)
class WindowArrays:
    x: np.ndarray
    y: np.ndarray
    input_dates: np.ndarray
    forecast_dates: np.ndarray


@dataclass(frozen=True)
class Standardizer:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float

    def transform_features(self, values: np.ndarray) -> np.ndarray:
        return (values - self.feature_mean) / self.feature_std

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        return (values - self.target_mean) / self.target_std

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_std + self.target_mean


def load_raw_power(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", na_values="?", nrows=nrows, low_memory=False)


def clean_raw_power(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.rename(columns=RAW_TO_CANONICAL).copy()
    required = {"date_raw", "time_raw", *NUMERIC_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame["datetime"] = pd.to_datetime(
        frame["date_raw"].astype(str) + " " + frame["time_raw"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime")

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.set_index("datetime")
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].interpolate(method="time", limit_direction="both")
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].ffill().bfill()
    frame = frame.reset_index()
    frame["date"] = frame["datetime"].dt.floor("D")
    return frame


def aggregate_daily(raw: pd.DataFrame) -> pd.DataFrame:
    frame = clean_raw_power(raw)
    frame["sub_metering_remainder"] = (
        frame["global_active_power"] * 1000.0 / 60.0
        - frame["sub_metering_1"]
        - frame["sub_metering_2"]
        - frame["sub_metering_3"]
    )

    agg_spec = {column: "sum" for column in SUM_COLUMNS}
    agg_spec.update({column: "mean" for column in MEAN_COLUMNS})
    daily = frame.groupby("date", as_index=False).agg(agg_spec)

    full_dates = pd.DataFrame(
        {
            "date": pd.date_range(
                daily["date"].min(),
                daily["date"].max(),
                freq="D",
            )
        }
    )
    daily = full_dates.merge(daily, on="date", how="left")
    value_columns = [column for column in daily.columns if column != "date"]
    daily[value_columns] = daily[value_columns].interpolate(limit_direction="both").ffill().bfill()
    return add_calendar_features(daily)


def monthly_weather_url(department: str = "92") -> str:
    return METEO_FRANCE_MONTHLY_URL_TEMPLATE.format(department=str(department).zfill(2))


def download_monthly_weather(
    output_dir: str | Path,
    department: str = "92",
    force: bool = False,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    department_code = str(department).zfill(2)
    path = output / f"MENSQ_{department_code}_previous-1950-2024.csv.gz"
    if path.exists() and not force:
        return path
    urlretrieve(monthly_weather_url(department_code), path)
    return path


def _haversine_km(lat: pd.Series, lon: pd.Series, target_lat: float, target_lon: float) -> pd.Series:
    radius = 6371.0
    lat1 = np.radians(target_lat)
    lat2 = np.radians(lat.astype(float))
    dlat = np.radians(lat.astype(float) - target_lat)
    dlon = np.radians(lon.astype(float) - target_lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return pd.Series(2 * radius * np.arcsin(np.sqrt(a)), index=lat.index)


def load_monthly_weather(
    path: str | Path,
    target_lat: float = SCEAUX_LATITUDE,
    target_lon: float = SCEAUX_LONGITUDE,
    max_distance_km: float | None = None,
) -> pd.DataFrame:
    weather = pd.read_csv(path, sep=";", compression="infer", low_memory=False)
    required = {"AAAAMM", "LAT", "LON", *WEATHER_FIELD_MAP.values()}
    missing = required.difference(weather.columns)
    if missing:
        raise ValueError(f"Missing weather columns: {sorted(missing)}")

    weather["month"] = weather["AAAAMM"].astype(str)
    weather = weather[weather["month"].str.match(r"^\d{6}$", na=False)].copy()
    weather["distance_km"] = _haversine_km(weather["LAT"], weather["LON"], target_lat, target_lon)
    if max_distance_km is not None:
        weather = weather[weather["distance_km"] <= max_distance_km].copy()

    for source in WEATHER_FIELD_MAP.values():
        weather[source] = pd.to_numeric(weather[source], errors="coerce")

    rows: list[dict[str, float | str]] = []
    for month, group in weather.groupby("month", sort=True):
        row: dict[str, float | str] = {"month": str(month)}
        for feature, source in WEATHER_FIELD_MAP.items():
            valid = group.dropna(subset=[source, "distance_km"])
            if valid.empty:
                row[feature] = np.nan
                continue
            weights = 1.0 / np.maximum(valid["distance_km"].to_numpy(dtype=float), 1.0) ** 2
            values = valid[source].to_numpy(dtype=float)
            row[feature] = float(np.average(values, weights=weights))
        rows.append(row)

    monthly = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    monthly[WEATHER_COLUMNS] = monthly[WEATHER_COLUMNS].interpolate(limit_direction="both").ffill().bfill()
    return monthly


def add_monthly_weather_features(
    daily: pd.DataFrame,
    monthly_weather: pd.DataFrame,
    lag_months: int = 1,
) -> pd.DataFrame:
    if lag_months < 0:
        raise ValueError("lag_months must be non-negative.")
    missing = {"month", *WEATHER_COLUMNS}.difference(monthly_weather.columns)
    if missing:
        raise ValueError(f"Missing monthly weather columns: {sorted(missing)}")

    frame = daily.copy()
    dates = pd.to_datetime(frame["date"])
    month_period = dates.dt.to_period("M") - lag_months
    frame["_weather_month"] = month_period.astype(str).str.replace("-", "", regex=False)

    weather = monthly_weather.loc[:, ["month", *WEATHER_COLUMNS]].copy()
    merged = frame.merge(weather, left_on="_weather_month", right_on="month", how="left")
    merged = merged.drop(columns=["_weather_month", "month"])
    merged[WEATHER_COLUMNS] = merged[WEATHER_COLUMNS].interpolate(limit_direction="both").ffill().bfill()
    return merged


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    dates = pd.to_datetime(frame["date"])
    dayofweek = dates.dt.dayofweek.to_numpy()
    dayofyear = dates.dt.dayofyear.to_numpy()
    month = dates.dt.month.to_numpy()

    frame["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    frame["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    frame["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 366)
    frame["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 366)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    return frame


def chronological_split(
    daily: pd.DataFrame,
    test_days: int = 365,
    input_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(daily) <= test_days + input_days:
        raise ValueError("Not enough daily rows for the requested split.")

    ordered = daily.sort_values("date").reset_index(drop=True)
    split_index = len(ordered) - test_days
    train = ordered.iloc[:split_index].reset_index(drop=True)
    test_start = max(0, split_index - input_days)
    test = ordered.iloc[test_start:].reset_index(drop=True)
    return train, test


def make_windows(
    daily: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    input_days: int,
    horizon: int,
    stride: int = 1,
) -> WindowArrays:
    if input_days <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("input_days, horizon, and stride must be positive.")

    frame = daily.sort_values("date").reset_index(drop=True)
    missing = set(feature_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    features = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float32)
    target = frame.loc[:, target_column].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(frame["date"]).to_numpy()

    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    input_dates: list[np.ndarray] = []
    forecast_dates: list[np.ndarray] = []

    last_start = len(frame) - input_days - horizon
    for start in range(0, last_start + 1, stride):
        input_end = start + input_days
        forecast_end = input_end + horizon
        x_values.append(features[start:input_end])
        y_values.append(target[input_end:forecast_end])
        input_dates.append(dates[start:input_end])
        forecast_dates.append(dates[input_end:forecast_end])

    if not x_values:
        return WindowArrays(
            x=np.empty((0, input_days, len(feature_columns)), dtype=np.float32),
            y=np.empty((0, horizon), dtype=np.float32),
            input_dates=np.empty((0, input_days), dtype="datetime64[ns]"),
            forecast_dates=np.empty((0, horizon), dtype="datetime64[ns]"),
        )

    return WindowArrays(
        x=np.stack(x_values).astype(np.float32),
        y=np.stack(y_values).astype(np.float32),
        input_dates=np.stack(input_dates),
        forecast_dates=np.stack(forecast_dates),
    )


def fit_standardizer(
    train_daily: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> Standardizer:
    feature_values = train_daily.loc[:, list(feature_columns)].to_numpy(dtype=np.float32)
    target_values = train_daily.loc[:, target_column].to_numpy(dtype=np.float32)

    feature_mean = feature_values.mean(axis=0)
    feature_std = feature_values.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    target_mean = float(target_values.mean())
    target_std = float(target_values.std())
    if target_std < 1e-6:
        target_std = 1.0

    return Standardizer(
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_mean=target_mean,
        target_std=target_std,
    )


def scale_daily(
    daily: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    standardizer: Standardizer,
) -> pd.DataFrame:
    frame = daily.copy()
    raw_features = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float32)
    raw_target = frame.loc[:, target_column].to_numpy(dtype=np.float32)
    frame.loc[:, list(feature_columns)] = standardizer.transform_features(raw_features)
    frame.loc[:, target_column] = standardizer.transform_target(raw_target)
    return frame


def save_prepared_splits(
    raw_path: str | Path,
    output_dir: str | Path,
    test_days: int = 365,
    input_days: int = 90,
    weather_path: str | Path | None = None,
    weather_lag_months: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    daily = aggregate_daily(load_raw_power(raw_path))
    if weather_path is not None:
        if weather_lag_months is None:
            raise ValueError("weather_lag_months must be set when weather_path is provided.")
        monthly_weather = load_monthly_weather(weather_path)
        daily = add_monthly_weather_features(daily, monthly_weather, lag_months=weather_lag_months)
    train, test = chronological_split(daily, test_days=test_days, input_days=input_days)

    daily.to_csv(output / "daily_power.csv", index=False)
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    return daily, train, test
