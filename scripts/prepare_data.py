from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast.data import WEATHER_COLUMNS, download_monthly_weather, save_prepared_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare daily train/test CSV files.")
    parser.add_argument("--raw-path", type=Path, default=ROOT / "household_power_consumption.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--input-days", type=int, default=90)
    parser.add_argument("--weather-mode", choices=["lagged", "current", "none"], default="lagged")
    parser.add_argument("--weather-path", type=Path, default=None)
    parser.add_argument("--weather-dir", type=Path, default=ROOT / "data" / "weather")
    parser.add_argument("--weather-department", default="92")
    parser.add_argument("--force-weather-download", action="store_true")
    args = parser.parse_args()

    weather_path = args.weather_path
    weather_lag_months = None
    if args.weather_mode != "none":
        weather_lag_months = 1 if args.weather_mode == "lagged" else 0
        if weather_path is None:
            weather_path = download_monthly_weather(
                args.weather_dir,
                department=args.weather_department,
                force=args.force_weather_download,
            )

    daily, train, test = save_prepared_splits(
        args.raw_path,
        args.output_dir,
        args.test_days,
        args.input_days,
        weather_path=weather_path,
        weather_lag_months=weather_lag_months,
    )
    if args.output_dir.resolve() == (ROOT / "data").resolve():
        train.to_csv(ROOT / "train.csv", index=False)
        test.to_csv(ROOT / "test.csv", index=False)
    print(f"daily_power.csv: {len(daily)} rows")
    print(f"train.csv: {len(train)} rows")
    print(f"test.csv: {len(test)} rows, including {args.input_days} warm-up days")
    if args.weather_mode == "none":
        print("weather: disabled")
    else:
        print(f"weather: {args.weather_mode}, source={weather_path}")
        print("weather features: " + ", ".join(column for column in WEATHER_COLUMNS if column in daily.columns))


if __name__ == "__main__":
    main()
