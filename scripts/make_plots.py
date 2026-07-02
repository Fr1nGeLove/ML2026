from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast.plotting import make_all_plots


if __name__ == "__main__":
    for path in make_all_plots(ROOT / "results", ROOT / "figures"):
        print(path)
