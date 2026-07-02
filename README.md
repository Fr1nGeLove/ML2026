# ML2026 Household Power Forecasting

本仓库用于对 UCI Individual Household Electric Power Consumption 数据集进行日级用电量预测。

仓库包含可运行代码、测试、环境配置和小型处理后 CSV 数据。

## 内容

```text
src/power_forecast/   数据处理、模型、训练、实验和绘图代码
scripts/              命令行入口
tests/                单元测试
data/                 已处理的小型 daily/train/test CSV
```

主要模型包括：

- LSTM
- Transformer
- PatchChannelMixer
- last value、moving average、weekly seasonal naive 三个简单基线

## 环境

推荐使用 Python 3.11。

```powershell
conda create -n ml_power python=3.11 -y
conda activate ml_power
pip install -r requirements.txt
```

如果需要 GPU 版 PyTorch，请根据本机 CUDA 版本从 PyTorch 官网选择安装命令。例如：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

也可以用 Conda 环境文件：

```powershell
conda env create -f environment.yml
conda activate ml_power
```

## 数据

仓库内已包含处理后的：

- `data/daily_power.csv`
- `data/train.csv`
- `data/test.csv`

如果要从原始数据重新生成，请下载 UCI 原始文件 `household_power_consumption.txt` 并放到仓库根目录：

[UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

默认预处理脚本会尝试下载 Météo-France 月度气候数据并缓存到 `data/weather/`：

[Météo-France monthly climatological data](https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles)

项目提供的是月度气候数据，不需要逐日天气数据。预处理时会把同一个月的 `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU` 映射到该月每天；其中 `RR` 除以 10。

## 当前发布版说明

当前版本使用 5 个 Météo-France 月度气候字段：

- `RR`：月降水量，预处理时除以 10 后作为 `weather_rain_mm`
- `NBJRR1`：月内日降水量大于等于 1mm 的天数
- `NBJRR5`：月内日降水量大于等于 5mm 的天数
- `NBJRR10`：月内日降水量大于等于 10mm 的天数
- `NBJBROU`：月内有雾天数

发布仓库中的 `data/daily_power.csv`、`data/train.csv`、`data/test.csv` 已由上述逻辑生成。

读者也可以下载原始 UCI 数据和月度气候数据后，用下面的命令重新生成。

## 常用命令

重新准备数据：

```powershell
python scripts\prepare_data.py
```

不使用气象特征重新准备数据：

```powershell
python scripts\prepare_data.py --weather-mode none
```

如需使用上一月气候统计以避免目标月份信息进入输入窗口，可以运行：

```powershell
python scripts\prepare_data.py --weather-mode lagged
```

快速跑通实验：

```powershell
python scripts\run_experiments.py --preset smoke --models last_value moving_average weekly_seasonal_naive lstm transformer patch_channel_mixer --horizons 90 --seeds 0
```

完整的实验：

```powershell
python scripts\run_experiments.py --preset full --models last_value moving_average weekly_seasonal_naive lstm transformer patch_channel_mixer --horizons 90 365 --seeds 0 1 2 3 4
```

实验指标和预测结果默认写入 `results/`，训练检查点写入 `models/`。

生成结果图：

```powershell
python scripts\make_plots.py
```

运行测试：

```powershell
python -m pytest -q
```

发布前验证过的最小复现流程：

```powershell
python scripts\prepare_data.py
python -m pytest -q
python scripts\run_experiments.py --preset smoke --models last_value moving_average weekly_seasonal_naive lstm transformer patch_channel_mixer --horizons 90 --seeds 0
```

## 说明

`data/test.csv` 前 90 天是为了构造第一个测试窗口而保留的历史输入，不作为测试标签参与指标计算。训练、验证和测试按时间顺序划分，标准化参数只由训练集拟合。
