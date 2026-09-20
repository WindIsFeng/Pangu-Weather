[English](README.md) | **简体中文**

# Pangu-Weather 台风事件批量预报

填写一份 case 清单，指定每个台风案例的起报时间和预报时长，运行一条命令，得到每个 case 的全球预测结果。
程序自动补齐缺失的 ERA5，直接读取 NetCDF，按原文贪婪算法推理并输出 NetCDF。没有 NPY 中间文件。

## 快速开始

在仓库根目录运行，依赖全部放在系统 conda 环境，不需要安装项目包：

```bash
conda env create -f environment.yml
conda run -n pangu python -m pangu_weather doctor --config configs/default.yaml --load-model 6
```

如果系统 `pangu` 环境已经创建，直接使用，无须重复安装。环境位置应为系统 conda 的 `envs/pangu`，
不要使用项目内的 `--prefix`、`.venv` 或 `pip --target`。当前 GPU 依赖使用 CUDA 12、cuDNN 9。
四个官方权重放在 `models/pangu_weather_{1,3,6,24}.onnx`，代码会按任务需要加载。

自动下载使用用户目录的 `.cdsapirc`，不把令牌写入仓库。需要先在 CDS 接受 ERA5
地面和气压层数据集的使用条款，参见 [CDS API 设置](https://cds.climate.copernicus.eu/how-to-api)。
只下载起报时刻的数据，不下载预报时段的未来 ERA5。

## 设置 case

参考 `configs/cases.example.csv`，复制为自己的清单：

```csv
case_id,storm_id,init_time,forecast_hours,output_interval_hours,name,basin
ragasa_01,ragasa_2025,2025-09-21T00:00:00Z,72,6,Ragasa,WP
ragasa_02,ragasa_2025,2025-09-22T00:00:00Z,120,6,Ragasa,WP
ragasa_03,ragasa_2025,2025-09-22T00:00:00Z,56,3,Ragasa,WP
```

- `case_id`：实验内唯一，使用英文字母、数字、下划线、点或连字符。
- `storm_id`：台风稳定标识。可以用最佳路径数据中的 ID，避免仅使用跨年重名的台风名称。
- `init_time`：带时区的整点时间，推荐写 UTC 的 `...T00:00:00Z`。
- `forecast_hours`：正整数小时，必填，不固定预报天数。
- `output_interval_hours`：正整数小时，列省略或留空时采用配置值，默认为 6。
- `name`、`basin`：可选信息，不影响全球预报计算。

时长不能整除输出间隔时，额外输出终点。例如 56 小时、间隔 6 小时，输出 6、12、…、54、56 小时。
一个 case 也使用同一 CSV 和运行命令。

## 运行

先查看时间范围、数据位置和模型调用，不下载、不加载模型、不写结果：

```bash
conda run -n pangu python -m pangu_weather run --config configs/default.yaml --cases configs/cases.example.csv --dry-run
```

确认清单后运行：

```bash
conda run --no-capture-output -n pangu python -m pangu_weather run --config configs/default.yaml --cases configs/cases.example.csv
```

`configs/default.yaml` 中的 `experiment` 决定实验名称。相对路径全部相对于配置文件所在目录。
默认在 GPU 0 顺序执行，避免并行进程争抢显存。24 GB GPU 默认仅缓存一个模型会话，切换模型时释放上一会话；更大显存可调整 `max_sessions`。CUDA 加载失败不会悄悄转为 CPU。
需要 CPU 时明确设置 `device: cpu`。

失败或中断后使用相同命令追加 `--resume`。完整且匹配的结果复用，未完成的预报从起报场重跑。
更改 case 清单、配置、代码、输入或模型时应使用新的实验名称；程序不默认覆盖旧结果。
退出码 0 表示全部完成或复用，1 表示存在失败或配置错误，130 表示用户中断。

## 使用已有 ERA5 NetCDF

在配置中同时指定地面和高空文件，可包含多个时间：

```yaml
surface_file: /data/era5/surface.nc
upper_file: /data/era5/upper.nc
download_missing: false
```

也可以使用模板，例如 `/data/era5/{init:%Y%m%dT%H}/surface.nc`。
若启用 `download_missing: true`，不存在的目标文件会自动下载；已有文件缺少所需时刻则报错，
不会覆盖已有多时间文件。程序会按实际坐标选时、排序和转置，并检查网格、气压层、单位及缺失值。
详细约定见 [设计说明](docs/design.md)。

## 读取结果

```text
outputs/<experiment>/
  cases.csv
  summary.csv
  batch.json
  batch.log
  cases/<case_id>/
    surface.nc
    upper.nc
    case.json
    run.log
  _forecasts/<fingerprint>/
    surface.nc
    upper.nc
    manifest.json
```

`summary.csv` 每行对应一个 case，记录 `complete`、`reused`、`failed` 等状态及错误原因。
个别任务失败不阻止其他任务。相同起报数据只下载一次，完全相同的全球预报只计算一次。
不同起报时间、预报时长和输出间隔分别记录结果。

```python
import xarray as xr

surface = xr.open_dataset("outputs/default/cases/ragasa_01/surface.nc", decode_timedelta=False)
upper = xr.open_dataset("outputs/default/cases/ragasa_01/upper.nc", decode_timedelta=False)
mslp_hpa = surface.msl / 100
z500 = upper.z.sel(level=500)  # z 是位势，单位 m²/s²
```

输出包含 `forecast_reference_time`、`lead_time`、`valid_time`、经纬度和真实气压层。
`lead_time` 单位为小时，初始场不混入预报结果。NetCDF 使用无损压缩并逐时写入。

**case 中的 NetCDF 是相对符号链接，迁移或交付时应保留整个实验目录，包括 `_forecasts/`。**
若只导出某个 case，可使用支持跟随链接的复制方式，同时保留其 `case.json` 和来源说明。

## 验证与旧代码

```bash
conda run -n pangu python -m pytest -q
```

GPU 数值对照测试不自动下载，需要先准备默认验证时刻的 ERA5，再显式运行：

```bash
PANGU_RUN_INTEGRATION=1 conda run --no-capture-output -n pangu python -m pytest tests/test_gpu_integration.py -q
```

本次真实 ERA5 批次及 GPU 对照已完成，详见 [验证记录](docs/validation.md)。

通过 `PANGU_TEST_CONFIG` 和 `PANGU_TEST_INIT` 可指定其他配置与起报时刻。
测试独立复现官方 24/6 小时迭代流程，逐输出比较新调度结果，并检查 3/1 小时模型执行。

旧脚本原样保存在 `legacy/`，不再作为入口。当前项目输出全球气象预报场；
台风中心追踪、最佳路径匹配与路径/强度误差计算是后续评估阶段的工作。

[论文](https://www.nature.com/articles/s41586-023-06185-3) · [官方实现](https://github.com/198808xc/Pangu-Weather)
