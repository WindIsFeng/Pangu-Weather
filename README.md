**English** | [简体中文](README.zh-CN.md)

# Pangu-Weather Batch Forecasting for Typhoon Events

Create a case list that specifies the initialization time and forecast duration for each typhoon case, then run a single command to generate global forecasts for every case.
The program automatically retrieves missing ERA5 data, reads NetCDF files directly, runs inference with the greedy algorithm described in the paper, and writes NetCDF output. No intermediate NPY files are created.

## Quick start

Run the following commands from the repository root. All dependencies are installed in a system Conda environment; the project itself does not need to be installed as a package:

```bash
conda env create -f environment.yml
conda run -n pangu python -m pangu_weather doctor --config configs/default.yaml --load-model 6
```

If the system `pangu` environment already exists, use it directly instead of reinstalling it. The environment should be located under the system Conda installation at `envs/pangu`.
Do not use a project-local `--prefix`, `.venv`, or `pip --target`. The current GPU dependencies use CUDA 12 and cuDNN 9.
Place the four official model files at `models/pangu_weather_{1,3,6,24}.onnx`; the program loads them as required by each task.

Automatic downloads use `.cdsapirc` in the user's home directory, so credentials are never written to the repository. Before downloading data, accept the terms for the ERA5 surface and pressure-level datasets in CDS; see [CDS API setup](https://cds.climate.copernicus.eu/how-to-api).
Only data at the initialization time is downloaded, not future ERA5 data covering the forecast period.

## Define cases

Use `configs/cases.example.csv` as a template and copy it to your own case list:

```csv
case_id,storm_id,init_time,forecast_hours,output_interval_hours,name,basin
ragasa_01,ragasa_2025,2025-09-21T00:00:00Z,72,6,Ragasa,WP
ragasa_02,ragasa_2025,2025-09-22T00:00:00Z,120,6,Ragasa,WP
ragasa_03,ragasa_2025,2025-09-22T00:00:00Z,56,3,Ragasa,WP
```

- `case_id`: Unique within an experiment. Use letters, digits, underscores, periods, or hyphens.
- `storm_id`: A stable typhoon identifier. Prefer an ID from best-track data instead of a storm name that may be reused in another year.
- `init_time`: An hour-aligned timestamp with a time zone. UTC in the form `...T00:00:00Z` is recommended.
- `forecast_hours`: Required positive integer number of hours. The forecast duration is not fixed.
- `output_interval_hours`: Positive integer number of hours. If the column is omitted or empty, the configured value is used; the default is 6.
- `name`, `basin`: Optional metadata that does not affect the global forecast calculation.

If the forecast duration is not divisible by the output interval, the final time is emitted as an additional output. For example, a 56-hour forecast with a 6-hour interval produces output at 6, 12, ..., 54, and 56 hours.
A single case uses the same CSV format and run command.

## Run forecasts

First inspect the time range, data locations, and model invocations without downloading data, loading models, or writing results:

```bash
conda run -n pangu python -m pangu_weather run --config configs/default.yaml --cases configs/cases.example.csv --dry-run
```

After reviewing the case list, run:

```bash
conda run --no-capture-output -n pangu python -m pangu_weather run --config configs/default.yaml --cases configs/cases.example.csv
```

The `experiment` field in `configs/default.yaml` determines the experiment name. All relative paths are resolved against the directory containing the configuration file.
By default, tasks run sequentially on GPU 0 to avoid competing for GPU memory. On a 24 GB GPU, only one model session is cached by default, and the previous session is released when switching models; increase `max_sessions` on GPUs with more memory. A CUDA loading failure never silently falls back to CPU.
To use the CPU, explicitly set `device: cpu`.

After a failure or interruption, rerun the same command with `--resume`. Complete, matching results are reused; incomplete forecasts restart from the initial field.
Use a new experiment name after changing the case list, configuration, code, inputs, or model files. Existing results are not overwritten by default.
Exit code 0 means all tasks completed or were reused, 1 means at least one task failed or the configuration was invalid, and 130 means the user interrupted the run.

## Use existing ERA5 NetCDF files

Specify both surface and upper-air files in the configuration. Each file may contain multiple times:

```yaml
surface_file: /data/era5/surface.nc
upper_file: /data/era5/upper.nc
download_missing: false
```

Templates such as `/data/era5/{init:%Y%m%dT%H}/surface.nc` are also supported.
With `download_missing: true`, missing target files are downloaded automatically. If an existing file does not contain the requested time, the program reports an error rather than overwriting a multi-time file.
Coordinates are selected, sorted, and transposed from their actual values, and the grid, pressure levels, units, and missing values are validated.
See the [design notes](docs/design.md) for detailed conventions.

## Read the results

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

Each row in `summary.csv` represents one case and records statuses such as `complete`, `reused`, or `failed`, together with any error message.
One failed task does not stop the others. Identical initialization data is downloaded only once, and identical global forecasts are computed only once.
Results for different initialization times, forecast durations, and output intervals are tracked separately.

```python
import xarray as xr

surface = xr.open_dataset("outputs/default/cases/ragasa_01/surface.nc", decode_timedelta=False)
upper = xr.open_dataset("outputs/default/cases/ragasa_01/upper.nc", decode_timedelta=False)
mslp_hpa = surface.msl / 100
z500 = upper.z.sel(level=500)  # z is geopotential in m²/s²
```

Output includes `forecast_reference_time`, `lead_time`, `valid_time`, latitude, longitude, and physical pressure levels.
`lead_time` is measured in hours, and the initial field is not mixed into the forecast output. NetCDF files use lossless compression and are written one time step at a time.

**The NetCDF files under each case are relative symbolic links. When moving or delivering results, preserve the entire experiment directory, including `_forecasts/`.**
To export only one case, use a copy method that follows symbolic links, and retain its `case.json` and provenance information.

## Validation and legacy code

```bash
conda run -n pangu python -m pytest -q
```

The GPU numerical comparison test does not download data automatically. Prepare ERA5 data for the default validation time, then run it explicitly:

```bash
PANGU_RUN_INTEGRATION=1 conda run --no-capture-output -n pangu python -m pytest tests/test_gpu_integration.py -q
```

The real ERA5 batch and GPU comparison for this version have been completed; see the [validation record](docs/validation.md).

Use `PANGU_TEST_CONFIG` and `PANGU_TEST_INIT` to select another configuration and initialization time.
The test independently reproduces the official 24/6-hour iteration procedure, compares every output with the new scheduler results, and verifies execution of the 3/1-hour models.

The original scripts are preserved unchanged in `legacy/` and are no longer entry points. The current project produces global weather forecast fields.
Typhoon-center tracking, best-track matching, and track/intensity error calculations belong to a later evaluation stage.

[Paper](https://www.nature.com/articles/s41586-023-06185-3) · [Official implementation](https://github.com/198808xc/Pangu-Weather)
