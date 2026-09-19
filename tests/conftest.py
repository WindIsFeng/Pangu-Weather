from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from pangu_weather import constants as C
from pangu_weather.config import Config


@pytest.fixture
def small_grid(monkeypatch):
    monkeypatch.setattr(C, "LATITUDES", (90.0, 0.0, -90.0))
    monkeypatch.setattr(C, "LONGITUDES", (0.0, 90.0, 180.0, 270.0))


@pytest.fixture
def init_time():
    return datetime(2025, 12, 31, 18, tzinfo=timezone.utc)


@pytest.fixture
def local_inputs(tmp_path, small_grid):
    """Sentinels encode time, variable, level and spatial position independently."""
    times = np.array(["2025-12-31T18:00", "2026-01-01T00:00"], dtype="datetime64[ns]")
    expected = {}
    for kind, names in (("surface", C.SURFACE), ("upper", C.UPPER)):
        shape = (2, len(names)) + ((len(C.LEVELS),) if kind == "upper" else ()) + (3, 4)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        expected[kind] = values
        coords = {"valid_time": times, "latitude": list(C.LATITUDES), "longitude": list(C.LONGITUDES)}
        dims = ("valid_time",) + (("level",) if kind == "upper" else ()) + ("latitude", "longitude")
        if kind == "upper":
            coords["level"] = list(C.LEVELS)
        ds = xr.Dataset({name: (dims, values[:, i], {"units": C.UNITS[name]}) for i, name in enumerate(names)}, coords=coords)
        if kind == "upper":
            ds.level.attrs["units"] = "hPa"
            ds = ds.sortby("level").rename(level="pressure_level")
        ds = ds.sortby("latitude").assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
        ds = ds.rename(valid_time="time").transpose(..., "longitude", "latitude")
        ds.to_netcdf(tmp_path / f"{kind}.nc", engine="netcdf4")
    return tmp_path, expected


@pytest.fixture
def config(tmp_path, local_inputs):
    root, _ = local_inputs
    models = tmp_path / "models"
    models.mkdir()
    for step in (1, 3, 6, 24):
        (models / f"pangu_weather_{step}.onnx").write_bytes(f"fake model {step}".encode())
    return Config("test", models, tmp_path / "cache", tmp_path / "outputs", device="cpu",
                  surface_file=str(root / "surface.nc"), upper_file=str(root / "upper.nc"), download_missing=False)
