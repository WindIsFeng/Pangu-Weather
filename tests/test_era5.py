import logging
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

from pangu_weather.era5 import ERA5Source, cds_request, read_fields


def test_coordinate_order_time_and_values(local_inputs, init_time):
    root, expected = local_inputs
    for kind in ("surface", "upper"):
        actual = read_fields(root / f"{kind}.nc", init_time, kind == "upper")
        np.testing.assert_array_equal(actual, expected[kind][0])
        later = read_fields(root / f"{kind}.nc", init_time + timedelta(hours=6), kind == "upper")
        np.testing.assert_array_equal(later, expected[kind][1])
        assert actual.dtype == np.float32
    assert not list(root.rglob("*.npy"))


@pytest.mark.parametrize("mutation,error", [("missing_level", "missing pressure level"),
    ("nan", "non-finite"), ("units", "expected units"), ("duplicate_lon", "grid"),
    ("ensemble", "ambiguous"), ("duplicate_time", "exactly one time")])
def test_bad_inputs(local_inputs, init_time, mutation, error):
    root, _ = local_inputs
    upper = mutation == "missing_level"
    path = root / ("upper.nc" if upper else "surface.nc")
    with xr.open_dataset(path) as source:
        ds = source.load()
    if mutation == "missing_level":
        ds = ds.isel(pressure_level=slice(1, None))
    elif mutation == "nan":
        ds["msl"].values[:] = np.nan
    elif mutation == "units":
        ds["msl"].attrs["units"] = "hPa"
    elif mutation == "duplicate_lon":
        ds = ds.assign_coords(longitude=[0, 0, 180, 270])
    elif mutation == "ensemble":
        ds = ds.expand_dims(number=[0, 1])
    else:
        ds = ds.assign_coords(time=[ds.time.values[0], ds.time.values[0]])
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match=error):
        read_fields(path, init_time, upper)


def test_source_cache_and_missing_time(config, init_time):
    source = ERA5Source(config, logging.getLogger("test"))
    first = source.get(init_time)
    assert source.get(init_time) is first
    with pytest.raises(ValueError, match="exactly one time"):
        source.get(init_time + timedelta(hours=12))


def test_download_request(init_time):
    request = cds_request(init_time, True)
    assert request["time"] == ["18:00"]
    assert request["day"] == ["31"]
    assert request["pressure_level"][0] == "1000"
    assert request["data_format"] == "netcdf"
