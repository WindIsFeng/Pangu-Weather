import json
import logging
import shutil
from dataclasses import replace

import pytest

from pangu_weather.__main__ import main
from pangu_weather.era5 import ERA5Source, cds_request
from pangu_weather.output import atomic_json, file_identity


def test_download_once_and_repair_incomplete_cache(config, init_time, local_inputs, monkeypatch):
    config = replace(config, surface_file=None, upper_file=None, download_missing=True)
    calls = []

    def download(path, init, upper, logger):
        calls.append(upper)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_inputs[0] / ("upper.nc" if upper else "surface.nc"), path)
        atomic_json(path.with_suffix(".json"), {"request": cds_request(init, upper), "file": file_identity(path)})

    monkeypatch.setattr("pangu_weather.era5._download", download)
    source = ERA5Source(config, logging.getLogger("test"))
    source.get(init_time)
    source.get(init_time)
    assert calls == [False, True]
    ERA5Source(config, logging.getLogger("test")).get(init_time)
    assert calls == [False, True]
    # A crash between renaming the downloaded NetCDF and writing its sidecar is repairable.
    next(config.data_dir.rglob("upper.json")).unlink()
    ERA5Source(config, logging.getLogger("test")).get(init_time)
    assert calls == [False, True, True]
    next(config.data_dir.rglob("surface.nc")).write_bytes(b"truncated")
    ERA5Source(config, logging.getLogger("test")).get(init_time)
    assert calls == [False, True, True, False]


def test_dry_run_has_no_side_effects(tmp_path, capsys):
    config = tmp_path / "settings.yaml"
    config.write_text("experiment: preview\noutput_dir: result\ndata_dir: cache\n")
    cases = tmp_path / "cases.csv"
    cases.write_text("case_id,storm_id,init_time,forecast_hours\na,s,2025-01-01T00:00:00Z,56\n")
    assert main(["run", "--config", str(config), "--cases", str(cases), "--dry-run"]) == 0
    assert not (tmp_path / "result").exists()
    assert not (tmp_path / "cache").exists()
    assert '"lead": 56' in capsys.readouterr().out


def test_invalid_cases_fail_before_any_outputs(tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text("experiment: invalid\noutput_dir: result\n")
    cases = tmp_path / "cases.csv"
    cases.write_text("case_id,storm_id,init_time,forecast_hours\na,s,2025-01-01T00:00:00Z,0\n")
    assert main(["run", "--config", str(config), "--cases", str(cases)]) == 1
    assert not (tmp_path / "result").exists()
