import csv
import json
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

from pangu_weather.cases import Case
from pangu_weather.pipeline import run_batch


class FakeRuntime:
    calls = []
    fail_step = None
    actual_providers = {"test": ["FakeExecutionProvider"]}

    def __init__(self, config, logger):
        pass

    def run(self, step, state):
        self.calls.append(step)
        if step == self.fail_step:
            raise RuntimeError("injected inference failure")
        # Order-sensitive transformation, unlike adding the lead time.
        return tuple(np.asarray(values * 2 + step, dtype=np.float32) for values in state)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def clear_runtime():
    FakeRuntime.calls = []
    FakeRuntime.fail_step = None


def read_summary(config):
    with (config.output_dir / config.experiment / "summary.csv").open() as stream:
        return list(csv.DictReader(stream))


def test_batch_reuse_time_mapping_and_resume(config, init_time, local_inputs):
    cases = [Case("a", "storm1", init_time, 30), Case("b", "storm2", init_time, 30),
             Case("c", "storm1", init_time + timedelta(hours=6), 5, 3)]
    assert run_batch(config, cases, runtime_factory=FakeRuntime) == 0
    assert FakeRuntime.calls == [6, 6, 6, 24, 6, 3, 1, 1]
    rows = read_summary(config)
    assert [r["status"] for r in rows] == ["complete", "reused", "complete"]
    root = config.output_dir / config.experiment
    assert (root / "cases/a/surface.nc").resolve() == (root / "cases/b/surface.nc").resolve()
    with xr.open_dataset(root / "cases/a/surface.nc", decode_timedelta=True) as ds:
        np.testing.assert_array_equal(ds.lead_time.values / np.timedelta64(1, "h"), [6, 12, 18, 24, 30])
        assert ds.valid_time.values[-1] == np.datetime64("2026-01-02T00:00")
        initial = local_inputs[1]["surface"][0, 0]
        np.testing.assert_array_equal(ds.msl.isel(valid_time=3).values, initial * 2 + 24)
        np.testing.assert_array_equal(ds.msl.isel(valid_time=4).values, (initial * 2 + 24) * 2 + 6)
        assert ds.msl.attrs["units"] == "Pa"
    with xr.open_dataset(root / "cases/c/upper.nc", decode_timedelta=True) as ds:
        assert list(ds.level.values) == [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        assert ds.level.attrs["units"] == "hPa"
        assert ds.valid_time.values[-1] == np.datetime64("2026-01-01T05:00")
    FakeRuntime.calls.clear()
    assert run_batch(config, cases, resume=True, runtime_factory=FakeRuntime) == 0
    assert not FakeRuntime.calls
    assert all(r["status"] == "reused" for r in read_summary(config))
    assert not list(root.rglob("*.npy"))
    with pytest.raises(FileExistsError):
        run_batch(config, cases, runtime_factory=FakeRuntime)


def test_failure_isolation_and_resume(config, init_time):
    cases = [Case("bad", "s", init_time, 24), Case("bad_shared", "s2", init_time, 24),
             Case("good", "s", init_time, 6)]
    FakeRuntime.fail_step = 24
    assert run_batch(config, cases, runtime_factory=FakeRuntime) == 1
    assert FakeRuntime.calls == [6, 6, 6, 24, 6]
    assert [r["status"] for r in read_summary(config)] == ["failed", "failed", "complete"]
    FakeRuntime.fail_step = None
    FakeRuntime.calls.clear()
    assert run_batch(config, cases, resume=True, runtime_factory=FakeRuntime) == 0
    assert FakeRuntime.calls == [6, 6, 6, 24]
    assert [r["status"] for r in read_summary(config)] == ["complete", "reused", "reused"]


def test_missing_time_does_not_abort_batch(config, init_time):
    cases = [Case("good", "s", init_time, 6), Case("bad", "s", init_time + timedelta(hours=12), 6)]
    assert run_batch(config, cases, runtime_factory=FakeRuntime) == 1
    assert [r["status"] for r in read_summary(config)] == ["complete", "failed"]


def test_truncated_product_restarts(config, init_time):
    case = Case("a", "s", init_time, 6)
    run_batch(config, [case], runtime_factory=FakeRuntime)
    path = config.output_dir / config.experiment / "cases/a/surface.nc"
    path.resolve().write_bytes(b"truncated")
    FakeRuntime.calls.clear()
    assert run_batch(config, [case], resume=True, runtime_factory=FakeRuntime) == 0
    assert FakeRuntime.calls == [6]


def test_changed_configuration_or_model_is_not_reused(config, init_time):
    case = Case("a", "s", init_time, 6)
    run_batch(config, [case], runtime_factory=FakeRuntime)
    with pytest.raises(ValueError, match="changed"):
        run_batch(replace(config, threads=2), [case], resume=True, runtime_factory=FakeRuntime)
    (config.model_dir / "pangu_weather_6.onnx").write_bytes(b"new model content")
    FakeRuntime.calls.clear()
    assert run_batch(config, [case], resume=True, runtime_factory=FakeRuntime) == 1
    assert not FakeRuntime.calls
    assert "identity changed" in read_summary(config)[0]["error"]
