"""Atomic records and streamed NetCDF products."""
import csv
import hashlib
import json
import os
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

from . import constants as C


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_csv(path: Path, rows, fields):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_identity(path: Path):
    """Hash content once per task invocation; never trust a filename as identity."""
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha.hexdigest()}


class ForecastWriter:
    def __init__(self, directory, case, forecast_id):
        self.directory, self.case, self.forecast_id = directory, case, forecast_id
        self.stack = ExitStack()
        self.datasets = {}
        self.index = 0

    def __enter__(self):
        import netCDF4 as nc
        try:
            for kind, names in (("surface", C.SURFACE), ("upper", C.UPPER)):
                ds = self.stack.enter_context(nc.Dataset(self.directory / f"{kind}.nc.partial", "w"))
                self.datasets[kind] = ds
                ds.Conventions = "CF-1.8"
                ds.forecast_id = self.forecast_id
                ds.status = "running"
                ds.createDimension("valid_time", len(self.case.leads))
                ds.createDimension("latitude", len(C.LATITUDES))
                ds.createDimension("longitude", len(C.LONGITUDES))
                for name, values, units in (("latitude", C.LATITUDES, "degrees_north"),
                                            ("longitude", C.LONGITUDES, "degrees_east")):
                    var = ds.createVariable(name, "f4", (name,))
                    var[:] = values
                    var.units = units
                    var.standard_name = name
                if kind == "upper":
                    ds.createDimension("level", len(C.LEVELS))
                    var = ds.createVariable("level", "i4", ("level",))
                    var[:] = C.LEVELS
                    var.units, var.standard_name, var.positive = "hPa", "air_pressure", "down"
                init = ds.createVariable("forecast_reference_time", "f8")
                init.units = "hours since 1970-01-01 00:00:00"
                init.calendar = "proleptic_gregorian"
                init.standard_name = "forecast_reference_time"
                init.assignValue(self.case.init_time.timestamp() / 3600)
                lead = ds.createVariable("lead_time", "i4", ("valid_time",))
                lead.units, lead.standard_name = "hours", "forecast_period"
                lead[:] = self.case.leads
                time = ds.createVariable("valid_time", "f8", ("valid_time",))
                time.units, time.calendar, time.standard_name = init.units, init.calendar, "time"
                time[:] = [(self.case.init_time + timedelta(hours=h)).timestamp() / 3600 for h in self.case.leads]
                dims = ("valid_time",) + (("level",) if kind == "upper" else ()) + ("latitude", "longitude")
                chunks = (1,) + ((1,) if kind == "upper" else ()) + (min(181, len(C.LATITUDES)), min(360, len(C.LONGITUDES)))
                for name in names:
                    var = ds.createVariable(name, "f4", dims, zlib=True, complevel=1, chunksizes=chunks)
                    var.units = C.UNITS[name]
                    var.coordinates = "forecast_reference_time lead_time"
            return self
        except BaseException:
            self.stack.close()
            raise

    def write(self, lead, state):
        from .era5 import validate_state
        validate_state(state)
        if self.index >= len(self.case.leads) or lead != self.case.leads[self.index]:
            raise ValueError(f"unexpected output lead: {lead}")
        upper, surface = state
        for kind, names, values in (("surface", C.SURFACE, surface), ("upper", C.UPPER, upper)):
            for i, name in enumerate(names):
                self.datasets[kind][name][self.index] = values[i]
        self.index += 1

    def __exit__(self, exc_type, exc, tb):
        complete = exc_type is None and self.index == len(self.case.leads)
        try:
            if complete:
                for ds in self.datasets.values():
                    ds.status = "complete"
        finally:
            self.stack.close()
        if complete:
            for kind in ("surface", "upper"):
                (self.directory / f"{kind}.nc.partial").replace(self.directory / f"{kind}.nc")
        elif exc_type is None:
            raise ValueError("forecast ended before all requested outputs were written")


def products_complete(directory, forecast_id, leads):
    import netCDF4 as nc
    import numpy as np
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("status") != "complete" or manifest.get("forecast_id") != forecast_id:
            return False
        for kind, names in (("surface", C.SURFACE), ("upper", C.UPPER)):
            path = directory / f"{kind}.nc"
            if path.stat().st_size != manifest["output_sizes"][kind]:
                return False
            with nc.Dataset(path) as ds:
                if ds.status != "complete" or ds.forecast_id != forecast_id:
                    return False
                if not np.array_equal(ds["lead_time"][:], leads):
                    return False
                for name in names:
                    if name not in ds.variables or ds[name].shape[0] != len(leads):
                        return False
        return True
    except (OSError, ValueError, KeyError, AttributeError, RuntimeError):
        return False


def link_products(case_dir, shared_dir):
    for name in ("surface.nc", "upper.nc"):
        target = case_dir / name
        temp = case_dir / (name + ".link")
        temp.unlink(missing_ok=True)
        temp.symlink_to(os.path.relpath(shared_dir / name, case_dir))
        temp.replace(target)
