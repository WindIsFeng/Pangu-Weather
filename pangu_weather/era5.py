"""Read ERA5 directly from NetCDF; NumPy exists only in memory."""
import json
from pathlib import Path

from . import constants as C
from .output import atomic_json, file_identity


def validate_state(state):
    import numpy as np
    shapes = ((len(C.UPPER), len(C.LEVELS), len(C.LATITUDES), len(C.LONGITUDES)),
              (len(C.SURFACE), len(C.LATITUDES), len(C.LONGITUDES)))
    for name, values, shape in zip(("upper", "surface"), state, shapes):
        if values.shape != shape or values.dtype != np.float32:
            raise ValueError(f"{name}: expected float32 {shape}, got {values.dtype} {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: missing or non-finite values")


def _unit_key(value):
    return str(value).lower().replace(" ", "").replace("**", "").replace("^", "")


def _check_units(name, value):
    accepted = {"Pa": {"pa", "pascal", "pascals"}, "K": {"k", "kelvin"},
                "m s-1": {"ms-1", "m/s"}, "m2 s-2": {"m2s-2", "m2/s2"},
                "kg kg-1": {"kgkg-1", "kg/kg", "1"}}
    if _unit_key(value) not in accepted[C.UNITS[name]]:
        raise ValueError(f"{name}: expected units {C.UNITS[name]}, got {value!r}")


def read_fields(path: Path, init_time, upper: bool):
    import numpy as np
    import xarray as xr
    with xr.open_dataset(path, engine="netcdf4") as source:
        ds = source
        for target, aliases in {"valid_time": ("time",), "level": ("pressure_level",),
                                 "latitude": ("lat",), "longitude": ("lon",)}.items():
            if target not in ds.coords:
                found = [a for a in aliases if a in ds.coords]
                if found:
                    ds = ds.rename({found[0]: target})
        if "valid_time" not in ds.coords:
            raise ValueError(f"{path}: missing time coordinate")
        time = ds["valid_time"]
        wanted = np.datetime64(init_time.replace(tzinfo=None), "ns")
        values = np.asarray(time.values).astype("datetime64[ns]")
        matches = np.flatnonzero(values.reshape(-1) == wanted)
        if matches.size != 1:
            raise ValueError(f"{path}: expected exactly one time {init_time.isoformat()}, found {matches.size}")
        if time.ndim == 1:
            ds = ds.isel({time.dims[0]: int(matches[0])})
        elif time.ndim != 0:
            raise ValueError("multidimensional time coordinates are unsupported")
        for name, expected in (("latitude", C.LATITUDES), ("longitude", C.LONGITUDES)):
            if name not in ds.coords or ds[name].dims != (name,):
                raise ValueError(f"{path}: missing or non-rectilinear {name}")
            actual = np.asarray(ds[name].values, dtype=float)
            if name == "longitude":
                actual = actual % 360
            order = np.argsort(actual)
            if name == "latitude":
                order = order[::-1]
            actual = actual[order]
            if actual.shape != (len(expected),) or not np.allclose(actual, expected, rtol=0, atol=1e-5):
                raise ValueError(f"{path}: {name} must cover the complete 0.25-degree Pangu grid without duplicates")
            ds = ds.isel({name: order}).assign_coords({name: actual})
        if upper:
            if "level" not in ds.coords or ds.level.dims != ("level",):
                raise ValueError(f"{path}: missing pressure levels")
            levels = np.asarray(ds.level.values, dtype=float)
            unit = _unit_key(ds.level.attrs.get("units", ""))
            if unit == "pa":
                levels = levels / 100
            elif unit not in ("hpa", "millibars", "millibar", "mbar"):
                raise ValueError(f"{path}: pressure level units must be hPa or Pa")
            if len(np.unique(levels)) != len(levels):
                raise ValueError("duplicate pressure levels")
            indices = []
            for level in C.LEVELS:
                found = np.flatnonzero(np.isclose(levels, level, rtol=0, atol=1e-5))
                if len(found) != 1:
                    raise ValueError(f"missing pressure level {level} hPa")
                indices.append(int(found[0]))
            ds = ds.isel(level=indices)
        dims = (("level",) if upper else ()) + ("latitude", "longitude")
        names = C.UPPER if upper else C.SURFACE
        result = np.empty((len(names),) + tuple(ds.sizes[d] for d in dims), dtype=np.float32)
        for i, name in enumerate(names):
            if name not in ds:
                raise ValueError(f"{path}: missing variable {name}")
            var = ds[name]
            _check_units(name, var.attrs.get("units"))
            for dim in set(var.dims) - set(dims):
                if var.sizes[dim] != 1:
                    raise ValueError(f"{name}: ambiguous additional dimension {dim}")
                var = var.isel({dim: 0})
            if set(var.dims) != set(dims):
                raise ValueError(f"{name}: expected dimensions {dims}, got {var.dims}")
            result[i] = var.transpose(*dims).values
            if not np.isfinite(result[i]).all():
                raise ValueError(f"{path}: {name} contains missing or non-finite values")
        return result


def source_paths(config, init_time):
    if config.surface_file:
        return tuple(Path(template.format(init=init_time)) for template in (config.surface_file, config.upper_file))
    base = config.data_dir / init_time.strftime("%Y%m%dT%H%M%SZ")
    return base / "surface.nc", base / "upper.nc"


def cds_request(init_time, upper):
    request = {"product_type": ["reanalysis"], "variable": list(C.CDS_UPPER if upper else C.CDS_SURFACE),
               "year": [init_time.strftime("%Y")], "month": [init_time.strftime("%m")],
               "day": [init_time.strftime("%d")], "time": [init_time.strftime("%H:00")],
               "data_format": "netcdf", "download_format": "unarchived", "grid": [0.25, 0.25]}
    if upper:
        request["pressure_level"] = [str(level) for level in C.LEVELS]
    return request


def _download(path, init_time, upper, logger):
    import cdsapi
    request = cds_request(init_time, upper)
    dataset = "reanalysis-era5-pressure-levels" if upper else "reanalysis-era5-single-levels"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".download")
    logger.info("Downloading %s for %s", dataset, init_time.isoformat())
    # Client already retries transient network failures; do not endlessly resubmit rejected requests.
    client = cdsapi.Client(timeout=120, retry_max=3, sleep_max=20, quiet=True, progress=False)
    try:
        client.retrieve(dataset, request, str(temp))
        read_fields(temp, init_time, upper)  # Never mark a partial/invalid response as cached input.
        temp.replace(path)
        atomic_json(path.with_suffix(".json"), {"dataset": dataset, "request": request,
                                              "file": file_identity(path)})
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


class ERA5Source:
    """One initial state in RAM; file hashes and failures cached for this invocation."""
    def __init__(self, config, logger):
        self.config, self.logger = config, logger
        self.cached_time = None
        self.cached_value = None
        self.hashes = {}
        self.failures = {}

    def get(self, init_time):
        if init_time in self.failures:
            raise RuntimeError(self.failures[init_time])
        if init_time == self.cached_time:
            return self.cached_value
        self.cached_time, self.cached_value = None, None
        try:
            paths = source_paths(self.config, init_time)
            identities, arrays = [], []
            for upper, path in zip((False, True), paths):
                if not path.exists():
                    if not self.config.download_missing:
                        raise FileNotFoundError(f"missing ERA5 input: {path}")
                    _download(path, init_time, upper, self.logger)
                stat = path.stat()
                hash_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
                if hash_key not in self.hashes:
                    self.hashes[hash_key] = file_identity(path)
                identity = self.hashes[hash_key]
                sidecar = path.with_suffix(".json")
                # Files explicitly supplied by the user need no sidecar. Managed cache does.
                if not self.config.surface_file:
                    try:
                        meta = json.loads(sidecar.read_text())
                        valid_cache = (meta.get("request") == cds_request(init_time, upper)
                                       and meta.get("file", {}).get("sha256") == identity["sha256"])
                    except (OSError, ValueError):
                        valid_cache = False
                    if not valid_cache:
                        if not self.config.download_missing:
                            raise ValueError(f"ERA5 cache metadata mismatch: {path}")
                        self.logger.info("Replacing incomplete or mismatched ERA5 cache: %s", path)
                        _download(path, init_time, upper, self.logger)
                        identity = file_identity(path)
                        stat = path.stat()
                        self.hashes[(str(path.resolve()), stat.st_size, stat.st_mtime_ns)] = identity
                identities.append(identity)
                arrays.append(read_fields(path, init_time, upper))
            state = (arrays[1], arrays[0])
            validate_state(state)
            self.cached_time = init_time
            self.cached_value = (state, {"surface": identities[0], "upper": identities[1]})
            return self.cached_value
        except Exception as exc:
            self.failures[init_time] = str(exc)
            raise
