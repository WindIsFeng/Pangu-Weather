"""Resolve every configured relative path against the YAML file directory."""
from dataclasses import asdict, dataclass
from pathlib import Path

from .cases import positive_int, safe_id


@dataclass(frozen=True)
class Config:
    experiment: str
    model_dir: Path
    data_dir: Path
    output_dir: Path
    device: str = "cuda"
    device_id: int = 0
    threads: int = 1
    max_sessions: int = 1
    output_interval_hours: int = 6
    surface_file: str | None = None
    upper_file: str | None = None
    download_missing: bool = True

    def record(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def load_config(path: Path) -> Config:
    import yaml

    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    unknown = set(raw) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    for name, default in (("model_dir", "../models"), ("data_dir", "../data/era5"),
                          ("output_dir", "../outputs")):
        value = Path(raw.get(name, default)).expanduser()
        raw[name] = (path.parent / value).resolve()
    for name in ("surface_file", "upper_file"):
        if raw.get(name):
            value = Path(raw[name]).expanduser()
            raw[name] = str(path.parent / value) if not value.is_absolute() else str(value)
    if bool(raw.get("surface_file")) != bool(raw.get("upper_file")):
        raise ValueError("surface_file and upper_file must be configured together")
    raw["experiment"] = safe_id(str(raw.get("experiment", "default")), "experiment")
    for name, default in (("threads", 1), ("max_sessions", 1), ("output_interval_hours", 6)):
        raw[name] = positive_int(raw.get(name, default), name)
    if raw.get("device", "cuda") not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    device_id = raw.get("device_id", 0)
    if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
        raise ValueError("device_id must be a nonnegative integer")
    if not isinstance(raw.get("download_missing", True), bool):
        raise ValueError("download_missing must be true or false")
    return Config(**raw)
