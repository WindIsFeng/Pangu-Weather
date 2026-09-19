"""The CSV is the only user-maintained task inventory."""
import csv
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


def parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("init_time must include a timezone (use ...T00:00:00Z)")
    dt = dt.astimezone(timezone.utc)
    if dt.minute or dt.second or dt.microsecond:
        raise ValueError("init_time must be an exact hour")
    return dt


def positive_int(value, name):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def safe_id(value: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{name} must use letters, digits, underscores, dots or hyphens")
    return value


@dataclass(frozen=True)
class Case:
    case_id: str
    storm_id: str
    init_time: datetime
    forecast_hours: int
    output_interval_hours: int = 6
    name: str = ""
    basin: str = ""

    @property
    def leads(self):
        return sorted(set(range(self.output_interval_hours, self.forecast_hours + 1,
                                self.output_interval_hours)) | {self.forecast_hours})

    def record(self):
        result = asdict(self)
        result["init_time"] = self.init_time.isoformat().replace("+00:00", "Z")
        return result


def read_cases(path: Path, default_interval: int = 6) -> list[Case]:
    required = {"case_id", "storm_id", "init_time", "forecast_hours"}
    allowed = required | {"output_interval_hours", "name", "basin"}
    cases, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not required <= set(fields) or set(fields) - allowed:
            raise ValueError(f"CSV columns must include {sorted(required)}; optional: {sorted(allowed-required)}")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("wrong number of CSV fields")
                row = {k: v.strip() for k, v in row.items()}
                case_id = safe_id(row["case_id"], "case_id")
                if case_id in seen:
                    raise ValueError(f"duplicate case_id: {case_id}")
                if not row["storm_id"]:
                    raise ValueError("storm_id cannot be empty")
                case = Case(case_id, row["storm_id"], parse_time(row["init_time"]),
                            positive_int(row["forecast_hours"], "forecast_hours"),
                            positive_int(row.get("output_interval_hours") or default_interval,
                                         "output_interval_hours"), row.get("name", ""), row.get("basin", ""))
                cases.append(case)
                seen.add(case_id)
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
    if not cases:
        raise ValueError("case list is empty")
    return cases
