"""One case inventory, one entry point, failure isolation and shared products."""
import fcntl
import importlib.metadata
import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .era5 import ERA5Source, source_paths
from .output import (ForecastWriter, atomic_csv, atomic_json, digest, file_identity,
                     link_products, products_complete)
from .runtime import Runtime
from .schedule import build_schedule, greedy_steps, predict


def now():
    return datetime.now(timezone.utc).isoformat()


def code_identity():
    return digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})


@contextmanager
def experiment_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process is using experiment {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def describe(config, cases):
    tasks = {}
    for case in cases:
        key = (case.init_time, case.forecast_hours, case.output_interval_hours)
        tasks.setdefault(key, []).append(case.case_id)
    print(f"Cases: {len(cases)}; unique forecasts: {len(tasks)}; initial times: {len({c.init_time for c in cases})}")
    print(f"Results: {config.output_dir / config.experiment}/cases/<case_id>/")
    for (init, hours, interval), ids in tasks.items():
        sample = next(c for c in cases if c.case_id == ids[0])
        schedule = build_schedule(sample.leads)
        print(json.dumps({"cases": ids, "init_time": init.isoformat(), "forecast_hours": hours,
                          "output_interval_hours": interval, "leads": sample.leads,
                          "input_files": [str(p) for p in source_paths(config, init)],
                          "steps": [{"lead": n.lead, "parent": n.parent, "model_hours": n.step} for n in schedule],
                          "required_models": [str(config.model_dir / f"pangu_weather_{s}.onnx")
                                              for s in sorted({n.step for n in schedule})]}, ensure_ascii=False))


def _logger(root):
    logger = logging.getLogger(f"pangu.{id(root)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(root / "batch.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def run_batch(config, cases, resume=False, source_factory=ERA5Source, runtime_factory=Runtime):
    root = config.output_dir / config.experiment
    with experiment_lock(root):
        snapshot = [c.record() for c in cases]
        definition = {"cases": snapshot, "config": config.record(), "code_sha256": code_identity()}
        batch_path = root / "batch.json"
        if batch_path.exists():
            if not resume:
                raise FileExistsError(f"experiment already exists: {root}; use --resume or a new experiment name")
            if json.loads(batch_path.read_text()) != definition:
                raise ValueError("case list, configuration or code changed; use a new experiment name")
        else:
            if (root / "cases").exists() or (root / "_forecasts").exists():
                raise ValueError("existing output has no batch definition; use a new experiment name")
            atomic_json(batch_path, definition)
            atomic_csv(root / "cases.csv", snapshot, list(snapshot[0]))
        logger = _logger(root)
        runtime = None
        try:
            source = source_factory(config, logger)
            package_versions = {}
            for package in ("numpy", "xarray", "netCDF4", "onnxruntime-gpu"):
                try:
                    package_versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    package_versions[package] = "unavailable"
            model_hashes = {}
            completed, failed = set(), {}
            rows = {c.case_id: {**c.record(), "status": "pending", "forecast_id": "",
                                "result_dir": f"cases/{c.case_id}", "error": ""} for c in cases}
            fields = list(next(iter(rows.values())))

            def summary():
                atomic_csv(root / "summary.csv", list(rows.values()), fields)

            summary()
            # Group by initial time to reuse the in-memory ERA5 state. Summary retains CSV order.
            for case in sorted(cases, key=lambda c: c.init_time):
                case_dir = root / "cases" / case.case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(case_dir / "run.log", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                logger.addHandler(handler)
                rows[case.case_id]["status"] = "running"
                summary()
                started = time.monotonic()
                shared_dir, forecast_id, owns_shared = None, "", False
                logger.info("Case %s: %s, %sh", case.case_id, case.init_time.isoformat(), case.forecast_hours)
                try:
                    needed = sorted({n.step for n in build_schedule(case.leads)})
                    for step in needed:
                        if step not in model_hashes:
                            path = config.model_dir / f"pangu_weather_{step}.onnx"
                            logger.info("Checking model %s", path.name)
                            model_hashes[step] = file_identity(path)
                    state, source_identity = source.get(case.init_time)
                    identity = {"init_time": case.record()["init_time"], "forecast_hours": case.forecast_hours,
                                "output_interval_hours": case.output_interval_hours, "leads": case.leads,
                                "source": source_identity, "models": {str(s): model_hashes[s] for s in needed},
                                "code_sha256": definition["code_sha256"], "packages": package_versions,
                                "device": config.device, "device_id": config.device_id, "threads": config.threads}
                    forecast_id = digest(identity)
                    shared_dir = root / "_forecasts" / forecast_id
                    record_path = case_dir / "case.json"
                    if record_path.exists():
                        previous = json.loads(record_path.read_text())
                        if previous.get("forecast_id") and previous["forecast_id"] != forecast_id:
                            raise ValueError("input or model identity changed; use a new experiment name")
                    if forecast_id in failed:
                        raise RuntimeError(f"shared forecast failed: {failed[forecast_id]}")
                    reused = forecast_id in completed or (resume and products_complete(shared_dir, forecast_id, case.leads))
                    if not reused:
                        owns_shared = True
                        shared_dir.mkdir(parents=True, exist_ok=True)
                        manifest = {"forecast_id": forecast_id, "status": "running", "started_at": now(),
                                    "identity": identity, "paths": {str(h): list(greedy_steps(h)) for h in case.leads}}
                        atomic_json(shared_dir / "manifest.json", manifest)
                        if runtime is None:
                            runtime = runtime_factory(config, logger)
                        with ForecastWriter(shared_dir, case, forecast_id) as writer:
                            for lead, predicted in predict(state, case.leads, runtime):
                                writer.write(lead, predicted)
                                logger.info("Wrote lead %sh", lead)
                                del predicted
                        manifest.update(status="complete", completed_at=now(),
                                        providers=runtime.actual_providers,
                                        output_sizes={kind: (shared_dir / f"{kind}.nc").stat().st_size
                                                      for kind in ("surface", "upper")})
                        atomic_json(shared_dir / "manifest.json", manifest)
                        owns_shared = False
                    completed.add(forecast_id)
                    link_products(case_dir, shared_dir)
                    status = "reused" if reused else "complete"
                    atomic_json(record_path, {**case.record(), "status": status, "forecast_id": forecast_id,
                                              "source": source_identity, "updated_at": now(),
                                              "manifest": f"../../_forecasts/{forecast_id}/manifest.json"})
                    rows[case.case_id].update(status=status, forecast_id=forecast_id)
                    logger.info("Case %s %s in %.1fs", case.case_id, status, time.monotonic()-started)
                    del state
                except BaseException as exc:
                    if not isinstance(exc, (Exception, KeyboardInterrupt)):
                        raise
                    status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    if owns_shared:
                        failed[forecast_id] = error
                        atomic_json(shared_dir / "manifest.json", {"forecast_id": forecast_id,
                                                                  "status": status, "error": error, "updated_at": now()})
                    rows[case.case_id].update(status=status, forecast_id=forecast_id, error=error)
                    record_path = case_dir / "case.json"
                    # Preserve a previous complete result if a changed source is rejected.
                    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
                    if previous.get("status") not in ("complete", "reused"):
                        atomic_json(record_path, {**case.record(), "status": status, "forecast_id": forecast_id,
                                                  "error": error, "updated_at": now()})
                    logger.error("Case %s: %s", case.case_id, error)
                    if isinstance(exc, KeyboardInterrupt):
                        raise
                finally:
                    summary()
                    logger.removeHandler(handler)
                    handler.close()
            counts = {s: sum(r["status"] == s for r in rows.values()) for s in ("complete", "reused", "failed")}
            logger.info("Finished: %s; summary: %s", counts, root / "summary.csv")
            return 1 if counts["failed"] else 0
        finally:
            if runtime is not None:
                runtime.close()
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
