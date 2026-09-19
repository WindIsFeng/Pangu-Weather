"""Run from the repository root; no editable install is needed."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch Pangu-Weather forecasts from a case CSV")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="prepare ERA5 and forecast every case")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true", help="print schedule without downloads or output writes")
    run.add_argument("--resume", action="store_true", help="reuse complete matching results; restart failed tasks")
    doctor = sub.add_parser("doctor", help="check environment; --load-model also verifies the provider initializes")
    doctor.add_argument("--config", type=Path, required=True)
    doctor.add_argument("--load-model", type=int, choices=(1, 3, 6, 24))
    args = parser.parse_args(argv)
    try:
        from .config import load_config
        config = load_config(args.config)
        if args.command == "doctor":
            return check_environment(config, args.load_model)
        from .cases import read_cases
        from .pipeline import describe, run_batch
        cases = read_cases(args.cases, config.output_interval_hours)
        if args.dry_run:
            describe(config, cases)
            return 0
        return run_batch(config, cases, resume=args.resume)
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def check_environment(config, step):
    import logging
    report = {"python": sys.executable, "config": config.record(), "packages": {},
              "cds_config_exists": (Path.home() / ".cdsapirc").is_file(),
              "models": {str(h): (config.model_dir / f"pangu_weather_{h}.onnx").is_file()
                         for h in (1, 3, 6, 24)}}
    missing = False
    for name in ("numpy", "xarray", "netCDF4", "PyYAML", "cdsapi", "onnxruntime-gpu"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "MISSING"
            missing = True
    import onnxruntime as ort
    report["available_providers"] = ort.get_available_providers()
    if step is not None:
        from .runtime import Runtime
        runtime = Runtime(config, logging.getLogger("doctor"))
        try:
            session = runtime.session(step)
            report["loaded_model"] = step
            report["actual_providers"] = session.get_providers()
            report["inputs"] = {v.name: v.shape for v in session.get_inputs()}
        finally:
            runtime.close()
    else:
        report["note"] = "Provider listing does not prove CUDA loads; use --load-model 6 to verify."
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
