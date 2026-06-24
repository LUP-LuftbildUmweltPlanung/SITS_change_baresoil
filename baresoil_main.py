# -*- coding: utf-8 -*-
"""
python3 baresoil_main.py \
  --project-name bare_soil_2018 \
  --aoi /data/aoi/germany.shp \
  --process-folder /data/process \
  --force-dir /force \
  --date-range "2018-01-01 2018-12-31"
"""
import argparse
import json
import time

from force.force_baresoil_utils import force_baresoil
from utils.baresoil_utils import baresoil


PROCESS_FOLDER = "/rvt_mount/process"

force_params = {
    "project_name": "bare_soil_project__",
    "aoi": "/rvt_mount/3DTests/data/bare_soil/3tiles.shp",
    "TSS_Sensors": "SEN2A SEN2B",
    "TSS_DATE_RANGE": "2018-01-01 2018-12-31",
}

force_advanced_params = {
    "process_folder": PROCESS_FOLDER,
    "force_dir": "/force",
    "TSS_ABOVE_NOISE": 0,
    "TSS_BELOW_NOISE": 0,
    "TSS_SPECTRAL_ADJUST": "FALSE",
    "hold": False,
    "TSS_NTHREAD_READ": 11,
    "TSS_NTHREAD_COMPUTE": 22,
    "TSS_NTHREAD_WRITE": 8,
    "TSS_BLOCK_SIZE": 3000,
}

analysis_params = {
    "project_name": force_params["project_name"],
    "bare_soil_lower": 173,
    "bare_soil_upper": 1371,
    "min_consecutive": 3,
    "mosaic": True,
    "aoi_path": force_params["aoi"],
    "overwrite_results": True,
    "debug_stats": True,
    "cleanup_tss": True,
    "write_stabilized_output": False,
    "stabilized_filter_size": 3,
    "write_weighted_output": False,
    "weighted_threshold_scale": 150,
}

analysis_advanced_params = {
    "process_folder": PROCESS_FOLDER,
    "tss_lst": None,
}

run_flags = {
    "run_force": True,
    "run_analysis": True,
}


def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the bare-soil FORCE workflow.")
    parser.add_argument("--project-name", help="Project name used under temp/results.")
    parser.add_argument("--aoi", help="Path to the AOI shapefile.")
    parser.add_argument("--date-range", help="FORCE date range, for example: '2018-01-01 2018-12-31'.")
    parser.add_argument("--process-folder", help="Processing root folder.")
    parser.add_argument("--force-dir", help="Mounted FORCE datacube root.")
    parser.add_argument("--sensors", help="Sensor allow-list, for example 'SEN2A SEN2B'.")
    parser.add_argument("--run-force", type=str_to_bool, help="Run the FORCE stage.")
    parser.add_argument("--run-analysis", type=str_to_bool, help="Run the analysis/export stage.")
    parser.add_argument("--hold", type=str_to_bool, help="Launch FORCE commands in xterm and keep the terminal open.")
    parser.add_argument("--force-only", action="store_true", help="Run FORCE only.")
    parser.add_argument("--analysis-only", action="store_true", help="Run analysis only.")
    parser.add_argument("--write-weighted-output", type=str_to_bool, help="Write the weighted annual product.")
    parser.add_argument("--weighted-threshold-scale", type=float, help="Distance scale used by the weighted product.")
    parser.add_argument("--min-consecutive", type=int, help="Minimum consecutive bare-soil detections.")
    parser.add_argument("--overwrite-results", type=str_to_bool, help="Overwrite existing outputs.")
    parser.add_argument("--cleanup-tss", type=str_to_bool, help="Remove intermediate TSS tiles after aggregation.")
    return parser.parse_args()


def apply_cli_overrides(args):
    if args.project_name:
        force_params["project_name"] = args.project_name
        analysis_params["project_name"] = args.project_name

    if args.aoi:
        force_params["aoi"] = args.aoi
        analysis_params["aoi_path"] = args.aoi

    if args.date_range:
        force_params["TSS_DATE_RANGE"] = args.date_range

    if args.process_folder:
        force_advanced_params["process_folder"] = args.process_folder
        analysis_advanced_params["process_folder"] = args.process_folder

    if args.force_dir:
        force_advanced_params["force_dir"] = args.force_dir

    if args.sensors:
        force_params["TSS_Sensors"] = args.sensors

    if args.hold is not None:
        force_advanced_params["hold"] = args.hold

    if args.write_weighted_output is not None:
        analysis_params["write_weighted_output"] = args.write_weighted_output

    if args.weighted_threshold_scale is not None:
        analysis_params["weighted_threshold_scale"] = args.weighted_threshold_scale

    if args.min_consecutive is not None:
        analysis_params["min_consecutive"] = args.min_consecutive

    if args.overwrite_results is not None:
        analysis_params["overwrite_results"] = args.overwrite_results

    if args.cleanup_tss is not None:
        analysis_params["cleanup_tss"] = args.cleanup_tss

    if args.run_force is not None:
        run_flags["run_force"] = args.run_force

    if args.run_analysis is not None:
        run_flags["run_analysis"] = args.run_analysis

    if args.force_only and args.analysis_only:
        raise ValueError("Use either --force-only or --analysis-only, not both.")
    if args.force_only:
        run_flags["run_force"] = True
        run_flags["run_analysis"] = False
    if args.analysis_only:
        run_flags["run_force"] = False
        run_flags["run_analysis"] = True


def validate_params():
    required_values = {
        "force_params.project_name": force_params["project_name"],
        "force_params.aoi": force_params["aoi"],
        "force_advanced_params.process_folder": force_advanced_params["process_folder"],
        "force_advanced_params.force_dir": force_advanced_params["force_dir"],
    }
    missing = [
        name for name, value in required_values.items()
        if value in (None, "", "/absolute/path/to/aoi.shp")
    ]
    if missing:
        raise ValueError(
            "Missing required parameters: "
            + ", ".join(missing)
            + ". Edit the defaults in baresoil_main.py or pass them on the command line."
        )

    if int(analysis_params["min_consecutive"]) < 1:
        raise ValueError("analysis_params['min_consecutive'] must be >= 1.")

    analysis_params["project_name"] = force_params["project_name"]
    analysis_params["aoi_path"] = force_params["aoi"]
    analysis_advanced_params["process_folder"] = force_advanced_params["process_folder"]


def print_effective_config():
    effective = {
        "force_params": force_params,
        "force_advanced_params": force_advanced_params,
        "analysis_params": analysis_params,
        "analysis_advanced_params": analysis_advanced_params,
        "run_flags": run_flags,
    }
    print("Effective configuration:")
    print(json.dumps(effective, indent=2))


def main():
    args = parse_args()
    apply_cli_overrides(args)
    validate_params()

    force_baresoil_time = 0
    baresoil_time = 0

    if run_flags["run_force"]:
        startzeit_force = time.time()
        force_baresoil(**force_params, **force_advanced_params)
        endzeit_force = time.time()
        force_baresoil_time = endzeit_force - startzeit_force
        print(f"tss executed in: {format_time(force_baresoil_time)}")
    else:
        print("Skipping FORCE processing (`run_force=False`).")

    if run_flags["run_analysis"]:
        startzeit_baresoil = time.time()
        baresoil(**analysis_params, **analysis_advanced_params)
        endzeit_baresoil = time.time()
        baresoil_time = endzeit_baresoil - startzeit_baresoil
        print(f"Analysis executed in: {format_time(baresoil_time)}")
    else:
        print("Skipping bare-soil analysis/export (`run_analysis=False`).")

    total_time = force_baresoil_time + baresoil_time
    print(f"Total execution time: {format_time(total_time)}")
    print_effective_config()


if __name__ == "__main__":
    main()
