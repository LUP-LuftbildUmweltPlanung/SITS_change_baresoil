# SITS_change_baresoil

Bare-soil annual mapping workflow based on FORCE time-series stacks and a PVIR2 threshold approach.

## What this repository does

The workflow has two stages:

1. `FORCE` stage:
   - builds the time-series stack over the AOI
   - supports tile resume, so a crashed run can continue from incomplete tiles
2. `Analysis` stage:
   - computes yearly bare-soil summaries from the FORCE outputs
   - exports a final annual raster named `bare-soil_<YEAR>_v1_0.tif`
   - can optionally export a weighted variant `bare-soil_<YEAR>_weighted_v1_0.tif`

The standard output has 3 bands:

- band 1: `bare_soil_ratio`
- band 2: `bare_soil_occurrences`
- band 3: `valid_observation_count`

## Requirements

System requirements:

- Python `3.9`
- Docker
- `xterm`
- GDAL command-line tools: `gdalbuildvrt`, `gdal_translate`, `gdalwarp`
- a mounted FORCE datacube

Python environment:

```bash
conda create --name sits-baresoil python=3.9
conda activate sits-baresoil
pip install -r requirements.txt
```

Ubuntu packages usually needed:

```bash
sudo apt-get update
sudo apt-get install -y xterm gdal-bin
```

FORCE is expected to be available through Docker. See the official FORCE Docker documentation:

- <https://force-eo.readthedocs.io/en/latest/setup/docker.html>

## Portable setup

This repository now supports two normal ways of running:

1. edit the defaults at the top of [baresoil_main.py](/rvt_mount/SITS_change_baresoil/baresoil_main.py:13)
2. keep the file generic and override values from the command line

For another VM, the important machine-specific values are:

- `project_name`
- `aoi`
- `process_folder`
- `force_dir`
- `TSS_DATE_RANGE`

## Running

Run the full workflow:

```bash
python3 baresoil_main.py
```

Run only FORCE:

```bash
python3 baresoil_main.py --force-only
```

Run only the analysis/mosaicing step:

```bash
python3 baresoil_main.py --analysis-only
```

Example with command-line overrides:

```bash
python3 baresoil_main.py \
  --project-name bare_soil_2018 \
  --aoi /data/aoi/germany.shp \
  --process-folder /data/process \
  --force-dir /force \
  --date-range "2018-01-01 2018-12-31" \
  --write-weighted-output true
```

## Configuration notes

Important parameter groups:

- `force_params`
  - AOI, date range, sensors, project name
- `force_advanced_params`
  - process folder, FORCE mount path, thread counts, FORCE noise-screening options
- `analysis_params`
  - bare-soil thresholds, minimum consecutive detections, weighted-output toggle
- `run_flags`
  - whether to run FORCE and/or the analysis stage

Current defaults keep:

- `min_consecutive = 3`
- standard output enabled
- weighted output optional via `write_weighted_output`

## Reproducibility notes

This repository now includes:

- command-line overrides for machine-specific paths
- resume-aware FORCE tile handling
- stricter output validation before reusing tiles
- a git ignore file to avoid committing local caches and IDE files

To reproduce a run on another VM, keep these inputs consistent:

- same FORCE datacube version
- same AOI
- same date range
- same thresholds
- same FORCE screening parameters

## Testing

Syntax check:

```bash
PYTHONPYCACHEPREFIX=/tmp python3 -m py_compile baresoil_main.py utils/baresoil_utils.py force/force_baresoil_utils.py
```

Unit tests, if `pytest` is installed:

```bash
python3 -m pytest -q tests
```

## Authors

- [Sebastian Valencia](https://github.com/Azarozo19)

## License

GPL-3.0
