import os
import time
import shutil
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import rasterio
from rasterio.merge import merge
import glob
from scipy.stats import linregress
import gc
import rasterstats
import geopandas as gpd
import rasterio
from utils.analysis_utils import *
from utils.residuals_utils import *
from utils.residuals_utils import extract_data
from utils.residuals_utils import get_output_array_full
from utils.residuals_utils import write_output_raster
from utils.residuals_utils import slice_by_date
from utils.residuals_utils import calculate_residuals

BAND_CHUNK_SIZE = 8

def format_time(seconds):
    """Format the time in hours, minutes, and seconds."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

startzeit = time.time()

def process_point_timeseries(raster_tsi_path, raster_tss_path, points_path, save_fig,
                             uncertainty="prc", id_column="id", title="Time Series Comparison", ylab="Value"):

    points = gpd.read_file(points_path)

    with_std = uncertainty in ["std", "prc"]

    data_tsi, dates_tsi, _, data_std, model = extract_data(raster_tsi_path, with_std)
    data_tss, dates_tss, _, __, model = extract_data(raster_tss_path, with_std=False)

    with rasterio.open(raster_tss_path) as src:
        affine = src.transform

    for idx, point in points.iterrows():
        tsi_time_series = []
        for i, step in enumerate(data_tsi.transpose(2, 0, 1)):
            time = dt.datetime.strptime(dates_tsi[i], '%Y-%m-%d').date()
            values = rasterstats.point_query(point.geometry, step, affine=affine, interpolate='nearest')
            tsi_time_series.append([time, np.nan if values[0] == -9999 else values])

        tss_time_series = []
        for i, step in enumerate(data_tss.transpose(2, 0, 1)):
            time = dt.datetime.strptime(dates_tss[i], '%Y-%m-%d').date()
            values = rasterstats.point_query(point.geometry, step, affine=affine, interpolate='nearest')
            tss_time_series.append([time, np.nan if values[0] == -9999 else values])

        threshold = None
        if with_std:
            threshold = rasterstats.point_query(point.geometry, data_std, affine=affine, interpolate='nearest')[0]

        plot_timeseries(tsi_time_series, tss_time_series, threshold, uncertainty, point,
                        with_std, save_fig, ylab, title, id_column)


def _parse_band_date(description):
    if not description:
        return None
    token = description[:8]
    try:
        return dt.datetime.strptime(token, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _compute_bare_soil_counts(raster_path, mask_path, lower, upper, min_consecutive):
    min_consecutive = max(1, int(min_consecutive))
    with rasterio.open(raster_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999
        height, width = src.height, src.width
        total_valid = np.zeros((height, width), dtype=np.uint16)
        bare_occurrences = np.zeros((height, width), dtype=np.uint16)
        run_counts = np.zeros((height, width), dtype=np.uint16)
        date_strings = []
        for desc in src.descriptions:
            parsed = _parse_band_date(desc)
            if parsed:
                date_strings.append(parsed)

        mask_src = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask_src = rasterio.open(mask_path)
                if mask_src.count != src.count:
                    print(f"Warning: mask band count mismatch for {mask_path}. Expected {src.count}, got {mask_src.count}. Ignoring mask.")
                    mask_src.close()
                    mask_src = None
            except Exception as exc:
                print(f"Warning: unable to open mask raster {mask_path}: {exc}")
                mask_src = None

        band_chunk = max(1, min(BAND_CHUNK_SIZE, src.count))
        indexes = list(range(1, src.count + 1))
        try:
            for chunk_start in range(0, src.count, band_chunk):
                chunk_indexes = indexes[chunk_start:chunk_start + band_chunk]
                data_chunk = src.read(chunk_indexes)
                if mask_src:
                    mask_chunk = mask_src.read(chunk_indexes)
                else:
                    mask_chunk = None

                for offset, band_idx in enumerate(chunk_indexes):
                    band = data_chunk[offset]
                    valid = band != nodata
                    threshold_hit = (band <= lower) | (band >= upper)
                    candidate = valid & threshold_hit

                    if mask_chunk is not None:
                        pass_mask = mask_chunk[offset] != 0
                        candidate &= pass_mask
                    else:
                        pass_mask = None

                    if np.any(candidate):
                        run_counts[candidate] += 1
                        just_reached = candidate & (run_counts == min_consecutive)
                        if np.any(just_reached):
                            increment = 1 if min_consecutive == 1 else min_consecutive
                            bare_occurrences[just_reached] += increment
                        continuing = candidate & (run_counts > min_consecutive)
                        if np.any(continuing):
                            bare_occurrences[continuing] += 1

                    breaker = valid & (~threshold_hit)
                    if pass_mask is not None:
                        breaker |= valid & (~pass_mask)
                    if np.any(breaker):
                        run_counts[breaker] = 0

                    total_valid += valid
        finally:
            if mask_src:
                mask_src.close()

    return date_strings, total_valid, bare_occurrences


def _parse_date_bounds(date_strings):
    if not date_strings:
        return None, None
    start_dt = dt.datetime.strptime(date_strings[0], "%Y-%m-%d")
    end_dt = dt.datetime.strptime(date_strings[-1], "%Y-%m-%d")
    return start_dt, end_dt


def _format_range_filename(start_dt, end_dt):
    if not start_dt or not end_dt:
        return "UNKNOWN_DATE_RANGE_BS.tif"
    return f"{start_dt.strftime('%B').upper()}_{start_dt.year}_{end_dt.strftime('%B').upper()}_{end_dt.year}_BS.tif"


def _parse_filename_bounds(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split('_')
    if len(parts) < 5 or parts[-1] != "BS":
        return None, None
    try:
        start_dt = dt.datetime.strptime(f"{parts[0].title()} {parts[1]}", "%B %Y")
        end_dt = dt.datetime.strptime(f"{parts[2].title()} {parts[3]}", "%B %Y")
    except ValueError:
        return None, None
    return start_dt, end_dt


def _write_bare_soil_summary(reference_raster, output_path, ratio, bare_counts, valid_counts):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(reference_raster) as src:
        meta = src.meta.copy()
        meta.update(dtype='float32', count=3, nodata=np.nan, compress='lzw')

    band_names = (
        'bare_soil_ratio',
        'bare_soil_occurrences',
        'valid_observation_count',
    )

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(ratio.astype('float32'), 1)
        dst.write(bare_counts.astype('float32'), 2)
        dst.write(valid_counts.astype('float32'), 3)
        for idx, name in enumerate(band_names, start=1):
            dst.set_band_description(idx, name)
        dst.descriptions = band_names


def baresoil(project_name,bare_soil_lower,bare_soil_upper,min_consecutive,mosaic,process_folder,tss_lst,overwrite_results=False,debug_stats=False,cleanup_tss=False,**kwargs):

    temp_folder = process_folder + "/temp"
    proc_folder = process_folder + "/results"

    if not tss_lst:
        discovered = sorted(glob.glob(f"{temp_folder}/{project_name}/tiles_tss/X*/*.tif"))
        tss_lst = [fp for fp in discovered if not fp.endswith("_mask.tif")]

    tss_lst = [fp for fp in tss_lst if not fp.endswith("_mask.tif")]

    if not tss_lst:
        print("No TSS rasters found; skipping bare soil analysis.")
        return

    min_consecutive = max(int(min_consecutive), 1)
    outputs = []
    global_start = None
    global_end = None

    for raster_tss in tss_lst:
        print("###" * 10)
        print(f"TSS:  {raster_tss}")
        output = raster_tss.replace(".tif", "_output")
        if os.path.exists(output):
            if overwrite_results:
                shutil.rmtree(output)
            else:
                print("Output folder in TSS already exists. Skipping processing ...")
                existing_outputs = sorted(glob.glob(os.path.join(output, "*_BS.tif")))
                if existing_outputs:
                    outputs.append(existing_outputs[0])
                    start_dt, end_dt = _parse_filename_bounds(existing_outputs[0])
                    if start_dt and end_dt:
                        global_start = start_dt if global_start is None else min(global_start, start_dt)
                        global_end = end_dt if global_end is None else max(global_end, end_dt)
                else:
                    outputs.append(os.path.join(output, "bare_soil_metrics.tif"))
                continue
        os.makedirs(output, exist_ok=True)

        mask_path = raster_tss.replace('.tif', '_mask.tif')
        date_strings, total_valid_counts, bare_occurrences_counts = _compute_bare_soil_counts(
            raster_tss,
            mask_path if os.path.exists(mask_path) else None,
            bare_soil_lower,
            bare_soil_upper,
            min_consecutive,
        )
        tile_start, tile_end = _parse_date_bounds(date_strings)
        if tile_start and tile_end:
            global_start = tile_start if global_start is None else min(global_start, tile_start)
            global_end = tile_end if global_end is None else max(global_end, tile_end)
        output_filename = _format_range_filename(tile_start, tile_end)
        output_file = os.path.join(output, output_filename)

        total_valid = total_valid_counts.astype(np.float32, copy=False)
        bare_occurrences = bare_occurrences_counts.astype(np.float32, copy=False)

        if debug_stats and bare_occurrences.size:
            nonzero_occ = bare_occurrences[bare_occurrences > 0]
            if nonzero_occ.size:
                unique_vals = np.unique(nonzero_occ)
                #print(f"    Bare occurrences unique values (pre-write): {unique_vals[:10]}")
            #else:
                #print("    Bare occurrences: no pixels passed the filter.")

        ratio = np.divide(
            bare_occurrences,
            total_valid,
            out=np.zeros_like(bare_occurrences, dtype=np.float32),
            where=total_valid > 0
        ) * 100
        ratio[total_valid == 0] = np.nan
        valid_ratio = ~np.isnan(ratio)
        ratio[valid_ratio] = np.floor(ratio[valid_ratio] + 0.5)

        _write_bare_soil_summary(raster_tss, output_file, ratio, bare_occurrences, total_valid)
        if debug_stats:
            with rasterio.open(output_file) as dbg_src:
                band2 = dbg_src.read(2)
                nonzero_after = band2[band2 > 0]
                if nonzero_after.size:
                    unique_after = np.unique(nonzero_after)
                    #print(f"    Bare occurrences unique values (post-write): {unique_after[:10]}")
                #else:
                    #print("    Bare occurrences after write: no non-zero values.")
        outputs.append(output_file)

    if mosaic and outputs:
        os.makedirs(proc_folder, exist_ok=True)
        project_folder = os.path.join(proc_folder, project_name)
        os.makedirs(project_folder, exist_ok=True)
        mosaic_filename = _format_range_filename(global_start, global_end)
        mosaic_output = os.path.join(project_folder, mosaic_filename)
        mosaic_rasters(
            outputs,
            mosaic_output,
            ['bare_soil_ratio', 'bare_soil_occurrences', 'valid_observation_count']
        )

    if cleanup_tss:
        tiles_root = os.path.join(process_folder, "temp", project_name, "tiles_tss")
        mask_root = os.path.join(process_folder, "temp", "_mask", project_name)
        for label, path in (("TSS tiles", tiles_root), ("mask tiles", mask_root)):
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                print(f"Removed {label}: {path}")
            else:
                print(f"{label} already removed or missing: {path}")

def mosaic_rasters(input_pattern, output_filename, band_descriptions=None):
    """
    Mosaic rasters matching the input pattern and save to output_filename.

    Parameters:
    - input_pattern: str, a wildcard pattern to match input raster files (e.g., "./tiles/*.tif").
    - output_filename: str, the name of the output mosaic raster file.
    """

    # Find all files matching the pattern
    src_files_to_mosaic = [rasterio.open(fp) for fp in input_pattern]

    # Mosaic the rasters
    mosaic, out_transform = merge(src_files_to_mosaic)

    # Get metadata from one of the input files
    out_meta = src_files_to_mosaic[0].meta.copy()

    # Update metadata with new dimensions, transform, and compression (optional)
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw"
    })

    # Write the mosaic raster to disk
    with rasterio.open(output_filename, "w", **out_meta) as dest:
        dest.write(mosaic)
        if band_descriptions:
            clean_names = []
            for i, desc in enumerate(band_descriptions, start=1):
                label = desc or f"band_{i}"
                dest.set_band_description(i, label)
                clean_names.append(label)
            dest.descriptions = tuple(clean_names)

    # Close the input files
    for src in src_files_to_mosaic:
        src.close()


endzeit = time.time()
print("###" * 10)
print("process finished in "+str((endzeit-startzeit)/60)+" minutes")
