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


def _mask_min_consecutive(candidate_mask, min_consecutive, breaker_mask=None):
    """Return mask where runs of True meet the min_consecutive requirement."""
    if min_consecutive <= 1:
        return candidate_mask.copy()

    if breaker_mask is None:
        breaker_mask = ~candidate_mask

    rows, cols, timesteps = candidate_mask.shape
    if timesteps < min_consecutive:
        return np.zeros_like(candidate_mask, dtype=bool)

    pixels = rows * cols
    candidate_flat = candidate_mask.reshape(pixels, timesteps)
    breaker_flat = breaker_mask.reshape(pixels, timesteps)
    qualified_flat = np.zeros_like(candidate_flat, dtype=bool)

    run_counts = np.zeros(pixels, dtype=np.int32)
    max_pending = min_consecutive - 1
    pending_buf = None
    if max_pending:
        pending_buf = np.full((max_pending, pixels), -1, dtype=np.int32)

    for t in range(timesteps):
        breakers = breaker_flat[:, t]
        if np.any(breakers):
            run_counts[breakers] = 0
            if pending_buf is not None:
                pending_buf[:, breakers] = -1

        candidates = candidate_flat[:, t]
        if not np.any(candidates):
            continue

        run_counts[candidates] += 1

        if pending_buf is not None:
            still_pending = candidates & (run_counts < min_consecutive)
            if np.any(still_pending):
                slots = run_counts[still_pending] - 1
                idxs = np.where(still_pending)[0]
                pending_buf[slots, idxs] = t

        just_reached = candidates & (run_counts == min_consecutive)
        if np.any(just_reached):
            idxs = np.where(just_reached)[0]
            qualified_flat[idxs, t] = True
            if pending_buf is not None:
                for slot in range(max_pending):
                    stored = pending_buf[slot, idxs]
                    valid = stored >= 0
                    if np.any(valid):
                        qualified_flat[idxs[valid], stored[valid]] = True
                pending_buf[:, idxs] = -1

        continuing = candidates & (run_counts > min_consecutive)
        if np.any(continuing):
            idxs = np.where(continuing)[0]
            qualified_flat[idxs, t] = True

    return qualified_flat.reshape(rows, cols, timesteps)


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

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(ratio.astype('float32'), 1)
        dst.write(bare_counts.astype('float32'), 2)
        dst.write(valid_counts.astype('float32'), 3)
        dst.set_band_description(1, 'bare_soil_ratio')
        dst.set_band_description(2, 'bare_soil_occurrences')
        dst.set_band_description(3, 'valid_observation_count')


def harmonic(project_name,bare_soil_lower,bare_soil_upper,min_consecutive,mosaic,process_folder,tss_lst,overwrite_results=False,debug_stats=False,**kwargs):

    temp_folder = process_folder + "/temp"
    proc_folder = process_folder + "/results"

    if not tss_lst:
        tss_lst = sorted(glob.glob(f"{temp_folder}/{project_name}/tiles_tss/X*/*.tif"))

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

        raster_tss_data, date_strings, _, _, _ = extract_data(raster_tss, with_std = False)

        tile_start, tile_end = _parse_date_bounds(date_strings)
        if tile_start and tile_end:
            global_start = tile_start if global_start is None else min(global_start, tile_start)
            global_end = tile_end if global_end is None else max(global_end, tile_end)
        output_filename = _format_range_filename(tile_start, tile_end)
        output_file = os.path.join(output, output_filename)

        valid_mask = ~np.isnan(raster_tss_data)
        total_valid = valid_mask.sum(axis=2).astype(np.float32)

        candidate_hits = (raster_tss_data <= bare_soil_lower) | (raster_tss_data >= bare_soil_upper)
        candidate_mask = valid_mask & candidate_hits
        breaker_mask = valid_mask & ~candidate_hits
        qualified_mask = _mask_min_consecutive(candidate_mask, min_consecutive, breaker_mask=breaker_mask)

        bare_occurrences = qualified_mask.sum(axis=2).astype(np.float32)
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

        raster_tss_data = None

    if mosaic and outputs:
        os.makedirs(proc_folder, exist_ok=True)
        project_folder = os.path.join(proc_folder, project_name)
        os.makedirs(project_folder, exist_ok=True)
        mosaic_filename = _format_range_filename(global_start, global_end)
        mosaic_output = os.path.join(project_folder, mosaic_filename)
        mosaic_rasters(outputs, mosaic_output)

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
            for i, desc in enumerate(band_descriptions, start=1):
                if desc:
                    dest.set_band_description(i, desc)

    # Close the input files
    for src in src_files_to_mosaic:
        src.close()


endzeit = time.time()
print("###" * 10)
print("process finished in "+str((endzeit-startzeit)/60)+" minutes")
