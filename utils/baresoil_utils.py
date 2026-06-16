import os
import time
import shutil
import subprocess
import tempfile
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import rasterio
import glob
from scipy.ndimage import generic_filter
from scipy.stats import linregress
import gc
import rasterstats
import geopandas as gpd
import rasterio
from pathlib import Path
from utils.analysis_utils import *
from utils.residuals_utils import *
from utils.residuals_utils import extract_data
from utils.residuals_utils import get_output_array_full
from utils.residuals_utils import write_output_raster
from utils.residuals_utils import slice_by_date
from utils.residuals_utils import calculate_residuals

BAND_CHUNK_SIZE = 8
REQUIRED_GDAL_TOOLS = ["gdalbuildvrt", "gdal_translate", "gdalwarp"]
DEFAULT_OUTPUT_NODATA = -9999.0
DEFAULT_COMPRESSION = "DEFLATE"
DEFAULT_ZLEVEL = "9"
DEFAULT_BIGTIFF = "YES"
DEFAULT_BLOCKSIZE = 512
BARESOIL_BAND_NAMES = (
    'bare_soil_ratio',
    'bare_soil_occurrences',
    'valid_observation_count',
)
WEIGHTED_BARESOIL_BAND_NAMES = (
    'weighted_bare_soil_ratio',
    'weighted_bare_soil_score',
    'valid_observation_count',
)

def format_time(seconds):
    """Format the time in hours, minutes, and seconds."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def ensure_gdal_tools():
    missing = [tool for tool in REQUIRED_GDAL_TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "Missing required GDAL tool(s): "
            + ", ".join(missing)
            + ". Install gdal-bin on the target machine."
        )


def run_command(cmd):
    print("Running command:")
    print(" ".join(str(part) for part in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(str(part) for part in cmd)}")


def validate_compression_method(compression_method):
    supported = {"DEFLATE", "LZW", "ZSTD", "PACKBITS", "NONE"}
    value = compression_method.upper()
    if value not in supported:
        raise ValueError(
            f"Unsupported compression method '{compression_method}'. "
            f"Supported values: {', '.join(sorted(supported))}."
        )
    return value


def normalize_bigtiff(value):
    supported = {"YES", "NO", "IF_NEEDED", "IF_SAFER"}
    normalized = value.upper()
    if normalized not in supported:
        raise ValueError(
            f"Unsupported BIGTIFF option '{value}'. "
            f"Supported values: {', '.join(sorted(supported))}."
        )
    return normalized


def infer_predictor(dtype, compression_method):
    if compression_method not in {"DEFLATE", "LZW", "ZSTD"}:
        return None
    if "float" in dtype.lower():
        return "3"
    return "2"


def build_creation_options(compression_method, predictor, zlevel, bigtiff, blocksize):
    options = [
        "-co", f"COMPRESS={compression_method}",
        "-co", f"BIGTIFF={bigtiff}",
        "-co", "TILED=YES",
        "-co", f"BLOCKXSIZE={blocksize}",
        "-co", f"BLOCKYSIZE={blocksize}",
    ]
    if predictor is not None:
        options.extend(["-co", f"PREDICTOR={predictor}"])
    if compression_method == "DEFLATE":
        options.extend(["-co", f"ZLEVEL={zlevel}"])
    elif compression_method == "ZSTD":
        options.extend(["-co", f"ZSTD_LEVEL={zlevel}"])
    return options


def derive_nodata_value(src, fallback_nodata=None):
    src_nodata = src.nodata
    if src_nodata is not None:
        return src_nodata

    if fallback_nodata is not None:
        return fallback_nodata

    src_dtype = src.dtypes[0]
    if src_dtype in ["int8", "byte"]:
        return -128
    if src_dtype == "uint8":
        return 255
    if src_dtype == "int16":
        return -32768
    if src_dtype == "uint16":
        return 65535
    if src_dtype == "int32":
        return -2147483648
    if src_dtype == "uint32":
        return 4294967295
    if src_dtype in ["float32", "float64"]:
        return DEFAULT_OUTPUT_NODATA

    raise ValueError(f"Unsupported dtype for nodata derivation: {src_dtype}")


def make_output_tile_name(tile_path):
    tile_dir = tile_path.parent.parent.name
    tile_name = tile_path.parent.name
    return f"{tile_dir}_{tile_name}.tif"


def write_tile_list(tile_paths, temp_dir):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", dir=temp_dir, delete=False) as tmp_file:
        for tile_path in tile_paths:
            tmp_file.write(f"{tile_path}\n")
        return Path(tmp_file.name)


def prepare_aoi_for_gdal(aoi_path, target_crs, temp_dir):
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI file contains no features: {aoi_path}")
    if target_crs and aoi.crs != target_crs:
        aoi = aoi.to_crs(target_crs)

    if hasattr(aoi.geometry, "make_valid"):
        aoi["geometry"] = aoi.geometry.make_valid()
    else:
        aoi["geometry"] = aoi.buffer(0)

    aoi = aoi[~aoi.geometry.is_empty & aoi.geometry.notnull()].copy()
    if aoi.empty:
        raise ValueError(f"AOI file has no valid geometries after repair: {aoi_path}")

    union_geom = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
    aoi_out = gpd.GeoDataFrame({"id": [1]}, geometry=[union_geom], crs=aoi.crs)

    fd, prepared_path = tempfile.mkstemp(suffix=".gpkg", dir=temp_dir)
    prepared_aoi_path = Path(prepared_path)
    prepared_aoi_path.unlink(missing_ok=True)
    try:
        os.close(fd)
    except OSError:
        pass

    aoi_out.to_file(prepared_aoi_path, driver="GPKG")
    return prepared_aoi_path, union_geom


def write_geometry_cutline(geometry, crs, temp_dir):
    cutline = gpd.GeoDataFrame({"id": [1]}, geometry=[geometry], crs=crs)
    fd, prepared_path = tempfile.mkstemp(suffix=".gpkg", dir=temp_dir)
    prepared_cutline_path = Path(prepared_path)
    prepared_cutline_path.unlink(missing_ok=True)
    try:
        os.close(fd)
    except OSError:
        pass

    cutline.to_file(prepared_cutline_path, driver="GPKG")
    return prepared_cutline_path


def build_vrt(tile_paths, vrt_output_path, temp_dir):
    tile_list_path = write_tile_list(tile_paths, temp_dir)
    try:
        run_command([
            "gdalbuildvrt",
            "-srcnodata", str(DEFAULT_OUTPUT_NODATA),
            "-vrtnodata", str(DEFAULT_OUTPUT_NODATA),
            "-input_file_list", str(tile_list_path),
            str(vrt_output_path),
        ])
        return vrt_output_path
    finally:
        if tile_list_path.exists():
            tile_list_path.unlink()


def build_final_raster(
    vrt_path,
    final_output_path,
    dtype,
    compression_method,
    predictor,
    zlevel,
    bigtiff,
    blocksize,
    output_nodata,
    aoi_path=None,
    num_threads="ALL_CPUS",
    cachemax_mb=512,
):
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    if aoi_path:
        with rasterio.open(vrt_path) as src:
            prepared_aoi_path, _ = prepare_aoi_for_gdal(Path(aoi_path), src.crs, final_output_path.parent)
            src_nodata = derive_nodata_value(src, fallback_nodata=output_nodata)
        try:
            cmd = [
                "gdalwarp",
                "-overwrite",
                "-cutline", str(prepared_aoi_path),
                "-crop_to_cutline",
                "-srcnodata", str(src_nodata),
                "-dstnodata", str(output_nodata),
                "-r", "near",
                "-multi",
                "-wo", f"NUM_THREADS={num_threads}",
                "-wm", str(cachemax_mb),
                "-of", "GTiff",
                "-ot", dtype.upper(),
            ]
            cmd.extend(build_creation_options(compression_method, predictor, zlevel, bigtiff, blocksize))
            cmd.extend([str(vrt_path), str(final_output_path)])
            run_command(cmd)
        finally:
            if prepared_aoi_path.exists():
                prepared_aoi_path.unlink()
    else:
        cmd = [
            "gdal_translate",
            "-of", "GTiff",
            "-a_nodata", str(output_nodata),
            "-ot", dtype.upper(),
        ]
        cmd.extend(build_creation_options(compression_method, predictor, zlevel, bigtiff, blocksize))
        cmd.extend([str(vrt_path), str(final_output_path)])
        run_command(cmd)
    return final_output_path


def set_band_descriptions(raster_path, band_names):
    with rasterio.open(raster_path, "r+") as dst:
        for idx, name in enumerate(band_names, start=1):
            if name:
                dst.set_band_description(idx, name)
        dst.descriptions = tuple(band_names)


def raster_matches_expectations(raster_path, expected_band_count=None):
    try:
        raster_path = Path(raster_path)
        if not raster_path.is_file() or raster_path.stat().st_size <= 0:
            return False
        with rasterio.open(raster_path) as src:
            if src.count <= 0:
                return False
            if expected_band_count is not None and src.count != expected_band_count:
                return False
            if src.width <= 0 or src.height <= 0:
                return False
            sample = src.read(1, window=((0, min(1, src.height)), (0, min(1, src.width))))
            if sample.size == 0:
                return False
        return True
    except Exception:
        return False


def export_baresoil_tiles(
    tile_paths,
    final_output_path,
    band_names=BARESOIL_BAND_NAMES,
    aoi_path=None,
    dtype="int16",
    num_threads="ALL_CPUS",
    cachemax_mb=512,
    overwrite_tiles=False,
    compression_method=DEFAULT_COMPRESSION,
    bigtiff=DEFAULT_BIGTIFF,
    zlevel=DEFAULT_ZLEVEL,
    blocksize=DEFAULT_BLOCKSIZE,
    output_nodata=DEFAULT_OUTPUT_NODATA,
):
    ensure_gdal_tools()
    compression_method = validate_compression_method(compression_method)
    bigtiff = normalize_bigtiff(bigtiff)
    predictor = infer_predictor(dtype, compression_method)

    final_output_path = Path(final_output_path)
    vrt_output_path = final_output_path.with_suffix(".vrt")
    temp_dir = final_output_path.parent.parent / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    source_tile_paths = [Path(tile_path) for tile_path in tile_paths]
    build_vrt(source_tile_paths, vrt_output_path, temp_dir)
    build_final_raster(
        vrt_output_path,
        final_output_path,
        dtype,
        compression_method,
        predictor,
        zlevel,
        bigtiff,
        blocksize,
        output_nodata,
        aoi_path=aoi_path,
        num_threads=num_threads,
        cachemax_mb=cachemax_mb,
    )
    set_band_descriptions(final_output_path, band_names)
    return str(final_output_path)


def _nanmedian_kernel(values):
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.nan
    return float(np.median(valid))


def _expand_window(window, width, height, halo):
    col_off = max(0, window.col_off - halo)
    row_off = max(0, window.row_off - halo)
    max_col = min(width, window.col_off + window.width + halo)
    max_row = min(height, window.row_off + window.height + halo)

    expanded = rasterio.windows.Window(
        col_off=col_off,
        row_off=row_off,
        width=max_col - col_off,
        height=max_row - row_off,
    )
    inner_row_start = window.row_off - row_off
    inner_col_start = window.col_off - col_off
    inner_rows = slice(inner_row_start, inner_row_start + window.height)
    inner_cols = slice(inner_col_start, inner_col_start + window.width)
    return expanded, inner_rows, inner_cols


def create_stabilized_baresoil_output(source_raster_path, filter_size=3):
    source_raster_path = Path(source_raster_path)
    stabilized_path = source_raster_path.with_name(
        source_raster_path.stem.replace("_v1_0", "_stabilized_v1_0") + source_raster_path.suffix
    )

    halo = filter_size // 2
    with rasterio.open(source_raster_path) as src:
        meta = src.meta.copy()
        meta.update(dtype="int16", nodata=DEFAULT_OUTPUT_NODATA, count=3)
        with rasterio.open(stabilized_path, "w", **meta) as dst:
            for _, window in src.block_windows(1):
                expanded, inner_rows, inner_cols = _expand_window(window, src.width, src.height, halo)

                band2 = src.read(2, window=expanded).astype(np.float32)
                band3 = src.read(3, window=expanded).astype(np.float32)

                band2[band2 == src.nodata] = np.nan
                filtered_band2 = generic_filter(
                    band2,
                    function=_nanmedian_kernel,
                    size=filter_size,
                    mode="constant",
                    cval=np.nan,
                )

                core_band2 = filtered_band2[inner_rows, inner_cols]
                core_band3 = band3[inner_rows, inner_cols]
                core_band3[core_band3 == src.nodata] = np.nan

                core_band1 = np.full(core_band2.shape, DEFAULT_OUTPUT_NODATA, dtype=np.float32)
                valid = np.isfinite(core_band2) & np.isfinite(core_band3) & (core_band3 > 0)
                core_band1[valid] = np.floor((100.0 * core_band2[valid] / core_band3[valid]) + 0.5)

                out_band1 = np.where(np.isfinite(core_band1), core_band1, DEFAULT_OUTPUT_NODATA).astype(np.int16)
                out_band2 = np.where(np.isfinite(core_band2), core_band2, DEFAULT_OUTPUT_NODATA).astype(np.int16)
                out_band3 = np.where(np.isfinite(core_band3), core_band3, DEFAULT_OUTPUT_NODATA).astype(np.int16)

                dst.write(out_band1, 1, window=window)
                dst.write(out_band2, 2, window=window)
                dst.write(out_band3, 3, window=window)

        set_band_descriptions(stabilized_path, BARESOIL_BAND_NAMES)

    print(f"Stabilized bare-soil output written to {stabilized_path}")
    return str(stabilized_path)

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


def _get_raster_layout(src, mask_path):
    descriptions = list(src.descriptions)
    mask_start_idx = next(
        (idx for idx, desc in enumerate(descriptions, start=1) if desc and desc.endswith("_MASK")),
        None,
    )

    if mask_start_idx is not None:
        value_count = mask_start_idx - 1
        mask_count = src.count - value_count
        if value_count > 0 and mask_count == value_count:
            date_strings = [
                parsed for desc in descriptions[:value_count]
                if (parsed := _parse_band_date(desc))
            ]
            return {
                "data_indexes": list(range(1, value_count + 1)),
                "mask_indexes": list(range(mask_start_idx, src.count + 1)),
                "mask_src": src,
                "date_strings": date_strings,
            }
        print(
            f"Warning: inline mask layout mismatch for {src.name}. "
            f"Expected equal value/mask bands, got {value_count}/{mask_count}. Ignoring inline mask."
        )

    date_strings = [parsed for desc in descriptions if (parsed := _parse_band_date(desc))]
    mask_src = None
    mask_indexes = None
    if mask_path and os.path.exists(mask_path):
        try:
            mask_src = rasterio.open(mask_path)
            if mask_src.count != src.count:
                print(
                    f"Warning: mask band count mismatch for {mask_path}. "
                    f"Expected {src.count}, got {mask_src.count}. Ignoring mask."
                )
                mask_src.close()
                mask_src = None
            else:
                mask_indexes = list(range(1, mask_src.count + 1))
        except Exception as exc:
            print(f"Warning: unable to open mask raster {mask_path}: {exc}")
            mask_src = None

    return {
        "data_indexes": list(range(1, src.count + 1)),
        "mask_indexes": mask_indexes,
        "mask_src": mask_src,
        "date_strings": date_strings,
    }


def _compute_bare_soil_counts(raster_path, mask_path, lower, upper, min_consecutive, weighted_threshold_scale):
    min_consecutive = max(1, int(min_consecutive))
    weighted_threshold_scale = max(float(weighted_threshold_scale), 1.0)
    with rasterio.open(raster_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999
        height, width = src.height, src.width
        total_valid = np.zeros((height, width), dtype=np.uint16)
        bare_occurrences = np.zeros((height, width), dtype=np.uint16)
        weighted_occurrences = np.zeros((height, width), dtype=np.float32)
        run_counts = np.zeros((height, width), dtype=np.uint16)
        run_weight_sums = np.zeros((height, width), dtype=np.float32)
        layout = _get_raster_layout(src, mask_path)
        data_indexes = layout["data_indexes"]
        mask_indexes = layout["mask_indexes"]
        mask_src = layout["mask_src"]
        date_strings = layout["date_strings"]

        band_chunk = max(1, min(BAND_CHUNK_SIZE, len(data_indexes)))
        try:
            for chunk_start in range(0, len(data_indexes), band_chunk):
                chunk_indexes = data_indexes[chunk_start:chunk_start + band_chunk]
                data_chunk = src.read(chunk_indexes)
                if mask_src and mask_indexes:
                    chunk_mask_indexes = mask_indexes[chunk_start:chunk_start + len(chunk_indexes)]
                    mask_chunk = mask_src.read(chunk_mask_indexes)
                else:
                    mask_chunk = None

                for offset, _band_idx in enumerate(chunk_indexes):
                    band = data_chunk[offset]
                    valid = band != nodata
                    threshold_hit = (band <= lower) | (band >= upper)
                    candidate = valid & threshold_hit
                    band_float = band.astype(np.float32, copy=False)

                    lower_distance = np.clip((lower - band_float) / weighted_threshold_scale, 0.0, 1.0)
                    upper_distance = np.clip((band_float - upper) / weighted_threshold_scale, 0.0, 1.0)
                    current_weight = np.maximum(lower_distance, upper_distance)

                    if mask_chunk is not None:
                        pass_mask = mask_chunk[offset] != 0
                        candidate &= pass_mask
                    else:
                        pass_mask = None

                    current_weight = np.where(candidate, current_weight, 0.0)

                    if np.any(candidate):
                        run_counts[candidate] += 1
                        run_weight_sums[candidate] += current_weight[candidate]
                        just_reached = candidate & (run_counts == min_consecutive)
                        if np.any(just_reached):
                            increment = 1 if min_consecutive == 1 else min_consecutive
                            bare_occurrences[just_reached] += increment
                            weighted_occurrences[just_reached] += run_weight_sums[just_reached]
                        continuing = candidate & (run_counts > min_consecutive)
                        if np.any(continuing):
                            bare_occurrences[continuing] += 1
                            weighted_occurrences[continuing] += current_weight[continuing]

                    breaker = valid & (~threshold_hit)
                    if pass_mask is not None:
                        breaker |= valid & (~pass_mask)
                    if np.any(breaker):
                        run_counts[breaker] = 0
                        run_weight_sums[breaker] = 0.0

                    total_valid += valid
        finally:
            if mask_src and mask_src is not src:
                mask_src.close()

    return date_strings, total_valid, bare_occurrences, weighted_occurrences


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


def _format_final_output_filename(start_dt, end_dt):
    if not start_dt or not end_dt:
        return "bare-soil_UNKNOWN_v1_0.tif"
    return f"bare-soil_{start_dt.year}_v1_0.tif"


def _format_weighted_final_output_filename(start_dt, end_dt):
    if not start_dt or not end_dt:
        return "bare-soil_UNKNOWN_weighted_v1_0.tif"
    return f"bare-soil_{start_dt.year}_weighted_v1_0.tif"


def _format_weighted_range_filename(start_dt, end_dt):
    if not start_dt or not end_dt:
        return "UNKNOWN_DATE_RANGE_WEIGHTED_BS.tif"
    return f"{start_dt.strftime('%B').upper()}_{start_dt.year}_{end_dt.strftime('%B').upper()}_{end_dt.year}_WEIGHTED_BS.tif"


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


def _write_bare_soil_summary(reference_raster, output_path, ratio, bare_counts, valid_counts, band_names=BARESOIL_BAND_NAMES, dtype="int16"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(reference_raster) as src:
        meta = src.meta.copy()
        meta.update(dtype=dtype, count=3, nodata=DEFAULT_OUTPUT_NODATA, compress='lzw')

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(ratio.astype(dtype), 1)
        dst.write(bare_counts.astype(dtype), 2)
        dst.write(valid_counts.astype(dtype), 3)
        for idx, name in enumerate(band_names, start=1):
            dst.set_band_description(idx, name)
        dst.descriptions = band_names


def baresoil(project_name,bare_soil_lower,bare_soil_upper,min_consecutive,mosaic,process_folder,tss_lst,overwrite_results=False,debug_stats=False,cleanup_tss=False,aoi_path=None,write_stabilized_output=False,stabilized_filter_size=3,write_weighted_output=False,weighted_threshold_scale=150,**kwargs):

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
    weighted_outputs = []
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
                existing_outputs = sorted(glob.glob(os.path.join(output, "*_BS.tif")))
                reusable_output = next(
                    (candidate for candidate in existing_outputs if raster_matches_expectations(candidate, expected_band_count=3)),
                    None,
                )
                if reusable_output:
                    print("Valid bare-soil tile output already exists. Skipping processing ...")
                    outputs.append(reusable_output)
                    start_dt, end_dt = _parse_filename_bounds(reusable_output)
                    if start_dt and end_dt:
                        global_start = start_dt if global_start is None else min(global_start, start_dt)
                        global_end = end_dt if global_end is None else max(global_end, end_dt)
                    continue
                else:
                    print("Existing bare-soil tile output is missing or invalid. Recomputing tile ...")
                    shutil.rmtree(output)
        os.makedirs(output, exist_ok=True)

        mask_path = raster_tss.replace('.tif', '_mask.tif')
        date_strings, total_valid_counts, bare_occurrences_counts, weighted_occurrences_counts = _compute_bare_soil_counts(
            raster_tss,
            mask_path if os.path.exists(mask_path) else None,
            bare_soil_lower,
            bare_soil_upper,
            min_consecutive,
            weighted_threshold_scale,
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
        ratio[total_valid == 0] = DEFAULT_OUTPUT_NODATA
        valid_ratio = ratio != DEFAULT_OUTPUT_NODATA
        ratio[valid_ratio] = np.floor(ratio[valid_ratio] + 0.5)

        no_data_mask = total_valid == 0
        bare_occurrences[no_data_mask] = DEFAULT_OUTPUT_NODATA
        total_valid[no_data_mask] = DEFAULT_OUTPUT_NODATA

        _write_bare_soil_summary(raster_tss, output_file, ratio, bare_occurrences, total_valid, dtype="int16")
        if write_weighted_output:
            weighted_output_file = os.path.join(output, _format_weighted_range_filename(tile_start, tile_end))
            weighted_occurrences = weighted_occurrences_counts.astype(np.float32, copy=False)
            weighted_ratio = np.divide(
                weighted_occurrences,
                total_valid_counts.astype(np.float32, copy=False),
                out=np.zeros_like(weighted_occurrences, dtype=np.float32),
                where=total_valid_counts > 0
            ) * 100
            weighted_ratio[total_valid_counts == 0] = DEFAULT_OUTPUT_NODATA
            valid_weighted_ratio = weighted_ratio != DEFAULT_OUTPUT_NODATA
            weighted_ratio[valid_weighted_ratio] = np.floor(weighted_ratio[valid_weighted_ratio] + 0.5)

            weighted_occurrences[no_data_mask] = DEFAULT_OUTPUT_NODATA
            _write_bare_soil_summary(
                raster_tss,
                weighted_output_file,
                weighted_ratio,
                weighted_occurrences,
                total_valid,
                band_names=WEIGHTED_BARESOIL_BAND_NAMES,
                dtype="float32",
            )
            weighted_outputs.append(weighted_output_file)
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
        mosaic_filename = _format_final_output_filename(global_start, global_end)
        mosaic_output = os.path.join(project_folder, mosaic_filename)
        if overwrite_results or not raster_matches_expectations(mosaic_output, expected_band_count=3):
            export_baresoil_tiles(
                tile_paths=outputs,
                final_output_path=mosaic_output,
                band_names=BARESOIL_BAND_NAMES,
                aoi_path=aoi_path,
                dtype="int16",
                overwrite_tiles=overwrite_results,
            )
        else:
            print(f"Valid final bare-soil mosaic already exists. Reusing: {mosaic_output}")

        if write_weighted_output and weighted_outputs:
            weighted_mosaic_output = os.path.join(project_folder, _format_weighted_final_output_filename(global_start, global_end))
            if overwrite_results or not raster_matches_expectations(weighted_mosaic_output, expected_band_count=3):
                export_baresoil_tiles(
                    tile_paths=weighted_outputs,
                    final_output_path=weighted_mosaic_output,
                    band_names=WEIGHTED_BARESOIL_BAND_NAMES,
                    aoi_path=aoi_path,
                    dtype="float32",
                    overwrite_tiles=overwrite_results,
                )
            else:
                print(f"Valid weighted bare-soil mosaic already exists. Reusing: {weighted_mosaic_output}")

        if write_stabilized_output:
            create_stabilized_baresoil_output(
                mosaic_output,
                filter_size=max(1, int(stabilized_filter_size)),
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

endzeit = time.time()
print("###" * 10)
print("process finished in "+str((endzeit-startzeit)/60)+" minutes")
