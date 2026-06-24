import os
import subprocess
import time
import shutil
import datetime as dt
import sys
import geopandas as gpd
import rasterio
from pathlib import Path

def replace_parameters(filename, replacements):
    with open(filename, 'r') as f:
        content = f.read()
        for key, value in replacements.items():
            content = content.replace(key, value)
    with open(filename, 'w') as f:
        f.write(content)

def extract_coordinates(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    #Skip the first line
    lines = lines[1:]
    #Extract X and Y values
    x_values = [int(line.split('_')[0][1:]) for line in lines]
    y_values = [int(line.split('_')[1][1:]) for line in lines]
    #Extract the desired values
    x_str = f"{min(x_values)} {max(x_values)}"
    y_str = f"{min(y_values)} {max(y_values)}"

    return x_str, y_str

def check_and_reproject_shapefile(shapefile_path, target_epsg=3035):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    # Check the current CRS of the shapefile
    if gdf.crs.to_epsg() != target_epsg:
        print("Reprojecting shapefile to EPSG: 3035")
        # Reproject the shapefile
        gdf = gdf.to_crs(epsg=target_epsg)
        # Define the new file path
        new_shapefile_path = shapefile_path.replace(".shp", "_3035.shp")
        # Save the reprojected shapefile
        gdf.to_file(new_shapefile_path, driver='ESRI Shapefile')
        print(f"Shapefile reprojected and saved to {new_shapefile_path}")
        return new_shapefile_path
    else:
        print("Shapefile is already in EPSG: 3035")
        return shapefile_path


def run_shell_command(cmd, hold=False, stage_name=None, log_path=None):
    stage_label = stage_name or "FORCE stage"
    print(f"[START] {stage_label}")
    print(f"Running command: {cmd}")
    start_time = time.time()
    if hold:
        result = subprocess.run(
            ['xterm', '-hold', '-e', 'bash', '-lc', cmd],
            check=False,
        )
    else:
        log_handle = open(log_path, "a", encoding="utf-8") if log_path else None
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                executable='/bin/bash',
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                sys.stdout.write(line)
                if log_handle:
                    log_handle.write(line)
            result_code = process.wait()
        finally:
            if log_handle:
                log_handle.close()
        result = subprocess.CompletedProcess(cmd, result_code)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")
    elapsed = time.time() - start_time
    print(f"[DONE] {stage_label} in {elapsed / 60:.2f} minutes")


def parse_date_range(date_range):
    start_token, end_token = date_range.split()
    start_date = dt.datetime.strptime(start_token, "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(end_token, "%Y-%m-%d").date()
    return start_date, end_date


def parse_band_date(description):
    if not description:
        return None
    token = description[:8]
    try:
        return dt.datetime.strptime(token, "%Y%m%d").date()
    except ValueError:
        return None


def validate_force_output_descriptions(descriptions, start_date, end_date):
    if not descriptions or not any(descriptions):
        return False

    mask_start_idx = next(
        (idx for idx, desc in enumerate(descriptions) if desc and desc.endswith("_MASK")),
        None,
    )
    if mask_start_idx is None:
        return False

    value_desc = descriptions[:mask_start_idx]
    mask_desc = descriptions[mask_start_idx:]
    if not value_desc or len(value_desc) != len(mask_desc):
        return False

    parsed_dates = []
    previous_date = None
    for value_description, mask_description in zip(value_desc, mask_desc):
        parsed_date = parse_band_date(value_description)
        if parsed_date is None:
            return False
        if parsed_date < start_date or parsed_date > end_date:
            return False
        if previous_date is not None and parsed_date < previous_date:
            return False
        if mask_description != f"{value_description}_MASK":
            return False
        parsed_dates.append(parsed_date)
        previous_date = parsed_date

    return len(set(parsed_dates)) == len(parsed_dates)


def raster_is_readable(raster_path, expected_band_count=None, require_band_descriptions=False):
    try:
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
            if require_band_descriptions:
                descriptions = list(src.descriptions)
                if not descriptions or not any(desc for desc in descriptions):
                    return False
        return True
    except Exception:
        return False


def force_tile_is_complete(raster_path, start_date, end_date):
    try:
        raster_path = Path(raster_path)
        if not raster_path.is_file() or raster_path.stat().st_size <= 0:
            return False
        with rasterio.open(raster_path) as src:
            if src.count <= 0 or src.width <= 0 or src.height <= 0:
                return False
            if src.count % 2 != 0:
                return False
            sample = src.read(1, window=((0, min(1, src.height)), (0, min(1, src.width))))
            if sample.size == 0:
                return False
            descriptions = list(src.descriptions)
        return validate_force_output_descriptions(descriptions, start_date, end_date)
    except Exception:
        return False


def tile_has_complete_output(tile_dir, start_date, end_date):
    if not tile_dir.is_dir():
        return False
    tif_files = sorted(tile_dir.glob("*.tif"))
    if not tif_files:
        return False
    return any(force_tile_is_complete(tif_file, start_date, end_date) for tif_file in tif_files)


def generate_tiles_to_process(process_folder, project_name, date_range):
    output_root = Path(process_folder) / "temp" / project_name / "tiles_tss"
    tile_extent_file = Path(process_folder) / "temp" / project_name / "tile_extent.txt"
    output_tile_list = Path(process_folder) / "temp" / project_name / "provenance" / "resume_tiles.txt"
    start_date, end_date = parse_date_range(date_range)

    if not tile_extent_file.is_file():
        raise FileNotFoundError(f"Tile extent file not found: {tile_extent_file}")

    with open(tile_extent_file, 'r') as file:
        lines = file.readlines()
        all_tiles = [line.strip() for line in lines if line.strip()][1:]

    all_tiles = sorted(set(all_tiles))
    tiles_to_process = []
    completed_tiles = 0

    for tile in all_tiles:
        tile_dir = output_root / tile
        if tile_has_complete_output(tile_dir, start_date, end_date):
            completed_tiles += 1
        else:
            tiles_to_process.append(tile)

    output_tile_list.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tile_list, 'w') as file:
        file.write(f"{len(tiles_to_process)}\n")
        for tile in tiles_to_process:
            file.write(f"{tile}\n")

    print(
        f"Tiles complete: {completed_tiles} | "
        f"tiles remaining: {len(tiles_to_process)} | "
        f"resume list written to: {output_tile_list}"
    )
    return output_tile_list, len(tiles_to_process)


def should_run_mask_stage(mask_project_dir, aoi_basename):
    mosaic_tif = mask_project_dir / aoi_basename.replace(".shp", ".tif")
    return not raster_is_readable(mosaic_tif)

def force_baresoil(project_name,aoi,TSS_Sensors,TSS_DATE_RANGE,process_folder,force_dir,TSS_SPECTRAL_ADJUST,TSS_ABOVE_NOISE,TSS_BELOW_NOISE,hold,TSS_NTHREAD_READ,TSS_NTHREAD_COMPUTE,
                   TSS_NTHREAD_WRITE,TSS_BLOCK_SIZE,**kwargs):

    force_dir = f"{force_dir}:{force_dir}"
    local_dir = f"{os.sep + process_folder.split(os.sep)[1]}:{os.sep + process_folder.split(os.sep)[1]}"
    scripts_skel = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/force/skel"
    force_skel = f"{scripts_skel}/force_cube_sceleton"
    temp_folder = process_folder + "/temp"
    mask_folder = process_folder + "/temp/_mask"
    project_temp_dir = Path(temp_folder) / project_name
    mask_project_dir = Path(mask_folder) / project_name
    log_dir = project_temp_dir / "provenance" / "logs"

    startzeit = time.time()

    aoi = check_and_reproject_shapefile(aoi)
    ### get force extend
    os.makedirs(project_temp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    subprocess.run(['sudo', 'chmod', '-R', '777', f"{temp_folder}"])

    shutil.copy(f"{force_skel}/datacube-definition.prj",f"{project_temp_dir}/datacube-definition.prj")

    tile_extent_path = project_temp_dir / "tile_extent.txt"
    if not tile_extent_path.is_file():
        cmd = f"sudo docker run -v {local_dir} -v {force_dir} davidfrantz/force " \
               f"force-tile-extent {aoi} {force_skel} {tile_extent_path}"
        run_shell_command(
            cmd,
            hold=hold,
            stage_name="force-tile-extent",
            log_path=log_dir / "force-tile-extent.log",
        )
    else:
        print(f"Reusing existing tile extent file: {tile_extent_path}")

    subprocess.run(['sudo', 'chmod', '-R', '777', f"{project_temp_dir}"])

    ### mask

    os.makedirs(mask_project_dir, exist_ok=True)
    shutil.copy(f"{force_skel}/datacube-definition.prj",f"{mask_project_dir}/datacube-definition.prj")

    if should_run_mask_stage(mask_project_dir, os.path.basename(aoi)):
        cmd = f"sudo docker run -v {local_dir} davidfrantz/force " \
              f"force-cube -o {mask_project_dir} " \
              f"{aoi}"
        run_shell_command(
            cmd,
            hold=hold,
            stage_name="force-cube mask generation",
            log_path=log_dir / "force-cube.log",
        )

        cmd = f"sudo docker run -v {local_dir} davidfrantz/force " \
              f"force-mosaic {mask_project_dir}"
        run_shell_command(
            cmd,
            hold=hold,
            stage_name="force-mosaic mask merge",
            log_path=log_dir / "force-mosaic.log",
        )
    else:
        print(f"Reusing existing mask mosaic in: {mask_project_dir}")

    subprocess.run(['sudo', 'chmod', '-R', '777', f"{mask_project_dir}"])

    #analysis_tss
    ###force param
    os.makedirs(project_temp_dir, exist_ok=True)
    os.makedirs(project_temp_dir / "provenance", exist_ok=True)
    os.makedirs(project_temp_dir / "tiles_tss", exist_ok=True)
    shutil.copy(f"{force_skel}/datacube-definition.prj",f"{project_temp_dir}/datacube-definition.prj")
    shutil.copy(f"{force_skel}/datacube-definition.prj",f"{project_temp_dir}/tiles_tss/datacube-definition.prj")
    shutil.copy(f"{scripts_skel}/UDF_NoCom.prm", f"{project_temp_dir}/pvir_tss.prm")
    shutil.copy(f"{scripts_skel}/pvir_skel_tss.py",f"{project_temp_dir}/pvir_tss.py")

    resume_tile_path, tiles_remaining = generate_tiles_to_process(process_folder, project_name, TSS_DATE_RANGE)
    if tiles_remaining == 0:
        print("All FORCE tiles are already complete. Skipping force-higher-level.")
        endzeit = time.time()
        print("FORCE-Processing beendet nach "+str((endzeit-startzeit)/60)+" Minuten")
        return

    X_TILE_RANGE, Y_TILE_RANGE = extract_coordinates(str(tile_extent_path))
    # Define replacements
    replacements = {
        # INPUT/OUTPUT DIRECTORIES
        f'DIR_LOWER = NULL':f'DIR_LOWER = {force_dir.split(":")[0]}/FORCE/C1/L2/ard',
        f'DIR_HIGHER = NULL':f'DIR_HIGHER = {project_temp_dir}/tiles_tss',
        f'DIR_PROVENANCE = NULL':f'DIR_PROVENANCE = {project_temp_dir}/provenance',
        # MASKING
        f'DIR_MASK = NULL':f'DIR_MASK = {mask_project_dir}',
        f'BASE_MASK = NULL':f'BASE_MASK = {os.path.basename(aoi).replace(".shp",".tif")}',
        # PARALLEL PROCESSING
        f'NTHREAD_READ = 8':f'NTHREAD_READ = {TSS_NTHREAD_READ}',
        f'NTHREAD_COMPUTE = 22':f'NTHREAD_COMPUTE = {TSS_NTHREAD_COMPUTE}',
        f'NTHREAD_WRITE = 4':f'NTHREAD_WRITE = {TSS_NTHREAD_WRITE}',
        # PROCESSING EXTENT AND RESOLUTION
        f'X_TILE_RANGE = 0 0':f'X_TILE_RANGE = {X_TILE_RANGE}',
        f'Y_TILE_RANGE = 0 0':f'Y_TILE_RANGE = {Y_TILE_RANGE}',
        f'FILE_TILE = NULL':f'FILE_TILE = {resume_tile_path}',
        f'BLOCK_SIZE = 0':f'BLOCK_SIZE = {TSS_BLOCK_SIZE}',
        # SENSOR ALLOW-LIST
        f'SENSORS = LND08 LND09 SEN2A SEN2B':f'SENSORS = {TSS_Sensors}',
        f'SPECTRAL_ADJUST = FALSE':f'SPECTRAL_ADJUST = {TSS_SPECTRAL_ADJUST}',
        # QAI SCREENING
        f'SCREEN_QAI = NODATA CLOUD_OPAQUE CLOUD_BUFFER CLOUD_CIRRUS CLOUD_SHADOW SNOW SUBZERO SATURATION':f'SCREEN_QAI = NODATA CLOUD_OPAQUE CLOUD_BUFFER CLOUD_CIRRUS CLOUD_SHADOW SNOW SUBZERO SATURATION',
        f'ABOVE_NOISE = 3':f'ABOVE_NOISE = {TSS_ABOVE_NOISE}',
        f'BELOW_NOISE = 1':f'BELOW_NOISE = {TSS_BELOW_NOISE}',
        # PROCESSING TIMEFRAME
        f'DATE_RANGE = 2010-01-01 2019-12-31':f'DATE_RANGE = {TSS_DATE_RANGE}',
        # PYTHON UDF PARAMETERS
        f'FILE_PYTHON = NULL':f'FILE_PYTHON = {project_temp_dir}/pvir_tss.py',
        f'PYTHON_TYPE = PIXEL':f'PYTHON_TYPE = PIXEL',
        f'OUTPUT_PYP = FALSE': f'OUTPUT_PYP = TRUE',
    }


    # Replace parameters in the file
    replace_parameters(f"{project_temp_dir}/pvir_tss.prm", replacements)

    cmd = f"sudo docker run -v {local_dir} -v {force_dir} davidfrantz/force " \
          f"force-higher-level {project_temp_dir}/pvir_tss.prm"

    subprocess.run(['sudo', 'chmod', '-R', '777', f"{temp_folder}"])
    run_shell_command(
        cmd,
        hold=hold,
        stage_name=f"force-higher-level ({tiles_remaining} tiles remaining)",
        log_path=log_dir / "force-higher-level.log",
    )

    subprocess.run(['sudo', 'chmod', '-R', '777', f"{project_temp_dir}"])
    endzeit = time.time()
    print("FORCE-Processing beendet nach "+str((endzeit-startzeit)/60)+" Minuten")
