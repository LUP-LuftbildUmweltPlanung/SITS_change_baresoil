import os
import subprocess
import time
import shutil
import geopandas as gpd

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

def force_baresoil(project_name,aoi,TSS_Sensors,TSS_DATE_RANGE,process_folder,force_dir,TSS_SPECTRAL_ADJUST,TSS_ABOVE_NOISE,TSS_BELOW_NOISE,hold,TSS_NTHREAD_READ,TSS_NTHREAD_COMPUTE,
                   TSS_NTHREAD_WRITE,TSS_BLOCK_SIZE,**kwargs):

    force_dir = f"{force_dir}:{force_dir}"
    local_dir = f"{os.sep + process_folder.split(os.sep)[1]}:{os.sep + process_folder.split(os.sep)[1]}"
    scripts_skel = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/force/skel"
    force_skel = f"{scripts_skel}/force_cube_sceleton"
    temp_folder = process_folder + "/temp"
    mask_folder = process_folder + "/temp/_mask"

    startzeit = time.time()

    aoi = check_and_reproject_shapefile(aoi)
    ### get force extend
    os.makedirs(f"{temp_folder}/{project_name}", exist_ok=True)

    subprocess.run(['sudo', 'chmod', '-R', '777', f"{temp_folder}"])

    shutil.copy(f"{force_skel}/datacube-definition.prj",f"{temp_folder}/{project_name}/datacube-definition.prj")

    cmd = f"sudo docker run -v {local_dir} -v {force_dir} davidfrantz/force " \
           f"force-tile-extent {aoi} {force_skel} {temp_folder}/{project_name}/tile_extent.txt"

    if hold == True:
        subprocess.run(['xterm','-hold','-e', cmd])
    else:
        subprocess.run(['xterm', '-e', cmd])

    subprocess.run(['sudo', 'chmod', '-R', '777', f"{temp_folder}/{project_name}"])

    ### mask

    os.makedirs(f"{mask_folder}/{project_name}", exist_ok=True)
    shutil.copy(f"{force_skel}/datacube-definition.prj",f"{mask_folder}/{project_name}/datacube-definition.prj")

    cmd = f"sudo docker run -v {local_dir} davidfrantz/force " \
          f"force-cube -o {mask_folder}/{project_name} " \
          f"{aoi}"

    if hold == True:
        subprocess.run(['xterm','-hold','-e', cmd])
    else:
        subprocess.run(['xterm', '-e', cmd])


    ###mask mosaic
    cmd = f"sudo docker run -v {local_dir} davidfrantz/force " \
          f"force-mosaic {mask_folder}/{project_name}"

    if hold == True:
        subprocess.run(['xterm','-hold','-e', cmd])
    else:
        subprocess.run(['xterm', '-e', cmd])

    subprocess.run(['sudo', 'chmod', '-R', '777', f"{mask_folder}/{project_name}"])



    #analysis_tss
    ###force param
    os.makedirs(f"{temp_folder}/{project_name}", exist_ok=True)
    os.makedirs(f"{temp_folder}/{project_name}/provenance", exist_ok=True)
    os.makedirs(f"{temp_folder}/{project_name}/tiles_tss", exist_ok=True)
    shutil.copy(f"{force_skel}/datacube-definition.prj",f"{temp_folder}/{project_name}/datacube-definition.prj")
    shutil.copy(f"{force_skel}/datacube-definition.prj",f"{temp_folder}/{project_name}/tiles_tss/datacube-definition.prj")
    shutil.copy(f"{scripts_skel}/UDF_NoCom.prm", f"{temp_folder}/{project_name}/pvir_tss.prm")
    shutil.copy(f"{scripts_skel}/pvir_skel_tss.py",f"{temp_folder}/{project_name}/pvir_tss.py")

    X_TILE_RANGE, Y_TILE_RANGE = extract_coordinates(f"{temp_folder}/{project_name}/tile_extent.txt")
    # Define replacements
    replacements = {
        # INPUT/OUTPUT DIRECTORIES
        f'DIR_LOWER = NULL':f'DIR_LOWER = {force_dir.split(":")[0]}/FORCE/C1/L2/ard',
        f'DIR_HIGHER = NULL':f'DIR_HIGHER = {temp_folder}/{project_name}/tiles_tss',
        f'DIR_PROVENANCE = NULL':f'DIR_PROVENANCE = {temp_folder}/{project_name}/provenance',
        # MASKING
        f'DIR_MASK = NULL':f'DIR_MASK = {mask_folder}/{project_name}',
        f'BASE_MASK = NULL':f'BASE_MASK = {os.path.basename(aoi).replace(".shp",".tif")}',
        # PARALLEL PROCESSING
        f'NTHREAD_READ = 8':f'NTHREAD_READ = {TSS_NTHREAD_READ}',
        f'NTHREAD_COMPUTE = 22':f'NTHREAD_COMPUTE = {TSS_NTHREAD_COMPUTE}',
        f'NTHREAD_WRITE = 4':f'NTHREAD_WRITE = {TSS_NTHREAD_WRITE}',
        # PROCESSING EXTENT AND RESOLUTION
        f'X_TILE_RANGE = 0 0':f'X_TILE_RANGE = {X_TILE_RANGE}',
        f'Y_TILE_RANGE = 0 0':f'Y_TILE_RANGE = {Y_TILE_RANGE}',
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
        f'FILE_PYTHON = NULL':f'FILE_PYTHON = {temp_folder}/{project_name}/pvir_tss.py',
        f'PYTHON_TYPE = PIXEL':f'PYTHON_TYPE = PIXEL',
        f'OUTPUT_PYP = FALSE': f'OUTPUT_PYP = TRUE',
    }


    # Replace parameters in the file
    replace_parameters(f"{temp_folder}/{project_name}/pvir_tss.prm", replacements)

    cmd = f"sudo docker run -it -v {local_dir} -v {force_dir} davidfrantz/force " \
          f"force-higher-level {temp_folder}/{project_name}/pvir_tss.prm"

    subprocess.run(['sudo', 'chmod', '-R', '777', f"{temp_folder}"])
    if hold == True:
        subprocess.run(['xterm', '-hold', '-e', cmd])
    else:
        subprocess.run(['xterm', '-e', cmd])

    subprocess.run(['sudo', 'chmod', '-R', '777', f"{temp_folder}/{project_name}"])
    endzeit = time.time()
    print("FORCE-Processing beendet nach "+str((endzeit-startzeit)/60)+" Minuten")
