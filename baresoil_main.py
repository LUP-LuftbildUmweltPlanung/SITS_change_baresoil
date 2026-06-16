# -*- coding: utf-8 -*-
"""

@author: sebastianvalencia (git Azarozo19)

"""
import time
import cProfile, pstats

from force.force_baresoil_utils import *
from utils.baresoil_utils import *

#### Default Index PVIR2 ####
PROCESS_FOLDER = "/rvt_mount/process"

force_params = {
    #########################
    #########Basics##########
    #########################
    "project_name": "bare_soil_filter_prmfile_", #Project Name that will be the name of output folder in temp & result subfolder test_full_tile_all_time
    "aoi": "/rvt_mount/3DTests/data/bare_soil/3tiles.shp", #Define Area of Interest as Shapefile

    #TimeSeriesStack (TSS) --> Real Spectral Values
    "TSS_Sensors": "SEN2A SEN2B", # Choose between Input Sensors
    "TSS_DATE_RANGE": "2018-01-01 2018-12-31",# TimeRange for index calculation.
}

force_advanced_params = {
    #BASIC
    "process_folder": PROCESS_FOLDER, # Folder where Data and Results will be processed (will be created if not existing)
    "force_dir": "/force", # mount directory for FORCE-Datacube - should look like /force_mount/FORCE/C1/L2/..

    # To disable filter set TS*_ABOVE_NOISE and! TS*_BELOW_NOISE to 0; it's recommended to TSS_ABOVE_NOISE and TSS_BELOW_NOISE to 0 to include all values and get comparable results
    "TSS_ABOVE_NOISE": 0, # noise filtering in spectral values above 3 x std; take care for not filtering real changes
    "TSS_BELOW_NOISE": 0, # get back values from qai masking below single std
    "TSS_SPECTRAL_ADJUST": "FALSE", #spectral adjustment will be necessary by using Sentinel 2 & Landsat together

    "hold": False,  # if True, cmd must be closed manually ## recommended for debugging FORCE

    #Streaming Mechanism
    "TSS_NTHREAD_READ": 11,
    "TSS_NTHREAD_COMPUTE": 22,
    "TSS_NTHREAD_WRITE": 8,
    "TSS_BLOCK_SIZE": 3000,
}

analysis_params = {
    ###########################
    ## Bare Soil Aggregation ###
    ###########################
    "project_name": force_params["project_name"],
    "bare_soil_lower": 173,   # values <= this threshold count as bare soil
    "bare_soil_upper": 1371,  # values >= this threshold count as bare soil
    "min_consecutive": 3,     # require at least this many consecutive valid detections
    "mosaic": True,           # Mosaic the per-tile results
    "aoi_path": force_params["aoi"], # Clip and export results using the same AOI-driven workflow as SITS_mowing
    "overwrite_results": True, # Whether to overwrite existing results (if False, will skip processing if results already exist)
    "debug_stats": True,       # Print additional statistics for debugging purposes
    "cleanup_tss": True,      # Remove intermediate TSS & mask tiles after aggregation
    "write_stabilized_output": False, # Write an additional output with median-filtered bare-soil occurrences
    "stabilized_filter_size": 3, # Median filter size for stabilized output
    "write_weighted_output": False, # Write an additional weighted product based on distance to the thresholds
    "weighted_threshold_scale": 150, # Distance scale for weighted detections; larger values make the weighting more gradual
}

analysis_advanced_params = {
    "process_folder": PROCESS_FOLDER,
    "tss_lst": None, # tss stacks will be automatically discovered inside the project folder
}

run_flags = {
    "run_force": True,
    "run_analysis": True,
}

def format_time(seconds):
    """Format the time in hours, minutes, and seconds."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


if __name__ == '__main__':
    #Uncomment the ones below for debugging purposses
    # profiler = cProfile.Profile()
    # profiler.enable()

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

    # Total time
    total_time = force_baresoil_time + baresoil_time
    print(f"Total execution time: {format_time(total_time)}")
    print("Force params:", force_params)
    print("Analysis params:", analysis_params)
    print("Run flags:", run_flags)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats()
    # stats.dump_stats("profile_results_fnq.prof")
