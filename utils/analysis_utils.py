import os
import time
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import fastnanquantile as fnq
import rasterio
from rasterio.merge import merge
import glob
from scipy.stats import linregress
import gc
from utils.residuals_utils import extract_data
from utils.residuals_utils import get_output_array_full
from utils.residuals_utils import write_output_raster
from utils.residuals_utils import slice_by_date
from utils.residuals_utils import calculate_residuals


def calculate_firstdate_whole(output_array_full, dates_nrt, forest_mask):
    print("###" * 10)
    print(f'calculate first date (residual related)\n')

    # Find the indices of the first non-NaN value along the third dimension
    first_non_nan_indices = np.argmax(~np.isnan(output_array_full), axis=2)
    # Initialize the output array to NaN
    output_array = np.full((3000, 3000), np.nan)

    # Set the values in the output array
    for x in range(output_array.shape[0]):
        for y in range(output_array.shape[1]):
            index = first_non_nan_indices[x, y]
            if not np.isnan(output_array_full[x, y, index]):
                output_array[x, y] = int(dates_nrt[index].replace('-', '')[2:])

    output_array[np.isnan(output_array)] = -32768
    missing_values = np.logical_and(output_array == -32768, ~forest_mask)
    output_array[missing_values] = 5000
    output_array = output_array.astype(int)

    return output_array


def calculate_int10p_whole(output_array_full, forest_mask):
    ############## Intensity and Count
    print("###" * 10)
    print(f'calculate intensity and count for whole time period (residual related)\n')
    # count number of positive and negative values in array
    positive_values = np.sum((output_array_full > 0), axis=2)
    negative_values = np.sum(output_array_full <= 0, axis=2)

    # create mask: if True = more positive than negative values; if false more negative than positive values
    mask_positive = positive_values > negative_values

    # initialise output array
    a_p10 = np.empty(output_array_full.shape[:2])  # Vorbelegen mit 9999

    # calculate 90 percentil for all pixel with more positive values within period
    if np.any(mask_positive):
        a_p10[mask_positive] = fnq.nanquantile(
            output_array_full[mask_positive], 0.9, axis=1
        )

    # calculate 10 percentil for all pixel with more negative values within period
    mask_negative = ~mask_positive
    if np.any(mask_negative):
        a_p10[mask_negative] = fnq.nanquantile(
            output_array_full[mask_negative], 0.1, axis=1
        )
    #a_p10 = np.nanpercentile(output_array_full, 10, axis=2)
    # Fill NaN values
    a_p10[np.isnan(a_p10)] = -32768
    missing_values = np.logical_and(a_p10 == -32768, ~forest_mask)
    a_p10[missing_values] = 5000

    mask_5000_9999 = np.isin(a_p10, [5000, -32768])
    a_p10_clipped = np.clip(a_p10, -4000, 4000)
    a_p10_clipped[mask_5000_9999] = a_p10[mask_5000_9999]
    a_p10 = a_p10_clipped.astype(np.int16)
    #a_p10 = a_p10.astype(int32)


    ## uncomment if you want to have information about amount of valid values
    # counts=(~np.isnan(output_array_full)).sum(axis=2)
    # counts[np.isnan(counts)]= 9999
    # counts = counts.astype(int)
    # write_output_raster(residuals_nrt,output, counts, f"\\count.tif",1)
    # a_p10[counts<20] = 9999
    # write_output_raster(residuals_nrt,output, a_p10, f"\\intensity_count_o20.tif",1)

    return a_p10


def calculate_intp10_period (raster_tss, output, start_date, end_date, period_length, dates_nrt, output_array_full, forest_mask, filtered=None, mode=None):
    ###############################################################
    ################ ITERATE OVER TIME PERIODS ####################
    ###############################################################

    # Loop over the date range and slice the data
    date = start_date
    print("###" * 10)
    print(f' starting slicing for time periods! (residual related) \n start date: {start_date} \n end date: {end_date} \n period length: {period_length} month')
    while date < end_date:

        # Print the start and end months for the current slice
        start_month = datetime.strptime(date, '%Y-%m').strftime('%B %Y')
        end_month = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length - 1)).strftime('%B %Y')

        sm_split = start_month.split(" ")
        em_split = end_month.split(" ")

        ## uncomment or adjust if some periods should be skipped
        if sm_split[0] == 'December':
            print(f"period started with December skipped ...")
            date = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length)).strftime('%Y-%m')
            continue

        # Slice the data for the current date range
        sliced_array = slice_by_date(output_array_full, dates_nrt, date, period_length)
        # sliced_array_thresh = slice_by_date(output_array_full_nothresh, dates_nrt, date, period_length)
        if mode == "thresholding":
            sliced_filter = slice_by_date(filtered, dates_nrt, date, period_length)
        # Move to the next date range
        date = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length)).strftime('%Y-%m')
        print("###" * 10)
        print(f'\n Sliced array for {start_month} - {end_month}')
        print(f'array shape: {sliced_array.shape}')

        # do the calculations
        print(f'calculate intensity ...')
        # a_p10 = np.nanpercentile(sliced_array, 10, axis=2)
        # sliced_array[sliced_array > 0] = np.nan
        startzeit_force = time.time()
        # force_harmonic(**params, **advanced_params)


        #######################################################
        sliced_array = np.array(sliced_array)

        # count number of positive and negative values in array
        positive_values = np.sum((sliced_array > 0), axis=2)
        negative_values = np.sum(sliced_array <= 0, axis=2)

        # create mask: if True = more positive than negative values; if false more negative than positive values
        mask_positive = positive_values > negative_values

        # initialise output array
        a_p10 = np.empty(sliced_array.shape[:2])  # Vorbelegen mit 9999

        # calculate 90 percentil for all pixel with more positive values within period
        if np.any(mask_positive):
            a_p10[mask_positive] = fnq.nanquantile(
                sliced_array[mask_positive], 0.9, axis=1
            )

        # calculate 10 percentil for all pixel with more negative values within period
        mask_negative = ~mask_positive
        if np.any(mask_negative):
            a_p10[mask_negative] = fnq.nanquantile(
                sliced_array[mask_negative], 0.1, axis=1
            )

        # change to integer
        a_p10[np.isnan(a_p10)] = -32768
        a_p10 = a_p10.astype(int)

#########################################################################################################

        if mode == "thresholding":
            sliced_filter_2d = np.any(sliced_filter, axis=2)
            sliced_filter_2d_nan = np.logical_and(a_p10 == -32768, sliced_filter_2d)
            # print(sum(sliced_filter_2d))
            a_p10[
                sliced_filter_2d_nan] = -32768  # Should be set to e.g. 0 if you want to seperate areas outside mask and no disturbance

        missing_values = np.logical_and(a_p10 == -32768, ~forest_mask)
        a_p10[missing_values] = 5000

        mask_5000_9999 = np.isin(a_p10, [5000, -32768])
        a_p10_clipped = np.clip(a_p10, -4000, 4000)
        a_p10_clipped[mask_5000_9999] = a_p10[mask_5000_9999]
        a_p10 = a_p10_clipped.astype(np.int16)
        #a_p10 = a_p10.astype(np.int32)

        # convert month strings to number
        month_map = {
            "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
            "july": "07", "august": "08","september": "09", "october": "10", "november": "11", "december": "12"
        }
        sm_month = month_map[sm_split[0].lower()]
        em_month = month_map[em_split[0].lower()]

        suffix = f"/vit_{sm_split[1]}{sm_month}_{em_split[1]}{em_month}_{mode}_v3.tif"

        write_output_raster(raster_tss, output, a_p10, suffix, 1)
