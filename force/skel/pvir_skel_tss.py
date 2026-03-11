import numpy as np
import time
from datetime import datetime, timedelta, date
# some global config variables
start = date.fromisoformat('1970-01-01')

def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """

    return [f'{(start+timedelta(days=int(dat))).year}{(start+timedelta(days=int(dat))).month:02d}{(start+timedelta(days=int(dat))).day:02d}_{sens.decode("utf-8")}' for dat, sens in zip(dates, sensors)]


def forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
    """
    inarray:   numpy.ndarray[nDates, nBands, nrows, ncols](Int16)
    outarray:  numpy.ndarray[nOutBands](Int16) initialized with no data values
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    nodata:    int
    nproc:     number of allowed processes/threads
    Write results into outarray.
    """

    inarray = inarray.astype(np.float32)
    inarray = inarray[:, :, 0, 0]
    invalid = inarray == nodata  ## bool mask values
    ## bool qai masks

    valid = np.where(inarray[:, 0] != nodata)[0]  # skip no data; just check first band

    if len(valid) == 0:
        return
    inarray[invalid] = np.nan

    #BANDS
    #green = np.argwhere(bandnames == b'GREEN')[0][0]
    #blue = np.argwhere(bandnames == b'BLUE')[0][0]
    red = np.argwhere(bandnames == b'RED')[0][0]
    # re1 = np.argwhere(bandnames == b'REDEDGE1')[0][0]
    # re2 = np.argwhere(bandnames == b'REDEDGE2')[0][0]
    nir = np.argwhere(bandnames == b'NIR')[0][0]
    bnir = np.argwhere(bandnames == b'BROADNIR')[0][0]
    swir1 = np.argwhere(bandnames == b'SWIR1')[0][0]
    swir2 = np.argwhere(bandnames == b'SWIR2')[0][0]

    vals = inarray[valid, :]

    #INDEXES

    # dswi = (vals[:,nir] + vals[:,green]) / (vals[:,swir1] + vals[:,red])
    # # NBR = (BNIR - SWIR2) / (BNIR + SWIR2)
    # nbr = (vals[:, bnir] - vals[:, swir2]) / (vals[:, bnir] + vals[:, swir2])
    # # NDVI = (BNIR - RED) / (BNIR + RED)
    # ndvi = (vals[:, bnir] - vals[:, red]) / (vals[:, bnir] + vals[:, red])
    pvir2 = (((vals[:, bnir] - vals[:, red]) / (vals[:, bnir] + vals[:, red])) + ((vals[:, bnir] - vals[:, swir2]) / (vals[:, bnir] + vals[:, swir2])))
    # # ARI = BNIR * ((1 / GREEN) - (1 / RE1))
    # ari = vals[:, bnir] * ((1 / vals[:, green]) - (1 / vals[:, re1]))
    # # CRI = (1 / BLUE) - (1 / GREEN)
    #ndvi = (vals[:, nir] - vals[:, red]) / (vals[:, nir] + vals[:, red])
    #dswi = (vals[:,nir] + vals[:,green]) / (vals[:,swir1] + vals[:,red])
    # # calculate NDTI ((Band 11(SWIR1) - Band 12(SWIR2)) / Band 11(SWIR1) + Band 12(SWIR2)))
    #ndti = (vals[:, swir1] - vals[:, swir2]) / (vals[:, swir1] + vals[:, swir2])
    #ndti[np.isnan(ndti)] = nodata

    swir_nir_ratio = np.divide(
        vals[:, swir1] - vals[:, nir],
        vals[:, swir1] + vals[:, nir],
        out=np.full(vals.shape[0], np.nan, dtype=np.float32),
        where=(vals[:, swir1] + vals[:, nir]) != 0
    )

    valid_condition = swir_nir_ratio >= 0.02

    scaled_pvir = pvir2 * 1000
    outarray[valid[valid_condition]] = scaled_pvir[valid_condition]
