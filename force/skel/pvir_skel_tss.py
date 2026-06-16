import numpy as np
from datetime import timedelta, date

# some global config variables
start = date.fromisoformat('1970-01-01')
BAND_INDEX = {}


def _get_band_index(bandnames, target_name):
    try:
        return BAND_INDEX[target_name]
    except KeyError:
        matches = np.flatnonzero(bandnames == target_name)
        if matches.size == 0:
            raise ValueError(f"Band '{target_name.decode('utf-8')}' not found in FORCE bandnames")
        index = int(matches[0])
        BAND_INDEX[target_name] = index
        return index

def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """

    base = [f'{(start+timedelta(days=int(dat))).year}{(start+timedelta(days=int(dat))).month:02d}{(start+timedelta(days=int(dat))).day:02d}_{sens.decode("utf-8")}' for dat, sens in zip(dates, sensors)]
    mask = [f"{label}_MASK" for label in base]
    BAND_INDEX.clear()
    for band_name in (b'RED', b'NIR', b'BROADNIR', b'SWIR1', b'SWIR2'):
        _get_band_index(bandnames, band_name)
    return base + mask


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
    inarray = inarray[:, :, 0, 0].astype(np.float32, copy=False)
    valid_mask = inarray[:, 0] != nodata
    if not np.any(valid_mask):
        return
    valid_idx = np.flatnonzero(valid_mask)
    vals = inarray[valid_idx, :]

    red = _get_band_index(bandnames, b'RED')
    bnir = _get_band_index(bandnames, b'BROADNIR')
    swir2 = _get_band_index(bandnames, b'SWIR2')

    fill = np.full(vals.shape[0], np.nan, dtype=np.float32)
    ndvi_like = np.divide(
        vals[:, bnir] - vals[:, red],
        vals[:, bnir] + vals[:, red],
        out=fill.copy(),
        where=(vals[:, bnir] + vals[:, red]) != 0
    )
    nbr2_like = np.divide(
        vals[:, bnir] - vals[:, swir2],
        vals[:, bnir] + vals[:, swir2],
        out=fill.copy(),
        where=(vals[:, bnir] + vals[:, swir2]) != 0
    )
    pvir2 = ndvi_like + nbr2_like

    nir_swir2_ratio = np.divide(
        vals[:, bnir],
        vals[:, swir2],
        out=fill.copy(),
        where=vals[:, swir2] != 0
    )

    valid_bands = (
        np.isfinite(vals[:, red]) &
        np.isfinite(vals[:, bnir]) &
        np.isfinite(vals[:, swir2]) &
        (vals[:, red] > 0) &
        (vals[:, bnir] > 0) &
        (vals[:, swir2] > 0)
    )
    valid_condition = valid_bands & np.isfinite(pvir2) & (nir_swir2_ratio >= 0.02)

    n_dates = inarray.shape[0]
    value_idx = np.arange(n_dates)
    mask_idx = value_idx + n_dates

    outarray[value_idx] = nodata
    outarray[mask_idx] = 0

    scaled_pvir = pvir2 * 1000
    outarray[value_idx[valid_idx]] = scaled_pvir
    outarray[mask_idx[valid_idx[valid_condition]]] = 1
