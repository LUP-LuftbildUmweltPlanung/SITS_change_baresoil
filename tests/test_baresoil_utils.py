import datetime as dt
import numpy as np
import rasterio
from rasterio.transform import from_origin

from force.force_baresoil_utils import force_tile_is_complete, generate_tiles_to_process
from utils.baresoil_utils import _compute_bare_soil_counts


def _write_raster(path, bands, descriptions, dtype="int16", nodata=-9999):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=len(bands),
        dtype=dtype,
        transform=from_origin(0, 1, 1, 1),
        nodata=nodata,
    ) as dst:
        for idx, band in enumerate(bands, start=1):
            dst.write(np.array([[band]], dtype=dtype), idx)
            dst.set_band_description(idx, descriptions[idx - 1])


def test_compute_bare_soil_counts_with_inline_mask(tmp_path):
    raster_path = tmp_path / "inline_mask.tif"
    _write_raster(
        raster_path,
        bands=[100, 100, 100, 1, 1, 0],
        descriptions=[
            "20200101_SEN2A",
            "20200111_SEN2A",
            "20200121_SEN2A",
            "20200101_SEN2A_MASK",
            "20200111_SEN2A_MASK",
            "20200121_SEN2A_MASK",
        ],
    )

    dates, total_valid, bare_occurrences, weighted_occurrences = _compute_bare_soil_counts(
        str(raster_path),
        None,
        lower=173,
        upper=1375,
        min_consecutive=2,
        weighted_threshold_scale=150,
    )

    assert dates == ["2020-01-01", "2020-01-11", "2020-01-21"]
    np.testing.assert_array_equal(total_valid, np.array([[3]], dtype=np.uint16))
    np.testing.assert_array_equal(bare_occurrences, np.array([[2]], dtype=np.uint16))
    np.testing.assert_allclose(weighted_occurrences, np.array([[0.97333336]], dtype=np.float32))


def test_compute_bare_soil_counts_with_external_mask(tmp_path):
    raster_path = tmp_path / "values.tif"
    mask_path = tmp_path / "values_mask.tif"
    _write_raster(
        raster_path,
        bands=[100, 100, 100],
        descriptions=[
            "20200101_SEN2A",
            "20200111_SEN2A",
            "20200121_SEN2A",
        ],
    )
    _write_raster(
        mask_path,
        bands=[1, 1, 0],
        descriptions=[
            "20200101_SEN2A_MASK",
            "20200111_SEN2A_MASK",
            "20200121_SEN2A_MASK",
        ],
        dtype="uint8",
        nodata=0,
    )

    dates, total_valid, bare_occurrences, weighted_occurrences = _compute_bare_soil_counts(
        str(raster_path),
        str(mask_path),
        lower=173,
        upper=1375,
        min_consecutive=2,
        weighted_threshold_scale=150,
    )

    assert dates == ["2020-01-01", "2020-01-11", "2020-01-21"]
    np.testing.assert_array_equal(total_valid, np.array([[3]], dtype=np.uint16))
    np.testing.assert_array_equal(bare_occurrences, np.array([[2]], dtype=np.uint16))
    np.testing.assert_allclose(weighted_occurrences, np.array([[0.48666668]], dtype=np.float32))


def test_force_tile_is_complete_rejects_out_of_range_dates(tmp_path):
    raster_path = tmp_path / "bad_dates.tif"
    _write_raster(
        raster_path,
        bands=[100, 100, 1, 1],
        descriptions=[
            "20191231_SEN2A",
            "20200111_SEN2A",
            "20191231_SEN2A_MASK",
            "20200111_SEN2A_MASK",
        ],
    )

    assert not force_tile_is_complete(raster_path, dt.date(2020, 1, 1), dt.date(2020, 12, 31))


def test_force_tile_is_complete_rejects_invalid_mask_layout(tmp_path):
    raster_path = tmp_path / "bad_mask_layout.tif"
    _write_raster(
        raster_path,
        bands=[100, 100, 1, 1],
        descriptions=[
            "20200101_SEN2A",
            "20200111_SEN2A",
            "20200101_SEN2A_MASK",
            "20200111_SEN2B_MASK",
        ],
    )

    assert not force_tile_is_complete(raster_path, dt.date(2020, 1, 1), dt.date(2020, 12, 31))


def test_generate_tiles_to_process_uses_strict_force_validation(tmp_path):
    process_folder = tmp_path
    project_name = "resume_strict"
    project_dir = process_folder / "temp" / project_name
    tiles_root = project_dir / "tiles_tss"
    provenance_dir = project_dir / "provenance"
    good_tile_dir = tiles_root / "X0001_Y0001"
    bad_tile_dir = tiles_root / "X0001_Y0002"
    good_tile_dir.mkdir(parents=True)
    bad_tile_dir.mkdir(parents=True)
    provenance_dir.mkdir(parents=True)

    (project_dir / "tile_extent.txt").write_text("2\nX0001_Y0001\nX0001_Y0002\n")

    _write_raster(
        good_tile_dir / "good.tif",
        bands=[100, 100, 1, 1],
        descriptions=[
            "20200101_SEN2A",
            "20200111_SEN2A",
            "20200101_SEN2A_MASK",
            "20200111_SEN2A_MASK",
        ],
    )
    _write_raster(
        bad_tile_dir / "bad.tif",
        bands=[100, 100, 1, 1],
        descriptions=[
            "20191231_SEN2A",
            "20200111_SEN2A",
            "20191231_SEN2A_MASK",
            "20200111_SEN2A_MASK",
        ],
    )

    resume_path, remaining = generate_tiles_to_process(str(process_folder), project_name, "2020-01-01 2020-12-31")

    assert remaining == 1
    assert resume_path.read_text().strip().splitlines() == ["1", "X0001_Y0002"]
