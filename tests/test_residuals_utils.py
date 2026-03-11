import numpy as np
from utils.residuals_utils import get_output_array_full

# Test function get_ouput_array_full
def test_detect_negative_disturbence():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[-20.0]],
        [[-20.0]],
        [[-20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, -20.0, -20.0, -20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_detect_positive_disturbence():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[20.0]],
        [[20.0]],
        [[20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, 20.0, 20.0, 20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_switch_to_positive_disturbence():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[-20.0]],
        [[-20.0]],
        [[-20.0]],
        [[20.0]],
        [[20.0]],
        [[20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, -20.0, -20.0, -20.0, 20.0, 20.0, 20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, False, False, False, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_switch_to_negative_disturbence():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[20.0]],
        [[20.0]],
        [[20.0]],
        [[-20.0]],
        [[-20.0]],
        [[-20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, 20.0, 20.0, 20.0, -20.0, -20.0, -20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, False, False, False, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_return_to_normal_state_after_negative_disturbence():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[-20.0]],
        [[-20.0]],
        [[-20.0]],
        [[0.0]],
        [[0.0]],
        [[0.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, -20.0, -20.0, -20.0, np.nan, np.nan, np.nan]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, False, True, True, True]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_return_to_normal_state_after_positive_disturbence():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[20.0]],
        [[20.0]],
        [[20.0]],
        [[0.0]],
        [[0.0]],
        [[0.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, 20.0, 20.0, 20.0, np.nan, np.nan, np.nan]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, False, True, True, True]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_skip_normal_values_within_negative_disturbence():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[-20.0]],
        [[-20.0]],
        [[-20.0]],
        [[0.0]],
        [[-20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, -20.0, -20.0, -20.0, np.nan, -20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, False, True, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_skip_normal_values_within_positive_disturbence():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[20.0]],
        [[20.0]],
        [[20.0]],
        [[0.0]],
        [[20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, 20.0, 20.0, 20.0, np.nan, 20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, False, True, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_skip_positive_anomaly_within_negative_disturbence():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[-20.0]],
        [[-20.0]],
        [[-20.0]],
        [[20.0]],
        [[-20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, -20.0, -20.0, -20.0, np.nan, -20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, False, True, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_skip_negative_anomaly_within_positive_disturbence():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[20.0]],
        [[20.0]],
        [[20.0]],
        [[-20.0]],
        [[20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, 20.0, 20.0, 20.0, np.nan, 20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, False, True, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_skip_nodata_values_while_negative_disturbence_counts_up():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[-20.0]],
        [[-20.0]],
        [[np.nan]],
        [[-20.0]],
        [[-20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, -20.0, -20.0, np.nan, -20.0, -20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, True, False, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_skip_nodata_values_while_positive_disturbence_counts_up():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[0.0]],
        [[20.0]],
        [[20.0]],
        [[np.nan]],
        [[20.0]],
        [[20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, 20.0, 20.0, np.nan, 20.0, 20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, False, False, True, False, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_multiple_changes_1():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[20.0]],
        [[20.0]],
        [[20.0]],
        [[np.nan]],
        [[20.0]],
        [[-20.0]],
        [[0.0]],
        [[20.0]],
        [[0.0]],
        [[np.nan]],
        [[0.0]],
        [[0.0]],
        [[-20.0]],
        [[20.0]],
        [[-20.0]],
        [[np.nan]],
        [[-20.0]],
        [[-20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[20.0, 20.0, 20.0, np.nan, 20.0, np.nan, np.nan, 20.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -20.0, np.nan, -20.0, -20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[False, False, False, True, False, True, True, False, True, True, True, True, True, True, False, True, False, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)

def test_multiple_changes_2():
    # Synthetisches raster
    nrt_raster_data_syn = np.array([
        [[-20.0]],
        [[20.0]],
        [[-20.0]],
        [[np.nan]],
        [[-20.0]],
        [[-20.0]],
        [[20.0]],
        [[20.0]],
        [[0.0]],
        [[20.0]],
        [[20.0]],
        [[np.nan]],
        [[20.0]],
        [[-20.0]],
        [[-20.0]],
        [[np.nan]],
        [[-20.0]],
        [[-20.0]]
    ], dtype=np.float32).transpose(1, 2, 0)  # Shape: (1, 1, 11)

    # Erwarteter Output
    expected_output = np.array([
        [[np.nan, np.nan, -20.0, np.nan, -20.0, -20.0, np.nan, np.nan, np.nan, 20.0, 20.0, np.nan, 20.0, -20.0, -20.0, np.nan, -20.0, -20.0]]
    ], dtype=np.float32)

    expected_filtered = np.array([
        [[True, True, False, True, False, False, True, True, True, False, False, True, False, False, False, True, False, False]]
    ], dtype=np.float32)

    # Synthetischer Threshold
    threshold = 10.0

    # Funktion aufrufen
    output_array, filtered = get_output_array_full(nrt_raster_data_syn, threshold=threshold)

    # Prüfen, ob die Werte stimmen (inkl. NaN-safe Vergleich)
    np.testing.assert_allclose(output_array, expected_output, equal_nan=True)
    np.testing.assert_allclose(filtered, expected_filtered, equal_nan=True)
