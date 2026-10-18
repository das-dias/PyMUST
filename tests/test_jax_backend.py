"""Parity tests: every backend="jax" code path must match the numpy reference."""

import numpy as np
import pytest
import scipy.ndimage

import pymust
from pymust.utils import Options, Param
from pymust.wfilt import wfilt

pytest.importorskip("jax")

RELATIVE_TOLERANCE = 1e-4
# JAX computes travel times in float32; the I/Q phase exp(1j*wc*tau) spans ~1e3 rad,
# so DAS weights carry ~1e-4 absolute phase error
DAS_RELATIVE_TOLERANCE = 1e-3
SOUND_SPEED_M_PER_S = 1540
TRACKING_SHIFT = (3, -2)


def assert_close_to_peak(jax_value, numpy_value, rtol=RELATIVE_TOLERANCE):
    jax_value, numpy_value = np.asarray(jax_value), np.asarray(numpy_value)
    assert jax_value.shape == numpy_value.shape
    peak = np.nanmax(np.abs(numpy_value))
    np.testing.assert_allclose(jax_value, numpy_value, rtol=0, atol=rtol * peak)


def run_both_backends(compute):
    return compute("numpy"), compute("jax")


def get_plane_wave_delays_3d(param, tilt_x_rad, tilt_y_rad):
    element_x, element_y = param.elements
    delays = (element_x * np.sin(tilt_x_rad) + element_y * np.sin(tilt_y_rad)) / SOUND_SPEED_M_PER_S
    return (delays - delays.min()).reshape((1, -1))


def get_linear_param():
    param = pymust.getparam("L11-5v")
    param.Nelements = 16
    return param


def get_matrix_param(elements_per_side=4):
    pitch_m = 0.3e-3
    centers_m = (np.arange(elements_per_side) - (elements_per_side - 1) / 2) * pitch_m
    element_x, element_y = np.meshgrid(centers_m, centers_m)
    return Param(
        fc=3e6,
        bandwidth=70,
        width=0.27e-3,
        height=0.27e-3,
        elements=np.stack([element_x.ravel(), element_y.ravel()]),
    )


@pytest.mark.parametrize(
    ("options", "elevation_y_m"),
    [({}, 0.0), ({"FullFrequencyDirectivity": True}, 0.0), ({}, 1e-3)],
    ids=["default", "full_frequency_directivity", "elevation_focusing"],
)
def test_pfield(options, elevation_y_m):
    param = get_linear_param()
    param.height = 5e-3
    param.focus = 0.02
    x, z = pymust.impolgrid(10, 0.04, np.pi / 4, param)
    y = np.full_like(x, elevation_y_m)
    delays = pymust.txdelayFocused(param, 0.005, 0.03)

    numpy_result, jax_result = run_both_backends(
        lambda backend: pymust.pfield(x, y, z, delays, param.copy(), Options(**options, Backend=backend))
    )
    assert_close_to_peak(jax_result[0], numpy_result[0])
    assert_close_to_peak(jax_result[1], numpy_result[1])


def test_pfield_mkmovie_mode():
    param = get_linear_param()
    xi, zi = np.meshgrid(np.linspace(-5e-3, 5e-3, 12), np.linspace(1e-3, 1e-2, 15))
    delays = pymust.txdelayPlane(param, 0.1)

    def compute_spectrum(backend):
        options = Options(
            CallFun="mkmovie",
            FrequencyStep=1e5,
            ElementSplitting=1,
            RC=np.array([1.0, 0.5]),
            x=np.array([0.0, 2e-3]),
            z=np.array([4e-3, 6e-3]),
            Backend=backend,
        )
        return pymust.pfield(xi, [], zi, delays, param.copy(), options)[1]

    numpy_spectrum, jax_spectrum = run_both_backends(compute_spectrum)
    assert_close_to_peak(jax_spectrum, numpy_spectrum)


def test_simus():
    param = get_linear_param()
    x = np.array([0.0, 0.3e-2, -0.3e-2])
    z = np.array([0.2e-2, 0.4e-2, 0.6e-2])
    rc = np.array([1.0, 0.7, 1.3])
    delays = pymust.txdelayPlane(param, 0.05)

    numpy_rf, jax_rf = run_both_backends(
        lambda backend: pymust.simus(x, np.zeros(3), z, rc, delays, param.copy(), Options(Backend=backend))[0]
    )
    assert_close_to_peak(jax_rf, numpy_rf)


def test_pfield3_and_simus3():
    param = get_matrix_param()
    delays = get_plane_wave_delays_3d(param, 0.1, 0.05)
    x, y, z = np.meshgrid(np.linspace(-3e-3, 3e-3, 6), np.linspace(-3e-3, 3e-3, 5), np.linspace(2e-3, 1e-2, 4))

    numpy_field, jax_field = run_both_backends(
        lambda backend: pymust.pfield3(x, y, z, delays, param.copy(), Options(Backend=backend))
    )
    assert_close_to_peak(jax_field[0], numpy_field[0])
    assert_close_to_peak(jax_field[1], numpy_field[1])

    scatterer_x = np.array([0.0, 1e-3])
    scatterer_y = np.array([0.0, -1e-3])
    scatterer_z = np.array([5e-3, 8e-3])
    rc = np.array([1.0, 0.8])
    numpy_rf, jax_rf = run_both_backends(
        lambda backend: pymust.simus3(
            scatterer_x, scatterer_y, scatterer_z, rc, delays, param.copy(), Options(Backend=backend)
        )[0]
    )
    assert_close_to_peak(jax_rf, numpy_rf)


@pytest.mark.parametrize("rx_angle_rad", [0.0, 0.2], ids=["no_rx_angle", "vector_doppler_rx_angle"])
def test_dasmtx_and_apply(rx_angle_rad, rng):
    param = get_linear_param()
    param.fs = 4 * param.fc
    param.RXangle = rx_angle_rad
    delays = pymust.txdelayPlane(param, 0.0)
    x, z = np.meshgrid(np.linspace(-4e-3, 4e-3, 20), np.linspace(5e-3, 2e-2, 30))
    signal_shape = (400, param.Nelements)

    numpy_matrix, jax_matrix = run_both_backends(
        lambda backend: pymust.dasmtx(1j * np.array(signal_shape), x, z, delays, param.copy(), backend=backend)
    )
    assert_close_to_peak(jax_matrix.toarray(), numpy_matrix.toarray(), DAS_RELATIVE_TOLERANCE)

    iq = rng.standard_normal(signal_shape) + 1j * rng.standard_normal(signal_shape)
    numpy_image, jax_image = run_both_backends(
        lambda backend: pymust.utils.applyDasMTX(numpy_matrix, iq, x.shape, backend=backend)
    )
    assert_close_to_peak(jax_image, numpy_image)


def test_dasmtx3():
    param = get_matrix_param()
    param.fs = 4 * param.fc
    delays = get_plane_wave_delays_3d(param, 0.0, 0.0)
    x, y, z = np.meshgrid(np.linspace(-2e-3, 2e-3, 5), np.linspace(-2e-3, 2e-3, 4), np.linspace(5e-3, 1e-2, 6))

    numpy_matrix, jax_matrix = run_both_backends(
        lambda backend: pymust.dasmtx3(
            1j * np.array([300, param.elements.shape[1]]), x, y, z, delays, param.copy(), backend=backend
        )
    )
    assert_close_to_peak(jax_matrix.toarray(), numpy_matrix.toarray(), DAS_RELATIVE_TOLERANCE)


@pytest.mark.parametrize("window_size", [1, 3, np.array([4, 5])], ids=["no_smoothing", "odd", "even"])
def test_iq2doppler(window_size, rng):
    iq = rng.standard_normal((12, 10, 8)) + 1j * rng.standard_normal((12, 10, 8))
    param = Param(fc=5e6, PRF=4e3, c=1540)

    numpy_result, jax_result = run_both_backends(
        lambda backend: pymust.iq2doppler(iq, param, window_size, backend=backend)
    )
    assert_close_to_peak(jax_result[0], numpy_result[0])
    assert_close_to_peak(jax_result[1], numpy_result[1])


@pytest.mark.parametrize(("method", "order"), [("poly", 0), ("poly", 2), ("dct", 1), ("dct", 3), ("svd", 2)])
def test_wfilt(method, order, rng):
    clutter = 5 * np.ones((6, 7, 10))
    signal = clutter + rng.standard_normal((6, 7, 10)) + 1j * rng.standard_normal((6, 7, 10))

    numpy_filtered, jax_filtered = run_both_backends(lambda backend: wfilt(signal, method, order, backend=backend))
    assert_close_to_peak(jax_filtered, numpy_filtered)
    assert abs(np.mean(numpy_filtered)) < 1


@pytest.mark.parametrize("window_sizes", [[[32, 32]], [[48, 48], [32, 32]]], ids=["one_pass", "two_passes"])
def test_sptrack(window_sizes, rng):
    speckle = scipy.ndimage.gaussian_filter(rng.standard_normal((140, 120)), 1.5)
    frames = np.stack(
        [
            np.roll(speckle, (TRACKING_SHIFT[0] * frame_index, TRACKING_SHIFT[1] * frame_index), axis=(0, 1))
            for frame_index in range(3)
        ],
        axis=2,
    )

    numpy_result, jax_result = run_both_backends(
        lambda backend: pymust.sptrack(frames.copy(), Param(winsize=np.array(window_sizes)), backend=backend)
    )
    for jax_value, numpy_value in zip(jax_result, numpy_result):
        assert_close_to_peak(jax_value, numpy_value)
    assert np.nanmedian(jax_result[0]) == pytest.approx(TRACKING_SHIFT[0], abs=0.2)
    assert np.nanmedian(jax_result[1]) == pytest.approx(TRACKING_SHIFT[1], abs=0.2)
