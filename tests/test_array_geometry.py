"""Regression tests for how element positions are read and used (see docs/transducer_array_support.md §5.1)."""

import numpy as np

import pymust
from pymust.dasmtx3 import vxdcr3
from pymust.utils import Param

SOUND_SPEED_M_PER_S = 1540
PITCH_M = 300e-6
NUM_SAMPLES = 600
VIRTUAL_SOURCE_M = np.array([1e-3, -0.5e-3, -8e-3])
# Virtual-source and element-based TX distances agree to ~5 µm (~0.04 samples at 12 MHz)
DAS_WEIGHT_TOLERANCE = 0.05


def get_matrix_param(elements_per_side=8):
    centers_m = (np.arange(elements_per_side) - (elements_per_side - 1) / 2) * PITCH_M
    element_x, element_y = np.meshgrid(centers_m, centers_m)
    return Param(
        fc=3e6,
        fs=12e6,
        bandwidth=70,
        width=270e-6,
        height=270e-6,
        elements=np.stack([element_x.ravel(), element_y.ravel()]),
    )


def get_diverging_delays(param):
    element_x, element_y = param.elements
    distance_m = np.sqrt(
        (element_x - VIRTUAL_SOURCE_M[0]) ** 2 + (element_y - VIRTUAL_SOURCE_M[1]) ** 2 + VIRTUAL_SOURCE_M[2] ** 2
    )
    return ((distance_m - distance_m.min()) / SOUND_SPEED_M_PER_S).reshape((1, -1)), distance_m.min()


def compute_das_matrix_3d(param, delays):
    x, y, z = np.meshgrid(np.linspace(-2e-3, 2e-3, 5), np.linspace(-2e-3, 2e-3, 5), np.linspace(5e-3, 2e-2, 6))
    num_elements = param.elements.shape[1]
    return pymust.dasmtx3(np.array([NUM_SAMPLES, num_elements]), x, y, z, delays, param.copy()).toarray()


def test_get_element_positions_defaults_to_linear_array():
    num_elements = 8
    element_x, element_z, element_angle, _ = Param(pitch=PITCH_M, Nelements=num_elements).getElementPositions()

    np.testing.assert_allclose(element_x.ravel(), (np.arange(num_elements) - (num_elements - 1) / 2) * PITCH_M)
    assert not element_z.any()
    assert not element_angle.any()


def test_txdelay3_on_plain_matrix_param():
    param = get_matrix_param()
    element_x, element_y = param.elements

    focus_m = (1e-3, -2e-3, 3e-2)
    focused = pymust.txdelay3(*focus_m, param.copy())
    distance_m = np.sqrt((element_x - focus_m[0]) ** 2 + (element_y - focus_m[1]) ** 2 + focus_m[2] ** 2)
    expected = -distance_m / SOUND_SPEED_M_PER_S
    np.testing.assert_allclose(focused.ravel(), expected - expected.min())

    tilt_x_rad, tilt_y_rad = 0.1, 0.2
    plane = pymust.txdelay3(param.copy(), tilt_x_rad, tilt_y_rad)
    propagation_x = np.sin(tilt_y_rad) * np.cos(tilt_x_rad)
    expected = (element_x * propagation_x - element_y * np.sin(tilt_x_rad)) / SOUND_SPEED_M_PER_S
    np.testing.assert_allclose(plane.ravel(), expected - expected.min())


def test_dasmtx_uses_param_elements():
    param = pymust.getparam("L11-5v")
    param.Nelements = 16
    param.fs = 4 * param.fc
    delays = pymust.txdelayPlane(param.copy(), 0.1)
    element_x, element_z, _, _ = param.getElementPositions()
    x, z = np.meshgrid(np.linspace(-3e-3, 3e-3, 8), np.linspace(5e-3, 1e-2, 8))

    def compute_das_matrix(elements=None):
        das_param = param.copy()
        if elements is not None:
            das_param.elements = elements
        return pymust.dasmtx(1j * np.array([NUM_SAMPLES, param.Nelements]), x, z, delays, das_param).toarray()

    default = compute_das_matrix()
    np.testing.assert_allclose(compute_das_matrix(np.vstack([element_x, element_z])), default)
    np.testing.assert_allclose(compute_das_matrix(element_x), default)
    shifted = compute_das_matrix(np.vstack([element_x + 0.5e-3, element_z + 1e-3]))
    assert not np.allclose(shifted, default)


def test_vxdcr3_recovers_virtual_source_sphere():
    param = get_matrix_param(elements_per_side=16)
    delays, source_distance_m = get_diverging_delays(param)
    element_x, element_y = param.elements

    virtual_x, virtual_y, virtual_z = vxdcr3(element_x, element_y, delays, SOUND_SPEED_M_PER_S)

    virtual_points = np.stack([virtual_x, virtual_y, virtual_z], axis=1)
    np.testing.assert_allclose(np.linalg.norm(virtual_points - VIRTUAL_SOURCE_M, axis=1), source_distance_m, atol=5e-6)


def test_dasmtx3_virtual_source_matches_element_based_tx_distance():
    param = get_matrix_param(elements_per_side=16)
    delays, _ = get_diverging_delays(param)
    element_based = compute_das_matrix_3d(param, delays)

    param.useVirtualSource = True
    virtual_source = compute_das_matrix_3d(param, delays)

    assert np.count_nonzero(element_based) > 0
    assert np.count_nonzero(virtual_source) == np.count_nonzero(element_based)
    np.testing.assert_allclose(virtual_source, element_based, rtol=0, atol=DAS_WEIGHT_TOLERANCE)


def test_dasmtx3_single_transmitting_element_uses_y():
    param = get_matrix_param()
    transmitting_element = 3
    assert param.elements[1, transmitting_element] != 0

    single_element = np.full((1, param.elements.shape[1]), np.nan)
    single_element[0, transmitting_element] = 0
    param.useVirtualSource = True
    virtual_source = compute_das_matrix_3d(param, single_element)

    # Non-virtual path with every other element delayed by 1 s picks the same element
    far_delays = np.ones((1, param.elements.shape[1]))
    far_delays[0, transmitting_element] = 0
    param.useVirtualSource = False
    np.testing.assert_allclose(virtual_source, compute_das_matrix_3d(param, far_delays))
