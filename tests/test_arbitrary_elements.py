"""Arbitrary element positions: pre-made arrays translate into PARAM.elements, and every
function uses the full position vector, so shifting the array together with the field
points, scatterers, foci and pixels leaves every result unchanged."""

import numpy as np
import pytest

import pymust
from pymust.dasmtx3 import vxdcr3
from pymust.utils import Param, plot_transducer_array

SOUND_SPEED_M_PER_S = 1540
PITCH_M = 300e-6
NUM_SAMPLES = 800
SHIFT_2D_M = np.array([1.5e-3, 4e-3])  # (x, z)
SHIFT_3D_M = np.array([1e-3, -2e-3, 3e-3])  # (x, y, z)
CAP_RADIUS_M = 20e-3
VIRTUAL_SOURCE_M = np.array([1e-3, -0.5e-3, -8e-3])
# pfield/simus accumulate in single precision, so shifted coordinates round differently
SINGLE_PRECISION_RTOL = 1e-4


def assert_close_to_peak(actual, desired, rtol=SINGLE_PRECISION_RTOL):
    actual, desired = np.asarray(actual), np.asarray(desired)
    assert actual.shape == desired.shape
    np.testing.assert_allclose(actual, desired, rtol=0, atol=rtol * np.max(np.abs(desired)))


def get_linear_param(num_elements=16):
    param = pymust.getparam("L11-5v")
    param.Nelements = num_elements
    param.fs = 4 * param.fc
    return param


def get_curved_2d_param(shift_m=(0.0, 0.0)):
    """16 elements on a shallow parabola in the x-z plane, given only by PARAM.elements."""
    element_x = (np.arange(16) - 7.5) * PITCH_M
    element_z = 1e-3 * (element_x / element_x.max()) ** 2
    return Param(
        fc=7.6e6,
        fs=4 * 7.6e6,
        bandwidth=77,
        width=270e-6,
        height=5e-3,
        elements=np.stack([element_x + shift_m[0], element_z + shift_m[1]]),
    )


def get_cap_3d_param(shift_m=(0.0, 0.0, 0.0), elements_per_side=8):
    """Matrix array on a spherical cap (a non-planar height field), given by 3-row PARAM.elements."""
    centers_m = (np.arange(elements_per_side) - (elements_per_side - 1) / 2) * PITCH_M
    element_x, element_y = (grid.ravel() for grid in np.meshgrid(centers_m, centers_m))
    element_z = CAP_RADIUS_M - np.sqrt(CAP_RADIUS_M**2 - element_x**2 - element_y**2)
    return Param(
        fc=3e6,
        fs=12e6,
        bandwidth=70,
        width=270e-6,
        height=270e-6,
        elements=np.stack([element_x + shift_m[0], element_y + shift_m[1], element_z + shift_m[2]]),
    )


def compute_2d_outputs(param, shift_m):
    """Every 2-D function evaluated with field points/foci/pixels shifted by shift_m = (dx, dz)."""
    shift_x, shift_z = shift_m
    x, z = np.meshgrid(np.linspace(-3e-3, 3e-3, 6) + shift_x, np.linspace(3e-3, 1.2e-2, 7) + shift_z)
    scatterer_x = np.array([0.0, 1e-3]) + shift_x
    scatterer_z = np.array([5e-3, 8e-3]) + shift_z

    outputs = {
        "txdelay_plane": pymust.txdelay(param.copy(), 0.1),
        "txdelay_focused": pymust.txdelay(1e-3 + shift_x, 2e-2 + shift_z, param.copy()),
        "txdelay_circular": pymust.txdelay(param.copy(), 0.1, np.pi / 3),
    }
    delays = outputs["txdelay_focused"]
    outputs["pfield"] = pymust.pfield(x, None, z, delays, param.copy())[0]
    outputs["simus"] = pymust.simus(
        scatterer_x, np.zeros(2), scatterer_z, np.ones(2), delays, param.copy(), pymust.utils.Options(dBThresh=-60)
    )[0]
    num_elements = delays.shape[1]
    outputs["dasmtx"] = pymust.dasmtx(1j * np.array([NUM_SAMPLES, num_elements]), x, z, delays, param.copy()).toarray()
    virtual_source_param = param.copy()
    virtual_source_param.useVirtualSource = True
    outputs["dasmtx_virtual_source"] = pymust.dasmtx(
        1j * np.array([NUM_SAMPLES, num_elements]), x, z, outputs["txdelay_circular"], virtual_source_param
    ).toarray()
    grid_x, grid_z = pymust.impolgrid(np.array([5, 4]), 2e-2, np.pi / 4, param.copy())
    outputs["impolgrid"] = np.stack([grid_x - shift_x, grid_z - shift_z])
    return outputs


def compute_3d_outputs(param, shift_m):
    """Every 3-D function evaluated with field points/foci/scatterers/pixels shifted by shift_m."""
    shift_x, shift_y, shift_z = shift_m
    x, y, z = np.meshgrid(
        np.linspace(-2e-3, 2e-3, 4) + shift_x,
        np.linspace(-2e-3, 2e-3, 3) + shift_y,
        np.linspace(4e-3, 1e-2, 3) + shift_z,
    )
    focus_line_x = np.array([0.0, 1e-3]) + shift_x
    focus_line_y = np.array([-1e-3, 1e-3]) + shift_y
    focus_line_z = np.array([2e-2, 2.5e-2]) + shift_z

    outputs = {
        "txdelay3_plane": pymust.txdelay3(param.copy(), 0.1, 0.2),
        "txdelay3_focused": pymust.txdelay3(1e-3 + shift_x, -2e-3 + shift_y, 3e-2 + shift_z, param.copy()),
        "txdelay3_focus_line": pymust.txdelay3(focus_line_x, focus_line_y, focus_line_z, param.copy()),
        "txdelay3_diverging": pymust.txdelay3(param.copy(), 0.1, -0.1, np.pi / 4),
    }
    delays = outputs["txdelay3_focused"]
    outputs["pfield3"] = pymust.pfield3(x, y, z, delays, param.copy())[0]
    outputs["simus3"] = pymust.simus3(
        np.array([0.0, 1e-3]) + shift_x,
        np.array([0.0, -1e-3]) + shift_y,
        np.array([5e-3, 8e-3]) + shift_z,
        np.ones(2),
        delays,
        param.copy(),
        pymust.utils.Options(dBThresh=-60),
    )[0]
    num_elements = delays.shape[1]
    outputs["dasmtx3"] = pymust.dasmtx3(np.array([NUM_SAMPLES, num_elements]), x, y, z, delays, param.copy()).toarray()
    virtual_source_param = param.copy()
    virtual_source_param.useVirtualSource = True
    outputs["dasmtx3_virtual_source"] = pymust.dasmtx3(
        np.array([NUM_SAMPLES, num_elements]), x, y, z, outputs["txdelay3_diverging"], virtual_source_param
    ).toarray()
    return outputs


def test_premade_linear_array_equals_its_explicit_elements():
    premade = get_linear_param()
    element_x, element_z, _, _ = premade.getElementPositions()
    explicit = premade.copy()
    explicit.elements = np.vstack([element_x, element_z])

    premade_outputs = compute_2d_outputs(premade, (0.0, 0.0))
    explicit_outputs = compute_2d_outputs(explicit, (0.0, 0.0))
    for name, premade_output in premade_outputs.items():
        np.testing.assert_allclose(explicit_outputs[name], premade_output, err_msg=name)

    x, y, z = np.meshgrid(np.linspace(-2e-3, 2e-3, 3), np.linspace(-1e-3, 1e-3, 3), np.linspace(5e-3, 1e-2, 3))
    delays = np.zeros((1, premade.Nelements))
    premade_field = pymust.pfield3(x, y, z, delays, premade.copy())[0]
    explicit_3d = premade.copy()
    explicit_3d.elements = np.vstack([element_x, np.zeros_like(element_x), element_z])
    np.testing.assert_allclose(pymust.pfield3(x, y, z, delays, explicit_3d)[0], premade_field)


def test_2d_functions_are_translation_invariant():
    reference = compute_2d_outputs(get_curved_2d_param(), (0.0, 0.0))
    shifted = compute_2d_outputs(get_curved_2d_param(SHIFT_2D_M), SHIFT_2D_M)
    for name, reference_output in reference.items():
        assert np.any(reference_output), name
        assert_close_to_peak(shifted[name], reference_output)


def test_3d_functions_are_translation_invariant():
    reference = compute_3d_outputs(get_cap_3d_param(), (0.0, 0.0, 0.0))
    shifted = compute_3d_outputs(get_cap_3d_param(SHIFT_3D_M), SHIFT_3D_M)
    for name, reference_output in reference.items():
        assert np.any(reference_output), name
        assert_close_to_peak(shifted[name], reference_output)


def test_txdelay3_plane_wave_follows_propagation_vector():
    param = get_cap_3d_param()
    element_x, element_y, element_z = param.elements
    tilt_x_rad, tilt_y_rad = 0.15, -0.25

    delays = pymust.txdelay3(param.copy(), tilt_x_rad, tilt_y_rad)

    propagation = np.array(
        [np.sin(tilt_y_rad) * np.cos(tilt_x_rad), -np.sin(tilt_x_rad), np.cos(tilt_y_rad) * np.cos(tilt_x_rad)]
    )
    expected = propagation @ np.stack([element_x, element_y, element_z]) / SOUND_SPEED_M_PER_S
    np.testing.assert_allclose(delays.ravel(), expected - expected.min())


def test_vxdcr3_finds_virtual_source_sphere_for_curved_array():
    param = get_cap_3d_param(elements_per_side=16)
    element_x, element_y, element_z = param.elements
    distance_m = np.linalg.norm(param.elements.T - VIRTUAL_SOURCE_M, axis=1)
    delays = (distance_m - distance_m.min()) / SOUND_SPEED_M_PER_S

    virtual_points = np.stack(vxdcr3(element_x, element_y, delays, SOUND_SPEED_M_PER_S, element_z), axis=1)

    np.testing.assert_allclose(np.linalg.norm(virtual_points - VIRTUAL_SOURCE_M, axis=1), distance_m.min(), atol=5e-6)


@pytest.mark.parametrize("probe_name", ["L11-5v", "C5-2v"])
def test_plot_transducer_array_draws_one_box_per_element(probe_name):
    pytest.importorskip("plotly")
    param = pymust.getparam(probe_name)
    thickness_m = 400e-6

    mesh = plot_transducer_array(param, thickness=thickness_m).data[0]

    vertices = np.stack([mesh.x, mesh.y, mesh.z], axis=1).reshape((param.Nelements, 8, 3)) / 1e3
    assert len(mesh.i) == 12 * param.Nelements
    first_box = vertices[0]
    edge_lengths = [
        np.linalg.norm(first_box[1] - first_box[0]),
        np.linalg.norm(first_box[2] - first_box[0]),
        np.linalg.norm(first_box[4] - first_box[0]),
    ]
    np.testing.assert_allclose(edge_lengths, [param.width, param.height, thickness_m])
    front_face_tilted = not np.isclose(first_box[4, 2], first_box[5, 2])
    assert front_face_tilted == np.isfinite(param.radius)
