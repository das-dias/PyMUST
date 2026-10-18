"""JAX kernels for PyMUST's hot loops, used when backend="jax" is selected."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

SINC_EPS = 1e-16
CROSS_CORRELATION_EPS = 1e-6


def compute_sinc(x):
    """Unnormalized sinc, sin(x)/x, matching pfield's ``mysinc``."""
    return jnp.sin(jnp.abs(x) + SINC_EPS) / (jnp.abs(x) + SINC_EPS)


@partial(jax.jit, static_argnames=("mode", "num_points"))
def sum_over_spectrum(
    exp_segments,
    exp_step,
    wavenumbers,
    tx_weights,
    is_out,
    directivity_args=(),
    mgbm_args=None,
    mgbm_update=None,
    rc=None,
    rx_weights=None,
    exp_rc=None,
    exp_step_rc=None,
    mode="pfield",
    num_points=0,
):
    """Frequency summation of PFIELD/PFIELD3, one ``lax.scan`` step per frequency.

    Returns the spectrum of shape [n_frequencies, n_outputs]: pressure at each
    point (mode "pfield"), received RF spectra per element ("simus") or
    pressure plus scatterer echoes on the movie grid ("mkmovie").
    """
    is_out = is_out.ravel()

    def compute_mgbm(wavenumber):
        alpha, gamma, beta2, coeff_a, coeff_b = mgbm_args
        inverse = 1 / (wavenumber * alpha[..., None] + coeff_b)
        gaussians = jnp.sqrt(jnp.pi * inverse) * jnp.exp(
            wavenumber**2 * beta2[..., None] / 4 * inverse + wavenumber * gamma[..., None]
        )
        return jnp.sum(coeff_a * gaussians, axis=-1)

    def compute_step(carry, per_frequency):
        exp_now, mgbm, exp_rc_now = carry
        wavenumber, tx_weight, rx_weight, update_mgbm = per_frequency

        directivity = 1.0
        for directivity_arg in directivity_args:
            directivity = directivity * compute_sinc(wavenumber * directivity_arg)
        rp_mono = jnp.mean(directivity * exp_now, axis=-1)

        if mgbm_args is not None:
            mgbm = jax.lax.cond(update_mgbm, compute_mgbm, lambda _: mgbm, wavenumber)
            rp_mono = rp_mono * mgbm

        rp_k = jnp.where(is_out, 0, rp_mono @ tx_weight)

        if mode == "simus":
            spectrum_k = rx_weight * ((rp_k * rc) @ rp_mono)
        elif mode == "mkmovie":
            spectrum_k = rp_k[:num_points]
            if exp_rc_now is not None:
                spectrum_k = spectrum_k + exp_rc_now @ (rp_k[num_points:] * rc)
                exp_rc_now = exp_rc_now * exp_step_rc
        else:
            spectrum_k = rp_k
        return (exp_now * exp_step, mgbm, exp_rc_now), spectrum_k

    mgbm_init = None if mgbm_args is None else jnp.zeros(exp_segments.shape[:2], exp_segments.dtype)
    _, spectrum = jax.lax.scan(
        compute_step,
        (exp_segments, mgbm_init, exp_rc),
        (wavenumbers, tx_weights, rx_weights, mgbm_update),
    )
    return spectrum


@partial(jax.jit, static_argnames=("window_rows", "window_cols", "frame_step"))
def track_windows(image, rows, cols, row_shifts, col_shifts, hann, window_rows, window_cols, frame_step):
    """Speckle-tracking cross-correlation of every interrogation window of one SPTRACK pass.

    Returns the row and column displacements (with parabolic subpixel fit) and
    the correlation coefficient of each window. NaN windows get NaN
    displacements and a zero coefficient.
    """
    window_shape = (window_rows, window_cols, image.shape[2])
    half_rows = window_rows // 2
    half_cols = window_cols // 2

    def compute_parabolic_offset(before, peak, after):
        return (before - after) / (2 * before - 4 * peak + 2 * after)

    def correlate_window(row, col, row_shift, col_shift):
        first = jax.lax.dynamic_slice(image, (row, col, 0), window_shape)[..., :-frame_step]
        second = jax.lax.dynamic_slice(image, (row + row_shift, col + col_shift, 0), window_shape)[..., frame_step:]
        has_nan = jnp.any(jnp.isnan(first + second))

        spectrum_first = jnp.fft.fft2((first - jnp.mean(first)) * hann[..., None], axes=(0, 1))
        spectrum_second = jnp.fft.fft2((second - jnp.mean(second)) * hann[..., None], axes=(0, 1))
        cross_spectrum = spectrum_second * jnp.conj(spectrum_first)

        normalized = jnp.mean(cross_spectrum / (jnp.abs(cross_spectrum) + CROSS_CORRELATION_EPS), axis=2)
        coefficient = jnp.max(jnp.fft.ifft2(normalized).real)

        correlation = jnp.fft.fftshift(jnp.fft.ifft2(jnp.sum(cross_spectrum, axis=2)).real)
        peak_row, peak_col = jnp.unravel_index(jnp.argmax(correlation), correlation.shape)

        row_offset = compute_parabolic_offset(
            correlation[jnp.maximum(peak_row - 1, 0), peak_col],
            correlation[peak_row, peak_col],
            correlation[jnp.minimum(peak_row + 1, window_rows - 1), peak_col],
        )
        col_offset = compute_parabolic_offset(
            correlation[peak_row, jnp.maximum(peak_col - 1, 0)],
            correlation[peak_row, peak_col],
            correlation[peak_row, jnp.minimum(peak_col + 1, window_cols - 1)],
        )
        row_displacement = row_shift + peak_row - half_rows
        row_displacement = row_displacement + jnp.where((peak_row > 0) & (peak_row < window_rows - 1), row_offset, 0)
        col_displacement = col_shift + peak_col - half_cols
        col_displacement = col_displacement + jnp.where((peak_col > 0) & (peak_col < window_cols - 1), col_offset, 0)

        return (
            jnp.where(has_nan, jnp.nan, row_displacement),
            jnp.where(has_nan, jnp.nan, col_displacement),
            jnp.where(has_nan, 0, coefficient),
        )

    return jax.vmap(correlate_window)(rows, cols, row_shifts, col_shifts)


def convolve_symmetric_same(image, kernel):
    """JAX equivalent of ``scipy.signal.convolve2d(image, kernel, 'same', boundary='symmetric')``."""
    pad_width = [(size // 2, size - 1 - size // 2) for size in kernel.shape]
    padded = jnp.pad(jnp.asarray(image), pad_width, mode="symmetric")
    return jax.scipy.signal.convolve2d(padded, jnp.asarray(kernel), mode="valid")


def apply_das_matrix(das_matrix, signal):
    """Beamform with a DAS matrix; pass a ``BCOO`` to skip the scipy conversion across frames."""
    if not isinstance(das_matrix, sparse.BCOO):
        das_matrix = sparse.BCOO.from_scipy_sparse(das_matrix)
    return np.asarray(das_matrix @ jnp.asarray(signal))


def compute_frequency_weights(frequencies, delays_tx, apodization, pulse_spectrum, probe_spectrum, rx_delays=None):
    """Host-side (float64) TX and RX weights of each frequency sample for ``sum_over_spectrum``."""
    tx_phases = 2 * np.pi * frequencies[:, None, None] * delays_tx
    tx_weights = np.sum(np.exp(1j * tx_phases), axis=1) * apodization * (pulse_spectrum * probe_spectrum)[:, None]
    if rx_delays is None:
        return tx_weights, None
    rx_phases = 2 * np.pi * frequencies[:, None] * rx_delays.reshape((1, -1))
    return tx_weights, probe_spectrum[:, None] * np.exp(1j * rx_phases)
