"""Low-level physics utilities for jax_hf (self-contained, no external deps)."""

from __future__ import annotations

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn, ifftshift
from jax.scipy.special import expit

from .linalg import eigh, normalize_block_specs


def hermitize(X: jax.Array) -> jax.Array:
    """Hermitize the last two axes: 0.5*(X + X†)."""
    return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))


def fermidirac(x: jax.Array, T: float) -> jax.Array:
    """Finite-T Fermi–Dirac occupation: 1/(1 + exp(x/T))."""
    return expit(-x / (T + 1e-12))


def electron_density(P: jax.Array) -> jax.Array:
    """Real trace of the one-particle density matrix (per k-point)."""
    return jnp.real(jnp.trace(P, axis1=-2, axis2=-1))


def density_spectrum(bands: jax.Array, mu: float, T: float) -> jax.Array:
    """Sum of Fermi occupations for each k over bands."""
    return fermidirac(bands - mu, T).sum(axis=-1)


def selfenergy_fft(
    VR: jax.Array,
    P: jax.Array,
    *,
    block_specs: Any | None = None,
    check_offdiag: bool | None = None,
    offdiag_atol: float = 1e-12,
    offdiag_rtol: float = 0.0,
) -> jax.Array:
    """Exchange self-energy Σ(k) with optional block-diagonal acceleration.

    Computes Σ(k) = -FFT⁻¹[FFT(P) · VR], aligned to meshgrid (ifftshift).

    When `block_specs` is provided, contimod-style specs are supported and the
    function will:
      1) check the off-block magnitude of `P` against `(atol + rtol*max|P|)`,
      2) run a reduced set of FFTs for the first compatible block partition,
      3) fall back to the full computation when no spec is compatible.
    """
    VR = jnp.asarray(VR)
    P = jnp.asarray(P)

    if block_specs is None:
        return _selfenergy_fft_full(VR, P)

    specs = normalize_block_specs(block_specs)
    if not specs:
        return _selfenergy_fft_full(VR, P)

    check = bool(check_offdiag) if check_offdiag is not None else True
    return _selfenergy_fft_block_specs(
        VR,
        P,
        specs,
        check_offdiag=check,
        offdiag_atol=float(offdiag_atol),
        offdiag_rtol=float(offdiag_rtol),
    )


def _selfenergy_fft_full(VR: jax.Array, P: jax.Array) -> jax.Array:
    P_fft = fftn(P, axes=(0, 1))
    sigma = -ifftn(P_fft * VR, axes=(0, 1))
    return ifftshift(sigma, axes=(0, 1))


def _mask_from_block_sizes(block_sizes: tuple[int, ...], n: int) -> jax.Array:
    mask = np.ones((int(n), int(n)), dtype=bool)
    start = 0
    for size in block_sizes:
        stop = start + int(size)
        mask[start:stop, start:stop] = False
        start = stop
    return jnp.asarray(mask)


def _mask_from_block_indices(block_indices: tuple[tuple[int, ...], ...], n: int) -> jax.Array:
    mask = np.ones((int(n), int(n)), dtype=bool)
    for idx in block_indices:
        idx_arr = np.asarray(idx, dtype=int)
        mask[np.ix_(idx_arr, idx_arr)] = False
    return jnp.asarray(mask)


def _slice_interaction(VR: jax.Array, idx: slice | jax.Array) -> jax.Array:
    if VR.shape[-2] == 1 and VR.shape[-1] == 1:
        return VR
    return VR[..., idx, idx]


def _take_interaction(VR: jax.Array, idx: jax.Array) -> jax.Array:
    if VR.shape[-2] == 1 and VR.shape[-1] == 1:
        return VR
    return jnp.take(jnp.take(VR, idx, axis=-2), idx, axis=-1)


def _selfenergy_fft_block_sizes(
    VR: jax.Array,
    P: jax.Array,
    block_sizes: tuple[int, ...],
) -> jax.Array:
    n = int(P.shape[-1])
    if sum(int(s) for s in block_sizes) != n:
        raise ValueError(f"block_sizes must sum to {n}.")

    out = jnp.zeros_like(P)
    start = 0
    for size in block_sizes:
        stop = start + int(size)
        s = slice(start, stop)
        sigma_block = _selfenergy_fft_full(_slice_interaction(VR, s), P[..., s, s])
        out = out.at[..., s, s].set(sigma_block)
        start = stop
    return out


def _selfenergy_fft_block_indices(
    VR: jax.Array,
    P: jax.Array,
    block_indices: tuple[tuple[int, ...], ...],
) -> jax.Array:
    out = jnp.zeros_like(P)
    for idx in block_indices:
        idx_j = jnp.asarray(idx, dtype=jnp.int32)
        sub = jnp.take(jnp.take(P, idx_j, axis=-2), idx_j, axis=-1)
        sigma_block = _selfenergy_fft_full(_take_interaction(VR, idx_j), sub)
        out = out.at[..., idx_j[:, None], idx_j[None, :]].set(sigma_block)
    return out


def _selfenergy_fft_block_specs(
    VR: jax.Array,
    P: jax.Array,
    block_specs: tuple[tuple[str, tuple[Any, ...]], ...],
    *,
    check_offdiag: bool,
    offdiag_atol: float,
    offdiag_rtol: float,
) -> jax.Array:
    n = int(P.shape[-1])
    abs_P = jnp.abs(P)
    scale = jnp.max(abs_P)
    tol = jnp.asarray(offdiag_atol, dtype=abs_P.dtype) + jnp.asarray(offdiag_rtol, dtype=abs_P.dtype) * scale

    def do_full(_):
        return _selfenergy_fft_full(VR, P)

    fn = do_full

    for kind, data in reversed(block_specs):
        kind = str(kind).strip().lower()
        if kind == "sizes":
            sizes = tuple(int(x) for x in data)
            mask = _mask_from_block_sizes(sizes, n)
            ok = (jnp.max(abs_P * mask) <= tol) if check_offdiag else jnp.array(True)

            def do_block(_, *, _sizes=sizes):
                return _selfenergy_fft_block_sizes(VR, P, _sizes)

        elif kind == "indices":
            blocks = tuple(tuple(int(i) for i in b) for b in data)
            mask = _mask_from_block_indices(blocks, n)
            ok = (jnp.max(abs_P * mask) <= tol) if check_offdiag else jnp.array(True)

            def do_block(_, *, _blocks=blocks):
                return _selfenergy_fft_block_indices(VR, P, _blocks)

        else:
            raise ValueError("block_specs kind must be 'sizes' or 'indices'.")

        prev_fn = fn
        fn = (lambda _prev, _ok, _do: (lambda _: jax.lax.cond(_ok, _do, _prev, operand=None)))(prev_fn, ok, do_block)

    return fn(None)


def find_chemical_potential(
    bands: jax.Array,
    weights: jax.Array,
    n_electrons: float,
    T: float,
    *,
    maxiter: int = 80,
) -> jax.Array:
    """Robust bracketed bisection that tolerates degenerate bands.

    Solves μ such that ∑_k w_k Σ_j f(ε_kj − μ) = n_electrons.
    Uses a conservative bracket around [min(bands), max(bands)] expanded by 10·T.
    """
    bands_min = jnp.min(bands)
    bands_max = jnp.max(bands)
    Tj = jnp.asarray(T, dtype=bands.real.dtype)
    span = bands_max - bands_min + 10.0 * jnp.maximum(Tj, jnp.array(1e-6, dtype=bands.real.dtype))
    lo = bands_min - span
    hi = bands_max + span

    def body(state, _):
        lo_curr, hi_curr = state
        mid = 0.5 * (lo_curr + hi_curr)
        occ = fermidirac(bands - mid, T)
        count = jnp.sum(jnp.asarray(weights)[..., None] * occ)
        too_high = count > n_electrons
        lo_next = jnp.where(too_high, lo_curr, mid)
        hi_next = jnp.where(too_high, mid, hi_curr)
        return (lo_next, hi_next), None

    (lo_fin, hi_fin), _ = jax.lax.scan(body, (lo, hi), xs=None, length=int(maxiter))
    return 0.5 * (lo_fin + hi_fin)


def density_matrix_from_fock(
    F: jax.Array,
    weights: jax.Array,
    n_electrons: float,
    T: float,
    *,
    eigh_block_specs: object | None = None,
    eigh_check_offdiag: bool | None = None,
    eigh_offdiag_atol: float = 1e-12,
    eigh_offdiag_rtol: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Build a physical density matrix P(k) from a (Hermitian) Fock matrix F(k)."""
    F = hermitize(jnp.asarray(F))
    eps, U = eigh(
        F,
        block_specs=eigh_block_specs,
        check_offdiag=eigh_check_offdiag,
        offdiag_atol=eigh_offdiag_atol,
        offdiag_rtol=eigh_offdiag_rtol,
    )
    mu = find_chemical_potential(eps, jnp.asarray(weights), n_electrons=float(n_electrons), T=float(T))
    occ = fermidirac(eps - mu, float(T))
    P = jnp.einsum("...in,...n,...jn->...ij", U, occ, jnp.conj(U))
    return hermitize(P), mu


def _resample_axis_periodic_linear(x: jax.Array, n_new: int, *, axis: int) -> jax.Array:
    x = jnp.asarray(x)
    if x.ndim == 0:
        raise ValueError("resample_kgrid expects an array with at least 1 dimension.")
    n_old = int(x.shape[axis])
    n_new = int(n_new)
    if n_new <= 0:
        raise ValueError("n_new must be a positive integer.")
    if n_old == n_new:
        return x

    real_dtype = x.real.dtype
    j = jnp.arange(n_new, dtype=real_dtype)
    # Coordinate convention: arrays are stored on a centered periodic grid (k=0 at index n//2).
    u = (j - jnp.asarray(n_new // 2, dtype=real_dtype)) * (jnp.asarray(n_old, dtype=real_dtype) / jnp.asarray(n_new, dtype=real_dtype))
    u = u + jnp.asarray(n_old // 2, dtype=real_dtype)

    i0 = jnp.floor(u).astype(jnp.int32)

    i0 = jnp.mod(i0, n_old)
    i1 = jnp.mod(i0 + 1, n_old)

    x0 = jnp.take(x, i0, axis=axis)
    x1 = jnp.take(x, i1, axis=axis)

    # reshape frac to broadcast along all non-resampled axes
    shape = [1] * x.ndim
    shape[axis] = n_new
    frac = (u - jnp.floor(u)).reshape(shape).astype(real_dtype)
    if jnp.iscomplexobj(x0):
        frac = frac.astype(x0.dtype)

    return (1.0 - frac) * x0 + frac * x1


def resample_kgrid(values: jax.Array, nk: int, *, method: str = "linear") -> jax.Array:
    """Resample a centered, periodic (nk,nk,...) k-grid array to a new nk.

    Notes
    -----
    - Assumes the first two axes are the 2D uniform k-grid, stored in *centered*
      meshgrid order (k=0 at index nk//2), as in the package regression tests.
    - This is intended for coarse-to-fine continuation and seeding, not for
      high-accuracy quadrature of sharply peaked functions.
    """
    nk = int(nk)
    x = jnp.asarray(values)
    if x.ndim < 2:
        raise ValueError("resample_kgrid expects an array with at least 2 dimensions (nk,nk,...).")
    if x.shape[0] == nk and x.shape[1] == nk:
        return x

    method = str(method).lower()
    if method != "linear":
        raise ValueError(f"Unsupported resample method {method!r}. Only 'linear' is supported.")

    y = _resample_axis_periodic_linear(x, nk, axis=0)
    y = _resample_axis_periodic_linear(y, nk, axis=1)
    return y
