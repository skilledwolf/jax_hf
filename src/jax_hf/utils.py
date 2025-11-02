"""Low-level physics utilities for jax_hf (self-contained, no external deps)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn, ifftshift
from jax.scipy.special import expit


def fermidirac(x: jax.Array, T: float) -> jax.Array:
    """Finite-T Fermi–Dirac occupation: 1/(1 + exp(x/T))."""
    return expit(-x / (T + 1e-12))


def electron_density(P: jax.Array) -> jax.Array:
    """Real trace of the one-particle density matrix (per k-point)."""
    return jnp.real(jnp.trace(P, axis1=-2, axis2=-1))


def density_spectrum(bands: jax.Array, mu: float, T: float) -> jax.Array:
    """Sum of Fermi occupations for each k over bands."""
    return fermidirac(bands - mu, T).sum(axis=-1)


def selfenergy_fft(VR: jax.Array, P: jax.Array) -> jax.Array:
    """Exchange self-energy Σ(k) = -FFT⁻¹[FFT(P) · VR], aligned to meshgrid (ifftshift)."""
    P_fft = fftn(P, axes=(0, 1))
    sigma = -ifftn(P_fft * VR, axes=(0, 1))
    return ifftshift(sigma, axes=(0, 1))


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
