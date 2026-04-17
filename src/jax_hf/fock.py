"""Fock matrix construction and energy evaluation (pure functions)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .utils import selfenergy_fft


def _herm(X: jax.Array) -> jax.Array:
    """Hermitize on the last two axes."""
    return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))


def build_fock(
    P: jax.Array,
    *,
    h: jax.Array,
    VR: jax.Array,
    refP: jax.Array,
    HH: jax.Array,
    w2d: jax.Array,
    include_exchange: bool,
    include_hartree: bool,
    exchange_hermitian_channel_packing: bool,
    exchange_block_specs: Any | None = None,
    exchange_check_offdiag: bool | None = None,
    exchange_offdiag_atol: float = 1e-12,
    exchange_offdiag_rtol: float = 0.0,
    project_fn=None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build Fock matrix and return (Sigma, H_hartree, F).

    F = project(hermitize(h + Sigma[P] + H[P]))
    """
    Sigma = (
        selfenergy_fft(
            VR,
            P - refP,
            block_specs=exchange_block_specs,
            check_offdiag=exchange_check_offdiag,
            offdiag_atol=exchange_offdiag_atol,
            offdiag_rtol=exchange_offdiag_rtol,
            _apply_ifftshift=False,
            hermitian_channel_packing=exchange_hermitian_channel_packing,
        )
        if include_exchange
        else jnp.zeros_like(h)
    )

    if include_hartree:
        dP = P - refP
        diag_real = jnp.real(jnp.diagonal(dP, axis1=-2, axis2=-1))
        n_vec = jnp.sum(w2d[..., None] * diag_real, axis=(0, 1))
        sigma_diag = HH @ n_vec
        H_mat = jnp.diag(sigma_diag.astype(h.real.dtype))
        H = H_mat[None, None, ...]
    else:
        H = jnp.zeros_like(h)

    F = _herm(h + Sigma + H)
    if project_fn is not None:
        F = _herm(jnp.asarray(project_fn(F), dtype=F.dtype))
    return Sigma, H, F


def hf_energy(
    P: jax.Array,
    *,
    h: jax.Array,
    Sigma: jax.Array,
    H: jax.Array,
    weights_b: jax.Array,
) -> jax.Array:
    """E = sum_k w_k Tr[(h + 0.5(Sigma+H)) P]."""
    return jnp.sum(
        jnp.real(jnp.einsum("...ij,...ji->...", weights_b * (h + 0.5 * (Sigma + H)), P))
    )


def occupation_entropy(p: jax.Array, w_norm: jax.Array) -> jax.Array:
    """S = -sum_k w_k sum_i [p*log(p) + (1-p)*log(1-p)]."""
    p_safe = jnp.clip(p, 1e-14, 1.0 - 1e-14)
    s = p_safe * jnp.log(p_safe) + (1.0 - p_safe) * jnp.log1p(-p_safe)
    return -jnp.sum(w_norm[..., None] * s)


def free_energy(
    E: jax.Array,
    p: jax.Array,
    w_norm: jax.Array,
    T: jax.Array,
) -> jax.Array:
    """Free energy Omega = E - T*S(p)."""
    T_val = jnp.maximum(jnp.asarray(T, dtype=p.dtype), jnp.asarray(1e-14, dtype=p.dtype))
    return E - T_val * occupation_entropy(p, w_norm)
