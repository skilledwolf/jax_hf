"""Generic symmetry projectors for JAX Hartree-Fock solvers.

This module provides ``make_project_fn``, a JAX-traceable callable builder
for averaging density / Fock matrices over unitary, spatial, and
time-reversal symmetries.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax


ProjectFn = Callable[[jax.Array], jax.Array]

__all__ = ["ProjectFn", "make_project_fn"]


def _flip_k(
    A: jax.Array,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> jax.Array:
    """Negate selected k-grid axes under either ``flip`` or ``mod`` convention."""
    if k_convention == "flip":
        return jnp.flip(A, axis=flip_axes)
    if k_convention == "mod":
        nk1, nk2 = A.shape[0], A.shape[1]
        i = (
            (-jnp.arange(nk1, dtype=jnp.int32)) % nk1
            if 0 in flip_axes
            else jnp.arange(nk1, dtype=jnp.int32)
        )
        j = (
            (-jnp.arange(nk2, dtype=jnp.int32)) % nk2
            if 1 in flip_axes
            else jnp.arange(nk2, dtype=jnp.int32)
        )
        return A[i[:, None], j[None, :], ...]
    raise ValueError(f"k_convention must be 'mod' or 'flip', got {k_convention!r}")


def _sum_unitary_conj(A: jax.Array, G: jax.Array) -> jax.Array:
    """Return ``sum_i g_i @ A @ g_i†`` without normalization."""
    ng = G.shape[0]

    def body(i, acc):
        g = G[i]
        gH = jnp.conj(jnp.swapaxes(g, -1, -2))
        return acc + (g @ A) @ gH

    return lax.fori_loop(0, ng, body, jnp.zeros_like(A))


def _avg_unitary_conj(A: jax.Array, G: jax.Array) -> jax.Array:
    """Average ``A`` over unitary conjugation by the elements of ``G``."""
    ng = G.shape[0]
    return _sum_unitary_conj(A, G) / jnp.asarray(float(ng), dtype=A.dtype)


def _avg_combined_group(
    A: jax.Array,
    G_same: jax.Array,
    G_flip: jax.Array,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> jax.Array:
    """Average over a group split into same-k and flipped-k elements."""
    acc = _sum_unitary_conj(A, G_same)
    A_neg = _flip_k(A, k_convention, flip_axes)
    acc = acc + _sum_unitary_conj(A_neg, G_flip)
    N = G_same.shape[0] + G_flip.shape[0]
    return acc / jnp.asarray(float(N), dtype=A.dtype)


def _avg_time_reversal(
    A: jax.Array,
    U: jax.Array,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> jax.Array:
    """Average ``A`` with its time-reversed partner."""
    UH = jnp.conj(jnp.swapaxes(U, -1, -2))
    A_neg = _flip_k(A, k_convention, flip_axes)
    A_tr = U @ jnp.conj(A_neg) @ UH
    return 0.5 * (A + A_tr)


def make_project_fn(
    *,
    unitary_group: jax.Array | None = None,
    spatial_group: jax.Array | None = None,
    spatial_k_convention: str = "mod",
    spatial_k_flip_axes: tuple[int, ...] = (0, 1),
    time_reversal_U: jax.Array | None = None,
    time_reversal_k_convention: str = "mod",
    time_reversal_k_flip_axes: tuple[int, ...] = (0, 1),
) -> ProjectFn:
    """Build a symmetry-averaging projection function."""
    has_group = unitary_group is not None or spatial_group is not None
    if not has_group and time_reversal_U is None:
        return lambda A: A

    G = None if unitary_group is None else jnp.asarray(unitary_group)
    S = None if spatial_group is None else jnp.asarray(spatial_group)
    U = None if time_reversal_U is None else jnp.asarray(time_reversal_U)
    s_k_conv = str(spatial_k_convention)
    s_k_axes = tuple(spatial_k_flip_axes)
    t_k_conv = str(time_reversal_k_convention)
    t_k_axes = tuple(time_reversal_k_flip_axes)

    def project(A: jax.Array) -> jax.Array:
        out = A
        if G is not None and S is not None:
            out = _avg_combined_group(out, G, S, s_k_conv, s_k_axes)
        elif G is not None:
            out = _avg_unitary_conj(out, G)
        elif S is not None:
            I_mat = jnp.eye(S.shape[-1], dtype=S.dtype)[None]
            out = _avg_combined_group(out, I_mat, S, s_k_conv, s_k_axes)
        if U is not None:
            out = _avg_time_reversal(out, U, t_k_conv, t_k_axes)
        return out

    return project
