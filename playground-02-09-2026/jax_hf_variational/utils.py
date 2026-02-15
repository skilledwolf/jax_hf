"""Standalone low-level utilities for ``jax_hf_variational``."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn, ifftshift
from jax.scipy.special import expit

BlockSpec = tuple[str, tuple[Any, ...]]


def _as_sequence(obj: Any) -> Sequence[Any]:
    if isinstance(obj, (list, tuple)):
        return obj
    if hasattr(obj, "tolist"):
        out = obj.tolist()
        if isinstance(out, list):
            return out
    raise TypeError(f"Expected a sequence; got {type(obj)}.")


def _normalize_one_spec(spec: Any) -> BlockSpec:
    if spec is None:
        raise TypeError("block_specs entries must be non-None.")

    if (
        isinstance(spec, tuple)
        and len(spec) == 2
        and isinstance(spec[0], str)
        and spec[0].strip().lower() in ("sizes", "indices")
    ):
        kind = spec[0].strip().lower()
        data = spec[1]
        if kind == "sizes":
            return ("sizes", tuple(int(x) for x in _as_sequence(data)))
        return ("indices", tuple(tuple(int(i) for i in _as_sequence(b)) for b in _as_sequence(data)))

    if isinstance(spec, dict):
        if "block_sizes" in spec:
            return ("sizes", tuple(int(x) for x in _as_sequence(spec["block_sizes"])))
        if "block_indices" in spec:
            return (
                "indices",
                tuple(tuple(int(i) for i in _as_sequence(b)) for b in _as_sequence(spec["block_indices"])),
            )
        raise TypeError("Spec dict must contain 'block_sizes' or 'block_indices'.")

    seq = _as_sequence(spec)
    if len(seq) == 0:
        raise TypeError("Spec sequences must be non-empty.")
    if all(isinstance(x, (int, np.integer)) for x in seq):
        return ("sizes", tuple(int(x) for x in seq))
    return ("indices", tuple(tuple(int(i) for i in _as_sequence(b)) for b in seq))


def normalize_block_specs(block_specs: Any) -> tuple[BlockSpec, ...] | None:
    """Normalize block specs into a canonical hashable tuple format."""
    if block_specs is None:
        return None

    if isinstance(block_specs, dict):
        items: Sequence[Any] = (block_specs,)
    else:
        items = block_specs

    if not isinstance(items, (list, tuple)):
        raise TypeError(
            "block_specs must be a sequence of specs or a single spec dict; "
            f"got {type(block_specs)}."
        )

    normalized: list[BlockSpec] = []
    for spec in items:
        normalized.append(_normalize_one_spec(spec))
    return tuple(normalized)


def hermitize(X: jax.Array) -> jax.Array:
    """Hermitize the last two axes: ``0.5 * (X + X^\u2020)``."""
    return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))


def fermidirac(x: jax.Array, T: float) -> jax.Array:
    """Finite-T Fermi-Dirac occupation: ``1/(1 + exp(x/T))``."""
    return expit(-x / (T + 1e-12))


def selfenergy_fft(
    VR: jax.Array,
    P: jax.Array,
    *,
    block_specs: Any | None = None,
    check_offdiag: bool | None = None,
    offdiag_atol: float = 1e-12,
    offdiag_rtol: float = 0.0,
) -> jax.Array:
    """Exchange self-energy ``Σ(k) = -FFT⁻¹[FFT(P) * VR]``.

    Supports optional block-structured acceleration using canonical block specs.
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
    """Bracketed bisection solve for ``mu`` with robust thermal padding."""
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


__all__ = [
    "fermidirac",
    "find_chemical_potential",
    "hermitize",
    "normalize_block_specs",
    "selfenergy_fft",
]
