"""Per-iteration solver wall-time benchmark with fixed iteration cap.

Uses a tiny positive gradient tolerance so the solver normally runs exactly
``max_iter`` steps unless it reaches exact stationarity. Reports total wall
time per completed iteration so implementations can be compared without
assuming that the cap was reached.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import jax_hf
from jax_hf import HartreeFockKernel, SolverConfig


def _make_kernel(nk: int, nb: int, T: float = 0.05, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    weights = jnp.ones((nk, nk), dtype=jnp.float32)
    h = np.zeros((nk, nk, nb, nb), dtype=np.complex64)
    diag = np.linspace(-1.0, 1.0, nb, dtype=np.float32)
    h[..., np.arange(nb), np.arange(nb)] = diag[None, None, :]
    A = rng.normal(scale=0.2, size=(nk, nk, nb, nb)).astype(np.float32) + \
        1j * rng.normal(scale=0.2, size=(nk, nk, nb, nb)).astype(np.float32)
    A = 0.5 * (A + np.conj(A.swapaxes(-1, -2)))
    h = h + A.astype(np.complex64)
    coulomb_q = jnp.full((nk, nk, 1, 1), 0.1, dtype=jnp.complex64)
    return HartreeFockKernel(
        weights=weights, hamiltonian=jnp.asarray(h),
        coulomb_q=coulomb_q, T=T,
    )


def _bench_fixed_iter(kernel, n_electrons: float, *, max_iter: int = 30,
                      n_warmup: int = 1, n_repeats: int = 5):
    """Force max_iter steps so we time per-iter cost only."""
    P0 = jnp.zeros_like(kernel.h)
    cfg = SolverConfig(max_iter=max_iter, tol_grad=1e-30)

    # Compile + warmup
    res = jax_hf.solve(kernel, P0, n_electrons, config=cfg)
    jax.block_until_ready(res.energy)
    for _ in range(n_warmup):
        res = jax_hf.solve(kernel, P0, n_electrons, config=cfg)
        jax.block_until_ready(res.energy)

    # Time
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        res = jax_hf.solve(kernel, P0, n_electrons, config=cfg)
        jax.block_until_ready(res.energy)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.min(times)), int(res.n_iter)


def main():
    import sys
    cases = [
        ("4x4x2",   4,   2,  1.0),
        ("16x16x4", 16,  4,  4.0),
        ("16x16x8", 16,  8,  8.0),
        ("32x32x16", 32, 16, 32.0),
    ]
    max_iter = 30
    print(f"Fixed iter benchmark (max_iter={max_iter})", flush=True)
    print(f"{'case':<10} {'mean':>10} {'min':>10} {'per-iter (ms)':>16}", flush=True)
    print("-" * 55, flush=True)
    for name, nk, nb, ne in cases:
        kernel = _make_kernel(nk, nb)
        try:
            mt, mnt, ni = _bench_fixed_iter(kernel, ne, max_iter=max_iter)
        except Exception as e:
            print(f"{name:<10}  FAILED: {e}", flush=True)
            continue
        per_iter = mt / ni * 1000.0
        per_iter_min = mnt / ni * 1000.0
        print(f"{name:<10} {mt*1000:>9.1f}ms {mnt*1000:>9.1f}ms {per_iter:>7.3f} ({per_iter_min:.3f} min)", flush=True)


if __name__ == "__main__":
    main()
