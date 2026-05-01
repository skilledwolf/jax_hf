"""Benchmark for the Cayley-related optimizations.

Measures per-iteration wall time and total convergence wall time on
representative test cases:

  * tiny: 4x4 grid, nb=2 (used by unit tests; BT loop usually trivial)
  * medium: 16x16 grid, nb=8 (representative of bilayer-projected models)
  * large: 32x32 grid, nb=16 (representative of multilayer)

Reports both first-call (compile + run) and warm-call wall times.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import jax_hf
from jax_hf import HartreeFockKernel, SolverConfig


def _make_kernel(nk: int, nb: int, T: float = 0.05, rng_seed: int = 0):
    """Build a synthetic kernel with off-diagonal hopping (mixed orbitals)."""
    rng = np.random.default_rng(rng_seed)
    weights = jnp.ones((nk, nk), dtype=jnp.float32)
    h = np.zeros((nk, nk, nb, nb), dtype=np.complex64)
    diag = np.linspace(-1.0, 1.0, nb, dtype=np.float32)
    h[..., np.arange(nb), np.arange(nb)] = diag[None, None, :]
    # Random off-diagonal hopping mixing flavors
    A = rng.normal(scale=0.2, size=(nk, nk, nb, nb)).astype(np.float32) + \
        1j * rng.normal(scale=0.2, size=(nk, nk, nb, nb)).astype(np.float32)
    A = 0.5 * (A + np.conj(A.swapaxes(-1, -2)))
    h = h + A.astype(np.complex64)
    coulomb_q = jnp.full((nk, nk, 1, 1), 0.1, dtype=jnp.complex64)
    return HartreeFockKernel(
        weights=weights, hamiltonian=jnp.asarray(h),
        coulomb_q=coulomb_q, T=T,
    )


def _bench_run(kernel, n_electrons: float, *, max_iter: int = 100,
               tol_E: float = 1e-7, n_warmup: int = 1, n_repeats: int = 3):
    """Benchmark a full solve.  Returns (compile_time, mean_run_time, min_run_time, n_iter, energy)."""
    P0 = jnp.zeros_like(kernel.h)
    cfg = SolverConfig(max_iter=max_iter, tol_E=tol_E)

    # First call: compile + run
    t0 = time.perf_counter()
    res = jax_hf.solve(kernel, P0, n_electrons, config=cfg)
    jax.block_until_ready(res.energy)
    compile_time = time.perf_counter() - t0

    # Warmup runs
    for _ in range(n_warmup):
        res = jax_hf.solve(kernel, P0, n_electrons, config=cfg)
        jax.block_until_ready(res.energy)

    # Timed runs
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        res = jax_hf.solve(kernel, P0, n_electrons, config=cfg)
        jax.block_until_ready(res.energy)
        times.append(time.perf_counter() - t0)

    return compile_time, float(np.mean(times)), float(np.min(times)), \
           int(res.n_iter), float(res.energy), bool(res.converged)


def main():
    cases = [
        ("tiny",   4,   2,  1.0),
        ("small", 16,   4,  4.0),
        ("medium", 16,  8,  8.0),
        ("large",  32, 16, 32.0),
    ]
    print(f"{'case':<8} {'nk':>4} {'nb':>3} {'compile':>8} {'mean':>8} {'min':>8} {'n_iter':>6} {'E':>14} {'cnv':>4}")
    print("-" * 75)
    for name, nk, nb, ne in cases:
        kernel = _make_kernel(nk, nb)
        ct, mt, mnt, ni, E, cnv = _bench_run(kernel, ne)
        print(f"{name:<8} {nk:>4} {nb:>3} {ct*1000:>7.1f}ms {mt*1000:>7.1f}ms {mnt*1000:>7.1f}ms "
              f"{ni:>6} {E:>14.7e} {str(cnv):>4}")


if __name__ == "__main__":
    main()
