"""Per-component breakdown via jax.named_scope + jax.profiler trace.

Wraps each body() section in a named_scope so the profiler can attribute
wall time to each piece in the actual fused JIT graph (no per-call
dispatch overhead inflation, no XLA-fusion-disabled artifacts).
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import jax_hf
from jax_hf import HartreeFockKernel, SolverConfig


def _bilayer_like_kernel(nk=49, nb=16, T=0.5):
    rng = np.random.default_rng(0)
    weights = jnp.ones((nk, nk), dtype=jnp.float32) / (nk * nk)
    h = np.zeros((nk, nk, nb, nb), dtype=np.complex64)
    diag = np.linspace(-0.5, 0.5, nb, dtype=np.float32)
    h[..., np.arange(nb), np.arange(nb)] = diag[None, None, :]
    A = rng.normal(scale=0.05, size=(nk, nk, nb, nb)).astype(np.float32) + \
        1j * rng.normal(scale=0.05, size=(nk, nk, nb, nb)).astype(np.float32)
    A = 0.5 * (A + np.conj(A.swapaxes(-1, -2)))
    h = h + A.astype(np.complex64)
    coulomb_q = jnp.full((nk, nk, 1, 1), 0.05, dtype=jnp.complex64)
    return HartreeFockKernel(
        weights=weights, hamiltonian=jnp.asarray(h),
        coulomb_q=coulomb_q, T=T,
    )


def _ablation(kernel, n_electrons, max_iter, *, n_warmup=2, n_repeats=5):
    """Measure full solve() time at fixed iter cap.  Returns (min wall, n_iter)."""
    P0 = jnp.zeros_like(kernel.h)
    # tol_E=-1.0 forces dE > tol always (since dE = |E-E_prev| >= 0 > -1)
    cfg = SolverConfig(max_iter=max_iter, tol_E=-1.0, tol_grad=0.0)
    res = jax_hf.solve(kernel, P0, n_electrons, config=cfg)
    jax.block_until_ready(res.energy)
    n_iter_actual = int(res.n_iter)
    for _ in range(n_warmup):
        res = jax_hf.solve(kernel, P0, n_electrons, config=cfg)
        jax.block_until_ready(res.energy)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        res = jax_hf.solve(kernel, P0, n_electrons, config=cfg)
        jax.block_until_ready(res.energy)
        times.append(time.perf_counter() - t0)
    return float(np.min(times)), n_iter_actual


def main():
    nk, nb, T = 49, 16, 0.5
    kernel = _bilayer_like_kernel(nk=nk, nb=nb, T=T)
    n_electrons = float(kernel.w2d.sum() * nb / 2.0)  # half filling

    # --- Ablation by varying max_iter ---
    # Total = compile + per_iter * n_iter.  Using two iter counts:
    #   t(N1) = c + per_iter * N1
    #   t(N2) = c + per_iter * N2
    # → per_iter = (t(N2) - t(N1)) / (N2 - N1)
    iters = [10, 30, 60]
    print(f"Slope-fit per-iteration timing (nk={nk}, nb={nb}, T={T})", flush=True)
    print("=" * 60, flush=True)
    print(f"{'max_iter':>10} {'n_iter':>8} {'min wall (ms)':>15}", flush=True)
    times = {}
    actual_iters = {}
    for n in iters:
        t_s, n_actual = _ablation(kernel, n_electrons, n)
        t = t_s * 1000.0
        times[n] = t
        actual_iters[n] = n_actual
        print(f"{n:>10} {n_actual:>8} {t:>15.1f}", flush=True)

    print(flush=True)
    # Use actual n_iter for the slope fit
    pairs = sorted(set(actual_iters.values()))
    if len(pairs) >= 2:
        n0, n1 = pairs[0], pairs[-1]
        # find max_iter values that gave these n_iter
        m0 = next(m for m in iters if actual_iters[m] == n0)
        m1 = next(m for m in iters if actual_iters[m] == n1)
        per_iter = (times[m1] - times[m0]) / (n1 - n0)
        overhead = times[m0] - per_iter * n0
        print(f"Per-iter slope (linear fit on n_iter={n0} and {n1}): {per_iter:.3f} ms", flush=True)
        print(f"Constant overhead (compile + finalise): {overhead:.1f} ms", flush=True)
    else:
        print(f"All runs converged at the same iter count ({pairs}); cannot fit slope.", flush=True)


if __name__ == "__main__":
    main()
