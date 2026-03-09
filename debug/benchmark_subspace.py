"""Benchmark: QR solver vs active-subspace QR solver.

Compares wall time, iteration count, and final energy for various problem
sizes and active_size settings.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

from jax_hf.main import HartreeFockKernel
import jax_hf


def make_model(nk: int, nb: int, T: float, coupling: float = 0.15, seed: int = 42):
    """Create a random Hermitian Hamiltonian with band structure."""
    rng = np.random.RandomState(seed)

    # Diagonal energies with clear band structure
    diag_energies = np.linspace(-2.0, 2.0, nb)
    H = np.zeros((nk, nk, nb, nb), dtype=np.complex64)
    for i in range(nb):
        H[..., i, i] = diag_energies[i]

    # Add random off-diagonal couplings
    for i in range(nb):
        for j in range(i + 1, nb):
            v = coupling * (rng.randn(nk, nk) + 1j * rng.randn(nk, nk)).astype(np.complex64)
            H[..., i, j] = v
            H[..., j, i] = np.conj(v)

    weights = np.ones((nk, nk), dtype=np.float32)
    Vq = (0.1 * np.ones((nk, nk, 1, 1), dtype=np.float32)).astype(np.complex64)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=jnp.asarray(H),
        coulomb_q=jnp.asarray(Vq),
        T=T,
        include_hartree=False,
        include_exchange=True,
    )
    P0 = 0.5 * jnp.broadcast_to(
        jnp.eye(nb, dtype=jnp.complex64), (nk, nk, nb, nb)
    )
    electrondensity0 = float(nb) / 2.0
    return kernel, P0, electrondensity0


def run_benchmark(nk, nb, T, active_sizes, max_iter=120, n_warmup=1, n_runs=3):
    print(f"\n{'='*70}")
    print(f"  nk={nk}x{nk}, nb={nb}, T={T}, max_iter={max_iter}")
    print(f"{'='*70}")

    kernel, P0, ne0 = make_model(nk, nb, T)
    common_kwargs = dict(
        electrondensity0=ne0,
        max_iter=max_iter,
        comm_tol=1e-5,
        p_tol=1e-5,
        inner_sweeps=2,
        line_search=True,
    )

    # --- QR baseline ---
    qr_runner = jax_hf.jit_variational_qr_iteration(kernel)
    for _ in range(n_warmup):
        P_qr, F_qr, E_qr, mu_qr, k_qr, hist_qr = qr_runner(P0, **common_kwargs)

    times_qr = []
    for _ in range(n_runs):
        jax.block_until_ready(P_qr)
        t0 = time.perf_counter()
        P_qr, F_qr, E_qr, mu_qr, k_qr, hist_qr = qr_runner(P0, **common_kwargs)
        jax.block_until_ready(P_qr)
        times_qr.append(time.perf_counter() - t0)

    t_qr = np.median(times_qr)
    k_qr_int = int(k_qr)
    E_qr_val = float(E_qr)
    dC_qr = float(hist_qr["dC"][max(k_qr_int - 1, 0)])

    print(f"\n  QR baseline:")
    print(f"    iters={k_qr_int:3d}  E={E_qr_val:.8f}  dC={dC_qr:.2e}  time={t_qr:.4f}s")

    # --- Subspace variants ---
    sub_runner = jax_hf.jit_variational_qr_subspace_iteration(kernel)
    for active_size in active_sizes:
        label = f"active_size={active_size}" if active_size is not None else "active_size=None"
        sub_kwargs = dict(**common_kwargs, active_size=active_size)

        for _ in range(n_warmup):
            P_sub, F_sub, E_sub, mu_sub, k_sub, hist_sub = sub_runner(P0, **sub_kwargs)

        times_sub = []
        for _ in range(n_runs):
            jax.block_until_ready(P_sub)
            t0 = time.perf_counter()
            P_sub, F_sub, E_sub, mu_sub, k_sub, hist_sub = sub_runner(P0, **sub_kwargs)
            jax.block_until_ready(P_sub)
            times_sub.append(time.perf_counter() - t0)

        t_sub = np.median(times_sub)
        k_sub_int = int(k_sub)
        E_sub_val = float(E_sub)
        dC_sub = float(hist_sub["dC"][max(k_sub_int - 1, 0)])
        dE = abs(E_sub_val - E_qr_val)
        speedup = t_qr / t_sub if t_sub > 0 else float("inf")

        n_active_used = np.array(hist_sub["n_active"][:k_sub_int])
        mask_used = np.array(hist_sub["active_mask"][:k_sub_int])

        print(f"\n  Subspace ({label}):")
        print(f"    iters={k_sub_int:3d}  E={E_sub_val:.8f}  dC={dC_sub:.2e}  time={t_sub:.4f}s")
        print(f"    |dE|={dE:.2e}  speedup={speedup:.2f}x")
        print(f"    n_active per iter: {n_active_used[:5]}{'...' if k_sub_int > 5 else ''}")

        # Check energy agreement
        if dE > 1e-3:
            print(f"    WARNING: energy disagrees by {dE:.2e}")


if __name__ == "__main__":
    print("Benchmarking QR vs active-subspace QR solver")
    print(f"JAX devices: {jax.devices()}")

    # Small problem: nb=4, nk=4
    run_benchmark(nk=4, nb=4, T=0.05, active_sizes=[None, 4, 3, 2])

    # Medium problem: nb=8, nk=4
    run_benchmark(nk=4, nb=8, T=0.05, active_sizes=[None, 8, 6, 4])

    # Larger k-mesh: nb=8, nk=12
    run_benchmark(nk=12, nb=8, T=0.05, active_sizes=[None, 6, 4])

    # Large bands: nb=16, nk=4
    run_benchmark(nk=4, nb=16, T=0.05, active_sizes=[None, 12, 8])
