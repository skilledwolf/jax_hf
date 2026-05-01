"""Profile _solve_mu convergence: residual after each Newton-bracket iter.

The Newton iteration with bracket safeguards converges quadratically once the
bracket is small.  With a warm-started mu (≈ true mu from prev outer iter)
we expect ~3–5 iters to machine precision.  Cold-start (mu=0) takes a few
more.  This benchmark measures both regimes.
"""

import numpy as np

import jax
import jax.numpy as jnp

from jax_hf.solver import _solve_mu


def _residual(mu, eps, w_norm, n_target_norm, T_r):
    """Residual = sum(w_norm * sigmoid((mu - eps)/T)) - n_target_norm."""
    p = jax.nn.sigmoid((mu - eps) / T_r)
    return float(jnp.abs(jnp.sum(w_norm[..., None] * p) - n_target_norm))


def main():
    rng = np.random.default_rng(0)
    nk1, nk2, nb = 49, 49, 16
    real_dtype = jnp.float32

    # Stress: metallic (continuous DOS at Fermi level), very low T
    eps_base = np.linspace(-1.0, 1.0, nb, dtype=np.float32)
    eps = jnp.asarray(
        eps_base[None, None, :] + 0.05 * rng.normal(size=(nk1, nk2, 1)).astype(np.float32)
    )
    w_norm = jnp.ones((nk1, nk2), dtype=real_dtype) / (nk1 * nk2)
    n_target_norm = jnp.asarray(nb / 2.0, dtype=real_dtype)
    T_r = jnp.asarray(0.005, dtype=real_dtype)  # very small T — stress case

    # True mu — get it by running with very large maxiter
    mu_true = _solve_mu(eps, w_norm, n_target_norm,
                        jnp.asarray(0.0, dtype=real_dtype), T_r, maxiter=200)
    res_true = _residual(mu_true, eps, w_norm, n_target_norm, T_r)
    print(f"Reference: mu_true = {float(mu_true):.10e}, |residual| = {res_true:.3e}", flush=True)
    print()

    print("Cold start (mu0 = 0):", flush=True)
    print(f"{'maxiter':>8} {'mu':>20} {'|residual|':>15} {'|mu - mu_true|':>18}", flush=True)
    for n in [3, 5, 8, 10, 15, 20, 25, 50]:
        mu = _solve_mu(eps, w_norm, n_target_norm,
                       jnp.asarray(0.0, dtype=real_dtype), T_r, maxiter=n)
        r = _residual(mu, eps, w_norm, n_target_norm, T_r)
        d = float(jnp.abs(mu - mu_true))
        print(f"{n:>8} {float(mu):>20.10e} {r:>15.3e} {d:>18.3e}", flush=True)

    print()
    print("Warm start (mu0 = mu_true + 1e-3 perturbation):", flush=True)
    print(f"{'maxiter':>8} {'mu':>20} {'|residual|':>15} {'|mu - mu_true|':>18}", flush=True)
    mu_warm = mu_true + jnp.asarray(1e-3, dtype=real_dtype)
    for n in [1, 2, 3, 5, 8, 10]:
        mu = _solve_mu(eps, w_norm, n_target_norm,
                       mu_warm, T_r, maxiter=n)
        r = _residual(mu, eps, w_norm, n_target_norm, T_r)
        d = float(jnp.abs(mu - mu_true))
        print(f"{n:>8} {float(mu):>20.10e} {r:>15.3e} {d:>18.3e}", flush=True)


if __name__ == "__main__":
    main()
