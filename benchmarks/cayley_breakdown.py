"""Per-component time breakdown of one DM body() iteration.

Builds a representative carry, then times each sub-operation in isolation:
  * build_fock (FFT exchange dominates)
  * Ft = Q† F Q
  * gradient + preconditioner
  * mu solve
  * eigh of i*d_Q (spectral-Cayley setup)
  * Ft_eig = V_d† Ft V_d
  * frozen-F evaluation per tau (Hadamard + matmul)
  * full backtracking loop (1 successful + 0 to ~3 BT trials)
  * post-line-search retraction (U from spectrum, Q@U, eps_new)

Each piece is JIT-compiled and timed independently with warmup.  Reports
mean wall time + % of the sum.  Not exact (per-call dispatch overhead is
amortised differently inside a fused JIT graph), but accurate enough to
identify the dominant terms.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import jax_hf
from jax_hf import HartreeFockKernel
from jax_hf.fock import build_fock, free_energy, hf_energy
from jax_hf.solver import (
    _cayley_spectral_setup,
    _cayley_unitary_from_spectrum,
    _diag_UFU_from_spectrum,
    _density_from_Qp,
    _fock_in_orbital_basis,
    _herm,
    _logit,
    _norm_matrix,
    _skew_hermitian,
    _solve_mu,
)


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


def _build_carry(kernel, n_electrons):
    """Run one solver step partially to get realistic mid-iteration carry."""
    args = kernel.as_args()
    h = args["h"]
    nk1, nk2, nb, _ = h.shape
    real_dtype = h.real.dtype

    # Initialize from F[P0=0]
    P0 = jnp.zeros_like(h)
    Sigma, H_h, F = build_fock(
        P0, h=h, VR=args["VR"], refP=args["refP"], HH=args["HH"], w2d=kernel.w2d,
        include_exchange=args["include_exchange"], include_hartree=args["include_hartree"],
        exchange_hermitian_channel_packing=args["exchange_hermitian_channel_packing"],
        contact_g=args["contact_g"], contact_Oi=args["contact_Oi"], contact_Oj=args["contact_Oj"],
    )
    eps0, Q = jnp.linalg.eigh(F)
    Q = Q.astype(h.dtype)
    eps = eps0.astype(real_dtype)
    w_norm = kernel.w2d / jnp.maximum(jnp.sum(kernel.w2d), 1e-30)
    n_target_norm = jnp.asarray(n_electrons, dtype=real_dtype) / jnp.maximum(jnp.sum(kernel.w2d), 1e-30)
    T_r = jnp.asarray(kernel.T, dtype=real_dtype)
    mu = _solve_mu(eps, w_norm, n_target_norm, jnp.asarray(0.0, dtype=real_dtype), T_r)
    p = jax.nn.sigmoid((mu - eps) / T_r).astype(real_dtype)

    # Build P, F, Ft, gradient
    P = _herm(_density_from_Qp(Q, p))
    Sigma, H_h, F = build_fock(
        P, h=h, VR=args["VR"], refP=args["refP"], HH=args["HH"], w2d=kernel.w2d,
        include_exchange=args["include_exchange"], include_hartree=args["include_hartree"],
        exchange_hermitian_channel_packing=args["exchange_hermitian_channel_packing"],
        contact_g=args["contact_g"], contact_Oi=args["contact_Oi"], contact_Oj=args["contact_Oj"],
    )
    Ft = _fock_in_orbital_basis(Q, F)
    eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)).astype(real_dtype)
    diff_p = p[..., None, :] - p[..., :, None]
    G_Q = _skew_hermitian(diff_p * Ft) * (1.0 - jnp.eye(nb, dtype=real_dtype))[None, None, ...]
    # crude preconditioner
    gap = eps[..., :, None] - eps[..., None, :]
    denom = jnp.sqrt(gap ** 2 + 0.01)
    d_Q = G_Q / denom

    return dict(
        kernel=kernel, h=h, P=P, F=F, Ft=Ft, eps=eps, Q=Q, p=p, mu=mu,
        d_Q=d_Q, w_norm=w_norm, n_target_norm=n_target_norm, T_r=T_r,
        weights_b=args["weights_b"],
    )


def _bench(fn, args_tuple, *, name, n_warmup=3, n_repeats=20):
    """Compile, warm up, and time."""
    fn_jit = jax.jit(fn)
    out = fn_jit(*args_tuple)
    jax.block_until_ready(out)
    for _ in range(n_warmup):
        out = fn_jit(*args_tuple)
        jax.block_until_ready(out)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn_jit(*args_tuple)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.min(times)), float(np.mean(times)), name


def main():
    nk, nb = 49, 16
    n_electrons = nk * nk * nb / 2.0  # half-filling
    print(f"Per-component breakdown — bilayer-like (nk={nk}, nb={nb}, T=0.5)", flush=True)
    print("=" * 78, flush=True)

    kernel = _bilayer_like_kernel(nk=nk, nb=nb)
    c = _build_carry(kernel, n_electrons)
    args = kernel.as_args()
    h = c["h"]
    P = c["P"]
    F = c["F"]
    Ft = c["Ft"]
    eps = c["eps"]
    Q = c["Q"]
    p = c["p"]
    mu = c["mu"]
    d_Q = c["d_Q"]
    w_norm = c["w_norm"]
    n_target_norm = c["n_target_norm"]
    T_r = c["T_r"]
    weights_b = c["weights_b"]

    # Pre-compute spectral setup outputs for downstream timings
    V_d, lam_d = _cayley_spectral_setup(d_Q)
    Ft_eig_pre = jnp.conj(jnp.swapaxes(V_d, -2, -1)) @ Ft @ V_d

    results = []

    # ======== 1. Fock build (the dominant Fock-side cost) ========
    def fock_build(P, h, VR, refP, HH, w2d):
        return build_fock(
            P, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=True, include_hartree=False,
            exchange_hermitian_channel_packing=args["exchange_hermitian_channel_packing"],
            contact_g=args["contact_g"], contact_Oi=args["contact_Oi"], contact_Oj=args["contact_Oj"],
        )
    results.append(_bench(fock_build, (P, h, args["VR"], args["refP"], args["HH"], kernel.w2d),
                          name="build_fock"))

    # ======== 2. Ft = Q† F Q ========
    results.append(_bench(_fock_in_orbital_basis, (Q, F),
                          name="Ft = Q^dag F Q"))

    # ======== 3. Energy evaluation ========
    Sigma_zero = jnp.zeros_like(F)
    H_zero = jnp.zeros_like(F)
    def energy_call(P, h, Sigma, H, weights_b):
        return hf_energy(P, h=h, Sigma=Sigma, H=H, weights_b=weights_b)
    results.append(_bench(energy_call, (P, h, Sigma_zero, H_zero, weights_b),
                          name="hf_energy (E)"))

    # ======== 4. Gradient (analytic) ========
    def gradient(p, Ft):
        diff_p = p[..., None, :] - p[..., :, None]
        offdiag = (1.0 - jnp.eye(nb, dtype=p.dtype))[None, None, ...]
        return _skew_hermitian(diff_p * Ft) * offdiag
    results.append(_bench(gradient, (p, Ft), name="gradient G_Q"))

    # ======== 5. mu solve (25 Newton-bracket iters) ========
    def mu_solve(eps, w_norm, n_target_norm, mu, T_r):
        return _solve_mu(eps, w_norm, n_target_norm, mu, T_r, maxiter=25)
    results.append(_bench(mu_solve, (eps, w_norm, n_target_norm, mu, T_r),
                          name="solve_mu (25 iters)"))

    # ======== 6. eigh of (i * d_Q) — fix(3) setup ========
    results.append(_bench(_cayley_spectral_setup, (d_Q,),
                          name="eigh(i*d_Q) [fix3 setup]"))

    # ======== 7. Ft_eig = V† Ft V ========
    def make_Ft_eig(V_d, Ft):
        return jnp.conj(jnp.swapaxes(V_d, -2, -1)) @ Ft @ V_d
    results.append(_bench(make_Ft_eig, (V_d, Ft), name="Ft_eig = V^dag Ft V"))

    # ======== 8. One frozen_F trial (Hadamard + matmul + clip + free_energy) ========
    def frozen_F_one_trial(V_d, Ft_eig, lam_d, tau, p, d_p, w_norm, T_r):
        eps_trial = _diag_UFU_from_spectrum(V_d, Ft_eig, lam_d, tau)
        p_trial = jnp.clip(p - tau * d_p, 1e-8, 1.0 - 1e-8)
        E_frozen = jnp.sum(w_norm[..., None] * p_trial * eps_trial)
        return free_energy(E_frozen, p_trial, w_norm, T_r)
    d_p_zero = jnp.zeros_like(p)  # gradient on p is zero by construction
    results.append(_bench(frozen_F_one_trial,
                          (V_d, Ft_eig_pre, lam_d, jnp.float32(0.5), p, d_p_zero, w_norm, T_r),
                          name="frozen_F per-tau trial"))

    # ======== 9. Cayley unitary from spectrum ========
    results.append(_bench(_cayley_unitary_from_spectrum,
                          (V_d, lam_d, jnp.float32(0.5)),
                          name="U(tau) from spectrum"))

    # ======== 10. Q @ U ========
    U_ref = _cayley_unitary_from_spectrum(V_d, lam_d, jnp.float32(0.5))
    def q_at_u(Q, U):
        return Q @ U
    results.append(_bench(q_at_u, (Q, U_ref), name="Q @ U"))

    # ======== 11. eps_new from spectrum (final retraction) ========
    results.append(_bench(_diag_UFU_from_spectrum,
                          (V_d, Ft_eig_pre, lam_d, jnp.float32(0.5)),
                          name="eps_new from spectrum"))

    # ---- Report ----
    total_min = sum(t_min for t_min, _, _ in results)
    total_mean = sum(t_mean for _, t_mean, _ in results)
    print(f"\n{'op':<35}  {'min (us)':>10}  {'mean (us)':>10}  {'% of sum':>10}", flush=True)
    print("-" * 78, flush=True)
    for t_min, t_mean, name in results:
        print(f"{name:<35}  {t_min*1e6:>10.1f}  {t_mean*1e6:>10.1f}  {t_min/total_min*100:>9.1f}%",
              flush=True)
    print("-" * 78, flush=True)
    print(f"{'sum':<35}  {total_min*1e6:>10.1f}  {total_mean*1e6:>10.1f}  100.0%", flush=True)
    print(flush=True)
    print("Notes:", flush=True)
    print("  - Each op is JIT-compiled and timed in isolation, so XLA fusion", flush=True)
    print("    that would happen in the body() jit may be missed here.  Treat", flush=True)
    print("    these as fraction-of-total estimates, not absolute body() costs.", flush=True)
    print("  - frozen_F trial cost x N_BT + (Q@U + U(tau) + eps_new) gives", flush=True)
    print("    the line-search section.  build_fock dominates for nb<<nk.", flush=True)


if __name__ == "__main__":
    main()
