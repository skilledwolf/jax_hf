"""
QR-retraction variational Hartree-Fock solver.

Uses the same frozen-F alternating minimization as variational.py, but replaces
Cayley/Jacobi orbital updates with a QR-retraction Riemannian gradient descent
on the unitary manifold.  The update stays entirely in the orbital basis:

    U  = QR(I - τ G)
    Q' = Q U
    Ft'= U† Ft U

where G = skew((p_j - p_i) Ft_ij / denom) is the preconditioned orbital
gradient, and Ft = Q†FQ is the Fock matrix in the orbital basis.

Because only the small nb×nb unitary U is factored and both Q and Ft are
updated via right-multiplication, the inner loop avoids recomputing Q†FQ
from the Hilbert-basis Fock matrix each step.

Advantages over Cayley:
  - No nb×nb linear solves (QR is cheaper for nb > ~12)
  - Orbital-basis updates avoid the cost of Q†FQ recomputation
  - More stable near degeneracies (gradient vanishes when p_i == p_j)
"""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from .linalg import normalize_block_specs
from .variational import (
    VariationalHFParams,
    _adjust_params_particle_number,
    _build_sigma_hartree,
    _density_from_Qp,
    _fock_in_orbital_basis,
    _frozen_F_energy,
    _hf_energy,
    _occupations_from_eps,
    _rotation_mask_from_block_sizes,
    _skew_hermitian,
    _solve_mu_fd_newton_bracket,
    _stationarity_norm,
    _weighted_rms_vec,
    init_variational_params_from_density,
)


# ----------------------------
# QR-specific orbital update
# ----------------------------


def _orbital_gradient(
    Ft: jax.Array, eps: jax.Array, gap: jax.Array, p: jax.Array, offdiag: jax.Array,
    p_floor: float, T: float, denom_scale: float,
) -> jax.Array:
    """
    Preconditioned skew-Hermitian Riemannian gradient with degenerate-subspace Jacobi fallback.

    Non-degenerate part: G = skew((p_j - p_i) * Ft / denom)  (preconditioned commutator)
    Degenerate part:     Jacobi angle × phase of Ft           (drives toward diagonalizing Ft)

    The energy-denominator preconditioning (denom = sqrt(gap² + λ²)) is critical for
    convergence: without it, large-gap pairs dominate the gradient and small rotations
    in near-degenerate subspaces are drowned out.

    Interpolated via occ_scale = |p_j - p_i| / (|p_j - p_i| + p_floor).
    """
    real_dtype = p.dtype
    n = Ft.shape[-1]
    tiny = jnp.asarray(1e-16, dtype=real_dtype)
    tiny_eps = jnp.asarray(1e-30, dtype=real_dtype)
    tiny_gap = jnp.asarray(1e-12, dtype=real_dtype)

    diff = p[..., None, :] - p[..., :, None]
    abs_diff = jnp.abs(diff)
    occ_scale = abs_diff / (abs_diff + p_floor)

    # Energy denominator preconditioning (same formula as inner solve / stationarity norm)
    eps_scale = jnp.sqrt(jnp.mean(eps ** 2) + tiny_eps)
    lam = jnp.maximum(
        jnp.asarray(T, dtype=real_dtype),
        jnp.asarray(denom_scale, dtype=real_dtype) * eps_scale,
    )
    denom = jnp.sqrt(gap ** 2 + lam ** 2)

    G_comm = _skew_hermitian(diff * Ft / denom) * offdiag

    # Jacobi fallback: anti-Hermitian generator that diagonalizes Ft in degenerate subspaces.
    # Key: use gap_ji = eps_j - eps_i (note j,i order) with pair_sign to break
    # the symmetry when gap=0, so that theta is antisymmetric and survives
    # anti-Hermitization.
    gap_ji = -gap  # (j,i) order
    orb_idx = jnp.arange(n, dtype=real_dtype)
    pair_sign = jnp.sign(orb_idx[None, :] - orb_idx[:, None])
    pair_sign = jnp.where(pair_sign == 0.0, 1.0, pair_sign)
    safe_gap_ji = jnp.where(jnp.abs(gap_ji) < tiny_gap, tiny_gap * pair_sign, gap_ji)

    abs_Ft = jnp.abs(Ft)
    theta = 0.5 * jnp.arctan(2.0 * abs_Ft / safe_gap_ji)
    phase = jnp.where(abs_Ft > tiny, Ft / abs_Ft, jnp.zeros_like(Ft))
    G_jac = _skew_hermitian(theta * phase * offdiag)

    return occ_scale * G_comm - (1.0 - occ_scale) * G_jac


def _adaptive_tau(
    G: jax.Array,
    eps: jax.Array,
    gap: jax.Array,
    w_norm: jax.Array,
    denom_scale: float,
    T: float,
) -> jax.Array:
    """Rayleigh quotient step size for orbital minimization."""
    real_dtype = jnp.real(eps).dtype

    tiny = jnp.asarray(1e-30, dtype=real_dtype)
    eps_scale = jnp.sqrt(jnp.mean(eps**2) + tiny)

    lam = jnp.maximum(
        jnp.asarray(T, dtype=real_dtype),
        jnp.asarray(denom_scale, dtype=real_dtype) * eps_scale,
    )

    G2 = jnp.abs(G)**2
    pair_weights = w_norm[..., None, None]

    # G is the preconditioned gradient (D = G_raw / denom).
    # The true directional gradient evaluated along D is sum(G_raw * D) = sum(|D|^2 * denom).
    denom = jnp.sqrt(gap**2 + lam**2)
    num = jnp.sum(pair_weights * G2 * denom)

    # The curvature evaluated along D is sum(|D|^2 * (|gap| + lam))
    den = jnp.sum(pair_weights * G2 * (jnp.abs(gap) + lam))

    tau = num / (den + tiny)
    return tau

def _qr_retraction_unitary(G: jax.Array, tau: jax.Array) -> jax.Array:
    """
    Orbital-basis QR retraction unitary U = QR(I - tau*G).

    The diagonal of R is only defined up to a unit-modulus phase for complex QR.
    Normalize the column phases so diag(R) is real and non-negative.
    """
    tiny = jnp.asarray(1e-30, dtype=jnp.real(G).dtype)
    n = G.shape[-1]
    eye = jnp.eye(n, dtype=G.dtype)
    U_trial = eye - tau * G
    U, R = jnp.linalg.qr(U_trial)

    phases = jnp.diagonal(R, axis1=-2, axis2=-1)
    phase_norm = jnp.where(jnp.abs(phases) > tiny, phases / jnp.abs(phases), jnp.ones_like(phases))
    return U * phase_norm[..., None, :]

def _apply_orbital_basis_update(Ft: jax.Array, U: jax.Array) -> jax.Array:
    """Update Ft in orbital basis under the right-unitary rotation U."""
    U_dag = U.conj().swapaxes(-1, -2)
    return U_dag @ Ft @ U

def _backtracking_qr_energy(
    Ft: jax.Array,
    G: jax.Array,
    *,
    p: jax.Array,
    w_norm: jax.Array,
    tau0: float,
    accept_ratio: float,
    shrink: float,
    max_backtrack: int,
) -> Tuple[jax.Array, jax.Array]:
    """
    Try tau = tau0, tau0*shrink, ... until the frozen-F energy does not increase.

    Returns (Ft_new, U). If no step is accepted, returns inputs unchanged and U=I.
    """
    real_dtype = p.dtype
    tau0 = jnp.asarray(tau0, dtype=real_dtype)
    shrink = jnp.asarray(shrink, dtype=real_dtype)
    accept_ratio = jnp.asarray(accept_ratio, dtype=real_dtype)

    E0 = _frozen_F_energy(Ft, p, w_norm)
    energy_tol = (1.0 - accept_ratio) * jnp.maximum(jnp.abs(E0), jnp.asarray(1.0, dtype=real_dtype))

    def cond(state):
        i, tau, accepted, Ft_best, U_best = state
        del tau, Ft_best, U_best
        return jnp.logical_and(i < jnp.int32(max_backtrack), jnp.logical_not(accepted))

    def body(state):
        i, tau, accepted, Ft_best, U_best = state

        U_trial = _qr_retraction_unitary(G, tau)
        Ft_trial = _apply_orbital_basis_update(Ft, U_trial)

        E_trial = _frozen_F_energy(Ft_trial, p, w_norm)
        ok = E_trial <= E0 + energy_tol

        def accept_step(_):
            return (Ft_trial, U_trial, jnp.bool_(True))

        def reject_step(_):
            return (Ft_best, U_best, accepted)

        Ft_best, U_best, accepted = lax.cond(ok, accept_step, reject_step, operand=None)

        tau = tau * shrink
        return (i + 1, tau, accepted, Ft_best, U_best)

    n = G.shape[-1]
    eye = jnp.zeros_like(G)
    eye = eye + jnp.eye(n, dtype=G.dtype)

    init_state = (jnp.int32(0), tau0, jnp.bool_(False), Ft, eye)
    _, _, _, Ft_out, U_out = lax.while_loop(cond, body, init_state)
    return Ft_out, U_out


# ----------------------------
# Frozen-F inner solve (QR)
# ----------------------------


def _frozenF_inner_solve_qr(
    Q: jax.Array,
    p: jax.Array,
    mu: jax.Array,
    *,
    Ft: jax.Array,
    w_norm: jax.Array,
    n_target_norm: jax.Array,
    T: float,
    inner_sweeps: int,
    q_sweeps: int,
    p_floor: float,
    denom_scale: float,
    max_rot: float,
    bt_accept: float,
    bt_shrink: float,
    bt_max: int,
    mu_maxiter: int,
    mu_tol: float,
    line_search: bool,
    rotation_block_sizes: tuple[int, ...] | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Cheap inner solve with F frozen, using QR-retraction orbital updates.
    """
    real_dtype = p.dtype
    max_rot_j = jnp.asarray(max_rot, dtype=real_dtype)
    tiny_eps = jnp.asarray(1e-30, dtype=real_dtype)
    clip_eps = jnp.asarray(1e-12, dtype=real_dtype)

    Ft0 = Ft
    n = Ft0.shape[-1]
    eye = jnp.eye(n, dtype=real_dtype)
    offdiag = (1.0 - eye)[None, None, ...]
    eye_u = jnp.zeros_like(Ft0)
    eye_u = eye_u + jnp.eye(n, dtype=Q.dtype)

    # Build rotation block mask (None → unrestricted).
    rot_mask: jax.Array | None = None
    if rotation_block_sizes is not None:
        rot_mask = _rotation_mask_from_block_sizes(rotation_block_sizes, n, real_dtype)

    def take_step(Ft_q: jax.Array, direction: jax.Array, p_q: jax.Array, tau: jax.Array) -> Tuple[jax.Array, jax.Array]:
        if line_search and bt_max > 0:
            Ft_new, U = _backtracking_qr_energy(
                Ft_q,
                direction,
                p=p_q,
                w_norm=w_norm,
                tau0=tau,
                accept_ratio=float(bt_accept),
                shrink=float(bt_shrink),
                max_backtrack=int(bt_max),
            )
        else:
            U = _qr_retraction_unitary(direction, tau)
            Ft_new = _apply_orbital_basis_update(Ft_q, U)
        return Ft_new, U

    def inner_sweep(_, state):
        Q_s, Ft_s, p_s, mu_s = state

        # 1) occupations from diagonal energies
        eps = jnp.real(jnp.diagonal(Ft_s, axis1=-2, axis2=-1)).astype(real_dtype)
        mu_s = _solve_mu_fd_newton_bracket(
            eps, w_norm, n_target_norm, mu_s, T,
            maxiter=int(mu_maxiter), tol=float(mu_tol),
        )
        p_s = _occupations_from_eps(eps, mu_s, T).astype(real_dtype)

        if q_sweeps == 1:
            eps_q = jnp.real(jnp.diagonal(Ft_s, axis1=-2, axis2=-1)).astype(real_dtype)
            gap_q = eps_q[..., :, None] - eps_q[..., None, :]
            G_step = _orbital_gradient(Ft_s, eps_q, gap_q, p_s, offdiag, float(p_floor), T, float(denom_scale))

            if rot_mask is not None:
                G_step = G_step * rot_mask

            tau = _adaptive_tau(G_step, eps_q, gap_q, w_norm, float(denom_scale), T)

            gen_norm = jnp.sqrt(jnp.sum(jnp.abs(G_step) ** 2, axis=(-2, -1)) + tiny_eps)
            eff_norm = tau * gen_norm
            clip_scale = jnp.minimum(1.0, max_rot_j / (eff_norm + clip_eps))
            G_step = G_step * clip_scale[..., None, None]

            Ft_s, U_step = take_step(Ft_s, G_step, p_s, tau)
            return (Q_s @ U_step, Ft_s, p_s, mu_s)

        # 2) Q sweeps via QR retraction (with optional Riemannian PR conjugate gradients)
        H_zero = jnp.zeros_like(Ft_s)

        def q_sweep(i_q, state2):
            Ft_q, U_total, G_prev, H_prev = state2

            eps_q = jnp.real(jnp.diagonal(Ft_q, axis1=-2, axis2=-1)).astype(real_dtype)
            gap_q = eps_q[..., :, None] - eps_q[..., None, :]

            G_new = _orbital_gradient(Ft_q, eps_q, gap_q, p_s, offdiag, float(p_floor), T, float(denom_scale))

            if rot_mask is not None:
                G_new = G_new * rot_mask

            def compute_beta():
                num = jnp.sum(jnp.real(jnp.conj(G_new) * (G_new - G_prev)), axis=(-2, -1))
                den = jnp.sum(jnp.real(jnp.conj(G_prev) * G_prev), axis=(-2, -1)) + tiny_eps
                return jnp.maximum(0.0, num / den)[..., None, None]

            beta = lax.cond(
                i_q > 0,
                compute_beta,
                lambda: jnp.zeros(G_new.shape[:-2] + (1, 1), dtype=real_dtype),
            )

            H_new = G_new + beta * H_prev

            tau = _adaptive_tau(H_new, eps_q, gap_q, w_norm, float(denom_scale), T)

            gen_norm = jnp.sqrt(jnp.sum(jnp.abs(H_new) ** 2, axis=(-2, -1)) + tiny_eps)
            eff_norm = tau * gen_norm
            clip_scale = jnp.minimum(1.0, max_rot_j / (eff_norm + clip_eps))
            H_step = H_new * clip_scale[..., None, None]

            Ft_new, U = take_step(Ft_q, H_step, p_s, tau)
            U_total = U_total @ U

            U_dag = U.conj().swapaxes(-1, -2)
            G_trans = U_dag @ G_new @ U
            H_trans = U_dag @ H_new @ U

            return (Ft_new, U_total, G_trans, H_trans)

        Ft_s, U_total, _, _ = lax.fori_loop(0, int(q_sweeps), q_sweep, (Ft_s, eye_u, H_zero, H_zero))
        Q_s = Q_s @ U_total
        return (Q_s, Ft_s, p_s, mu_s)

    Q_out, _, p_out, mu_out = lax.fori_loop(0, int(inner_sweeps), inner_sweep, (Q, Ft0, p, mu))
    return Q_out, p_out, mu_out


# ---------------------------------------
# Main JIT-friendly outer optimizer
# ---------------------------------------


def variational_qr_optimize(
    params0: VariationalHFParams,
    electrondensity0: float,
    *,
    # big dynamic inputs (match HartreeFockKernel.as_args())
    h: jax.Array,
    weights_b: jax.Array,
    weight_sum: jax.Array,
    VR: jax.Array,
    T: float,
    refP: jax.Array,
    HH: jax.Array,
    include_hartree: bool,
    include_exchange: bool,

    # outer loop controls
    max_iter: int = 80,
    comm_tol: float = 5e-4,
    p_tol: float = 5e-6,
    e_tol: float = 0.0,

    # inner loop (frozen-F) controls
    inner_sweeps: int = 2,
    q_sweeps: int = 1,
    p_floor: float = 0.10,
    denom_scale: float = 1e-3,
    max_rot: float = 0.60,
    bt_accept: float = 1.0,
    bt_shrink: float = 0.5,
    bt_max: int = 5,
    mu_maxiter: int = 25,
    mu_tol: float = 1e-12,
    line_search: bool = False,

    # rotation constraint (optional)
    rotation_block_sizes: tuple[int, ...] | None = None,

    # exchange knobs (optional)
    exchange_block_specs: Any | None = None,
    exchange_check_offdiag: bool | None = None,
    exchange_offdiag_atol: float = 1e-12,
    exchange_offdiag_rtol: float = 0.0,

    # symmetry projection (optional)
    project_fn=None,
):
    """
    QR-retraction variational HF solver.

    Same algorithm as variational_hartreefock_optimize (frozen-F alternating
    minimization) but with QR-retraction orbital updates instead of Cayley.

    Returns:
      P_fin, F_fin, E_fin, mu_fin, n_iter, history, params_fin
    """
    target_dtype = h.dtype
    real_dtype = jnp.zeros((), dtype=target_dtype).real.dtype

    _project = project_fn if project_fn is not None else (lambda A: A)

    # normalized weights (sum to 1)
    w2d = jnp.asarray(weights_b[..., 0, 0], dtype=real_dtype)
    weight_sum = jnp.asarray(weight_sum, dtype=real_dtype)
    w_norm = w2d / jnp.maximum(weight_sum, jnp.asarray(1e-30, dtype=real_dtype))

    n_target = jnp.asarray(electrondensity0, dtype=real_dtype)
    n_target_norm = n_target / jnp.maximum(weight_sum, jnp.asarray(1e-30, dtype=real_dtype))

    Q0 = jnp.asarray(params0.Q, dtype=target_dtype)
    p0 = jnp.asarray(params0.p, dtype=real_dtype)
    mu0 = jnp.asarray(params0.mu, dtype=real_dtype)

    hist_E = jnp.zeros(int(max_iter), dtype=real_dtype)
    hist_dC = jnp.zeros(int(max_iter), dtype=real_dtype)
    hist_dP = jnp.zeros(int(max_iter), dtype=real_dtype)
    hist_dE = jnp.zeros(int(max_iter), dtype=real_dtype)
    hist_mu = jnp.zeros(int(max_iter), dtype=real_dtype)

    comm_tol_r = jnp.asarray(comm_tol, dtype=real_dtype)
    p_tol_r = jnp.asarray(p_tol, dtype=real_dtype)
    e_tol_r = jnp.asarray(e_tol, dtype=real_dtype)

    # Cache last-built F if we finish on a no-update iteration.
    F_last0 = jnp.zeros_like(h)
    E_last0 = jnp.asarray(jnp.nan, dtype=real_dtype)
    fresh0 = jnp.bool_(False)

    carry0 = (
        jnp.int32(0),
        Q0,
        p0,
        mu0,
        jnp.asarray(jnp.inf, dtype=real_dtype),  # dC_prev
        jnp.asarray(jnp.inf, dtype=real_dtype),  # dP_prev
        jnp.asarray(jnp.inf, dtype=real_dtype),  # dE_prev
        jnp.asarray(jnp.nan, dtype=real_dtype),  # E_prev
        F_last0,
        E_last0,
        fresh0,
        hist_E, hist_dC, hist_dP, hist_dE, hist_mu,
    )

    def cond(carry):
        k = carry[0]
        dC_prev = carry[4]
        dP_prev = carry[5]
        dE_prev = carry[6]
        not_converged = jnp.logical_or(dC_prev > comm_tol_r, dP_prev > p_tol_r)
        not_converged = jnp.where(
            e_tol_r > 0,
            jnp.logical_or(not_converged, dE_prev > e_tol_r),
            not_converged,
        )
        return jnp.logical_and(k < max_iter, not_converged)

    def body(carry):
        (k, Q, p, mu, _dC_prev, _dP_prev, _dE_prev, E_prev,
         _F_last, _E_last, _fresh, hE, hC, hP, hdE, hM) = carry

        # Build density and Fock at current state
        P = _project(_density_from_Qp(Q, p))
        Sigma, H_shift, F = _build_sigma_hartree(
            P,
            h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange,
            include_hartree=include_hartree,
            exchange_block_specs=exchange_block_specs,
            exchange_check_offdiag=exchange_check_offdiag,
            exchange_offdiag_atol=exchange_offdiag_atol,
            exchange_offdiag_rtol=exchange_offdiag_rtol,
        )
        F = _project(F)
        E = _hf_energy(P, h=h, Sigma=Sigma, H=H_shift, weights_b=weights_b)
        E_real = jnp.real(E)

        # Energy change
        dE = jnp.abs(E_real - E_prev)
        dE = jnp.where(jnp.isfinite(dE), dE, jnp.asarray(jnp.inf, dtype=real_dtype))

        # HF stationarity measure
        Ft = _fock_in_orbital_basis(Q, F)
        dC = _stationarity_norm(Ft, p, w_norm, float(p_floor), T, float(denom_scale))

        # Fixed-F occupation residual at the current orbitals.
        eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)).astype(real_dtype)
        mu_fd = _solve_mu_fd_newton_bracket(
            eps, w_norm, n_target_norm, mu, T,
            maxiter=int(mu_maxiter), tol=float(mu_tol),
        )
        p_fd = _occupations_from_eps(eps, mu_fd, T).astype(real_dtype)
        dP = _weighted_rms_vec(p_fd - p, w_norm)

        # Record history
        hE = hE.at[k].set(E_real)
        hC = hC.at[k].set(dC)
        hP = hP.at[k].set(dP)
        hM = hM.at[k].set(mu_fd)
        hdE = hdE.at[k].set(dE)

        need_occ_update = dP > p_tol_r
        need_orbital_update = dC > comm_tol_r
        need_update = jnp.logical_or(need_occ_update, need_orbital_update)

        def do_update(args):
            Q_u, _p_u, _mu_u = args

            def orbital_update(inner_args):
                Q_i, p_i, mu_i = inner_args
                return _frozenF_inner_solve_qr(
                    Q_i, p_i, mu_i,
                    Ft=Ft, w_norm=w_norm, n_target_norm=n_target_norm, T=T,
                    inner_sweeps=int(inner_sweeps),
                    q_sweeps=int(q_sweeps),
                    p_floor=float(p_floor),
                    denom_scale=float(denom_scale),
                    max_rot=float(max_rot),
                    bt_accept=float(bt_accept),
                    bt_shrink=float(bt_shrink),
                    bt_max=int(bt_max),
                    mu_maxiter=int(mu_maxiter),
                    mu_tol=float(mu_tol),
                    line_search=bool(line_search),
                    rotation_block_sizes=rotation_block_sizes,
                )

            def occ_only_update(inner_args):
                Q_i, p_i, mu_i = inner_args
                del p_i, mu_i
                return (Q_i, p_fd, mu_fd)

            Q1, p1, mu1 = lax.cond(
                need_orbital_update,
                orbital_update,
                occ_only_update,
                operand=(Q_u, p_fd, mu_fd),
            )
            return (Q1, p1, mu1, dP)

        def no_update(args):
            Q_u, p_u, mu_u = args
            return (Q_u, p_u, mu_u, jnp.asarray(0.0, dtype=real_dtype))

        Q_new, p_new, mu_new, dP_prev = lax.cond(need_update, do_update, no_update, operand=(Q, p, mu))

        # F is fresh only if we didn't change (Q,p) this iteration.
        fresh = jnp.logical_not(need_update)
        F_last = jnp.where(need_update, _F_last, F)
        E_last = jnp.where(need_update, _E_last, E_real)

        return (k + 1, Q_new, p_new, mu_new, dC, dP_prev, dE, E_real,
                F_last, E_last, fresh, hE, hC, hP, hdE, hM)

    (
        k_fin, Q_fin, p_fin, mu_fin, _dC_fin, _dP_fin, _dE_fin, _E_prev_fin,
        F_last, E_last, fresh,
        hist_E, hist_dC, hist_dP, hist_dE, hist_mu,
    ) = lax.while_loop(cond, body, carry0)

    # Final consistent outputs at returned state.
    def finalize_rebuild(_):
        P_fin = _project(_density_from_Qp(Q_fin, p_fin))
        Sigma_fin, H_fin, F_fin = _build_sigma_hartree(
            P_fin,
            h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange,
            include_hartree=include_hartree,
            exchange_block_specs=exchange_block_specs,
            exchange_check_offdiag=exchange_check_offdiag,
            exchange_offdiag_atol=exchange_offdiag_atol,
            exchange_offdiag_rtol=exchange_offdiag_rtol,
        )
        F_fin = _project(F_fin)
        E_fin = _hf_energy(P_fin, h=h, Sigma=Sigma_fin, H=H_fin, weights_b=weights_b)
        return P_fin, F_fin, jnp.real(E_fin)

    def finalize_reuse(_):
        P_fin = _project(_density_from_Qp(Q_fin, p_fin))
        return P_fin, F_last, E_last

    P_fin, F_fin, E_fin = lax.cond(fresh, finalize_reuse, finalize_rebuild, operand=None)

    history = dict(E=hist_E, dC=hist_dC, dP=hist_dP, dE=hist_dE, mu=hist_mu)
    params_fin = VariationalHFParams(Q=Q_fin, p=p_fin, mu=mu_fin)
    return P_fin, F_fin, E_fin, mu_fin, k_fin, history, params_fin


# ---------------------------------------
# Kernel-style wrapper (matches jax_hf API)
# ---------------------------------------


def jit_variational_qr_iteration(hf_step):
    """
    Create a JIT-compiled QR-retraction variational solver.

    Same API as jit_variational_hartreefock_iteration but uses QR orbital updates.
    """
    compiled = jax.jit(
        variational_qr_optimize,
        static_argnames=(
            "max_iter", "comm_tol", "p_tol", "e_tol",
            "inner_sweeps", "q_sweeps",
            "p_floor", "denom_scale", "max_rot",
            "bt_accept", "bt_shrink", "bt_max",
            "mu_maxiter", "mu_tol",
            "line_search",
            "rotation_block_sizes",
            "include_hartree", "include_exchange",
            "exchange_block_specs", "exchange_check_offdiag",
            "exchange_offdiag_atol", "exchange_offdiag_rtol",
            "project_fn",
        ),
    )
    _jit_init = jax.jit(
        init_variational_params_from_density,
        static_argnames=("method", "occ_clip"),
    )
    _jit_adjust_N = jax.jit(_adjust_params_particle_number)

    def run(
        P0: jax.Array | None = None,
        electrondensity0: float = 0.0,
        *,
        params0: VariationalHFParams | None = None,
        init_method: str = "identity",
        return_params: bool = False,
        **kwargs,
    ):
        if params0 is None:
            if P0 is None:
                raise ValueError("Either P0 or params0 must be provided")
            params0_local = _jit_init(
                P0,
                electrondensity0,
                weights_b=hf_step.weights_b,
                weight_sum=hf_step.weight_sum,
                method=init_method,
            )
        else:
            if not isinstance(params0, VariationalHFParams):
                raise TypeError("params0 must be VariationalHFParams(Q, p, mu)")
            params0_local = _jit_adjust_N(
                params0,
                electrondensity0,
                weights_b=hf_step.weights_b,
                weight_sum=hf_step.weight_sum,
            )

        if kwargs.get("exchange_block_specs") is not None:
            kwargs["exchange_block_specs"] = normalize_block_specs(kwargs["exchange_block_specs"])

        P_fin, F_fin, E_fin, mu_fin, n_iter, history, params_fin = compiled(
            params0_local,
            electrondensity0,
            **hf_step.as_args(),
            **kwargs,
        )

        if return_params:
            return P_fin, F_fin, E_fin, mu_fin, n_iter, history, params_fin
        return P_fin, F_fin, E_fin, mu_fin, n_iter, history

    return run
