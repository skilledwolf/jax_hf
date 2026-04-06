"""
Frozen-Fock variational Hartree-Fock solver with QR retraction.

This implementation alternates:

  1. At fixed orbitals Q, update occupations p with the same FD/Newton-bracket
     step used by the rest of the variational code.
  2. At fixed updated occupations p, minimize the frozen-Fock energy
     E(Omega) = Tr(diag(p) U'FtU) over right-unitary orbital rotations
     using preconditioned gradient descent with QR retraction.  The
     diagonal Hessian preconditioner gives a step ≈ Jacobi rotation
     angle per element, typically converging the inner loop in 1-2 steps.
  3. Rebuild the mean field after every outer update for self-consistency.

The gradient uses the closed-form commutator formula G_ij = (p_j-p_i)*Ft_ij,
avoiding JAX autodiff overhead.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from .utils import validate_electron_count
from .variational import (
    VariationalHFParams,
    _adjust_params_particle_number,
    _build_sigma_hartree,
    _density_from_Qp,
    _fock_in_orbital_basis,
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
# QR-specific orbital update and helpers
# ----------------------------

def _frobenius_ip(a: jax.Array, b: jax.Array) -> jax.Array:
    """Per-k-point Frobenius inner product: Re(Tr(a† b)), reducing over (..., n, n)."""
    return jnp.sum(jnp.real(jnp.conj(a) * b), axis=(-2, -1))


def _inner_product_w(X: jax.Array, Y: jax.Array, w_norm: jax.Array) -> jax.Array:
    per_k = _frobenius_ip(X, Y)
    return jnp.sum(w_norm * per_k) / jnp.maximum(1e-30, jnp.sum(w_norm))


def _block_slices_from_sizes(block_sizes: tuple[int, ...], n: int) -> tuple[slice, ...]:
    if sum(int(size) for size in block_sizes) != int(n):
        raise ValueError(f"block_sizes must sum to {n}, got {block_sizes!r}.")

    start = 0
    slices = []
    for size in block_sizes:
        stop = start + int(size)
        slices.append(slice(start, stop))
        start = stop
    return tuple(slices)


def _exchange_block_specs_from_block_sizes(
    block_sizes: tuple[int, ...] | None,
) -> tuple[tuple[str, tuple[Any, ...]], ...] | None:
    if block_sizes is None:
        return None
    return (("sizes", tuple(int(size) for size in block_sizes)),)


def _qr_retraction_unitary(
    G: jax.Array,
    tau: jax.Array,
    *,
    block_slices: tuple[slice, ...] | None = None,
) -> jax.Array:
    tiny = jnp.asarray(1e-30, dtype=jnp.real(G).dtype)
    tau_bc = jnp.asarray(tau, dtype=jnp.real(G).dtype)[..., None, None]
    n = G.shape[-1]

    def _normalize_qr_columns(U: jax.Array, R: jax.Array) -> jax.Array:
        phases = jnp.diagonal(R, axis1=-2, axis2=-1)
        phase_norm = jnp.where(
            jnp.abs(phases) > tiny,
            phases / jnp.abs(phases),
            jnp.ones_like(phases),
        )
        return U * phase_norm[..., None, :]

    if block_slices is None or len(block_slices) <= 1:
        U_trial = jnp.eye(n, dtype=G.dtype) - tau_bc * G
        U, R = jnp.linalg.qr(U_trial)
        return _normalize_qr_columns(U, R)

    out = jnp.zeros_like(G)
    for s in block_slices:
        size = int(s.stop - s.start)
        U_block, R_block = jnp.linalg.qr(
            jnp.eye(size, dtype=G.dtype) - tau_bc * G[..., s, s]
        )
        out = out.at[..., s, s].set(_normalize_qr_columns(U_block, R_block))
    return out


def _apply_right_unitary(
    X: jax.Array,
    U: jax.Array,
    *,
    block_slices: tuple[slice, ...] | None = None,
) -> jax.Array:
    if block_slices is None or len(block_slices) <= 1:
        return X @ U

    out = jnp.zeros_like(X)
    for s in block_slices:
        out = out.at[..., :, s].set(X[..., :, s] @ U[..., s, s])
    return out


def _project_tangent(
    Omega: jax.Array,
    *,
    rot_mask: jax.Array | None,
) -> jax.Array:
    Omega = _skew_hermitian(Omega)
    if rot_mask is not None:
        Omega = Omega * rot_mask
    return Omega


def _commutator_norm(
    Ft: jax.Array, p: jax.Array, w_norm: jax.Array,
    T: float, denom_scale: float,
) -> jax.Array:
    """Commutator-only stationarity norm: ||(p_j - p_i) Ft_ij / denom||.

    Unlike `_stationarity_norm`, this omits the Jacobi-gauge term that
    measures off-diagonal Fock elements within degenerate-occupation subspaces.
    The RTR gradient is zero in those subspaces, so including the Jacobi term
    would make the solver report non-convergence for a quantity it cannot
    optimise.
    """
    real_dtype = p.dtype
    tiny_eps = jnp.asarray(1e-30, dtype=real_dtype)
    diff = p[..., None, :] - p[..., :, None]

    n = Ft.shape[-1]
    offdiag = (1.0 - jnp.eye(n, dtype=real_dtype))[None, None, ...]

    eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)).astype(real_dtype)
    gap = eps[..., :, None] - eps[..., None, :]
    eps_scale = jnp.sqrt(jnp.mean(eps ** 2) + tiny_eps)
    lam = jnp.maximum(
        jnp.asarray(T, dtype=real_dtype),
        jnp.asarray(denom_scale, dtype=real_dtype) * eps_scale,
    )
    denom = jnp.sqrt(gap ** 2 + lam ** 2)

    comm = (diff * Ft * offdiag) / denom
    per_k = jnp.sum(jnp.abs(comm) ** 2, axis=(-2, -1))
    return jnp.sqrt(jnp.sum(w_norm * per_k) / jnp.maximum(jnp.sum(w_norm), tiny_eps))


# ----------------------------
# Trust-Region Steihaug-Toint tCG
# ----------------------------

def _steihaug_tcg(G, hvp_fn, delta, max_iter, tol, w_norm):
    """
    Minimize the quadratic trust-region model

        m(s) = <G, s> + 1/2 <s, H s>

    with weighted Frobenius inner product, approximately via Steihaug CG.
    """
    s0 = jnp.zeros_like(G)
    r0 = G
    p0 = -G
    Hs0 = jnp.zeros_like(G)

    r0_norm = jnp.sqrt(jnp.maximum(0.0, _inner_product_w(r0, r0, w_norm)))
    tol_cg = r0_norm * jnp.minimum(tol, jnp.sqrt(r0_norm))

    def cond(carry):
        i, s, r, p, Hs, converged = carry
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

    def body(carry):
        i, s, r, p, Hs, converged = carry

        Hp = hvp_fn(p)
        kappa = _inner_product_w(p, Hp, w_norm)

        def on_neg_curve(_):
            a = _inner_product_w(p, p, w_norm)
            b = _inner_product_w(s, p, w_norm)
            c = _inner_product_w(s, s, w_norm) - delta**2
            tau = (-b + jnp.sqrt(jnp.maximum(0.0, b**2 - a * c))) / jnp.maximum(1e-30, a)
            s_out = s + tau * p
            Hs_out = Hs + tau * Hp
            return s_out, Hs_out, r, p, jnp.bool_(True)

        def on_pos_curve(_):
            rr = _inner_product_w(r, r, w_norm)
            alpha = rr / jnp.maximum(1e-30, kappa)
            s_new = s + alpha * p
            Hs_new = Hs + alpha * Hp

            ss_new = _inner_product_w(s_new, s_new, w_norm)

            def on_boundary(_):
                a = _inner_product_w(p, p, w_norm)
                b = _inner_product_w(s, p, w_norm)
                c = _inner_product_w(s, s, w_norm) - delta**2
                tau = (-b + jnp.sqrt(jnp.maximum(0.0, b**2 - a * c))) / jnp.maximum(1e-30, a)
                s_bnd = s + tau * p
                Hs_bnd = Hs + tau * Hp
                return s_bnd, Hs_bnd, r, p, jnp.bool_(True)

            def inside(_):
                r_new = r + alpha * Hp
                rr_new = _inner_product_w(r_new, r_new, w_norm)
                beta = rr_new / jnp.maximum(1e-30, rr)
                p_new = -r_new + beta * p
                conv = jnp.sqrt(jnp.maximum(0.0, rr_new)) <= tol_cg
                return s_new, Hs_new, r_new, p_new, conv

            return lax.cond(ss_new >= delta**2, on_boundary, inside, operand=None)

        s_next, Hs_next, r_next, p_next, conv_next = lax.cond(
            kappa <= 0.0,
            on_neg_curve,
            on_pos_curve,
            operand=None,
        )
        return (i + 1, s_next, r_next, p_next, Hs_next, conv_next)

    # Correct early exit: respect tol_cg, not a hard-coded 1e-12.
    init_state = (jnp.int32(0), s0, r0, p0, Hs0, r0_norm <= tol_cg)
    final_carry = lax.while_loop(cond, body, init_state)
    return final_carry[1], final_carry[4], final_carry[0]


# ---------------------------------------
# Main JIT-friendly outer optimizer
# ---------------------------------------

def variational_rtr_optimize(
    params0: VariationalHFParams,
    electrondensity0: float,
    *,
    h: jax.Array,
    weights_b: jax.Array,
    weight_sum: jax.Array,
    VR: jax.Array,
    T: float,
    refP: jax.Array,
    HH: jax.Array,
    include_hartree: bool,
    include_exchange: bool,
    exchange_hermitian_channel_packing: bool,
    max_iter: int = 80,
    comm_tol: float = 1e-5,
    p_tol: float = 1e-2,
    e_tol: float = 0.0,
    # RTR specific
    max_cg_iter: int = 3,
    cg_tol: float = 1e-2,
    max_rot: float = 0.60,
    p_floor: float = 0.10,
    denom_scale: float = 1e-3,
    mu_maxiter: int = 25,
    mu_tol: float = 1e-9,
    block_sizes: tuple[int, ...] | None = None,
    exchange_check_offdiag: bool | None = None,
    exchange_offdiag_atol: float = 1e-12,
    exchange_offdiag_rtol: float = 0.0,
    project_fn=None,
):
    exchange_block_specs = _exchange_block_specs_from_block_sizes(block_sizes)
    target_dtype = h.dtype
    real_dtype = jnp.zeros((), dtype=target_dtype).real.dtype

    tiny = jnp.asarray(1e-30, dtype=real_dtype)
    _project = project_fn if project_fn is not None else (lambda A: A)

    w2d = jnp.asarray(weights_b[..., 0, 0], dtype=real_dtype)
    weight_sum = jnp.asarray(weight_sum, dtype=real_dtype)
    w_norm = w2d / jnp.maximum(weight_sum, tiny)

    n_target = jnp.asarray(electrondensity0, dtype=real_dtype)
    n_target_norm = n_target / jnp.maximum(weight_sum, tiny)

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
    max_rot_r = jnp.asarray(max_rot, dtype=real_dtype)
    delta0 = jnp.asarray(0.5 * max_rot, dtype=real_dtype)

    n = Q0.shape[-1]
    rot_mask = _rotation_mask_from_block_sizes(block_sizes, n, real_dtype) if block_sizes else None
    block_slices = _block_slices_from_sizes(block_sizes, n) if block_sizes else None

    # Jacobi-fallback constants: when occupations are (nearly) degenerate,
    # the commutator gradient (p_j - p_i) * Ft_ij vanishes.  A Jacobi-angle
    # direction drives Q toward diagonalising Ft in those subspaces,
    # mirroring the QR solver's _orbital_gradient interpolation.
    p_floor_r = jnp.asarray(p_floor, dtype=real_dtype)
    tiny_jac = jnp.asarray(1e-16, dtype=real_dtype)
    tiny_gap = jnp.asarray(1e-12, dtype=real_dtype)
    orb_idx = jnp.arange(n, dtype=real_dtype)
    pair_sign = jnp.sign(orb_idx[None, :] - orb_idx[:, None])
    pair_sign = jnp.where(pair_sign == 0.0, 1.0, pair_sign)
    offdiag_mask = 1.0 - jnp.eye(n, dtype=real_dtype)

    # Pre-step: diagonalise Fock in the orbital basis to break degenerate-
    # occupation symmetry.  Only used when no symmetry projection is active;
    # with project_fn the global eigh reorders eigenvectors by eigenvalue,
    # mixing symmetry sectors (e.g. 3 identical flavors in SVP).  In that
    # case the Jacobi fallback in the inner loop handles the degeneracy.
    if project_fn is None:
        P_pre = _project(_density_from_Qp(Q0, p0))
        _Sig_pre, _H_pre, F_pre = _build_sigma_hartree(
            P_pre,
            h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            exchange_block_specs=exchange_block_specs,
            exchange_check_offdiag=exchange_check_offdiag,
            exchange_offdiag_atol=exchange_offdiag_atol,
            exchange_offdiag_rtol=exchange_offdiag_rtol,
        )
        F_pre = _project(F_pre)
        Ft_pre = _fock_in_orbital_basis(Q0, F_pre)
        if block_slices is not None:
            V_pre = jnp.zeros_like(Ft_pre)
            for s in block_slices:
                _ev, _V = jnp.linalg.eigh(Ft_pre[..., s, s])
                V_pre = V_pre.at[..., s, s].set(_V)
        else:
            _ev, V_pre = jnp.linalg.eigh(Ft_pre)
        Q0 = _apply_right_unitary(Q0, V_pre, block_slices=block_slices)

    carry0 = (
        jnp.int32(0),
        Q0,
        p0,
        mu0,
        jnp.asarray(jnp.nan, dtype=real_dtype),   # previous iterate energy
        jnp.asarray(jnp.inf, dtype=real_dtype),   # dC
        jnp.asarray(jnp.inf, dtype=real_dtype),   # dP
        jnp.asarray(jnp.inf, dtype=real_dtype),   # dE
        delta0,
        hist_E,
        hist_dC,
        hist_dP,
        hist_dE,
        hist_mu,
    )

    def cond(carry):
        k = carry[0]
        dC_prev, dP_prev, dE_prev = carry[5], carry[6], carry[7]
        not_converged = jnp.logical_or(dC_prev > comm_tol_r, dP_prev > p_tol_r)
        not_converged = jnp.where(
            e_tol_r > 0,
            jnp.logical_or(not_converged, dE_prev > e_tol_r),
            not_converged,
        )
        return jnp.logical_and(k < max_iter, not_converged)

    def body(carry):
        (
            k,
            Q,
            p,
            mu,
            E_prev_iter,
            _dC_prev,
            _dP_prev,
            _dE_prev,
            delta,
            hE,
            hC,
            hP,
            hdE,
            hM,
        ) = carry

        # ----- Current state: Fock at (Q, p) = output of previous iteration -----
        P_cur = _project(_density_from_Qp(Q, p))
        Sigma_cur, H_cur, F_cur = _build_sigma_hartree(
            P_cur,
            h=h,
            VR=VR,
            refP=refP,
            HH=HH,
            w2d=w2d,
            include_exchange=include_exchange,
            include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            exchange_block_specs=exchange_block_specs,
            exchange_check_offdiag=exchange_check_offdiag,
            exchange_offdiag_atol=exchange_offdiag_atol,
            exchange_offdiag_rtol=exchange_offdiag_rtol,
        )
        F_cur = _project(F_cur)
        E_cur = jnp.real(_hf_energy(P_cur, h=h, Sigma=Sigma_cur, H=H_cur, weights_b=weights_b))
        Ft_cur = _fock_in_orbital_basis(Q, F_cur)
        eps_cur = jnp.real(jnp.diagonal(Ft_cur, axis1=-2, axis2=-1)).astype(real_dtype)

        # Self-consistency metric including both commutator gradient and
        # Jacobi-gauge term for degenerate-occupation subspaces.
        dC = _stationarity_norm(
            Ft_cur, p, w_norm, float(p_floor), T, float(denom_scale),
        )

        # ----- Occupation update at fixed orbitals -----
        mu_work = _solve_mu_fd_newton_bracket(
            eps_cur,
            w_norm,
            n_target_norm,
            mu,
            T,
            maxiter=int(mu_maxiter),
            tol=float(mu_tol),
        )
        p_work = _occupations_from_eps(eps_cur, mu_work, T).astype(real_dtype)
        dP = _weighted_rms_vec(p_work - p, w_norm)

        # ----- Orbital RTR step at fixed updated occupations p_work -----
        # Frozen-F objective: E_frozen(Omega) = Tr(diag(p) U'FtU).
        # Gradient and HVP use closed-form commutator formulas:
        #   G_ij      = (p_i - p_j) * Ft_ij           (element-wise)
        #   (H*v)_ij  = (p_i - p_j) * [v, Ft]_ij      (matrix commutator)

        # Block-diagonal Fock for gradient/HVP (off-block elements don't
        # contribute to the frozen-F objective).
        if block_slices is not None:
            Ft_diag = jnp.zeros_like(Ft_cur)
            for sl in block_slices:
                Ft_diag = Ft_diag.at[..., sl, sl].set(Ft_cur[..., sl, sl])
        else:
            Ft_diag = Ft_cur

        # Analytical Riemannian gradient with Jacobi fallback.
        # Non-degenerate pairs use the commutator gradient G = (p_j-p_i)*Ft,
        # preconditioned by the Hessian diagonal.  Degenerate pairs (p_i≈p_j)
        # use the Jacobi rotation angle that drives Q toward diagonalising Ft.
        # Interpolated via occ_scale = |diff_p|/(|diff_p|+p_floor).
        diff_p = p_work[..., None, :] - p_work[..., :, None]
        abs_diff_p = jnp.abs(diff_p)
        occ_scale = abs_diff_p / (abs_diff_p + p_floor_r)

        # Commutator gradient norm (for outer-loop dC metric, unchanged).
        G0 = _project_tangent(diff_p * Ft_diag, rot_mask=rot_mask)
        grad_norm = jnp.sqrt(jnp.maximum(0.0, _inner_product_w(G0, G0, w_norm)))

        # Jacobi contribution norm: off-diagonal |Ft| in degenerate subspaces.
        jac_contrib = (1.0 - occ_scale) * jnp.abs(Ft_diag) * offdiag_mask
        jac_norm = jnp.sqrt(jnp.maximum(
            0.0, jnp.sum(w_norm[..., None, None] * jac_contrib ** 2),
        ))
        total_grad_norm = grad_norm + jac_norm

        def do_tr_step(_):
            # Inner preconditioned-gradient loop on the frozen-Fock
            # energy at fixed occupations p_work.  Each step interpolates
            # the preconditioned commutator direction (non-degenerate pairs)
            # with the Jacobi rotation angle (degenerate pairs).
            inner_tol = jnp.asarray(cg_tol, dtype=real_dtype) * total_grad_norm

            def inner_cond(carry):
                i, _Q_i, _Ft_i, s_norm_prev = carry
                return jnp.logical_and(i < max_cg_iter, s_norm_prev > inner_tol)

            def inner_body(carry):
                i, Q_i, Ft_i, _ = carry

                # Block-diagonal Fock for gradient computation.
                Ft_work_i = Ft_i
                if block_slices is not None:
                    Ft_work_i = jnp.zeros_like(Ft_i)
                    for sl in block_slices:
                        Ft_work_i = Ft_work_i.at[..., sl, sl].set(
                            Ft_i[..., sl, sl]
                        )

                # --- Commutator step (non-degenerate pairs) ---
                G_comm_i = diff_p * Ft_work_i
                eps_i = jnp.real(
                    jnp.diagonal(Ft_i, axis1=-2, axis2=-1)
                )
                gap_i = eps_i[..., :, None] - eps_i[..., None, :]
                T_reg = jnp.asarray(T, dtype=real_dtype)
                d_i = jnp.maximum(
                    jnp.abs(diff_p * gap_i), T_reg * T_reg
                )
                s_comm_i = _project_tangent(
                    -G_comm_i / d_i, rot_mask=rot_mask
                )

                # --- Jacobi step (degenerate pairs) ---
                gap_ji = -gap_i
                abs_Ft_i = jnp.abs(Ft_work_i)
                safe_gap_ji = jnp.where(
                    jnp.abs(gap_ji) < tiny_gap,
                    tiny_gap * pair_sign,
                    gap_ji,
                )
                theta_i = 0.5 * jnp.arctan(
                    2.0 * abs_Ft_i / safe_gap_ji
                )
                phase_i = jnp.where(
                    abs_Ft_i > tiny_jac,
                    Ft_work_i / abs_Ft_i,
                    jnp.zeros_like(Ft_work_i),
                )
                s_jac_i = _project_tangent(
                    theta_i * phase_i, rot_mask=rot_mask
                )

                # --- Interpolated step ---
                s_i = occ_scale * s_comm_i + (1.0 - occ_scale) * s_jac_i

                # Unclamped step norm (for convergence check).
                s_norm_i = jnp.sqrt(
                    jnp.maximum(0.0, _inner_product_w(s_i, s_i, w_norm))
                )

                # Clamp step norm to max_rot.
                alpha = jnp.minimum(
                    1.0,
                    max_rot_r / jnp.maximum(1e-30, s_norm_i),
                )
                s_i = alpha * s_i
                U_i = _qr_retraction_unitary(
                    s_i, -1.0, block_slices=block_slices
                )
                Q_new = _apply_right_unitary(
                    Q_i, U_i, block_slices=block_slices
                )

                # Rotate Fock matrix to new orbital basis.
                if block_slices is not None:
                    Ft_new = jnp.zeros_like(Ft_i)
                    for sl in block_slices:
                        Ub = U_i[..., sl, sl]
                        Ft_new = Ft_new.at[..., sl, sl].set(
                            jnp.conj(jnp.swapaxes(Ub, -2, -1))
                            @ Ft_i[..., sl, sl]
                            @ Ub
                        )
                else:
                    Ft_new = (
                        jnp.conj(jnp.swapaxes(U_i, -2, -1))
                        @ Ft_i
                        @ U_i
                    )

                return (i + 1, Q_new, Ft_new, s_norm_i)

            init_inner = (jnp.int32(0), Q, Ft_cur, total_grad_norm)
            _, Q_next, _, _ = lax.while_loop(
                inner_cond, inner_body, init_inner
            )
            return Q_next, delta

        def no_tr_step(_):
            return Q, delta

        # Occupations are always updated; the orbital step is only needed
        # when there is a non-negligible commutator gradient OR off-diagonal
        # Fock elements in degenerate-occupation subspaces (Jacobi).
        need_tr = total_grad_norm > 1e-14
        Q_next, delta_next = lax.cond(need_tr, do_tr_step, no_tr_step, operand=None)

        dE = jnp.where(jnp.isfinite(E_prev_iter), jnp.abs(E_cur - E_prev_iter), jnp.inf)

        hE = hE.at[k].set(E_cur)
        hC = hC.at[k].set(dC)
        hP = hP.at[k].set(dP)
        hdE = hdE.at[k].set(dE)
        hM = hM.at[k].set(mu_work)

        return (
            k + 1,
            Q_next,
            p_work,
            mu_work,
            E_cur,
            dC,
            dP,
            dE,
            delta_next,
            hE,
            hC,
            hP,
            hdE,
            hM,
        )

    (
        k_fin,
        Q_fin,
        p_fin,
        mu_fin,
        _E_prev_fin,
        _dC_fin,
        _dP_fin,
        _dE_fin,
        _delta_fin,
        hist_E,
        hist_dC,
        hist_dP,
        hist_dE,
        hist_mu,
    ) = lax.while_loop(cond, body, carry0)

    P_fin = _project(_density_from_Qp(Q_fin, p_fin))
    Sigma_fin, H_fin, F_fin = _build_sigma_hartree(
        P_fin,
        h=h,
        VR=VR,
        refP=refP,
        HH=HH,
        w2d=w2d,
        include_exchange=include_exchange,
        include_hartree=include_hartree,
        exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
        exchange_block_specs=exchange_block_specs,
        exchange_check_offdiag=exchange_check_offdiag,
        exchange_offdiag_atol=exchange_offdiag_atol,
        exchange_offdiag_rtol=exchange_offdiag_rtol,
    )
    F_fin = _project(F_fin)
    E_fin = jnp.real(_hf_energy(P_fin, h=h, Sigma=Sigma_fin, H=H_fin, weights_b=weights_b))

    history = dict(E=hist_E, dC=hist_dC, dP=hist_dP, dE=hist_dE, mu=hist_mu)
    params_fin = VariationalHFParams(Q=Q_fin, p=p_fin, mu=mu_fin)
    return P_fin, F_fin, E_fin, mu_fin, k_fin, history, params_fin


def jit_variational_rtr_iteration(hf_step):
    compiled = jax.jit(
        variational_rtr_optimize,
        static_argnames=(
            "max_iter",
            "comm_tol",
            "p_tol",
            "e_tol",
            "max_cg_iter",
            "cg_tol",
            "max_rot",
            "p_floor",
            "denom_scale",
            "mu_maxiter",
            "mu_tol",
            "block_sizes",
            "include_hartree",
            "include_exchange",
            "exchange_hermitian_channel_packing",
            "exchange_check_offdiag",
            "exchange_offdiag_atol",
            "exchange_offdiag_rtol",
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
        validate_electron_count(
            hf_step.w2d,
            hf_step.h.shape[-1],
            electrondensity0,
            context="electrondensity0",
        )
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

        legacy_block_kwargs = tuple(
            key
            for key in ("rotation_block_sizes", "exchange_block_specs")
            if key in kwargs
        )
        if legacy_block_kwargs:
            raise TypeError(f"Remove legacy kwargs {legacy_block_kwargs!r}.")
        if kwargs.get("block_sizes") is not None:
            kwargs["block_sizes"] = tuple(int(size) for size in kwargs["block_sizes"])

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
