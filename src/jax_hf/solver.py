"""Direct minimization solver: preconditioned Riemannian CG on Stiefel x Simplex.

Minimizes the Hartree-Fock free energy Omega(Q, p) = E[P] - T*S(p) over
unitary orbitals Q (Stiefel manifold) and occupations p (capped simplex),
with one Fock build per iteration.

Algorithm:
  1. Build F[P] (one exchange FFT — the dominant cost)
  2. Analytical Riemannian gradient from Fock matrix (no autodiff)
  3. Energy-gap preconditioning (diagonal Hessian)
  4. Polak-Ribiere CG direction with periodic restart
  5. Backtracking line search on frozen-F free energy
  6. QR retraction for orbitals, simplex projection for occupations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .fock import build_fock, hf_energy, free_energy
from .utils import validate_electron_count


# ---------------------------------------------------------------------------
# Small linear algebra helpers
# ---------------------------------------------------------------------------

def _herm(X: jax.Array) -> jax.Array:
    return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))


def _skew_hermitian(A: jax.Array) -> jax.Array:
    return 0.5 * (A - jnp.conj(jnp.swapaxes(A, -1, -2)))


def _density_from_Qp(Q: jax.Array, p: jax.Array) -> jax.Array:
    """P = Q diag(p) Q†."""
    return jnp.einsum("...in,...n,...jn->...ij", Q, p, jnp.conj(Q))


def _fock_in_orbital_basis(Q: jax.Array, F: jax.Array) -> jax.Array:
    """Ft = Q† F Q."""
    return jnp.einsum("...in,...ij,...jm->...nm", jnp.conj(Q), F, Q)


def _logit(p: jax.Array) -> jax.Array:
    # Use 1e-6 bound (not 1e-14) so float32 can represent 1-eps != 1.
    p_safe = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p_safe) - jnp.log1p(-p_safe)


# ---------------------------------------------------------------------------
# Chemical potential solver (Newton-bracket)
# ---------------------------------------------------------------------------

def _solve_mu(
    eps: jax.Array,
    w_norm: jax.Array,
    n_target_norm: jax.Array,
    mu0: jax.Array,
    T: jax.Array,
    maxiter: int = 25,
) -> jax.Array:
    """Find mu so sum_k w_k sum_i sigmoid((mu - eps)/T) = n_target_norm."""
    real_dtype = eps.dtype
    Tj = jnp.maximum(jnp.asarray(T, dtype=real_dtype), jnp.asarray(1e-12, dtype=real_dtype))

    e_min, e_max = jnp.min(eps), jnp.max(eps)
    lo = e_min - 50.0 * Tj
    hi = e_max + 50.0 * Tj
    mu = jnp.clip(jnp.asarray(mu0, dtype=real_dtype), lo, hi)
    w_b = w_norm[..., None]

    def body(state, _):
        mu, lo, hi = state
        x = (mu - eps) / Tj
        p = jax.nn.sigmoid(x)
        N = jnp.sum(w_b * p)
        Z = jnp.sum(w_b * p * (1.0 - p) / Tj)
        g = N - n_target_norm

        lo = jnp.where(g < 0, mu, lo)
        hi = jnp.where(g > 0, mu, hi)

        Z_safe = jnp.maximum(Z, jnp.asarray(1e-18, dtype=real_dtype))
        mu_new = mu - g / Z_safe
        mu_bis = 0.5 * (lo + hi)
        out_of = jnp.logical_or(mu_new <= lo, mu_new >= hi)
        mu_new = jnp.where(out_of, mu_bis, mu_new)
        mu_new = jnp.clip(mu_new, lo, hi)
        mu_new = jnp.where(jnp.isfinite(mu_new), mu_new, mu_bis)
        return (mu_new, lo, hi), None

    (mu_fin, _, _), _ = lax.scan(body, (mu, lo, hi), xs=None, length=maxiter)
    return mu_fin


# ---------------------------------------------------------------------------
# QR retraction
# ---------------------------------------------------------------------------

def _qr_retract(G: jax.Array, tau: jax.Array) -> jax.Array:
    """U = QR(I - tau*G) with phase normalization."""
    tiny = jnp.asarray(1e-30, dtype=jnp.real(G).dtype)
    tau_bc = jnp.asarray(tau, dtype=jnp.real(G).dtype)[..., None, None]
    n = G.shape[-1]
    U_trial = jnp.eye(n, dtype=G.dtype) - tau_bc * G
    U, R = jnp.linalg.qr(U_trial)
    phases = jnp.diagonal(R, axis1=-2, axis2=-1)
    phase_norm = jnp.where(
        jnp.abs(phases) > tiny, phases / jnp.abs(phases), jnp.ones_like(phases),
    )
    return U * phase_norm[..., None, :]


def _cayley_retract(d: jax.Array, tau: jax.Array) -> jax.Array:
    """Cayley retraction: U = (I - tau*d/2) (I + tau*d/2)^-1.

    For skew-Hermitian d, (I+A) and (I-A) with A = tau*d/2 commute, so
    U = solve(I + A, I - A) = (I + A)^-1 (I - A) is unitary.

    Agrees with exp(-tau*d) to first order in tau.  Cheaper than QR
    retraction (one LU solve vs one QR, ~2x fewer flops on nb*nb blocks).

    Valid as long as ||tau*d|| < 2 (eigenvalues of (I + tau*d/2) stay away
    from zero).  With max_step <= 1 and our typical tau <= 1, this is safe.
    """
    tau_bc = jnp.asarray(tau, dtype=jnp.real(d).dtype)[..., None, None]
    n = d.shape[-1]
    eye = jnp.eye(n, dtype=d.dtype)
    A = 0.5 * tau_bc * d
    return jnp.linalg.solve(eye + A, eye - A)


# ---------------------------------------------------------------------------
# Spectral Cayley: amortise eigh of (i*d) across line search backtracking.
#
# For skew-Hermitian d, i*d is Hermitian.  Diagonalise once:
#     i*d = V diag(lam) V†      (lam real, V unitary)
# then for any tau:
#     U(tau) = (I - tau*d/2)(I + tau*d/2)^-1
#            = V · diag((1 + i*tau*lam/2)/(1 - i*tau*lam/2)) · V†
# The factor c(tau, lam) = (1 + iτλ/2)/(1 − iτλ/2) has |c|=1 ⇒ U is exactly
# unitary independently of how many times we evaluate it.
#
# In the line search we only need diag(U(tau)† Ft U(tau)) (the trial orbital
# energies).  Pre-rotating Ft once into the eigenbasis of i*d gives
#     Ft_eig = V† Ft V
# and per-tau the diagonal is computed with one Hadamard scaling and one
# matmul, with no LU solve and no full Ft_trial:
#     M(tau) = c̄(tau) Ft_eig c(tau)              (Hadamard, O(nb²))
#     diag(U†FtU) = (V @ M(tau) ⊙ conj(V)).sum(-1)  (one matmul, O(nb³))
# ---------------------------------------------------------------------------

def _cayley_spectral_setup(d: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Eigendecompose i*d for a skew-Hermitian d.  Returns (V, lam)."""
    iA = (1j * d).astype(d.dtype)
    iA = 0.5 * (iA + jnp.conj(jnp.swapaxes(iA, -1, -2)))  # exact Hermitisation
    lam, V = jnp.linalg.eigh(iA)
    return V, lam.astype(jnp.real(d).dtype)


def _cayley_factor(lam: jax.Array, tau: jax.Array, complex_dtype) -> jax.Array:
    """c(τ, λ) = (1 + iτλ/2) / (1 - iτλ/2) — |c|=1 by construction."""
    half = jnp.asarray(0.5, dtype=lam.dtype)
    arg = jnp.asarray(tau, dtype=lam.dtype) * lam * half
    iexp = (1j * arg).astype(complex_dtype)
    one = jnp.asarray(1.0, dtype=complex_dtype)
    return (one + iexp) / (one - iexp)


def _cayley_unitary_from_spectrum(V: jax.Array, lam: jax.Array,
                                  tau: jax.Array) -> jax.Array:
    """U(τ) = V · diag(c(τ, λ)) · V†."""
    c = _cayley_factor(lam, tau, V.dtype)
    return (V * c[..., None, :]) @ jnp.conj(jnp.swapaxes(V, -1, -2))


def _diag_UFU_from_spectrum(V: jax.Array, Ft_eig: jax.Array,
                            lam: jax.Array, tau: jax.Array) -> jax.Array:
    """Compute diag(U(τ)† Ft U(τ)) using precomputed Ft_eig = V† Ft V.

    Cost: one Hadamard (nb²) plus one matmul (nb³) — versus the LU+two-matmul
    path in :func:`_cayley_retract` followed by a full U†FtU triple product.
    """
    c = _cayley_factor(lam, tau, V.dtype)
    cbar = jnp.conj(c)
    M = (cbar[..., :, None]) * Ft_eig * (c[..., None, :])
    A = V @ M
    diag = jnp.real(jnp.sum(A * jnp.conj(V), axis=-1))
    return diag



# ---------------------------------------------------------------------------
# Weighted inner products
# ---------------------------------------------------------------------------

def _ip_matrix(X: jax.Array, Y: jax.Array, w_norm: jax.Array) -> jax.Array:
    """Weighted Frobenius inner product: sum_k w_k Re(Tr(X†Y))."""
    per_k = jnp.sum(jnp.real(jnp.conj(X) * Y), axis=(-2, -1))
    return jnp.sum(w_norm * per_k)


def _ip_vec(x: jax.Array, y: jax.Array, w_norm: jax.Array) -> jax.Array:
    """Weighted vector inner product: sum_k w_k sum_i x_i y_i."""
    return jnp.sum(w_norm[..., None] * x * y)


def _norm_matrix(X: jax.Array, w_norm: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.maximum(0.0, _ip_matrix(X, X, w_norm)))


def _norm_vec(x: jax.Array, w_norm: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.maximum(0.0, _ip_vec(x, x, w_norm)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverConfig:
    """Solver controls. Intentionally minimal."""

    max_iter: int = 200
    tol_E: float = 1e-7         # convergence on energy change per step (primary)
    tol_grad: float = 0.0       # convergence on gradient norm (0 = disabled)
    denom_scale: float = 1e-3   # regularization for energy-gap preconditioner
    max_step: float = 0.6       # max orbital rotation norm per step
    cg_restart: int = 10        # restart CG every N steps
    bt_shrink: float = 0.5      # backtracking shrink factor
    bt_max: int = 8             # max backtracking steps
    mu_maxiter: int = 25        # chemical potential solver iterations
    block_sizes: tuple[int, ...] | None = None
    project_fn: Any = None


class SolveResult(NamedTuple):
    Q: jax.Array             # converged orbitals (nk1, nk2, nb, nb)
    p: jax.Array             # converged occupations (nk1, nk2, nb)
    mu: jax.Array            # chemical potential
    density: jax.Array       # P = Q diag(p) Q†
    fock: jax.Array          # F at convergence
    energy: jax.Array        # HF energy (scalar)
    n_iter: jax.Array        # iteration count
    converged: jax.Array     # bool
    history: dict[str, jax.Array]


def _exchange_block_specs(block_sizes):
    if block_sizes is None:
        return None
    return (("sizes", tuple(int(s) for s in block_sizes)),)


def _solve_impl(
    P0: jax.Array,
    n_electrons: float,
    tol_E: float,
    tol_grad: float,
    max_step: float,
    bt_shrink: float,
    denom_scale: float,
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
    contact_g: jax.Array,
    contact_Oi: jax.Array,
    contact_Oj: jax.Array,
    max_iter: int,
    bt_max: int,
    cg_restart: int,
    mu_maxiter: int,
    block_sizes: tuple | None,
    project_fn,
) -> SolveResult:
    """Direct minimization via preconditioned Riemannian CG.

    One Fock build per iteration.  Jointly optimizes orbitals (Q on Stiefel
    manifold via Cayley retraction) and occupations (p on capped simplex).

    Implementation detail — users should call :func:`solve_direct_minimization`
    (or the :func:`solve` alias) instead.  JIT caching is handled by that wrapper.

    Only structurally-static arguments (``max_iter``, ``bt_max``,
    ``cg_restart``, ``mu_maxiter``, ``block_sizes``, ``project_fn``, plus the
    Hartree/exchange/hcp flags from the kernel) are marked as static in the
    JIT decorator.  Float tolerances and step sizes stay dynamic so that
    tuning them does not trigger recompilation.
    """
    target_dtype = h.dtype
    real_dtype = jnp.zeros((), dtype=target_dtype).real.dtype
    tiny = jnp.asarray(1e-30, dtype=real_dtype)

    _project = project_fn if project_fn is not None else (lambda A: A)
    block_specs = _exchange_block_specs(block_sizes)

    w2d = jnp.asarray(weights_b[..., 0, 0], dtype=real_dtype)
    weight_sum_r = jnp.asarray(weight_sum, dtype=real_dtype)
    w_norm = w2d / jnp.maximum(weight_sum_r, tiny)

    n_target = jnp.asarray(n_electrons, dtype=real_dtype)
    n_target_norm = n_target / jnp.maximum(weight_sum_r, tiny)

    T_r = jnp.maximum(jnp.asarray(T, dtype=real_dtype), jnp.asarray(1e-12, dtype=real_dtype))
    tol_E_r = jnp.asarray(tol_E, dtype=real_dtype)
    tol_grad_r = jnp.asarray(tol_grad, dtype=real_dtype)
    max_step_r = jnp.asarray(max_step, dtype=real_dtype)
    bt_shrink_r = jnp.asarray(bt_shrink, dtype=real_dtype)
    denom_scale_r = jnp.asarray(denom_scale, dtype=real_dtype)

    nb = h.shape[-1]
    offdiag = (1.0 - jnp.eye(nb, dtype=real_dtype))[None, None, ...]

    # ---- Initialize (Q, p) from Fock at P0 ----
    # Build initial Fock matrix and diagonalize to get a good starting point.
    # This handles P0=0 gracefully (gives non-interacting ground state).
    P0_h = _herm(jnp.asarray(P0, dtype=target_dtype))
    Sigma0, H0, F0 = build_fock(
        P0_h, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
        include_exchange=include_exchange, include_hartree=include_hartree,
        exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
        contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
        exchange_block_specs=block_specs,
        project_fn=project_fn,
    )
    eps0, Q0 = jnp.linalg.eigh(F0)
    eps0 = eps0.astype(real_dtype)
    Q0 = Q0.astype(target_dtype)
    mu0 = _solve_mu(eps0, w_norm, n_target_norm,
                    jnp.asarray(0.0, dtype=real_dtype), T_r, maxiter=mu_maxiter)
    p0 = jax.nn.sigmoid((mu0 - eps0) / T_r).astype(real_dtype)

    # ---- History buffers ----
    hist_E = jnp.zeros(max_iter, dtype=real_dtype)
    hist_grad = jnp.zeros(max_iter, dtype=real_dtype)

    # ---- Carry for lax.while_loop ----
    # (iter, Q, p, mu, d_Q, d_p, G_Q_prev, g_p_prev, grad_norm, E_prev, dE,
    #  hist_E, hist_grad)
    zeros_Q = jnp.zeros_like(Q0)
    zeros_p = jnp.zeros_like(p0)

    carry0 = (
        jnp.int32(0),       # k
        Q0,                  # Q
        p0,                  # p
        mu0,                 # mu
        zeros_Q,             # d_Q (CG search direction, orbital)
        zeros_p,             # d_p (CG search direction, occupation)
        zeros_Q,             # G_Q_prev (previous orbital gradient)
        zeros_p,             # g_p_prev (previous occupation gradient)
        jnp.asarray(jnp.inf, dtype=real_dtype),  # grad_norm
        jnp.asarray(jnp.inf, dtype=real_dtype),  # E_prev (last recorded energy)
        jnp.asarray(jnp.inf, dtype=real_dtype),  # dE = |E - E_prev|
        hist_E,
        hist_grad,
    )

    def cond(carry):
        # Converge when energy change is below tol_E.  If tol_grad > 0, also
        # require gradient norm below tol_grad before declaring convergence.
        k, _, _, _, _, _, _, _, grad_norm, _, dE, _, _ = carry
        energy_not_converged = dE > tol_E_r
        grad_check_active = tol_grad_r > 0.0
        grad_not_converged = jnp.logical_and(grad_check_active, grad_norm > tol_grad_r)
        return jnp.logical_and(
            k < max_iter,
            jnp.logical_or(energy_not_converged, grad_not_converged),
        )

    def body(carry):
        (k, Q, p, mu, d_Q_prev, d_p_prev,
         G_Q_prev, g_p_prev, _grad_norm_prev, E_prev, _dE_prev, hE, hG) = carry

        # ========== 1. Build Fock matrix (THE expensive step) ==========
        # P_cur is already projected to the symmetric subspace above.  When
        # h, V_q, and the contact terms are themselves symmetric (the only
        # use case where project_fn makes physical sense), F[symmetric P]
        # is automatically symmetric to FFT roundoff (verified ~5e-16 on
        # the bilayer regression).  Skip the redundant inner projection of
        # F to save one symmetry-group sweep per outer iter.
        P_cur = _herm(_project(_density_from_Qp(Q, p)))
        Sigma, H_h, F = build_fock(
            P_cur, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
            exchange_block_specs=block_specs,
            project_fn=None,
        )
        E = hf_energy(P_cur, h=h, Sigma=Sigma, H=H_h, weights_b=weights_b)
        Ft = _fock_in_orbital_basis(Q, F)
        eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)).astype(real_dtype)

        # ========== 2. Gradient ==========
        # Orbital gradient: G_ij = skew_herm((p_j - p_i) * Ft_ij)
        diff_p = p[..., None, :] - p[..., :, None]
        G_Q = _skew_hermitian(diff_p * Ft) * offdiag

        # Occupation gradient: g_i = eps_i + T*logit(p_i) - mu
        # Use the carry's mu (from last iter's post-retraction solve) as a
        # warm guess.  g_p only enters the line-search direction d_p; the
        # final p_new is set via a fresh _solve_mu after retraction, so a
        # small staleness in mu only perturbs the τ choice.  Skipping this
        # mu_solve cuts ~14% of body work per iter.
        g_p = (eps + T_r * _logit(p) - mu).astype(real_dtype)

        # Convergence uses only the orbital gradient (commutator [F, P]).
        # The occupation gradient is zero by construction since p is always
        # set to the FD optimum at the end of each step.
        grad_norm = _norm_matrix(G_Q, w_norm)

        # ========== 3. Precondition ==========
        # Orbital: scale by inverse energy gaps
        gap = eps[..., :, None] - eps[..., None, :]
        eps_scale = jnp.sqrt(jnp.mean(eps ** 2) + tiny)
        lam = jnp.maximum(T_r, denom_scale_r * eps_scale)
        denom = jnp.sqrt(gap ** 2 + lam ** 2)
        H_Q = G_Q / denom

        # Occupation: scale by inverse FD curvature  (d²S/dp² = 1/(p(1-p)T))
        p_safe = jnp.clip(p, 1e-8, 1.0 - 1e-8)
        fd_curv = p_safe * (1.0 - p_safe) / T_r
        h_p = g_p * fd_curv

        # ========== 4. CG direction (Polak-Ribiere) ==========
        # On first iteration (k=0), just use steepest descent
        # For CG: transport previous gradient/direction to current tangent space
        # Since Q changed by Q_new = Q_old @ U, the transport is U† * prev * U.
        # We approximate transport as identity (reset via periodic restart).

        # Preconditioned Polak-Ribiere+: beta = <g, M^-1 g - M^-1 g_prev> / <g_prev, g_prev>
        # Here H_Q = M^-1 G_Q is the preconditioned orbital gradient, h_p = M^-1 g_p likewise.
        # The denominator uses the unpreconditioned squared norm of the previous
        # gradient — always well-scaled and avoids the division-then-multiplication
        # round-trip an earlier draft had.
        pr_num = (_ip_matrix(G_Q, H_Q, w_norm) - _ip_matrix(G_Q_prev, H_Q, w_norm)
                  + _ip_vec(g_p, h_p, w_norm) - _ip_vec(g_p_prev, h_p, w_norm))
        pr_den = _ip_matrix(G_Q_prev, G_Q_prev, w_norm) + _ip_vec(g_p_prev, g_p_prev, w_norm)
        beta = jnp.where(pr_den > tiny, jnp.maximum(0.0, pr_num / pr_den), 0.0)
        beta = jnp.minimum(beta, 5.0)  # cap to prevent runaway
        # Reset to steepest descent on first step and every cg_restart steps
        beta = jnp.where(
            jnp.logical_or(k == 0, k % cg_restart == 0),
            0.0, beta,
        )

        # Search direction: positive preconditioned gradient.
        # The QR retraction QR(I - tau*d) steps in the -d direction,
        # so d = +H_Q gives descent along -H_Q.
        d_Q = H_Q + beta * d_Q_prev
        d_p = h_p + beta * d_p_prev

        # ========== 5. Line search (backtracking on frozen-F free energy) ==========
        # Frozen-F free energy: Omega(tau) = Tr[diag(p_trial) U†FtU] - T*S(p_trial)
        #
        # Spectral Cayley path: eigendecompose (i * d_Q) once and pre-rotate Ft
        # into the d_Q eigenbasis.  Each line-search trial then costs one
        # Hadamard scaling + one matmul (versus an LU solve + two matmuls in
        # the LU-based Cayley).  The accepted U(tau_final) is reconstructed
        # spectrally afterwards, and eps_new is read from the same spectral
        # form so we never materialise U†FtU explicitly.
        V_d, lam_d = _cayley_spectral_setup(d_Q)
        Ft_eig = jnp.conj(jnp.swapaxes(V_d, -2, -1)) @ Ft @ V_d

        d_Q_norm = _norm_matrix(d_Q, w_norm)
        tau0 = jnp.minimum(1.0, max_step_r / jnp.maximum(d_Q_norm, tiny))

        def frozen_F_free_energy(tau_val):
            eps_trial = _diag_UFU_from_spectrum(V_d, Ft_eig, lam_d, tau_val)
            p_trial = jnp.clip(p - tau_val * d_p, 1e-8, 1.0 - 1e-8)
            E_frozen = jnp.sum(w_norm[..., None] * p_trial * eps_trial)
            return free_energy(E_frozen, p_trial, w_norm, T_r)

        # Inlined frozen_F_free_energy(tau=0): eps_trial = eps and
        # p_trial = clip(p), avoiding the spectral diag computation entirely
        # at the trivial trial point.  Mathematically identical to the
        # original frozen_F_free_energy(0.0) call.
        p0_clip = jnp.clip(p, 1e-8, 1.0 - 1e-8)
        E_frozen0 = jnp.sum(w_norm[..., None] * p0_clip * eps)
        Omega0 = free_energy(E_frozen0, p0_clip, w_norm, T_r)

        def bt_cond(state):
            i, tau, accepted = state
            return jnp.logical_and(i < bt_max, jnp.logical_not(accepted))

        def bt_body(state):
            i, tau, _ = state
            Omega_trial = frozen_F_free_energy(tau)
            accepted = Omega_trial < Omega0
            tau_next = jnp.where(accepted, tau, tau * bt_shrink_r)
            return (i + 1, tau_next, accepted)

        _, tau_final, bt_accepted = lax.while_loop(
            bt_cond, bt_body,
            (jnp.int32(0), tau0, jnp.bool_(False)),
        )
        # If no step accepted, take a tiny step
        tau_final = jnp.where(bt_accepted, tau_final, tau0 * bt_shrink_r ** bt_max)

        # ========== 6. Retraction (eigen-free inner loop) ==========
        # Use diag(Q_new† F Q_new) as approximate eigenvalues for FD
        # re-occupation.  Exact within p-degenerate blocks; slightly
        # approximate for near-degenerate states but the CG direction
        # already kills pure-gauge components via the diff_p factor.
        U = _cayley_unitary_from_spectrum(V_d, lam_d, tau_final)
        Q_new = Q @ U
        # eps_new = diag(Q_new† F Q_new) = diag(U† Ft U) — read directly from
        # the cached spectral form without an extra matmul triple-product.
        eps_new = _diag_UFU_from_spectrum(V_d, Ft_eig, lam_d, tau_final).astype(real_dtype)
        mu_new = _solve_mu(eps_new, w_norm, n_target_norm, mu, T_r, maxiter=mu_maxiter)
        p_new = jax.nn.sigmoid((mu_new - eps_new) / T_r).astype(real_dtype)

        # ========== 7. Record history + energy change ==========
        hE = hE.at[k].set(E)
        hG = hG.at[k].set(grad_norm)
        dE = jnp.abs(E - E_prev)

        return (k + 1, Q_new, p_new, mu_new, d_Q, d_p, G_Q, g_p, grad_norm,
                E, dE, hE, hG)

    # ---- Run ----
    (k_fin, Q_fin, p_fin, mu_fin, _, _, _, _, grad_fin, E_fin_loop, dE_fin,
     hist_E, hist_grad) = lax.while_loop(cond, body, carry0)

    # ---- Finalize: one eigh to produce clean Fock eigenvectors ----
    # Build Fock at current (Q, p), then rotate Q so the orbitals diagonalize
    # the Fock matrix.  This gives the canonical output expected by consumers
    # (eigenvalues in eps_final, orbitals as rows of Q).  P is gauge-invariant
    # within degenerate p-blocks so this shouldn't change the energy much;
    # we rebuild the Fock one more time with the clean density for a consistent
    # final result.
    P_pre = _herm(_project(_density_from_Qp(Q_fin, p_fin)))
    _, _, F_pre = build_fock(
        P_pre, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
        include_exchange=include_exchange, include_hartree=include_hartree,
        exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
        contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
        exchange_block_specs=block_specs,
        project_fn=None,
    )
    Ft_fin = _fock_in_orbital_basis(Q_fin, F_pre)
    eps_fin, V_fin = jnp.linalg.eigh(Ft_fin)
    eps_fin = eps_fin.astype(real_dtype)
    Q_fin = Q_fin @ V_fin.astype(target_dtype)
    mu_fin = _solve_mu(eps_fin, w_norm, n_target_norm, mu_fin, T_r, maxiter=mu_maxiter)
    p_fin = jax.nn.sigmoid((mu_fin - eps_fin) / T_r).astype(real_dtype)

    P_fin = _herm(_project(_density_from_Qp(Q_fin, p_fin)))
    Sigma_fin, H_fin, F_fin = build_fock(
        P_fin, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
        include_exchange=include_exchange, include_hartree=include_hartree,
        exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
        contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
        exchange_block_specs=block_specs,
        project_fn=None,
    )
    E_fin = hf_energy(P_fin, h=h, Sigma=Sigma_fin, H=H_fin, weights_b=weights_b)

    # Convergence flag: loop exited because of tolerance (not max_iter).
    converged = k_fin < max_iter

    return SolveResult(
        Q=Q_fin,
        p=p_fin,
        mu=mu_fin,
        density=P_fin,
        fock=F_fin,
        energy=E_fin,
        n_iter=k_fin,
        converged=converged,
        history=dict(E=hist_E, grad_norm=hist_grad),
    )


_solve_jitted = jax.jit(
    _solve_impl,
    static_argnames=(
        # Kernel-side static flags (affect Fock build dispatch)
        "include_hartree",
        "include_exchange",
        "exchange_hermitian_channel_packing",
        # SolverConfig fields that are structurally static (integers used as
        # loop bounds, callables, or tuples affecting graph shape)
        "max_iter",
        "bt_max",
        "cg_restart",
        "mu_maxiter",
        "block_sizes",
        "project_fn",
    ),
)


def solve_direct_minimization(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: SolverConfig | None = None,
) -> SolveResult:
    """Direct-minimization Hartree-Fock solver.

    Preconditioned Riemannian CG on Stiefel x capped simplex, with one
    Fock build per iteration.  See :class:`SolverConfig` for tuning knobs.

    Parameters
    ----------
    kernel
        :class:`HartreeFockKernel` holding problem arrays (pre-computed FFT
        of the interaction kernel, Hamiltonian, reference density, etc).
    P0
        Initial density matrix, shape ``(nk1, nk2, nb, nb)``.  If the seed
        is arbitrary (e.g. zeros), the solver auto-diagonalises ``F[P0]``
        internally before starting the CG loop.
    n_electrons
        Target electron count (sum of weights × occupations).
    config
        Solver controls.  Defaults to :class:`SolverConfig()`.

    Returns
    -------
    SolveResult
        Named tuple with ``Q, p, mu, density, fock, energy, n_iter,
        converged, history``.  ``density`` = ``Q @ diag(p) @ Q†`` is the
        canonical self-consistent density matrix.

    Notes
    -----
    The inner loop is eigen-free (Cayley retraction + ``diag(Ft)`` for FD
    re-occupation).  One eigendecomposition at convergence produces
    canonical Fock eigenvectors in the returned ``Q``.

    JIT cache strategy: only *structurally static* config fields trigger
    recompilation — loop-bound integers (``max_iter``, ``bt_max``,
    ``cg_restart``, ``mu_maxiter``), the optional ``block_sizes`` tuple,
    and ``project_fn`` (callable identity).  Float tolerances and step
    sizes (``tol_E``, ``tol_grad``, ``max_step``, ``bt_shrink``,
    ``denom_scale``) stay dynamic and can be tuned between calls without
    triggering a recompile.
    """
    if config is None:
        config = SolverConfig()
    validate_electron_count(
        kernel.w2d,
        kernel.h.shape[-1],
        n_electrons,
        context="n_electrons",
    )
    return _solve_jitted(
        jnp.asarray(P0),
        float(n_electrons),
        float(config.tol_E),
        float(config.tol_grad),
        float(config.max_step),
        float(config.bt_shrink),
        float(config.denom_scale),
        **kernel.as_args(),
        max_iter=int(config.max_iter),
        bt_max=int(config.bt_max),
        cg_restart=int(config.cg_restart),
        mu_maxiter=int(config.mu_maxiter),
        block_sizes=config.block_sizes,
        project_fn=config.project_fn,
    )


# Public alias: direct minimization is the default solver.
solve = solve_direct_minimization
