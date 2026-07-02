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

import warnings
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


def _project_occ(dp: jax.Array, w2d: jax.Array) -> jax.Array:
    """Project an occupation variation onto the particle-conserving subspace.

    Removes the component that changes the total electron count, i.e. enforces
    ``sum_k w_k sum_b dp[k, b] = 0`` by subtracting a uniform shift.
    """
    nb = dp.shape[-1]
    num = jnp.sum(w2d[..., None] * dp)
    den = jnp.sum(w2d) * nb
    return dp - num / den


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverConfig:
    """Solver controls. Intentionally minimal."""

    max_iter: int = 200
    tol_E: float = 1e-7         # energy criterion (governs only when tol_grad == 0)
    # Gradient stopping criterion.  When > 0 it is the SOLE and SUFFICIENT
    # stop for both optimizers: the run ends (converged=True) as soon as the
    # w-normalised Frobenius norm of the orbital gradient — the quantity
    # logged in history["grad_norm"] — drops to tol_grad, and tol_E is
    # inactive.  0 (default) stops on the windowed-energy criterion instead
    # (CG path; the Newton path then falls back to an internal 1e-6).
    tol_grad: float = 0.0
    denom_scale: float = 1e-3   # regularization for energy-gap preconditioner
    max_step: float = 0.6       # max orbital rotation norm per step
    cg_restart: int = 10        # restart CG every N steps
    bt_shrink: float = 0.5      # backtracking shrink factor
    bt_max: int = 8             # max backtracking steps
    mu_maxiter: int = 25        # chemical potential solver iterations
    # Windowed energy convergence (CG path): stop when the energy improved by
    # < tol_E over the last `plateau_window` iters (robust to per-step CG noise
    # that makes a single-iteration |dE| test stop early).  0 = per-iteration.
    plateau_window: int = 5
    # Orbital optimizer: "cg" (preconditioned Riemannian CG, default) or
    # "newton" (trust-region Newton — a Steihaug truncated-CG on the joint
    # (Q, p) response Hessian, one Fock build per Hessian-vector product).
    # Newton needs far fewer Fock builds than CG on stiff problems; it
    # converges on the gradient norm, so it uses ``tol_grad`` (which defaults
    # to 1e-6 internally when left at 0) instead of ``tol_E``, and ignores the
    # CG-only knobs ``max_step``, ``cg_restart``, ``bt_shrink``, ``bt_max``.
    # Newton is a second-order method and needs float64 precision (enable x64);
    # in float32 it warns and may not converge.  CG works in either precision.
    optimizer: str = "cg"
    tr_delta0: float = 0.5      # Newton: initial trust radius
    tr_cg_max: int = 20         # Newton: max Steihaug inner CG iters per outer step
    # Deflation (optimizer="newton" only): add a repulsive Gaussian bias around
    # each density in ``deflation_targets`` so the solve is pushed out of those
    # basins and converges to a DISTINCT HF solution.  ``deflation_targets`` has
    # shape ``(n_solutions, nk1, nk2, nb, nb)`` — or None for no deflation;
    # ``deflation_sigma`` is the bias height (0 = off) and ``deflation_length``
    # its width in weighted-Frobenius density distance.  Usually driven by
    # :func:`jax_hf.solve_deflated` rather than set by hand.
    deflation_targets: Any = None
    deflation_sigma: float = 0.0
    deflation_length: float = 1.0
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
    plateau_window: int,
    block_sizes: tuple | None,
    project_fn,
    fock_build_fn=None,
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
    _fock_fn = fock_build_fn if fock_build_fn is not None else build_fock
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
    Sigma0, H0, F0 = _fock_fn(
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
        # Stopping rule (mirrors cpp_hf solve_dm):
        #
        # tol_grad > 0 — the gradient criterion is the sole and SUFFICIENT
        # stop: continue only while grad_norm > tol_grad; the energy criterion
        # is inactive.  grad_norm in the carry was measured at the previous
        # body's pre-step point and is the last recorded history entry, so the
        # loop stops with the histories ending at the qualifying (E, grad)
        # pair; the returned state is one accepted descent step past it.
        #
        # tol_grad == 0 — windowed energy test: stop only when the energy
        # improved by less than tol_E over the last `plateau_window` recorded
        # iters.  The line search makes E monotone, so this is a reliable
        # stop, whereas a single-iteration |dE| test can dip below tol_E
        # during CG warm-up or after a backtracked step and quit prematurely.
        k, _, _, _, _, _, _, _, grad_norm, _, dE, hist_E, _ = carry
        if plateau_window > 0:
            e_now = hist_E[jnp.maximum(k - 1, 0)]
            e_past = hist_E[jnp.maximum(k - 1 - plateau_window, 0)]
            energy_not_converged = jnp.where(
                k > plateau_window, (e_past - e_now) > tol_E_r, dE > tol_E_r,
            )
        else:
            energy_not_converged = dE > tol_E_r
        not_converged = jnp.where(
            tol_grad_r > 0.0, grad_norm > tol_grad_r, energy_not_converged,
        )
        return jnp.logical_and(k < max_iter, not_converged)

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
        Sigma, H_h, F = _fock_fn(
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
    _, _, F_pre = _fock_fn(
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
    Sigma_fin, H_fin, F_fin = _fock_fn(
        P_fin, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
        include_exchange=include_exchange, include_hartree=include_hartree,
        exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
        contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
        exchange_block_specs=block_specs,
        project_fn=None,
    )
    E_fin = hf_energy(P_fin, h=h, Sigma=Sigma_fin, H=H_fin, weights_b=weights_b)

    # Convergence flag.  Gradient-primary when tol_grad > 0: grad_fin is the
    # last measured gradient — the entry the loop stopped on, or the final
    # recorded one when max_iter ran out (which cpp_hf's mid-iteration break
    # would have accepted; the plain k_fin < max_iter test would miss it).
    # tol_grad == 0 keeps the legacy flag: loop exited early on the energy
    # criterion.
    converged = jnp.where(
        tol_grad_r > 0.0, grad_fin <= tol_grad_r, k_fin < max_iter,
    )

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
        "plateau_window",
        "block_sizes",
        "project_fn",
        # Optional Fock build override (e.g. superlattice streaming Fock).
        # Captured by identity, so passing the same closure is cache-friendly.
        "fock_build_fn",
    ),
)


def _solve_newton_impl(
    P0: jax.Array,
    n_electrons: float,
    tol_grad: float,
    tr_delta0: float,
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
    deflation_targets: jax.Array,
    deflation_sigma: float,
    deflation_length: float,
    has_deflation: bool,
    max_iter: int,
    tr_cg_max: int,
    mu_maxiter: int,
    block_sizes: tuple | None,
    project_fn,
    fock_build_fn=None,
) -> SolveResult:
    """Trust-region Newton via Steihaug truncated-CG on the joint (Q, p) Hessian.

    Minimizes the same free energy as :func:`_solve_impl`, but with a
    second-order step: the joint (Q, p) Hessian-vector product uses the exact
    linear interaction response ``Sigma[dP] = F[dP] - h`` (one Fock build each),
    and a Steihaug truncated-CG solves the trust-region subproblem.  Converges
    on the gradient norm (commutator ``[F, P]``); far fewer Fock builds than CG
    on stiff problems.  Implementation mirrors ``cpp_hf``'s ``solve_rtr``.
    """
    target_dtype = h.dtype
    real_dtype = jnp.zeros((), dtype=target_dtype).real.dtype
    tiny = jnp.asarray(1e-30, dtype=real_dtype)

    _project = project_fn if project_fn is not None else (lambda A: A)
    _fock_fn = fock_build_fn if fock_build_fn is not None else build_fock
    block_specs = _exchange_block_specs(block_sizes)

    w2d = jnp.asarray(weights_b[..., 0, 0], dtype=real_dtype)
    weight_sum_r = jnp.asarray(weight_sum, dtype=real_dtype)
    w_norm = w2d / jnp.maximum(weight_sum_r, tiny)
    ones_w = jnp.ones_like(w2d)

    n_target = jnp.asarray(n_electrons, dtype=real_dtype)
    n_target_norm = n_target / jnp.maximum(weight_sum_r, tiny)

    T_r = jnp.maximum(jnp.asarray(T, dtype=real_dtype), jnp.asarray(1e-12, dtype=real_dtype))
    tol_g = jnp.asarray(tol_grad, dtype=real_dtype)
    denom_scale_r = jnp.asarray(denom_scale, dtype=real_dtype)
    delta0 = jnp.asarray(tr_delta0, dtype=real_dtype)
    delta_min = jnp.asarray(1e-10, dtype=real_dtype)
    occ_clip = jnp.asarray(1e-8, dtype=real_dtype)
    # Relative energy-noise floor for the trust-region ratio: ~4500 * machine
    # epsilon, i.e. ~1e-12 in float64 (cpp_hf's hard-coded value) and ~5e-4 in
    # float32, tracking the precision of the Fock build that produces E.
    noise_floor = jnp.maximum(
        jnp.asarray(1e-12, dtype=real_dtype),
        jnp.asarray(4500.0 * jnp.finfo(real_dtype).eps, dtype=real_dtype),
    )

    nb = h.shape[-1]
    eye = jnp.eye(nb, dtype=target_dtype)
    offdiag = (1.0 - jnp.eye(nb, dtype=real_dtype))[None, None, ...]
    refP_zero = jnp.zeros_like(refP)

    def _fock_full(P):
        return _fock_fn(
            P, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
            exchange_block_specs=block_specs, project_fn=None,
        )

    def _fock_response(dP):
        # Linear interaction response Sigma[dP] = F[dP] - h, with refP = 0 so
        # the interaction sees dP directly (exact since Sigma is linear in P).
        _, _, F_resp = _fock_fn(
            dP, h=h, VR=VR, refP=refP_zero, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
            exchange_block_specs=block_specs, project_fn=None,
        )
        return F_resp - h

    # Joint metric: orbital part unweighted, occupation part weighted by w2d.
    def _jip(aX, ap, bX, bp):
        return _ip_matrix(aX, bX, ones_w) + _ip_vec(ap, bp, w2d)

    def _jip_norm(aX, ap):
        return jnp.sqrt(jnp.maximum(0.0, _jip(aX, ap, aX, ap)))

    # Deflation bias (Newton-only): a repulsive Gaussian hill around each
    # already-found density, so a re-solve is pushed out of those basins and
    # converges to a DISTINCT HF solution.
    #   Phi      = sigma * sum_i exp(-d_i^2 / (2 L^2)),
    #   d_i^2    = sum_k w_k ||P_k - target_{i,k}||_F^2   (gauge-invariant in P)
    #   S_pen    = sum_i 2 phi'(d_i^2) (P - target_i),    phi'(x) = -phi(x)/(2L^2)
    # The force S_pen is folded into the Fock (F_eff = F + S_pen) so the gradient
    # sees the bias, and pen_diag = 2 sum_i phi'(d_i^2) is the diagonal penalty
    # curvature injected into the Hessian-vector product.  ``has_deflation`` is a
    # static flag, so a normal Newton solve compiles with none of this.
    defl_inv2L2 = 1.0 / (2.0 * jnp.maximum(
        jnp.asarray(deflation_length, dtype=real_dtype),
        jnp.asarray(1e-12, dtype=real_dtype)) ** 2)
    defl_sigma = jnp.asarray(deflation_sigma, dtype=real_dtype)

    def _deflation_bias(P):
        diff = P[None] - deflation_targets               # (n_defl, nk1, nk2, nb, nb)
        per_k = jnp.sum(jnp.abs(diff) ** 2, axis=(-2, -1))       # (n_defl, nk1, nk2)
        d2 = jnp.sum(w2d[None] * per_k, axis=(1, 2))             # (n_defl,)
        phi = defl_sigma * jnp.exp(-d2 * defl_inv2L2)
        phip = -phi * defl_inv2L2                                # phi'(d^2) < 0
        coeff = (2.0 * phip).astype(target_dtype)
        S_pen = jnp.sum(coeff[:, None, None, None, None] * diff, axis=0)
        return jnp.sum(phi), S_pen, 2.0 * jnp.sum(phip)

    def _eval_state(Q, p):
        """Energy, orbital-basis Fock, gradient norm, penalty curvature at (Q, p).

        One Fock build.  Convergence is judged on the orbital gradient
        (equivalently the commutator ``[F, P]``); the occupation sector is kept
        Fermi-Dirac optimal by construction at every step and is relaxed once at
        the cold start (see ``_relax_occupations``).  With deflation the Fock
        carries the penalty force and ``E`` is the biased free energy (the
        physical energy is recomputed unbiased at the finalize).
        """
        P = _herm(_project(_density_from_Qp(Q, p)))
        Sigma, H_h, F = _fock_full(P)
        E = hf_energy(P, h=h, Sigma=Sigma, H=H_h, weights_b=weights_b)
        if has_deflation:
            Phi, S_pen, pen_diag = _deflation_bias(P)
            E = E + Phi
            F = F + S_pen
        else:
            pen_diag = jnp.zeros((), dtype=real_dtype)
        Ft = _fock_in_orbital_basis(Q, F)
        diff_p = p[..., None, :] - p[..., :, None]
        G = _skew_hermitian(diff_p * Ft) * offdiag
        grad_norm = _norm_matrix(G, w_norm)
        return E, Ft, grad_norm, pen_diag

    # ---- Initialize (Q, p) from Fock at P0 ----
    P0_h = _herm(jnp.asarray(P0, dtype=target_dtype))
    _, _, F0 = _fock_full(P0_h)
    eps0, Q0 = jnp.linalg.eigh(F0)
    eps0 = eps0.astype(real_dtype)
    Q0 = Q0.astype(target_dtype)
    mu0 = _solve_mu(eps0, w_norm, n_target_norm,
                    jnp.asarray(0.0, dtype=real_dtype), T_r, maxiter=mu_maxiter)
    p0 = jax.nn.sigmoid((mu0 - eps0) / T_r).astype(real_dtype)
    E0, Ft0, grad0, pen0 = _eval_state(Q0, p0)

    # Cold-start occupation relaxation.  The orbital gradient diff_p * Ft is
    # blind to the occupations, so if it is already below tolerance at the cold
    # start the orbital sector may simply be decoupled from a not-yet-relaxed
    # occupation sector (e.g. diagonal h with scalar exchange).  In that case
    # relax the occupations to self-consistency at the fixed initial orbitals
    # before trusting the convergence test, so we do not stop at the
    # non-interacting occupations.  This only runs when grad0 <= tol_g, so a
    # normal problem (large initial gradient) pays nothing.
    def _relax_occupations(operands):
        p, mu, _E, _Ft, _g, _pen = operands

        def relax_step(_i, state):
            p_i, mu_i = state
            P = _herm(_project(_density_from_Qp(Q0, p_i)))
            _, _, F = _fock_full(P)
            eps_i = jnp.real(jnp.diagonal(
                _fock_in_orbital_basis(Q0, F), axis1=-2, axis2=-1)).astype(real_dtype)
            mu_i = _solve_mu(eps_i, w_norm, n_target_norm, mu_i, T_r, maxiter=mu_maxiter)
            return jax.nn.sigmoid((mu_i - eps_i) / T_r).astype(real_dtype), mu_i

        p_r, mu_r = lax.fori_loop(0, 30, relax_step, (p, mu))
        E_r, Ft_r, g_r, pen_r = _eval_state(Q0, p_r)
        return p_r, mu_r, E_r, Ft_r, g_r, pen_r

    p0, mu0, E0, Ft0, grad0, pen0 = lax.cond(
        grad0 <= tol_g, _relax_occupations, lambda o: o,
        (p0, mu0, E0, Ft0, grad0, pen0),
    )

    hist_E = jnp.zeros(max_iter, dtype=real_dtype)
    hist_grad = jnp.zeros(max_iter, dtype=real_dtype)

    # carry: (k, Q, p, mu, Ft, E, Delta, grad_norm, pen_diag, hist_E, hist_grad)
    carry0 = (jnp.int32(0), Q0, p0, mu0, Ft0, E0, delta0, grad0, pen0,
              hist_E, hist_grad)

    def outer_cond(carry):
        k, _, _, _, _, _, Delta, grad_norm, _, _, _ = carry
        return jnp.logical_and(
            k < max_iter,
            jnp.logical_and(grad_norm > tol_g, Delta > delta_min),
        )

    def outer_body(carry):
        k, Q, p, mu, Ft, E, Delta, grad_norm, pen_diag, hE, hG = carry
        hE = hE.at[k].set(E)
        hG = hG.at[k].set(grad_norm)

        eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)).astype(real_dtype)
        diff_p = p[..., None, :] - p[..., :, None]
        G = _skew_hermitian(diff_p * Ft) * offdiag

        eps_scale = jnp.sqrt(jnp.mean(eps ** 2) + tiny)
        lam_pc = jnp.maximum(T_r, denom_scale_r * eps_scale)

        # Block preconditioner: orbital 1/sqrt(gap^2 + lam^2), occupation
        # p(1-p)/T, then project the occupation onto particle conservation.
        gap = eps[..., :, None] - eps[..., None, :]
        denomX = jnp.sqrt(gap ** 2 + lam_pc ** 2)
        pc = jnp.clip(p, occ_clip, 1.0 - occ_clip)
        fd_curv = pc * (1.0 - pc) / T_r

        def _prec(aX, ap):
            return aX / denomX, _project_occ(ap * fd_curv, w2d)

        # Joint (Q, p) Hessian-vector product (one Fock build via _fock_response).
        def _hvp(X, dp):
            M = X * diff_p + dp[..., :, None] * eye  # [X, diag p] off-diag + diag(dp)
            dP = jnp.einsum("...in,...nm,...jm->...ij", Q, M, jnp.conj(Q))
            dP = _herm(dP)
            resp = _fock_response(dP)
            if has_deflation:
                resp = resp + pen_diag * dP             # diagonal penalty curvature
            St = _fock_in_orbital_basis(Q, resp)        # response in orbital basis
            A = jnp.matmul(Ft, X) - jnp.matmul(X, Ft) + St  # [Ft, X] + St
            diff_dp = dp[..., None, :] - dp[..., :, None]
            C = diff_dp * Ft + diff_p * A
            HX = _skew_hermitian(C) * offdiag
            A_diag = jnp.real(jnp.diagonal(A, axis1=-2, axis2=-1)).astype(real_dtype)
            Hp = A_diag + (T_r / (pc * (1.0 - pc))) * dp
            return HX, _project_occ(Hp, w2d)

        # ---- Steihaug truncated CG on the joint quadratic model ----
        rX0 = -G
        rp0 = jnp.zeros_like(p)
        zX0, zp0 = _prec(rX0, rp0)
        rz0 = _jip(rX0, rp0, zX0, zp0)
        zerosQ = jnp.zeros_like(Q)
        zerosp = jnp.zeros_like(p)
        # carry: (cg, done, vX, vp, HvX, Hvp, rX, rp, dX, dp, rz)
        cg_carry0 = (jnp.int32(0), jnp.bool_(False),
                     zerosQ, zerosp, zerosQ, zerosp,
                     rX0, rp0, zX0, zp0, rz0)

        def cg_cond(c):
            return jnp.logical_and(c[0] < tr_cg_max, jnp.logical_not(c[1]))

        def cg_body(c):
            cg, _done, vX, vp, HvX, Hvp, rX, rp, dX, dp, rz = c
            HdX, Hdp = _hvp(dX, dp)
            dHd = _jip(dX, dp, HdX, Hdp)
            dd = _jip(dX, dp, dX, dp)
            neg_curv = dHd <= 1e-14 * dd

            # to_boundary: positive root t of ||v + t d||^2 = Delta^2.
            b_coef = 2.0 * _jip(vX, vp, dX, dp)
            cc = _jip(vX, vp, vX, vp) - Delta * Delta
            disc = jnp.maximum(b_coef ** 2 - 4.0 * dd * cc, 0.0)
            t_bd = (-b_coef + jnp.sqrt(disc)) / (2.0 * dd + 1e-30)
            vX_bd, vp_bd = vX + t_bd * dX, vp + t_bd * dp
            HvX_bd, Hvp_bd = HvX + t_bd * HdX, Hvp + t_bd * Hdp

            # Full CG step (guard alpha denom so negative-curvature stays finite).
            alpha = rz / jnp.where(neg_curv, jnp.ones_like(dHd), dHd)
            vX_a, vp_a = vX + alpha * dX, vp + alpha * dp
            norm_va = _jip_norm(vX_a, vp_a)
            cross = norm_va >= Delta
            take_boundary = jnp.logical_or(neg_curv, cross)

            HvX_a, Hvp_a = HvX + alpha * HdX, Hvp + alpha * Hdp
            rX_a, rp_a = rX - alpha * HdX, rp - alpha * Hdp
            rn = _jip_norm(rX_a, rp_a)
            inner_conv = rn < 0.05 * grad_norm
            zX_a, zp_a = _prec(rX_a, rp_a)
            rz_new = _jip(rX_a, rp_a, zX_a, zp_a)
            beta = rz_new / jnp.where(jnp.abs(rz) > tiny, rz, jnp.ones_like(rz))
            dX_a, dp_a = zX_a + beta * dX, zp_a + beta * dp

            stop = jnp.logical_or(take_boundary, inner_conv)
            vX_n = jnp.where(take_boundary, vX_bd, vX_a)
            vp_n = jnp.where(take_boundary, vp_bd, vp_a)
            HvX_n = jnp.where(take_boundary, HvX_bd, HvX_a)
            Hvp_n = jnp.where(take_boundary, Hvp_bd, Hvp_a)
            rX_n = jnp.where(take_boundary, rX, rX_a)
            rp_n = jnp.where(take_boundary, rp, rp_a)
            dX_n = jnp.where(stop, dX, dX_a)
            dp_n = jnp.where(stop, dp, dp_a)
            rz_n = jnp.where(stop, rz, rz_new)
            return (cg + 1, stop, vX_n, vp_n, HvX_n, Hvp_n,
                    rX_n, rp_n, dX_n, dp_n, rz_n)

        cg_fin = lax.while_loop(cg_cond, cg_body, cg_carry0)
        _, _, vX, vp, HvX, Hvp, _, _, _, _, _ = cg_fin

        # Predicted reduction m(0) - m(v) in TRUE energy units: the orbital
        # gradient/Hessian action are per-k (unweighted) but the energy carries
        # the BZ measure w2d, so both terms here must be w2d-weighted for the
        # trust-region ratio to compare against the actual energy change.
        gv = _ip_matrix(G, vX, w2d)
        vHv = _ip_matrix(vX, HvX, w2d) + _ip_vec(vp, Hvp, w2d)
        pred = -(gv + 0.5 * vHv)

        # Retract Q <- Q . Cayley(tau=-1, vX): Cayley(tau, d) ~ I - tau d, and
        # Steihaug returns vX = -H^{-1} G (the descent direction), so tau = -1
        # moves along +vX.  Trial occupations come from the frozen Ft.
        V_v, lam_v = _cayley_spectral_setup(vX)
        U = _cayley_unitary_from_spectrum(V_v, lam_v, jnp.asarray(-1.0, dtype=real_dtype))
        Qt = Q @ U
        Ft_froz = _fock_in_orbital_basis(U, Ft)  # U† Ft U
        eps_t = jnp.real(jnp.diagonal(Ft_froz, axis1=-2, axis2=-1)).astype(real_dtype)
        mu_t = _solve_mu(eps_t, w_norm, n_target_norm, mu, T_r, maxiter=mu_maxiter)
        p_trial = jax.nn.sigmoid((mu_t - eps_t) / T_r).astype(real_dtype)

        # True energy + gradient at the trial point (one Fock build per outer).
        E_trial, Ft_trial, grad_trial, pen_trial = _eval_state(Qt, p_trial)

        # Energy-noise floor below which the predicted reduction is meaningless
        # and the step is judged on the gradient instead.  Scaled to the working
        # precision: ~1e-12 in float64 (matching cpp_hf), ~5e-4 in float32, so
        # the trust-region ratio is never trusted below the actual Fock-build
        # round-off (otherwise float32 noise makes every ratio garbage).
        noise = noise_floor * jnp.maximum(jnp.abs(E), 1.0)
        use_grad_branch = pred <= noise

        # Branch 1 (pred below energy noise): judge by the gradient, never climb.
        accept_g = jnp.logical_and(grad_trial <= grad_norm, E_trial <= E + noise)
        Delta_g = jnp.where(accept_g, Delta, 0.5 * Delta)
        # Branch 2 (standard trust-region ratio).
        ratio = (E - E_trial) / jnp.where(use_grad_branch, jnp.ones_like(pred), pred)
        step_norm = _jip_norm(vX, vp)
        Delta_r = jnp.where(
            ratio < 0.25, 0.25 * Delta,
            jnp.where(jnp.logical_and(ratio > 0.75, step_norm > 0.9 * Delta),
                      jnp.minimum(2.0 * Delta, 5.0), Delta),
        )
        accept_r = ratio > 0.1

        accept = jnp.where(use_grad_branch, accept_g, accept_r)
        Delta_new = jnp.where(use_grad_branch, Delta_g, Delta_r)

        Q_new = jnp.where(accept, Qt, Q)
        p_new = jnp.where(accept, p_trial, p)
        Ft_new = jnp.where(accept, Ft_trial, Ft)
        E_new = jnp.where(accept, E_trial, E)
        mu_new = jnp.where(accept, mu_t, mu)
        grad_new = jnp.where(accept, grad_trial, grad_norm)
        pen_new = jnp.where(accept, pen_trial, pen_diag)

        return (k + 1, Q_new, p_new, mu_new, Ft_new, E_new, Delta_new,
                grad_new, pen_new, hE, hG)

    (k_fin, Q_fin, p_fin, mu_fin, _Ft_fin, _E_loop, _Delta_fin, grad_fin,
     _pen_fin, hist_E, hist_grad) = lax.while_loop(outer_cond, outer_body, carry0)

    # ---- Finalize: rebuild the Fock at the converged (Q, p) and report.
    # Unlike the CG path we do NOT re-diagonalize and re-occupy here: at finite
    # temperature with near-degenerate occupations the orbital gradient
    # diff_p * Ft can mask sizeable off-diagonal Ft entries, so an eigh +
    # Fermi-Dirac re-occupation acts as one extra (unconverged) SCF step that
    # would move the density away from the stationary point the loop reached.
    # Reporting the loop's own (Q, p) keeps the solution self-consistent
    # (mirrors cpp_hf's solve_rtr finalize).
    P_fin = _herm(_project(_density_from_Qp(Q_fin, p_fin)))
    Sigma_fin, H_fin, F_fin = _fock_full(P_fin)
    E_fin = hf_energy(P_fin, h=h, Sigma=Sigma_fin, H=H_fin, weights_b=weights_b)

    converged = grad_fin <= tol_g

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


_solve_newton_jitted = jax.jit(
    _solve_newton_impl,
    static_argnames=(
        "include_hartree",
        "include_exchange",
        "exchange_hermitian_channel_packing",
        "has_deflation",
        "max_iter",
        "tr_cg_max",
        "mu_maxiter",
        "block_sizes",
        "project_fn",
        "fock_build_fn",
    ),
)


def solve_direct_minimization(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: SolverConfig | None = None,
    fock_build_fn=None,
) -> SolveResult:
    """Direct-minimization Hartree-Fock solver.

    With ``config.optimizer="cg"`` (default): preconditioned Riemannian CG on
    Stiefel x capped simplex, one Fock build per iteration.  With
    ``config.optimizer="newton"``: trust-region Newton (Steihaug truncated-CG
    on the joint (Q, p) response Hessian), which converges on the gradient norm
    and typically needs far fewer Fock builds on stiff problems.  See
    :class:`SolverConfig` for tuning knobs.

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

    optimizer = str(config.optimizer).lower()
    if optimizer not in ("cg", "newton"):
        raise ValueError(
            f"Unknown optimizer {config.optimizer!r}; expected 'cg' or 'newton'."
        )

    deflation_on = (
        config.deflation_targets is not None and float(config.deflation_sigma) > 0.0
    )
    if deflation_on and optimizer != "newton":
        raise ValueError(
            "deflation is only supported with optimizer='newton' "
            f"(got optimizer={config.optimizer!r})."
        )

    if optimizer == "newton":
        # Trust-region Newton is a second-order method: its Steihaug Hessian-
        # vector products go through the FFT Fock build, whose float32 round-off
        # (~1e-6) swamps the small Hessian eigenvalues on near-degenerate
        # problems and yields ascent directions.  cpp_hf sidesteps this by
        # computing in double precision throughout; warn so a float32 caller
        # knows to enable x64 rather than silently get converged=False.
        if jnp.finfo(jnp.asarray(kernel.h).real.dtype).bits < 64:
            warnings.warn(
                "optimizer='newton' needs float64 precision to converge "
                "reliably; the kernel is float32. Enable x64 via "
                "jax.config.update('jax_enable_x64', True) (and build the "
                "kernel in float64). The CG optimizer works in float32.",
                stacklevel=2,
            )
        # Trust-region Newton converges on the gradient norm; default tol_grad
        # to 1e-6 when the caller leaves it at 0 (mirrors cpp_hf's solve_rtr).
        tol_grad = float(config.tol_grad) if config.tol_grad > 0.0 else 1e-6

        # Deflation targets -> (n_solutions, nk1, nk2, nb, nb), matching the
        # kernel dtype.  An empty (0, ...) stack with has_deflation=False is the
        # no-deflation path (the bias machinery is then compiled out entirely).
        h_dtype = jnp.asarray(kernel.h).dtype
        nk1, nk2, nb = kernel.h.shape[0], kernel.h.shape[1], kernel.h.shape[-1]
        if deflation_on:
            defl_arr = jnp.asarray(config.deflation_targets, dtype=h_dtype)
            if defl_arr.ndim == 4:
                defl_arr = defl_arr[None, ...]
            if defl_arr.ndim != 5 or defl_arr.shape[1:] != (nk1, nk2, nb, nb):
                raise ValueError(
                    "deflation_targets must have shape "
                    "(n_solutions, nk1, nk2, nb, nb) matching the kernel; got "
                    f"{tuple(jnp.asarray(config.deflation_targets).shape)}."
                )
        else:
            defl_arr = jnp.zeros((0, nk1, nk2, nb, nb), dtype=h_dtype)

        return _solve_newton_jitted(
            jnp.asarray(P0),
            float(n_electrons),
            tol_grad,
            float(config.tr_delta0),
            float(config.denom_scale),
            **kernel.as_args(),
            deflation_targets=defl_arr,
            deflation_sigma=float(config.deflation_sigma),
            deflation_length=float(config.deflation_length),
            has_deflation=bool(deflation_on),
            max_iter=int(config.max_iter),
            tr_cg_max=int(config.tr_cg_max),
            mu_maxiter=int(config.mu_maxiter),
            block_sizes=config.block_sizes,
            project_fn=config.project_fn,
            fock_build_fn=fock_build_fn,
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
        plateau_window=int(config.plateau_window),
        block_sizes=config.block_sizes,
        project_fn=config.project_fn,
        fock_build_fn=fock_build_fn,
    )


# Public alias: direct minimization is the default solver.
solve = solve_direct_minimization
