"""
Option A (Frozen-F alternating minimization), eigen-decomposition-free:

Outer loop (expensive): build F[P] once per iteration (same cost-dominant step as SCF).
Inner loop (cheap, no new exchange builds): with F frozen,
  - update occupations p exactly from diag(F_t) using a robust 1D mu solve
  - update orbitals Q by commutator-based Cayley/Jacobi sweeps, with automatic backtracking
    on the commutator norm (still using only frozen F_t).

This removes the need for fine-tuned gradient steps / BB / Anderson in most cases, while
keeping performance comparable (1 exchange build per outer iteration + small nbxnb solves).

Notes:
- Works best with finite T (smearing). If T=0 is passed, we use T_eff=max(T,1e-12).
- Stabilized for degenerate metallic subspaces via occupation-difference damping (p_floor).
"""

from __future__ import annotations

from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from .linalg import normalize_block_specs
from .utils import selfenergy_fft


# ----------------------------
# Small linear algebra helpers
# ----------------------------


def _herm(X: jax.Array) -> jax.Array:
    """Hermitize on the last two axes."""
    return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))


def _skew_hermitian(A: jax.Array) -> jax.Array:
    """Anti-Hermitize on the last two axes: (A - A†)/2."""
    return 0.5 * (A - jnp.conj(jnp.swapaxes(A, -1, -2)))


def _density_from_Qp(Q: jax.Array, p: jax.Array) -> jax.Array:
    """P = Q diag(p) Q†, batched over leading axes.  Hermitian by construction (p is real)."""
    return jnp.einsum("...in,...n,...jn->...ij", Q, p, jnp.conj(Q))


def _fock_in_orbital_basis(Q: jax.Array, F: jax.Array) -> jax.Array:
    """F̃ = Q† F Q.  Hermitian by construction (F is Hermitian)."""
    return jnp.einsum("...in,...ij,...jm->...nm", jnp.conj(Q), F, Q)


def _cayley_unitary(Omega: jax.Array, I: jax.Array) -> jax.Array:
    """U = (I - Omega/2)^-1 (I + Omega/2). Omega should be anti-Hermitian."""
    A = I - 0.5 * Omega
    B = I + 0.5 * Omega
    return jnp.linalg.solve(A, B)


def _weighted_rms_matrix(X: jax.Array, w_norm: jax.Array, eps: float = 1e-30) -> jax.Array:
    """Weighted RMS for matrices X(k,i,j): sqrt( sum_k w * sum_ij |X|^2 )."""
    per_k = jnp.sum(jnp.abs(X) ** 2, axis=(-2, -1))
    return jnp.sqrt(jnp.sum(w_norm * per_k) / jnp.maximum(jnp.sum(w_norm), eps))


def _weighted_rms_vec(X: jax.Array, w_norm: jax.Array, eps: float = 1e-30) -> jax.Array:
    """Weighted RMS for vectors X(k,i): sqrt( sum_k w * sum_i X^2 )."""
    return jnp.sqrt(jnp.sum(w_norm[..., None] * (X * X)) / jnp.maximum(jnp.sum(w_norm), eps))


def _logit(p: jax.Array, eps: float = 1e-12) -> jax.Array:
    p = jnp.clip(p, eps, 1.0 - eps)
    return jnp.log(p) - jnp.log1p(-p)


# --------------------------------------
# Particle-number constraint: solve delta (for init / enforcing N on p)
# --------------------------------------


def _solve_delta_newton_bracket(
    logits: jax.Array,          # (nk1,nk2,nb) real
    w_norm: jax.Array,          # (nk1,nk2) real, sum(w_norm)=1
    n_target_norm: jax.Array,   # scalar real, in [0, nb]
    delta0: jax.Array,          # scalar real initial guess
    *,
    maxiter: int = 12,
    tol: float = 1e-12,
) -> jax.Array:
    """Solve delta so sum_k w_norm sum_i sigmoid(logits+delta) == n_target_norm."""
    logits = jnp.asarray(logits)
    real_dtype = logits.dtype

    s_min = jnp.min(logits)
    s_max = jnp.max(logits)
    lo = jnp.asarray(-50.0, dtype=real_dtype) - s_max
    hi = jnp.asarray(+50.0, dtype=real_dtype) - s_min

    delta = jnp.clip(jnp.asarray(delta0, dtype=real_dtype), lo, hi)

    w_norm_b = w_norm[..., None]

    def count_and_slope(delta_val):
        x = logits + delta_val
        p = jax.nn.sigmoid(x)
        N = jnp.sum(w_norm_b * p)
        dp = p * (1.0 - p)
        Z = jnp.sum(w_norm_b * dp)
        return N, Z

    def body(state, _):
        delta, lo, hi = state
        N, Z = count_and_slope(delta)
        g = N - n_target_norm

        lo = jnp.where(g < 0, delta, lo)
        hi = jnp.where(g > 0, delta, hi)

        Z_safe = jnp.maximum(Z, jnp.asarray(1e-18, dtype=real_dtype))
        delta_new = delta - g / Z_safe

        delta_bis = 0.5 * (lo + hi)
        out_of = jnp.logical_or(delta_new <= lo, delta_new >= hi)
        delta_new = jnp.where(out_of, delta_bis, delta_new)

        delta_new = jnp.clip(delta_new, lo, hi)
        delta_new = jnp.where(jnp.isfinite(delta_new), delta_new, delta_bis)
        return (delta_new, lo, hi), None

    (delta_fin, lo_fin, hi_fin), _ = lax.scan(body, (delta, lo, hi), xs=None, length=int(maxiter))

    N_fin, _ = count_and_slope(delta_fin)
    g_fin = jnp.abs(N_fin - n_target_norm)
    return jnp.where(g_fin > jnp.asarray(tol, dtype=real_dtype), 0.5 * (lo_fin + hi_fin), delta_fin)


# --------------------------------------
# Chemical potential solve for Fermi-Dirac occupations (frozen F inner loop)
# --------------------------------------


def _solve_mu_fd_newton_bracket(
    eps: jax.Array,             # (nk1,nk2,nb) real energies
    w_norm: jax.Array,          # (nk1,nk2) real, sum=1
    n_target_norm: jax.Array,   # scalar real in [0,nb]
    mu0: jax.Array,             # scalar real initial guess
    T: float,
    *,
    maxiter: int = 25,
    tol: float = 1e-12,
) -> jax.Array:
    """Solve mu so sum_k w_norm sum_i sigmoid((mu - eps)/T) == n_target_norm."""
    real_dtype = eps.dtype
    Tj = jnp.asarray(T, dtype=real_dtype)
    Tj = jnp.maximum(Tj, jnp.asarray(1e-12, dtype=real_dtype))

    e_min = jnp.min(eps)
    e_max = jnp.max(eps)
    lo = e_min - 50.0 * Tj
    hi = e_max + 50.0 * Tj

    mu = jnp.clip(jnp.asarray(mu0, dtype=real_dtype), lo, hi)

    w_norm_b = w_norm[..., None]

    def count_and_slope(mu_val):
        x = (mu_val - eps) / Tj
        p = jax.nn.sigmoid(x)
        N = jnp.sum(w_norm_b * p)
        dp = (p * (1.0 - p)) / Tj
        Z = jnp.sum(w_norm_b * dp)
        return N, Z

    def body(state, _):
        mu, lo, hi = state
        N, Z = count_and_slope(mu)
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

    (mu_fin, lo_fin, hi_fin), _ = lax.scan(body, (mu, lo, hi), xs=None, length=int(maxiter))

    N_fin, _ = count_and_slope(mu_fin)
    g_fin = jnp.abs(N_fin - n_target_norm)
    return jnp.where(g_fin > jnp.asarray(tol, dtype=real_dtype), 0.5 * (lo_fin + hi_fin), mu_fin)


def _occupations_from_eps(eps: jax.Array, mu: jax.Array, T: float) -> jax.Array:
    real_dtype = eps.dtype
    Tj = jnp.asarray(T, dtype=real_dtype)
    Tj = jnp.maximum(Tj, jnp.asarray(1e-12, dtype=real_dtype))
    return jax.nn.sigmoid((mu - eps) / Tj)


# ----------------------------
# Mean-field build: Sigma / Hartree / F
# ----------------------------


def _build_sigma_hartree(
    P: jax.Array,
    *,
    h: jax.Array,
    VR: jax.Array,
    refP: jax.Array,
    HH: jax.Array,
    w2d: jax.Array,                 # (nk1,nk2) real physical weights
    include_exchange: bool,
    include_hartree: bool,
    exchange_hermitian_channel_packing: bool,
    exchange_block_specs: Any | None = None,
    exchange_check_offdiag: bool | None = None,
    exchange_offdiag_atol: float = 1e-12,
    exchange_offdiag_rtol: float = 0.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Return (Sigma, H, F) at density P."""
    Sigma = (
        selfenergy_fft(
            VR,
            P - refP,               # P is Hermitian by construction (_density_from_Qp hermitizes)
            block_specs=exchange_block_specs,
            check_offdiag=exchange_check_offdiag,
            offdiag_atol=exchange_offdiag_atol,
            offdiag_rtol=exchange_offdiag_rtol,
            _apply_ifftshift=False,  # VR has shift phase absorbed
            hermitian_channel_packing=exchange_hermitian_channel_packing,
        )
        if include_exchange
        else jnp.zeros_like(h)
    )

    if include_hartree:
        dP = P - refP              # P already Hermitian
        diag_real = jnp.real(jnp.diagonal(dP, axis1=-2, axis2=-1))  # (nk1,nk2,nb)
        n_vec = jnp.sum(w2d[..., None] * diag_real, axis=(0, 1))    # (nb,)
        sigma_diag = HH @ n_vec                                     # (nb,)
        H_mat = jnp.diag(sigma_diag.astype(h.real.dtype))
        H = H_mat[None, None, ...]
    else:
        H = jnp.zeros_like(h)

    F = _herm(h + Sigma + H)
    return Sigma, H, F


def _hf_energy(
    P: jax.Array,
    *,
    h: jax.Array,
    Sigma: jax.Array,
    H: jax.Array,
    weights_b: jax.Array,   # (nk1,nk2,1,1)
) -> jax.Array:
    """E = sum_k w_k Tr[(h + 0.5(Sigma+H)) P]."""
    return jnp.sum(jnp.real(jnp.einsum("...ij,...ji->...", weights_b * (h + 0.5 * (Sigma + H)), P)))


# ----------------------------
# Params for frozen-F variational solver
# ----------------------------


class VariationalHFParams(NamedTuple):
    """
    Parameters for the frozen-F variational solver.

    Q(k): unitary orbital matrix (columns = orbitals).
    p(k): occupations in (0,1).
    mu:  scalar chemical potential (used as initial guess inside inner sweeps).
    """
    Q: jax.Array      # (nk1,nk2,nb,nb) complex
    p: jax.Array      # (nk1,nk2,nb) real
    mu: jax.Array     # scalar real


# ----------------------------
# Frozen-F inner solve (cheap)
# ----------------------------


def _comm_norm(Ft: jax.Array, diff: jax.Array, w_norm: jax.Array, offdiag: jax.Array) -> jax.Array:
    return _weighted_rms_matrix(diff * Ft * offdiag, w_norm)


def _stationarity_norm(
    Ft: jax.Array, p: jax.Array, w_norm: jax.Array,
    p_floor: float, T: float, denom_scale: float,
) -> jax.Array:
    """
    Energy-denominator weighted stationarity norm (true Grassmann gradient norm):
      - gradient term: (p_j - p_i) F_ij / denom  for non-degenerate occupations
      - Jacobi gauge term: F_ij / denom  in near-degenerate occupation subspaces
    where denom = sqrt(gap² + λ²), λ = max(T, denom_scale * eps_scale).
    """
    real_dtype = p.dtype
    tiny_eps = jnp.asarray(1e-30, dtype=real_dtype)
    diff = p[..., None, :] - p[..., :, None]
    occ_scale = jnp.abs(diff) / (jnp.abs(diff) + jnp.asarray(p_floor, dtype=real_dtype))

    n = Ft.shape[-1]
    offdiag = (1.0 - jnp.eye(n, dtype=real_dtype))[None, None, ...]

    # Energy denominator (same formula as inner solve)
    eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)).astype(real_dtype)
    gap = eps[..., :, None] - eps[..., None, :]
    eps_scale = jnp.sqrt(jnp.mean(eps ** 2) + tiny_eps)
    lam = jnp.maximum(
        jnp.asarray(T, dtype=real_dtype),
        jnp.asarray(denom_scale, dtype=real_dtype) * eps_scale,
    )
    denom = jnp.sqrt(gap ** 2 + lam ** 2)

    comm = (diff * Ft * offdiag) / denom
    jac = ((1.0 - occ_scale) * Ft * offdiag) / denom

    per_k = jnp.sum(jnp.abs(comm) ** 2 + jnp.abs(jac) ** 2, axis=(-2, -1))
    return jnp.sqrt(jnp.sum(w_norm * per_k) / jnp.maximum(jnp.sum(w_norm), tiny_eps))


def _frozen_F_energy(Ft: jax.Array, p: jax.Array, w_norm: jax.Array) -> jax.Array:
    """Frozen-F energy: E = sum_k w_k sum_i p_i Ft_ii."""
    eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1))
    return jnp.sum(w_norm[..., None] * p * eps)


def _backtracking_cayley_right(
    Q: jax.Array,
    Ft: jax.Array,
    Omega_dir: jax.Array,     # anti-Hermitian direction in orbital basis
    *,
    diff: jax.Array,
    p: jax.Array,
    offdiag: jax.Array,
    cayley_eye: jax.Array,
    w_norm: jax.Array,
    tau0: float,
    accept_ratio: float,
    shrink: float,
    max_backtrack: int,
) -> Tuple[jax.Array, jax.Array]:
    """
    Try tau= tau0, tau0*shrink, ... until commutator norm decreases sufficiently
    AND frozen-F energy does not increase.
    Returns (Q_new, Ft_new). If no step is accepted, returns inputs unchanged.
    """
    real_dtype = diff.dtype
    tau0 = jnp.asarray(tau0, dtype=real_dtype)
    shrink = jnp.asarray(shrink, dtype=real_dtype)
    accept_ratio = jnp.asarray(accept_ratio, dtype=real_dtype)

    norm0 = _comm_norm(Ft, diff, w_norm, offdiag)
    E0 = _frozen_F_energy(Ft, p, w_norm)

    def cond(state):
        i, tau, accepted, Q_best, Ft_best = state
        del tau, Q_best, Ft_best
        return jnp.logical_and(i < jnp.int32(max_backtrack), jnp.logical_not(accepted))

    def body(state):
        i, tau, accepted, Q_best, Ft_best = state

        Omega = (tau * Omega_dir).astype(Q.dtype)
        U = _cayley_unitary(Omega, cayley_eye)
        Udag = jnp.conj(jnp.swapaxes(U, -1, -2))
        Ft_trial = Udag @ (Ft @ U)

        norm_trial = _comm_norm(Ft_trial, diff, w_norm, offdiag)
        E_trial = _frozen_F_energy(Ft_trial, p, w_norm)
        ok = jnp.logical_and(
            norm_trial <= accept_ratio * norm0,
            E_trial <= E0,
        )

        def accept_step(_):
            return (Q @ U, Ft_trial, jnp.bool_(True))

        def reject_step(_):
            return (Q_best, Ft_best, accepted)

        Q_best, Ft_best, accepted = lax.cond(ok, accept_step, reject_step, operand=None)

        tau = tau * shrink
        return (i + 1, tau, accepted, Q_best, Ft_best)

    init_state = (jnp.int32(0), tau0, jnp.bool_(False), Q, Ft)
    _, _, _, Q_out, Ft_out = lax.while_loop(cond, body, init_state)
    return Q_out, Ft_out


def _rotation_mask_from_block_sizes(block_sizes: tuple[int, ...], n: int, dtype) -> jax.Array:
    """Build binary (nb,nb) mask with 1s inside blocks and 0s between them."""
    mask = jnp.zeros((n, n), dtype=dtype)
    start = 0
    for size in block_sizes:
        stop = start + int(size)
        mask = mask.at[start:stop, start:stop].set(1.0)
        start = stop
    return mask


def _frozenF_inner_solve(
    Q: jax.Array,
    p: jax.Array,
    mu: jax.Array,
    *,
    F: jax.Array,
    Ft: jax.Array,
    w_norm: jax.Array,
    n_target_norm: jax.Array,
    T: float,
    inner_sweeps: int,
    q_sweeps: int,
    p_floor: float,
    denom_scale: float,
    max_rot: float,
    bt_tau0: float,
    bt_accept: float,
    bt_shrink: float,
    bt_max: int,
    mu_maxiter: int,
    mu_tol: float,
    rotation_block_sizes: tuple[int, ...] | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Cheap inner solve with F frozen:
      - update occupations exactly (FD) from diag(Ft) + mu
      - update Q by Cayley/Jacobi sweeps using commutator-based Newton direction
      - optional backtracking on commutator norm (all using Ft, no new exchange)

    Ft = Q†FQ should be precomputed by the caller to avoid redundant work.

    If rotation_block_sizes is given (e.g. (4,4,4,4) for a 16-band model with
    4 symmetry sectors), the rotation generator Ω is masked to be block-diagonal.
    This confines orbital mixing to within each sector, preventing the solver
    from drifting into symmetry-broken basins.
    """
    real_dtype = p.dtype
    p_floor = jnp.asarray(p_floor, dtype=real_dtype)
    max_rot = jnp.asarray(max_rot, dtype=real_dtype)
    Tj = jnp.asarray(T, dtype=real_dtype)
    denom_scale_j = jnp.asarray(denom_scale, dtype=real_dtype)
    tiny_eps = jnp.asarray(1e-30, dtype=real_dtype)
    tiny_gap = jnp.asarray(1e-12, dtype=real_dtype)
    tiny_abs_f = jnp.asarray(1e-16, dtype=real_dtype)
    clip_eps = jnp.asarray(1e-12, dtype=real_dtype)

    Ft0 = Ft

    # Hoist loop-invariant constant arrays (depend only on nb, not on Q/p state).
    n = Ft0.shape[-1]
    eye = jnp.eye(n, dtype=real_dtype)
    offdiag = 1.0 - eye
    orb_idx = jnp.arange(n, dtype=real_dtype)
    pair_sign = jnp.sign(orb_idx[None, :] - orb_idx[:, None])
    pair_sign = jnp.where(pair_sign == 0.0, 1.0, pair_sign)
    cayley_eye = jnp.eye(n, dtype=Q.dtype)

    # Build rotation block mask (None → unrestricted, else block-diagonal).
    rot_mask: jax.Array | None = None
    if rotation_block_sizes is not None:
        rot_mask = _rotation_mask_from_block_sizes(rotation_block_sizes, n, real_dtype)

    def inner_sweep(_, state):
        Q_s, Ft_s, p_s, mu_s = state

        # 1) occupations from diagonal energies
        eps = jnp.real(jnp.diagonal(Ft_s, axis1=-2, axis2=-1)).astype(real_dtype)
        mu_s = _solve_mu_fd_newton_bracket(
            eps, w_norm, n_target_norm, mu_s, T,
            maxiter=int(mu_maxiter), tol=float(mu_tol),
        )
        p_s = _occupations_from_eps(eps, mu_s, T).astype(real_dtype)

        diff = p_s[..., None, :] - p_s[..., :, None]
        abs_diff = jnp.abs(diff)
        occ_scale = abs_diff / (abs_diff + p_floor)

        # 2) Q sweeps
        def q_sweep(_, state2):
            Q_q, Ft_q = state2
            eps_q = jnp.real(jnp.diagonal(Ft_q, axis1=-2, axis2=-1)).astype(real_dtype)

            comm = diff * Ft_q

            gap = eps_q[..., :, None] - eps_q[..., None, :]
            # denom_scale is dimensionless; scale it by a typical energy scale
            eps_scale = jnp.sqrt(jnp.mean(eps_q ** 2) + tiny_eps)
            lam_gap = jnp.maximum(
                Tj,
                denom_scale_j * eps_scale,
            )
            denom = jnp.sqrt(gap ** 2 + lam_gap ** 2)

            # Jacobi fallback keeps motion in exactly degenerate occupancy subspaces.
            gap_ji = eps_q[..., None, :] - eps_q[..., :, None]
            safe_gap_ji = jnp.where(jnp.abs(gap_ji) < tiny_gap, tiny_gap * pair_sign, gap_ji)

            abs_F = jnp.abs(Ft_q)
            theta = 0.5 * jnp.arctan(2.0 * abs_F / safe_gap_ji)
            phase = jnp.where(
                abs_F < tiny_abs_f,
                jnp.asarray(1.0, dtype=Ft_q.dtype),
                Ft_q / abs_F,
            )
            Omega_jac = theta * phase * offdiag

            Omega_dir = -(comm / denom) * occ_scale + (1.0 - occ_scale) * Omega_jac
            Omega_dir = _skew_hermitian(Omega_dir)

            # Zero diagonal explicitly (numerical safety)
            Omega_dir = Omega_dir * (1.0 - eye)

            # Restrict rotations to within symmetry blocks (if requested).
            if rot_mask is not None:
                Omega_dir = Omega_dir * rot_mask

            # per-k clipping on generator norm
            gen_norm = jnp.sqrt(jnp.sum(jnp.abs(Omega_dir) ** 2, axis=(-2, -1)) + tiny_eps)
            clip_scale = jnp.minimum(1.0, max_rot / (gen_norm + clip_eps))
            Omega_dir = Omega_dir * clip_scale[..., None, None]

            Q_new, Ft_new = _backtracking_cayley_right(
                Q_q, Ft_q, Omega_dir,
                diff=diff,
                p=p_s,
                offdiag=offdiag,
                cayley_eye=cayley_eye,
                w_norm=w_norm,
                tau0=float(bt_tau0),
                accept_ratio=float(bt_accept),
                shrink=float(bt_shrink),
                max_backtrack=int(bt_max),
            )
            return (Q_new, Ft_new)

        Q_s, Ft_s = lax.fori_loop(0, int(q_sweeps), q_sweep, (Q_s, Ft_s))
        return (Q_s, Ft_s, p_s, mu_s)

    Q_out, _, p_out, mu_out = lax.fori_loop(0, int(inner_sweeps), inner_sweep, (Q, Ft0, p, mu))
    return Q_out, p_out, mu_out


# ---------------------------------------
# Main JIT-friendly outer optimizer
# ---------------------------------------


def variational_hartreefock_optimize(
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
    exchange_hermitian_channel_packing: bool,

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
    bt_tau0: float = 1.0,
    bt_accept: float = 0.999,
    bt_shrink: float = 0.5,
    bt_max: int = 5,
    mu_maxiter: int = 25,
    mu_tol: float = 1e-12,

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
    Option A: Frozen-F alternating minimization with exact occupations + Cayley/Jacobi rotations.

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

    # Cache last-built F if we finish on a no-update iteration (avoids a final extra build).
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
        jnp.asarray(jnp.nan, dtype=real_dtype),  # E_prev (for dE computation)
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
        # When e_tol > 0, also require energy convergence
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
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            exchange_block_specs=exchange_block_specs,
            exchange_check_offdiag=exchange_check_offdiag,
            exchange_offdiag_atol=exchange_offdiag_atol,
            exchange_offdiag_rtol=exchange_offdiag_rtol,
        )
        F = _project(F)
        E = _hf_energy(P, h=h, Sigma=Sigma, H=H_shift, weights_b=weights_b)
        E_real = jnp.real(E)

        # Energy change (inf on first iteration when E_prev is nan)
        dE = jnp.abs(E_real - E_prev)
        dE = jnp.where(jnp.isfinite(dE), dE, jnp.asarray(jnp.inf, dtype=real_dtype))

        # HF stationarity measure at current state (includes a degenerate-subspace gauge term)
        Ft = _fock_in_orbital_basis(Q, F)
        dC = _stationarity_norm(Ft, p, w_norm, float(p_floor), T, float(denom_scale))

        # Record history for the CURRENT state
        hE = hE.at[k].set(E_real)
        hC = hC.at[k].set(dC)
        hM = hM.at[k].set(mu)
        hdE = hdE.at[k].set(dE)

        # If converged, do not update (prevents post-convergence drift)
        need_update = dC > comm_tol_r

        def do_update(args):
            Q_u, p_u, mu_u = args
            Q1, p1, mu1 = _frozenF_inner_solve(
                Q_u, p_u, mu_u,
                F=F, Ft=Ft, w_norm=w_norm, n_target_norm=n_target_norm, T=T,
                inner_sweeps=int(inner_sweeps),
                q_sweeps=int(q_sweeps),
                p_floor=float(p_floor),
                denom_scale=float(denom_scale),
                max_rot=float(max_rot),
                bt_tau0=float(bt_tau0),
                bt_accept=float(bt_accept),
                bt_shrink=float(bt_shrink),
                bt_max=int(bt_max),
                mu_maxiter=int(mu_maxiter),
                mu_tol=float(mu_tol),
                rotation_block_sizes=rotation_block_sizes,
            )
            dP = _weighted_rms_vec(p1 - p_u, w_norm)
            return (Q1, p1, mu1, dP)

        def no_update(args):
            Q_u, p_u, mu_u = args
            return (Q_u, p_u, mu_u, jnp.asarray(0.0, dtype=real_dtype))

        Q_new, p_new, mu_new, dP = lax.cond(need_update, do_update, no_update, operand=(Q, p, mu))
        hP = hP.at[k].set(dP)

        # F is fresh only if we didn't change (Q,p) this iteration.
        fresh = jnp.logical_not(need_update)
        F_last = jnp.where(need_update, _F_last, F)
        E_last = jnp.where(need_update, _E_last, E_real)

        return (k + 1, Q_new, p_new, mu_new, dC, dP, dE, E_real,
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
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
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
# Convenience: initialize params from P0
# ---------------------------------------


def init_variational_params_from_density(
    P0: jax.Array,
    electrondensity0: float,
    *,
    weights_b: jax.Array,
    weight_sum: jax.Array,
    method: str = "identity",  # "identity" | "pdiag" | "uniform" | "eigh"
    occ_clip: float = 1e-6,
) -> VariationalHFParams:
    """
    Initialize (Q, p, mu) from an initial density P0.

    - identity/pdiag: Q=I, p from diag(P0)
    - uniform:        Q=I, p uniform matching N (most robust)
    - eigh:           one-time eigendecomposition of P0 for best warm start (optional)
    """
    P0 = _herm(jnp.asarray(P0))
    dtype = P0.dtype
    real_dtype = P0.real.dtype

    w2d = jnp.asarray(weights_b[..., 0, 0], dtype=real_dtype)
    weight_sum = jnp.asarray(weight_sum, dtype=real_dtype)
    w_norm = w2d / jnp.maximum(weight_sum, jnp.asarray(1e-30, dtype=real_dtype))
    n_target_norm = jnp.asarray(electrondensity0, dtype=real_dtype) / jnp.maximum(weight_sum, jnp.asarray(1e-30, dtype=real_dtype))

    nb = P0.shape[-1]
    Q0 = jnp.eye(nb, dtype=dtype)[None, None, ...]
    Q0 = jnp.broadcast_to(Q0, P0.shape)

    method_l = method.lower()
    if method_l in ("identity", "pdiag"):
        p0 = jnp.real(jnp.diagonal(P0, axis1=-2, axis2=-1)).astype(real_dtype)
        p0 = jnp.clip(p0, occ_clip, 1.0 - occ_clip)
    elif method_l == "uniform":
        pbar = jnp.clip(n_target_norm / jnp.asarray(nb, dtype=real_dtype), occ_clip, 1.0 - occ_clip)
        p0 = jnp.full(P0.shape[:-1], pbar, dtype=real_dtype)
    elif method_l == "eigh":
        evals, evecs = jnp.linalg.eigh(P0)
        p0 = jnp.clip(jnp.real(evals).astype(real_dtype), occ_clip, 1.0 - occ_clip)
        Q0 = evecs.astype(dtype)
    else:
        raise ValueError(f"Unknown init method {method!r}")

    # Enforce correct N by shifting in logit space (independent of F)
    logits = _logit(p0, eps=occ_clip)
    delta = _solve_delta_newton_bracket(logits, w_norm, n_target_norm, jnp.asarray(0.0, dtype=real_dtype), maxiter=25)
    p0 = jax.nn.sigmoid(logits + delta).astype(real_dtype)

    mu0 = jnp.asarray(0.0, dtype=real_dtype)
    return VariationalHFParams(Q=Q0, p=p0, mu=mu0)


def _adjust_params_particle_number(
    params: VariationalHFParams,
    electrondensity0: float,
    *,
    weights_b: jax.Array,
    weight_sum: jax.Array,
    occ_clip: float = 1e-6,
) -> VariationalHFParams:
    """Shift occupations in logit space so params match a new target N.

    Preserves Q (orbital basis) and mu exactly; only p is adjusted.
    """
    real_dtype = params.p.dtype
    w2d = jnp.asarray(weights_b[..., 0, 0], dtype=real_dtype)
    ws = jnp.maximum(jnp.asarray(weight_sum, dtype=real_dtype), 1e-30)
    w_norm = w2d / ws
    n_target_norm = jnp.asarray(electrondensity0, dtype=real_dtype) / ws

    p = jnp.clip(params.p, occ_clip, 1.0 - occ_clip)
    logits = _logit(p, eps=occ_clip)
    delta = _solve_delta_newton_bracket(
        logits, w_norm, n_target_norm,
        jnp.asarray(0.0, dtype=real_dtype), maxiter=25,
    )
    p_new = jax.nn.sigmoid(logits + delta).astype(real_dtype)
    return VariationalHFParams(Q=params.Q, p=p_new, mu=params.mu)


# ---------------------------------------
# Kernel-style wrapper (matches jax_hf API)
# ---------------------------------------
def jit_variational_hartreefock_iteration(hf_step):
    """
    Create a JIT-compiled frozen-F variational solver with the same style as jit_hartreefock_iteration.
    """
    compiled = jax.jit(
        variational_hartreefock_optimize,
        static_argnames=(
            "max_iter", "comm_tol", "p_tol", "e_tol",
            "inner_sweeps", "q_sweeps",
            "p_floor", "denom_scale", "max_rot",
            "bt_tau0", "bt_accept", "bt_shrink", "bt_max",
            "mu_maxiter", "mu_tol",
            "rotation_block_sizes",
            "include_hartree", "include_exchange",
            "exchange_hermitian_channel_packing",
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
