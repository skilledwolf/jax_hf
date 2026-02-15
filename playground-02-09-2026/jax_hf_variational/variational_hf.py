"""
Variational finite-T Hartree-Fock (HF) solver on a uniform k-grid.

Key design points:
- Variational in density matrix P: minimize A[P] = E[P] - T S[P] (Mermin free energy).
- Hard constraints via parameterization:
    P(k) = Q(k) diag(occ(k)) Q(k)^H
  with Q(k) unitary and occ in (0,1).
- Fixed electron number enforced via a scalar shift delta in logits:
    occ = sigmoid(logits + delta), delta chosen so weighted sum equals N_e.
- No eigen-decompositions in the optimization loop.
- Unitary optimization via Cayley retraction + "Riemannian Adam"-style preconditioning.
- Optional: pinning field ramp, tether ramp, symmetry projection (true group average).

IMPORTANT correctness note:
FFT acceleration assumes a *uniform full k-grid* (translation-invariant quadrature).
If you use wedge-reduced meshes with nonuniform weights, the convolution is not valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn, ifftshift

from .utils import hermitize, fermidirac, find_chemical_potential, selfenergy_fft, normalize_block_specs


ProjectFn = Callable[[jax.Array], jax.Array]


# -----------------------------
# Settings / Results
# -----------------------------

@dataclass(frozen=True)
class VariationalHFSettings:
    # Iterations
    max_steps: int = 400

    # Optimizer hyperparams
    lr_Q: float = 1e-2
    lr_logits: float = 3e-2
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    # Clip *preconditioned* per-k steps (post-Adam)
    clip_Q: float | None = 1.0
    clip_logits: float | None = 1.0

    # Optional logits magnitude clip (helps for very small T / near-idempotent occ)
    logits_clip_value: float | None = 60.0

    # Stopping
    tol_comm: float | None = 2e-3
    tol_grad: float = 5e-2
    tol_dA_rel: float | None = None  # relative free-energy change tolerance

    # Electron-number constraint shift delta (Newton)
    occ_shift_iters: int = 8
    occ_shift_step_clip: float | None = 2.0

    # Temperature floor for numerical stability (smearing)
    # If user requests T=0, this provides a controlled small smearing; A is then
    # computed at T_eff (to stay variationally consistent with gradients).
    T_min: float = 1e-6

    # Optional pinning-field schedule: h -> h + pin_scale(t) * pin_field
    pin_strength: float = 0.0
    pin_strength_final: float = 0.0
    pin_ramp_steps: int = 100

    # Optional tether penalty schedule: 0.5*kappa(t)*||P - P_ref||^2
    tether_strength: float = 0.0
    tether_strength_final: float = 0.0
    tether_ramp_steps: int = 0

    # Optional periodic QR re-orthonormalization (Cayley should keep Q unitary).
    reorth_every: int = 0

    # Exchange block acceleration options (passed to utils.selfenergy_fft)
    exchange_block_specs: Any | None = None
    exchange_check_offdiag: bool = False  # default off for performance
    exchange_offdiag_atol: float = 1e-12
    exchange_offdiag_rtol: float = 0.0

    # FFT uniform-weight check
    require_uniform_weights: bool = True
    uniform_weight_rtol: float = 1e-8
    uniform_weight_atol: float = 1e-12

    # Time-reversal k-index convention:
    # - "mod": (-i mod nk1, -j mod nk2) (recommended for FFT-order grids)
    # - "flip": (nk1-1-i, nk2-1-j) (matches old code; only correct for some layouts)
    time_reversal_k_convention: str = "mod"

    # Orbital-rotation (Q) update preconditioner: scale Omega_{ij} by an
    # approximate inverse Hessian using "orbital energy" denominators.
    # If enabled: Omega_{ij} <- Omega_{ij} / (|ε_i - ε_j| + lambda).
    q_gap_precond: bool = False
    q_gap_precond_lambda: float = 1e-2
    q_gap_precond_start_step: int = 20
    q_gap_precond_comm_max: float | None = 1e-2
    q_gap_precond_occ_floor: float = 5e-2

    # Optional quasi-Newton updates (no eigendecomps) intended to reduce the
    # required number of outer iterations.
    #
    # For logits: take a damped Newton-like step in the dimensionless variables
    # η = logits + delta using the residual (g_occ - mu_proj):
    #   η <- η - α * (g_occ - mu_proj) / T_eff
    logits_newton_step: bool = True
    logits_newton_damping: float = 0.7
    logits_newton_occ_floor: float = 1e-3  # suppress updates when occ is nearly idempotent

    # For Q: take a preconditioned step that approximately cancels the
    # (occ_j - occ_i) factor and divides by a Rayleigh-quotient gap.
    q_newton_step: bool = True
    q_newton_step_size: float = 0.7
    q_newton_gap_lambda: float = 1e-2
    q_newton_occ_floor: float = 5e-2


class VariationalHFResult(NamedTuple):
    P: jax.Array
    F: jax.Array
    E: jax.Array
    A: jax.Array
    S: jax.Array
    mu: jax.Array
    Q: jax.Array
    logits: jax.Array
    delta: jax.Array
    T_eff: jax.Array
    n_steps: jax.Array
    converged: jax.Array


# -----------------------------
# Linear algebra helpers
# -----------------------------

def _skew_hermitian(A: jax.Array) -> jax.Array:
    return 0.5 * (A - jnp.conj(jnp.swapaxes(A, -1, -2)))


def _project_unitary_qr(Q: jax.Array) -> jax.Array:
    Q = jnp.asarray(Q)
    Qp, R = jnp.linalg.qr(Q)
    diag = jnp.diagonal(R, axis1=-2, axis2=-1)
    phase = diag / (jnp.abs(diag) + jnp.asarray(1e-12, dtype=diag.dtype))
    return Qp * jnp.conj(phase)[..., None, :]


def _cayley_update(Q: jax.Array, Omega: jax.Array) -> jax.Array:
    """Unitary Cayley update: Q <- Q (I - Omega/2)^-1 (I + Omega/2)."""
    n = Q.shape[-1]
    I = jnp.eye(n, dtype=Q.dtype)
    A = I - 0.5 * Omega
    B = I + 0.5 * Omega
    U = jnp.linalg.solve(A, B)
    return Q @ U


def _entropy_from_occ(w2d: jax.Array, occ: jax.Array) -> jax.Array:
    real_dtype = occ.dtype
    eps = jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
    one = jnp.asarray(1.0, dtype=real_dtype)
    o = jnp.clip(occ, eps, one - eps)
    return -jnp.sum(w2d[..., None] * (o * jnp.log(o) + (one - o) * jnp.log(one - o)))


def _schedule_linear(step: jax.Array, *, start: float, end: float, ramp_steps: int, dtype) -> jax.Array:
    """JIT-safe linear schedule. If ramp_steps<=0: constant = start."""
    start = jnp.asarray(float(start), dtype=dtype)
    end = jnp.asarray(float(end), dtype=dtype)
    rs = int(ramp_steps)
    if rs <= 0:
        return start
    t = jnp.minimum(jnp.asarray(step, dtype=dtype) / jnp.asarray(float(rs), dtype=dtype), jnp.asarray(1.0, dtype=dtype))
    return (1.0 - t) * start + t * end


def _wtd_trace(weights_b: jax.Array, A: jax.Array, B: jax.Array) -> jax.Array:
    # sum_k w_k Re Tr[A(k) B(k)]
    return jnp.sum(jnp.real(jnp.einsum("...ij,...ji->...", weights_b * A, B)))


def _wtd_mean_fro_norm_sq(X: jax.Array, *, w2d: jax.Array, weight_sum: jax.Array) -> jax.Array:
    per_k = jnp.sum(jnp.abs(X) ** 2, axis=(-2, -1))
    real_dtype = jnp.zeros((), dtype=per_k.dtype).real.dtype
    denom = jnp.maximum(jnp.asarray(weight_sum, dtype=real_dtype), jnp.asarray(1e-30, dtype=real_dtype))
    return jnp.sum(jnp.asarray(w2d, dtype=real_dtype) * per_k) / denom


def _commutator_mean_sq(F: jax.Array, P: jax.Array, *, w2d: jax.Array, weight_sum: jax.Array) -> jax.Array:
    R = F @ P - P @ F
    per_k = jnp.sum(jnp.abs(R) ** 2, axis=(-2, -1))
    real_dtype = jnp.zeros((), dtype=per_k.dtype).real.dtype
    denom = jnp.maximum(jnp.asarray(weight_sum, dtype=real_dtype), jnp.asarray(1e-30, dtype=real_dtype))
    return jnp.sum(jnp.asarray(w2d, dtype=real_dtype) * per_k) / denom


def _commutator_rms(F: jax.Array, P: jax.Array, *, w2d: jax.Array, weight_sum: jax.Array) -> jax.Array:
    comm2 = _commutator_mean_sq(F, P, w2d=w2d, weight_sum=weight_sum)
    real_dtype = jnp.zeros((), dtype=comm2.dtype).real.dtype
    return jnp.sqrt(comm2 + jnp.asarray(1e-30, dtype=real_dtype))


def _occ_shift_newton(
    logits: jax.Array,
    w2d: jax.Array,
    n_electrons: jax.Array,
    delta0: jax.Array,
    *,
    iters: int,
    step_clip: float | None,
) -> jax.Array:
    """Solve scalar delta so sum_k w_k sum_n sigmoid(logits+delta) == n_electrons."""
    logits = jnp.asarray(logits)
    real_dtype = logits.dtype
    w = jnp.asarray(w2d, dtype=real_dtype)[..., None]
    n_target = jnp.asarray(n_electrons, dtype=real_dtype)
    delta0 = jnp.asarray(delta0, dtype=real_dtype)
    tiny = jnp.asarray(1e-18, dtype=real_dtype)

    def body(_, delta):
        occ = jax.nn.sigmoid(logits + delta)
        N = jnp.sum(w * occ).astype(real_dtype)
        dN = jnp.sum(w * (occ * (1.0 - occ))).astype(real_dtype)
        step = (n_target - N) / (dN + tiny)
        if step_clip is not None:
            sc = jnp.asarray(float(step_clip), dtype=real_dtype)
            step = jnp.clip(step, -sc, sc)
        # If dN is extremely small, Newton can go wild; clipping helps, but we
        # also softly damp in that regime.
        return delta + step

    return jax.lax.fori_loop(0, int(iters), body, delta0)


# -----------------------------
# Symmetry projection (group average + optional time reversal)
# -----------------------------

@dataclass(frozen=True)
class SymmetryProjector:
    unitary_group: jax.Array | None = None  # (ng, d, d)
    time_reversal_U: jax.Array | None = None  # (d, d)
    k_convention: str = "mod"  # "mod" or "flip"

    def __call__(self, A: jax.Array) -> jax.Array:
        out = A
        if self.unitary_group is not None:
            out = self._avg_unitary_conj(out)
        if self.time_reversal_U is not None:
            out = self._avg_time_reversal(out)
        return out

    def _avg_unitary_conj(self, A: jax.Array) -> jax.Array:
        G = jnp.asarray(self.unitary_group)
        if G.ndim != 3:
            raise ValueError("unitary_group must have shape (ng, d, d).")
        ng = int(G.shape[0])

        def body(i, acc):
            g = G[i]
            gH = jnp.conj(jnp.swapaxes(g, -1, -2))
            return acc + (g @ A) @ gH

        acc0 = jnp.zeros_like(A)
        acc = jax.lax.fori_loop(0, ng, body, acc0)
        return acc / jnp.asarray(float(ng), dtype=A.dtype)

    def _avg_time_reversal(self, A: jax.Array) -> jax.Array:
        U = jnp.asarray(self.time_reversal_U)
        UH = jnp.conj(jnp.swapaxes(U, -1, -2))

        nk1, nk2 = int(A.shape[0]), int(A.shape[1])

        if self.k_convention == "flip":
            A_neg = jnp.flip(A, axis=(0, 1))
        elif self.k_convention == "mod":
            i = (-(jnp.arange(nk1, dtype=jnp.int32))) % nk1
            j = (-(jnp.arange(nk2, dtype=jnp.int32))) % nk2
            A_neg = A[i[:, None], j[None, :], ...]
        else:
            raise ValueError("time_reversal_k_convention must be 'mod' or 'flip'.")

        A_tr = U @ jnp.conj(A_neg) @ UH
        return 0.5 * (A + A_tr)


def make_project_fn(
    *,
    unitary_group: jax.Array | None = None,
    time_reversal_U: jax.Array | None = None,
    time_reversal_k_convention: str = "mod",
) -> ProjectFn:
    proj = SymmetryProjector(
        unitary_group=unitary_group,
        time_reversal_U=time_reversal_U,
        k_convention=time_reversal_k_convention,
    )
    return lambda A: proj(A)


# -----------------------------
# Hartree term (orbital-resolved density-density coupling)
# -----------------------------

def _hartree_from_density_fft(
    VR_h: jax.Array,
    P_diff: jax.Array,
    *,
    weights_b: jax.Array,
) -> jax.Array:
    """
    Compute Hartree self-energy Σ_H(k), assuming a density-density interaction.

    Let rho_j(k) = (P_diff)_{jj}(k) (real).
    Interaction kernel V_h(q) may be:
      - scalar: VR_h[..., 0, 0] (shape (nk1,nk2,1,1))
      - orbital coupling matrix: VR_h[..., i, j] (shape (nk1,nk2,d,d))

    We compute:
      phi_i(k) = sum_j [ V_ij * rho_j ](k)   (convolution on the k-grid)
      Σ_H(k) = diag(phi(k))
    """
    # Diagonal densities (nk1,nk2,d), real
    rho = jnp.real(jnp.diagonal(P_diff, axis1=-2, axis2=-1))
    # FFT in k-space
    rho_fft = fftn(rho, axes=(0, 1))

    if VR_h.shape[-2:] == (1, 1):
        # scalar coupling: all orbitals see the same potential from total density
        rho_tot_fft = jnp.sum(rho_fft, axis=-1)  # (nk1,nk2)
        phi_fft = rho_tot_fft * VR_h[..., 0, 0]  # (nk1,nk2)
        phi = ifftshift(ifftn(phi_fft, axes=(0, 1)), axes=(0, 1))
        d = int(P_diff.shape[-1])
        I = jnp.eye(d, dtype=P_diff.dtype)
        return (jnp.real(phi).astype(P_diff.dtype))[..., None, None] * I

    # general orbital coupling: phi_fft[..., i] = sum_j VR_h[..., i, j] * rho_fft[..., j]
    phi_fft = jnp.einsum("...ij,...j->...i", VR_h, rho_fft)  # (nk1,nk2,d)
    phi = ifftshift(ifftn(phi_fft, axes=(0, 1)), axes=(0, 1))  # (nk1,nk2,d)
    phi_r = jnp.real(phi).astype(P_diff.dtype)

    d = int(P_diff.shape[-1])
    I = jnp.eye(d, dtype=P_diff.dtype)
    return jnp.einsum("...i,ij->...ij", phi_r, I)


# -----------------------------
# Main solver
# -----------------------------

class VariationalHF:
    """
    Variational HF solver with:
    - Exchange (Fock) via FFT convolution on P_diff = P - refP
    - Optional Hartree via FFT convolution on diagonal density of P_diff
    """

    def __init__(
        self,
        *,
        h: jax.Array,                    # (nk1, nk2, d, d)
        weights: jax.Array,              # (nk1, nk2)
        V_exchange_q: jax.Array,         # (nk1,nk2) or (nk1,nk2,1,1) or (nk1,nk2,d,d)
        V_hartree_q: jax.Array | None = None,  # same shape conventions; if None, Hartree is disabled (zero)
        reference_density: jax.Array | None = None,
        project_fn: ProjectFn | None = None,
        settings: VariationalHFSettings | None = None,
    ):
        self.settings = settings or VariationalHFSettings()

        self.h = hermitize(jnp.asarray(h))
        self.w2d = jnp.asarray(weights, dtype=self.h.real.dtype)
        self.weights_b = self.w2d[..., None, None]
        self.weight_sum = jnp.sum(self.w2d)

        # Enforce uniform weights for FFT-accelerated convolution (correctness)
        if self.settings.require_uniform_weights:
            self._check_uniform_weights(self.w2d)

        nk_tot = int(self.w2d.shape[0] * self.w2d.shape[1])
        w_mean = self.weight_sum / jnp.asarray(max(nk_tot, 1), dtype=self.h.real.dtype)
        self.w2d_grad = self.w2d / jnp.maximum(w_mean, jnp.asarray(1e-30, dtype=self.h.real.dtype))
        self.weights_b_grad = self.w2d_grad[..., None, None]

        self.project = project_fn or (lambda A: A)

        self.refP = (
            self.project(hermitize(jnp.asarray(reference_density, dtype=self.h.dtype)))
            if reference_density is not None
            else jnp.zeros_like(self.h)
        )

        # Normalize exchange block specs once (hashable / canonical)
        self.exchange_block_specs = normalize_block_specs(self.settings.exchange_block_specs)

        # Parse interaction kernels
        Vx = jnp.asarray(V_exchange_q)
        if Vx.ndim == 2:
            Vx = Vx[..., None, None]
        Vx = jnp.asarray(Vx, dtype=self.h.dtype)

        if V_hartree_q is None:
            # Default: exchange-only (no Hartree).
            self._hartree_enabled = False
            Vh = jnp.zeros_like(Vx)
        else:
            self._hartree_enabled = True
            Vh = jnp.asarray(V_hartree_q)
            if Vh.ndim == 2:
                Vh = Vh[..., None, None]
            Vh = jnp.asarray(Vh, dtype=self.h.dtype)

        # Precompute FFT of weights*V(q) for convolution.
        # NOTE: correctness assumes weights are uniform (checked above).
        self.VR_x = fftn(self.weights_b * Vx, axes=(0, 1))
        self.VR_h = fftn(self.weights_b * Vh, axes=(0, 1)) if self._hartree_enabled else None

        # JIT step + loop
        self._run = jax.jit(self._run_impl)

    def _check_uniform_weights(self, w2d: jax.Array) -> None:
        # One-time host check; if you really want to bypass, disable require_uniform_weights.
        import numpy as np
        w = np.asarray(jax.device_get(w2d))
        w_mean = float(np.mean(w))
        if w_mean == 0.0:
            raise ValueError("weights mean is zero; FFT convolution is not meaningful.")
        max_dev = float(np.max(np.abs(w - w_mean)))
        if max_dev > self.settings.uniform_weight_atol + self.settings.uniform_weight_rtol * abs(w_mean):
            raise ValueError(
                "Non-uniform k-point weights detected. FFT-based convolution assumes a uniform full grid.\n"
                "Use a full uniform mesh (weights constant), or disable require_uniform_weights if you are\n"
                "absolutely sure your convolution remains valid (rare)."
            )

    def init_from_fock(self, *, F0: jax.Array, n_electrons: float, T_eff: float) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Initialize (Q, logits, delta) by diagonalizing a Fock-like matrix F0 (one-time)."""
        eps, U = jnp.linalg.eigh(hermitize(F0))
        mu = find_chemical_potential(eps, self.w2d, n_electrons=float(n_electrons), T=float(T_eff))
        occ = fermidirac(eps - mu, float(T_eff)).astype(self.h.real.dtype)

        eps_occ = jnp.asarray(jnp.finfo(occ.dtype).eps, dtype=occ.dtype)
        one = jnp.asarray(1.0, dtype=occ.dtype)
        occ = jnp.clip(occ, eps_occ, one - eps_occ)
        logits = jnp.log(occ) - jnp.log(one - occ)
        Q0 = _project_unitary_qr(U)
        delta0 = jnp.asarray(0.0, dtype=occ.dtype)
        return Q0, logits, delta0

    # -----------------------------
    # Core evaluation pieces
    # -----------------------------

    def _sigma_exchange(self, P_diff: jax.Array) -> jax.Array:
        """Exchange Σ_X(P_diff) with correct ref-subtraction convention."""
        # Fast path if no block specs: compute directly here.
        if self.exchange_block_specs is None:
            P_fft = fftn(P_diff, axes=(0, 1))
            sig = -ifftn(P_fft * self.VR_x, axes=(0, 1))
            sig = ifftshift(sig, axes=(0, 1))
            return self.project(hermitize(sig))

        # Block-spec path (uses your utils implementation)
        sig = selfenergy_fft(
            self.VR_x,
            P_diff,
            block_specs=self.exchange_block_specs,
            check_offdiag=self.settings.exchange_check_offdiag,
            offdiag_atol=self.settings.exchange_offdiag_atol,
            offdiag_rtol=self.settings.exchange_offdiag_rtol,
        )
        return self.project(hermitize(sig))

    def _sigma_hartree(self, P_diff: jax.Array) -> jax.Array:
        if not self._hartree_enabled:
            return jnp.zeros_like(P_diff)
        assert self.VR_h is not None
        sig = _hartree_from_density_fft(self.VR_h, P_diff, weights_b=self.weights_b)
        return self.project(hermitize(sig))

    def _build_P(self, Q: jax.Array, occ: jax.Array) -> jax.Array:
        # P = Q diag(occ) Q^H, built without explicit diag
        Qocc = Q * occ[..., None, :]
        P = Qocc @ jnp.conj(jnp.swapaxes(Q, -1, -2))
        return self.project(hermitize(P))

    # -----------------------------
    # The jitted solve loop
    # -----------------------------

    class _State(NamedTuple):
        Q: jax.Array
        logits: jax.Array
        delta: jax.Array

        mL: jax.Array
        vL: jax.Array
        tL: jax.Array  # int32

        mQ: jax.Array
        vQ: jax.Array  # (nk1,nk2) scalar second moment
        tQ: jax.Array  # int32

        step: jax.Array  # int32
        A_prev: jax.Array
        done: jax.Array  # bool

        # Best-state tracking (for robust branch selection)
        best_comm_all: jax.Array
        best_Q_all: jax.Array
        best_logits_all: jax.Array
        best_delta_all: jax.Array

        best_comm_post: jax.Array
        best_Q_post: jax.Array
        best_logits_post: jax.Array
        best_delta_post: jax.Array

        best_A: jax.Array
        best_Q_A: jax.Array
        best_logits_A: jax.Array
        best_delta_A: jax.Array

    def _run_impl(
        self,
        state0: _State,
        *,
        n_e: jax.Array,
        T_eff: jax.Array,
        pin_field: jax.Array,
        tether_P: jax.Array,
        select_after: jax.Array,
        select_comm: jax.Array,
    ) -> _State:
        s = self.settings

        has_pin = (s.pin_strength != 0.0) or (s.pin_strength_final != 0.0)
        has_tether = (s.tether_strength != 0.0) or (s.tether_strength_final != 0.0)

        tether_P = self.project(hermitize(jnp.asarray(tether_P, dtype=self.h.dtype))) if has_tether else tether_P
        pin_field = hermitize(jnp.asarray(pin_field, dtype=self.h.dtype)) if has_pin else pin_field

        def cond(st: VariationalHF._State) -> jax.Array:
            return jnp.logical_and(st.step < jnp.asarray(int(s.max_steps), dtype=jnp.int32), jnp.logical_not(st.done))

        def body(st: VariationalHF._State) -> VariationalHF._State:
            step_i = st.step

            # schedules (all JIT-safe)
            pin_scale = _schedule_linear(
                step_i, start=s.pin_strength, end=s.pin_strength_final, ramp_steps=s.pin_ramp_steps, dtype=self.h.real.dtype
            ) if has_pin else jnp.asarray(0.0, dtype=self.h.real.dtype)

            tether_kappa = _schedule_linear(
                step_i,
                start=s.tether_strength,
                end=s.tether_strength_final,
                ramp_steps=s.tether_ramp_steps,
                dtype=self.h.real.dtype,
            ) if has_tether else jnp.asarray(0.0, dtype=self.h.real.dtype)

            # electron count shift
            delta = _occ_shift_newton(
                st.logits,
                self.w2d,
                n_e,
                jax.lax.stop_gradient(st.delta),
                iters=int(s.occ_shift_iters),
                step_clip=s.occ_shift_step_clip,
            )
            occ = jax.nn.sigmoid(st.logits + delta)

            # build P and P_diff
            P = self._build_P(st.Q, occ)
            P_diff = hermitize(P - self.refP)

            # one-body (pin if active)
            h_eff = hermitize(self.h + pin_scale * pin_field) if has_pin else self.h

            # self-energies (consistent ref-subtraction)
            # If no exchange block specs, we already compute exchange via direct FFT.
            Sigma_x = self._sigma_exchange(P_diff)
            Sigma_h = self._sigma_hartree(P_diff)

            F = hermitize(h_eff + Sigma_h + Sigma_x)

            comm = _commutator_rms(F, P, w2d=self.w2d, weight_sum=self.weight_sum)

            # free energy components (variationally consistent with P_diff convention)
            E1 = _wtd_trace(self.weights_b, h_eff, P)
            Eh = 0.5 * _wtd_trace(self.weights_b, Sigma_h, P_diff)
            Ex = 0.5 * _wtd_trace(self.weights_b, Sigma_x, P_diff)
            E = E1 + Eh + Ex

            S_ent = _entropy_from_occ(self.w2d, occ)
            A = E - T_eff * S_ent

            # Optional tether stabilizer (continuation)
            if has_tether:
                P_ref = jax.lax.stop_gradient(tether_P)
                D = hermitize(P - P_ref)
                tether2 = _wtd_mean_fro_norm_sq(D, w2d=self.w2d, weight_sum=self.weight_sum)
                tether_loss = 0.5 * tether_kappa * tether2
            else:
                D = jnp.zeros_like(P)
                tether_loss = jnp.asarray(0.0, dtype=self.h.real.dtype)

            # --- Best-state selection snapshots (current state, before the update) ---
            in_window = step_i >= jnp.asarray(select_after, dtype=jnp.int32)
            better_all = comm < st.best_comm_all
            better_post = jnp.logical_and(in_window, comm < st.best_comm_post)
            acceptable = jnp.logical_and(in_window, comm <= jnp.asarray(select_comm, dtype=self.h.real.dtype))
            better_A = jnp.logical_and(acceptable, A < st.best_A)

            def pick(b, new, old):
                return jax.lax.cond(b, lambda _: new, lambda _: old, operand=None)

            best_comm_all = jnp.where(better_all, comm, st.best_comm_all)
            best_Q_all = pick(better_all, st.Q, st.best_Q_all)
            best_logits_all = pick(better_all, st.logits, st.best_logits_all)
            best_delta_all = pick(better_all, delta, st.best_delta_all)

            best_comm_post = jnp.where(better_post, comm, st.best_comm_post)
            best_Q_post = pick(better_post, st.Q, st.best_Q_post)
            best_logits_post = pick(better_post, st.logits, st.best_logits_post)
            best_delta_post = pick(better_post, delta, st.best_delta_post)

            best_A = jnp.where(better_A, A, st.best_A)
            best_Q_A = pick(better_A, st.Q, st.best_Q_A)
            best_logits_A = pick(better_A, st.logits, st.best_logits_A)
            best_delta_A = pick(better_A, delta, st.best_delta_A)

            # --- Analytic gradients (no autodiff) ---
            # Fuse all "Fock-like" contributions into a single matrix for Q-basis transform.
            # For tether: d/dP (0.5*kappa*||P-Pref||^2) = kappa*(P-Pref) / weight_sum (in our mean-square normalization).
            if has_tether:
                tether_coeff = tether_kappa / jnp.maximum(self.weight_sum, jnp.asarray(1e-30, dtype=self.h.real.dtype))
                F_total = F + tether_coeff * D
            else:
                F_total = F

            Q = st.Q
            QH = jnp.conj(jnp.swapaxes(Q, -1, -2))
            M_total = QH @ F_total @ Q  # (nk1,nk2,d,d)
            diag_M = jnp.real(jnp.diagonal(M_total, axis1=-2, axis2=-1)).astype(self.h.real.dtype)

            # Occupation gradient: g_occ = diag(M) + T * logit(occ)
            # and logit(occ) = logits + delta (since occ = sigmoid(logits+delta))
            g_occ = diag_M + jnp.asarray(T_eff, dtype=diag_M.dtype) * (st.logits + delta)

            d_occ = occ * (1.0 - occ)

            # Enforce fixed-N constraint by projecting out the component along d_occ:
            w = self.w2d[..., None].astype(diag_M.dtype)
            tiny = jnp.asarray(1e-18, dtype=diag_M.dtype)
            mu_proj = jnp.sum(w * g_occ * d_occ) / (jnp.sum(w * d_occ) + tiny)

            # Weighted constrained gradient for logits
            w_grad = self.w2d_grad[..., None].astype(diag_M.dtype)
            g_logits = w_grad * (g_occ - mu_proj) * d_occ

            # Unitary gradient generator:
            # Omega = skew( w_k * (M_total * (occ_j - occ_i)) )
            diff = occ[..., None, :] - occ[..., :, None]
            Omega_grad = _skew_hermitian(self.weights_b_grad * (M_total * diff))
            Omega_update = Omega_grad
            if s.q_gap_precond and (not s.q_newton_step):
                # Gap (Jacobi) preconditioner: larger steps for near-degenerate
                # rotations, damped by lambda to avoid instability.
                gap = diag_M[..., :, None] - diag_M[..., None, :]
                lam_user = jnp.asarray(float(s.q_gap_precond_lambda), dtype=diag_M.dtype)
                lam_user = jnp.maximum(lam_user, jnp.asarray(1e-12, dtype=diag_M.dtype))
                # Use at least O(T) damping to avoid huge steps when the Rayleigh
                # quotient "gaps" are accidentally tiny early in the solve.
                lam = jnp.maximum(lam_user, jnp.asarray(T_eff, dtype=diag_M.dtype))
                occ_floor = jnp.asarray(float(s.q_gap_precond_occ_floor), dtype=diag_M.dtype)
                occ_floor = jnp.maximum(occ_floor, jnp.asarray(0.0, dtype=diag_M.dtype))
                diff_scale = jnp.maximum(jnp.abs(diff).astype(diag_M.dtype), occ_floor)
                denom = (jnp.abs(gap) + lam) * diff_scale
                Omega_gap = _skew_hermitian(Omega_grad / denom)

                # Smoothly turn on the preconditioner (avoid hard toggling).
                alpha = jnp.asarray(1.0, dtype=self.h.real.dtype)
                if s.q_gap_precond_comm_max is not None:
                    comm_max = jnp.asarray(float(s.q_gap_precond_comm_max), dtype=self.h.real.dtype)
                    alpha = jnp.clip((comm_max - comm) / jnp.maximum(comm_max, jnp.asarray(1e-30, dtype=self.h.real.dtype)), 0.0, 1.0)
                start_step = int(s.q_gap_precond_start_step)
                if start_step > 0:
                    alpha = alpha * (step_i >= jnp.asarray(start_step, dtype=jnp.int32)).astype(self.h.real.dtype)

                Omega_update = _skew_hermitian((1.0 - alpha) * Omega_grad + alpha * Omega_gap)

            # Gradient norms (RMS per k-point) for stopping
            per_k_Q = jnp.sum(jnp.abs(Omega_grad) ** 2, axis=(-2, -1))
            gnorm_Q = jnp.sqrt(jnp.mean(per_k_Q))

            per_k_L = jnp.sum(g_logits ** 2, axis=-1)
            gnorm_L = jnp.sqrt(jnp.mean(per_k_L))

            # --- Logits update ---
            if s.logits_newton_step:
                # Damped Newton-like residual update in η = logits + delta:
                #   η <- η - α * (g_occ - mu_proj) / T_eff
                #
                # With mu_proj defined via the fixed-N projection, this update is
                # (to first order) electron-number preserving.
                damp = jnp.asarray(float(s.logits_newton_damping), dtype=self.h.real.dtype)
                damp = jnp.clip(damp, jnp.asarray(0.0, dtype=self.h.real.dtype), jnp.asarray(1.0, dtype=self.h.real.dtype))

                Tden = jnp.maximum(jnp.asarray(T_eff, dtype=self.h.real.dtype), jnp.asarray(1e-12, dtype=self.h.real.dtype))
                eta_curr = (st.logits + delta).astype(self.h.real.dtype)
                resid = (g_occ - mu_proj).astype(self.h.real.dtype)

                # Suppress updates for nearly-idempotent occupations (robust at small T).
                occ_floor = jnp.asarray(float(s.logits_newton_occ_floor), dtype=self.h.real.dtype)
                occ_floor = jnp.maximum(occ_floor, jnp.asarray(0.0, dtype=self.h.real.dtype))
                d_scale = d_occ / (d_occ + occ_floor)

                eta_next = eta_curr - damp * d_scale * (resid / Tden)
                logits_next = (eta_next - delta).astype(st.logits.dtype)

                # Clip per-k step (consistent with clip_logits semantics).
                if s.clip_logits is not None:
                    c_eff = jnp.asarray(float(s.clip_logits), dtype=self.h.real.dtype)
                    dL = (logits_next - st.logits).astype(self.h.real.dtype)
                    per_k_step = jnp.sqrt(jnp.sum(dL ** 2, axis=-1) + jnp.asarray(1e-30, dtype=self.h.real.dtype))
                    scale = jnp.minimum(1.0, c_eff / (per_k_step + jnp.asarray(1e-12, dtype=self.h.real.dtype)))
                    logits_next = (st.logits + dL * scale[..., None]).astype(st.logits.dtype)

                if s.logits_clip_value is not None:
                    Lc = jnp.asarray(float(s.logits_clip_value), dtype=logits_next.dtype)
                    logits_next = jnp.clip(logits_next, -Lc, Lc)

                tL = st.tL
                mL = st.mL
                vL = st.vL
            else:
                # --- Adam on logits ---
                tL = st.tL + jnp.asarray(1, dtype=jnp.int32)
                mL = s.b1 * st.mL + (1.0 - s.b1) * g_logits
                vL = s.b2 * st.vL + (1.0 - s.b2) * (g_logits ** 2)

                b1t_L = jnp.asarray(s.b1, dtype=mL.dtype) ** tL
                b2t_L = jnp.asarray(s.b2, dtype=vL.dtype) ** tL
                mLh = mL / (1.0 - b1t_L)
                vLh = vL / (1.0 - b2t_L)
                stepL = mLh / (jnp.sqrt(vLh) + jnp.asarray(s.eps, dtype=vLh.dtype))

                if s.clip_logits is not None:
                    c_eff = jnp.asarray(float(s.clip_logits), dtype=self.h.real.dtype)
                    per_k_step = jnp.sqrt(jnp.sum(stepL ** 2, axis=-1) + jnp.asarray(1e-30, dtype=self.h.real.dtype))
                    scale = jnp.minimum(1.0, c_eff / (per_k_step + jnp.asarray(1e-12, dtype=self.h.real.dtype)))
                    stepL = stepL * scale[..., None]

                logits_next = st.logits - jnp.asarray(s.lr_logits, dtype=st.logits.dtype) * stepL
                if s.logits_clip_value is not None:
                    Lc = jnp.asarray(float(s.logits_clip_value), dtype=logits_next.dtype)
                    logits_next = jnp.clip(logits_next, -Lc, Lc)

            # --- Q (unitary) update ---
            if s.q_newton_step:
                # Damped Newton-like orbital rotation step (no eigendecomps):
                # Omega_newton ~ M_ij / (ε_i - ε_j), with a soft suppression for
                # nearly equal occupations (Δocc ~ 0).
                gap = diag_M[..., :, None] - diag_M[..., None, :]
                lam_user = jnp.asarray(float(s.q_newton_gap_lambda), dtype=gap.dtype)
                lam_user = jnp.maximum(lam_user, jnp.asarray(1e-12, dtype=gap.dtype))
                lam = jnp.maximum(lam_user, jnp.asarray(T_eff, dtype=gap.dtype))
                sign_gap = jnp.where(gap >= 0.0, jnp.asarray(1.0, dtype=gap.dtype), jnp.asarray(-1.0, dtype=gap.dtype))
                denom = gap + sign_gap * lam

                diff_abs = jnp.abs(diff).astype(gap.dtype)
                occ_floor = jnp.asarray(float(s.q_newton_occ_floor), dtype=gap.dtype)
                occ_floor = jnp.maximum(occ_floor, jnp.asarray(0.0, dtype=gap.dtype))
                occ_scale = diff_abs / (diff_abs + occ_floor)

                stepQ = _skew_hermitian(self.weights_b_grad * (M_total / denom) * occ_scale)
                if s.clip_Q is not None:
                    c_eff = jnp.asarray(float(s.clip_Q), dtype=self.h.real.dtype)
                    per_k_step = jnp.sqrt(jnp.sum(jnp.abs(stepQ) ** 2, axis=(-2, -1)) + jnp.asarray(1e-30, dtype=self.h.real.dtype))
                    scale = jnp.minimum(1.0, c_eff / (per_k_step + jnp.asarray(1e-12, dtype=self.h.real.dtype)))
                    stepQ = stepQ * scale[..., None, None]

                q_step = jnp.asarray(float(s.q_newton_step_size), dtype=self.h.real.dtype)
                q_step = jnp.clip(q_step, jnp.asarray(0.0, dtype=self.h.real.dtype), jnp.asarray(2.0, dtype=self.h.real.dtype))
                Q_next = _cayley_update(Q, -q_step * stepQ)

                tQ = st.tQ
                mQ = st.mQ
                vQ = st.vQ
            else:
                # --- Riemannian Adam on Omega + Cayley update for Q ---
                tQ = st.tQ + jnp.asarray(1, dtype=jnp.int32)
                mQ = s.b1 * st.mQ + (1.0 - s.b1) * Omega_update

                per_k_sq = jnp.sum(jnp.real(Omega_update * jnp.conj(Omega_update)), axis=(-2, -1))
                vQ = s.b2 * st.vQ + (1.0 - s.b2) * per_k_sq

                b1t_Q = jnp.asarray(s.b1, dtype=self.h.real.dtype) ** tQ
                b2t_Q = jnp.asarray(s.b2, dtype=self.h.real.dtype) ** tQ
                mQh = mQ / (1.0 - b1t_Q)
                vQh = vQ / (1.0 - b2t_Q)

                stepQ = mQh / (jnp.sqrt(vQh)[..., None, None] + jnp.asarray(s.eps, dtype=self.h.real.dtype))
                stepQ = _skew_hermitian(stepQ)

                if s.clip_Q is not None:
                    c_eff = jnp.asarray(float(s.clip_Q), dtype=self.h.real.dtype)
                    per_k_step = jnp.sqrt(jnp.sum(jnp.abs(stepQ) ** 2, axis=(-2, -1)) + jnp.asarray(1e-30, dtype=self.h.real.dtype))
                    scale = jnp.minimum(1.0, c_eff / (per_k_step + jnp.asarray(1e-12, dtype=self.h.real.dtype)))
                    stepQ = stepQ * scale[..., None, None]

                Q_next = _cayley_update(Q, -jnp.asarray(s.lr_Q, dtype=self.h.real.dtype) * stepQ)

            # Optional re-orthonormalization
            if int(s.reorth_every) > 0:
                do = (step_i + 1) % int(s.reorth_every) == 0
                Q_next = jax.lax.cond(do, lambda x: _project_unitary_qr(x), lambda x: x, Q_next)

            # stopping (purely on-device)
            comm_done = True
            if s.tol_comm is not None:
                comm_done = comm <= jnp.asarray(float(s.tol_comm), dtype=self.h.real.dtype)
            if s.tol_dA_rel is None:
                done = jnp.logical_and(
                    gnorm_Q <= jnp.asarray(s.tol_grad, dtype=self.h.real.dtype),
                    gnorm_L <= jnp.asarray(s.tol_grad, dtype=self.h.real.dtype),
                )
            else:
                dA = jnp.abs(A - st.A_prev)
                denom = jnp.maximum(jnp.asarray(1.0, dtype=self.h.real.dtype), jnp.abs(A))
                done = jnp.logical_and(
                    jnp.logical_and(
                        gnorm_Q <= jnp.asarray(s.tol_grad, dtype=self.h.real.dtype),
                        gnorm_L <= jnp.asarray(s.tol_grad, dtype=self.h.real.dtype),
                    ),
                    dA <= jnp.asarray(float(s.tol_dA_rel), dtype=self.h.real.dtype) * denom,
                )
            done = jnp.logical_and(done, jnp.asarray(comm_done))

            return VariationalHF._State(
                Q=Q_next,
                logits=logits_next,
                delta=delta,
                mL=mL,
                vL=vL,
                tL=tL,
                mQ=mQ,
                vQ=vQ,
                tQ=tQ,
                step=step_i + jnp.asarray(1, dtype=jnp.int32),
                A_prev=A,
                done=done,
                best_comm_all=best_comm_all,
                best_Q_all=best_Q_all,
                best_logits_all=best_logits_all,
                best_delta_all=best_delta_all,
                best_comm_post=best_comm_post,
                best_Q_post=best_Q_post,
                best_logits_post=best_logits_post,
                best_delta_post=best_delta_post,
                best_A=best_A,
                best_Q_A=best_Q_A,
                best_logits_A=best_logits_A,
                best_delta_A=best_delta_A,
            )

        return jax.lax.while_loop(cond, body, state0)

    # -----------------------------
    # Public solve
    # -----------------------------

    def solve(
        self,
        *,
        n_electrons: float,
        T: float,
        Q0: jax.Array | None = None,
        logits0: jax.Array | None = None,
        delta0: jax.Array | None = None,
        pin_field: jax.Array | None = None,
        tether_P: jax.Array | None = None,
        select_comm: float | None = None,
    ) -> VariationalHFResult:
        s = self.settings

        n_e = jnp.asarray(float(n_electrons), dtype=self.h.real.dtype)
        T_in = float(T)
        T_eff = jnp.asarray(max(T_in, float(s.T_min)), dtype=self.h.real.dtype)

        if pin_field is None:
            pin_field = jnp.zeros_like(self.h)
        if tether_P is None:
            tether_P = jnp.zeros_like(self.h)

        if Q0 is None or logits0 is None or delta0 is None:
            # Initialize from pinned h if pin is active at start
            pin0 = float(s.pin_strength)
            h_eff0 = hermitize(self.h + pin0 * hermitize(jnp.asarray(pin_field, dtype=self.h.dtype)))
            Q0, logits0, delta0 = self.init_from_fock(F0=h_eff0, n_electrons=float(n_electrons), T_eff=float(T_eff))

        Q0 = _project_unitary_qr(jnp.asarray(Q0, dtype=self.h.dtype))
        logits0 = jnp.asarray(logits0, dtype=self.h.real.dtype)
        delta0 = jnp.asarray(delta0, dtype=self.h.real.dtype)

        # Selection window (avoid "best state" being picked while stabilizers are ramping).
        select_after = 0
        if (s.pin_strength != s.pin_strength_final) and int(s.pin_ramp_steps) > 0:
            select_after = int(s.pin_ramp_steps)
        if (s.tether_strength != s.tether_strength_final) and int(s.tether_ramp_steps) > 0:
            select_after = max(select_after, int(s.tether_ramp_steps))
        select_after_j = jnp.asarray(int(select_after), dtype=jnp.int32)

        if select_comm is None:
            if s.tol_comm is None:
                select_comm_thr = jnp.asarray(jnp.inf, dtype=self.h.real.dtype)
            else:
                select_comm_thr = jnp.asarray(float(s.tol_comm), dtype=self.h.real.dtype)
        else:
            select_comm_thr = jnp.asarray(float(select_comm), dtype=self.h.real.dtype)

        st0 = VariationalHF._State(
            Q=Q0,
            logits=logits0,
            delta=delta0,
            mL=jnp.zeros_like(logits0),
            vL=jnp.zeros_like(logits0),
            tL=jnp.asarray(0, dtype=jnp.int32),
            mQ=jnp.zeros_like(Q0),
            vQ=jnp.zeros(self.w2d.shape, dtype=self.h.real.dtype),
            tQ=jnp.asarray(0, dtype=jnp.int32),
            step=jnp.asarray(0, dtype=jnp.int32),
            A_prev=jnp.asarray(jnp.inf, dtype=self.h.real.dtype),
            done=jnp.asarray(False),
            best_comm_all=jnp.asarray(jnp.inf, dtype=self.h.real.dtype),
            best_Q_all=Q0,
            best_logits_all=logits0,
            best_delta_all=delta0,
            best_comm_post=jnp.asarray(jnp.inf, dtype=self.h.real.dtype),
            best_Q_post=Q0,
            best_logits_post=logits0,
            best_delta_post=delta0,
            best_A=jnp.asarray(jnp.inf, dtype=self.h.real.dtype),
            best_Q_A=Q0,
            best_logits_A=logits0,
            best_delta_A=delta0,
        )

        st_fin = self._run(
            st0,
            n_e=n_e,
            T_eff=T_eff,
            pin_field=pin_field,
            tether_P=tether_P,
            select_after=select_after_j,
            select_comm=select_comm_thr,
        )

        # Choose representative state (matches the selection logic of the original playground solver):
        # 1) lowest A among states with comm <= select_comm_thr (after ramps),
        # 2) otherwise lowest comm after ramps,
        # 3) otherwise lowest comm overall.
        import numpy as np
        best_A = float(jax.device_get(st_fin.best_A))
        best_comm_post = float(jax.device_get(st_fin.best_comm_post))
        if np.isfinite(best_A):
            Q_sel, L_sel, d_sel = st_fin.best_Q_A, st_fin.best_logits_A, st_fin.best_delta_A
        elif np.isfinite(best_comm_post):
            Q_sel, L_sel, d_sel = st_fin.best_Q_post, st_fin.best_logits_post, st_fin.best_delta_post
        else:
            Q_sel, L_sel, d_sel = st_fin.best_Q_all, st_fin.best_logits_all, st_fin.best_delta_all

        # Final evaluation (unpinned should be achieved by choosing pin_final=0 if desired)
        delta_fin = _occ_shift_newton(
            L_sel,
            self.w2d,
            n_e,
            jax.lax.stop_gradient(d_sel),
            iters=int(s.occ_shift_iters),
            step_clip=s.occ_shift_step_clip,
        )
        occ = jax.nn.sigmoid(L_sel + delta_fin)
        P = self._build_P(Q_sel, occ)
        P_diff = hermitize(P - self.refP)

        # Use final pin strength (as per schedule endpoint) for reporting h_eff
        has_pin = (s.pin_strength != 0.0) or (s.pin_strength_final != 0.0)
        pin_end = float(s.pin_strength_final) if has_pin else 0.0
        h_eff = hermitize(self.h + pin_end * hermitize(jnp.asarray(pin_field, dtype=self.h.dtype))) if has_pin else self.h

        Sigma_x = self._sigma_exchange(P_diff)
        Sigma_h = self._sigma_hartree(P_diff)
        F = hermitize(h_eff + Sigma_h + Sigma_x)

        E1 = _wtd_trace(self.weights_b, h_eff, P)
        Eh = 0.5 * _wtd_trace(self.weights_b, Sigma_h, P_diff)
        Ex = 0.5 * _wtd_trace(self.weights_b, Sigma_x, P_diff)
        E = E1 + Eh + Ex

        S_ent = _entropy_from_occ(self.w2d, occ)
        A = E - T_eff * S_ent

        eps, _U = jnp.linalg.eigh(F)
        mu = find_chemical_potential(eps, self.w2d, n_electrons=float(n_electrons), T=float(T_eff))

        return VariationalHFResult(
            P=P,
            F=F,
            E=E,
            A=A,
            S=S_ent,
            mu=mu,
            Q=Q_sel,
            logits=L_sel,
            delta=delta_fin,
            T_eff=T_eff,
            n_steps=st_fin.step,
            converged=st_fin.done,
        )
