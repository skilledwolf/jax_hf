"""Variational finite-T Hartree-Fock (HF) solver on a k-grid.

This lives in `playground-02-06-2026/` so we can iterate without modifying
the `jax_hf` package yet.

Design goals (best-practice, pragmatic):
- Variational in the density matrix: minimize free energy A[P] = E[P] - T S[P].
- Hard enforce constraints via parameterization:
    P(k) = Q(k) diag(occ(k)) Q(k)^H,  with Q(k) unitary and occ in (0,1).
- Enforce fixed electron number N via a single scalar shift delta in logits.
- Avoid differentiating through eigen-decompositions in the optimization loop.
- Optimize Q on the unitary manifold using a Cayley update and Riemannian Adam.
- Optimize logits with Adam.
- Optional pinning-field and commutator-penalty schedules to help branch selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

from jax_hf.utils import fermidirac, find_chemical_potential, hermitize, selfenergy_fft


ProjectFn = Callable[[jax.Array], jax.Array]


@dataclass(frozen=True)
class VariationalHFSettings:
    # Iterations
    max_steps: int = 400
    log_every: int = 20

    # Optimizer hyperparams
    lr_Q: float = 1e-2
    lr_logits: float = 3e-2
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    clip_Q: float | None = 10.0
    clip_logits: float | None = 10.0

    # Stopping conditions
    tol_comm: float = 2e-3
    tol_grad: float = 5e-2

    # Electron-number constraint shift delta (Newton)
    occ_shift_iters: int = 8
    occ_shift_step_clip: float | None = 2.0

    # Optional commutator penalty schedule: lam(t) * ||[F,P]||^2
    comm_penalty: float = 1e-2
    comm_penalty_final: float = 0.0
    comm_penalty_ramp_steps: int = 150

    # Optional pinning-field schedule: h -> h + pin_scale(t) * pin_field
    pin_strength: float = 0.0
    pin_strength_final: float = 0.0
    pin_ramp_steps: int = 100

    # Optional periodic QR re-orthonormalization (Cayley should keep Q unitary).
    reorth_every: int = 0


class VariationalHFResult(NamedTuple):
    P: jax.Array
    F: jax.Array
    E: jax.Array
    A: jax.Array
    mu: jax.Array
    Q: jax.Array
    logits: jax.Array
    delta: jax.Array
    history: dict[str, jax.Array]


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


def _hf_energy(weights_b: jax.Array, h_eff: jax.Array, sigma: jax.Array, P: jax.Array) -> jax.Array:
    return jnp.sum(jnp.real(jnp.einsum("...ij,...ji->...", weights_b * (h_eff + 0.5 * sigma), P)))


def _entropy_from_occ(w2d: jax.Array, occ: jax.Array) -> jax.Array:
    real_dtype = occ.dtype
    eps = jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
    one = jnp.asarray(1.0, dtype=real_dtype)
    o = jnp.clip(occ, eps, one - eps)
    return -jnp.sum(w2d[..., None] * (o * jnp.log(o) + (one - o) * jnp.log(one - o)))


def _commutator_mean_sq(F: jax.Array, P: jax.Array, *, w2d: jax.Array, weight_sum: jax.Array) -> jax.Array:
    R = F @ P - P @ F
    per_k = jnp.sum(jnp.abs(R) ** 2, axis=(-2, -1))
    real_dtype = jnp.zeros((), dtype=per_k.dtype).real.dtype
    denom = jnp.maximum(jnp.asarray(weight_sum, dtype=real_dtype), jnp.asarray(1e-30, dtype=real_dtype))
    return jnp.sum(w2d * per_k) / denom


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
            sc = jnp.asarray(step_clip, dtype=real_dtype)
            step = jnp.clip(step, -sc, sc)
        return delta + step

    return jax.lax.fori_loop(0, int(iters), body, delta0)


def _schedule_linear(step: int, *, start: float, end: float, ramp_steps: int) -> jax.Array:
    if int(ramp_steps) > 0:
        t = min(float(step) / float(ramp_steps), 1.0)
    else:
        t = 0.0
    return jnp.asarray((1.0 - t) * float(start) + t * float(end), dtype=jnp.float32)


def make_project_fn(
    *,
    symmetry_conj_generators: jax.Array | None = None,
    time_reversal_U: jax.Array | None = None,
) -> ProjectFn:
    """Build a projection function enforcing symmetries by averaging.

    - Unitary conjugation generators g: A <- (A + g A g^H)/2
    - Antiunitary time reversal with U: A(k) <- (A(k) + U conj(A(-k)) U^H)/2
    """

    sym_gens = None if symmetry_conj_generators is None else jnp.asarray(symmetry_conj_generators)
    tr_U = None if time_reversal_U is None else jnp.asarray(time_reversal_U)

    def _project_under_conjugations(A: jax.Array) -> jax.Array:
        if sym_gens is None:
            return A
        if sym_gens.ndim != 3:
            raise ValueError("symmetry_conj_generators must have shape (n, d, d).")

        def body(i, acc):
            g = sym_gens[i]
            gH = jnp.conj(jnp.swapaxes(g, -1, -2))
            return 0.5 * (acc + (g @ acc) @ gH)

        return jax.lax.fori_loop(0, sym_gens.shape[0], body, A)

    def _project_time_reversal(A: jax.Array) -> jax.Array:
        if tr_U is None:
            return A
        U = tr_U
        UH = jnp.conj(jnp.swapaxes(U, -1, -2))
        A_flip = jnp.flip(A, axis=(0, 1))
        A_tr = (U @ jnp.conj(A_flip)) @ UH
        return 0.5 * (A + A_tr)

    def project(A: jax.Array) -> jax.Array:
        return _project_time_reversal(_project_under_conjugations(A))

    return project


class VariationalHF:
    """Single variational solver (branch exploration is handled by wrappers)."""

    def __init__(
        self,
        *,
        h: jax.Array,  # (nk1, nk2, d, d)
        weights: jax.Array,  # (nk1, nk2)
        coulomb_q: jax.Array,  # (nk1, nk2) or (nk1, nk2, 1, 1) or (nk1, nk2, d, d)
        reference_density: jax.Array | None = None,
        project_fn: ProjectFn | None = None,
        settings: VariationalHFSettings | None = None,
    ):
        self.settings = settings or VariationalHFSettings()
        self.h = hermitize(jnp.asarray(h))
        self.w2d = jnp.asarray(weights, dtype=self.h.real.dtype)
        self.weights_b = self.w2d[..., None, None]
        self.weight_sum = jnp.sum(self.w2d)

        self.project = project_fn or (lambda A: A)

        self.refP = (
            hermitize(jnp.asarray(reference_density, dtype=self.h.dtype))
            if reference_density is not None
            else jnp.zeros_like(self.h)
        )

        Vq = jnp.asarray(coulomb_q)
        if Vq.ndim == 2:
            Vq = Vq[..., None, None]
        self.VR = jnp.fft.fftn(self.weights_b * jnp.asarray(Vq, dtype=self.h.dtype), axes=(0, 1))

        # JIT the per-step update once.
        self._step = jax.jit(self._step_impl)

    def sigma_of_P(self, P: jax.Array) -> jax.Array:
        Sigma = hermitize(selfenergy_fft(self.VR, hermitize(P - self.refP)))
        return self.project(hermitize(Sigma))

    def init_from_fock(self, *, F0: jax.Array, n_electrons: float, T: float) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Initialize (Q, logits, delta) by diagonalizing a Fock-like matrix F0."""
        eps, U = jnp.linalg.eigh(hermitize(F0))
        mu = find_chemical_potential(eps, self.w2d, n_electrons=float(n_electrons), T=float(T))
        occ = fermidirac(eps - mu, float(T)).astype(self.h.real.dtype)

        eps_occ = jnp.asarray(jnp.finfo(occ.dtype).eps, dtype=occ.dtype)
        one = jnp.asarray(1.0, dtype=occ.dtype)
        occ = jnp.clip(occ, eps_occ, one - eps_occ)
        logits = jnp.log(occ) - jnp.log(one - occ)
        Q0 = _project_unitary_qr(U)
        delta0 = jnp.asarray(0.0, dtype=occ.dtype)
        return Q0, logits, delta0

    def _step_impl(
        self,
        state: dict[str, jax.Array],
        *,
        pin_scale: jax.Array,
        comm_lam: jax.Array,
        n_e: jax.Array,
        T: jax.Array,
        pin_field: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        s = self.settings
        Q = state["Q"]
        logits = state["logits"]
        delta0 = jax.lax.stop_gradient(state["delta"])

        # --- Physical P, Sigma, F ---
        delta = _occ_shift_newton(
            logits,
            self.w2d,
            n_e,
            delta0,
            iters=int(s.occ_shift_iters),
            step_clip=s.occ_shift_step_clip,
        )
        occ = jax.nn.sigmoid(logits + delta)

        P = self.project(hermitize(jnp.einsum("...in,...n,...jn->...ij", Q, occ, jnp.conj(Q))))
        h_eff = hermitize(self.h + pin_scale * pin_field)
        Sigma = self.sigma_of_P(P)
        F = hermitize(h_eff + Sigma)

        E = _hf_energy(self.weights_b, h_eff, Sigma, P)
        S_ent = _entropy_from_occ(self.w2d, occ)
        A = E - T * S_ent

        comm2 = _commutator_mean_sq(F, P, w2d=self.w2d, weight_sum=self.weight_sum)
        comm = jnp.sqrt(comm2 + jnp.asarray(1e-30, dtype=comm2.dtype))

        # --- Analytic gradients (no autodiff) ---
        # Work in the Q-basis: M = Q^H F Q.
        M = jnp.einsum("...ji,...jk,...kl->...il", jnp.conj(Q), F, Q)
        diag_M = jnp.real(jnp.diagonal(M, axis1=-2, axis2=-1)).astype(self.h.real.dtype)

        # Occupation gradient (KKT form): g_occ = diag(M) + T * logit(occ).
        # Since occ = sigmoid(logits + delta), logit(occ) = logits + delta.
        g_occ = diag_M + jnp.asarray(T, dtype=diag_M.dtype) * (logits + delta)
        d_occ = occ * (1.0 - occ)
        w = self.w2d[..., None].astype(diag_M.dtype)
        tiny = jnp.asarray(1e-18, dtype=diag_M.dtype)
        mu_proj = jnp.sum(w * g_occ * d_occ) / (jnp.sum(w * d_occ) + tiny)
        g_logits = (g_occ - mu_proj) * d_occ

        # Unitary gradient generator: Omega = [M, diag(occ)] = M_ij (occ_j - occ_i).
        diff = occ[..., None, :] - occ[..., :, None]
        Omega = _skew_hermitian(M * diff)

        # Report RMS Frobenius norms per k-point (stable across nk).
        per_k_Q = jnp.sum(jnp.abs(Omega) ** 2, axis=(-2, -1))
        gnorm_Q = jnp.sqrt(jnp.mean(per_k_Q))
        per_k_L = jnp.sum(g_logits**2, axis=-1)
        gnorm_L = jnp.sqrt(jnp.mean(per_k_L))

        if s.clip_Q is not None:
            # Per-k clipping with a clip budget that keeps backwards-compatible
            # scaling with nk. (Old behavior clipped the global Frobenius norm.)
            k_scale = jnp.sqrt(jnp.asarray(Omega.shape[0] * Omega.shape[1], dtype=self.h.real.dtype))
            c_eff = jnp.asarray(float(s.clip_Q), dtype=self.h.real.dtype) / k_scale
            per_k_norm = jnp.sqrt(per_k_Q + jnp.asarray(1e-30, dtype=self.h.real.dtype))
            scale = jnp.minimum(1.0, c_eff / (per_k_norm + jnp.asarray(1e-12, dtype=self.h.real.dtype)))
            Omega = Omega * scale[..., None, None]
        if s.clip_logits is not None:
            k_scale = jnp.sqrt(jnp.asarray(g_logits.shape[0] * g_logits.shape[1], dtype=self.h.real.dtype))
            c_eff = jnp.asarray(float(s.clip_logits), dtype=self.h.real.dtype) / k_scale
            per_k_norm = jnp.sqrt(per_k_L + jnp.asarray(1e-30, dtype=self.h.real.dtype))
            scale = jnp.minimum(1.0, c_eff / (per_k_norm + jnp.asarray(1e-12, dtype=self.h.real.dtype)))
            g_logits = g_logits * scale[..., None]

        # Adam on logits
        tL = state["tL"] + 1
        mL = s.b1 * state["mL"] + (1.0 - s.b1) * g_logits
        vL = s.b2 * state["vL"] + (1.0 - s.b2) * (g_logits**2)
        b1t_L = jnp.asarray(s.b1, dtype=mL.dtype) ** tL
        b2t_L = jnp.asarray(s.b2, dtype=vL.dtype) ** tL
        mLh = mL / (1.0 - b1t_L)
        vLh = vL / (1.0 - b2t_L)
        logits_next = logits - jnp.asarray(s.lr_logits, dtype=logits.dtype) * mLh / (jnp.sqrt(vLh) + s.eps)

        # Momentum on Omega + Cayley update for Q.
        tQ = state["tQ"] + 1
        mQ = s.b1 * state["mQ"] + (1.0 - s.b1) * Omega
        b1t_Q = jnp.asarray(s.b1, dtype=self.h.real.dtype) ** tQ
        stepQ = _skew_hermitian(mQ / (1.0 - b1t_Q))
        Q_next = _cayley_update(Q, -jnp.asarray(s.lr_Q, dtype=self.h.real.dtype) * stepQ)

        # Optional re-orthonormalization (rarely needed).
        if int(s.reorth_every) > 0:
            do = (state["iter"] + 1) % int(s.reorth_every) == 0
            Q_next = jax.lax.cond(do, lambda x: _project_unitary_qr(x), lambda x: x, Q_next)

        next_state = dict(
            Q=Q_next,
            logits=logits_next,
            delta=delta,
            mL=mL,
            vL=vL,
            tL=tL,
            mQ=mQ,
            tQ=tQ,
            iter=state["iter"] + 1,
        )

        metrics = dict(
            A=A,
            E=E,
            loss=A + comm_lam * comm2,
            comm=comm,
            comm2=comm2,
            gnorm_Q=gnorm_Q,
            gnorm_logits=gnorm_L,
            delta=delta,
        )
        return next_state, metrics

    def solve(
        self,
        *,
        n_electrons: float,
        T: float,
        Q0: jax.Array | None = None,
        logits0: jax.Array | None = None,
        delta0: jax.Array | None = None,
        pin_field: jax.Array | None = None,
        # Optional schedule overrides (if None, use settings)
        pin_strength: float | None = None,
        pin_strength_final: float | None = None,
        pin_ramp_steps: int | None = None,
        comm_penalty: float | None = None,
        comm_penalty_final: float | None = None,
        comm_penalty_ramp_steps: int | None = None,
        max_steps: int | None = None,
        verbose: bool = True,
    ) -> VariationalHFResult:
        s = self.settings
        n_e = jnp.asarray(n_electrons, dtype=self.h.real.dtype)
        Tj = jnp.asarray(T, dtype=self.h.real.dtype)

        pin_strength = float(s.pin_strength) if pin_strength is None else float(pin_strength)
        pin_strength_final = (
            float(s.pin_strength_final) if pin_strength_final is None else float(pin_strength_final)
        )
        pin_ramp_steps = int(s.pin_ramp_steps) if pin_ramp_steps is None else int(pin_ramp_steps)

        comm_penalty = float(s.comm_penalty) if comm_penalty is None else float(comm_penalty)
        comm_penalty_final = (
            float(s.comm_penalty_final) if comm_penalty_final is None else float(comm_penalty_final)
        )
        comm_penalty_ramp_steps = (
            int(s.comm_penalty_ramp_steps) if comm_penalty_ramp_steps is None else int(comm_penalty_ramp_steps)
        )

        steps = int(s.max_steps) if max_steps is None else int(max_steps)

        if pin_field is None:
            pin_field = jnp.zeros_like(self.h)
        pin_field = hermitize(jnp.asarray(pin_field, dtype=self.h.dtype))

        if Q0 is None or logits0 is None or delta0 is None:
            F0 = hermitize(self.h + pin_strength * pin_field)
            Q0, logits0, delta0 = self.init_from_fock(F0=F0, n_electrons=float(n_electrons), T=float(T))

        Q0 = _project_unitary_qr(jnp.asarray(Q0, dtype=self.h.dtype))
        logits0 = jnp.asarray(logits0, dtype=self.h.real.dtype)
        delta0 = jnp.asarray(delta0, dtype=self.h.real.dtype)

        state: dict[str, jax.Array] = dict(
            Q=Q0,
            logits=logits0,
            delta=delta0,
            mL=jnp.zeros_like(logits0),
            vL=jnp.zeros_like(logits0),
            tL=jnp.asarray(0, dtype=jnp.int32),
            mQ=jnp.zeros_like(Q0),
            tQ=jnp.asarray(0, dtype=jnp.int32),
            iter=jnp.asarray(0, dtype=jnp.int32),
        )

        hist: dict[str, list[jax.Array]] = {k: [] for k in ["A", "E", "comm", "gnorm_Q", "gnorm_logits", "delta"]}

        best_state = None
        best_comm = float("inf")
        best_state_all = None
        best_comm_all = float("inf")
        # When using a pinning-field ramp, the commutator can be artificially small
        # early in the trajectory while the pin is still active. We want the *unpinned*
        # stationary point, so only start selecting "best-by-comm" once the pin has
        # reached its final value (or immediately if there's no ramp).
        select_after = 0
        if pin_strength != pin_strength_final and pin_ramp_steps > 0:
            select_after = int(pin_ramp_steps)

        for k in range(steps):
            pin_scale = _schedule_linear(k, start=pin_strength, end=pin_strength_final, ramp_steps=pin_ramp_steps).astype(
                self.h.real.dtype
            )
            comm_lam = _schedule_linear(
                k,
                start=comm_penalty,
                end=comm_penalty_final,
                ramp_steps=comm_penalty_ramp_steps,
            ).astype(self.h.real.dtype)

            state, metrics = self._step(state, pin_scale=pin_scale, comm_lam=comm_lam, n_e=n_e, T=Tj, pin_field=pin_field)

            for key in ["A", "E", "comm", "gnorm_Q", "gnorm_logits", "delta"]:
                hist[key].append(metrics[key])

            comm_val = float(metrics["comm"])
            if comm_val < best_comm_all:
                best_comm_all = comm_val
                best_state_all = dict(state)
            if (k >= select_after) and (comm_val < best_comm):
                best_comm = comm_val
                best_state = dict(state)

            if verbose and s.log_every and (k % int(s.log_every) == 0 or k == steps - 1):
                print(
                    f"[varHF] step={k:04d}  A={float(metrics['A']): .6e}  E={float(metrics['E']): .6e}  "
                    f"comm={float(metrics['comm']):.3e}  |gQ|={float(metrics['gnorm_Q']):.3e}  |gL|={float(metrics['gnorm_logits']):.3e}"
                )

            if (
                float(metrics["comm"]) <= float(s.tol_comm)
                and float(metrics["gnorm_Q"]) <= float(s.tol_grad)
                and float(metrics["gnorm_logits"]) <= float(s.tol_grad)
            ):
                break

        # If we never entered the post-ramp window (e.g. very short runs), fall back
        # to best state over the full trajectory.
        if best_state is None:
            assert best_state_all is not None
            best_state = best_state_all
        Q = best_state["Q"]
        logits = best_state["logits"]
        delta = best_state["delta"]

        # Build physical P and evaluate the *unpinned* HF energy for reporting.
        delta_fin = _occ_shift_newton(
            logits,
            self.w2d,
            n_e,
            jax.lax.stop_gradient(delta),
            iters=int(s.occ_shift_iters),
            step_clip=s.occ_shift_step_clip,
        )
        occ = jax.nn.sigmoid(logits + delta_fin)
        P = self.project(hermitize(jnp.einsum("...in,...n,...jn->...ij", Q, occ, jnp.conj(Q))))
        Sigma = self.sigma_of_P(P)
        F = hermitize(self.h + Sigma)
        E = _hf_energy(self.weights_b, self.h, Sigma, P)
        S_ent = _entropy_from_occ(self.w2d, occ)
        A = E - Tj * S_ent

        eps, _U = jnp.linalg.eigh(F)
        mu = find_chemical_potential(eps, self.w2d, n_electrons=float(n_electrons), T=float(T))

        history = {k: jnp.asarray(v) for k, v in hist.items()}
        return VariationalHFResult(P=P, F=F, E=E, A=A, mu=mu, Q=Q, logits=logits, delta=delta_fin, history=history)
