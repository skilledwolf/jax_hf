"""Reference SCF solver for generating gold-standard baselines.

Roothaan iteration with selectable acceleration:
  1. Build Fock F[P]
  2. Diagonalize -> eigenvalues, eigenvectors
  3. Find mu via Fermi-Dirac, construct P_new = V diag(f) V†
  4. Update the density via ``acceleration``:
       - "linear": P <- (1 - mixing) P + mixing P_new
       - "diis":   Pulay commutator-DIIS extrapolation of the Fock
       - "oda":    optimal damping (exact analytic line search on E)
     optionally clipped to a ``trust_radius`` step.
  5. Check convergence on density change and commutator norm.

This solver is intentionally separate from the direct-minimization solver and
serves as a reference / fallback.  DIIS and ODA mirror cpp_hf's solver_scf.hpp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .fock import build_fock, hf_energy
from .utils import find_chemical_potential, fermidirac


@dataclass(frozen=True)
class SCFConfig:
    max_iter: int = 200
    density_tol: float = 1e-7
    comm_tol: float = 1e-6
    mixing: float = 0.5
    level_shift: float = 0.0
    # Acceleration scheme: "linear" (alpha-mixing, default), "diis" (Pulay
    # commutator-DIIS extrapolation of the Fock), or "oda" (optimal damping --
    # exact analytic line search on E((1-lam)P + lam P_new), one extra Fock
    # build per iter).
    acceleration: str = "linear"
    diis_size: int = 6        # DIIS history depth (Pulay)
    diis_start: int = 2       # begin DIIS extrapolation after this many iters
    # DIIS damping: F_used = damp * F_extrap + (1 - damp) * F_current.
    # 1.0 = pure DIIS (default); ~0.7 suppresses late-iteration oscillation.
    diis_damping: float = 1.0
    # Trust radius on the density step (weighted Frobenius norm).  0.0 = off.
    # Bounds the per-iter motion so DIIS extrapolations cannot launch the
    # iteration to unphysical states (e.g. ungated Coulomb at high doping).
    trust_radius: float = 0.0
    project_fn: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.acceleration not in ("linear", "diis", "oda"):
            raise ValueError(
                "SCFConfig.acceleration must be 'linear', 'diis', or 'oda'; "
                f"got {self.acceleration!r}"
            )
        if self.diis_size <= 0:
            raise ValueError("diis_size must be positive")


@dataclass(frozen=True)
class SCFResult:
    density_matrix: Any
    fock_matrix: Any
    energy: Any
    chemical_potential: Any
    iterations: int
    converged: bool
    message: str
    history: dict[str, Any] = field(default_factory=dict)


def _herm(X):
    return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))


def _weighted_matrix_norm(matrix, weights, weight_sum):
    per_k = jnp.sum(jnp.abs(matrix) ** 2, axis=(-2, -1))
    return jnp.sqrt(jnp.sum(weights * per_k) / weight_sum)


def _density_from_fock(fock, *, weights, n_electrons, temperature, level_shift):
    """Diagonalize Fock, find mu, build P = V diag(f) V†."""
    eigenvalues, orbitals = jnp.linalg.eigh(_herm(fock))
    mu_raw = find_chemical_potential(eigenvalues, weights, n_electrons, temperature)
    occ_raw = fermidirac(eigenvalues - mu_raw, temperature)
    shifted = eigenvalues + level_shift * (1.0 - occ_raw)
    mu = find_chemical_potential(shifted, weights, n_electrons, temperature)
    occupations = fermidirac(shifted - mu, temperature)
    density = jnp.einsum("...in,...n,...jn->...ij", orbitals, occupations, jnp.conj(orbitals))
    return _herm(density), mu


def _diis_coefficients(R_hist, active, w2d):
    """Pulay C-DIIS coefficients from the commutator-residual history.

    Solves the augmented system ``[[B, -1], [-1, 0]] [c; lam] = [0; -1]`` with
    ``B[i, j] = sum_k w_k Re tr(R_i_k† R_j_k)``, masked so inactive history
    slots get ``c_i = 0``.  ``B`` is normalized by its largest entry first (the
    Pulay solution is invariant to that scaling) for float32 conditioning.
    Returns ``(coeffs, ok)`` where ``ok`` is False if the solve is singular.
    """
    N = R_hist.shape[0]
    real_dtype = jnp.real(R_hist).dtype
    B = jnp.real(jnp.einsum(
        "iklab,jklab->ij", jnp.conj(R_hist), R_hist * w2d[None, :, :, None, None]))
    B = B / jnp.maximum(jnp.max(jnp.abs(B)), jnp.asarray(1e-30, real_dtype))
    active_f = active.astype(real_dtype)
    A = jnp.zeros((N + 1, N + 1), real_dtype)
    A = A.at[:N, :N].set(B)
    A = A.at[:N, :N].add(jnp.diag(1.0 - active_f))   # inactive slots -> c_i = 0
    A = A.at[:N, N].set(-active_f)
    A = A.at[N, :N].set(-active_f)
    rhs = jnp.zeros(N + 1, real_dtype).at[N].set(-1.0)
    sol = jnp.linalg.solve(A, rhs)
    return sol[:N], jnp.all(jnp.isfinite(sol))


def solve_scf(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: SCFConfig | None = None,
    fock_build_fn: Callable | None = None,
) -> SCFResult:
    """Reference self-consistent field (SCF) solver.

    Standard Roothaan iteration: build Fock, diagonalise, re-occupy via
    Fermi-Dirac, update the density (linear mixing, Pulay DIIS, or optimal
    damping per ``config.acceleration``), repeat until converged.  Useful as a
    baseline or fallback when the direct-minimization solver struggles.

    Parameters
    ----------
    kernel
        :class:`HartreeFockKernel` with Hamiltonian, interaction, reference
        density, etc.
    P0
        Initial density matrix, shape ``(nk1, nk2, nb, nb)``.
    n_electrons
        Target electron count.
    config
        :class:`SCFConfig`.  ``acceleration`` selects linear / diis / oda;
        ``diis_size``, ``diis_start``, ``diis_damping`` tune DIIS; ``trust_radius``
        clips the per-iteration density step.  Defaults to ``SCFConfig()``.
    fock_build_fn
        Optional Fock builder for non-standard exchange kernels (e.g. the
        superlattice streaming Fock).  Signature must match
        :func:`jax_hf.fock.build_fock`'s call site.  ``None`` uses the default
        k-space FFT exchange.  Captured at trace time, so the JIT'd inner loop
        stays fully XLA-native.

    Returns
    -------
    SCFResult
        ``density_matrix, fock_matrix, energy, chemical_potential,
        iterations, converged, message, history``.
    """
    if config is None:
        config = SCFConfig()

    args = kernel.as_args()
    h = args["h"]

    density, fock, energy, mu, iterations, converged, \
        hist_E, hist_density, hist_comm = _run_scf(
            jnp.asarray(P0, dtype=h.dtype),
            h=h, weights_b=args["weights_b"], weight_sum=args["weight_sum"],
            VR=args["VR"], T=args["T"], refP=args["refP"], HH=args["HH"],
            w2d=kernel.w2d,
            n_electrons=float(n_electrons),
            include_hartree=args["include_hartree"],
            include_exchange=args["include_exchange"],
            exchange_hermitian_channel_packing=args["exchange_hermitian_channel_packing"],
            contact_g=args["contact_g"], contact_Oi=args["contact_Oi"],
            contact_Oj=args["contact_Oj"],
            mixing=float(config.mixing),
            level_shift=float(config.level_shift),
            density_tol=float(config.density_tol),
            comm_tol=float(config.comm_tol),
            diis_start=int(config.diis_start),
            diis_damping=float(config.diis_damping),
            trust_radius=float(config.trust_radius),
            max_iter=int(config.max_iter),
            acceleration=str(config.acceleration),
            diis_size=int(config.diis_size),
            project_fn=config.project_fn,
            fock_build_fn=fock_build_fn if fock_build_fn is not None else build_fock,
        )

    n_iter = int(iterations)
    is_converged = bool(converged)
    if is_converged:
        message = f"converged in {n_iter} iterations"
    else:
        dr = float(hist_density[n_iter - 1]) if n_iter > 0 else float("nan")
        cr = float(hist_comm[n_iter - 1]) if n_iter > 0 else float("nan")
        message = f"stopped after {n_iter} iterations (density_res={dr:.3e}, comm_res={cr:.3e})"

    return SCFResult(
        density_matrix=density,
        fock_matrix=fock,
        energy=energy,
        chemical_potential=mu,
        iterations=n_iter,
        converged=is_converged,
        message=message,
        history={
            "E": hist_E[:n_iter],
            "density_residual": hist_density[:n_iter],
            "commutator_residual": hist_comm[:n_iter],
        },
    )


@partial(
    jax.jit,
    static_argnames=(
        "include_hartree", "include_exchange",
        "exchange_hermitian_channel_packing",
        "max_iter", "acceleration", "diis_size", "project_fn", "fock_build_fn",
    ),
)
def _run_scf(
    density0,
    *,
    h, weights_b, weight_sum, VR, T, refP, HH, w2d,
    n_electrons,
    include_hartree, include_exchange,
    exchange_hermitian_channel_packing,
    contact_g, contact_Oi, contact_Oj,
    mixing, level_shift, density_tol, comm_tol,
    diis_start, diis_damping, trust_radius, max_iter,
    acceleration, diis_size,
    project_fn,
    fock_build_fn,
):
    real_dtype = jnp.zeros((), dtype=h.dtype).real.dtype
    weights_2d = w2d
    ws = jnp.maximum(weight_sum, jnp.asarray(1e-30, dtype=real_dtype))
    tiny = jnp.asarray(1e-30, dtype=real_dtype)

    hist_E = jnp.zeros(max_iter, dtype=real_dtype)
    hist_density = jnp.zeros(max_iter, dtype=real_dtype)
    hist_comm = jnp.zeros(max_iter, dtype=real_dtype)

    _project = project_fn if project_fn is not None else (lambda A: A)

    def _fock(density_h):
        return fock_build_fn(
            density_h, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
            project_fn=project_fn,
        )

    def _build_and_occupy(density):
        density_h = _herm(jnp.asarray(_project(density), dtype=h.dtype))
        Sigma, H, F = _fock(density_h)
        E = hf_energy(density_h, h=h, Sigma=Sigma, H=H, weights_b=weights_b)
        P_new, mu = _density_from_fock(
            F, weights=weights_2d, n_electrons=n_electrons,
            temperature=T, level_shift=level_shift,
        )
        P_new = _herm(jnp.asarray(_project(P_new), dtype=h.dtype))
        return P_new, F, E, mu

    def _linear(density, delta):
        return _herm(jnp.asarray(_project(density + mixing * delta), dtype=h.dtype))

    use_diis = acceleration == "diis"
    n_hist = diis_size if use_diis else 0
    F_hist0 = jnp.zeros((n_hist,) + h.shape, dtype=h.dtype)
    R_hist0 = jnp.zeros((n_hist,) + h.shape, dtype=h.dtype)

    def cond(carry):
        k, _, _, _, _, converged, _, _, _, _, _ = carry
        return jnp.logical_and(k < max_iter, jnp.logical_not(converged))

    def body(carry):
        k, density, _fk, _en, _mu, _conv, F_hist, R_hist, hE, hD, hC = carry

        P_new, F, E, mu = _build_and_occupy(density)
        delta = P_new - density
        comm = F @ density - density @ F
        d_res = _weighted_matrix_norm(delta, weights_2d, ws)
        c_res = _weighted_matrix_norm(comm, weights_2d, ws)
        converged = jnp.logical_and(d_res <= density_tol, c_res <= comm_tol)

        # --- density update per acceleration scheme (branch at trace time) ---
        if acceleration == "diis":
            idx = k % n_hist
            F_hist = F_hist.at[idx].set(F)
            R_hist = R_hist.at[idx].set(comm)
            count = jnp.minimum(k + 1, n_hist)
            active = jnp.arange(n_hist) < count
            coeffs, ok = _diis_coefficients(R_hist, active, weights_2d)
            F_extrap = jnp.einsum("i,iklab->klab", coeffs.astype(h.dtype), F_hist)
            damp = jnp.clip(diis_damping, 0.0, 1.0).astype(h.dtype)
            F_extrap = damp * F_extrap + (1.0 - damp) * F
            P_diis, _ = _density_from_fock(
                _herm(F_extrap), weights=weights_2d, n_electrons=n_electrons,
                temperature=T, level_shift=level_shift,
            )
            P_diis = _herm(jnp.asarray(_project(P_diis), dtype=h.dtype))
            engaged = jnp.logical_and(count >= jnp.maximum(2, diis_start), ok)
            updated = jnp.where(engaged, P_diis, _linear(density, delta))
        elif acceleration == "oda":
            # One extra Fock build at P_new gives the energy endpoint E(lam=1).
            Sigma_n, H_n, _ = _fock(P_new)
            E_new = hf_energy(P_new, h=h, Sigma=Sigma_n, H=H_n, weights_b=weights_b)
            tr = jnp.real(jnp.einsum("...ij,...ji->...", F, delta))  # tr(F dP) per k
            c1 = jnp.sum(weights_2d * tr)
            c2 = (E_new - E) - c1
            lam = jnp.where(
                c2 > 1e-12, jnp.clip(-c1 / (2.0 * c2), 0.0, 1.0),
                jnp.where(c1 < 0.0, jnp.asarray(1.0, real_dtype),
                          jnp.asarray(0.0, real_dtype)),
            )
            updated = _herm(jnp.asarray(
                _project(density + lam.astype(h.dtype) * delta), dtype=h.dtype))
        else:  # linear
            updated = _linear(density, delta)

        # --- trust-region clip on the proposed step (when not converged) -----
        step = updated - density
        step_norm = _weighted_matrix_norm(step, weights_2d, ws)
        do_clip = jnp.logical_and(trust_radius > 0.0, step_norm > trust_radius)
        scale = jnp.where(do_clip, trust_radius / jnp.maximum(step_norm, tiny), 1.0)
        clipped = _herm(jnp.asarray(
            _project(density + scale.astype(h.dtype) * step), dtype=h.dtype))
        updated = jnp.where(do_clip, clipped, updated)

        density_out = jnp.where(converged, P_new, updated)

        hE = hE.at[k].set(E)
        hD = hD.at[k].set(d_res)
        hC = hC.at[k].set(c_res)
        return (k + 1, density_out, F, E, mu, converged,
                F_hist, R_hist, hE, hD, hC)

    carry0 = (
        jnp.int32(0),
        jnp.asarray(density0, dtype=h.dtype),
        jnp.zeros_like(h),
        jnp.asarray(0.0, dtype=real_dtype),
        jnp.asarray(0.0, dtype=real_dtype),
        jnp.bool_(False),
        F_hist0, R_hist0,
        hist_E, hist_density, hist_comm,
    )

    k, density, fock, energy, mu, converged, _Fh, _Rh, hE, hD, hC = \
        jax.lax.while_loop(cond, body, carry0)

    # Final evaluation at the converged density (self-consistent density returned).
    _P_final, F_final, E_final, mu_final = _build_and_occupy(density)
    return density, F_final, E_final, mu_final, k, converged, hE, hD, hC
