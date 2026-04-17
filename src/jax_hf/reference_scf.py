"""Reference SCF solver for generating gold-standard baselines.

Simple linear-mixing self-consistent field iteration:
  1. Build Fock F[P]
  2. Diagonalize → eigenvalues, eigenvectors
  3. Find mu via Fermi-Dirac, construct P_new = V diag(f) V†
  4. Mix: P ← (1 - alpha) P + alpha P_new
  5. Check convergence on density change and commutator norm

This solver is intentionally kept simple and separate from the direct
minimization solver.  Its purpose is to provide reference energies for
regression testing.
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
    project_fn: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")


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
    # Level shift: shift virtual eigenvalues up to aid convergence.
    # Always applied (just zero when level_shift=0).
    mu_raw = find_chemical_potential(eigenvalues, weights, n_electrons, temperature)
    occ_raw = fermidirac(eigenvalues - mu_raw, temperature)
    shifted = eigenvalues + level_shift * (1.0 - occ_raw)
    mu = find_chemical_potential(shifted, weights, n_electrons, temperature)
    occupations = fermidirac(shifted - mu, temperature)
    density = jnp.einsum("...in,...n,...jn->...ij", orbitals, occupations, jnp.conj(orbitals))
    return _herm(density), mu


def solve_scf(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: SCFConfig | None = None,
) -> SCFResult:
    """Reference self-consistent field (SCF) solver with linear mixing.

    Standard Roothaan iteration: build Fock, diagonalise, re-occupy via
    Fermi-Dirac, linearly mix densities, repeat until converged.  Useful as
    a baseline or fallback when the direct-minimization solver struggles.

    Parameters
    ----------
    kernel
        :class:`HartreeFockKernel` with Hamiltonian, interaction, reference
        density, etc.  Hartree and exchange inclusion controlled at kernel
        construction time.
    P0
        Initial density matrix, shape ``(nk1, nk2, nb, nb)``.
    n_electrons
        Target electron count.
    config
        :class:`SCFConfig` controls (max_iter, mixing, density_tol,
        comm_tol, level_shift, project_fn).  Defaults to ``SCFConfig()``.

    Returns
    -------
    SCFResult
        ``density_matrix, fock_matrix, energy, chemical_potential,
        iterations, converged, message, history``.

    Notes
    -----
    The inner loop uses dense ``eigh`` at every SCF step (not eigen-free),
    which makes per-iteration cost slightly higher than direct minimization
    for small problems.  On the bilayer benchmark, direct minimization
    converges ~8x faster end-to-end but SCF is more robust at phase
    boundaries.
    """
    if config is None:
        config = SCFConfig()

    args = kernel.as_args()
    h = args["h"]
    weights_b = args["weights_b"]
    weight_sum = args["weight_sum"]
    VR = args["VR"]
    T = args["T"]
    refP = args["refP"]
    HH = args["HH"]
    include_hartree = args["include_hartree"]
    include_exchange = args["include_exchange"]
    hcp = args["exchange_hermitian_channel_packing"]

    w2d = kernel.w2d
    project_fn = config.project_fn

    density, fock, energy, mu, iterations, converged, \
        hist_E, hist_density, hist_comm = _run_scf(
            jnp.asarray(P0, dtype=h.dtype),
            h=h, weights_b=weights_b, weight_sum=weight_sum,
            VR=VR, T=T, refP=refP, HH=HH, w2d=w2d,
            n_electrons=float(n_electrons),
            include_hartree=include_hartree,
            include_exchange=include_exchange,
            exchange_hermitian_channel_packing=hcp,
            mixing=float(config.mixing),
            level_shift=float(config.level_shift),
            density_tol=float(config.density_tol),
            comm_tol=float(config.comm_tol),
            max_iter=int(config.max_iter),
            project_fn=project_fn,
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
        "max_iter", "project_fn",
    ),
)
def _run_scf(
    density0,
    *,
    h, weights_b, weight_sum, VR, T, refP, HH, w2d,
    n_electrons,
    include_hartree, include_exchange,
    exchange_hermitian_channel_packing,
    mixing, level_shift, density_tol, comm_tol, max_iter,
    project_fn,
):
    real_dtype = jnp.zeros((), dtype=h.dtype).real.dtype
    weights_2d = w2d
    ws = jnp.maximum(weight_sum, jnp.asarray(1e-30, dtype=real_dtype))

    hist_E = jnp.zeros(max_iter, dtype=real_dtype)
    hist_density = jnp.zeros(max_iter, dtype=real_dtype)
    hist_comm = jnp.zeros(max_iter, dtype=real_dtype)

    _project = project_fn if project_fn is not None else (lambda A: A)

    def _build_and_occupy(density):
        density_h = _herm(jnp.asarray(_project(density), dtype=h.dtype))
        Sigma, H, F = build_fock(
            density_h, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            project_fn=project_fn,
        )
        E = hf_energy(density_h, h=h, Sigma=Sigma, H=H, weights_b=weights_b)
        P_new, mu = _density_from_fock(
            F, weights=weights_2d, n_electrons=n_electrons,
            temperature=T, level_shift=level_shift,
        )
        P_new = _herm(jnp.asarray(_project(P_new), dtype=h.dtype))
        return P_new, F, E, mu, Sigma, H

    def cond(carry):
        k, _, _, _, _, converged, _, _, _ = carry
        return jnp.logical_and(k < max_iter, jnp.logical_not(converged))

    def body(carry):
        k, density, _fock, _energy, _mu, _conv, hE, hD, hC = carry

        P_new, F, E, mu, _S, _H = _build_and_occupy(density)

        delta = P_new - density
        comm = F @ density - density @ F
        d_res = _weighted_matrix_norm(delta, weights_2d, ws)
        c_res = _weighted_matrix_norm(comm, weights_2d, ws)
        converged = jnp.logical_and(d_res <= density_tol, c_res <= comm_tol)

        mixed = _herm(jnp.asarray(
            _project(density + mixing * delta), dtype=h.dtype,
        ))
        density_out = jnp.where(converged, P_new, mixed)

        hE = hE.at[k].set(E)
        hD = hD.at[k].set(d_res)
        hC = hC.at[k].set(c_res)

        return (k + 1, density_out, F, E, mu, converged, hE, hD, hC)

    carry0 = (
        jnp.int32(0),
        jnp.asarray(density0, dtype=h.dtype),
        jnp.zeros_like(h),
        jnp.asarray(0.0, dtype=real_dtype),
        jnp.asarray(0.0, dtype=real_dtype),
        jnp.bool_(False),
        hist_E, hist_density, hist_comm,
    )

    k, density, fock, energy, mu, converged, hE, hD, hC = \
        jax.lax.while_loop(cond, body, carry0)

    # Final evaluation at converged density
    P_final, F_final, E_final, mu_final, _S, _H = _build_and_occupy(density)
    # Use the self-consistent density for the final energy
    return density, F_final, E_final, mu_final, k, converged, hE, hD, hC
