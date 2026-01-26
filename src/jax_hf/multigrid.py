from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .main import HartreeFockKernel, jit_hartreefock_iteration
from .utils import density_matrix_from_fock, hermitize, resample_kgrid


class HFRunResult(NamedTuple):
    density: jax.Array
    fock: jax.Array
    energy: jax.Array
    mu: jax.Array
    n_iter: jax.Array
    history: dict[str, jax.Array]


class MultigridHFResult(NamedTuple):
    coarse: HFRunResult | None
    fine: HFRunResult
    Sigma_seed_f: jax.Array | None
    P0_seed_f: jax.Array | None


def coarse_to_fine_scf(
    *,
    weights_f: jax.Array,
    hamiltonian_f: jax.Array,
    coulomb_q_f: jax.Array,
    P0_f: jax.Array,
    electrondensity0: float,
    T: float,
    nk_coarse: int | None = None,
    resample_method: str = "linear",
    include_hartree: bool = False,
    include_exchange: bool = True,
    reference_density_f: jax.Array | None = None,
    hartree_matrix: jax.Array | None = None,
    coarse_scf_kwargs: dict[str, Any] | None = None,
    fine_scf_kwargs: dict[str, Any] | None = None,
) -> MultigridHFResult:
    """Coarse-to-fine continuation on uniform 2D k-grids.

    This mirrors the common workflow in continuum-model HF runs:

    1) Evaluate (H(k), V(q)) on a fine grid once.
    2) Resample to a smaller `nk_coarse` grid and run the SCF loop there.
    3) Interpolate the converged mean-field correction Σ(k) back to the fine grid
       and convert it into a physical density-matrix seed for a fine-grid SCF run.
    """
    coarse_scf_kwargs = dict(coarse_scf_kwargs or {})
    fine_scf_kwargs = dict(fine_scf_kwargs or {})

    weights_f = jnp.asarray(weights_f)
    hamiltonian_f = jnp.asarray(hamiltonian_f)
    coulomb_q_f = jnp.asarray(coulomb_q_f)
    P0_f = hermitize(jnp.asarray(P0_f))
    reference_density_f = hermitize(jnp.asarray(reference_density_f)) if reference_density_f is not None else None

    if weights_f.ndim != 2:
        raise ValueError(f"weights_f must have shape (nk,nk), got {weights_f.shape}")
    if hamiltonian_f.ndim < 4:
        raise ValueError(f"hamiltonian_f must have shape (nk,nk,nb,nb), got {hamiltonian_f.shape}")
    nk_f = int(weights_f.shape[0])
    if weights_f.shape[1] != nk_f:
        raise ValueError(f"weights_f must be square (nk,nk), got {weights_f.shape}")
    if hamiltonian_f.shape[0] != nk_f or hamiltonian_f.shape[1] != nk_f:
        raise ValueError("hamiltonian_f first two axes must match weights_f grid.")
    if P0_f.shape != hamiltonian_f.shape:
        raise ValueError(f"P0_f must have shape {hamiltonian_f.shape}, got {P0_f.shape}")
    if reference_density_f is not None and reference_density_f.shape != hamiltonian_f.shape:
        raise ValueError(f"reference_density_f must have shape {hamiltonian_f.shape}, got {reference_density_f.shape}")
    if coulomb_q_f.shape[0] != nk_f or coulomb_q_f.shape[1] != nk_f:
        raise ValueError("coulomb_q_f first two axes must match weights_f grid.")

    # ---- fine kernel/runner (always needed) ----
    kernel_f = HartreeFockKernel(
        weights=weights_f,
        hamiltonian=hamiltonian_f,
        coulomb_q=coulomb_q_f,
        T=float(T),
        include_hartree=bool(include_hartree),
        include_exchange=bool(include_exchange),
        reference_density=reference_density_f,
        hartree_matrix=hartree_matrix,
    )
    run_f = jit_hartreefock_iteration(kernel_f)

    # ---- no coarse stage requested -> just run fine ----
    if nk_coarse is None or int(nk_coarse) == nk_f:
        P_fin, F_fin, E_fin, mu_fin, k_fin, hist_fin = run_f(
            P0_f, float(electrondensity0), **fine_scf_kwargs
        )
        return MultigridHFResult(
            coarse=None,
            fine=HFRunResult(P_fin, F_fin, E_fin, mu_fin, k_fin, hist_fin),
            Sigma_seed_f=None,
            P0_seed_f=None,
        )

    nk_c = int(nk_coarse)
    if nk_c <= 0:
        raise ValueError("nk_coarse must be a positive integer or None.")

    # ---- coarse stage: resample from fine (no re-evaluation) ----
    weights_c = jnp.real(resample_kgrid(weights_f, nk_c, method=resample_method))
    wsum_f = jnp.sum(weights_f)
    wsum_c = jnp.sum(weights_c)
    if float(wsum_c) == 0.0:
        raise ValueError("Resampled coarse weights sum to zero; cannot renormalize.")
    weights_c = weights_c * (wsum_f / wsum_c)

    hamiltonian_c = hermitize(resample_kgrid(hamiltonian_f, nk_c, method=resample_method))
    coulomb_q_c = resample_kgrid(coulomb_q_f, nk_c, method=resample_method)
    P0_c = hermitize(resample_kgrid(P0_f, nk_c, method=resample_method))
    reference_density_c = (
        hermitize(resample_kgrid(reference_density_f, nk_c, method=resample_method))
        if reference_density_f is not None
        else None
    )

    kernel_c = HartreeFockKernel(
        weights=weights_c,
        hamiltonian=hamiltonian_c,
        coulomb_q=coulomb_q_c,
        T=float(T),
        include_hartree=bool(include_hartree),
        include_exchange=bool(include_exchange),
        reference_density=reference_density_c,
        hartree_matrix=hartree_matrix,
    )
    run_c = jit_hartreefock_iteration(kernel_c)

    P_c, F_c, E_c, mu_c, k_c, hist_c = run_c(P0_c, float(electrondensity0), **coarse_scf_kwargs)
    coarse = HFRunResult(P_c, F_c, E_c, mu_c, k_c, hist_c)

    # Σ_total(k) := F(k) - h0(k); includes Hartree shift + exchange.
    Sigma_c = hermitize(F_c - hamiltonian_c)
    Sigma_seed_f = hermitize(resample_kgrid(Sigma_c, nk_f, method=resample_method))
    P0_seed_f, _mu_seed = density_matrix_from_fock(
        hermitize(hamiltonian_f + Sigma_seed_f),
        weights_f,
        n_electrons=float(electrondensity0),
        T=float(T),
    )

    P_fin, F_fin, E_fin, mu_fin, k_fin, hist_fin = run_f(P0_seed_f, float(electrondensity0), **fine_scf_kwargs)
    fine = HFRunResult(P_fin, F_fin, E_fin, mu_fin, k_fin, hist_fin)

    return MultigridHFResult(
        coarse=coarse,
        fine=fine,
        Sigma_seed_f=Sigma_seed_f,
        P0_seed_f=P0_seed_f,
    )

