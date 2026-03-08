from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .main import HartreeFockKernel, jit_hartreefock_iteration
from .utils import density_matrix_from_fock, hermitize, resample_kgrid
from .variational import jit_variational_hartreefock_iteration
from .variational_qr import jit_variational_qr_iteration


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
    T_coarse: float | None = None,
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
    if nk_coarse is None or int(nk_coarse) >= nk_f:
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

    T_c = float(T_coarse) if T_coarse is not None else float(T)
    kernel_c = HartreeFockKernel(
        weights=weights_c,
        hamiltonian=hamiltonian_c,
        coulomb_q=coulomb_q_c,
        T=T_c,
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


# ---------------------------------------------------------------------------
# Coarse-to-fine for the variational (frozen-F) solver
# ---------------------------------------------------------------------------


def _resample_to_coarse(
    *,
    weights_f: jax.Array,
    hamiltonian_f: jax.Array,
    coulomb_q_f: jax.Array,
    P0_f: jax.Array,
    reference_density_f: jax.Array | None,
    nk_coarse: int,
    resample_method: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Downsample fine-grid arrays to a coarse k-grid."""
    nk_c = int(nk_coarse)
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
    return weights_c, hamiltonian_c, coulomb_q_c, P0_c, reference_density_c


class VariationalRunResult(NamedTuple):
    density: jax.Array
    fock: jax.Array
    energy: jax.Array
    mu: jax.Array
    n_iter: jax.Array
    history: dict[str, jax.Array]
    params: Any  # VariationalHFParams


class MultigridVariationalResult(NamedTuple):
    coarse: VariationalRunResult | None
    fine: VariationalRunResult
    P0_seed_f: jax.Array | None


def coarse_to_fine_variational(
    *,
    weights_f: jax.Array,
    hamiltonian_f: jax.Array,
    coulomb_q_f: jax.Array,
    P0_f: jax.Array,
    electrondensity0: float,
    T: float,
    T_coarse: float | None = None,
    nk_coarse: int | None = None,
    resample_method: str = "linear",
    include_hartree: bool = False,
    include_exchange: bool = True,
    reference_density_f: jax.Array | None = None,
    hartree_matrix: jax.Array | None = None,
    coarse_var_kwargs: dict[str, Any] | None = None,
    fine_var_kwargs: dict[str, Any] | None = None,
    solver: str = "cayley",
) -> MultigridVariationalResult:
    """Coarse-to-fine continuation using the variational (frozen-F) solver.

    Mirrors :func:`coarse_to_fine_scf` but uses the variational solver:

    1) Resample H(k), V(q), P0(k) to a coarser *nk_coarse* grid.
    2) Run the variational solver to convergence on the coarse grid.
    3) Extract the mean-field correction Σ(k) = F(k) − h(k) on the coarse grid,
       interpolate it back to the fine grid, and build a physical density-matrix
       seed P0_seed via eigendecomposition of (h_f + Σ_f).
    4) Run the variational solver on the fine grid starting from P0_seed with
       ``init_method="eigh"`` to extract natural orbitals.

    Parameters
    ----------
    weights_f, hamiltonian_f, coulomb_q_f : array
        Fine-grid k-mesh weights, Hamiltonian, and Coulomb kernel.
    P0_f : array
        Initial density-matrix seed on the fine grid.
    electrondensity0 : float
        Target electron density (per degeneracy).
    T : float
        Smearing temperature.
    nk_coarse : int or None
        Coarse-grid size.  ``None`` or equal to the fine grid size skips
        the coarse stage entirely.
    coarse_var_kwargs, fine_var_kwargs : dict or None
        Extra keyword arguments forwarded to the variational runner for
        the coarse and fine stages respectively (e.g. ``project_fn``,
        ``exchange_block_specs``, ``max_iter``, ``comm_tol``).
    solver : str
        Which variational solver to use: ``"cayley"`` (default) or ``"qr"``.

    Returns
    -------
    MultigridVariationalResult
        Named tuple with ``coarse``, ``fine`` (both :class:`VariationalRunResult`),
        and ``P0_seed_f`` (the upsampled seed used for the fine stage).
    """
    if solver == "cayley":
        _make_runner = jit_variational_hartreefock_iteration
    elif solver == "qr":
        _make_runner = jit_variational_qr_iteration
    else:
        raise ValueError(f"Unknown solver {solver!r}; expected 'cayley' or 'qr'")

    coarse_var_kwargs = dict(coarse_var_kwargs or {})
    fine_var_kwargs = dict(fine_var_kwargs or {})

    weights_f = jnp.asarray(weights_f)
    hamiltonian_f = jnp.asarray(hamiltonian_f)
    coulomb_q_f = jnp.asarray(coulomb_q_f)
    P0_f = hermitize(jnp.asarray(P0_f))
    reference_density_f = hermitize(jnp.asarray(reference_density_f)) if reference_density_f is not None else None

    nk_f = int(weights_f.shape[0])

    # ---- fine kernel/runner ----
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
    run_f = _make_runner(kernel_f)

    # ---- no coarse stage → fine only ----
    if nk_coarse is None or int(nk_coarse) >= nk_f:
        P_fin, F_fin, E_fin, mu_fin, k_fin, hist_fin, params_fin = run_f(
            P0_f, float(electrondensity0), return_params=True, **fine_var_kwargs,
        )
        return MultigridVariationalResult(
            coarse=None,
            fine=VariationalRunResult(P_fin, F_fin, E_fin, mu_fin, k_fin, hist_fin, params_fin),
            P0_seed_f=None,
        )

    nk_c = int(nk_coarse)

    # ---- coarse stage ----
    weights_c, hamiltonian_c, coulomb_q_c, P0_c, reference_density_c = _resample_to_coarse(
        weights_f=weights_f,
        hamiltonian_f=hamiltonian_f,
        coulomb_q_f=coulomb_q_f,
        P0_f=P0_f,
        reference_density_f=reference_density_f,
        nk_coarse=nk_c,
        resample_method=resample_method,
    )

    T_c = float(T_coarse) if T_coarse is not None else float(T)
    kernel_c = HartreeFockKernel(
        weights=weights_c,
        hamiltonian=hamiltonian_c,
        coulomb_q=coulomb_q_c,
        T=T_c,
        include_hartree=bool(include_hartree),
        include_exchange=bool(include_exchange),
        reference_density=reference_density_c,
        hartree_matrix=hartree_matrix,
    )
    run_c = _make_runner(kernel_c)

    P_c, F_c, E_c, mu_c, k_c, hist_c, params_c = run_c(
        P0_c, float(electrondensity0), return_params=True, **coarse_var_kwargs,
    )
    coarse = VariationalRunResult(P_c, F_c, E_c, mu_c, k_c, hist_c, params_c)

    # Σ(k) = F(k) - h(k) on coarse grid → upsample → fine-grid seed
    Sigma_c = hermitize(F_c - hamiltonian_c)
    Sigma_seed_f = hermitize(resample_kgrid(Sigma_c, nk_f, method=resample_method))
    P0_seed_f, _mu_seed = density_matrix_from_fock(
        hermitize(hamiltonian_f + Sigma_seed_f),
        weights_f,
        n_electrons=float(electrondensity0),
        T=float(T),
    )

    # ---- fine stage from upsampled seed ----
    fine_var_kwargs.setdefault("init_method", "eigh")
    P_fin, F_fin, E_fin, mu_fin, k_fin, hist_fin, params_fin = run_f(
        P0_seed_f, float(electrondensity0), return_params=True, **fine_var_kwargs,
    )
    fine = VariationalRunResult(P_fin, F_fin, E_fin, mu_fin, k_fin, hist_fin, params_fin)

    return MultigridVariationalResult(
        coarse=coarse,
        fine=fine,
        P0_seed_f=P0_seed_f,
    )
