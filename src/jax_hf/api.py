from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import jax.numpy as jnp

from .main import HartreeFockKernel, jit_hartreefock_iteration
from .mixing import PRECOND_AUTO
from .utils import density_matrix_from_fock, hermitize, resample_kgrid
from .variational import VariationalHFParams
from .variational_qr import jit_variational_qr_iteration
from .variational_rtr import jit_variational_rtr_iteration


SolverName = Literal["scf", "qr", "rtr"]


@dataclass(frozen=True)
class HFProblem:
    """Array-based Hartree-Fock problem definition shared by all solvers."""

    weights: Any
    hamiltonian: Any
    coulomb_q: Any
    T: float
    include_hartree: bool = False
    include_exchange: bool = True
    reference_density: Any | None = None
    hartree_matrix: Any | None = None

    def kernel(self) -> HartreeFockKernel:
        return HartreeFockKernel(
            weights=self.weights,
            hamiltonian=self.hamiltonian,
            coulomb_q=self.coulomb_q,
            T=float(self.T),
            include_hartree=bool(self.include_hartree),
            include_exchange=bool(self.include_exchange),
            reference_density=self.reference_density,
            hartree_matrix=self.hartree_matrix,
        )


@dataclass(frozen=True)
class SCFRunConfig:
    """Controls for the DIIS-style SCF iteration."""

    max_iter: int = 100
    comm_tol: float = 5e-3
    diis_size: int = 4
    log_every: int | None = None
    mixing_alpha: float = 1.0
    precond_delta: float = 5e-3
    precond_mode: int | str = PRECOND_AUTO
    precond_auto_nb: int = 128
    level_shift: float = 0.0
    mu_method: str = "bisection"
    eigh_block_specs: object | None = None
    eigh_check_offdiag: bool | None = None
    eigh_offdiag_atol: float = 1e-12
    eigh_offdiag_rtol: float = 0.0
    project_fn: Any = None

    def as_run_kwargs(self) -> dict[str, Any]:
        return {
            "max_iter": int(self.max_iter),
            "comm_tol": float(self.comm_tol),
            "diis_size": int(self.diis_size),
            "log_every": self.log_every,
            "mixing_alpha": float(self.mixing_alpha),
            "precond_delta": float(self.precond_delta),
            "precond_mode": self.precond_mode,
            "precond_auto_nb": int(self.precond_auto_nb),
            "level_shift": float(self.level_shift),
            "mu_method": str(self.mu_method),
            "eigh_block_specs": self.eigh_block_specs,
            "eigh_check_offdiag": self.eigh_check_offdiag,
            "eigh_offdiag_atol": float(self.eigh_offdiag_atol),
            "eigh_offdiag_rtol": float(self.eigh_offdiag_rtol),
            "project_fn": self.project_fn,
        }


@dataclass(frozen=True)
class QRRunConfig:
    """Controls for the variational QR Hartree-Fock solver."""

    max_iter: int = 80
    comm_tol: float = 1e-5
    p_tol: float = 1e-2
    e_tol: float = 1e-10
    inner_sweeps: int = 2
    q_sweeps: int = 1
    p_floor: float = 0.10
    denom_scale: float = 1e-3
    max_rot: float = 0.60
    bt_accept: float = 0.999
    bt_shrink: float = 0.5
    bt_max: int = 5
    mu_maxiter: int = 25
    mu_tol: float = 1e-12
    line_search: bool = True
    block_sizes: tuple[int, ...] | None = None
    optimizer: str = "cg"
    lbfgs_m: int = 5
    exchange_check_offdiag: bool | None = None
    exchange_offdiag_atol: float = 1e-12
    exchange_offdiag_rtol: float = 0.0
    project_fn: Any = None
    init_method: str = "identity"

    def as_run_kwargs(self) -> dict[str, Any]:
        return {
            "max_iter": int(self.max_iter),
            "comm_tol": float(self.comm_tol),
            "p_tol": float(self.p_tol),
            "e_tol": float(self.e_tol),
            "inner_sweeps": int(self.inner_sweeps),
            "q_sweeps": int(self.q_sweeps),
            "p_floor": float(self.p_floor),
            "denom_scale": float(self.denom_scale),
            "max_rot": float(self.max_rot),
            "bt_accept": float(self.bt_accept),
            "bt_shrink": float(self.bt_shrink),
            "bt_max": int(self.bt_max),
            "mu_maxiter": int(self.mu_maxiter),
            "mu_tol": float(self.mu_tol),
            "line_search": bool(self.line_search),
            "block_sizes": self.block_sizes,
            "optimizer": str(self.optimizer),
            "lbfgs_m": int(self.lbfgs_m),
            "exchange_check_offdiag": self.exchange_check_offdiag,
            "exchange_offdiag_atol": float(self.exchange_offdiag_atol),
            "exchange_offdiag_rtol": float(self.exchange_offdiag_rtol),
            "project_fn": self.project_fn,
        }


@dataclass(frozen=True)
class RTRRunConfig:
    """Controls for the variational RTR Hartree-Fock solver."""

    max_iter: int = 80
    comm_tol: float = 1e-5
    p_tol: float = 1e-2
    e_tol: float = 0.0
    max_cg_iter: int = 15
    cg_tol: float = 1e-2
    max_rot: float = 0.60
    denom_scale: float = 1e-3
    mu_maxiter: int = 25
    mu_tol: float = 1e-9
    block_sizes: tuple[int, ...] | None = None
    exchange_check_offdiag: bool | None = None
    exchange_offdiag_atol: float = 1e-12
    exchange_offdiag_rtol: float = 0.0
    project_fn: Any = None
    init_method: str = "identity"

    def as_run_kwargs(self) -> dict[str, Any]:
        return {
            "max_iter": int(self.max_iter),
            "comm_tol": float(self.comm_tol),
            "p_tol": float(self.p_tol),
            "e_tol": float(self.e_tol),
            "max_cg_iter": int(self.max_cg_iter),
            "cg_tol": float(self.cg_tol),
            "max_rot": float(self.max_rot),
            "denom_scale": float(self.denom_scale),
            "mu_maxiter": int(self.mu_maxiter),
            "mu_tol": float(self.mu_tol),
            "block_sizes": self.block_sizes,
            "exchange_check_offdiag": self.exchange_check_offdiag,
            "exchange_offdiag_atol": float(self.exchange_offdiag_atol),
            "exchange_offdiag_rtol": float(self.exchange_offdiag_rtol),
            "project_fn": self.project_fn,
        }


@dataclass(frozen=True)
class SolveStageResult:
    """Standardized public result for one solver stage."""

    solver: SolverName
    density: Any
    fock: Any
    energy: Any
    mu: Any
    n_iter: int
    history: dict[str, np.ndarray]
    params: Any | None = None


@dataclass(frozen=True)
class SolveResult:
    """Standardized public solve result, with optional coarse stage."""

    solver: SolverName
    fine: SolveStageResult
    coarse: SolveStageResult | None = None
    Sigma_seed_f: Any | None = None
    P0_seed_f: Any | None = None

    @property
    def density(self) -> Any:
        return self.fine.density

    @property
    def fock(self) -> Any:
        return self.fine.fock

    @property
    def energy(self) -> Any:
        return self.fine.energy

    @property
    def mu(self) -> Any:
        return self.fine.mu

    @property
    def n_iter(self) -> int:
        return self.fine.n_iter

    @property
    def history(self) -> dict[str, np.ndarray]:
        return self.fine.history

    @property
    def params(self) -> Any | None:
        return self.fine.params


@dataclass(frozen=True)
class DensityMatrixSeed:
    """Explicit density-matrix warm start for a solve."""

    density_matrix: Any


@dataclass(frozen=True)
class VariationalSeed:
    """Explicit variational-parameter warm start for a solve."""

    params: VariationalHFParams


@dataclass(frozen=True)
class ContinuationConfig:
    """Explicit coarse-to-fine continuation controls."""

    nk_coarse: int
    coarse_problem: HFProblem | None = None
    coarse_seed: DensityMatrixSeed | None = None
    coarse_n_electrons_per_degeneracy: float | None = None
    coarse_config: Any | None = None
    fine_config: Any | None = None
    T_coarse: float | None = None
    resample_method: str = "linear"


def solve(
    problem: HFProblem,
    *,
    solver: str = "scf",
    seed: DensityMatrixSeed | VariationalSeed | None = None,
    continuation: ContinuationConfig | None = None,
    P0: Any | None = None,
    params0: VariationalHFParams | None = None,
    n_electrons_per_degeneracy: float | None = None,
    electrondensity0: float | None = None,
    config: Any | None = None,
    nk_coarse: int | None = None,
    coarse_problem: HFProblem | None = None,
    coarse_P0: Any | None = None,
    coarse_n_electrons_per_degeneracy: float | None = None,
    coarse_config: Any | None = None,
    fine_config: Any | None = None,
    T_coarse: float | None = None,
    resample_method: str = "linear",
) -> SolveResult:
    """Solve one Hartree-Fock problem with a single public dispatch point.

    Parameters
    ----------
    problem
        Fine-grid problem definition.
    solver
        One of ``"scf"``, ``"qr"``, or ``"rtr"``. Common aliases such as
        ``"jax"`` and ``"variational_qr"`` are accepted.
    seed
        Explicit density-matrix or variational warm start. The legacy ``P0`` and
        ``params0`` inputs are still accepted.
    n_electrons_per_degeneracy
        Target electron density integrated over mesh weights, divided by any
        external degeneracy handled by the caller.
    config
        Solver-specific config object or a dict of fields for the selected solver.
    continuation
        Explicit coarse-to-fine continuation controls. The legacy
        ``nk_coarse``/``coarse_*`` inputs are still accepted.
    """

    solver_key = _normalize_solver(solver)
    n_target = _resolve_density_target(
        n_electrons_per_degeneracy=n_electrons_per_degeneracy,
        electrondensity0=electrondensity0,
    )
    P0, params0 = _normalize_seed_inputs(seed=seed, P0=P0, params0=params0)
    continuation_cfg = _normalize_continuation(
        continuation=continuation,
        nk_coarse=nk_coarse,
        coarse_problem=coarse_problem,
        coarse_P0=coarse_P0,
        coarse_n_electrons_per_degeneracy=coarse_n_electrons_per_degeneracy,
        coarse_config=coarse_config,
        fine_config=fine_config,
        T_coarse=T_coarse,
        resample_method=resample_method,
    )
    base_config = _normalize_config(solver_key, config)
    coarse_config_input = (
        continuation_cfg.coarse_config
        if continuation_cfg is not None and continuation_cfg.coarse_config is not None
        else coarse_config
    )
    fine_config_input = (
        continuation_cfg.fine_config
        if continuation_cfg is not None and continuation_cfg.fine_config is not None
        else fine_config
    )
    coarse_cfg = _normalize_config(
        solver_key,
        coarse_config_input
        if coarse_config_input is not None
        else _default_coarse_config(solver_key, base_config),
    )
    fine_cfg = _normalize_config(
        solver_key,
        fine_config_input if fine_config_input is not None else base_config,
    )

    if continuation_cfg is None:
        fine = _solve_stage(
            problem,
            solver=solver_key,
            P0=P0,
            params0=params0,
            n_target=n_target,
            config=fine_cfg,
        )
        return SolveResult(solver=solver_key, fine=fine)

    nk_coarse = int(continuation_cfg.nk_coarse)
    nk_f = int(np.asarray(problem.hamiltonian).shape[0])
    if nk_coarse >= nk_f and continuation_cfg.coarse_problem is None:
        fine = _solve_stage(
            problem,
            solver=solver_key,
            P0=P0,
            params0=params0,
            n_target=n_target,
            config=fine_cfg,
        )
        return SolveResult(solver=solver_key, fine=fine)
    if nk_coarse <= 0:
        raise ValueError("nk_coarse must be a positive integer or None.")

    P0_f = _density_seed_from_inputs(problem, P0=P0, params0=params0)
    if continuation_cfg.coarse_problem is None:
        coarse_problem = _resample_problem(
            problem,
            nk_coarse=nk_coarse,
            T_coarse=continuation_cfg.T_coarse,
            method=continuation_cfg.resample_method,
        )
    else:
        coarse_problem = _prepared_coarse_problem(
            continuation_cfg.coarse_problem,
            T_coarse=continuation_cfg.T_coarse,
        )

    P0_c = (
        _resample_density_seed(P0_f, nk_coarse=nk_coarse, method=continuation_cfg.resample_method)
        if continuation_cfg.coarse_seed is None
        else hermitize(jnp.asarray(continuation_cfg.coarse_seed.density_matrix))
    )

    coarse_target = (
        float(continuation_cfg.coarse_n_electrons_per_degeneracy)
        if continuation_cfg.coarse_n_electrons_per_degeneracy is not None
        else n_target
    )

    coarse = _solve_stage(
        coarse_problem,
        solver=solver_key,
        P0=P0_c,
        params0=None,
        n_target=coarse_target,
        config=coarse_cfg,
    )

    Sigma_c = hermitize(jnp.asarray(coarse.fock) - jnp.asarray(coarse_problem.hamiltonian))
    Sigma_seed_f = hermitize(resample_kgrid(Sigma_c, nk_f, method=continuation_cfg.resample_method))
    P0_seed_f, _mu_seed = density_matrix_from_fock(
        hermitize(jnp.asarray(problem.hamiltonian) + Sigma_seed_f),
        jnp.asarray(problem.weights),
        n_electrons=float(n_target),
        T=float(problem.T),
    )

    fine = _solve_stage(
        problem,
        solver=solver_key,
        P0=P0_seed_f,
        params0=None,
        n_target=n_target,
        config=fine_cfg,
    )
    return SolveResult(
        solver=solver_key,
        fine=fine,
        coarse=coarse,
        Sigma_seed_f=Sigma_seed_f,
        P0_seed_f=P0_seed_f,
    )


def run_scf(
    problem: HFProblem,
    *,
    P0: Any,
    n_electrons_per_degeneracy: float | None = None,
    electrondensity0: float | None = None,
    config: SCFRunConfig | dict[str, Any] | None = None,
) -> SolveStageResult:
    """Run the DIIS-style SCF solver and return a standardized stage result."""

    return solve(
        problem,
        solver="scf",
        P0=P0,
        n_electrons_per_degeneracy=n_electrons_per_degeneracy,
        electrondensity0=electrondensity0,
        config=config,
    ).fine


def run_scf_coarse_to_fine(
    problem: HFProblem,
    *,
    P0_f: Any,
    n_electrons_per_degeneracy: float | None = None,
    electrondensity0: float | None = None,
    T_coarse: float | None = None,
    nk_coarse: int | None = None,
    resample_method: str = "linear",
    coarse_config: SCFRunConfig | dict[str, Any] | None = None,
    fine_config: SCFRunConfig | dict[str, Any] | None = None,
) -> SolveResult:
    """Run coarse-to-fine DIIS continuation via the public solve API."""

    return solve(
        problem,
        solver="scf",
        P0=P0_f,
        n_electrons_per_degeneracy=n_electrons_per_degeneracy,
        electrondensity0=electrondensity0,
        nk_coarse=nk_coarse,
        coarse_config=coarse_config,
        fine_config=fine_config,
        T_coarse=T_coarse,
        resample_method=resample_method,
    )


def run_variational_qr(
    problem: HFProblem,
    *,
    P0: Any | None = None,
    params0: VariationalHFParams | None = None,
    n_electrons_per_degeneracy: float | None = None,
    electrondensity0: float | None = None,
    config: QRRunConfig | dict[str, Any] | None = None,
) -> SolveStageResult:
    """Run the variational QR solver and return a standardized stage result."""

    return solve(
        problem,
        solver="qr",
        P0=P0,
        params0=params0,
        n_electrons_per_degeneracy=n_electrons_per_degeneracy,
        electrondensity0=electrondensity0,
        config=config,
    ).fine


def run_variational_qr_coarse_to_fine(
    problem: HFProblem,
    *,
    P0_f: Any | None = None,
    params0: VariationalHFParams | None = None,
    n_electrons_per_degeneracy: float | None = None,
    electrondensity0: float | None = None,
    T_coarse: float | None = None,
    nk_coarse: int | None = None,
    resample_method: str = "linear",
    coarse_config: QRRunConfig | dict[str, Any] | None = None,
    fine_config: QRRunConfig | dict[str, Any] | None = None,
) -> SolveResult:
    """Run coarse-to-fine QR continuation via the public solve API."""

    return solve(
        problem,
        solver="qr",
        P0=P0_f,
        params0=params0,
        n_electrons_per_degeneracy=n_electrons_per_degeneracy,
        electrondensity0=electrondensity0,
        nk_coarse=nk_coarse,
        coarse_config=coarse_config,
        fine_config=fine_config,
        T_coarse=T_coarse,
        resample_method=resample_method,
    )


def run_variational_rtr(
    problem: HFProblem,
    *,
    P0: Any | None = None,
    params0: VariationalHFParams | None = None,
    n_electrons_per_degeneracy: float | None = None,
    electrondensity0: float | None = None,
    config: RTRRunConfig | dict[str, Any] | None = None,
) -> SolveStageResult:
    """Run the variational RTR solver and return a standardized stage result."""

    return solve(
        problem,
        solver="rtr",
        P0=P0,
        params0=params0,
        n_electrons_per_degeneracy=n_electrons_per_degeneracy,
        electrondensity0=electrondensity0,
        config=config,
    ).fine


def run_variational_rtr_coarse_to_fine(
    problem: HFProblem,
    *,
    P0_f: Any | None = None,
    params0: VariationalHFParams | None = None,
    n_electrons_per_degeneracy: float | None = None,
    electrondensity0: float | None = None,
    T_coarse: float | None = None,
    nk_coarse: int | None = None,
    resample_method: str = "linear",
    coarse_config: RTRRunConfig | dict[str, Any] | None = None,
    fine_config: RTRRunConfig | dict[str, Any] | None = None,
) -> SolveResult:
    """Run coarse-to-fine RTR continuation via the public solve API."""

    return solve(
        problem,
        solver="rtr",
        P0=P0_f,
        params0=params0,
        n_electrons_per_degeneracy=n_electrons_per_degeneracy,
        electrondensity0=electrondensity0,
        nk_coarse=nk_coarse,
        coarse_config=coarse_config,
        fine_config=fine_config,
        T_coarse=T_coarse,
        resample_method=resample_method,
    )


def _normalize_solver(solver: str) -> SolverName:
    key = str(solver).strip().lower().replace("-", "_")
    if key in ("scf", "jax", "jax_hf", "jaxhf"):
        return "scf"
    if key in ("qr", "variational_qr", "qr_variational"):
        return "qr"
    if key in ("rtr", "variational_rtr", "rtr_variational"):
        return "rtr"
    raise ValueError(f"Unknown solver {solver!r}. Expected one of 'scf', 'qr', or 'rtr'.")


def _normalize_seed_inputs(
    *,
    seed: DensityMatrixSeed | VariationalSeed | None,
    P0: Any | None,
    params0: VariationalHFParams | None,
) -> tuple[Any | None, VariationalHFParams | None]:
    if seed is None:
        return P0, params0
    if P0 is not None or params0 is not None:
        raise ValueError("Use either seed=... or legacy P0/params0 inputs, not both.")
    if isinstance(seed, DensityMatrixSeed):
        return seed.density_matrix, None
    if isinstance(seed, VariationalSeed):
        return None, seed.params
    raise TypeError(f"Unsupported seed type {type(seed).__name__}.")


def _normalize_continuation(
    *,
    continuation: ContinuationConfig | None,
    nk_coarse: int | None,
    coarse_problem: HFProblem | None,
    coarse_P0: Any | None,
    coarse_n_electrons_per_degeneracy: float | None,
    coarse_config: Any | None,
    fine_config: Any | None,
    T_coarse: float | None,
    resample_method: str,
) -> ContinuationConfig | None:
    legacy_requested = any(
        value is not None
        for value in (
            nk_coarse,
            coarse_problem,
            coarse_P0,
            coarse_n_electrons_per_degeneracy,
            coarse_config,
            fine_config,
            T_coarse,
        )
    ) or resample_method != "linear"
    if continuation is None:
        if nk_coarse is None:
            if legacy_requested:
                raise ValueError(
                    "Legacy coarse-to-fine inputs require nk_coarse to be set."
                )
            return None
        coarse_seed = None if coarse_P0 is None else DensityMatrixSeed(coarse_P0)
        return ContinuationConfig(
            nk_coarse=int(nk_coarse),
            coarse_problem=coarse_problem,
            coarse_seed=coarse_seed,
            coarse_n_electrons_per_degeneracy=coarse_n_electrons_per_degeneracy,
            coarse_config=coarse_config,
            fine_config=fine_config,
            T_coarse=T_coarse,
            resample_method=resample_method,
        )
    if legacy_requested:
        raise ValueError(
            "Use either continuation=ContinuationConfig(...) or legacy nk_coarse/coarse_* inputs, not both."
        )
    return continuation


def _resolve_density_target(
    *,
    n_electrons_per_degeneracy: float | None,
    electrondensity0: float | None,
) -> float:
    if n_electrons_per_degeneracy is None and electrondensity0 is None:
        raise TypeError("n_electrons_per_degeneracy is required.")
    if n_electrons_per_degeneracy is None:
        return float(electrondensity0)
    if electrondensity0 is not None and float(electrondensity0) != float(n_electrons_per_degeneracy):
        raise ValueError(
            "Received both n_electrons_per_degeneracy and electrondensity0 with different values."
        )
    return float(n_electrons_per_degeneracy)


def _normalize_config(solver: SolverName, config: Any | None):
    config_cls = _config_class(solver)
    if config is None:
        cfg = config_cls()
    elif isinstance(config, config_cls):
        cfg = config
    elif isinstance(config, dict):
        cfg = config_cls(**config)
    else:
        raise TypeError(
            f"{solver!r} solver expects config of type {config_cls.__name__} or dict, "
            f"got {type(config).__name__}."
        )
    if getattr(cfg, "project_fn", None) is not None:
        cfg = replace(cfg, project_fn=_wrap_project_fn(cfg.project_fn))
    return cfg


def _default_coarse_config(solver: SolverName, config):
    if solver in ("qr", "rtr") and getattr(config, "init_method", None) != "eigh":
        return replace(config, init_method="eigh")
    return config


def _config_class(solver: SolverName):
    if solver == "scf":
        return SCFRunConfig
    if solver == "qr":
        return QRRunConfig
    if solver == "rtr":
        return RTRRunConfig
    raise AssertionError(f"Unhandled solver {solver!r}")


def _wrap_project_fn(project_fn):
    def wrapped(A):
        Ah = hermitize(jnp.asarray(A))
        return hermitize(jnp.asarray(project_fn(Ah), dtype=Ah.dtype))

    return wrapped


def _density_seed_from_inputs(
    problem: HFProblem,
    *,
    P0: Any | None,
    params0: VariationalHFParams | None,
) -> jnp.ndarray:
    if params0 is not None:
        return _density_from_params(params0, dtype=jnp.asarray(problem.hamiltonian).dtype)
    if P0 is None:
        raise ValueError("Either P0 or params0 must be provided.")
    return hermitize(jnp.asarray(P0))


def _density_from_params(params0: VariationalHFParams, *, dtype) -> jnp.ndarray:
    q = jnp.asarray(params0.Q, dtype=dtype)
    p = jnp.asarray(params0.p, dtype=jnp.zeros((), dtype=dtype).real.dtype)
    return hermitize(jnp.einsum("...ia,...a,...ja->...ij", q, p, jnp.conj(q)))


def _resample_problem(
    problem: HFProblem,
    *,
    nk_coarse: int,
    T_coarse: float | None,
    method: str,
) -> HFProblem:
    weights_f = jnp.asarray(problem.weights)
    weights_c = jnp.real(resample_kgrid(weights_f, nk_coarse, method=method))
    wsum_f = jnp.sum(weights_f)
    wsum_c = jnp.sum(weights_c)
    if float(wsum_c) == 0.0:
        raise ValueError("Resampled coarse weights sum to zero; cannot renormalize.")
    weights_c = weights_c * (wsum_f / wsum_c)

    reference_density_c = None
    if problem.reference_density is not None:
        reference_density_c = hermitize(
            resample_kgrid(jnp.asarray(problem.reference_density), nk_coarse, method=method)
        )

    return HFProblem(
        weights=weights_c,
        hamiltonian=hermitize(resample_kgrid(jnp.asarray(problem.hamiltonian), nk_coarse, method=method)),
        coulomb_q=resample_kgrid(jnp.asarray(problem.coulomb_q), nk_coarse, method=method),
        T=float(problem.T if T_coarse is None else T_coarse),
        include_hartree=bool(problem.include_hartree),
        include_exchange=bool(problem.include_exchange),
        reference_density=reference_density_c,
        hartree_matrix=problem.hartree_matrix,
    )


def _prepared_coarse_problem(problem: HFProblem, *, T_coarse: float | None) -> HFProblem:
    if T_coarse is None:
        return problem
    return replace(problem, T=float(T_coarse))


def _resample_density_seed(P0_f: Any, *, nk_coarse: int, method: str) -> jnp.ndarray:
    return hermitize(resample_kgrid(jnp.asarray(P0_f), nk_coarse, method=method))


def _solve_stage(
    problem: HFProblem,
    *,
    solver: SolverName,
    P0: Any | None,
    params0: VariationalHFParams | None,
    n_target: float,
    config,
) -> SolveStageResult:
    kernel = problem.kernel()

    if solver == "scf":
        if params0 is not None:
            raise ValueError("params0 is only supported for variational solvers.")
        if P0 is None:
            raise ValueError("P0 is required for the SCF solver.")
        run = jit_hartreefock_iteration(kernel)
        density, fock, energy, mu, n_iter, history = run(
            hermitize(jnp.asarray(P0)),
            float(n_target),
            **config.as_run_kwargs(),
        )
        return _make_stage_result(
            solver=solver,
            density=density,
            fock=fock,
            energy=energy,
            mu=mu,
            n_iter=n_iter,
            history=history,
            params=None,
        )

    if solver == "qr":
        run = jit_variational_qr_iteration(kernel)
    elif solver == "rtr":
        run = jit_variational_rtr_iteration(kernel)
    else:
        raise AssertionError(f"Unhandled solver {solver!r}")

    if params0 is None:
        if P0 is None:
            raise ValueError("Either P0 or params0 must be provided.")
        density, fock, energy, mu, n_iter, history, params = run(
            hermitize(jnp.asarray(P0)),
            electrondensity0=float(n_target),
            init_method=str(config.init_method),
            return_params=True,
            **config.as_run_kwargs(),
        )
    else:
        if not isinstance(params0, VariationalHFParams):
            raise TypeError("params0 must be VariationalHFParams(Q, p, mu)")
        density, fock, energy, mu, n_iter, history, params = run(
            electrondensity0=float(n_target),
            params0=params0,
            return_params=True,
            **config.as_run_kwargs(),
        )
    return _make_stage_result(
        solver=solver,
        density=density,
        fock=fock,
        energy=energy,
        mu=mu,
        n_iter=n_iter,
        history=history,
        params=params,
    )


def _make_stage_result(
    *,
    solver: SolverName,
    density: Any,
    fock: Any,
    energy: Any,
    mu: Any,
    n_iter: Any,
    history: dict[str, Any],
    params: Any | None,
) -> SolveStageResult:
    n_iter_int = int(n_iter)
    return SolveStageResult(
        solver=solver,
        density=density,
        fock=fock,
        energy=energy,
        mu=mu,
        n_iter=n_iter_int,
        history=_standardize_history(history, n_iter_int),
        params=params,
    )


def _standardize_history(history: dict[str, Any], n_iter: int) -> dict[str, np.ndarray]:
    trimmed = {key: np.asarray(values)[:n_iter] for key, values in history.items()}
    base = _history_array(trimmed.get("E"), n_iter)
    dC = _history_array(trimmed.get("dC"), n_iter, like=base)
    dP = _history_array(trimmed.get("dP"), n_iter, like=base)
    mu = _history_array(trimmed.get("mu"), n_iter, like=base)
    dE = _history_array(trimmed.get("dE"), n_iter, like=base)
    if "dE" not in trimmed:
        dE = _compute_dE(base)

    out = {
        "E": base,
        "dC": dC,
        "dP": dP,
        "dE": dE,
        "mu": mu,
        "n_iter": np.asarray(n_iter, dtype=int),
    }
    for key, values in trimmed.items():
        out.setdefault(key, values)
    return out


def _history_array(values: np.ndarray | None, n_iter: int, *, like: np.ndarray | None = None) -> np.ndarray:
    if values is not None:
        return np.asarray(values)
    dtype = np.asarray(like).dtype if like is not None and np.asarray(like).size else float
    return np.full((n_iter,), np.nan, dtype=dtype)


def _compute_dE(energies: np.ndarray) -> np.ndarray:
    energies = np.asarray(energies)
    if energies.size == 0:
        return energies.astype(float)
    dE = np.full_like(energies, np.nan, dtype=float)
    if energies.size > 1:
        dE[1:] = np.abs(np.diff(np.real(energies)))
    return dE


__all__ = [
    "HFProblem",
    "SCFRunConfig",
    "QRRunConfig",
    "RTRRunConfig",
    "DensityMatrixSeed",
    "VariationalSeed",
    "ContinuationConfig",
    "SolveStageResult",
    "SolveResult",
    "solve",
    "run_scf",
    "run_scf_coarse_to_fine",
    "run_variational_qr",
    "run_variational_qr_coarse_to_fine",
    "run_variational_rtr",
    "run_variational_rtr_coarse_to_fine",
]
