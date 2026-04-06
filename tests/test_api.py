import numpy as np
import jax.numpy as jnp
import pytest

from jax_hf import (
    ContinuationConfig,
    DEFAULT_SOLVER,
    DensityMatrixSeed,
    HFProblem,
    QRRunConfig,
    SCFRunConfig,
    SolveResult,
    SolveStageResult,
    VariationalHFParams,
    VariationalSeed,
    run_scf,
    solve,
)


def _two_band_problem(nk: int = 1):
    weights = jnp.ones((nk, nk), dtype=jnp.float32)
    hamiltonian = np.zeros((nk, nk, 2, 2), dtype=np.complex64)
    hamiltonian[..., 0, 0] = -0.5
    hamiltonian[..., 1, 1] = 0.5
    coulomb_q = jnp.zeros((nk, nk, 1, 1), dtype=jnp.complex64)
    return HFProblem(
        weights=weights,
        hamiltonian=jnp.asarray(hamiltonian),
        coulomb_q=coulomb_q,
        T=1e-2,
    )


def test_run_scf_returns_standardized_history():
    problem = _two_band_problem()
    P0 = jnp.zeros_like(problem.hamiltonian)

    result = run_scf(
        problem,
        P0=P0,
        n_electrons_per_degeneracy=1.0,
        config=SCFRunConfig(max_iter=12, comm_tol=1e-8, diis_size=2),
    )

    assert isinstance(result, SolveStageResult)
    assert result.solver == "scf"
    assert result.density.shape == problem.hamiltonian.shape
    assert int(result.n_iter) <= 12
    for key in ("E", "dC", "dP", "dE", "mu"):
        assert key in result.history
        assert len(result.history[key]) == result.n_iter
    np.testing.assert_allclose(
        np.real(np.trace(np.asarray(result.density[0, 0]))),
        1.0,
        atol=1e-5,
        rtol=1e-5,
    )


def test_solve_accepts_explicit_density_seed():
    problem = _two_band_problem()
    result = solve(
        problem,
        solver="scf",
        seed=DensityMatrixSeed(jnp.zeros_like(problem.hamiltonian)),
        n_electrons_per_degeneracy=1.0,
        config=SCFRunConfig(max_iter=12, comm_tol=1e-8, diis_size=2),
    )

    assert isinstance(result, SolveResult)
    assert result.solver == "scf"
    assert result.fock.shape == problem.hamiltonian.shape
    assert result.history["n_iter"] == result.n_iter


def test_solve_defaults_to_scf():
    problem = _two_band_problem()
    result = solve(
        problem,
        seed=DensityMatrixSeed(jnp.zeros_like(problem.hamiltonian)),
        n_electrons_per_degeneracy=1.0,
        config=SCFRunConfig(max_iter=12, comm_tol=1e-8, diis_size=2),
    )

    assert DEFAULT_SOLVER == "scf"
    assert isinstance(result, SolveResult)
    assert result.solver == "scf"
    assert result.params is None


def test_run_scf_rejects_unreachable_density_target():
    problem = _two_band_problem()

    with pytest.raises(ValueError, match="physically reachable range"):
        run_scf(
            problem,
            P0=jnp.zeros_like(problem.hamiltonian),
            n_electrons_per_degeneracy=3.0,
            config=SCFRunConfig(max_iter=12, comm_tol=1e-8, diis_size=2),
        )


def test_solve_rejects_mismatched_config_type():
    problem = _two_band_problem()

    with pytest.raises(TypeError, match="SCFRunConfig"):
        solve(
            problem,
            solver="scf",
            P0=jnp.zeros_like(problem.hamiltonian),
            n_electrons_per_degeneracy=1.0,
            config=QRRunConfig(),
        )


def test_solve_rejects_mixed_seed_interfaces():
    problem = _two_band_problem()

    with pytest.raises(ValueError, match="either seed=... or legacy P0/params0"):
        solve(
            problem,
            solver="scf",
            seed=DensityMatrixSeed(jnp.zeros_like(problem.hamiltonian)),
            P0=jnp.zeros_like(problem.hamiltonian),
            n_electrons_per_degeneracy=1.0,
            config=SCFRunConfig(),
        )


def test_solve_supports_variational_seed_through_continuation_config():
    problem = _two_band_problem(nk=2)
    q0 = np.broadcast_to(np.eye(2, dtype=np.complex64), problem.hamiltonian.shape)
    p0 = np.broadcast_to(np.asarray([0.75, 0.25], dtype=np.float32), problem.hamiltonian.shape[:-1])
    params0 = VariationalSeed(
        VariationalHFParams(
        Q=jnp.asarray(q0),
        p=jnp.asarray(p0),
        mu=jnp.asarray(0.0, dtype=jnp.float32),
        )
    )

    result = solve(
        problem,
        solver="qr",
        seed=params0,
        n_electrons_per_degeneracy=1.0,
        continuation=ContinuationConfig(nk_coarse=1),
        config=QRRunConfig(max_iter=8, comm_tol=1e-6, p_tol=1e-6),
    )

    assert isinstance(result, SolveResult)
    assert result.coarse is not None
    assert result.params is not None
    assert result.P0_seed_f is not None
    assert result.Sigma_seed_f is not None
    assert result.fock.shape == problem.hamiltonian.shape
