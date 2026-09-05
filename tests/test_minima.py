import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_hf import HartreeFockKernel, SCFConfig, solve_minimum

jax.config.update("jax_enable_x64", True)


def saddle_kernel():
    return HartreeFockKernel(np.ones((1, 1)), np.zeros((1, 1, 2, 2), complex),
                            np.ones((1, 1, 1, 1)), T=.1)


def test_recovery_escapes_exact_degenerate_paramagnetic_saddle():
    result = solve_minimum(saddle_kernel(), np.eye(2)[None, None]/2, 1.)
    assert result.converged and result.stability.stable
    assert result.escapes >= 1 and result.total_iterations > result.solution.iterations
    assert result.fock_builds > result.solution.history["fock_builds"]
    assert np.all(np.diff(result.free_energies) < 0)
    p = np.linalg.eigvalsh(result.solution.density_matrix)
    assert p.max() > .99 and p.min() < .01
    np.testing.assert_allclose(p.sum(), 1., atol=1e-13)


def test_zero_escape_budget_reports_saddle_without_false_success():
    result = solve_minimum(saddle_kernel(), np.eye(2)[None, None]/2, 1., max_escapes=0)
    assert result.solution.converged and not result.converged and result.stability.stable is False
    assert result.escapes == 0


def test_recovery_rejects_conflicting_symmetry_constraint():
    with pytest.raises(ValueError, match="unconstrained"):
        solve_minimum(saddle_kernel(), np.eye(2)[None, None]/2, 1.,
                      config=SCFConfig(project_fn=lambda P: P))


def test_nonstationary_iterate_is_not_reported_as_a_minimum():
    h = np.diag([-1., 1.]).astype(complex)[None, None]
    kernel = HartreeFockKernel(np.ones((1, 1)), h, np.full((1, 1, 1, 1), .1), T=.1)
    result = solve_minimum(kernel, np.eye(2)[None, None]/2, 1.,
        config=SCFConfig(acceleration="rdiis", max_iter=1, density_tol=1e-8, comm_tol=1e-8))
    assert not result.converged and result.stability is None
    assert result.escapes == 0 and result.total_iterations == 1


def test_custom_fock_is_used_in_every_recovery_stage():
    # Standard kernel is noninteracting/stable; the custom interaction is
    # exchange-dominated and must cause the exact same escape as saddle_kernel.
    kernel = HartreeFockKernel(np.ones((1, 1)), np.zeros((1, 1, 2, 2), complex),
                              np.zeros((1, 1, 1, 1)), T=.1)
    def custom(P, *, h, refP, **kwargs):
        sigma = -(P-refP)
        return sigma, jnp.zeros_like(P), h+sigma
    result = solve_minimum(kernel, np.eye(2)[None, None]/2, 1., fock_build_fn=custom)
    assert result.converged and result.stability.stable and result.escapes >= 1


def test_near_saddle_local_stall_can_propose_a_validated_escape(monkeypatch):
    from dataclasses import replace
    import jax_hf.minima as module
    actual_solve = module.solve_scf
    calls = []
    def first_local_budget_is_short(*args, config, **kwargs):
        local_config = replace(config, max_iter=1) if not calls else config
        result = actual_solve(*args, config=local_config, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(module, "solve_scf", first_local_budget_is_short)
    initial = np.diag([.5+2e-8, .5-2e-8])[None, None]
    result = solve_minimum(saddle_kernel(), initial, 1.)
    assert not calls[0].converged
    residual = max(float(calls[0].history["density_residual"][-1]),
                   float(calls[0].history["commutator_residual"][-1]))
    assert 1e-8 < residual < 1e-6
    assert result.converged and result.stability.stable and result.escapes >= 1
    assert np.all(np.diff(result.free_energies) < 0)


def test_float32_escape_requires_descent_above_kernel_precision(monkeypatch):
    from jax_hf.reference_scf import SCFResult
    from jax_hf.stability import StabilityResult
    import jax_hf.minima as module

    h = np.zeros((1, 1, 2, 2), np.complex64)
    kernel = HartreeFockKernel(np.ones((1, 1), np.float32), h,
                              np.zeros((1, 1, 1, 1), np.float32), T=1e-9)
    density = np.eye(2, dtype=np.complex64)[None, None]/2
    local = SCFResult(density, h, 0., 0., 1, True, "test stationary state", {
        "free_energy": np.array([0.]), "density_residual": np.array([0.]),
        "commutator_residual": np.array([0.]), "fock_builds": 1,
    })
    direction = np.diag([1., -1.]).astype(complex)[None, None]
    saddle = StabilityResult(-1., 0., True, True, False, 0., 0., 1, direction)
    monkeypatch.setattr(module, "solve_scf", lambda *args, **kwargs: local)
    monkeypatch.setattr(module, "stability_analysis", lambda *args, **kwargs: saddle)
    evaluated = []

    def precision_limited_trial(P, **kwargs):
        evaluated.append(P.dtype)
        return jnp.zeros_like(P), jnp.asarray(-1e-7, dtype=jnp.float32)

    monkeypatch.setattr(module, "_evaluate_fock", precision_limited_trial)
    result = solve_minimum(kernel, density, 1., eigen_tol=1e-4,
        config=SCFConfig(acceleration="rdiis", density_tol=1e-4, comm_tol=1e-4))
    assert not result.converged and result.escapes == 0
    assert len(evaluated) == 12 and all(dtype == jnp.complex64 for dtype in evaluated)
    assert "no finite feasible descent" in result.message
