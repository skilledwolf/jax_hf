"""Tests for the new direct-minimization solver."""

import numpy as np
import pytest

import jax.numpy as jnp

from jax_hf import HartreeFockKernel, SolverConfig, SolveResult, solve


def _comm_rms(F, P, weights_2d):
    """Weighted RMS of the commutator [F, P]."""
    comm = F @ P - P @ F
    sq = jnp.abs(comm) ** 2
    per_k = jnp.sum(sq, axis=(-2, -1))
    weight_sum = jnp.sum(weights_2d)
    return float(jnp.sqrt(jnp.sum(weights_2d * per_k) / jnp.maximum(weight_sum, 1e-30)))


def _make_two_band_kernel(nk=1, T=0.2, exchange_strength=0.25):
    """Two-band model with weak exchange on an nk x nk grid."""
    weights = jnp.ones((nk, nk), dtype=jnp.float32)
    hamiltonian = np.zeros((nk, nk, 2, 2), dtype=np.complex64)
    hamiltonian[..., 0, 0] = -0.5
    hamiltonian[..., 1, 1] = 0.5
    coulomb_q = jnp.full((nk, nk, 1, 1), exchange_strength, dtype=jnp.complex64)
    return HartreeFockKernel(
        weights=weights,
        hamiltonian=jnp.asarray(hamiltonian),
        coulomb_q=coulomb_q,
        T=T,
    )


def _solve_problem(kernel, n_electrons=1.0, config=None):
    """Helper: solve, return result."""
    if config is None:
        config = SolverConfig(max_iter=100, tol_E=1e-8)
    P0 = jnp.zeros_like(kernel.h)
    return solve(kernel, P0, n_electrons, config=config)


class TestBasicConvergence:
    def test_converges_on_tiny_model(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        assert int(result.n_iter) <= 100
        assert bool(result.converged)
        assert np.isfinite(float(result.energy))
        assert np.isfinite(float(result.mu))

    def test_density_is_hermitian(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        np.testing.assert_allclose(
            np.array(result.density),
            np.array(jnp.conj(jnp.swapaxes(result.density, -1, -2))),
            atol=1e-6,
        )

    def test_self_consistency(self):
        """At convergence, [F, P] should be near zero."""
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        rms = _comm_rms(result.fock, result.density, kernel.w2d)
        assert rms < 1e-4

    def test_particle_number_conserved(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        n_total = float(jnp.sum(
            kernel.w2d[..., None] * result.p
        ))
        np.testing.assert_allclose(n_total, 1.0, atol=1e-4)

    def test_history_has_correct_length(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        n = int(result.n_iter)
        # History arrays are pre-allocated to max_iter, but first n entries
        # should be populated with finite values
        assert n > 0
        assert np.all(np.isfinite(np.array(result.history["E"][:n])))
        assert np.all(np.isfinite(np.array(result.history["grad_norm"][:n])))


class TestNonInteracting:
    """Non-interacting limit: exchange=0, solution is just diag(h)."""

    def test_converges_to_exact_occupations(self):
        kernel = _make_two_band_kernel(exchange_strength=0.0, T=0.01)
        result = _solve_problem(kernel, config=SolverConfig(max_iter=50, tol_E=1e-10))

        # With eps = [-0.5, 0.5] and T=0.01, occupation should be ~[1, 0]
        p = np.array(result.p[0, 0])
        assert p[0] > 0.99 or p[1] > 0.99  # one of them should be ~1


class TestMultiKPoint:
    def test_2x2_grid_converges(self):
        kernel = _make_two_band_kernel(nk=2, T=0.1)
        result = _solve_problem(
            kernel, n_electrons=4.0,
            config=SolverConfig(max_iter=100, tol_E=1e-7),
        )

        assert bool(result.converged)
        rms = _comm_rms(result.fock, result.density, kernel.w2d)
        assert rms < 1e-3

    def test_4x4_grid_converges(self):
        kernel = _make_two_band_kernel(nk=4, T=0.1)
        result = _solve_problem(
            kernel,
            n_electrons=16.0,
            config=SolverConfig(max_iter=200, tol_E=1e-7),
        )

        assert bool(result.converged)


class TestSolveResult:
    def test_result_is_named_tuple(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)
        assert isinstance(result, SolveResult)

    def test_shapes_match_problem(self):
        kernel = _make_two_band_kernel(nk=2)
        result = _solve_problem(kernel, n_electrons=4.0)

        assert result.density.shape == kernel.h.shape
        assert result.fock.shape == kernel.h.shape
        assert result.Q.shape == kernel.h.shape
        assert result.p.shape == kernel.h.shape[:-1]


class TestEdgeCases:
    def test_rejects_unreachable_density(self):
        kernel = _make_two_band_kernel()
        P0 = jnp.zeros_like(kernel.h)

        with pytest.raises(ValueError, match="physically reachable range"):
            solve(kernel, P0, n_electrons=3.0)

    def test_zero_exchange_strength(self):
        """Should converge quickly with no interaction."""
        kernel = _make_two_band_kernel(exchange_strength=0.0)
        result = _solve_problem(kernel, config=SolverConfig(max_iter=50, tol_E=1e-8))

        assert int(result.n_iter) <= 50
        assert np.isfinite(float(result.energy))
