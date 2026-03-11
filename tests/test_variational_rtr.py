from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jax_hf
from jax_hf.main import HartreeFockKernel
from jax_hf.variational import (
    init_variational_params_from_density,
)


def _electron_count(P: jax.Array, weights: jax.Array) -> float:
    tr = jnp.real(jnp.trace(P, axis1=-2, axis2=-1))
    return float(jnp.sum(weights * tr))


def test_init_variational_params_from_density_enforces_particle_number():
    weights = jnp.ones((1, 1), dtype=jnp.float32)
    hamiltonian = jnp.array([[[[-0.5, 0.0], [0.0, 0.5]]]], dtype=jnp.complex64)
    coulomb_q = jnp.array([[[[0.25]]]], dtype=jnp.complex64)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.2,
        include_hartree=False,
        include_exchange=True,
    )

    P0 = jnp.array([[[[0.6, 0.0], [0.0, 0.4]]]], dtype=jnp.complex64)
    params = init_variational_params_from_density(
        P0,
        electrondensity0=1.0,
        weights_b=kernel.weights_b,
        weight_sum=kernel.weight_sum,
        method="identity",
    )

    total = float(jnp.sum(kernel.w2d[..., None] * params.p))
    assert total == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_variational_solver_smoke_conserves_density_and_supports_param_reuse():
    weights = jnp.ones((1, 1), dtype=jnp.float32)
    hamiltonian = jnp.array([[[[-0.5, 0.0], [0.0, 0.5]]]], dtype=jnp.complex64)
    coulomb_q = jnp.array([[[[0.25]]]], dtype=jnp.complex64)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.2,
        include_hartree=False,
        include_exchange=True,
    )
    runner = jax_hf.jit_variational_rtr_iteration(kernel)

    P0 = jnp.array([[[[0.6, 0.0], [0.0, 0.4]]]], dtype=jnp.complex64)

    P_fin, F_fin, E_fin, mu_fin, k_fin, history, params_fin = runner(
        P0,
        electrondensity0=1.0,
        max_iter=80,
        comm_tol=1e-6,
        p_tol=1e-6,
        max_rot=0.25,
        return_params=True,
        init_method="identity",
        block_sizes=(1, 1),
    )

    assert int(k_fin) <= 80
    assert np.isfinite(np.array(E_fin)).all()
    assert np.isfinite(np.array(mu_fin)).all()

    np.testing.assert_allclose(
        np.array(P_fin),
        np.array(jnp.conj(jnp.swapaxes(P_fin, -1, -2))),
        atol=1e-6,
    )

    # Fixed-N constraint is enforced by the Fermi-Dirac chemical-potential solve.
    n_fin = _electron_count(P_fin, weights)
    assert n_fin == pytest.approx(1.0, rel=1e-6, abs=1e-6)

    assert history["dC"].shape[0] == 80
    assert history["dP"].shape[0] == 80
    assert history["mu"].shape[0] == 80

    # Continuation path: pass params back in and ensure API works.
    P2, F2, E2, mu2, k2, history2, params2 = runner(
        P_fin,
        electrondensity0=1.0,
        params0=params_fin,
        max_iter=5,
        comm_tol=1e-6,
        p_tol=1e-6,
        return_params=True,
    )

    assert P2.shape == P_fin.shape
    assert F2.shape == F_fin.shape
    assert np.isfinite(np.array(E2)).all()
    assert np.isfinite(np.array(mu2)).all()
    assert int(k2) <= 5
    assert params2.Q.shape == params_fin.Q.shape
    assert history2["E"].shape[0] == 5


def test_variational_solver_default_init_method_matches_identity():
    weights = jnp.ones((1, 1), dtype=jnp.float32)
    hamiltonian = jnp.array([[[[-0.5, 0.0], [0.0, 0.5]]]], dtype=jnp.complex64)
    coulomb_q = jnp.array([[[[0.25]]]], dtype=jnp.complex64)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.2,
        include_hartree=False,
        include_exchange=True,
    )
    runner = jax_hf.jit_variational_rtr_iteration(kernel)

    P0 = jnp.array([[[[0.6, 0.0], [0.0, 0.4]]]], dtype=jnp.complex64)

    P_def, _F_def, E_def, mu_def, k_def, hist_def = runner(
        P0,
        electrondensity0=1.0,
        max_iter=40,
        comm_tol=1e-6,
        p_tol=1e-6,
    )
    P_id, _F_id, E_id, mu_id, k_id, hist_id = runner(
        P0,
        electrondensity0=1.0,
        max_iter=40,
        comm_tol=1e-6,
        p_tol=1e-6,
        init_method="identity",
    )

    np.testing.assert_allclose(np.array(P_def), np.array(P_id), atol=1e-7, rtol=1e-7)
    assert float(E_def) == pytest.approx(float(E_id), rel=1e-7, abs=1e-7)
    assert float(mu_def) == pytest.approx(float(mu_id), rel=1e-7, abs=1e-7)
    assert int(k_def) == int(k_id)
    np.testing.assert_allclose(np.array(hist_def["dC"]), np.array(hist_id["dC"]), atol=1e-7, rtol=1e-7)


def test_variational_solver_reduces_occupancy_residual_on_random_system():
    key_h, key_i = jax.random.split(jax.random.PRNGKey(123))
    nk1, nk2, nb = 2, 2, 3

    a = jax.random.normal(key_h, (nk1, nk2, nb, nb), dtype=jnp.float32)
    b = jax.random.normal(key_i, (nk1, nk2, nb, nb), dtype=jnp.float32)
    hamiltonian = (a + 1j * b).astype(jnp.complex64)
    hamiltonian = 0.5 * (hamiltonian + jnp.conj(jnp.swapaxes(hamiltonian, -1, -2)))

    weights = jnp.ones((nk1, nk2), dtype=jnp.float32)
    coulomb_q = (0.1 * jnp.ones((nk1, nk2, 1, 1), dtype=jnp.float32)).astype(jnp.complex64)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.01,
        include_hartree=False,
        include_exchange=True,
    )
    runner = jax_hf.jit_variational_rtr_iteration(kernel)

    P0 = jnp.eye(nb, dtype=jnp.complex64)[None, None, ...]
    P0 = jnp.broadcast_to(P0, (nk1, nk2, nb, nb)) * 0.5

    P_fin, _F_fin, _E_fin, _mu_fin, k_fin, history = runner(
        P0,
        electrondensity0=1.7,
        max_iter=120,
        comm_tol=1e-6,
        p_tol=1e-6,
        max_rot=0.25,
    )

    last = max(int(k_fin) - 1, 0)
    d_p_last = float(history["dP"][last])
    assert d_p_last < 5e-2

    n_fin = _electron_count(P_fin, weights)
    assert n_fin == pytest.approx(1.7, rel=1e-5, abs=1e-5)


def test_variational_solver_can_break_degenerate_occupancy_symmetry():
    """Regression: Q updates must not freeze when p_i == p_j initially."""
    weights = jnp.ones((1, 1), dtype=jnp.float32)
    v = 0.2
    hamiltonian = jnp.array([[[[0.0, v], [v, 0.0]]]], dtype=jnp.complex64)
    coulomb_q = (1e-6 * jnp.ones((1, 1, 1, 1), dtype=jnp.float32)).astype(jnp.complex64)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.02,
        include_hartree=False,
        include_exchange=True,
    )
    runner = jax_hf.jit_variational_rtr_iteration(kernel)

    # Start from exactly degenerate occupations: p1 = p2 = 0.5.
    P0 = (0.5 * jnp.eye(2, dtype=jnp.complex64))[None, None, ...]

    P_fin, _F_fin, E_fin, _mu_fin, _k_fin, _history = runner(
        P0,
        electrondensity0=1.0,
        max_iter=120,
        comm_tol=1e-8,
        p_tol=1e-8,
        max_rot=0.7,
        init_method="identity",
    )

    # If Q rotations freeze, solver gets stuck at P = 0.5 I with near-zero energy.
    assert float(E_fin) < -5e-2
    assert float(jnp.abs(P_fin[0, 0, 0, 1])) > 1e-1


def test_variational_solver_stops_before_max_iter_with_moderate_p_tol():
    key_h, key_i = jax.random.split(jax.random.PRNGKey(321))
    nk1, nk2, nb = 2, 2, 3

    a = jax.random.normal(key_h, (nk1, nk2, nb, nb), dtype=jnp.float32)
    b = jax.random.normal(key_i, (nk1, nk2, nb, nb), dtype=jnp.float32)
    hamiltonian = (a + 1j * b).astype(jnp.complex64)
    hamiltonian = 0.5 * (hamiltonian + jnp.conj(jnp.swapaxes(hamiltonian, -1, -2)))

    weights = jnp.ones((nk1, nk2), dtype=jnp.float32)
    coulomb_q = (0.1 * jnp.ones((nk1, nk2, 1, 1), dtype=jnp.float32)).astype(jnp.complex64)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.01,
        include_hartree=False,
        include_exchange=True,
    )
    runner = jax_hf.jit_variational_rtr_iteration(kernel)

    P0 = jnp.eye(nb, dtype=jnp.complex64)[None, None, ...]
    P0 = jnp.broadcast_to(P0, (nk1, nk2, nb, nb)) * 0.5

    max_iter = 120
    comm_tol = 8e-4
    p_tol = 8e-4
    _P_fin, _F_fin, _E_fin, _mu_fin, k_fin, history = runner(
        P0,
        electrondensity0=1.7,
        max_iter=max_iter,
        comm_tol=comm_tol,
        p_tol=p_tol,
    )

    assert int(k_fin) < max_iter
    last = max(int(k_fin) - 1, 0)
    assert float(history["dC"][last]) <= comm_tol
    assert float(history["dP"][last]) <= p_tol


def test_rotation_block_sizes_preserves_block_diagonal_density():
    """rotation_block_sizes=(2,2) must keep P block-diagonal if started so."""
    nk1, nk2, nb = 1, 1, 4
    # Block-diagonal Hamiltonian: two 2×2 blocks with different energies.
    h_block = np.zeros((nk1, nk2, nb, nb), dtype=np.complex64)
    h_block[0, 0, :2, :2] = np.array([[-1.0, 0.3], [0.3, 1.0]])
    h_block[0, 0, 2:, 2:] = np.array([[-0.5, 0.2], [0.2, 0.8]])
    hamiltonian = jnp.asarray(h_block)

    weights = jnp.ones((nk1, nk2), dtype=jnp.float32)
    coulomb_q = (0.1 * jnp.ones((nk1, nk2, 1, 1), dtype=jnp.float32)).astype(jnp.complex64)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.1,
        include_hartree=False,
        include_exchange=True,
    )
    runner = jax_hf.jit_variational_rtr_iteration(kernel)

    # Block-diagonal initial density
    P0 = 0.5 * jnp.eye(nb, dtype=jnp.complex64)[None, None, ...]

    # With rotation_block_sizes: off-block entries of P must stay zero.
    P_fin, _F, _E, _mu, _k, _hist = runner(
        P0,
        electrondensity0=2.0,
        max_iter=60,
        comm_tol=1e-6,
        p_tol=1e-6,
        block_sizes=(2, 2),
    )
    P_arr = np.array(P_fin[0, 0])
    offblock = np.concatenate([P_arr[:2, 2:].ravel(), P_arr[2:, :2].ravel()])
    assert np.max(np.abs(offblock)) < 1e-10, (
        f"rotation_block_sizes should prevent inter-block mixing; "
        f"max off-block |P| = {np.max(np.abs(offblock)):.2e}"
    )

    # Without constraint: solver is free to mix blocks (off-block may be nonzero).
    P_free, _, _, _, _, _ = runner(
        P0,
        electrondensity0=2.0,
        max_iter=60,
        comm_tol=1e-6,
        p_tol=1e-6,
    )
    # (We don't assert off-block is nonzero — it depends on the problem —
    #  but the constrained run must be at least as block-diagonal.)
