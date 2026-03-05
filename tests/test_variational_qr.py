from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jax_hf
from jax_hf.main import HartreeFockKernel
from jax_hf.variational import (
    VariationalHFParams,
    init_variational_params_from_density,
)


def _electron_count(P: jax.Array, weights: jax.Array) -> float:
    tr = jnp.real(jnp.trace(P, axis1=-2, axis2=-1))
    return float(jnp.sum(weights * tr))


def test_variational_qr_smoke_conserves_density_and_supports_param_reuse():
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
    runner = jax_hf.jit_variational_qr_iteration(kernel)

    P0 = jnp.array([[[[0.6, 0.0], [0.0, 0.4]]]], dtype=jnp.complex64)

    P_fin, F_fin, E_fin, mu_fin, k_fin, history, params_fin = runner(
        P0,
        electrondensity0=1.0,
        max_iter=80,
        comm_tol=1e-6,
        p_tol=1e-6,
        return_params=True,
        init_method="identity",
        exchange_block_specs=[{"block_sizes": [1, 1]}],
        exchange_check_offdiag=True,
    )

    assert int(k_fin) <= 80
    assert np.isfinite(np.array(E_fin)).all()
    assert np.isfinite(np.array(mu_fin)).all()

    np.testing.assert_allclose(
        np.array(P_fin),
        np.array(jnp.conj(jnp.swapaxes(P_fin, -1, -2))),
        atol=1e-6,
    )

    # Fixed-N constraint
    n_fin = _electron_count(P_fin, weights)
    assert n_fin == pytest.approx(1.0, rel=1e-6, abs=1e-6)

    assert history["dC"].shape[0] == 80
    assert history["dP"].shape[0] == 80
    assert history["dE"].shape[0] == 80
    assert history["mu"].shape[0] == 80

    # Continuation path: pass params back in
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


@pytest.mark.parametrize("line_search", [False, True])
def test_variational_qr_reduces_occupancy_residual_on_random_system(line_search: bool):
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
    runner = jax_hf.jit_variational_qr_iteration(kernel)

    P0 = jnp.eye(nb, dtype=jnp.complex64)[None, None, ...]
    P0 = jnp.broadcast_to(P0, (nk1, nk2, nb, nb)) * 0.5

    P_fin, _F_fin, _E_fin, _mu_fin, k_fin, history = runner(
        P0,
        electrondensity0=1.7,
        max_iter=120,
        comm_tol=1e-6,
        p_tol=1e-6,
        line_search=line_search,
    )

    last = max(int(k_fin) - 1, 0)
    d_p_last = float(history["dP"][last])
    assert d_p_last < 5e-2

    n_fin = _electron_count(P_fin, weights)
    assert n_fin == pytest.approx(1.7, rel=1e-5, abs=1e-5)


def test_variational_qr_updates_occupations_even_when_orbitals_are_stationary():
    weights = jnp.ones((1, 1), dtype=jnp.float32)
    hamiltonian = jnp.array([[[[-1.0, 0.0], [0.0, 1.0]]]], dtype=jnp.complex64)
    coulomb_q = (1e-9 * jnp.ones((1, 1, 1, 1), dtype=jnp.float32)).astype(jnp.complex64)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.2,
        include_hartree=False,
        include_exchange=True,
    )
    runner = jax_hf.jit_variational_qr_iteration(kernel)

    P0 = jnp.array([[[[0.9, 0.0], [0.0, 0.1]]]], dtype=jnp.complex64)

    P_fin, _F_fin, _E_fin, _mu_fin, k_fin, history = runner(
        P0,
        electrondensity0=1.0,
        max_iter=8,
        comm_tol=1e-8,
        p_tol=1e-8,
        init_method="identity",
    )

    p_fin = np.array(jnp.real(jnp.diagonal(P_fin, axis1=-2, axis2=-1)))[0, 0]
    assert int(k_fin) >= 2
    assert float(history["dC"][0]) == pytest.approx(0.0, abs=1e-12)
    assert float(history["dP"][0]) > 1e-2
    assert p_fin[0] > 0.98
    assert p_fin[1] < 0.02


def test_variational_qr_multisweep_cg_path_runs_with_batched_kmesh():
    key_h, key_i = jax.random.split(jax.random.PRNGKey(777))
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
        T=0.02,
        include_hartree=False,
        include_exchange=True,
    )
    runner = jax_hf.jit_variational_qr_iteration(kernel)

    P0 = jnp.eye(nb, dtype=jnp.complex64)[None, None, ...]
    P0 = jnp.broadcast_to(P0, (nk1, nk2, nb, nb)) * 0.5

    P_fin, _F_fin, _E_fin, _mu_fin, k_fin, history = runner(
        P0,
        electrondensity0=1.8,
        max_iter=80,
        comm_tol=1e-6,
        p_tol=1e-6,
        q_sweeps=2,
        line_search=True,
    )

    last = max(int(k_fin) - 1, 0)
    assert int(k_fin) <= 80
    assert np.isfinite(np.array(history["dC"][last]))
    assert np.isfinite(np.array(history["dP"][last]))
    assert _electron_count(P_fin, weights) == pytest.approx(1.8, rel=1e-5, abs=1e-5)


def test_variational_qr_can_break_degenerate_occupancy_symmetry():
    """Q updates must not freeze when p_i == p_j initially."""
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
    runner = jax_hf.jit_variational_qr_iteration(kernel)

    # Start from exactly degenerate occupations: p1 = p2 = 0.5.
    P0 = (0.5 * jnp.eye(2, dtype=jnp.complex64))[None, None, ...]

    P_fin, _F_fin, E_fin, _mu_fin, _k_fin, _history = runner(
        P0,
        electrondensity0=1.0,
        max_iter=120,
        comm_tol=1e-8,
        p_tol=1e-8,
        init_method="identity",
    )

    assert float(E_fin) < -5e-2
    assert float(jnp.abs(P_fin[0, 0, 0, 1])) > 1e-1


def test_variational_qr_stops_before_max_iter():
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
    runner = jax_hf.jit_variational_qr_iteration(kernel)

    P0 = jnp.eye(nb, dtype=jnp.complex64)[None, None, ...]
    P0 = jnp.broadcast_to(P0, (nk1, nk2, nb, nb)) * 0.5

    max_iter = 200
    comm_tol = 8e-4
    p_tol = 8e-4
    _P_fin, _F_fin, _E_fin, _mu_fin, k_fin, history = runner(
        P0,
        electrondensity0=1.7,
        max_iter=max_iter,
        comm_tol=comm_tol,
        p_tol=p_tol,
        inner_sweeps=4,
    )

    assert int(k_fin) < max_iter
    last = max(int(k_fin) - 1, 0)
    assert float(history["dC"][last]) <= comm_tol
    assert float(history["dP"][last]) <= p_tol


def test_variational_qr_rotation_block_sizes():
    """rotation_block_sizes=(2,2) must keep P block-diagonal if started so."""
    nk1, nk2, nb = 1, 1, 4
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
    runner = jax_hf.jit_variational_qr_iteration(kernel)

    P0 = 0.5 * jnp.eye(nb, dtype=jnp.complex64)[None, None, ...]

    P_fin, _F, _E, _mu, _k, _hist = runner(
        P0,
        electrondensity0=2.0,
        max_iter=60,
        comm_tol=1e-6,
        p_tol=1e-6,
        rotation_block_sizes=(2, 2),
    )
    P_arr = np.array(P_fin[0, 0])
    offblock = np.concatenate([P_arr[:2, 2:].ravel(), P_arr[2:, :2].ravel()])
    assert np.max(np.abs(offblock)) < 1e-10, (
        f"rotation_block_sizes should prevent inter-block mixing; "
        f"max off-block |P| = {np.max(np.abs(offblock)):.2e}"
    )
