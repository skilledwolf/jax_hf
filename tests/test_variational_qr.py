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
from jax_hf.variational_qr import _adaptive_tau
from jax_hf.variational_qr import _apply_orbital_basis_update
from jax_hf.variational_qr import _apply_right_unitary
from jax_hf.variational_qr import _block_slices_from_sizes
from jax_hf.variational_qr import _exchange_block_specs_from_block_sizes
from jax_hf.variational_qr import _frobenius_ip
from jax_hf.variational_qr import _lbfgs_direction
from jax_hf.variational_qr import _orbital_gradient
from jax_hf.variational_qr import _qr_retraction_unitary


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
        block_sizes=(1, 1),
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


def test_variational_qr_multisweep_path_runs_with_batched_kmesh():
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


def test_orbital_gradient_uses_jacobi_fallback_for_equal_occupations():
    Ft = jnp.array([[[[0.0, 0.3], [0.3, 0.0]]]], dtype=jnp.complex64)
    eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)).astype(jnp.float32)
    gap = eps[..., :, None] - eps[..., None, :]
    p = jnp.array([[[0.5, 0.5]]], dtype=jnp.float32)
    offdiag = jnp.reshape(1.0 - jnp.eye(2, dtype=jnp.float32), (1, 1, 2, 2))

    G = _orbital_gradient(Ft, eps, gap, p, offdiag, p_floor=0.1, T=0.02, denom_scale=1e-3)

    G_arr = np.array(G)
    assert np.max(np.abs(G_arr)) > 1e-3
    np.testing.assert_allclose(G_arr, -np.swapaxes(np.conj(G_arr), -1, -2), atol=1e-7)


def test_orbital_gradient_preconditioner_is_per_k():
    Ft = jnp.array(
        [
            [[[0.0, 0.2], [0.2, 0.0]]],
            [[[100.0, 0.2], [0.2, 100.0]]],
        ],
        dtype=jnp.complex64,
    )
    eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)).astype(jnp.float32)
    gap = eps[..., :, None] - eps[..., None, :]
    p = jnp.array([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=jnp.float32)
    offdiag = jnp.reshape(1.0 - jnp.eye(2, dtype=jnp.float32), (1, 1, 2, 2))

    G = _orbital_gradient(Ft, eps, gap, p, offdiag, p_floor=0.1, T=0.02, denom_scale=1e-3)

    assert float(jnp.abs(G[1, 0, 0, 1])) < float(jnp.abs(G[0, 0, 0, 1]))


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


def test_variational_qr_block_sizes():
    """block_sizes=(2,2) must keep P block-diagonal if started so."""
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
        block_sizes=(2, 2),
    )
    P_arr = np.array(P_fin[0, 0])
    offblock = np.concatenate([P_arr[:2, 2:].ravel(), P_arr[2:, :2].ravel()])
    assert np.max(np.abs(offblock)) < 1e-10, (
        f"block_sizes should prevent inter-block mixing; "
        f"max off-block |P| = {np.max(np.abs(offblock)):.2e}"
    )


def test_adaptive_tau_is_per_k():
    eps = jnp.array(
        [
            [[0.0, 1.0], [0.0, 4.0]],
            [[0.0, 2.0], [0.0, 8.0]],
        ],
        dtype=jnp.float32,
    )
    gap = eps[..., :, None] - eps[..., None, :]

    base = jnp.array([[0.0, 1.0], [-1.0, 0.0]], dtype=jnp.complex64)
    scales = jnp.array([[1.0, 0.5], [2.0, 1.5]], dtype=jnp.float32)
    G = scales[..., None, None] * base

    tau = _adaptive_tau(G, eps, gap, denom_scale=1e-3, T=0.05)

    assert tau.shape == eps.shape[:-1]
    assert np.isfinite(np.array(tau)).all()
    assert len(np.unique(np.round(np.array(tau).ravel(), 6))) > 1


def test_qr_exchange_block_specs_follow_block_sizes():
    specs = _exchange_block_specs_from_block_sizes((2, 2))
    assert specs == (("sizes", (2, 2)),)


def test_variational_qr_rejects_legacy_duplicate_block_kwargs():
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

    with pytest.raises(TypeError, match="single shared `block_sizes` kwarg"):
        runner(P0, electrondensity0=1.0, rotation_block_sizes=(1, 1))

    with pytest.raises(TypeError, match="single shared `block_sizes` kwarg"):
        runner(P0, electrondensity0=1.0, exchange_block_specs=[{"block_sizes": [1, 1]}])


def test_blocked_qr_retraction_matches_dense_qr_for_block_diagonal_generator():
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (1, 1, 4, 4), dtype=jnp.float32)
    b = jax.random.normal(jax.random.fold_in(key, 1), (1, 1, 4, 4), dtype=jnp.float32)
    raw = (a + 1j * b).astype(jnp.complex64)

    G = jnp.zeros_like(raw)
    G = G.at[..., :2, :2].set(raw[..., :2, :2])
    G = G.at[..., 2:, 2:].set(raw[..., 2:, 2:])
    G = 0.5 * (G - jnp.conj(jnp.swapaxes(G, -1, -2)))

    tau = jnp.array([[0.2]], dtype=jnp.float32)
    block_slices = _block_slices_from_sizes((2, 2), 4)

    U_dense = _qr_retraction_unitary(G, tau)
    U_block = _qr_retraction_unitary(G, tau, block_slices=block_slices)

    np.testing.assert_allclose(np.array(U_block), np.array(U_dense), rtol=1e-6, atol=1e-6)


def test_blocked_orbital_basis_update_matches_dense_similarity():
    key = jax.random.PRNGKey(1)
    a = jax.random.normal(key, (1, 1, 4, 4), dtype=jnp.float32)
    b = jax.random.normal(jax.random.fold_in(key, 1), (1, 1, 4, 4), dtype=jnp.float32)
    Ft = (a + 1j * b).astype(jnp.complex64)
    Ft = 0.5 * (Ft + jnp.conj(jnp.swapaxes(Ft, -1, -2)))

    g_a = jax.random.normal(jax.random.fold_in(key, 2), (1, 1, 4, 4), dtype=jnp.float32)
    g_b = jax.random.normal(jax.random.fold_in(key, 3), (1, 1, 4, 4), dtype=jnp.float32)
    raw = (g_a + 1j * g_b).astype(jnp.complex64)
    G = jnp.zeros_like(raw)
    G = G.at[..., :2, :2].set(raw[..., :2, :2])
    G = G.at[..., 2:, 2:].set(raw[..., 2:, 2:])
    G = 0.5 * (G - jnp.conj(jnp.swapaxes(G, -1, -2)))

    tau = jnp.array([[0.15]], dtype=jnp.float32)
    block_slices = _block_slices_from_sizes((2, 2), 4)
    U = _qr_retraction_unitary(G, tau, block_slices=block_slices)

    Ft_dense = _apply_orbital_basis_update(Ft, U)
    Ft_block = _apply_orbital_basis_update(Ft, U, block_slices=block_slices)

    np.testing.assert_allclose(np.array(Ft_block), np.array(Ft_dense), rtol=1e-6, atol=1e-6)


def test_blocked_right_unitary_matches_dense_product():
    key = jax.random.PRNGKey(4)
    a = jax.random.normal(key, (1, 1, 6, 4), dtype=jnp.float32)
    b = jax.random.normal(jax.random.fold_in(key, 1), (1, 1, 6, 4), dtype=jnp.float32)
    X = (a + 1j * b).astype(jnp.complex64)

    g_a = jax.random.normal(jax.random.fold_in(key, 2), (1, 1, 4, 4), dtype=jnp.float32)
    g_b = jax.random.normal(jax.random.fold_in(key, 3), (1, 1, 4, 4), dtype=jnp.float32)
    raw = (g_a + 1j * g_b).astype(jnp.complex64)
    G = jnp.zeros_like(raw)
    G = G.at[..., :2, :2].set(raw[..., :2, :2])
    G = G.at[..., 2:, 2:].set(raw[..., 2:, 2:])
    G = 0.5 * (G - jnp.conj(jnp.swapaxes(G, -1, -2)))

    tau = jnp.array([[0.1]], dtype=jnp.float32)
    block_slices = _block_slices_from_sizes((2, 2), 4)
    U = _qr_retraction_unitary(G, tau, block_slices=block_slices)

    X_dense = _apply_right_unitary(X, U)
    X_block = _apply_right_unitary(X, U, block_slices=block_slices)

    np.testing.assert_allclose(np.array(X_block), np.array(X_dense), rtol=1e-6, atol=1e-6)


# -------------------------------------------------------
# L-BFGS unit tests
# -------------------------------------------------------


def test_lbfgs_direction_returns_gradient_when_no_history():
    """With count=0, L-BFGS should return a scaled copy of the gradient."""
    G = jnp.array([[[[0.0, 0.3], [-0.3, 0.0]]]], dtype=jnp.complex64)
    m = 3
    S = jnp.zeros((m,) + G.shape, dtype=G.dtype)
    Y = jnp.zeros((m,) + G.shape, dtype=G.dtype)
    count = jnp.int32(0)

    H = _lbfgs_direction(G, S, Y, count, m)

    # gamma=1 when count=0, so H should equal G
    np.testing.assert_allclose(np.array(H), np.array(G), atol=1e-6)


def test_lbfgs_direction_uses_curvature_from_history():
    """With valid history, L-BFGS direction should differ from steepest descent."""
    key = jax.random.PRNGKey(42)
    nk1, nk2, nb = 2, 2, 3
    m = 3

    def make_skew(k):
        a = jax.random.normal(k, (nk1, nk2, nb, nb), dtype=jnp.float32)
        b = jax.random.normal(jax.random.fold_in(k, 1), (nk1, nk2, nb, nb), dtype=jnp.float32)
        raw = (a + 1j * b).astype(jnp.complex64)
        return 0.5 * (raw - jnp.conj(jnp.swapaxes(raw, -1, -2)))

    S = jnp.stack([make_skew(jax.random.fold_in(key, i)) * 0.1 for i in range(m)])
    Y = jnp.stack([make_skew(jax.random.fold_in(key, i + 10)) for i in range(m)])
    G = make_skew(jax.random.fold_in(key, 99))
    count = jnp.int32(m)

    H = _lbfgs_direction(G, S, Y, count, m)

    # Direction should be anti-Hermitian
    np.testing.assert_allclose(
        np.array(H), -np.conj(np.swapaxes(np.array(H), -1, -2)), atol=1e-5,
    )
    # Should differ from raw gradient (curvature correction applied)
    assert not np.allclose(np.array(H), np.array(G), atol=1e-3)


def test_frobenius_ip_matches_trace():
    a = jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=jnp.complex64)
    b = jnp.array([[[[5.0, 6.0], [7.0, 8.0]]]], dtype=jnp.complex64)
    ip = _frobenius_ip(a, b)
    expected = jnp.real(jnp.trace(a.conj().swapaxes(-1, -2) @ b, axis1=-2, axis2=-1))
    np.testing.assert_allclose(np.array(ip), np.array(expected), atol=1e-6)


# -------------------------------------------------------
# L-BFGS integration tests (full solver)
# -------------------------------------------------------


def test_variational_qr_lbfgs_smoke():
    """L-BFGS optimizer converges and conserves density."""
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

    P_fin, F_fin, E_fin, mu_fin, k_fin, history = runner(
        P0,
        electrondensity0=1.0,
        max_iter=80,
        comm_tol=1e-6,
        p_tol=1e-6,
        optimizer="lbfgs",
        lbfgs_m=3,
        inner_sweeps=3,
        q_sweeps=2,
    )

    assert int(k_fin) <= 80
    assert np.isfinite(np.array(E_fin)).all()
    n_fin = _electron_count(P_fin, weights)
    assert n_fin == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_variational_qr_lbfgs_matches_cg_energy():
    """L-BFGS and CG should converge to the same energy on a random system."""
    key_h, key_i = jax.random.split(jax.random.PRNGKey(456))
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

    P0 = jnp.eye(nb, dtype=jnp.complex64)[None, None, ...]
    P0 = jnp.broadcast_to(P0, (nk1, nk2, nb, nb)) * 0.5

    shared = dict(
        electrondensity0=1.7,
        max_iter=200,
        comm_tol=1e-5,
        p_tol=1e-5,
        inner_sweeps=4,
        q_sweeps=3,
    )

    runner_cg = jax_hf.jit_variational_qr_iteration(kernel)
    runner_lbfgs = jax_hf.jit_variational_qr_iteration(kernel)

    _, _, E_cg, _, k_cg, _ = runner_cg(P0, optimizer="cg", **shared)
    _, _, E_lbfgs, _, k_lbfgs, _ = runner_lbfgs(P0, optimizer="lbfgs", lbfgs_m=5, **shared)

    # Both should converge
    assert int(k_cg) < 200
    assert int(k_lbfgs) < 200
    # Energies should agree
    np.testing.assert_allclose(float(E_lbfgs), float(E_cg), atol=1e-4)


def test_variational_qr_lbfgs_with_block_sizes():
    """L-BFGS respects block-diagonal structure."""
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
        block_sizes=(2, 2),
        optimizer="lbfgs",
        lbfgs_m=3,
        inner_sweeps=3,
        q_sweeps=2,
    )
    P_arr = np.array(P_fin[0, 0])
    offblock = np.concatenate([P_arr[:2, 2:].ravel(), P_arr[2:, :2].ravel()])
    assert np.max(np.abs(offblock)) < 1e-10
