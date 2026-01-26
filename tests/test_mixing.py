import numpy as np
import pytest

import jax.numpy as jnp

from jax_hf import mixing


def test_normalize_precond_mode_strings():
    assert mixing.normalize_precond_mode("eigh") == mixing.PRECOND_EIGH
    assert mixing.normalize_precond_mode("AUTO") == mixing.PRECOND_AUTO
    assert mixing.normalize_precond_mode(mixing.PRECOND_DIAG) == mixing.PRECOND_DIAG


def test_normalize_precond_mode_invalid():
    with pytest.raises(ValueError):
        mixing.normalize_precond_mode("nope")


def test_reuse_available_shape_checks():
    F = jnp.zeros((2, 2), dtype=jnp.float32)
    eps = jnp.zeros((2,), dtype=jnp.float32)
    C = jnp.zeros((2, 2), dtype=jnp.float32)
    assert mixing._reuse_available(F, eps, C)
    assert not mixing._reuse_available(F, jnp.zeros((2, 2), dtype=jnp.float32), C)
    assert not mixing._reuse_available(F, eps, jnp.zeros((3, 3), dtype=jnp.float32))


def test_orbital_preconditioner_diag_matches_formula():
    F = jnp.diag(jnp.array([1.0, 3.0], dtype=jnp.float32))
    comm = jnp.array([[0.0, 2.0], [-2.0, 0.0]], dtype=jnp.float32)
    delta = 0.5
    eps = jnp.diagonal(F)
    denom = (eps[:, None] - eps[None, :]) + delta
    expected = -comm / denom
    actual = mixing.orbital_preconditioner(comm, F, delta=delta, mode=mixing.PRECOND_DIAG)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-6, atol=1e-6)


def test_orbital_preconditioner_reuse_matches_eigh():
    F = jnp.array([[2.0, 0.3], [0.3, 1.0]], dtype=jnp.float32)
    comm = jnp.array([[0.0, 1.2], [-1.2, 0.0]], dtype=jnp.float32)
    eps, C = jnp.linalg.eigh(F)
    out_eigh = mixing.orbital_preconditioner(comm, F, delta=0.1, mode=mixing.PRECOND_EIGH)
    out_reuse = mixing.orbital_preconditioner(
        comm,
        F,
        delta=0.1,
        mode=mixing.PRECOND_REUSE,
        eps=eps,
        C=C,
    )
    np.testing.assert_allclose(np.array(out_reuse), np.array(out_eigh), rtol=1e-6, atol=1e-6)


def test_diis_update_fallback_when_insufficient_history():
    state = mixing.diis_init(3, (2, 2), dtype=jnp.float32)
    P_new = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    P_current = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.float32)
    resid = jnp.arange(4, dtype=jnp.float32)
    state_out, P_out = mixing.diis_update(
        state,
        P_new,
        resid,
        P_current=P_current,
        blend_keep=0.7,
        blend_new=0.3,
    )
    expected = 0.7 * P_current + 0.3 * P_new
    np.testing.assert_allclose(np.array(P_out), np.array(expected), rtol=1e-6, atol=1e-6)
    assert int(state_out.n_entries) == 1


def test_diis_update_stores_entries_in_order():
    state = mixing.diis_init(3, (2, 2), dtype=jnp.float32)
    P_new1 = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    resid1 = jnp.arange(4, dtype=jnp.float32)
    state1, _ = mixing.diis_update(state, P_new1, resid1, P_current=P_new1)

    P_new2 = jnp.array([[2.0, 0.0], [0.0, 2.0]], dtype=jnp.float32)
    resid2 = jnp.arange(4, dtype=jnp.float32) + 10.0
    state2, _ = mixing.diis_update(state1, P_new2, resid2, P_current=P_new2)

    np.testing.assert_allclose(np.array(state2.residuals[0]), np.array(resid1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(state2.residuals[1]), np.array(resid2), rtol=1e-6, atol=1e-6)


def test_ediis_update_returns_input_for_single_entry():
    state = mixing.ediis_init(3, (2, 2), dtype=jnp.complex64)
    P = jnp.eye(2, dtype=jnp.complex64)
    F = jnp.eye(2, dtype=jnp.complex64)
    E = jnp.array(1.0, dtype=jnp.float32)
    sqrt_weights = jnp.array(1.0, dtype=jnp.float32)
    state_out, P_out = mixing.ediis_update(state, P, F, E, sqrt_weights)
    np.testing.assert_allclose(np.array(P_out), np.array(P), rtol=1e-6, atol=1e-6)
    assert int(state_out.n_entries) == 1


def test_broyden_update_first_call_no_change():
    state = mixing.broyden_init(2, (2, 2), dtype=jnp.float32)
    P = jnp.array([[1.0, 0.2], [0.2, 1.5]], dtype=jnp.float32)
    resid = jnp.ones(P.size, dtype=jnp.float32)
    state_out, P_out = mixing.broyden_update(state, P, resid, α=1.0)
    np.testing.assert_allclose(np.array(P_out), np.array(P), rtol=1e-6, atol=1e-6)
    assert int(state_out.count) == 1


@pytest.mark.parametrize(
    "comm_rms, expected_phase",
    [
        (10.0, mixing.PHASE_EDIIS),
        (0.5, mixing.PHASE_CDIIS),
        (0.01, mixing.PHASE_BROYDEN),
    ],
)
def test_mixer_update_phase_selection(comm_rms, expected_phase):
    state = mixing.mixer_init(3, (2, 2), dtype=jnp.complex64)
    P_new = jnp.eye(2, dtype=jnp.complex64)
    P_cur = 0.9 * jnp.eye(2, dtype=jnp.complex64)
    F = jnp.array([[1.0, 0.2], [0.2, 1.5]], dtype=jnp.complex64)
    E = jnp.array(0.0, dtype=jnp.float32)
    comm = jnp.array([[0.0, 0.1], [-0.1, 0.0]], dtype=jnp.complex64)
    sqrt_weights = jnp.array(1.0, dtype=jnp.float32)
    state_out, _ = mixing.mixer_update(
        state,
        P_new,
        P_cur,
        F,
        E,
        comm,
        comm_rms=jnp.array(comm_rms, dtype=jnp.float32),
        sqrt_weights=sqrt_weights,
        to_cdiis=1.0,
        to_broyden=0.1,
        precond_mode=mixing.PRECOND_DIAG,
    )
    assert int(state_out.phase) == int(expected_phase)
