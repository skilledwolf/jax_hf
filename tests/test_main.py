import numpy as np
import pytest

import jax.numpy as jnp

from jax_hf.main import HartreeFockKernel, jit_hartreefock_iteration
from jax_hf.utils import fermidirac, find_chemical_potential, selfenergy_fft


def _comm_rms(R: jnp.ndarray, weights_2d: jnp.ndarray) -> float:
    """Weighted RMS of the commutator."""
    sq = jnp.abs(R) ** 2
    per_k = jnp.sum(sq, axis=(-2, -1))
    weight_sum = jnp.sum(weights_2d)
    rms = jnp.sqrt(jnp.sum(weights_2d * per_k) / jnp.maximum(weight_sum, jnp.array(1e-30)))
    return float(rms)


def test_find_chemical_potential_hits_target_density():
    # Simple two-level system with a single k-point.
    bands = jnp.array([[[0.0, 1.0]]], dtype=jnp.float32)  # shape (1,1,2)
    weights = jnp.ones((1, 1), dtype=jnp.float32)
    mu = find_chemical_potential(bands, weights, n_electrons=1.0, T=0.1)
    occ = fermidirac(bands - mu, 0.1)
    total = float(jnp.sum(weights[..., None] * occ))
    assert abs(total - 1.0) < 1e-4


def test_selfenergy_fft_single_point_matches_definition():
    # For a single k-point grid, FFT/ifft reduce to identity, so Σ = -VR * P.
    VR = jnp.array([[[[0.3]]]], dtype=jnp.complex64)  # shape (1,1,1,1)
    P = jnp.array([[[[1.0 + 0.5j, 0.2], [0.2, 0.8 - 0.1j]]]], dtype=jnp.complex64)
    sigma = selfenergy_fft(VR, P)
    expected = -VR * P
    np.testing.assert_allclose(np.array(sigma), np.array(expected), rtol=1e-6, atol=1e-6)


def test_hartreefock_iteration_converges_on_tiny_model():
    # Inspired by the continuum-model examples: include exchange feedback on a 1x1 grid.
    weights = jnp.ones((1, 1), dtype=jnp.float32)
    hamiltonian = jnp.array([[[[-0.5, 0.0], [0.0, 0.5]]]], dtype=jnp.complex64)  # (1,1,2,2)
    coulomb_q = jnp.array([[[[0.25]]]], dtype=jnp.complex64)  # weak exchange strength

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.2,
        include_hartree=False,
        include_exchange=True,
    )

    P0 = jnp.array([[[[0.6, 0.0], [0.0, 0.4]]]], dtype=jnp.complex64)
    runner = jit_hartreefock_iteration(kernel)

    P_fin, F_fin, E_fin, mu_fin, k_fin, history = runner(
        P0,
        electrondensity0=1.0,
        max_iter=60,
        comm_tol=1e-6,
        diis_size=4,
        precond_mode="diag",
    )

    # Converged within budget and produced finite values.
    assert int(k_fin) <= 60
    assert np.isfinite(np.array(E_fin)).all()
    assert np.isfinite(np.array(mu_fin)).all()

    # Final density is Hermitian.
    np.testing.assert_allclose(
        np.array(P_fin),
        np.array(jnp.conj(jnp.swapaxes(P_fin, -1, -2))),
        atol=1e-7,
    )

    # Self-consistency: commutator RMS well below tolerance used in the loop.
    comm = F_fin @ P_fin - P_fin @ F_fin
    rms = _comm_rms(comm, weights)
    assert rms < 1e-6

    # History stored the last RMS value in the first k_fin slots.
    last_recorded = float(history["dC"][int(k_fin) - 1])
    assert last_recorded == pytest.approx(rms, rel=1e-3, abs=1e-8)
