import numpy as np
import pytest

import jax.numpy as jnp

from jax_hf.main import HartreeFockKernel, jit_hartreefock_iteration
from jax_hf.utils import density_matrix_from_fock, fermidirac, find_chemical_potential, selfenergy_fft


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


def test_find_chemical_potential_rejects_unreachable_density():
    bands = jnp.array([[[0.0]]], dtype=jnp.float32)
    weights = jnp.ones((1, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="physically reachable range"):
        find_chemical_potential(bands, weights, n_electrons=2.0, T=0.1)


def test_density_matrix_from_fock_rejects_unreachable_density():
    F = jnp.array([[[[0.0]]]], dtype=jnp.complex64)
    weights = jnp.ones((1, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="physically reachable range"):
        density_matrix_from_fock(F, weights, n_electrons=2.0, T=0.1)


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

    # History dC = max(commutator_rms, density_change_rms); at convergence both
    # components are below the comm_tol used in the loop.
    last_recorded = float(history["dC"][int(k_fin) - 1])
    assert last_recorded < 1e-6


def test_hartreefock_iteration_rejects_nonpositive_diis_size():
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

    runner = jit_hartreefock_iteration(kernel)
    P0 = jnp.array([[[[0.6, 0.0], [0.0, 0.4]]]], dtype=jnp.complex64)

    with pytest.raises(ValueError, match="diis_size must be a positive integer"):
        runner(
            P0,
            electrondensity0=1.0,
            max_iter=60,
            comm_tol=1e-6,
            diis_size=0,
            precond_mode="diag",
        )


def test_hartreefock_kernel_auto_enables_hermitian_channel_packing_for_real_scalar_coulomb():
    weights = jnp.ones((2, 2), dtype=jnp.float32)
    hamiltonian = jnp.zeros((2, 2, 3, 3), dtype=jnp.complex64)
    coulomb_q = jnp.ones((2, 2, 1, 1), dtype=jnp.float32)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.1,
        include_hartree=False,
        include_exchange=True,
    )

    assert kernel.exchange_hermitian_channel_packing is True


@pytest.mark.parametrize("method", ["bisection", "newton"])
def test_find_chemical_potential_hits_target_multiband(method):
    """Both solvers hit target density on a multi-k, multi-band system."""
    rng = np.random.RandomState(42)
    nk1, nk2, nb = 4, 4, 6
    bands = jnp.array(rng.randn(nk1, nk2, nb).astype(np.float32))
    weights = jnp.ones((nk1, nk2), dtype=jnp.float32) / (nk1 * nk2)
    n_target = 3.0
    T = 0.05

    mu = find_chemical_potential(bands, weights, n_target, T, method=method)
    occ = fermidirac(bands - mu, T)
    total = float(jnp.sum(weights[..., None] * occ))
    assert abs(total - n_target) < 1e-4


@pytest.mark.parametrize("method", ["bisection", "newton"])
def test_find_chemical_potential_cold_limit(method):
    """At very low T, both solvers must still find the correct mu."""
    bands = jnp.array([[[0.0, 1.0, 2.0]]], dtype=jnp.float64)
    weights = jnp.ones((1, 1), dtype=jnp.float64)

    mu = find_chemical_potential(bands, weights, n_electrons=2.0, T=1e-8, method=method)
    occ = fermidirac(bands - mu, 1e-8)
    total = float(jnp.sum(weights[..., None] * occ))
    assert abs(total - 2.0) < 1e-6


def test_hartreefock_iteration_converges_with_newton_mu():
    """SCF converges with mu_method='newton' on the tiny model."""
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
    runner = jit_hartreefock_iteration(kernel)

    P_fin, F_fin, E_fin, mu_fin, k_fin, history = runner(
        P0,
        electrondensity0=1.0,
        max_iter=60,
        comm_tol=1e-6,
        diis_size=4,
        precond_mode="diag",
        mu_method="newton",
    )

    assert int(k_fin) <= 60
    assert np.isfinite(np.array(E_fin)).all()

    comm = F_fin @ P_fin - P_fin @ F_fin
    rms = _comm_rms(comm, weights)
    assert rms < 1e-6


def test_hartreefock_iteration_converges_with_level_shift():
    """Level shift doesn't break convergence and the final result is self-consistent."""
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
    runner = jit_hartreefock_iteration(kernel)

    P_fin, F_fin, E_fin, mu_fin, k_fin, history = runner(
        P0,
        electrondensity0=1.0,
        max_iter=60,
        comm_tol=1e-6,
        diis_size=4,
        precond_mode="diag",
        level_shift=0.3,
    )

    assert int(k_fin) <= 60
    assert np.isfinite(np.array(E_fin)).all()

    # Final result should still be self-consistent (level shift only aids convergence,
    # not applied at finalization).
    comm = F_fin @ P_fin - P_fin @ F_fin
    rms = _comm_rms(comm, weights)
    assert rms < 1e-5


def test_hartreefock_iteration_project_fn_enforces_symmetry():
    weights = jnp.ones((1, 1), dtype=jnp.float32)
    hamiltonian = jnp.zeros((1, 1, 2, 2), dtype=jnp.complex64)
    coulomb_q = jnp.array([[[[0.6]]]], dtype=jnp.complex64)

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=0.05,
        include_hartree=False,
        include_exchange=True,
    )

    P0 = jnp.array([[[[0.9, 0.0], [0.0, 0.1]]]], dtype=jnp.complex64)
    runner = jit_hartreefock_iteration(kernel)

    P_unproj, _, _, _, _, _ = runner(
        P0,
        electrondensity0=1.0,
        max_iter=60,
        comm_tol=1e-7,
        diis_size=4,
        precond_mode="diag",
    )

    def project_fn(A):
        swap = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=A.dtype)
        return 0.5 * (A + swap @ A @ jnp.conj(swap.T))

    P_fin, F_fin, *_ = runner(
        P0,
        electrondensity0=1.0,
        max_iter=60,
        comm_tol=1e-7,
        diis_size=4,
        precond_mode="diag",
        project_fn=project_fn,
    )

    assert float(abs(P_unproj[0, 0, 0, 0] - P_unproj[0, 0, 1, 1])) > 1e-3
    np.testing.assert_allclose(np.array(P_fin), np.array(project_fn(P_fin)), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.array(F_fin), np.array(project_fn(F_fin)), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        np.array(P_fin[0, 0, 0, 0]),
        np.array(P_fin[0, 0, 1, 1]),
        atol=1e-6,
        rtol=1e-6,
    )
