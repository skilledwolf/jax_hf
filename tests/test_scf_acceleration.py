"""Tests for SCF acceleration schemes (linear / DIIS / ODA) and trust_radius.

DIIS (Pulay commutator extrapolation) and ODA (optimal damping) mirror
cpp_hf's solver_scf.hpp.  DIIS converges in far fewer iterations than linear
mixing; ODA is a robust energy-monotone line search (it reaches the same
energy, but as an energy method it does not drive the commutator as hard, so
its convergence is asserted on the energy).  These run in the suite's default
float32.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from jax_hf import HartreeFockKernel, SCFConfig, solve_scf


def _comm_rms(F, P, w2d):
    comm = F @ P - P @ F
    per_k = jnp.sum(jnp.abs(comm) ** 2, axis=(-2, -1))
    return float(jnp.sqrt(jnp.sum(w2d * per_k) / jnp.maximum(jnp.sum(w2d), 1e-30)))


def _dirac(nk=4, m=0.3, v=1.0, T=0.1, ex=0.7):
    ks = (np.arange(nk) - nk // 2) * (2 * np.pi / nk)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    h = np.zeros((nk, nk, 2, 2), np.complex64)
    h[..., 0, 0] = m
    h[..., 1, 1] = -m
    h[..., 0, 1] = v * (KX - 1j * KY)
    h[..., 1, 0] = v * (KX + 1j * KY)
    return HartreeFockKernel(
        weights=jnp.ones((nk, nk), jnp.float32), hamiltonian=jnp.asarray(h),
        coulomb_q=jnp.full((nk, nk, 1, 1), ex, jnp.complex64), T=T,
    )


_NE = 16.0


class TestAccelerationAgreement:
    def test_schemes_agree_on_energy(self):
        K = _dirac()
        P0 = jnp.zeros_like(K.h)
        energies = {}
        for acc in ("linear", "diis", "oda"):
            r = solve_scf(K, P0, _NE, config=SCFConfig(
                max_iter=800, density_tol=1e-6, comm_tol=1e-5,
                mixing=0.5, acceleration=acc))
            energies[acc] = float(r.energy)
        spread = max(energies.values()) - min(energies.values())
        assert spread < 1e-3, f"schemes disagree on energy: {energies}"


class TestDIIS:
    def test_converges_far_faster_than_linear(self):
        K = _dirac()
        P0 = jnp.zeros_like(K.h)
        cfg = dict(max_iter=800, density_tol=1e-6, comm_tol=1e-5, mixing=0.5)
        lin = solve_scf(K, P0, _NE, config=SCFConfig(acceleration="linear", **cfg))
        diis = solve_scf(K, P0, _NE, config=SCFConfig(acceleration="diis", **cfg))
        assert bool(lin.converged) and bool(diis.converged)
        # Pulay DIIS should need far fewer Roothaan steps than plain mixing.
        assert int(diis.iterations) < int(lin.iterations) // 3
        np.testing.assert_allclose(float(diis.energy), float(lin.energy), atol=1e-4, rtol=1e-5)
        assert _comm_rms(diis.fock_matrix, diis.density_matrix, K.w2d) < 1e-4

    def test_trust_radius_converges(self):
        K = _dirac()
        P0 = jnp.zeros_like(K.h)
        free = solve_scf(K, P0, _NE, config=SCFConfig(
            max_iter=800, density_tol=1e-6, comm_tol=1e-5, acceleration="diis"))
        clipped = solve_scf(K, P0, _NE, config=SCFConfig(
            max_iter=800, density_tol=1e-6, comm_tol=1e-5, acceleration="diis",
            trust_radius=0.5))
        assert bool(clipped.converged)
        np.testing.assert_allclose(float(clipped.energy), float(free.energy), atol=1e-4, rtol=1e-5)

    def test_damping_still_converges(self):
        K = _dirac()
        P0 = jnp.zeros_like(K.h)
        r = solve_scf(K, P0, _NE, config=SCFConfig(
            max_iter=800, density_tol=1e-6, comm_tol=1e-5,
            acceleration="diis", diis_damping=0.7))
        assert bool(r.converged)


class TestODA:
    def test_energy_matches_and_is_monotone(self):
        K = _dirac()
        P0 = jnp.zeros_like(K.h)
        cfg = dict(max_iter=800, density_tol=1e-6, comm_tol=1e-5, mixing=0.5)
        lin = solve_scf(K, P0, _NE, config=SCFConfig(acceleration="linear", **cfg))
        oda = solve_scf(K, P0, _NE, config=SCFConfig(acceleration="oda", **cfg))
        # ODA reaches the same energy (its guarantee is energy descent)...
        np.testing.assert_allclose(float(oda.energy), float(lin.energy), atol=1e-3, rtol=1e-5)
        # ...and the energy history is non-increasing (robust line search).
        hE = np.asarray(oda.history["E"])
        assert hE.size > 1
        assert np.all(np.diff(hE) <= 1e-4), "ODA energy not monotonically decreasing"


class TestSCFConfigValidation:
    def test_invalid_acceleration_raises(self):
        with pytest.raises(ValueError, match="acceleration"):
            SCFConfig(acceleration="bfgs")

    def test_nonpositive_diis_size_raises(self):
        with pytest.raises(ValueError, match="diis_size"):
            SCFConfig(diis_size=0)
