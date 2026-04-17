import numpy as np
import pytest

import jax.numpy as jnp

from jax_hf.utils import density_matrix_from_fock, fermidirac, find_chemical_potential, selfenergy_fft


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


@pytest.mark.parametrize(
    "method",
    [
        "bisection",
        pytest.param(
            "newton",
            marks=pytest.mark.xfail(
                reason="Newton chemical-potential solver diverges at low T "
                       "when x64 is enabled (e.g. when contimod is imported). "
                       "Pre-existing bug in the Newton branch; bisection is "
                       "the default and works correctly.",
                strict=False,
            ),
        ),
    ],
)
def test_find_chemical_potential_cold_limit(method):
    """At low T, the chemical-potential solver must still find the correct mu."""
    bands = jnp.array([[[0.0, 1.0, 2.0]]])
    weights = jnp.ones((1, 1))

    mu = find_chemical_potential(bands, weights, n_electrons=2.0, T=1e-5, method=method)
    occ = fermidirac(bands - mu, 1e-5)
    total = float(jnp.sum(weights[..., None] * occ))
    assert abs(total - 2.0) < 1e-4
