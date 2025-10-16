import jax
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn, fftshift
from jax.scipy.special import expit
import optimistix

###################################################################
# Low-level physics functions using JAX (file: jax_hf/utils.py)
###################################################################

def fermidirac(x: jax.Array, T: float) -> jax.Array:
    """Finite-T Fermi–Dirac occupation: 1/(1 + exp(x/T))."""
    return expit(-x / (T + 1e-12))

def electron_density(P: jax.Array) -> jax.Array:
    """Real trace of density matrix."""
    return jnp.real(jnp.trace(P, axis1=-2, axis2=-1))

def density_spectrum(bands: jax.Array, mu: float, T: float) -> jax.Array:
    """Sum of Fermi occupations for given band energies."""
    return fermidirac(bands - mu, T).sum(axis=-1)

def selfenergy_fft(VR: jax.Array, P: jax.Array) -> jax.Array:
    """Hartree self-energy Σ(k) = -FFT⁻¹[FFT(P) * VR], shifted to center."""
    P_fft = fftn(P, axes=(0, 1))
    sigma = -ifftn(P_fft * VR, axes=(0, 1))
    return fftshift(sigma, axes=(0, 1))

def find_chemical_potential(
    bands: jax.Array,
    weights: jax.Array,
    n_electrons: float,
    T: float,
    tol: float = 1e-3,
    maxiter: int = 400
 ) -> float:
    """
    Solve μ such that ∑ weights * fermi(bands - μ) = n_electrons via bisection.
    """
    mu_low, mu_high = jnp.min(bands), jnp.max(bands)
    solver = optimistix.Bisection(tol, tol, False)

    @jax.jit
    def resid(mu: float, args=None) -> jax.Array:
        return jnp.real(jnp.sum(weights * density_spectrum(bands, mu, T)) - n_electrons)

    result = optimistix.root_find(
        resid,
        solver,
        mu_high,
        options={"lower": mu_low, "upper": mu_high},
        max_steps=maxiter
    )
    return result.value
