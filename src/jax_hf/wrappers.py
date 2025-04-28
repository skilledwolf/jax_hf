import jax
import jax.numpy as jnp

from .jax_modules import HartreeFockStep
from .utils import electron_density, selfenergy_fft


def HartreeFock_from_contimod(h, Vq: jax.Array) -> HartreeFockStep:
    weights = jnp.array(h.kMesh.weight, dtype=jnp.float32)
    hamil = jnp.array(h.hs, dtype=jnp.complex64)
    n_elec_matrix = jnp.array(h.densitymatrix, dtype=jnp.complex64)
    n_elec = jnp.real(weights * electron_density(n_elec_matrix))
    return HartreeFockStep(weights, hamil, Vq, n_elec, h.T)

def contimod_from_HartreeFock(h, model, P):
    h_mf = h.copy()  # Assuming 'h' has a copy method to clone its current state
    Sigma = selfenergy_fft(model.VR, P)
    h_mf.hs += Sigma
    h_mf._chemicalpotential = model._chemical_potential(P)  # Assuming chemical potential is a direct attribute

    return h_mf