import jax 
import jax.numpy as jnp

# from .jax_modules import HartreeFockStep

from .utils import selfenergy_fft
from .mixing import mixer_init, mixer_update

from functools import partial
from jax import lax, debug  # ← new import

from functools import partial
from jax import lax, debug
import jax.numpy as jnp

from .utils import find_chemical_potential, fermidirac, selfenergy_fft

import jax
import jax.numpy as jnp
from jax.numpy.linalg import eigh
from jax.numpy.fft import fftn

import math
from typing import NamedTuple, Tuple


class HartreeFockKernel:
    """Build and diagonalize Fock operator → return (P_new, E_HF)."""
    def __init__(
        self,
        weights: jax.Array,
        hamiltonian: jax.Array,
        coulomb_q: jax.Array,
        # electrondensity: float,
        T: float
    ):
        self.weights = weights
        self.h = hamiltonian
        self.VR = fftn(weights * coulomb_q, axes=(0, 1))[..., None, None]
        # self.n_electrons = electrondensity
        self.T = T

    def _chemical_potential(self, P, electrondensity):
        Ph = 0.5 * (P + jnp.conj(jnp.swapaxes(P, -1, -2)))
        Σ       = selfenergy_fft(self.VR, Ph)
        Hf      = self.h + Σ
        bands, ψ = eigh(Hf)                            # ψ[..., i, n]
        return find_chemical_potential(bands, self.weights, electrondensity, self.T)

    def __call__(self, P: jax.Array, electrondensity) -> Tuple[jax.Array, float]:
        Ph = 0.5 * (P + jnp.conj(jnp.swapaxes(P, -1, -2)))
        sigma = selfenergy_fft(self.VR, Ph)
        fock = self.h + sigma
        bands, psi = eigh(fock)

        mu = find_chemical_potential(bands, self.weights, electrondensity, self.T)
        occ = fermidirac(bands - mu, self.T)

        P_new = jnp.einsum('...in,...n,...jn->...ij', psi, occ, jnp.conj(psi))

        Hhalf = self.h + 0.5 * sigma
        E_hf = jnp.real(
            jnp.einsum(
                'klim,klmi',
                self.weights[..., None, None] * Hhalf,
                P_new
            )
        )
        return P_new, E_hf
    

def hartreefock_iteration(
    P0: jax.Array, 
    electrondensity: float,
    hf_step: HartreeFockKernel, 
    *,
    max_iter:  int   = 100,
    comm_tol:  float = 5e-3,
    diis_size: int   = 4,
    log_every: int | None = 1):

    mixer_st  = mixer_init(diis_size, P0.shape)

    # history buffers
    hist_E  = jnp.zeros(max_iter, dtype=jnp.float64)
    hist_dC = jnp.zeros(max_iter, dtype=jnp.float64)

    # carry = (iter, P, mixer_state, dC_prev, E_prev, hist_E, hist_dC)
    carry0 = (jnp.int32(0), P0, mixer_st, jnp.inf, jnp.inf,
                hist_E, hist_dC)

    def body(carry):
        k, P, mix_st, _dC_prev, E_prev, hE, hC = carry

        # 1) SCF step
        P_new, E_new = hf_step(P, electrondensity)
        Σ_new        = selfenergy_fft(hf_step.VR, P_new)
        F_new        = hf_step.h + Σ_new

        comm = (hf_step.weights[..., None, None] *
                (F_new @ P_new - P_new @ F_new)) / hf_step.weights.sum()
        dC = jnp.max(jnp.abs(comm))

        # 2) optional energy / comm prints
        if log_every is not None:
            should_log = (k % log_every) == 0
            _ = lax.cond(
                should_log,
                lambda args: (debug.print(
                    "iter {k:3d} :  E = {E:.8f}  |comm| = {c:.2e}",
                    k=args[0], E=args[1], c=args[2], ordered=True
                ), jnp.int32(0))[1],
                lambda _: jnp.int32(0),
                operand=(k, jnp.real(E_new), dC),
            )

        # 3) DIIS / mixing (+ optional mixing‐mode prints)
        was_ediis = mix_st.use_ediis
        mix_st, P_mix = mixer_update(mix_st, P_new, F_new, E_new, comm)
        is_ediis = mix_st.use_ediis

        if log_every is not None:
            switched = jnp.not_equal(was_ediis, is_ediis)

            def _announce_ediis(_):
                debug.print("→ entering EDIIS mixing", ordered=True)
                return jnp.int32(0)

            def _announce_cdiis(_):
                debug.print("→ entering CDIIS mixing", ordered=True)
                return jnp.int32(0)

            _ = lax.cond(
                switched,
                lambda mode: lax.cond(mode,
                                        _announce_ediis,
                                        _announce_cdiis,
                                        operand=None),
                lambda _: jnp.int32(0),
                operand=is_ediis,
            )

        # 4) write history
        hE = hE.at[k].set(jnp.real(E_new))
        hC = hC.at[k].set(dC)

        return (k + 1, P_mix, mix_st, dC, jnp.real(E_new), hE, hC)

    def cond(carry):
        k, _, _, dC, _, _, _ = carry
        return jnp.logical_and(k < max_iter, dC > comm_tol)

    # run the SCF loop
    k_fin, P_fin, mix_fin, dC_fin, E_prev, hist_E, hist_dC = \
        lax.while_loop(cond, body, carry0)

    # final energy
    E_fin = jnp.real(hf_step(P_fin, electrondensity)[1])

    # 5) convergence warning if we hit max_iter without converging
    # if log_every is not None:
    did_fail = jnp.logical_and(k_fin == max_iter, dC_fin > comm_tol)

    def _warn(_):
        debug.print(
            f"Warning: SCF didn’t converge after {max_iter} iters (|comm|>{comm_tol:.1e})",
            ordered=True
        )
        return jnp.int32(0)

    _ = lax.cond(
        did_fail,
        _warn,
        lambda _: jnp.int32(0),
        operand=None,
    )

    history = dict(E=hist_E, dC=hist_dC)
    return P_fin, E_fin, k_fin, history


def jit_hartreefock_iteration(
    hf_step: HartreeFockKernel):

    # make iteration parameters and density static 
    compiled_hf_iteration_fn = jax.jit(
        partial(
            hartreefock_iteration,
            hf_step=hf_step
        ),
        static_argnames=["max_iter", "comm_tol", "diis_size", "log_every"]
    )

    return compiled_hf_iteration_fn


# -------------------------------------------------------------------------
# 10) Usage example
# -------------------------------------------------------------------------
# hf_kernel = jax_hf.HartreeFockKernel(
#     jnp.array(h.kMesh.weight, dtype=jnp.float32),
#     jnp.array(h.hs, dtype=jnp.complex64),
#     jnp.array(Vq.magnitude, dtype=jnp.complex64),
#     T = h.T
# )
# hartree_fock_iteration = jax_hf.jit_hartreefock_iteration(hf_kernel)
#
# iteration_settings = {
#     "max_iter": 35,
#     "comm_tol": 1e-3,
#     "diis_size": 10,
#     "log_every": None,
# }
#
# hf_run   = jax_hf.build_hf_run(model, max_iter=35, comm_tol=1e-3, diis_size=10, log_every=None)
# P_conv, E_sym, n_iter, history = hartree_fock_iteration(h.zero*(1+0j), n_cn+n_e, **iteration_settings)
# hmf = jax_hf.wrappers.contimod_from_HartreeFock(h, model, P_conv)
# print(f"Converged in {int(n_iter)} steps: E = {E_conv:.8f}")