import jax 
import jax.numpy as jnp

from .jax_modules import HartreeFockStep

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

###################################################################
# Flax.nnx modules (file: jax_hf/flax_modules.py)
###################################################################

class HartreeFockStep:
    """Build and diagonalize Fock operator → return (P_new, E_HF)."""
    def __init__(
        self,
        weights: jax.Array,
        hamiltonian: jax.Array,
        coulomb_q: jax.Array,
        n_electrons: float,
        T: float
    ):
        self.weights = weights
        self.h = hamiltonian
        self.VR = fftn(weights * coulomb_q, axes=(0, 1))[..., None, None]
        self.n_electrons = n_electrons
        self.T = T

    def _chemical_potential(self, P):
        Ph = 0.5 * (P + jnp.conj(jnp.swapaxes(P, -1, -2)))
        Σ       = selfenergy_fft(self.VR, Ph)
        Hf      = self.h + Σ
        bands, ψ = eigh(Hf)                            # ψ[..., i, n]
        return find_chemical_potential(bands, self.weights, self.n_electrons, self.T)

    def __call__(self, P: jax.Array) -> Tuple[jax.Array, float]:
        Ph = 0.5 * (P + jnp.conj(jnp.swapaxes(P, -1, -2)))
        sigma = selfenergy_fft(self.VR, Ph)
        fock = self.h + sigma
        bands, psi = eigh(fock)

        mu = find_chemical_potential(bands, self.weights, self.n_electrons, self.T)
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
    


def build_hf_run(
    hf_step: HartreeFockStep,
    *,
    max_iter:  int   = 100,
    comm_tol:  float = 5e-3,
    diis_size: int   = 4,
    log_every: int | None = 1,     # None → no prints
):
    """
    Returns a **jit-compiled** function
    
        P_final, E_final, n_iter, history = hf_run(P0)
    
    *history* is a dict with two 1-D DeviceArrays of length `max_iter`
    (unused tail entries are zero).  You can slice with `[:n_iter]`
    after the call to keep only the converged part.
    """

    # @partial(jax.jit, static_argnames=("log_every",))
    @jax.jit
    def hf_run(P0: jax.Array):
        mixer_st  = mixer_init(diis_size, P0.shape)

        # history buffers (fixed length → static shape)
        hist_E  = jnp.zeros(max_iter, dtype=jnp.float64)
        hist_dC = jnp.zeros(max_iter, dtype=jnp.float64)

        # carry = (iter, P, mixer_state, dC_prev, E_prev, hist_E, hist_dC)
        carry0 = (jnp.int32(0), P0, mixer_st, jnp.inf, jnp.inf,
                  hist_E, hist_dC)

        # ------------------------------------------------------------------
        def body(carry):
            k, P, mix_st, _dC_prev, E_prev, hE, hC = carry

            # --- 1) SCF map ----------------------------------------------------------
            P_new, E_new = hf_step(P)                # complex energy
            Σ_new        = selfenergy_fft(hf_step.VR, P_new)
            F_new        = hf_step.h + Σ_new

            comm  = (hf_step.weights[..., None, None] *
                    (F_new @ P_new - P_new @ F_new)) / hf_step.weights.sum()
            dC    = jnp.max(jnp.abs(comm))

            was_ediis = mix_st.use_ediis

            # --- 2) optional printing -----------------------------------------------
            if log_every is not None:                           # static test
                should_log = (k % log_every) == 0               # dynamic (traced) bool

                def _do_print(args):
                    kk, EE, cc = args
                    # NOTE: ordered=True keeps multiple prints in program order
                    debug.print(
                        "iter {k:3d} :  E = {E:.8f}  |comm| = {c:.2e}",
                        k=kk, E=EE, c=cc, ordered=True
                    )
                    return jnp.int32(0)     # dummy value to satisfy lax.cond

                # lax.cond executes exactly one branch at run-time
                _ = lax.cond(should_log,
                            _do_print,
                            lambda _: jnp.int32(0),            # no-op branch
                            operand=(k, jnp.real(E_new), dC))

            # --- 3) DIIS / mixing ----------------------------------------------------
            mix_st, P_mix = mixer_update(mix_st, P_new, F_new, E_new, comm)

            is_ediis  = mix_st.use_ediis             # jnp.bool_
            switched  = jnp.not_equal(was_ediis, is_ediis)   # jnp.bool_

            # Define two tiny helpers – each emits one fixed string
            def _announce_ediis(_):
                debug.print("→ entering EDIIS mixing", ordered=True)
                return jnp.int32(0)                  # dummy identical return types

            def _announce_cdiis(_):
                debug.print("→ entering CDIIS mixing", ordered=True)
                return jnp.int32(0)

            # If `switched` is true, choose which message by `is_ediis`
            def _do_announce(mode):
                return lax.cond(mode,
                                _announce_ediis,
                                _announce_cdiis,
                                operand=None)

            _ = lax.cond(switched,        # only run when a change happened
                        _do_announce,
                        lambda _: jnp.int32(0),    # no-op branch
                        operand=is_ediis)

            # --- 4) write history ----------------------------------------------------
            hE = hE.at[k].set(jnp.real(E_new))
            hC = hC.at[k].set(dC)

            return (k + 1, P_mix, mix_st, dC, jnp.real(E_new), hE, hC)

        # ------------------------------------------------------------------
        def cond(carry):
            k, _, _, dC, _, _, _ = carry
            return jnp.logical_and(k < max_iter, dC > comm_tol)

        k_fin, P_fin, _, _, _, hist_E, hist_dC = lax.while_loop(cond, body, carry0)
        E_fin = jnp.real(hf_step(P_fin)[1])

        history = dict(E=hist_E, dC=hist_dC)
        return P_fin, E_fin, k_fin, history

    return hf_run


# -------------------------------------------------------------------------
# 10) Usage example
# -------------------------------------------------------------------------
# hf_step = HartreeFock_from_contimod(h, Vq)\# build and jit your HF step
# hf_run = build_hf_run(hf_step, max_iter=100, comm_tol=5e-3, diis_size=4)\# run SCF
# P_conv, E_conv, n_iter = hf_run(P_init)
# print(f"Converged in {int(n_iter)} steps: E = {E_conv:.8f}")