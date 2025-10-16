from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import debug, lax
from jax.numpy.fft import fftn
from jax.numpy.linalg import eigh

from .mixing import mixer_init, mixer_init_like, mixer_update
from .utils import fermidirac, find_chemical_potential, selfenergy_fft


class HartreeFockResult(NamedTuple):
    density: jax.Array
    energy: jax.Array
    fock_input: jax.Array
    self_energy_input: jax.Array
    fock_density: jax.Array
    self_energy_density: jax.Array


class HartreeFockKernel:
    def __init__(self, weights, hamiltonian, coulomb_q, T: float):
        self.h        = jnp.asarray(hamiltonian)
        self.weights  = jnp.asarray(weights[..., None, None], dtype=hamiltonian.dtype)
        self.weight_sum = jnp.sum(self.weights)
        # keep static trailing dims for cheap multiply/broadcast in k-space
        self.VR       = fftn(self.weights * jnp.asarray(coulomb_q, dtype=hamiltonian.dtype), axes=(0, 1))
        self.T        = T

    def __call__(self, P: jax.Array, electrondensity):
        # symmetrize input density
        Ph     = 0.5 * (P + jnp.conj(jnp.swapaxes(P, -1, -2)))
        sigma  = selfenergy_fft(self.VR, Ph)
        fock   = self.h + sigma
        bands, psi = eigh(fock)

        mu  = find_chemical_potential(bands, self.weights, electrondensity, self.T)
        occ = fermidirac(bands - mu, self.T)
        P_new = jnp.einsum('...in,...n,...jn->...ij', psi, occ, jnp.conj(psi))

        # use the input-side sigma for the HF energy (same as before)
        E_hf = jnp.real(
            jnp.einsum('klim,klmi', (self.weights * (self.h + 0.5 * sigma)), P_new)
        )
        return P_new, E_hf

    def _chemical_potential(self, P, electrondensity):
        Ph = 0.5 * (P + jnp.conj(jnp.swapaxes(P, -1, -2)))
        Σ       = selfenergy_fft(self.VR, Ph)
        Hf      = self.h + Σ
        bands, ψ = eigh(Hf)                            # ψ[..., i, n]
        return find_chemical_potential(bands, self.weights, electrondensity, self.T)
    

def hartreefock_iteration(
    P0: jax.Array, 
    electrondensity0: float,
    hf_step: HartreeFockKernel, 
    *,
    max_iter:  int   = 100,
    comm_tol:  float = 5e-3,
    diis_size: int   = 4,
    log_every: int | None = 1):

    # Single conversion: pin everything to the kernel Hamiltonian dtype
    target_dtype = hf_step.h.dtype
    P0 = jnp.asarray(P0, dtype=target_dtype)

    mixer_st  = mixer_init_like(P0, diis_size)

    real_dtype = jnp.zeros((), dtype=target_dtype).real.dtype
    electrondensity = jnp.array(electrondensity0, dtype=real_dtype)

    hist_E  = jnp.zeros(max_iter, dtype=real_dtype)
    hist_dC = jnp.zeros(max_iter, dtype=real_dtype)

    # Make the infinities match the loop’s real dtype
    inf_r = jnp.asarray(jnp.inf, dtype=real_dtype)

    # carry = (iter, P, mixer_state, dC_prev, E_prev, hist_E, hist_dC)
    carry0 = (jnp.int32(0), P0, mixer_st, inf_r, inf_r, hist_E, hist_dC)

    def body(carry):
        k, P, mix_st, _dC_prev, E_prev, hE, hC = carry

        # 1) SCF step
        hf_result = hf_step(P, electrondensity)
        P_new, E_new = hf_step(P, electrondensity)
        F_new        = hf_step.h + selfenergy_fft(hf_step.VR, P_new)

        comm = (hf_step.weights *
                (F_new @ P_new - P_new @ F_new)) / hf_step.weight_sum
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
                lambda mode: lax.cond(mode, _announce_ediis, _announce_cdiis, operand=None),
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
    k_fin, P_fin, mix_fin, dC_fin, E_prev, hist_E, hist_dC = lax.while_loop(cond, body, carry0)

    # final energy
    E_fin = jnp.real(E_prev)

    # 5) convergence warning if we hit max_iter without converging
    did_fail = jnp.logical_and(k_fin == max_iter, dC_fin > comm_tol)

    def _warn(_):
        debug.print(
            f"Warning: SCF didn’t converge after {max_iter} iters (|comm|>{comm_tol:.1e})",
            ordered=True
        )
        return jnp.int32(0)

    _ = lax.cond(did_fail, _warn, lambda _: jnp.int32(0), operand=None)

    # compute the final mean field Hamiltonian
    F_fin = hf_step.h + selfenergy_fft(hf_step.VR, P_fin)
    mu_fin = hf_step._chemical_potential(P_fin, electrondensity)

    history = dict(E=hist_E, dC=hist_dC)
    return P_fin, F_fin, E_fin, mu_fin, k_fin, history


def jit_hartreefock_iteration(hf_step: HartreeFockKernel):
    compiled_hf_iteration_fn = jax.jit(
        partial(hartreefock_iteration, hf_step=hf_step),
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
# P_conv, F_conv, E_sym, n_iter, history = hartree_fock_iteration(h.zero*(1+0j), n_cn+n_e, **iteration_settings)
# hmf = jax_hf.wrappers.contimod_from_HartreeFock(h, model, P_conv)
# print(f"Converged in {int(n_iter)} steps: E = {E_conv:.8f}")
