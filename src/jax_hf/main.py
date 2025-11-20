from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.numpy.linalg import eigh

from .mixing import mixer_init_like, mixer_update
from .utils import fermidirac, find_chemical_potential, selfenergy_fft


class HartreeFockKernel:
    """
    Thin container for precomputations; keeps your __call__ one-step API,
    but we do NOT call it inside the jitted loop (to avoid closure capture).
    """
    def __init__(self, weights, hamiltonian, coulomb_q, T: float,
                 include_hartree: bool=False, include_exchange: bool=True,
                 reference_density: jnp.ndarray | None=None, hartree_matrix: jnp.ndarray | None=None):
        h = jnp.asarray(hamiltonian)
        self.h = h  # complex
        # broadcast weights for cheap elementwise ops in k-space
        w2d = jnp.asarray(weights, dtype=h.real.dtype)  # weights are physically real
        self.weights_b = w2d[..., None, None]           # keep real to avoid downstream casting warnings
        self.weight_sum = jnp.sum(w2d)
        self.w2d = w2d

        Vq = jnp.asarray(coulomb_q)
        if jnp.iscomplexobj(Vq):
            imag_max = float(jnp.max(jnp.abs(jnp.imag(Vq))))
            if imag_max <= 1e-8:
                Vq = jnp.real(Vq)  # small phase noise; use real Coulomb
            # else keep complex Vq as provided
        else:
            Vq = Vq.astype(h.real.dtype)
        self.VR = jnp.fft.fftn(self.weights_b * jnp.asarray(Vq, dtype=h.dtype), axes=(0, 1))
        self.T = float(T)
        # options
        self.include_hartree = bool(include_hartree)
        self.include_exchange = bool(include_exchange)
        if (not self.include_hartree) and (not self.include_exchange):
            raise ValueError("HartreeFockKernel must include at least one of Hartree or exchange.")
        # reference density for subtraction (full matrix on grid)
        if reference_density is not None:
            ref = jnp.asarray(reference_density, dtype=h.dtype)
            if ref.shape != h.shape:
                raise ValueError(f"reference_density must have shape {h.shape}, got {ref.shape}")
            self.refP = ref
        else:
            self.refP = jnp.zeros_like(h)
        # Hartree matrix (nb, nb)
        if self.include_hartree:
            if hartree_matrix is None:
                raise ValueError("include_hartree=True requires hartree_matrix to be provided")
            if reference_density is None:
                raise ValueError("include_hartree=True requires reference_density to be provided")
            HH = jnp.asarray(hartree_matrix, dtype=h.real.dtype)
            if HH.shape != h.shape[-2:]:
                raise ValueError(f"hartree_matrix must have shape {h.shape[-2:]}, got {HH.shape}")
            self.HH = HH
        else:
            self.HH = jnp.zeros(h.shape[-2:], dtype=h.real.dtype)

    def _exchange_sigma(self, P: jax.Array) -> jax.Array:
        if not self.include_exchange:
            return jnp.zeros_like(self.h)
        return _herm(selfenergy_fft(self.VR, _herm(P - self.refP)))

    def _hartree_shift(self, P: jax.Array) -> jax.Array:
        if not self.include_hartree:
            return jnp.zeros_like(self.h)
        dP = _herm(P) - self.refP
        # Per-orbital fluctuation density integrated over k: n_vec[α]
        diag_real = jnp.real(jnp.diagonal(dP, axis1=-2, axis2=-1))  # (..., nb)
        n_vec = jnp.sum(self.w2d[..., None] * diag_real, axis=(0, 1))  # (nb,)
        # Hartree self-energy is diagonal: (Σ_H)_αα = sum_β U_{αβ} n_β
        sigma_diag = self.HH @ n_vec  # (nb,)
        H_mat = jnp.diag(sigma_diag.astype(self.h.real.dtype))  # (nb,nb)
        return H_mat[None, None, ...]

    def fock_matrix(self, P: jax.Array) -> jax.Array:
        return _herm(self.h + self._exchange_sigma(P) + self._hartree_shift(P))

    def __call__(self, P: jax.Array, electrondensity: float):
        # one-step HF update (convenience; *not* used by the jitted loop)
        F     = self.fock_matrix(P)
        eps, U = jnp.linalg.eigh(F)
        mu    = find_chemical_potential(eps, self.w2d, electrondensity, self.T)
        occ   = fermidirac(eps - mu, self.T)
        P_new = _density_from(U, occ)
        # Energy with Σ[P_new] and Hartree shift at P_new
        Sigma_new = self._exchange_sigma(P_new)
        H_new = self._hartree_shift(P_new)
        F_new = _herm(self.h + Sigma_new + H_new)
        E_new = jnp.sum(jnp.real(jnp.einsum("...ij,...ji->...", self.weights_b * (self.h + 0.5*(Sigma_new + H_new)), P_new)))
        return HFStepResult(P_new, F_new, E_new, mu)

    def as_args(self):
        # everything the jitted loop needs, as dynamic inputs (no constant capture)
        return dict(
            h=self.h,
            weights_b=self.weights_b,
            weight_sum=self.weight_sum,
            VR=self.VR,
            T=self.T,
            refP=self.refP,
            HH=self.HH,
            include_hartree=self.include_hartree,
            include_exchange=self.include_exchange,
        )



def _herm(X):
    return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))


def _density_from(U, occ):
    # U[..., i, n], occ[..., n]
    return jnp.einsum("...in,...n,...jn->...ij", U, occ, jnp.conj(U))


def _hf_energy(weights_b, h, sigma, P):
    # E = sum_k w_k Tr[(h + 0.5Σ) P]
    return jnp.sum(
        jnp.real(jnp.einsum("...ij,...ji->...", weights_b * (h + 0.5 * sigma), P))
    )


def hartreefock_iteration(
    P0: jax.Array,
    electrondensity0: float,
    *,
    # ---- pass all BIG arrays at call time (dynamic, not captured) ----
    h: jax.Array,                 # (nk1,nk2,nb,nb), complex
    weights_b: jax.Array,         # (nk1,nk2,1,1), real/complex matching h.dtype
    weight_sum: jax.Array,        # scalar real
    VR: jax.Array,                # (nk1,nk2,1,1), complex FFT(weights*V)
    T: float,
    refP: jax.Array,
    HH: jax.Array,
    include_hartree: bool,
    include_exchange: bool,

    # ---- small controls (static) ----
    max_iter:  int   = 100,
    comm_tol:  float = 5e-3,
    diis_size: int   = 4,
    log_every: int | None = None,   # kept for API compatibility; no prints inside JIT
    mixing_alpha: float = 1.0,
    precond_delta: float = 5e-3,
):
    # unify dtype once
    target_dtype = h.dtype
    P0 = jnp.asarray(P0, dtype=target_dtype)
    real_dtype = jnp.zeros((), dtype=target_dtype).real.dtype

    # weights for mu: (nk1,nk2)
    w2d = jnp.asarray(weights_b[..., 0, 0], dtype=real_dtype)
    n_e = jnp.asarray(electrondensity0, dtype=real_dtype)

    # init mixer/state/history
    mix_st  = mixer_init_like(P0, diis_size)
    hist_E  = jnp.zeros(max_iter, dtype=real_dtype)
    hist_dC = jnp.zeros(max_iter, dtype=real_dtype)
    comm_tol_r = jnp.asarray(comm_tol, dtype=real_dtype)

    # carry = (k, P, mixer_state, dC_prev, E_prev, hist_E, hist_dC)
    carry0 = (jnp.int32(0), _herm(P0), mix_st, jnp.asarray(jnp.inf, dtype=real_dtype),
              jnp.asarray(jnp.inf, dtype=real_dtype), hist_E, hist_dC)

    def cond(carry):
        k, _, _, dC, *_ = carry
        return jnp.logical_and(k < max_iter, dC > comm_tol_r)

    def body(carry):
        k, P, mix_st, _dC_prev, E_prev, hE, hC = carry

        # ---- 1) Build Fock at INPUT density, diagonalize, make new density
        Sigma_in = selfenergy_fft(VR, _herm(P - refP)) if include_exchange else jnp.zeros_like(h)
        if include_hartree:
            dP_in = _herm(P) - refP
            diag_real_in = jnp.real(jnp.diagonal(dP_in, axis1=-2, axis2=-1))
            n_vec_in = jnp.sum(w2d[..., None] * diag_real_in, axis=(0, 1))
            sigma_diag_in = HH @ n_vec_in
            H_in = jnp.diag(sigma_diag_in.astype(h.real.dtype))[None, None, ...]
        else:
            H_in = jnp.zeros_like(h)
        F_in     = h + Sigma_in + H_in
        eps, U   = eigh(F_in)
        mu       = find_chemical_potential(eps, w2d, n_e, T)
        occ      = fermidirac(eps - mu, T)
        P_new    = _density_from(U, occ)

        # ---- 2) Build Fock at NEW density, energy, and commutator residual
        Sigma_new = selfenergy_fft(VR, _herm(P_new - refP)) if include_exchange else jnp.zeros_like(h)
        if include_hartree:
            dP_new = _herm(P_new) - refP
            diag_real_new = jnp.real(jnp.diagonal(dP_new, axis1=-2, axis2=-1))
            n_vec_new = jnp.sum(w2d[..., None] * diag_real_new, axis=(0, 1))
            sigma_diag_new = HH @ n_vec_new
            H_new = jnp.diag(sigma_diag_new.astype(h.real.dtype))[None, None, ...]
        else:
            H_new = jnp.zeros_like(h)
        F_new     = h + Sigma_new + H_new
        # E = sum_k w_k Tr[(h + 0.5(Σ+H)) P]
        E_new     = jnp.sum(jnp.real(jnp.einsum("...ij,...ji->...", weights_b * (h + 0.5*(Sigma_new + H_new)), P_new)))

        R  = F_new @ P_new - P_new @ F_new
        # weighted RMS(comm): sqrt(sum w ||R||_F^2 / sum w)
        sq = jnp.abs(R) ** 2
        per_k = jnp.sum(sq, axis=(-2, -1))
        dC = jnp.sqrt(jnp.sum(w2d * per_k) / jnp.maximum(weight_sum, jnp.array(1e-30, dtype=real_dtype)))

        # ---- 3) DIIS / mixing
        mix_st, P_mix = mixer_update(
            mix_st, P_new, P, F_new, E_new, (jnp.sqrt(jnp.maximum(w2d, 0.0))[..., None, None] * R),
            comm_rms=dC, sqrt_weights=jnp.sqrt(jnp.maximum(w2d, 0.0)),
            to_cdiis=9.0 * comm_tol_r, to_broyden=1.5 * comm_tol_r,
            mixing_alpha=mixing_alpha, precond_delta=precond_delta,
            cdiis_blend_keep=0.5, cdiis_blend_new=0.5,
        )

        # ---- 4) history
        hE = hE.at[k].set(jnp.real(E_new))
        hC = hC.at[k].set(dC)

        return (k + 1, _herm(P_mix), mix_st, dC, jnp.real(E_new), hE, hC)

    k_fin, P_fin, mix_fin, dC_fin, E_prev, hist_E, hist_dC = lax.while_loop(cond, body, carry0)

    # finalize (compute last Fock & mu)
    F_fin  = h + selfenergy_fft(VR, P_fin)
    eps_f, _ = eigh(F_fin)
    mu_fin = find_chemical_potential(eps_f, w2d, n_e, T)
    E_fin  = jnp.real(E_prev)

    history = dict(E=hist_E, dC=hist_dC)
    return P_fin, F_fin, E_fin, mu_fin, k_fin, history


def jit_hartreefock_iteration(hf_step: HartreeFockKernel):
    _compiled = jax.jit(
        hartreefock_iteration,                     # the while_loop version
        static_argnames=("max_iter", "comm_tol", "diis_size", "log_every", "include_hartree", "include_exchange"),
    )
    def run(P0, electrondensity0, **kwargs):
        return _compiled(P0, electrondensity0, **hf_step.as_args(), **kwargs)
    return run
class HFStepResult(NamedTuple):
    density: jax.Array
    fock: jax.Array
    energy: jax.Array
    mu: jax.Array
