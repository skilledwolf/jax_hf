"""Hartree-Fock kernel: precomputed arrays for the JIT solver loop."""

from __future__ import annotations

from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from .utils import selfenergy_fft, validate_electron_count


class HartreeFockKernel:
    """Precomputed arrays for the jitted solver loop.

    Absorbs the ifftshift phase into VR so the hot loop skips the data
    permutation.  All arrays are exposed via ``as_args()`` for passing
    as dynamic inputs to ``jax.jit``-compiled functions.
    """

    def __init__(
        self,
        weights,
        hamiltonian,
        coulomb_q,
        T: float,
        include_hartree: bool = False,
        include_exchange: bool = True,
        reference_density=None,
        hartree_matrix=None,
    ):
        h = jnp.asarray(hamiltonian)
        self.h = h
        w2d = jnp.asarray(weights, dtype=h.real.dtype)
        self.weights_b = w2d[..., None, None]
        self.weight_sum = jnp.sum(w2d)
        self.w2d = w2d

        Vq = jnp.asarray(coulomb_q)
        if jnp.iscomplexobj(Vq):
            imag_max = float(jnp.max(jnp.abs(jnp.imag(Vq))))
            if imag_max <= 1e-8:
                Vq = jnp.real(Vq)
        else:
            Vq = Vq.astype(h.real.dtype)
        self.exchange_hermitian_channel_packing = bool(
            Vq.shape[-2:] == (1, 1) and not jnp.iscomplexobj(Vq)
        )
        self.VR = jnp.fft.fftn(self.weights_b * jnp.asarray(Vq, dtype=h.dtype), axes=(0, 1))
        # Pre-absorb ifftshift phase into VR.
        n0, n1 = int(h.shape[0]), int(h.shape[1])
        s0, s1 = n0 // 2, n1 // 2
        phase0 = np.exp(2j * np.pi * np.arange(n0) * s0 / n0)
        phase1 = np.exp(2j * np.pi * np.arange(n1) * s1 / n1)
        shift_phase = jnp.asarray(
            (phase0[:, None] * phase1[None, :])[..., None, None],
            dtype=self.VR.dtype,
        )
        self._VR_shifted = self.VR * shift_phase
        self.T = float(T)
        self.include_hartree = bool(include_hartree)
        self.include_exchange = bool(include_exchange)
        if (not self.include_hartree) and (not self.include_exchange):
            raise ValueError("HartreeFockKernel must include at least one of Hartree or exchange.")

        if reference_density is not None:
            ref = jnp.asarray(reference_density, dtype=h.dtype)
            if ref.shape != h.shape:
                raise ValueError(f"reference_density must have shape {h.shape}, got {ref.shape}")
            self.refP = ref
        else:
            self.refP = jnp.zeros_like(h)

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

    def as_args(self):
        """Dynamic inputs for jitted solver functions (no constant capture)."""
        return dict(
            h=self.h,
            weights_b=self.weights_b,
            weight_sum=self.weight_sum,
            VR=self._VR_shifted,
            T=self.T,
            refP=self.refP,
            HH=self.HH,
            include_hartree=self.include_hartree,
            include_exchange=self.include_exchange,
            exchange_hermitian_channel_packing=self.exchange_hermitian_channel_packing,
        )
