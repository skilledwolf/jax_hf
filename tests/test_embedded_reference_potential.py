"""Mirror of cpp_hf/tests/test_embedded_reference_potential.py for jax_hf.

Same identities, same conventions — verifies the JAX implementation
behaves identically to the cpp_hf one for the embedded reference
potential mode.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from jax_hf import HartreeFockKernel
from jax_hf.fock import build_fock, hf_energy


def _setup(nk=4, n_orb=2, T=0.05, seed=0):
    weights = np.full((nk, nk), 1.0 / (nk * nk), dtype=np.float64)
    h_diag = np.array([-0.5, +0.5])
    h = np.broadcast_to(np.diag(h_diag), (nk, nk, n_orb, n_orb)).astype(np.complex128)
    h = np.ascontiguousarray(h)
    kx = np.linspace(-0.5, 0.5, nk, endpoint=False)
    h_pert = (kx[:, None, None, None] * np.array([[0.0, 1.0], [1.0, 0.0]])[None, None]
              ).astype(np.complex128)
    h = h + h_pert
    h = np.ascontiguousarray(h)
    Vq = np.full((nk, nk, 1, 1), 0.3, dtype=np.complex128)
    HH = np.array([[0.5, -0.1], [-0.1, 0.5]], dtype=np.float64)
    refP = np.zeros((nk, nk, n_orb, n_orb), dtype=np.complex128)
    refP[..., 0, 0] = 0.4
    refP[..., 1, 1] = 0.6
    return weights, h, Vq, HH, T, refP


def _make_kernels(*, embed: bool, with_hartree: bool, refP_arg, **extra):
    weights, h, Vq, HH, T, refP = _setup()
    return HartreeFockKernel(
        weights=weights, hamiltonian=h, coulomb_q=Vq, T=T,
        include_hartree=with_hartree,
        include_exchange=True,
        reference_density=refP_arg if refP_arg is not None else refP,
        hartree_matrix=HH if with_hartree else None,
        embed_reference_potential=embed,
        **extra,
    )


def _build_F(kernel, P):
    Sigma, H, F = build_fock(
        jnp.asarray(P, dtype=jnp.complex128),
        h=kernel.h, VR=kernel._VR_shifted, refP=kernel.refP,
        HH=kernel.HH, w2d=kernel.w2d,
        include_exchange=kernel.include_exchange,
        include_hartree=kernel.include_hartree,
        exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
        contact_g=kernel.contact_g,
        contact_Oi=kernel.contact_Oi,
        contact_Oj=kernel.contact_Oj,
    )
    return np.asarray(Sigma), np.asarray(H), np.asarray(F)


@pytest.mark.parametrize("with_hartree", [True, False])
def test_embedded_F_at_refP_equals_h_plus_V_HF_refP(with_hartree):
    kernel_embed = _make_kernels(
        embed=True, with_hartree=with_hartree, refP_arg=None,
        center_embedded_hartree=False,
    )
    kernel_plain = _make_kernels(
        embed=False, with_hartree=with_hartree, refP_arg=None,
    )
    refP = np.asarray(kernel_plain.refP)
    _, _, F_embed = _build_F(kernel_embed, refP)
    _, _, F_plain = _build_F(kernel_plain, refP)

    np.testing.assert_allclose(F_plain, np.asarray(kernel_plain.h), atol=1e-6)
    np.testing.assert_allclose(F_embed, np.asarray(kernel_embed.h), atol=1e-6)
    np.testing.assert_allclose(
        F_embed - F_plain,
        np.asarray(kernel_embed.h - kernel_plain.h),
        atol=1e-6,
    )


@pytest.mark.parametrize("with_hartree", [True, False])
def test_F_difference_independent_of_P(with_hartree):
    kernel_embed = _make_kernels(
        embed=True, with_hartree=with_hartree, refP_arg=None,
        center_embedded_hartree=False,
    )
    kernel_plain = _make_kernels(
        embed=False, with_hartree=with_hartree, refP_arg=None,
    )
    rng = np.random.default_rng(42)
    P = (rng.standard_normal(kernel_plain.h.shape)
         + 1j * rng.standard_normal(kernel_plain.h.shape))
    P = 0.5 * (P + np.conj(np.swapaxes(P, -1, -2)))
    P = P.astype(np.complex128)

    _, _, F_embed = _build_F(kernel_embed, P)
    _, _, F_plain = _build_F(kernel_plain, P)
    diff = F_embed - F_plain
    expected = np.asarray(kernel_embed.h - kernel_plain.h)
    np.testing.assert_allclose(diff, expected, atol=1e-5)


@pytest.mark.parametrize("with_hartree", [True, False])
def test_embedding_no_op_when_refP_zero(with_hartree):
    weights, h, Vq, HH, T, _ = _setup()
    refP_zero = np.zeros_like(h)
    if not with_hartree:
        kwargs = dict(include_hartree=False, hartree_matrix=None)
    else:
        kwargs = dict(include_hartree=True, hartree_matrix=HH)
    kernel = HartreeFockKernel(
        weights=weights, hamiltonian=h, coulomb_q=Vq, T=T,
        include_exchange=True,
        reference_density=refP_zero,
        embed_reference_potential=True,
        center_embedded_hartree=False,
        **kwargs,
    )
    np.testing.assert_allclose(np.asarray(kernel.h), h, atol=1e-6)
    assert float(kernel.embedded_energy_offset) == 0.0


def test_embedded_energy_offset_is_correct():
    kernel_embed = _make_kernels(
        embed=True, with_hartree=True, refP_arg=None,
        center_embedded_hartree=False,
    )
    kernel_plain = _make_kernels(
        embed=False, with_hartree=True, refP_arg=None,
    )
    refP = np.asarray(kernel_plain.refP)
    Sigma_p, H_p, _ = _build_F(kernel_plain, refP)
    Sigma_e, H_e, _ = _build_F(kernel_embed, refP)
    E_plain = float(hf_energy(jnp.asarray(refP), h=kernel_plain.h,
                              Sigma=jnp.asarray(Sigma_p),
                              H=jnp.asarray(H_p),
                              weights_b=kernel_plain.weights_b))
    E_embed = float(hf_energy(jnp.asarray(refP), h=kernel_embed.h,
                              Sigma=jnp.asarray(Sigma_e),
                              H=jnp.asarray(H_e),
                              weights_b=kernel_embed.weights_b))
    delta = E_embed - E_plain
    expected = 2.0 * float(kernel_embed.embedded_energy_offset)
    np.testing.assert_allclose(delta, expected, atol=1e-5, rtol=1e-6)
