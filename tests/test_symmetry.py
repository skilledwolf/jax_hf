from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from jax_hf.symmetry import make_project_fn


def test_make_project_fn_identity_when_no_args():
    proj = make_project_fn()
    A = jnp.ones((2, 2, 3, 3), dtype=jnp.complex64)
    out = proj(A)
    np.testing.assert_array_equal(np.array(A), np.array(out))


def test_unitary_group_averaging_produces_invariant_output():
    nb = 4
    eye = np.eye(nb, dtype=np.complex64)
    g = np.diag(np.array([1, 1, -1, -1], dtype=np.complex64))
    G = jnp.stack([jnp.asarray(eye), jnp.asarray(g)], axis=0)

    proj = make_project_fn(unitary_group=G)

    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, (3, 3, nb, nb))
    A = (a + 1j * jax.random.normal(jax.random.PRNGKey(7), (3, 3, nb, nb))).astype(
        jnp.complex64
    )

    A_proj = proj(A)

    for i in range(G.shape[0]):
        gi = G[i]
        giH = jnp.conj(gi.T)
        rotated = (gi @ A_proj) @ giH
        np.testing.assert_allclose(
            np.array(rotated), np.array(A_proj), atol=1e-6, rtol=1e-6
        )


def test_unitary_group_averaging_is_hermitian_preserving():
    nb = 3
    eye = np.eye(nb, dtype=np.complex64)
    perm = np.zeros((nb, nb), dtype=np.complex64)
    perm[0, 1] = perm[1, 2] = perm[2, 0] = 1.0
    perm2 = perm @ perm
    G = jnp.stack([jnp.asarray(eye), jnp.asarray(perm), jnp.asarray(perm2)], axis=0)

    proj = make_project_fn(unitary_group=G)

    key = jax.random.PRNGKey(99)
    a = jax.random.normal(key, (2, 2, nb, nb))
    A = (a + 1j * jax.random.normal(jax.random.PRNGKey(100), (2, 2, nb, nb))).astype(
        jnp.complex64
    )
    A_herm = 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))

    result = proj(A_herm)
    np.testing.assert_allclose(
        np.array(result),
        np.array(jnp.conj(jnp.swapaxes(result, -1, -2))),
        atol=1e-6,
    )


def test_time_reversal_averaging_flip():
    nk1, nk2, nb = 4, 4, 2
    U_tr = jnp.asarray(np.array([[0, -1j], [1j, 0]], dtype=np.complex64))

    proj = make_project_fn(
        time_reversal_U=U_tr,
        time_reversal_k_convention="flip",
    )

    key = jax.random.PRNGKey(55)
    a = jax.random.normal(key, (nk1, nk2, nb, nb))
    A = (a + 1j * jax.random.normal(jax.random.PRNGKey(56), (nk1, nk2, nb, nb))).astype(
        jnp.complex64
    )

    A_proj = proj(A)

    UH = jnp.conj(U_tr.T)
    A_neg = jnp.flip(A_proj, axis=(0, 1))
    A_tr = U_tr @ jnp.conj(A_neg) @ UH
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_tr), atol=1e-6, rtol=1e-6
    )


def test_time_reversal_averaging_mod():
    nk1, nk2, nb = 5, 5, 2
    U_tr = jnp.eye(nb, dtype=jnp.complex64)

    proj = make_project_fn(
        time_reversal_U=U_tr,
        time_reversal_k_convention="mod",
    )

    key = jax.random.PRNGKey(77)
    a = jax.random.normal(key, (nk1, nk2, nb, nb))
    A = (a + 1j * jax.random.normal(jax.random.PRNGKey(78), (nk1, nk2, nb, nb))).astype(
        jnp.complex64
    )

    A_proj = proj(A)

    UH = jnp.conj(U_tr.T)
    i_idx = (-jnp.arange(nk1, dtype=jnp.int32)) % nk1
    j_idx = (-jnp.arange(nk2, dtype=jnp.int32)) % nk2
    A_neg = A_proj[i_idx[:, None], j_idx[None, :], ...]
    A_tr = U_tr @ jnp.conj(A_neg) @ UH
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_tr), atol=1e-6, rtol=1e-6
    )


def test_combined_unitary_and_time_reversal():
    nk1, nk2, nb = 4, 4, 2
    eye = np.eye(nb, dtype=np.complex64)
    sigma_z = np.diag(np.array([1, -1], dtype=np.complex64))
    G = jnp.stack([jnp.asarray(eye), jnp.asarray(sigma_z)], axis=0)
    U_tr = jnp.eye(nb, dtype=jnp.complex64)

    proj = make_project_fn(
        unitary_group=G,
        time_reversal_U=U_tr,
        time_reversal_k_convention="flip",
    )

    key = jax.random.PRNGKey(11)
    a = jax.random.normal(key, (nk1, nk2, nb, nb))
    A = (a + 1j * jax.random.normal(jax.random.PRNGKey(12), (nk1, nk2, nb, nb))).astype(
        jnp.complex64
    )

    A_proj = proj(A)

    for i in range(G.shape[0]):
        gi = G[i]
        giH = jnp.conj(gi.T)
        rotated = (gi @ A_proj) @ giH
        np.testing.assert_allclose(
            np.array(rotated), np.array(A_proj), atol=1e-5, rtol=1e-5
        )

    UH = jnp.conj(U_tr.T)
    A_neg = jnp.flip(A_proj, axis=(0, 1))
    A_tr = U_tr @ jnp.conj(A_neg) @ UH
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_tr), atol=1e-5, rtol=1e-5
    )


def test_projection_is_idempotent():
    nb = 3
    eye = np.eye(nb, dtype=np.complex64)
    perm = np.zeros((nb, nb), dtype=np.complex64)
    perm[0, 1] = perm[1, 2] = perm[2, 0] = 1.0
    perm2 = perm @ perm
    G = jnp.stack([jnp.asarray(eye), jnp.asarray(perm), jnp.asarray(perm2)], axis=0)

    proj = make_project_fn(unitary_group=G)

    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (2, 2, nb, nb), dtype=jnp.float32).astype(
        jnp.complex64
    )

    once = proj(A)
    twice = proj(once)
    np.testing.assert_allclose(np.array(once), np.array(twice), atol=1e-6)


def test_spatial_group_produces_k_flip_invariant_output():
    nk1, nk2, nb = 4, 4, 2
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    S = jnp.asarray(sigma_x)[None]

    proj = make_project_fn(spatial_group=S, spatial_k_convention="flip")

    key = jax.random.PRNGKey(33)
    A = jax.random.normal(key, (nk1, nk2, nb, nb)).astype(jnp.complex64)

    A_proj = proj(A)

    sxH = jnp.conj(sigma_x.T)
    A_neg = jnp.flip(A_proj, axis=(0, 1))
    A_rotated = jnp.asarray(sigma_x) @ A_neg @ jnp.asarray(sxH)
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_rotated), atol=1e-6, rtol=1e-6
    )


def test_combined_same_k_and_flip_k_group():
    nk1, nk2, nb = 4, 4, 2
    eye = np.eye(nb, dtype=np.complex64)
    sigma_z = np.diag(np.array([1, -1], dtype=np.complex64))
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)

    G_same = jnp.stack([jnp.asarray(eye), jnp.asarray(sigma_z)], axis=0)
    G_flip = jnp.asarray(sigma_x)[None]

    proj = make_project_fn(
        unitary_group=G_same,
        spatial_group=G_flip,
        spatial_k_convention="flip",
    )

    key = jax.random.PRNGKey(44)
    A = jax.random.normal(key, (nk1, nk2, nb, nb)).astype(jnp.complex64)
    A_proj = proj(A)

    A_neg = jnp.flip(A, axis=(0, 1))
    expected = (
        A
        + jnp.asarray(sigma_z) @ A @ jnp.conj(jnp.asarray(sigma_z).T)
        + jnp.asarray(sigma_x) @ A_neg @ jnp.conj(jnp.asarray(sigma_x).T)
    ) / 3.0
    np.testing.assert_allclose(
        np.array(A_proj), np.array(expected), atol=1e-6, rtol=1e-6
    )


def test_kx_only_flip_is_idempotent():
    nk1, nk2, nb = 6, 4, 2
    eye = np.eye(nb, dtype=np.complex64)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    G_same = jnp.asarray(eye)[None]
    G_flip = jnp.asarray(sigma_x)[None]

    proj = make_project_fn(
        unitary_group=G_same,
        spatial_group=G_flip,
        spatial_k_convention="flip",
        spatial_k_flip_axes=(0,),
    )

    key = jax.random.PRNGKey(66)
    A = jax.random.normal(key, (nk1, nk2, nb, nb)).astype(jnp.complex64)

    once = proj(A)
    twice = proj(once)
    np.testing.assert_allclose(np.array(once), np.array(twice), atol=1e-5)


