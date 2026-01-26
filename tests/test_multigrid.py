import numpy as np

import jax.numpy as jnp

from jax_hf.multigrid import coarse_to_fine_scf
from jax_hf.utils import resample_kgrid


def test_resample_kgrid_preserves_constants():
    x = (1.25 - 0.3j) * jnp.ones((12, 12, 2, 2), dtype=jnp.complex64)
    y = resample_kgrid(x, 7)
    assert y.shape == (7, 7, 2, 2)
    np.testing.assert_allclose(np.array(y), (1.25 - 0.3j), rtol=0, atol=1e-6)


def test_coarse_to_fine_scf_runs():
    nk_f = 12
    nk_c = 6
    kmax = 0.8
    T = 0.25

    k = np.linspace(-kmax, kmax, nk_f, endpoint=False, dtype=np.float32)
    KX, KY = np.meshgrid(k, k, indexing="ij")

    # 2-band Hermitian Hamiltonian.
    t = 1.1
    m0 = 0.7
    m1 = 0.3
    d0 = 0.2 * (np.cos(KX) + np.cos(KY))
    dz = m0 + m1 * (2.0 - np.cos(KX) - np.cos(KY))
    f = t * (np.sin(KX) + 1j * np.sin(KY))

    H = np.zeros((nk_f, nk_f, 2, 2), dtype=np.complex64)
    H[..., 0, 0] = d0 + dz
    H[..., 1, 1] = d0 - dz
    H[..., 0, 1] = f
    H[..., 1, 0] = np.conj(f)

    # Uniform weights (sum=1).
    weights = np.ones((nk_f, nk_f), dtype=np.float32) / (nk_f * nk_f)

    # Screened Coulomb kernel in q-space; stored as (nk,nk,1,1).
    q = np.sqrt(KX * KX + KY * KY)
    Vq = (0.6 / (q + 0.4)).astype(np.float32)[..., None, None]

    # Half-filling for 2 bands with weights.sum()=1.
    electrondensity0 = 1.0

    # Simple diagonal initial density (already Hermitian).
    P0 = np.zeros((nk_f, nk_f, 2, 2), dtype=np.complex64)
    P0[..., 0, 0] = 0.6
    P0[..., 1, 1] = 0.4

    out = coarse_to_fine_scf(
        weights_f=weights,
        hamiltonian_f=H,
        coulomb_q_f=Vq,
        P0_f=P0,
        electrondensity0=electrondensity0,
        T=T,
        nk_coarse=nk_c,
        include_hartree=False,
        include_exchange=True,
        coarse_scf_kwargs=dict(max_iter=25, comm_tol=1e-3, diis_size=4, precond_mode="diag"),
        fine_scf_kwargs=dict(max_iter=25, comm_tol=1e-3, diis_size=4, precond_mode="diag"),
    )

    assert out.coarse is not None
    assert out.Sigma_seed_f is not None
    assert out.P0_seed_f is not None

    assert out.coarse.density.shape == (nk_c, nk_c, 2, 2)
    assert out.fine.density.shape == (nk_f, nk_f, 2, 2)
    assert out.Sigma_seed_f.shape == (nk_f, nk_f, 2, 2)

    # Hermiticity checks.
    P_fin = np.array(out.fine.density)
    np.testing.assert_allclose(P_fin, np.conj(np.swapaxes(P_fin, -1, -2)), atol=1e-6)

