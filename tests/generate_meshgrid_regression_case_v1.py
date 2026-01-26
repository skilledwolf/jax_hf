from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Keep this script reproducible by default.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax.numpy as jnp  # noqa: E402

from jax_hf.main import HartreeFockKernel, jit_hartreefock_iteration  # noqa: E402
from jax_hf.utils import fermidirac, find_chemical_potential  # noqa: E402


def generate(path: Path) -> None:
    nk = 12
    kmax = 0.8
    T = 0.25

    # Centered meshgrid (k=0 at center index).
    k = np.linspace(-kmax, kmax, nk, endpoint=False, dtype=np.float32)
    KX, KY = np.meshgrid(k, k, indexing="ij")

    # 4-band Hermitian Hamiltonian with nontrivial couplings.
    t = 1.1
    m0 = 0.7
    m1 = 0.3

    d0 = 0.2 * (np.cos(KX) + np.cos(KY))
    dz = m0 + m1 * (2.0 - np.cos(KX) - np.cos(KY))
    f = t * (np.sin(KX) + 1j * np.sin(KY))

    H2 = np.zeros((nk, nk, 2, 2), dtype=np.complex64)
    H2[..., 0, 0] = d0 + dz
    H2[..., 1, 1] = d0 - dz
    H2[..., 0, 1] = f
    H2[..., 1, 0] = np.conj(f)

    H = np.zeros((nk, nk, 4, 4), dtype=np.complex64)
    I2 = np.eye(2, dtype=np.complex64)
    delta = 0.15
    H[..., :2, :2] = H2 + delta * I2
    H[..., 2:, 2:] = H2 - delta * I2

    coupling = (0.07 * (np.cos(KX) - np.cos(KY))).astype(np.float32)
    H[..., 0, 2] = coupling
    H[..., 2, 0] = coupling
    H[..., 1, 3] = -coupling
    H[..., 3, 1] = -coupling

    ic = (0.05 * np.sin(KX - KY)).astype(np.float32)
    H[..., 0, 3] = 1j * ic
    H[..., 3, 0] = -1j * ic

    # Uniform weights (sum=1).
    weights = np.ones((nk, nk), dtype=np.float32) / (nk * nk)

    # Screened Coulomb in q-space; stored as (nk,nk,1,1) so broadcasting matches.
    q = np.sqrt(KX * KX + KY * KY)
    Vq = (0.6 / (q + 0.4)).astype(np.float32)[..., None, None]

    # Reference and initial densities from non-interacting bands.
    bands, U = jnp.linalg.eigh(jnp.asarray(H))
    weights_j = jnp.asarray(weights)

    n_ref = 2.0
    mu_ref = find_chemical_potential(bands, weights_j, n_electrons=n_ref, T=T)
    occ_ref = fermidirac(bands - mu_ref, T)
    P_ref = jnp.einsum("...in,...n,...jn->...ij", U, occ_ref, jnp.conj(U))

    n_e = 2.15
    mu0 = find_chemical_potential(bands, weights_j, n_electrons=n_e, T=T)
    occ0 = fermidirac(bands - mu0, T)
    P0 = jnp.einsum("...in,...n,...jn->...ij", U, occ0, jnp.conj(U))

    # Hartree coupling matrix (real).
    HH = (
        np.array(
            [
                [1.0, 0.2, 0.1, 0.0],
                [0.2, 1.1, 0.0, 0.1],
                [0.1, 0.0, 1.2, 0.2],
                [0.0, 0.1, 0.2, 1.3],
            ],
            dtype=np.float32,
        )
        * 0.4
    )

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=H,
        coulomb_q=Vq,
        T=T,
        include_hartree=True,
        include_exchange=True,
        reference_density=np.array(P_ref),
        hartree_matrix=HH,
    )

    run = jit_hartreefock_iteration(kernel)
    max_iter = 60
    comm_tol = 1e-6
    diis_size = 6

    P_fin, F_fin, E_fin, mu_fin, k_fin, history = run(
        P0,
        electrondensity0=n_e,
        max_iter=max_iter,
        comm_tol=comm_tol,
        diis_size=diis_size,
        precond_mode="auto",
        precond_auto_nb=4,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.array("v1"),
        nk=np.array(nk, dtype=np.int32),
        kmax=np.array(kmax, dtype=np.float32),
        T=np.array(T, dtype=np.float32),
        weights=weights,
        hamiltonian=H,
        coulomb_q=Vq,
        hartree_matrix=HH,
        reference_density=np.array(P_ref),
        P0=np.array(P0),
        electrondensity0=np.array(n_e, dtype=np.float32),
        # solver controls
        max_iter=np.array(max_iter, dtype=np.int32),
        comm_tol=np.array(comm_tol, dtype=np.float32),
        diis_size=np.array(diis_size, dtype=np.int32),
        precond_mode=np.array("auto"),
        precond_auto_nb=np.array(4, dtype=np.int32),
        # expected outputs
        P_expected=np.array(P_fin),
        F_expected=np.array(F_fin),
        E_expected=np.array(E_fin),
        mu_expected=np.array(mu_fin),
        k_expected=np.array(k_fin),
        hist_E_expected=np.array(history["E"]),
        hist_dC_expected=np.array(history["dC"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "data" / "meshgrid_regression_case_v1.npz",
        help="Output .npz path",
    )
    args = parser.parse_args()
    generate(args.out)
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

