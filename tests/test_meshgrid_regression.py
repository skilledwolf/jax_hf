from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp

from jax_hf.main import HartreeFockKernel, jit_hartreefock_iteration


def test_meshgrid_regression_case_v1_matches_reference():
    case_path = Path(__file__).resolve().parent / "data" / "meshgrid_regression_case_v1.npz"
    data = np.load(case_path)

    # Inputs
    weights = data["weights"]
    hamiltonian = data["hamiltonian"]
    coulomb_q = data["coulomb_q"]
    T = float(data["T"])
    P0 = data["P0"]
    electrondensity0 = float(data["electrondensity0"])
    reference_density = data["reference_density"]
    hartree_matrix = data["hartree_matrix"]

    # Solver controls
    max_iter = int(data["max_iter"])
    comm_tol = float(data["comm_tol"])
    diis_size = int(data["diis_size"])
    precond_mode = str(data["precond_mode"])
    precond_auto_nb = int(data["precond_auto_nb"])

    # Expected outputs
    P_expected = data["P_expected"]
    F_expected = data["F_expected"]
    E_expected = float(data["E_expected"])
    mu_expected = float(data["mu_expected"])
    k_expected = int(data["k_expected"])
    hist_E_expected = data["hist_E_expected"]
    hist_dC_expected = data["hist_dC_expected"]

    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=T,
        include_hartree=True,
        include_exchange=True,
        reference_density=reference_density,
        hartree_matrix=hartree_matrix,
    )

    run = jit_hartreefock_iteration(kernel)
    P_fin, F_fin, E_fin, mu_fin, k_fin, history = run(
        jnp.asarray(P0),
        electrondensity0=electrondensity0,
        max_iter=max_iter,
        comm_tol=comm_tol,
        diis_size=diis_size,
        precond_mode=precond_mode,
        precond_auto_nb=precond_auto_nb,
    )

    np.testing.assert_allclose(np.array(P_fin), P_expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(F_fin), F_expected, rtol=1e-6, atol=1e-6)
    assert float(E_fin) == pytest.approx(E_expected, rel=1e-6, abs=1e-6)
    assert float(mu_fin) == pytest.approx(mu_expected, rel=1e-6, abs=1e-6)
    assert int(k_fin) == k_expected

    np.testing.assert_allclose(np.array(history["E"]), hist_E_expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(history["dC"]), hist_dC_expected, rtol=1e-6, atol=1e-6)
