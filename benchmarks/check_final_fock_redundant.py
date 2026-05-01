"""Check whether the second build_fock in the solver finalize is redundant.

After the loop, the finalize block:
  1. P_pre = project(Q_fin @ diag(p_fin) @ Q_fin†)
  2. F_pre = build_fock(P_pre)
  3. eigh(Q_fin† F_pre Q_fin) → eps_fin, V_fin
  4. Q_fin ← Q_fin @ V_fin   (canonicalize)
  5. mu_fin from eps_fin, p_fin from FD(eps_fin - mu_fin)
  6. P_fin = project(Q_fin @ diag(p_fin) @ Q_fin†)
  7. F_fin = build_fock(P_fin)        ← redundant if P_fin == P_pre
  8. E_fin from F_fin

For converged solutions, the canonicalization (step 4) is a unitary rotation
within the gauge degrees of freedom — P should be unchanged.  Test it.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jax_hf import HartreeFockKernel, SolverConfig, solve

DATA_PATH = Path(__file__).parent.parent / "tests" / "data" / "bilayer_reference.npz"


def main():
    if not DATA_PATH.exists():
        print(f"Reference file not found: {DATA_PATH}")
        return
    ref = np.load(str(DATA_PATH), allow_pickle=False)

    try:
        import contimod as cm
        from contimod.utils.spectrum_fermi import FermiParams
        from contimod.meanfield.init_guess import init_to_density_matrix
        import contimod_graphene.symmetry as cg_symmetry
    except ImportError:
        print("contimod not available; skipping")
        return

    nk = int(ref["nk"])
    kmax = float(ref["kmax"])
    U_meV = float(ref["U_meV"])
    temperature = float(ref["temperature"])
    epsilon_r = float(ref["epsilon_r"])
    d_gate = float(ref["d_gate"])
    init_scale = float(ref["init_scale"])

    model = cm.graphene.MultilayerAB(valleyful=True, spinful=True, U=U_meV)
    h_template = model.discretize(nk=nk, kmax=kmax)
    h_template.fermi = FermiParams(T=temperature, mu=0.0)
    ne_cn = float(h_template.state.compute_density_from_filling(0.5 - 1e-6))
    weights = np.asarray(h_template.kmesh.weights)
    Vq = cm.coulomb.dualgate_coulomb(
        h_template.kmesh.distance_grid, epsilon_r=epsilon_r, d_gate=d_gate,
    )
    Vq = np.asarray(Vq.magnitude)[..., None, None]

    identity_op = np.asarray(model.identity)
    s3 = np.asarray(model.spin_op(3))
    v3 = np.asarray(model.valley_op(3))
    projector_sv = 0.25 * (identity_op + s3) @ (identity_op + v3)
    sv_contrast = -projector_sv + 3.0 * (identity_op - projector_sv)
    project_fn = cg_symmetry.make_svp_project_fn(
        s3=jnp.asarray(s3), v3=jnp.asarray(v3), n_orb=4,
        outlier_sv=(+1, +1), k_convention="flip", k_flip_axes=(0,),
    )

    PER_CM = 0.246e-7
    n_cm12 = 0.05
    dd = n_cm12 * 1e12 * (PER_CM ** 2)
    h_run = h_template.copy()
    h_run.fermi = FermiParams(T=temperature, mu=0.0)
    h_run.compute_chemicalpotential(density=float(ne_cn + dd))
    n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))
    seed = -init_scale * h_template.get_operator(sv_contrast)
    P0 = init_to_density_matrix(h_run, seed, density=None,
                                T=temperature, init_kind="auto")

    kernel = HartreeFockKernel(
        weights=weights, hamiltonian=np.asarray(h_template.hs),
        coulomb_q=Vq, T=temperature,
        include_hartree=False, include_exchange=True,
    )
    cfg = SolverConfig(max_iter=200, tol_E=1e-7, project_fn=project_fn)
    res = solve(kernel, jnp.asarray(P0), n_e, config=cfg)

    # Manually reproduce P_pre and P_fin to compare them directly.
    # First, run the loop to get the pre-canonicalization Q, p.
    # We don't have direct access, but we can call solve with max_iter=1 and
    # then iterate manually... easier: just check energy stability across
    # canonicalization by running with one fewer iter and comparing.
    Q = res.Q
    p = res.p

    # P from canonical (Q, p) — what P_fin looks like
    P_canonical = jnp.einsum("...in,...n,...jn->...ij", Q, p, jnp.conj(Q))
    P_canonical = 0.5 * (P_canonical + jnp.conj(jnp.swapaxes(P_canonical, -1, -2)))
    P_canonical_proj = jnp.asarray(project_fn(P_canonical), dtype=Q.dtype)

    # The result.density IS the canonical, projected P_fin.
    # Compare to a hypothetical P_pre — but we don't have direct access to it.
    # Instead, compute the full energy via the canonical result and compare
    # to result.energy.
    print(f"Bilayer SVP +0.05, nk={nk}, nb={kernel.h.shape[-1]}, T={temperature}")
    print(f"Solver result: n_iter={int(res.n_iter)}, converged={bool(res.converged)}")
    print(f"E (returned by solver):  {float(res.energy):.10e}")

    # The density returned IS the canonical one (post-canonicalization).
    # If the canonicalization is gauge-only, then the energy should be the
    # same as if we'd skipped the second build_fock.
    P_diff_from_density = float(jnp.max(jnp.abs(P_canonical_proj - res.density)))
    print(f"||P(Q,p) - result.density||_max = {P_diff_from_density:.3e}")

    # Build F at result.density and compute energy directly
    from jax_hf.fock import build_fock, hf_energy
    Sigma, H, F = build_fock(
        res.density, h=kernel.h, VR=kernel._VR_shifted, refP=kernel.refP,
        HH=kernel.HH, w2d=kernel.w2d,
        include_exchange=True, include_hartree=False,
        exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
        contact_g=kernel.contact_g, contact_Oi=kernel.contact_Oi,
        contact_Oj=kernel.contact_Oj,
        project_fn=None,
    )
    E_recompute = hf_energy(res.density, h=kernel.h, Sigma=Sigma, H=H, weights_b=kernel.weights_b)
    print(f"E (recomputed at density): {float(E_recompute):.10e}")
    print(f"|ΔE| = {float(jnp.abs(E_recompute - res.energy)):.3e}")


if __name__ == "__main__":
    main()
