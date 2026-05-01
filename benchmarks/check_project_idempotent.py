"""Check whether project_fn(F[P]) == F[P] when P is already projected.

If yes, the project_fn call inside build_fock is redundant when called from
the DM solver body (which already projects P beforehand).  This would let
us drop one ~8-element symmetry-group sweep per outer iter.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jax_hf import HartreeFockKernel, SolverConfig, solve
from jax_hf.fock import build_fock

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
        from jax_hf.symmetry import make_project_fn
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
    s1, s2, s3 = [np.asarray(model.spin_op(i)) for i in (1, 2, 3)]
    v1, v3 = np.asarray(model.valley_op(1)), np.asarray(model.valley_op(3))
    U_tr = cg_symmetry.make_time_reversal_U(s2, v1)

    G_pm = cg_symmetry.make_pm_group(identity_op, s1, s2, s3, v3)
    project_fn_pm = make_project_fn(
        unitary_group=G_pm,
        time_reversal_U=jnp.asarray(U_tr),
        time_reversal_k_convention="flip",
    )

    projector_sv = 0.25 * (identity_op + s3) @ (identity_op + v3)
    sv_contrast = -projector_sv + 3.0 * (identity_op - projector_sv)
    project_fn_svp = cg_symmetry.make_svp_project_fn(
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

    for branch_name, project_fn, seed_op in [
        ("PM",  project_fn_pm,  h_template.get_operator("zero")),
        ("SVP", project_fn_svp, -init_scale * h_template.get_operator(sv_contrast)),
    ]:
        P0 = init_to_density_matrix(h_run, seed_op, density=None,
                                    T=temperature, init_kind="auto")

        kernel = HartreeFockKernel(
            weights=weights, hamiltonian=np.asarray(h_template.hs),
            coulomb_q=Vq, T=temperature,
            include_hartree=False, include_exchange=True,
        )
        cfg = SolverConfig(max_iter=200, tol_E=1e-7, project_fn=project_fn)
        res = solve(kernel, jnp.asarray(P0), n_e, config=cfg)

        # Now check: is build_fock(P) symmetric without internal projection?
        P_sym = jnp.asarray(project_fn(res.density), dtype=kernel.h.dtype)
        # Confirm P_sym is symmetric (project is idempotent)
        P_sym2 = project_fn(P_sym)
        P_sym_diff = float(jnp.max(jnp.abs(P_sym - P_sym2)))

        # Build F without internal project
        _, _, F_no_proj = build_fock(
            P_sym, h=kernel.h, VR=kernel._VR_shifted, refP=kernel.refP,
            HH=kernel.HH, w2d=kernel.w2d,
            include_exchange=True, include_hartree=False,
            exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
            contact_g=kernel.contact_g, contact_Oi=kernel.contact_Oi,
            contact_Oj=kernel.contact_Oj,
            project_fn=None,  # ← no internal projection
        )
        # Build F with internal project
        _, _, F_with_proj = build_fock(
            P_sym, h=kernel.h, VR=kernel._VR_shifted, refP=kernel.refP,
            HH=kernel.HH, w2d=kernel.w2d,
            include_exchange=True, include_hartree=False,
            exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
            contact_g=kernel.contact_g, contact_Oi=kernel.contact_Oi,
            contact_Oj=kernel.contact_Oj,
            project_fn=project_fn,
        )
        F_diff = float(jnp.max(jnp.abs(F_no_proj - F_with_proj)))
        F_rel = F_diff / float(jnp.max(jnp.abs(F_with_proj)) + 1e-30)

        # Also: is F_no_proj already symmetric?
        F_proj_of_no_proj = jnp.asarray(project_fn(F_no_proj), dtype=F_no_proj.dtype)
        F_self_diff = float(jnp.max(jnp.abs(F_no_proj - F_proj_of_no_proj)))
        F_self_rel = F_self_diff / float(jnp.max(jnp.abs(F_no_proj)) + 1e-30)

        print(f"\n{branch_name} branch:")
        print(f"  P projection idempotent (||P - proj(P)||_max): {P_sym_diff:.3e}")
        print(f"  F(no internal proj) vs F(with internal proj):  diff = {F_diff:.3e}, rel = {F_rel:.3e}")
        print(f"  F(no internal proj) already symmetric:          ||F - proj(F)||_max = {F_self_diff:.3e}, rel = {F_self_rel:.3e}")


if __name__ == "__main__":
    main()
