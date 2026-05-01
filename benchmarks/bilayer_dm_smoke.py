"""Single-point DM smoke test against the bilayer regression setup.

Times one direct-minimization solve on the SVP +0.05 cm^-12 point and
reports iter count, energy, and wall time.  Runs without contimod by
loading the saved reference data.  Used for fast iteration during
performance tuning.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jax_hf
from jax_hf import HartreeFockKernel, SolverConfig

DATA_PATH = Path(__file__).parent.parent / "tests" / "data" / "bilayer_reference.npz"


def main():
    if not DATA_PATH.exists():
        print(f"Reference file not found: {DATA_PATH}")
        return
    ref = np.load(str(DATA_PATH), allow_pickle=False)

    # Try to use contimod for the actual setup
    try:
        import contimod as cm
        from contimod.utils.spectrum_fermi import FermiParams
        from contimod.meanfield.init_guess import init_to_density_matrix
        import contimod_graphene.symmetry as cg_symmetry
        from jax_hf.symmetry import make_project_fn
    except ImportError:
        print("contimod not available; skipping bilayer smoke")
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
    config = SolverConfig(max_iter=200, tol_E=1e-7, project_fn=project_fn)

    print(f"Bilayer SVP +0.05 cm^-12, nk={nk}, nb={kernel.h.shape[-1]}, T={temperature}")

    # Compile + warmup
    print("Compiling...", flush=True)
    t0 = time.perf_counter()
    res = jax_hf.solve(kernel, jnp.asarray(P0), n_e, config=config)
    jax.block_until_ready(res.energy)
    compile_t = time.perf_counter() - t0
    print(f"  compile + first run: {compile_t*1000:.1f}ms, "
          f"n_iter={int(res.n_iter)}, E={float(res.energy):.7e}", flush=True)

    # Warm runs
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        res = jax_hf.solve(kernel, jnp.asarray(P0), n_e, config=config)
        jax.block_until_ready(res.energy)
        times.append(time.perf_counter() - t0)
    mean_t = np.mean(times)
    min_t = np.min(times)
    n_iter = int(res.n_iter)
    print(f"  warm runs: mean={mean_t*1000:.1f}ms min={min_t*1000:.1f}ms "
          f"n_iter={n_iter} per_iter={min_t/n_iter*1000:.3f}ms",
          flush=True)
    print(f"  energy: {float(res.energy):.7e}, converged={bool(res.converged)}",
          flush=True)


if __name__ == "__main__":
    main()
