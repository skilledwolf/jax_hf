"""
Reproduce notebook execution pattern: single var_runner, all SVP density
points in the same order. The standalone per-point test converges fine,
so the failure must depend on execution order or shared state.
"""
from __future__ import annotations

import time

import contimod as cm
import jax
import jax.numpy as jnp
import numpy as np
from contimod.meanfield.init_guess import init_to_density_matrix
from contimod.utils.spectrum_fermi import FermiParams
from jax import config

from jax_hf.main import HartreeFockKernel
from jax_hf.symmetry import make_project_fn, make_svp_project_fn
from jax_hf.variational import jit_variational_hartreefock_iteration

config.update("jax_enable_x64", True)

NK = 101
KMAX = 0.14
U = 40.0
TEMPERATURE = 0.5
EPSILON_R = 10.0 / (2.0 * np.pi)
D_GATE = 40.0
INIT_SCALE = 45.0
PER_CM = 0.246e-7

H = cm.graphene.MultilayerAB(valleyful=True, spinful=True, U=float(U))
h_template = H.discretize(nk=NK, kmax=KMAX)
h_template.fermi = FermiParams(T=TEMPERATURE, mu=0.0)
ne_cn = float(h_template.state.compute_density_from_filling(0.5 - 1e-6))
weights = np.asarray(h_template.kmesh.weights)

Vq = cm.coulomb.dualgate_coulomb(
    h_template.kmesh.distance_grid,
    epsilon_r=EPSILON_R,
    d_gate=D_GATE,
)
Vq = np.asarray(Vq.magnitude)[..., None, None]

# ---- Seeds ----
identity_op = np.asarray(H.identity)
s1, s2, s3 = [np.asarray(H.spin_op(i)) for i in (1, 2, 3)]
v1, v3 = np.asarray(H.valley_op(1)), np.asarray(H.valley_op(3))
U_tr = v1 @ (1j * s2)

projector_sv = 0.25 * (identity_op + s3) @ (identity_op + v3)
sv_contrast = -projector_sv + 3.0 * (identity_op - projector_sv)

seeds = {
    "PM": h_template.get_operator("zero"),
    "SVP": -float(INIT_SCALE) * h_template.get_operator(sv_contrast),
}

# ---- Projections ----
G_pm = jnp.stack(
    [jnp.asarray(S @ V) for S in [identity_op, s1, s2, s3] for V in [identity_op, v3]],
    axis=0,
)
project_fn_pm = make_project_fn(
    unitary_group=G_pm,
    time_reversal_U=jnp.asarray(U_tr),
    time_reversal_k_convention="flip",
)
project_fn_svp = make_svp_project_fn(
    s3=jnp.asarray(s3),
    v3=jnp.asarray(v3),
    n_orb=4,
    outlier_sv=(+1, +1),
    k_convention="flip",
    k_flip_axes=(0,),
)


def init_density_from_seed(h, init_op):
    P0 = init_to_density_matrix(
        h,
        init_op,
        density=None,
        T=float(h.fermi.T),
        init_kind="auto",
    )
    return np.asarray(P0)


kernel = HartreeFockKernel(
    weights=weights,
    hamiltonian=np.asarray(h_template.hs),
    coulomb_q=Vq,
    T=TEMPERATURE,
    include_hartree=False,
    include_exchange=True,
)
# SINGLE shared var_runner, same as notebook
var_runner = jit_variational_hartreefock_iteration(kernel)

density_points = np.linspace(-0.60, -0.05, 15)
density_points = np.asarray(sorted(density_points, key=lambda n: abs(float(n))))

# ---- First run CN reference (exactly like notebook) ----
h_cn = h_template.copy()
h_cn.fermi = FermiParams(T=TEMPERATURE, mu=0.0)
h_cn.compute_chemicalpotential(density=ne_cn)
n_e_cn = float(h_cn.state.compute_density() / float(h_cn.degeneracy))
P0_cn = init_density_from_seed(h_cn, seeds["PM"])

P_cn, F_cn, E_cn, mu_cn, k_cn, hist_cn = var_runner(
    jnp.asarray(P0_cn),
    electrondensity0=n_e_cn,
    init_method="eigh",
    max_iter=100,
    comm_tol=3e-4,
    p_tol=3e-4,
    project_fn=project_fn_pm,
    max_rot=0.6,
    inner_sweeps=4,
)
print(f"CN ref: iters={int(k_cn)}, E={float(E_cn):.6f}")

# ---- Run PM branch first (like notebook) ----
print("\n=== PM ===")
for n_cm12 in density_points:
    dd = (n_cm12 * 1e12) * (PER_CM**2)
    h_run = h_template.copy()
    h_run.fermi = FermiParams(T=TEMPERATURE, mu=0.0)
    h_run.compute_chemicalpotential(density=float(ne_cn + dd))
    n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))
    P0 = init_density_from_seed(h_run, seeds["PM"])

    P_var, F_var, E_var, mu_var, k_var, hist_var = var_runner(
        jnp.asarray(P0),
        electrondensity0=n_e,
        init_method="eigh",
        max_iter=100,
        comm_tol=3e-4,
        p_tol=3e-4,
        project_fn=project_fn_pm,
        max_rot=0.6,
        inner_sweeps=4,
    )
    jax.block_until_ready((P_var, E_var))
    k_int = int(k_var)
    dC = float(np.asarray(hist_var["dC"])[k_int - 1]) if k_int > 0 else float("nan")
    conv = "OK" if k_int < 100 else "FAIL"
    print(f"  n={n_cm12:+.4f}  it={k_int:3d}  dC={dC:.2e}  {conv}")

# ---- Then SVP (like notebook) ----
print("\n=== SVP ===")
for n_cm12 in density_points:
    dd = (n_cm12 * 1e12) * (PER_CM**2)
    h_run = h_template.copy()
    h_run.fermi = FermiParams(T=TEMPERATURE, mu=0.0)
    h_run.compute_chemicalpotential(density=float(ne_cn + dd))
    n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))
    P0 = init_density_from_seed(h_run, seeds["SVP"])

    t0 = time.perf_counter()
    P_var, F_var, E_var, mu_var, k_var, hist_var = var_runner(
        jnp.asarray(P0),
        electrondensity0=n_e,
        init_method="eigh",
        max_iter=100,
        comm_tol=3e-4,
        p_tol=3e-4,
        project_fn=project_fn_svp,
        max_rot=0.6,
        inner_sweeps=4,
    )
    jax.block_until_ready((P_var, E_var))
    dt = time.perf_counter() - t0
    k_int = int(k_var)
    dC = float(np.asarray(hist_var["dC"])[k_int - 1]) if k_int > 0 else float("nan")
    conv = "OK" if k_int < 100 else "FAIL"
    print(f"  n={n_cm12:+.4f}  it={k_int:3d}  dC={dC:.2e}  {conv}  {dt:.1f}s")
