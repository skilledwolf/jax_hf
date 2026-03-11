"""
Compare CG vs L-BFGS on tetralayer ABAB graphene (32 bands, 64×64 k-grid).

Uses contimod's MultilayerABAB with spinful+valleyful to get 32-band system.
"""
from __future__ import annotations

import time

import contimod as cm
import contimod_graphene.symmetry as cg_symmetry
import jax
import jax.numpy as jnp
import numpy as np
from contimod.meanfield.init_guess import init_to_density_matrix
from contimod.utils.spectrum_fermi import FermiParams
from jax import config

from jax_hf.main import HartreeFockKernel
from jax_hf.symmetry import make_project_fn
from jax_hf.variational_qr import jit_variational_qr_iteration

config.update("jax_enable_x64", True)

# ---- Physical parameters ----
NK = 64
KMAX = 0.14
U = 40.0
TEMPERATURE = 0.5
EPSILON_R = 10.0 / (2.0 * np.pi)
D_GATE = 40.0
INIT_SCALE = 45.0

print(f"Setting up tetralayer ABAB: NK={NK}, U={U}, T={TEMPERATURE}")
H = cm.graphene.MultilayerABAB(valleyful=True, spinful=True, U=float(U))
h_template = H.discretize(nk=NK, kmax=KMAX)
h_template.fermi = FermiParams(T=TEMPERATURE, mu=0.0)
ne_cn = float(h_template.state.compute_density_from_filling(0.5 - 1e-6))

nb = h_template.hs.shape[-1]
print(f"  nb = {nb}  (shape: {h_template.hs.shape})")

weights = np.asarray(h_template.kmesh.weights)

Vq = cm.coulomb.dualgate_coulomb(
    h_template.kmesh.distance_grid,
    epsilon_r=EPSILON_R,
    d_gate=D_GATE,
)
Vq = np.asarray(Vq.magnitude)[..., None, None]

# ---- PM symmetry ----
identity_op = np.asarray(H.identity)
s1, s2, s3 = [np.asarray(H.spin_op(i)) for i in (1, 2, 3)]
v1, v3 = np.asarray(H.valley_op(1)), np.asarray(H.valley_op(3))
U_tr = cg_symmetry.make_time_reversal_U(s2, v1)

G_pm = cg_symmetry.make_pm_group(identity_op, s1, s2, s3, v3)
project_fn_pm = make_project_fn(
    unitary_group=G_pm,
    time_reversal_U=jnp.asarray(U_tr),
    time_reversal_k_convention="flip",
)

# ---- Initial density matrix ----
h_run = h_template.copy()
h_run.fermi = FermiParams(T=TEMPERATURE, mu=0.0)
h_run.compute_chemicalpotential(density=ne_cn)
n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))

P0_op = h_template.get_operator("zero")
P0 = np.asarray(init_to_density_matrix(
    h_run, P0_op, density=None, T=float(h_run.fermi.T), init_kind="auto",
))
print(f"  P0 shape: {P0.shape}, n_e = {n_e:.6f}")

# ---- Kernel ----
kernel = HartreeFockKernel(
    weights=weights,
    hamiltonian=np.asarray(h_template.hs),
    coulomb_q=Vq,
    T=TEMPERATURE,
    include_hartree=False,
    include_exchange=True,
)

# ---- Shared solver params ----
solver_kwargs = dict(
    electrondensity0=n_e,
    init_method="eigh",
    max_iter=100,
    comm_tol=3e-4,
    p_tol=3e-4,
    project_fn=project_fn_pm,
    max_rot=0.6,
    inner_sweeps=4,
    q_sweeps=3,
)

# ---- Run CG ----
print("\n" + "=" * 70)
print("Running CG optimizer...")
print("=" * 70)

runner_cg = jit_variational_qr_iteration(kernel)
t0 = time.perf_counter()
P_cg, F_cg, E_cg, mu_cg, k_cg, hist_cg = runner_cg(
    jnp.asarray(P0), optimizer="cg", **solver_kwargs,
)
jax.block_until_ready((P_cg, E_cg))
dt_cg = time.perf_counter() - t0

k_cg_int = int(k_cg)
dC_cg = float(np.asarray(hist_cg["dC"])[k_cg_int - 1]) if k_cg_int > 0 else float("nan")
conv_cg = "OK" if k_cg_int < 100 else "FAIL"
print(f"  CG:  iters={k_cg_int:3d}  E={float(E_cg):.8f}  dC={dC_cg:.2e}  {conv_cg}  time={dt_cg:.1f}s")

# ---- Run L-BFGS ----
print("\n" + "=" * 70)
print("Running L-BFGS optimizer...")
print("=" * 70)

runner_lbfgs = jit_variational_qr_iteration(kernel)
t0 = time.perf_counter()
P_lb, F_lb, E_lb, mu_lb, k_lb, hist_lb = runner_lbfgs(
    jnp.asarray(P0), optimizer="lbfgs", lbfgs_m=5, **solver_kwargs,
)
jax.block_until_ready((P_lb, E_lb))
dt_lb = time.perf_counter() - t0

k_lb_int = int(k_lb)
dC_lb = float(np.asarray(hist_lb["dC"])[k_lb_int - 1]) if k_lb_int > 0 else float("nan")
conv_lb = "OK" if k_lb_int < 100 else "FAIL"
print(f"  LBFGS: iters={k_lb_int:3d}  E={float(E_lb):.8f}  dC={dC_lb:.2e}  {conv_lb}  time={dt_lb:.1f}s")

# ---- Summary ----
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  System: tetralayer ABAB, {NK}x{NK} k-grid, {nb} bands")
print(f"  CG:     {k_cg_int:3d} iters, {dt_cg:7.1f}s, E={float(E_cg):.8f}")
print(f"  L-BFGS: {k_lb_int:3d} iters, {dt_lb:7.1f}s, E={float(E_lb):.8f}")
if k_cg_int > 0 and k_lb_int > 0:
    print(f"  Iter ratio: L-BFGS/CG = {k_lb_int/k_cg_int:.2f}")
    print(f"  Time ratio: L-BFGS/CG = {dt_lb/dt_cg:.2f}")
    print(f"  Energy diff: {abs(float(E_lb) - float(E_cg)):.2e}")
