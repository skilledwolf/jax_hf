"""
Compare QR (CG + L-BFGS) vs RTR on tetralayer ABAB graphene (32 bands, 64×64 k-grid).

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
from jax_hf.variational_rtr import jit_variational_rtr_iteration

config.update("jax_enable_x64", True)

# ---- Physical parameters ----
NK = 64
KMAX = 0.14
U = 40.0
TEMPERATURE = 0.5
EPSILON_R = 10.0 / (2.0 * np.pi)
D_GATE = 40.0

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

P0_jax = jnp.asarray(P0)

# ---- Shared convergence criteria ----
COMM_TOL = 3e-4
P_TOL = 3e-4
MAX_ITER = 25

# ---- Run QR-CG ----
print("\n" + "=" * 70)
print("Running QR-CG optimizer...")
print("=" * 70)

runner_cg = jit_variational_qr_iteration(kernel)
t0 = time.perf_counter()
P_cg, F_cg, E_cg, mu_cg, k_cg, hist_cg = runner_cg(
    P0_jax,
    electrondensity0=n_e,
    optimizer="cg",
    init_method="eigh",
    max_iter=MAX_ITER,
    comm_tol=COMM_TOL,
    p_tol=P_TOL,
    project_fn=project_fn_pm,
    max_rot=0.6,
    inner_sweeps=4,
    q_sweeps=3,
)
jax.block_until_ready((P_cg, E_cg))
dt_cg = time.perf_counter() - t0

k_cg_int = int(k_cg)
dC_cg = float(np.asarray(hist_cg["dC"])[max(k_cg_int - 1, 0)])
print(f"  QR-CG:  iters={k_cg_int:3d}  E={float(E_cg):.8f}  dC={dC_cg:.2e}  time={dt_cg:.1f}s")

# ---- Run QR-LBFGS ----
print("\n" + "=" * 70)
print("Running QR-LBFGS optimizer...")
print("=" * 70)

runner_lb = jit_variational_qr_iteration(kernel)
t0 = time.perf_counter()
P_lb, F_lb, E_lb, mu_lb, k_lb, hist_lb = runner_lb(
    P0_jax,
    electrondensity0=n_e,
    optimizer="lbfgs",
    lbfgs_m=5,
    init_method="eigh",
    max_iter=MAX_ITER,
    comm_tol=COMM_TOL,
    p_tol=P_TOL,
    project_fn=project_fn_pm,
    max_rot=0.6,
    inner_sweeps=4,
    q_sweeps=3,
)
jax.block_until_ready((P_lb, E_lb))
dt_lb = time.perf_counter() - t0

k_lb_int = int(k_lb)
dC_lb = float(np.asarray(hist_lb["dC"])[max(k_lb_int - 1, 0)])
print(f"  QR-LBFGS: iters={k_lb_int:3d}  E={float(E_lb):.8f}  dC={dC_lb:.2e}  time={dt_lb:.1f}s")

# ---- Run RTR (same tol) ----
print("\n" + "=" * 70)
print("Running RTR optimizer (same tol)...")
print("=" * 70)

runner_rtr = jit_variational_rtr_iteration(kernel)
t0 = time.perf_counter()
P_rtr, F_rtr, E_rtr, mu_rtr, k_rtr, hist_rtr = runner_rtr(
    P0_jax,
    electrondensity0=n_e,
    init_method="eigh",
    max_iter=MAX_ITER,
    comm_tol=COMM_TOL,
    p_tol=P_TOL,
    project_fn=project_fn_pm,
    max_rot=0.6,
    max_cg_iter=15,
    cg_tol=1e-2,
)
jax.block_until_ready((P_rtr, E_rtr))
dt_rtr = time.perf_counter() - t0

k_rtr_int = int(k_rtr)
dC_rtr = float(np.asarray(hist_rtr["dC"])[max(k_rtr_int - 1, 0)])
print(f"  RTR:    iters={k_rtr_int:3d}  E={float(E_rtr):.8f}  dC={dC_rtr:.2e}  time={dt_rtr:.1f}s")

# ---- Run RTR (tight tol, more iters) ----
print("\n" + "=" * 70)
print("Running RTR optimizer (tight tol, 200 iters)...")
print("=" * 70)

runner_rtr2 = jit_variational_rtr_iteration(kernel)
t0 = time.perf_counter()
P_rtr2, F_rtr2, E_rtr2, mu_rtr2, k_rtr2, hist_rtr2 = runner_rtr2(
    P0_jax,
    electrondensity0=n_e,
    init_method="eigh",
    max_iter=25,
    comm_tol=1e-6,
    p_tol=1e-6,
    e_tol=1e-8,
    project_fn=project_fn_pm,
    max_rot=0.6,
    max_cg_iter=25,
    cg_tol=1e-4,
)
jax.block_until_ready((P_rtr2, E_rtr2))
dt_rtr2 = time.perf_counter() - t0

k_rtr2_int = int(k_rtr2)
dC_rtr2 = float(np.asarray(hist_rtr2["dC"])[max(k_rtr2_int - 1, 0)])
print(f"  RTR-tight: iters={k_rtr2_int:3d}  E={float(E_rtr2):.8f}  dC={dC_rtr2:.2e}  time={dt_rtr2:.1f}s")

# ---- Summary ----
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  System: tetralayer ABAB, {NK}x{NK} k-grid, {nb} bands")
print(f"  Convergence: comm_tol={COMM_TOL:.0e}, p_tol={P_TOL:.0e}")
print()

results = [
    ("QR-CG",    k_cg_int,  dt_cg,  float(E_cg),  dC_cg),
    ("QR-LBFGS", k_lb_int,  dt_lb,  float(E_lb),  dC_lb),
    ("RTR",      k_rtr_int, dt_rtr, float(E_rtr), dC_rtr),
    ("RTR-tight", k_rtr2_int, dt_rtr2, float(E_rtr2), dC_rtr2),
]

print(f"  {'Solver':<12s} {'Iters':>5s} {'Time(s)':>8s} {'Energy':>14s} {'dC_final':>10s} {'Conv':>5s}")
print(f"  {'-'*12} {'-'*5} {'-'*8} {'-'*14} {'-'*10} {'-'*5}")
for name, k, dt, E, dC in results:
    conv = "OK" if k < MAX_ITER else "FAIL"
    print(f"  {name:<12s} {k:5d} {dt:8.1f} {E:14.8f} {dC:10.2e} {conv:>5s}")

ref_E = float(E_cg)
print()
for name, k, dt, E, dC in results:
    print(f"  {name:<12s} dE_vs_CG = {E - ref_E:+.2e}")

# ---- Convergence trace (first 30 iters) ----
print("\n" + "=" * 70)
print("CONVERGENCE TRACE (first 30 iters)")
print("=" * 70)
all_traces = [
    ("QR-CG",     hist_cg,   k_cg_int),
    ("QR-LBFGS",  hist_lb,   k_lb_int),
    ("RTR",       hist_rtr,  k_rtr_int),
    ("RTR-tight", hist_rtr2, k_rtr2_int),
]
header = "  " + f"{'iter':>4s}" + "".join(f"  {name+' dC':>12s}" for name, _, _ in all_traces)
print(header)
max_trace = max(k for _, _, k in all_traces)
for i in range(min(40, max_trace)):
    vals = []
    for _, hist, k_int in all_traces:
        dC_i = float(np.asarray(hist["dC"])[i]) if i < k_int else float("nan")
        vals.append(dC_i)
    if all(v != v for v in vals):
        break
    row = f"  {i:4d}" + "".join(f"  {v:12.2e}" for v in vals)
    print(row)
