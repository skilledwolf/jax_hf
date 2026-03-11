"""Quick benchmark: RTR vs QR on the bilayer CN reference point."""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

from jax_hf.main import HartreeFockKernel
from jax_hf.variational_rtr import jit_variational_rtr_iteration
from jax_hf.variational_qr import jit_variational_qr_iteration

import contimod as cm
from contimod.meanfield.init_guess import init_to_density_matrix
from contimod.meanfield.symmetry import build_graphene_projections
from contimod.utils.spectrum_fermi import FermiParams

# --- Build model ---
NK = 81
KMAX = 0.14
U = 40.0
TEMPERATURE = 0.5
EPSILON_R = 10.0 / (2.0 * np.pi)
D_GATE = 40.0

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

_project_fns = build_graphene_projections(H)
project_fn_pm = _project_fns["PM"]

kernel = HartreeFockKernel(
    weights=weights,
    hamiltonian=np.asarray(h_template.hs),
    coulomb_q=Vq,
    T=TEMPERATURE,
    include_hartree=False,
    include_exchange=True,
)

# --- Prepare CN reference ---
h_cn = h_template.copy()
h_cn.fermi = FermiParams(T=TEMPERATURE, mu=0.0)
h_cn.compute_chemicalpotential(density=ne_cn)
n_e_cn = float(h_cn.state.compute_density() / float(h_cn.degeneracy))

seeds_zero = H.discretize(nk=NK, kmax=KMAX)
seeds_zero.fermi = FermiParams(T=TEMPERATURE, mu=0.0)
P0_cn = np.asarray(init_to_density_matrix(h_cn, seeds_zero.get_operator("zero"),
                                           density=None, T=TEMPERATURE, init_kind="auto"))

MAX_ITER = 100
COMM_TOL = 1e-4
OCC_TOL = 2e-3

# --- RTR with different max_cg_iter ---
rtr_runner_cache = {}

def run_rtr(max_cg_iter, n_trials=3):
    if max_cg_iter not in rtr_runner_cache:
        rtr_runner_cache[max_cg_iter] = jit_variational_rtr_iteration(kernel)
    rtr_runner = rtr_runner_cache[max_cg_iter]

    print(f"\n--- RTR max_cg_iter={max_cg_iter} ---")
    for trial in range(n_trials):
        t0 = time.perf_counter()
        P, F, E, mu, k, hist = rtr_runner(
            jnp.asarray(P0_cn),
            electrondensity0=n_e_cn,
            max_iter=MAX_ITER,
            max_cg_iter=max_cg_iter,
            comm_tol=COMM_TOL,
            p_tol=OCC_TOL,
            project_fn=project_fn_pm,
        )
        jax.block_until_ready((P, E))
        dt = time.perf_counter() - t0
        n_it = int(k)
        dC = float(np.asarray(hist["dC"])[n_it - 1]) if n_it > 0 else float("nan")
        dP = float(np.asarray(hist["dP"])[n_it - 1]) if n_it > 0 else float("nan")
        print(f"  trial {trial}: E={float(E):.8f}  iters={n_it:3d}  dC={dC:.2e}  dP={dP:.2e}  time={dt:.2f}s")

        if trial == n_trials - 1:
            print(f"  Convergence trace:")
            for i in range(min(n_it, 20)):
                e = float(np.asarray(hist["E"])[i])
                dc = float(np.asarray(hist["dC"])[i])
                dp = float(np.asarray(hist["dP"])[i])
                de = float(np.asarray(hist["dE"])[i])
                print(f"    iter {i+1:3d}: E={e:.8f}  dC={dc:.2e}  dP={dp:.2e}  dE={de:.2e}")

for cg in [15, 10, 5, 3]:
    run_rtr(cg)

# --- QR ---
print("\n--- QR ---")
qr_runner = jit_variational_qr_iteration(kernel)
for trial in range(3):
    t0 = time.perf_counter()
    P, F, E, mu, k, hist = qr_runner(
        jnp.asarray(P0_cn),
        electrondensity0=n_e_cn,
        max_iter=MAX_ITER,
        comm_tol=COMM_TOL,
        p_tol=OCC_TOL,
        project_fn=project_fn_pm,
    )
    jax.block_until_ready((P, E))
    dt = time.perf_counter() - t0
    dC = float(np.asarray(hist["dC"])[int(k) - 1]) if int(k) > 0 else float("nan")
    dP = float(np.asarray(hist["dP"])[int(k) - 1]) if int(k) > 0 else float("nan")
    print(f"  trial {trial}: E={float(E):.8f}  iters={int(k):3d}  dC={dC:.2e}  dP={dP:.2e}  time={dt:.2f}s")
