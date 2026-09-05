"""Optional independent JAX/C++ solver parity check (requires cpp_hf)."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json
import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
import jax_hf as jhf
import cpp_hf as chf
rows = []
for seed in range(4):
    rng = np.random.default_rng(414 + seed)
    shape=(2,2,4,4)
    A=rng.normal(size=shape)+1j*rng.normal(size=shape)
    h=.5*(A+A.conj().swapaxes(-1,-2))
    P=np.broadcast_to(.4*np.eye(4),shape).astype(complex).copy()
    P[0,0,0,1]=.1j; P[0,0,1,0]=-.1j
    w=np.array([[.1,.2],[.3,.4]])
    V=np.full((2,2,1,1),.3+seed*.1)
    ref=np.broadcast_to(.5*np.eye(4),shape).astype(complex).copy()
    HH=(.4+seed*3)*(np.eye(4)+.2*np.ones((4,4)))
    kw=dict(weights=w,hamiltonian=h,coulomb_q=V,T=.3,include_hartree=True,
            reference_density=ref,hartree_matrix=HH)
    jk=jhf.HartreeFockKernel(**kw); ck=chf.HartreeFockKernel(**kw)
    for method in ('rdiis','anderson'):
        cfg=dict(acceleration=method,diis_size=8,mixing=1.,max_iter=150,
                 density_tol=1e-8,comm_tol=1e-8)
        a=jhf.solve_scf(jk,P,1.6,config=jhf.SCFConfig(**cfg))
        b=chf.solve_scf(ck,P,1.6,config=chf.SCFConfig(**cfg))
        row=dict(seed=seed,method=method,jax_converged=a.converged,cpp_converged=b.converged,
                 jax_iterations=a.iterations,cpp_iterations=b.iterations,
                 energy_error=float(abs(a.energy-b.energy)),
                 density_max_error=float(np.max(abs(np.asarray(a.density_matrix)-b.density_matrix))))
        rows.append(row)
        print(json.dumps(row), flush=True)
assert all(r['jax_converged'] and r['cpp_converged'] and r['energy_error']<1e-9
           and r['density_max_error']<1e-7 for r in rows)
