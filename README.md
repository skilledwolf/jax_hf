## jax_hf — JAX Hartree–Fock on k‑grids

[![PyPI](https://img.shields.io/pypi/v/jax-hf.svg)](https://pypi.org/project/jax-hf/)
[![Python](https://img.shields.io/pypi/pyversions/jax-hf.svg)](https://pypi.org/project/jax-hf/)
[![Wheel](https://img.shields.io/pypi/wheel/jax-hf.svg)](https://pypi.org/project/jax-hf/#files)
[![License](https://img.shields.io/pypi/l/jax-hf.svg)](LICENSE)
[![Build](https://github.com/skilledwolf/jax_hf/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/skilledwolf/jax_hf/actions/workflows/build-and-test.yml)
[![Release](https://github.com/skilledwolf/jax_hf/actions/workflows/release.yml/badge.svg)](https://github.com/skilledwolf/jax_hf/actions/workflows/release.yml)

jax_hf provides a JAX implementation of a Hartree–Fock self‑consistent field
(SCF) loop on uniform 2D k‑grids, with optional JIT compilation.

- FFT‑based exchange in k‑space
- Dense Hermitian diagonalization (JAX eigh)
- DIIS/EDIIS‑style mixing (robust convergence)
- NumPy/JAX‑friendly API, easy to integrate with other JAX code

Project links:
- PyPI: https://pypi.org/project/jax-hf/
- Source: https://github.com/skilledwolf/jax_hf

## Installation

Users (PyPI):
```bash
pip install jax-hf
```

Note: jax_hf depends on JAX. For CPU‑only installs, pip will usually pull a
working wheel automatically. For GPU, follow JAX’s official install guide to
select the correct extras/wheels for your CUDA/cuDNN stack. 

Developers (editable install):
```bash
git clone https://github.com/skilledwolf/jax_hf.git
cd jax_hf
pip install -e .
```

## Quick start

```python
import numpy as np
import jax.numpy as jnp
import jax_hf

# Grid and shapes
nk = 128; d = 2
weights = np.ones((nk, nk)) * ((2/nk)*(2/nk) / (2*np.pi)**2)  # scalar mesh measure
H = np.zeros((nk, nk, d, d), dtype=np.complex128)
K = np.linspace(-1.0, 1.0, nk)
Vq = (1.0 / np.sqrt((K[:, None]**2 + K[None, :]**2) + 0.1)).astype(np.complex128)[..., None, None]
P0 = np.zeros_like(H)

# Target electron density (half‑filling)
ne_target = 0.5 * d * weights.sum()

# Build HF kernel (JAX arrays inside)
kernel = jax_hf.HartreeFockKernel(
    weights,                 # (nk, nk)
    H,                       # (nk, nk, d, d)
    Vq,                      # (nk, nk, 1, 1) or (nk, nk, d, d)
    T=0.5,                   # temperature
)

# JIT‑compile the SCF iteration function (optional but recommended)
hf_iter = jax_hf.jit_hartreefock_iteration(kernel)

P_fin, F_fin, E_fin, mu_fin, n_iter, history = hf_iter(
    P0, float(ne_target),
    max_iter=50, comm_tol=1e-3, diis_size=6, log_every=None,
)
print("iters:", int(n_iter), "mu:", float(mu_fin), "E:", float(E_fin))
```

## Prepared-problem API

If you want a stable public surface above the low-level JIT wrappers, use the
prepared-problem API:

```python
import jax_hf
import jax.numpy as jnp

problem = jax_hf.HFProblem(
    weights=weights,
    hamiltonian=H,
    coulomb_q=Vq,
    T=0.5,
)

result = jax_hf.solve(
    problem,
    solver="scf",
    seed=jax_hf.DensityMatrixSeed(jnp.zeros_like(H)),
    n_electrons_per_degeneracy=float(ne_target),
    config=jax_hf.SCFRunConfig(max_iter=50, comm_tol=1e-3, diis_size=6),
)
print(result.fine.n_iter, float(result.mu))

qr = jax_hf.solve(
    problem,
    solver="qr",
    seed=jax_hf.DensityMatrixSeed(jnp.zeros_like(H)),
    n_electrons_per_degeneracy=float(ne_target),
    config=jax_hf.QRRunConfig(max_iter=80, comm_tol=1e-5, p_tol=1e-3),
)
```

`solve(...)` returns a standardized `SolveResult` with `fine` and optional
`coarse` stage results. The convenience wrappers `run_scf(...)`,
`run_variational_qr(...)`, and `run_variational_rtr(...)` call the same path.
The legacy `electrondensity0` argument is still accepted as an alias for
`n_electrons_per_degeneracy`, but new code should prefer the latter.
Likewise, the legacy `P0` / `params0` / `nk_coarse` inputs still work, but the
preferred interface is `seed=...` plus `continuation=ContinuationConfig(...)`.

## Coarse-to-fine continuation (`nk_coarse`)

For large k-grids it can be helpful to converge on a smaller coarse grid and use
the resulting mean-field self-energy as a seed for a fine-grid run. jax_hf
includes a small helper that resamples (H, Vq, P) between uniform centered
grids:

```python
import jax_hf

out = jax_hf.solve(
    problem,
    solver="scf",
    seed=jax_hf.DensityMatrixSeed(P0),
    n_electrons_per_degeneracy=float(ne_target),
    continuation=jax_hf.ContinuationConfig(
        nk_coarse=64,
        coarse_config=jax_hf.SCFRunConfig(max_iter=80, comm_tol=1e-3, diis_size=6),
        fine_config=jax_hf.SCFRunConfig(max_iter=50, comm_tol=1e-4, diis_size=6),
    ),
)
print("coarse iters:", int(out.coarse.n_iter) if out.coarse else None)
print("fine iters:", int(out.fine.n_iter))
```

## API

```python
class HartreeFockKernel:
    def __init__(self, weights, hamiltonian, coulomb_q, T: float):
        ...

def hartreefock_iteration(
    P0, electrondensity0, hf_step: HartreeFockKernel,
    *, max_iter=100, comm_tol=5e-3, diis_size=4, log_every: int | None = 1,
):
    """Runs SCF and returns (P_fin, F_fin, E_fin, mu_fin, n_iter, history)."""

def jit_hartreefock_iteration(hf_step: HartreeFockKernel):
    """Returns a jitted version of hartreefock_iteration with static args."""
```

- shapes: `weights` is (nk, nk), `hamiltonian` is (nk, nk, d, d),
  `coulomb_q` is (nk, nk, 1, 1) or (nk, nk, d, d), `P0` matches (nk, nk, d, d).
- returns: converged density `P_fin`, mean‑field `F_fin`, total energy,
  chemical potential `mu_fin`, iteration count, and a small `history` dict with
  energy/commutator traces.

## Versioning

Versions are derived from git tags using setuptools_scm. Tags like `v1.2.3`
produce version `1.2.3`; non‑tag builds produce development versions.

## License

BSD 2‑Clause — see `LICENSE`.

Third‑party notices: see `THIRD_PARTY_NOTICES.md`.

**Author: Dr. Tobias Wolf**
