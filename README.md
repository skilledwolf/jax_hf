## jax_hf — JAX Hartree–Fock on k‑grids

[![PyPI](https://img.shields.io/pypi/v/jax-hf.svg)](https://pypi.org/project/jax-hf/)
[![Python](https://img.shields.io/pypi/pyversions/jax-hf.svg)](https://pypi.org/project/jax-hf/)
[![Wheel](https://img.shields.io/pypi/wheel/jax-hf.svg)](https://pypi.org/project/jax-hf/#files)
[![License](https://img.shields.io/pypi/l/jax-hf.svg)](LICENSE)
[![Build](https://github.com/skilledwolf/jax_hf/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/skilledwolf/jax_hf/actions/workflows/build-and-test.yml)
[![Release](https://github.com/skilledwolf/jax_hf/actions/workflows/release.yml/badge.svg)](https://github.com/skilledwolf/jax_hf/actions/workflows/release.yml)

jax_hf provides two JAX-jitted solvers for the Hartree–Fock free-energy
minimisation problem on 2D k-meshes:

* **Direct minimisation** (primary): preconditioned Riemannian CG on
  Stiefel × capped simplex, eigen-free inner loop, Cayley retraction,
  one Fock build per iteration.
* **Reference SCF** (baseline / fallback): standard Roothaan iteration
  with linear mixing.

Exchange and Hartree can both be included, and the exchange kernel may
be layer-resolved.  See `examples/` for density-scan scripts on a
bilayer graphene model.

> **v2.0.0 note:** This release is a clean-slate rewrite.  The
> entire public API has changed relative to the deprecated v1.x line
> (which was already a skeleton in v1.1.0).  See
> [`MIGRATION.md`](MIGRATION.md) for the migration guide.

### Install

```bash
pip install jax-hf
```

### Minimal example

```python
import jax.numpy as jnp
import jax_hf

# Build a HartreeFockKernel: precomputes the FFT of the interaction kernel,
# the Hartree matrix, etc., ready for JIT.
kernel = jax_hf.HartreeFockKernel(
    weights=weights,          # (nk1, nk2) k-point weights
    hamiltonian=hamiltonian,  # (nk1, nk2, nb, nb) single-particle Hamiltonian
    coulomb_q=coulomb_q,      # (nk1, nk2, 1, 1) scalar or (nk1, nk2, nb, nb) layer-resolved
    T=0.1,
    include_hartree=False,    # set True for Hartree; also pass reference_density + hartree_matrix
    include_exchange=True,
)

# Solve (direct minimisation, default)
result = jax_hf.solve(kernel, P0=jnp.zeros_like(hamiltonian), n_electrons=N)
print(result.energy, result.converged, result.n_iter)
# result.density, result.fock, result.Q, result.p, result.mu, result.history

# Or use SCF as a fallback baseline
result_scf = jax_hf.solve_scf(kernel, P0=jnp.zeros_like(hamiltonian), n_electrons=N)
```

### Config

Both solvers take a Config dataclass with sensible defaults:

```python
jax_hf.SolverConfig(max_iter=200, tol_E=1e-7, max_step=0.6, project_fn=None, ...)
jax_hf.SCFConfig(max_iter=200, mixing=0.3, density_tol=1e-7, comm_tol=1e-6, ...)
```

`project_fn` lets you enforce symmetry constraints (spin, valley, time
reversal, spatial) on the density and Fock at every iteration.  See
`jax_hf.symmetry.make_project_fn`.

### Public API

| Name | Purpose |
|---|---|
| `HartreeFockKernel` | Problem + precomputed arrays |
| `solve` (alias `solve_direct_minimization`), `SolverConfig`, `SolveResult` | Primary solver |
| `solve_scf`, `SCFConfig`, `SCFResult` | Reference SCF solver |
| `build_fock`, `hf_energy`, `free_energy`, `occupation_entropy` | HF objective building blocks |

Lower-level modules (`jax_hf.utils`, `jax_hf.symmetry`, `jax_hf.linalg`,
`jax_hf.fock`) expose the individual pieces for users who need them.

### Examples

* `examples/multilayer_graphene_density_scan.py` — PM/SVP density scan
  for bilayer graphene, direct minimisation, Fock only
* `examples/multilayer_graphene_density_scan_extended.py` — adds
  spin-polarised and "SVP flipped" branches (4 total)
* `examples/multilayer_graphene_density_scan_hartree.py` — same four
  branches with layer-resolved Coulomb and Hartree included
* `examples/multilayer_graphene_reference_scf_scan.py` — SCF baseline
  scan for side-by-side comparison

### Running tests

```bash
pytest tests/
```

The bilayer regression tests (`tests/test_bilayer_regression.py`) require
`contimod` and `contimod_graphene` and will be skipped otherwise.
