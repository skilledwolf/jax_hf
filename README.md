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
  Stiefel × capped simplex with Cayley retraction and true-free-energy
  backtracking.  An optional **trust-region Newton** variant
  (`optimizer="newton"`) usually needs fewer outer iterations and falls
  back to damped fixed-point steps near occupation degeneracies.
  **Deflation** (`solve_deflated`) finds *distinct* self-consistent
  solutions on non-convex landscapes.
* **Self-consistent field**: safeguarded full-density-residual **RDIIS** and
  **Anderson**, alongside linear mixing, commutator DIIS and ODA.
  `solve_minimum` adds local stability checks and bounded saddle recovery.

Exchange and Hartree can both be included, and the exchange kernel may
be layer-resolved.  See `examples/` for density-scan scripts on a
bilayer graphene model.

> **v2.0.0 note:** This release is a clean-slate rewrite.  The
> entire public API has changed relative to the deprecated v1.x line
> (which was already a skeleton in v1.1.0).  See
> [`MIGRATION.md`](MIGRATION.md) for the migration guide.

### Install

Python 3.11 or newer is required. The supported runtime dependency floors are
JAX 0.5.3, NumPy 1.26 and SciPy 1.12; CI tests the JAX/NumPy minimum combination on Python 3.11
and current dependency releases on Python 3.11–3.14.

```bash
pip install jax-hf
```

The default JAX installation is CPU-capable. Follow JAX's platform-specific
installation instructions when you need a CUDA, ROCm, or TPU build.

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
jax_hf.SolverConfig(max_iter=200, tol_E=1e-7, optimizer="cg")   # or optimizer="newton"
jax_hf.SCFConfig(max_iter=200, acceleration="rdiis", diis_size=8)
```

`project_fn` lets you enforce symmetry constraints (spin, valley, time
reversal, spatial) on the density and Fock at every iteration.  See
`jax_hf.symmetry.make_project_fn`.

### Newton, deflation, and SCF acceleration

```python
from jax_hf import SolverConfig, SCFConfig, solve, solve_scf, solve_deflated

# Trust-region Newton: often fewer outer iterations on stiff problems.
# (A second-order method — enable x64 / build the kernel in float64.)
r = solve(kernel, P0, N, config=SolverConfig(optimizer="newton", tol_grad=1e-6))

# Deflation: find *distinct* HF solutions (e.g. competing broken-symmetry
# phases).  Returns them sorted by energy; `best` is the ground-state candidate.
res = solve_deflated(kernel, P0, N, n_solutions=4)
print(res.energies, res.best.energy)

# Safeguarded full-residual DIIS includes the occupation residual.
r = solve_scf(kernel, P0, N, config=SCFConfig(acceleration="rdiis", diis_size=8))
print(r.history["fock_builds"])
```

Enable double precision **before constructing the kernel** for tight Newton or
stability tolerances:

```python
import jax
jax.config.update("jax_enable_x64", True)
# Construct a float64/complex128 kernel, then:
minimum = jax_hf.solve_minimum(kernel, P0, N)
print(minimum.converged, minimum.solution.energy, minimum.escapes)
```

Small fixed-point residuals do not rule out saddles. `solve_minimum` checks
curvature in the supplied basis and accepts only feasible, free-energy-lowering
escape steps. Distinct stable phases still require multiple seeds or
continuation and comparison of their **free energies**. The standalone diagnostic
is `jax_hf.stability_analysis(kernel, P, N)`. These outer routines use NumPy/SciPy
host orchestration with JAX Fock kernels; they are not themselves JIT-compatible.
They support affine, variational, JAX-differentiable `fock_build_fn` callbacks.

Saddle recovery first applies a fixed-particle Fermi map to the perturbed Fock
field, retaining the unstable tangent while respecting saturated occupation
bounds. If these trials do not give significant free-energy descent, the solver
tries projected-density perturbations. Both constructions evaluate and retain
the actual kernel-dtype density; unresolved descent remains a reported failure.

RDIIS/Anderson remain JAX-native and preserve custom Fock support. Their history
stores two density-sized arrays per history slot, plus working buffers; choose
the history size with the problem's memory cost in mind. Existing CG and linear
SCF defaults remain unchanged. The [port validation note](notes/solver_study_port_2026-09-05.md)
documents the kernel/Newton corrections, parity checks, and scope.

### Orbital-resolved superlattices

`jax_hf.superlattice.build_extended_layout(..., coulomb_V_orbital=V_orbital)`
tabulates the full `V_orbital(qmag)` tensor of shape
`(*qmag.shape, dim_orb, dim_orb)`. The returned layout carries it into
`make_superlattice_fock_fn` and `make_superlattice_build_fock_fn`; an explicit
`V_lag_fft_orbital=` override is also accepted by both factories. The scalar
path and its defaults are unchanged. Radial density-density interactions
should be real and symmetric in their orbital indices.

The orbital path streams one `(delta_G, alpha, beta)` channel through a 2D
FFT buffer, retaining every inter-orbital and inter-G coherence. Working FFT
memory scales as `N_ext_x * N_ext_y`; the shared orbital kernel table scales
as `N_ext_x * N_ext_y * dim_orb**2`, with no `n_delta` multiplier. Density,
Fock, solver history and eigensystem storage remain additional costs. This
memory bound does not promise faster execution than the C++ backend.

The custom build-Fock adapter includes orbital Hartree, reference subtraction
and contact terms and supports JAX forward-mode response (`jax.jvp`), including
at zero density. Supply it through `fock_build_fn=` to SCF, Newton or stability
analysis. Its orbital Hartree matrix is supplied separately as
`HH_GG_orbital`; keeping or dropping finite q=0 blocks is the caller's choice.

### Public API

| Name | Purpose |
|---|---|
| `HartreeFockKernel` | Problem + precomputed arrays |
| `solve` (alias `solve_direct_minimization`), `SolverConfig`, `SolveResult` | Primary solver (CG + trust-region Newton) |
| `solve_deflated`, `DeflatedResult` | Find distinct HF solutions (deflated Newton) |
| `solve_scf`, `SCFConfig`, `SCFResult` | SCF (RDIIS / Anderson / linear / DIIS / ODA) |
| `solve_minimum`, `MinimumResult` | Local stability checks and bounded saddle recovery |
| `stability_analysis`, `StabilityResult` | Fixed-particle finite-temperature density Hessian diagnostic |
| `build_fock`, `hf_energy`, `free_energy`, `occupation_entropy` | HF objective building blocks |
| `solve_continuation`, `ContinuationResult`, `resample_kgrid` | Coarse → fine multigrid driver + k-grid resampler |

Lower-level modules (`jax_hf.utils`, `jax_hf.symmetry`, `jax_hf.linalg`,
`jax_hf.fock`) expose the individual pieces for users who need them.

### Coarse → fine continuation

For large fine grids, `solve_continuation` runs a cheap coarse solve first
and uses its density to seed the fine solve. The two stages can mix and
match direct minimisation and SCF:

```python
import jax_hf
from jax_hf import SCFConfig, SolverConfig

coarse = jax_hf.HartreeFockKernel(weights_c, h_c, Vq_c, T=0.1)
fine   = jax_hf.HartreeFockKernel(weights_f, h_f, Vq_f, T=0.1)

result = jax_hf.solve_continuation(
    coarse, fine, P0_coarse=jnp.zeros_like(h_c),
    n_electrons_coarse=N, n_electrons_fine=N,
    coarse_config=SCFConfig(max_iter=50, mixing=0.5),   # robust coarse
    fine_config=SolverConfig(max_iter=200, tol_E=1e-8), # fast fine
)
# result.coarse, result.fine (each a SolveResult or SCFResult)
# result.P0_fine (resampled coarse density used to seed the fine solve)
```

The driver is intentionally algorithm-agnostic: it resamples the coarse
density onto the fine grid via `resample_kgrid` and hands off. Callers
that need physics-aware seeding (reference-density interpolation,
self-energy seeds, filling-consistent electron counts across grids) should
construct both kernels themselves.

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

Install the checkout and its development dependencies first:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

The bilayer regression tests (`tests/test_bilayer_regression.py`) require
`contimod` and `contimod_graphene` and will be skipped otherwise.

Because `contimod` is not publicly accessible, the default GitHub Actions
suite does not attempt to install it. Repository maintainers can turn the
bilayer matrix into an additional CI and release gate by setting the
`RUN_BILAYER_REGRESSIONS` repository variable to `true` and adding a
`BILAYER_REPOSITORY_TOKEN` secret with read access to both dependency
repositories. When enabled, a failing bilayer job prevents a tagged release
from publishing.
