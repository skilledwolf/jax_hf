# jax_hf v2.0.0 migration guide

v2.0.0 is a clean-slate rewrite.  It replaces the v1.x SCF-centric solver
with a **direct-minimisation** solver (preconditioned Riemannian CG on
Stiefel × capped simplex, eigen-free inner loop, Cayley retraction).

The SCF algorithm is still available as a separate `solve_scf` baseline,
but the v1 public API (`hartreefock_iteration`, `jit_hartreefock_iteration`,
`HartreeFockResult`, the `mixing` module) has been removed outright rather
than shimmed.  v1.1.0 had already been declared a deprecated skeleton, so
we are not surprising anyone by breaking compatibility here.

If you need v1 behaviour as a *baseline* (not as an import-compatible
shim), use `jax_hf.solve_scf` — it is the same class of algorithm (Fock
build → eigh → Fermi-Dirac re-occupation → linear mix) with a cleaner API.

## API mapping

| v1.0.2 (last working v1) | v2.0.0 |
|---|---|
| `jax_hf.HartreeFockKernel(weights, hamiltonian, coulomb_q, T)` | `jax_hf.HartreeFockKernel(weights, hamiltonian, coulomb_q, T, include_hartree=False, include_exchange=True, reference_density=None, hartree_matrix=None)` |
| `jax_hf.jit_hartreefock_iteration(kernel)` → `runner(P0, electrondensity0=…, max_iter=…, comm_tol=…, diis_size=…, precond_mode=…)` | `jax_hf.solve(kernel, P0, n_electrons, config=SolverConfig(…))` for direct minimisation, or `jax_hf.solve_scf(kernel, P0, n_electrons, config=SCFConfig(…))` for SCF |
| `jax_hf.HartreeFockResult(P_fin, F_fin, E_fin, mu_fin, k_fin, history)` | `jax_hf.SolveResult(Q, p, mu, density, fock, energy, n_iter, converged, history)` (direct minimisation) or `jax_hf.SCFResult(density_matrix, fock_matrix, energy, chemical_potential, iterations, converged, message, history)` (SCF) |
| `jax_hf.mixing` (DIIS utilities) | **Removed.**  Linear mixing lives inside `solve_scf`; DIIS is not currently reimplemented. |
| `jax_hf.jax_modules`, `jax_hf.wrappers` | **Removed** (implementation detail). |

## Drop-in translation

### v1.0.2

```python
import jax_hf

kernel = jax_hf.HartreeFockKernel(
    weights=weights, hamiltonian=h, coulomb_q=Vq, T=0.1,
)
runner = jax_hf.jit_hartreefock_iteration(kernel)
P_fin, F_fin, E_fin, mu_fin, k_fin, history = runner(
    P0, electrondensity0=N,
    max_iter=100, comm_tol=1e-5, diis_size=4, precond_mode="diag",
)
```

### v2.0.0 — direct minimisation (recommended)

```python
import jax_hf

kernel = jax_hf.HartreeFockKernel(
    weights=weights, hamiltonian=h, coulomb_q=Vq, T=0.1,
    include_hartree=False, include_exchange=True,
)
config = jax_hf.SolverConfig(max_iter=100, tol_E=1e-7)
result = jax_hf.solve(kernel, P0, n_electrons=N, config=config)

# result.density, result.fock, result.energy, result.mu, result.n_iter
# result.history["E"], result.history["grad_norm"]
```

### v2.0.0 — SCF baseline

```python
import jax_hf

kernel = jax_hf.HartreeFockKernel(...)  # same constructor as above
config = jax_hf.SCFConfig(
    max_iter=300, mixing=0.3, density_tol=1e-7, comm_tol=1e-6,
)
result = jax_hf.solve_scf(kernel, P0, n_electrons=N, config=config)

# result.density_matrix, result.fock_matrix, result.energy,
# result.chemical_potential, result.iterations, result.converged
```

Note the SCF result uses `.density_matrix` / `.fock_matrix` /
`.iterations`, while the direct solver uses `.density` / `.fock` /
`.n_iter`.  This is intentional — the two algorithms expose slightly
different information and it would be misleading to paper over the
difference with a unified result type.

## New capabilities in v2.0.0

* **Hartree term:** pass `include_hartree=True`, `reference_density=<P_ref>`
  and `hartree_matrix=<HH>` when constructing the kernel.  The Hartree
  contribution is computed relative to `reference_density`.  See
  `examples/multilayer_graphene_density_scan_hartree.py`.
* **Layer-resolved exchange:** `coulomb_q` may be shape
  `(nk1, nk2, nb, nb)` (per-orbital) in addition to the scalar
  `(nk1, nk2, 1, 1)` form used by the dualgate Coulomb kernel.
* **Symmetry projection:** `SolverConfig(project_fn=…)` and
  `SCFConfig(project_fn=…)` accept a callable that projects density /
  Fock onto a symmetry-invariant subspace.  See
  `jax_hf.symmetry.make_project_fn`.
* **HF objective building blocks:** `jax_hf.build_fock`,
  `jax_hf.hf_energy`, `jax_hf.free_energy`, `jax_hf.occupation_entropy`
  are public and can be called on arbitrary densities (e.g. for
  diagnostics or to build custom solvers).
* **Convergence on energy change:** the direct solver stops when
  `|E_k − E_{k−1}| < tol_E` rather than on a commutator norm.  An
  optional gradient-norm backstop (`tol_grad`) is available.

## Convergence tolerances: rough equivalents

| v1.0.2 SCF | v2.0.0 direct | v2.0.0 SCF |
|---|---|---|
| `comm_tol=1e-5` | `tol_E ≈ 1e-7` | `comm_tol=1e-5` |
| `comm_tol=1e-6` | `tol_E ≈ 1e-8` | `comm_tol=1e-6` |
| `diis_size=4` | (not applicable — DM has its own line search) | (linear mixing only; DIIS is not currently reimplemented) |

For bilayer graphene at `nk=49`, the direct solver typically converges
to `tol_E=1e-7` in ~5–15 iterations per density point (wall-time
dominated by the FFT in the Fock build).

## Why the clean break?

v1.1.0 was already marked deprecated with a pointer to a separate
package.  Rather than carry forward v1 shims indefinitely, we took the
opportunity to ship a solver that is:

* Substantially faster (direct minimisation needs ~5× fewer Fock builds
  than SCF on our bilayer benchmark).
* More robust at phase boundaries (cold-seed per density point +
  explicit symmetry projection handles branch drift cleanly).
* Smaller and easier to maintain (the solver + SCF fallback + Fock
  evaluator total well under 1000 lines).

If you need v1 behaviour and can't migrate, pin to
`jax-hf==1.0.2` (the last v1 release with a working solver).
