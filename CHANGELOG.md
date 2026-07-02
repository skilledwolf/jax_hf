# Changelog

All notable changes to **jax_hf** are documented here.
This project adheres to [Semantic Versioning](https://semver.org).

## [2.2.1] — 2026-07-02

### Changed
- **`tol_grad` is now a sufficient stopping criterion for the CG path**,
  matching `cpp_hf` v1.1.1: with `tol_grad > 0` the run stops — and reports
  `converged=True` — at the first iterate whose orbital-gradient norm drops
  to `tol_grad`; `tol_E` is then inactive. Previously the CG loop required
  the windowed-energy plateau AND the gradient criterion, so soft-mode runs
  could sit below `tol_grad` for hundreds of iterations and finish
  `converged=False` at `max_iter`. `tol_grad = 0` (default) keeps the
  windowed-energy criterion; iterate trajectories are unchanged. The Newton
  path was already gradient-only and is untouched.
- `converged` for gradient-stopped runs is derived from the final measured
  gradient (`grad_fin <= tol_grad`), so a run whose last recorded iterate
  qualifies exactly at `max_iter` is no longer misflagged unconverged.

### Added
- `TestGradientStop` regression tests (first-crossing stop, same-solution
  check against the energy stop, unreachable tolerance flags unconverged).
  Float32 calibration notes live in the test comments: the orbital-gradient
  noise floor is ~1e-5, and a gradient stop leaves O(1e-4) occupation-channel
  energy settling relative to the tighter windowed-energy stop.

## [2.2.0] — 2026-06-10

This release brings the JAX solver to feature parity with the C++ reference
`cpp_hf`, and is verified behaviorally equivalent to it (see Notes).

### Added
- **Trust-region Newton solver** — `SolverConfig(optimizer="newton")`. A Steihaug
  truncated-CG on the joint (Q, p) response Hessian (exact linear interaction
  response, one Fock build per Hessian-vector product). Converges superlinearly
  — a handful of outer steps versus hundreds for CG on stiff problems. Tuned via
  `tr_delta0`, `tr_cg_max`, `tol_grad`. Second-order method: needs float64
  (warns on float32).
- **Deflation** — `solve_deflated` / `DeflatedResult`. Finds *distinct*
  self-consistent HF solutions on non-convex landscapes by adding a repulsive
  Gaussian bias around already-found densities and re-polishing each result
  unbiased. Returns solutions sorted by energy.
- **SCF acceleration** — `SCFConfig.acceleration ∈ {"linear", "diis", "oda"}`:
  Pulay commutator-DIIS (typically ~10× fewer iterations than linear mixing,
  tunable via `diis_size` / `diis_start` / `diis_damping`) and ODA optimal
  damping. Plus a `trust_radius` clip on the per-iteration density step.
- **Windowed CG convergence** — `SolverConfig.plateau_window` (default 5): stop
  on the energy improvement over a sliding window rather than a single step,
  which is robust to per-step CG noise. `0` restores the per-iteration test.
- **`embed_reference_potential`** mode in `HartreeFockKernel`: fold
  `V_HF[reference_density]` into the Hamiltonian so the solver works with
  absolute (rather than displacement) densities.
- **Contact terms** — q-independent flavor-bilinear interactions, via
  `HartreeFockKernel(contact_terms=[(g, O_i, O_j), ...])`.
- **Superlattice / moiré streaming Fock** — JIT streaming exchange and
  orbital-resolved Hartree (`build_extended_layout`,
  `make_superlattice_fock_fn`, `make_superlattice_build_fock_fn`), including
  caller-supplied q=0 layer Hartree (a nonzero diagonal-G block now warns rather
  than raising).

### Changed
- Direct-minimization solver: spectral Cayley retraction, with redundant
  symmetry projections and chemical-potential solves removed from the inner loop.

### Notes
- Verified behaviorally equivalent to `cpp_hf` to machine precision on CG,
  trust-region Newton, SCF (linear/DIIS/ODA), and deflation, and *more* correct
  on the decoupled cold-start case (a cold-start occupation relaxation recovers
  the exact ground state where the C++ reference stops at unrelaxed occupations).

## [2.1.0]

### Added
- `solve_continuation` / `ContinuationResult` — coarse→fine multigrid driver
  that seeds a fine solve from a cheap coarse one; `resample_kgrid`.

## [2.0.0]

- Clean-slate rewrite: direct-minimization solver plus a reference SCF baseline.
  The entire public API changed relative to the deprecated v1.x line — see
  [`MIGRATION.md`](MIGRATION.md).
