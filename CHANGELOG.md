# Changelog

All notable changes to **jax_hf** are documented here.
This project adheres to [Semantic Versioning](https://semver.org).

## Unreleased

### Correctness
- Project the CG Fock consistently with the density, and keep CG/Newton
  occupations synchronized with projected trial densities before evaluating
  entropy or accepting a step.
- Return the density actually certified by legacy SCF. Use occupation-aware
  chemical-potential brackets and representable-update stopping to preserve
  dilute fillings and resolve low-temperature occupations near remote bands.
- Preserve complex symmetry operators on real densities. Expose
  `embedded_hartree_shift` and correct the centered reference energy conversion
  using the actual quadrature-weighted particle count.
- Require the full weighted `||P - P_FD(F)||` residual before either direct
  optimizer reports convergence. Near-equal occupations can suppress the
  commutator/orbital gradient even when the density is not self-consistent.
- Use true finite-temperature free energy for CG/Newton decisions, reject
  objective increases as plateaus, apply the occupation displacement used by
  the Newton trust model, and use checked damped fixed-point steps when the
  local parameterization stalls. `history["E"]` remains the internal HF
  energy; `history["free_energy"]` records the optimized objective.
- Correct reference-subtracted interaction energy so its derivative is the
  returned Fock matrix, place nonuniform quadrature weights on the convolved
  density, and make float32 entropy finite at occupations exactly 0 and 1.
- Include contact interactions in embedded-reference and superlattice Fock
  paths, including the embedded energy offset.
- Make finite-temperature ODA descend free energy, repair low-temperature
  Newton chemical-potential bracketing, handle exact empty/full fillings, and
  reject negative temperatures and invalid solver inputs before JIT.

### Performance notes
- Reuse the initial SCF chemical-potential solve when level shifting is zero.
- Reuse accepted projected CG trial eigensystems, while retaining the original
  unprojected reconstruction order for float32 convergence.
- Apply unitary and spatial group averages in `make_project_fn` as one
  precomputed GEMM on `vec(A)` when `nb <= 24` (the `(nb², nb²)` matrix is
  formed once from the concrete unitaries). On the nk=49, nb=16 bilayer the
  PM projector drops from 31 ms to 5.7 ms per call; it is applied about six
  times per CG iteration (every density that enters a Fock build, plus the
  Fermi-Dirac fixed point), so the CG solve runs 1.75x faster end to end
  (514 -> 293 ms per iteration in the same process). Above the threshold the
  per-element conjugation loop is kept.
- Fix a dtype promotion in the projector: a complex128 group or time-reversal
  unitary applied to a complex64 density promoted the `fori_loop` carry and
  raised under x64; the operators now follow the density dtype.
- CG backtracking now evaluates the true nonlinear objective and may perform
  multiple Fock builds per outer iteration. Newton may use a damped
  fixed-point fallback near occupation degeneracies. These checks trade some
  per-iteration cost for reliable stationarity and monotone free energy.
- Fix the finite-temperature occupation step at saturated occupations. At low
  temperature nearly every occupation sits exactly at 0 or 1 in floating point
  (98.5% on the bilayer graphene regression problem), and the logit of such a
  band is clipped, so `eps + T*logit(p)` can miss the chemical potential by
  hundreds of meV. Feeding that into the uniform particle-conservation shift
  spread the error onto the partially filled bands and made the occupation
  search direction an ascent direction, so the line search rejected every
  candidate and the solver advanced only through its Fermi-Dirac fallback --
  effectively undamped Roothaan iteration at about 10 Fock builds per step.
  Saturated bands are now treated as active bound constraints: they are held
  fixed and excluded from the particle-conservation projection, which is
  enforced over the resolvable bands (curvature-weighted on the Newton path).
- Stop the chemical-potential solve once the particle count and the update
  reach working precision instead of always running `mu_maxiter` steps, and
  scale its count tolerance to the target electron count rather than the band
  capacity. The CG occupations come straight from this solve, so its accuracy
  carries the particle count; the previous scaling left a relative particle
  error of order `capacity / n_target` at low filling.
- Measured on the bilayer graphene regression problem (nk=49, nb=16, float64,
  CPU) over six branch/density points and two stopping criteria: CG spends 9.9
  Fock builds per iteration before the fix and 2.9-4.3 after, for a 1.55x
  wall-time reduction over the twelve CG runs (1.37x over Newton, 1.02x over
  SCF). Every converged energy is bit-identical, and iteration counts are
  unchanged except at PM +0.05 with `tol_grad=1e-5`, which takes 32 iterations
  instead of 11 -- the masked gradient is a different search direction, and on
  that one point it takes a longer route to the same solution. Synthetic
  multiband problems are unchanged (4, 47 and 3 iterations, as before).

### Packaging and CI
- Replay the saved bilayer quadrature explicitly so contimod grid-default
  changes do not invalidate the existing reference energies.
- Require PEP 639-capable setuptools, preserve versions in tagged
  `git archive` exports, and reject `0.0` or stale-module wheels during CI.
- Declare the directly imported NumPy dependency, set the tested JAX/NumPy
  floors, and remove unused Optax and Optimistix runtime dependencies.
- Test Python 3.11–3.14 and provide a credential-gated bilayer regression job
  that becomes a release gate when enabled.
- Remove obsolete v1 diagnostic scripts and validate README Python snippets
  and tracked diagnostics in the test suite.

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
  They run in float64: the float32 orbital-gradient noise floor is
  platform-dependent (~1e-5 on macOS ARM, ~2e-4 on Linux CI), so float32
  gradient tolerances are not portable.

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
