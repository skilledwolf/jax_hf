# TASKS

Legend: `[ ]` not started, `[~]` in progress, `[x]` done.

## Completed scrutiny

- [x] Scrutinize core solver implementation.
- [x] Scrutinize public API, tests, and runtime import behavior.
- [x] Scrutinize packaging, CI, and documentation.

## Discovered during scrutiny

- [x] Remove import-time JAX device allocations from `src/jax_hf/mixing.py` default arguments and add an import smoke test.
- [x] Decide whether `solve()` should default to QR or SCF, then either restore the previous default or document and version the breaking change.
- [x] Validate requested electron counts against the physically reachable range before solving or constructing density matrices.
- [x] Reject `diis_size <= 0` with a clear validation error and regression coverage.
- [x] Add shape validation to the variational coarse-to-fine path so bad inputs fail with targeted errors.
- [x] Make `.github/workflows/build-and-test.yml` run the test suite instead of metadata-only smoke checks.
- [x] Gate `.github/workflows/release.yml` on real runtime verification before publishing to PyPI.
- [x] Add a reproducible developer/test install path, such as a `dev` extra or dedicated requirements file, and document it.
- [x] Document the Python `>=3.11` requirement in the installation section.
- [x] Reconcile the README API section with the actual package exports, including whether `hartreefock_iteration` should be public.
- [x] Investigate the current regression mismatch in `tests/test_meshgrid_regression.py` where the fixture expects `k_fin == 15` but the present run returns `13`.
- [x] Investigate the remaining `vcs_versioning` warning emitted during isolated builds.  *(No longer reproduced by `python -m build -s -w` on v2.2.1; archive builds still need the separate setuptools-scm fix below.)*
- [x] Remove the avoidable `tau=0` Cayley solve in the direct-minimization line search and evaluate caching the accepted trial unitary.  *(Done: τ=0 is now inlined; the line-search trials and the post-line-search retraction share a single `eigh(i*d_Q)` decomposition that replaces the LU solves and the U†FtU triple product with one Hadamard scaling + one matmul.  Per-iter speedup ~10–35% at nb≥16, neutral on bilayer DM regression suite.  See `benchmarks/cayley_baseline.txt` for measurements.)*
- [x] Pass `contact_terms` through the direct-minimization hot-loop Fock build and add regression coverage for contact interactions.

## v2.1

- [x] Add `solve_continuation` / `ContinuationResult` — a thin coarse → fine multigrid driver so downstream callers don't re-implement the coarse-solve-then-resample-then-fine-solve sequence. Ships with validation tests for config dispatch, kernel shape mismatches, and convergence-to-same-fixed-point.

## Discovered during 2026-07-30 v2.2.1 scrutiny

- [x] Re-audit v2.2.1 core numerics, solver/API behavior, tests, packaging, CI, and documentation.
- [x] Require an occupation/KKT or `P - P_FD(F)` residual before either direct optimizer reports convergence; the current orbital-only test can return a maximally inconsistent density.
- [x] Make CG/Newton optimization and convergence decisions use the finite-temperature free energy, reject signed energy increases as convergence, and recheck stationarity after CG canonicalization.
- [x] Correct the reference-subtracted interaction energy so `dE/dP` equals the Fock matrix returned by `build_fock`.
- [x] Make ODA include entropy at finite temperature (or explicitly restrict ODA to `T=0`).
- [x] Make float32 occupation entropy endpoint-safe so saturated occupations do not produce `NaN` and disable CG line-search acceptance.
- [x] Apply nonuniform k-point quadrature weights to the density in FFT exchange rather than to the transfer-momentum interaction kernel, with a nonuniform-weight regression test.
- [x] Repair the low-temperature Newton chemical-potential solver and turn its existing xfail into a passing regression.
- [x] Validate SCF electron counts before entering JIT, and handle exact empty/full fillings without reporting a mismatched particle count as converged.
- [x] Reject negative temperatures (or implement consistent negative-temperature semantics in both solvers and chemical-potential paths).
- [x] Include contact interactions in `embed_reference_potential` and `make_superlattice_build_fock_fn`, including the embedded energy offset, or reject those combinations explicitly.
- [x] Make the Newton trial step consistent with its Steihaug model: either apply the computed occupation step `vp` or remove it from predicted reduction and trust-radius calculations.
- [x] Validate `P0` shapes and add `SolverConfig` domain validation before JIT compilation.
- [x] Raise the build-system setuptools floor to a version that supports the declared SPDX `license` / `license-files` metadata.
- [x] Make release CI run the optional bilayer regression matrix and test the advertised Python range, not only the dependency-light suite on Python 3.12. *(The dependency-light matrix is unconditional; the private-dependency bilayer job is a documented credential-gated release check.)*
- [x] Preserve setuptools-scm versions in tagged VCS archives instead of silently building version `0.0`.
- [x] Make local release builds clean/reproducible so stale files in `build/lib` cannot leak removed v1 modules into a new wheel.
- [x] Port or remove tracked example/debug scripts that still import removed v1 modules such as `jax_hf.main` and `jax_hf.variational*`, and smoke-test shipped examples.
- [x] Declare NumPy directly, remove unused Optimistix/Optax runtime dependencies unless intentional, and document the tested dependency floor.
- [x] Fix README installation/testing prerequisites and make every fenced Python example syntactically valid.
- [x] Remove or update optimizer diagnostics invalidated by the corrected true-objective and finalization behavior.

## Downstream validation

- [x] Re-run the `../contimod/examples` workflows that use the `jax_hf` backend against the current checkout, compare their convergence and physical outputs with saved/reference results, and record any regressions or qualifications.
  - Representative reduced runs covered the meshgrid, scalar superlattice, orbital-resolved, and TDHF consumers in examples 11, 12, 20, 50, 53, 60, 62, and 63, plus 77 focused downstream tests. General/scalar JAX results agreed closely with C++; production-scale scans were runtime-limited. The confirmed example regressions and documentation qualifications are tracked below.
- [ ] Update Contimod collective-mode response-Fock calls in `realtime_tdhf.py`, `lanczos_tdhf.py`, and `acfd_tdhf.py` that reuse `kernel.as_args()["VR"]`: after quadrature weights moved from the interaction transform onto the density, raw `selfenergy_fft(VR, dP)` calls over-scale exchange; pass `weights_b * dP` or route through a supported response builder and add a regression. (`acfd_tdhf_2deg.py` constructs a weighted transform itself and is unaffected.)
- [ ] Update the Contimod tetralayer cRPA README's obsolete warning that `jax_hf` reports energy with `P` instead of `P - P_ref`; the corrected backend energy now agrees with an independent reference-consistent reconstruction to floating-point precision.
- [ ] Update Contimod example 11's branch-tracker convergence export for the current JAX history schema: `dC_final` is now saved as all `NaN` because `jax_hf` no longer exposes the old `dC` diagnostic, even though the solves and order-parameter outputs remain finite.
- [ ] Regenerate Contimod example 53's hard-coded reference narrative and plots for the current model artifact: current `hf.py`, `cpp_hf`, and `jax_hf` agree near `-2751.7041862` meV/cell, while the documented `-2755.6632` result and plots predate the current model file.
- [ ] Give Contimod example 50's `--nk` option a plotting-safe lower bound (at least 4) or handle smaller meshes explicitly; the `nk=3` JAX solve converges, but its `RectBivariateSpline` contour plot fails after the solve.
- [x] Implement orbital-resolved exchange in the JAX superlattice adapter used by Contimod example 63; the full orbital tensor now reaches streaming exchange, Hartree construction, reference embedding and custom-Fock/JVP solvers.

## Discovered during 2026-09-02 performance assessment

- [x] Fix the finite-temperature occupation gradient at saturated occupations. `_logit` clips at `1e-14`, so for bands with `|mu - eps| > ~32 T` the raw gradient `eps + T*logit(p)` missed `mu` by up to ~8e2 meV on the nk=49 bilayer (98.5% of occupations are saturated at T=0.5 meV); `_project_occ` spread that spurious constant onto the partially filled bands, the occupation direction became an ascent direction (analytic slope -1.5e-2 vs measured +2.7e-1), and the CG line search was rejected on every iteration (8 trials, `tau_final = 0`), so the solver advanced only through the FD fallback at `lam = 1`. *(Done: saturated bands are masked out of the particle-conservation projection and given zero curvature, on both the CG and Newton paths. Real bilayer, six branch/density points: 9.9 -> 2.9-4.3 Fock builds per CG iteration, 1.55x over the twelve CG runs, every converged energy bit-identical. Synthetic multiband problems unchanged. Coverage in `tests/test_solver_saturation.py`.)*
- [ ] Investigate the PM +0.05 CG iteration-count regression. With `tol_grad=1e-5` that point now takes 32 iterations instead of 11 (same converged energy, and unchanged at `tol_E=1e-7`). The masked occupation gradient is a genuinely different search direction; on that one point it takes a longer route. It is the only regression among 24 measured branch/density/criterion combinations, but it is worth understanding before tightening the default tolerances.
- [x] Stop `_solve_mu` on convergence instead of always running `mu_maxiter` steps. *(Done, with the count tolerance scaled to the target electron count rather than the band capacity: the CG occupations come straight from this solve, and the capacity scaling left a ~1.5e-4 worst-case relative particle error in float32 at low filling versus ~4e-5 now.)*
- [ ] Reduce the remaining per-iteration eigendecompositions in `_solve_impl`. Batched `eigh` of `(nk^2, nb, nb)` is the dominant CPU primitive (32 ms for 49^2 x 16 x 16 complex64, versus 1.9 ms for `build_fock`) and each iteration still performs three: the Fermi-Dirac residual, the spectral Cayley setup, and the fallback's `P_mix`. Two approaches were implemented and rejected, both with measurements:
  - *Gating the residual and the fallback* on the loop being able to stop or the line search failing. The fallback is load-bearing for the convergence **rate**, not merely a safety net: gating it cost more iterations than it saved per iteration (synthetic nk=49 nb=16 went from 4 and 47 iterations to non-convergence in 300; the bilayer PM +0.05 point went from 10 to 92).
  - *Reusing the `lam = 1` eigendecomposition* in the fallback, where `P_mix` equals the already-decomposed `P_fd`. Not numerically neutral when a `project_fn` is supplied: the reused spectrum is that of the unprojected `P_fd`, and the bilayer PM +0.05 trajectory changed (11 -> 12 iterations at `tol_grad=1e-5`; it helped there, but by luck). It would be sound if restricted to `project_fn is None`, which excludes the production case.
- [ ] Speed up the CG path on the hard doped points. `SVP -0.42` still needs 105 iterations to `tol_E=1e-7` and does not reach `tol_grad=1e-5` in 200, while SCF DIIS converges in 18. A redesign that replaces the occupation gradient step with a damped frozen-Fock Fermi-Dirac re-occupation inside the line search was prototyped: it reaches 2.0 Fock builds per iteration and 3.0x wall time on the bilayer, but converges 15x more slowly on a warm, weakly interacting synthetic problem (4 -> 62 iterations at `tol_grad=1e-6`), because it replaces a genuine joint CG direction with a linearly convergent fixed-point map wherever occupations are unsaturated. Reviving it needs CG acceleration over the Fermi-Dirac direction, not just the direction itself.
- [ ] Investigate CPU parallelism and a GPU path. Solves use ~1.2 of 14 cores. Sharding k over host devices speeds batched matmuls ~7x but the LAPACK `eigh` FFI only ~1.3x (serialized), and k-sharded FFTs are slower. Batched Jacobi `eigh` on an NVIDIA GPU is the natural path; Metal lacks batched complex `eigh`.
- [x] Restore reproducible bilayer references across contimod grid defaults. The saved energies agree with the current solvers on the original endpoint grid; persist `kx`, `ky`, and the quadrature weight and replay them explicitly. All 30 converged reference assertions pass without regenerating expected energies.

## 2026-09-05 solver-study port

- [x] Port safeguarded full-density residual DIIS/Anderson while preserving existing JAX API and custom Fock support.
- [x] Correct joint Newton geometry, occupation steps and interaction curvature with finite-difference regressions.
- [x] Restore exchange reciprocity and add finite-temperature local stability checks and bounded saddle recovery.
- [x] Validate parity against cpp_hf and document measured scope and limitations. (211 core tests passed in the actual checkout; eight independent C++ parity cases agreed to <1e-12. The saved optional bilayer fixture remains a separately tracked pre-existing issue.)

- [x] Preserve finite-temperature saddle-escape descent near saturated occupations using a fixed-particle Fermi-map retraction, with projected-density fallback and verified free-energy acceptance from the actual returned density and energy. (Final full actual-checkout validation: 215 core tests passed in 87.63 seconds with no warnings, including saturated coupled-band curvature and stale legacy SCF history regressions.)

- [x] Implement memory-bounded orbital-resolved superlattice exchange, preserving scalar behavior and supporting custom-Fock/JVP adapters. Validation: 57 focused JAX/Contimod tests passed; the reduced 228-band hexalayer Fock agrees with C++ to 5.7e-14. Its cold RDIIS solve reaches the 80-iteration budget without convergence, so this validates the operator and does not establish phase-diagram convergence.

## 2026-10-13 cpp_hf fixes parity

- [x] Correct direct-solver projection/entropy consistency and low-temperature/dilute chemical-potential accuracy, with JAX regression coverage.
- [x] Preserve the density certified by legacy SCF and reuse the chemical potential when level shifting is disabled.
- [x] Preserve complex symmetry operators on real densities and correct the centered embedded reference energy conversion.
- [x] Check applicable performance improvements, validate the optional bilayer quadrature, and rerun the test suite.

Validation: 286 core tests and 30 bilayer tests passed on CPU (Python 3.12, JAX 0.7.2); the two existing skips are the saved unconverged SVP reference point. Focused float32 checks also passed with x64 disabled. JAX already reused the stability eigensystem and installed editable source in CI; FFTW thread controls do not apply to this backend.
