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
- [ ] Remove the misleading `superlattice_jax` CLI option from Contimod example 63, or implement orbital-resolved exchange in the JAX superlattice adapter; the example requires `coulomb_V_orbital` and currently raises `NotImplementedError` immediately.
