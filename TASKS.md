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
- [ ] Investigate the remaining `vcs_versioning` warning emitted during isolated builds.
- [x] Remove the avoidable `tau=0` Cayley solve in the direct-minimization line search and evaluate caching the accepted trial unitary.  *(Done: τ=0 is now inlined; the line-search trials and the post-line-search retraction share a single `eigh(i*d_Q)` decomposition that replaces the LU solves and the U†FtU triple product with one Hadamard scaling + one matmul.  Per-iter speedup ~10–35% at nb≥16, neutral on bilayer DM regression suite.  See `benchmarks/cayley_baseline.txt` for measurements.)*
- [ ] Pass `contact_terms` through the direct-minimization hot-loop Fock build and add regression coverage for contact interactions.

## v2.1

- [x] Add `solve_continuation` / `ContinuationResult` — a thin coarse → fine multigrid driver so downstream callers don't re-implement the coarse-solve-then-resample-then-fine-solve sequence. Ships with validation tests for config dispatch, kernel shape mismatches, and convergence-to-same-fixed-point.
