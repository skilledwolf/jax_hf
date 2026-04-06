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
