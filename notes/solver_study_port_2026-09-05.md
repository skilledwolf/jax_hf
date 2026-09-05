# Solver-study port, 5 September 2026

The JAX backend now provides safeguarded full-density residual DIIS (`rdiis`)
and Anderson (`anderson`). These fit the fixed-particle Fermi residual, including
occupations, rather than only the commutator. The implementation is native JAX,
with direct residual-difference SVD, feasible weighted spectral projection,
free-energy backtracking, affine Fock reuse, history reset/fallback, and an exact
rebuild of the returned state's physical Fock. Existing defaults and result
formats are preserved; the new configuration field is appended.

`solve_minimum` adds host-side NumPy/SciPy local stability testing and bounded
saddle escape. It tests the fixed-particle Hermitian density Hessian, including
variations between equal occupations. A successful result requires stationarity,
a converged and independently checked eigenpair, and no resolved negative
curvature. Trial escapes must actually lower the free energy beyond the kernel's
precision. The diagnostic is a local check in the supplied basis, not a proof of
a global ground state. Default tight tolerances require JAX double precision.
Affine, variational, JAX-differentiable custom Fock builders are supported through
SCF, exact JVP response, and finite-step energy evaluation. The full stability and
minimum drivers are not JIT-compatible.

The kernel now removes the skew-adjoint part of the FFT exchange multiplier.
Radial sampling on an even oblique grid need not obey periodic inversion; the
old Fock could therefore differ from the derivative of the reported quadratic
energy. The correction preserves that scalar energy at fixed unembedded h and
reference, restores its derivative, and records `exchange_reciprocity_defect`.
It does not correct circular-convolution finite-size error. Embedded h is built
from the corrected response. Invalid quadrature weights are rejected.

The Newton corrections use one normalized quadrature metric, the covariant
unitary connection, the complete Gaussian-deflation Hessian, and the actual
coupled orbital/occupation Newton step with one fraction-to-boundary scale.
Density, effective Fock and response are projected consistently. The standard
response omits the one-body term directly; custom responses use JAX JVP.
The legacy CG implementation and JIT configuration were kept unchanged.

## Validation and scope

**211 core tests passed** in the actual sibling checkout, including the new
regressions. The optional saved bilayer fixture is reported separately below.

The independent eight-case [C++ comparison](solver_study_parity_2026-09-05.jsonl)
uses four complex, coupled Hartree/exchange/reference-subtracted kernels and two
methods. All eight converge with identical iteration counts; maximum energy
difference is 7.7e-13 and maximum density-element difference is 8.0e-13.
Reproduce with `python benchmarks/cpp_solver_parity.py` when cpp_hf is installed.
The package itself has no cpp_hf dependency.

New regressions cover transported Hessian finite differences and adjointness,
full deflation, custom Fock response, occupation feasibility and saturation,
fixed-particle response at degeneracy, kernel reciprocity/energy preservation,
false stability certificates, failed eigensolves, and bounded saddle recovery.
The optional saved bilayer reference still predates the corrected energy
convention; its pre-existing mismatch is tracked in TASKS.md and its fixture was
not rewritten as part of this port.

This ports the validated numerical ingredients from the sibling C++ study.
It does not establish the same wall-time speedups in JAX: compilation, CPU
batched eigensolves, and history/SVD memory differ. RDIIS stores two full
complex density-sized history stacks plus SVD and state temporaries. Benchmark
the intended problem and device, including compilation when relevant. The
experimental Hartree preconditioner and automatic Newton hybrid were not ported;
neither improved the main C++ comparison consistently enough to justify extra
JAX API and implementation complexity in this change.
