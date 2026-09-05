"""Stability-checked local minimization with bounded negative-curvature recovery."""
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
from scipy.special import xlogy
from .reference_scf import SCFConfig, SCFResult, solve_scf
from .stability import StabilityResult, stability_analysis, _evaluate_fock, _fock_kwargs


@dataclass(frozen=True)
class MinimumResult:
    solution: SCFResult
    stability: StabilityResult | None
    converged: bool
    escapes: int
    total_iterations: int
    fock_builds: int
    free_energies: tuple[float, ...]
    message: str


def _feasible_density(P, weights, number):
    """Weighted Frobenius projection onto the fixed-trace density spectrahedron."""
    p, Q = np.linalg.eigh(.5*(P+P.conj().swapaxes(-2,-1)))
    lo, hi = -float(p.max())-1., 1.-float(p.min())+1.
    for _ in range(60):
        mid = .5*(lo+hi)
        if np.sum(weights[...,None]*np.clip(p+mid,0.,1.)) < number:
            lo = mid
        else:
            hi = mid
    p = np.clip(p+.5*(lo+hi),0.,1.)
    return np.ascontiguousarray((Q*p[...,None,:])@Q.conj().swapaxes(-2,-1))


def solve_minimum(kernel, P0, n_electrons, *, config=None, max_escapes=4,
                  eigen_tol=1e-7, stability_maxiter=200,
                  fock_build_fn=None) -> MinimumResult:
    """Find a stationary density with no resolved negative curvature.

    Each local solve has ``config.max_iter`` iterations; at most
    ``max_escapes + 1`` local solves are performed. All additional Fock builds,
    including the stability test and trial perturbations, are counted for
    RDIIS/Anderson. A legacy SCF method without a counter returns -1 for the
    total build count. The default is safeguarded RDIIS at 1e-8 residuals
    with 500 iterations.
    ``converged`` requires both a converged local solve and a conclusive stable
    Hessian test. This is a local minimum check within the supplied basis, not
    a global ground-state certificate or a test of omitted spin/valley sectors.

    Extra symmetry callbacks are not supported: recovery must be allowed to
    break symmetry. ``fock_build_fn`` follows the affine, variational,
    JAX-differentiable contract of :func:`stability_analysis`, and is used
    consistently in SCF, curvature, and finite-step energy checks. The host
    driver uses NumPy/SciPy orchestration and is not itself JIT-compatible.
    Enable JAX double precision before constructing the kernel for the default
    tolerances. The returned SCF solution contains its density and Fock matrix.
    """
    if not isinstance(max_escapes,(int,np.integer)) or isinstance(max_escapes,bool) or max_escapes < 0:
        raise ValueError('max_escapes must be a nonnegative integer')
    if kernel.T <= 0 or np.any(np.asarray(kernel.w2d) <= 0):
        raise ValueError('solve_minimum requires positive temperature and weights')
    if (not np.isfinite(eigen_tol) or eigen_tol <= 0
            or not isinstance(stability_maxiter, (int, np.integer))
            or isinstance(stability_maxiter, bool) or stability_maxiter <= 0):
        raise ValueError('eigen_tol and stability_maxiter must be positive')
    precision = np.finfo(np.asarray(kernel.h).real.dtype).eps
    if eigen_tol < 10*precision:
        raise ValueError('stability tolerance requires greater kernel precision; '
                         'enable jax_enable_x64 before constructing the kernel')
    if config is None:
        config = SCFConfig(acceleration='rdiis',diis_size=8,max_iter=500,
                           density_tol=1e-8,comm_tol=1e-8)
    if config.project_fn is not None:
        raise ValueError('solve_minimum requires an unconstrained kernel; remove project_fn')
    cfg = config
    args = _fock_kwargs(kernel, fock_build_fn)
    w = np.asarray(kernel.w2d)
    iterations = builds = escapes = 0
    energies = []
    initial = P0
    stability = None
    message = 'Local solve did not converge'
    for cycle in range(max_escapes+1):
        result = solve_scf(kernel,initial,n_electrons,config=cfg,
                           fock_build_fn=fock_build_fn)
        iterations += result.iterations
        # The new RDIIS/Anderson paths count every build. For legacy methods
        # without counters, -1 propagates "unavailable" rather than fabricating
        # a total from iteration counts.
        local_builds = result.history.get('fock_builds', -1)
        if builds >= 0:
            builds = builds + int(local_builds) if local_builds >= 0 else -1
        omega = float(result.history['free_energy'][-1])
        energies.append(omega)
        # A saddle can stall just above a tight requested residual. A nearby
        # Fermi-response Hessian can still propose an escape direction; actual
        # energy descent validates it. Only strict local convergence plus a
        # stable eigenvalue test can produce a successful final result.
        recovery_tol = max(1e-6,cfg.density_tol,cfg.comm_tol)
        residual = max(float(result.history['density_residual'][-1]),
                       float(result.history['commutator_residual'][-1]))
        if not result.converged and residual > recovery_tol:
            stability = None
            break
        stability = stability_analysis(kernel,result.density_matrix,n_electrons,
            stationarity_tol=recovery_tol,eigen_tol=eigen_tol,
            maxiter=stability_maxiter, fock_build_fn=fock_build_fn)
        if builds >= 0:
            builds += 1 + stability.response_builds
        if stability.stable is True and not result.converged:
            message = "No negative curvature found, but the requested residual was not reached"
            break
        if stability.stable is True:
            return MinimumResult(result,stability,True,escapes,iterations,builds,
                                 tuple(energies),'Converged to a locally stable density')
        if stability.stable is None:
            message = 'Stability eigenvalue solve was inconclusive'
            break
        if cycle == max_escapes:
            message = 'Negative curvature remains after the escape budget'
            break
        direction = stability.direction
        norm = np.sqrt(np.sum(w*np.sum(abs(direction)**2,axis=(-2,-1)))/w.sum())
        direction = direction/max(norm,np.finfo(float).tiny)
        best, best_energy = None, omega
        # Projection restores occupation bounds, including saturated bands.
        # Check actual free energies: negative tangent curvature by itself is
        # insufficient to justify a finite, projected perturbation.
        noise = 64*precision*max(1.,abs(omega))
        for amplitude in (.25,.1,.03,.01,.003,.001):
            for sign in (-1.,1.):
                trial = _feasible_density(result.density_matrix+sign*amplitude*direction,w,n_electrons)
                # Judge and retain exactly the density representable by this
                # kernel. A host float64 perturbation must not silently change
                # the precision of a float32 custom Fock builder.
                device_trial = jnp.asarray(trial, dtype=kernel.h.dtype)
                trial = np.asarray(device_trial)
                E = float(_evaluate_fock(device_trial, **args)[1])
                if builds >= 0:
                    builds += 1
                p = np.clip(np.linalg.eigvalsh(trial),0.,1.)
                trial_energy = E+kernel.T*np.sum(w[...,None]*(xlogy(p,p)+xlogy(1-p,1-p)))
                if trial_energy < best_energy-noise:
                    best, best_energy = trial, float(trial_energy)
        if best is None:
            message = 'Negative curvature found, but no finite feasible descent step resolved'
            break
        initial = best
        escapes += 1
    return MinimumResult(result,stability,False,escapes,iterations,builds,tuple(energies),message)
