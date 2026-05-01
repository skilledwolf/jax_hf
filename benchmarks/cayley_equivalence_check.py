"""Numerical equivalence checks for the Cayley line-search optimizations.

Probes:
 (1) Inlined Omega0 vs frozen_F(0.0) — should match to roundoff.
 (3) Spectral Cayley vs LU-based Cayley — should match to roundoff.
"""

import numpy as np
import jax
import jax.numpy as jnp


def _herm(X):
    return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))


def _skew(X):
    return 0.5 * (X - jnp.conj(jnp.swapaxes(X, -1, -2)))


def _cayley_lu(d, tau):
    """Reference LU-based Cayley retraction: U = (I+A)^-1 (I-A) with A = tau*d/2."""
    tau_bc = jnp.asarray(tau, dtype=jnp.real(d).dtype)[..., None, None]
    n = d.shape[-1]
    eye = jnp.eye(n, dtype=d.dtype)
    A = 0.5 * tau_bc * d
    return jnp.linalg.solve(eye + A, eye - A)


def _spectral_cayley_setup(d):
    """Eigendecompose i*d (Hermitian since d is skew-Hermitian)."""
    iA = 1j * d
    iA = _herm(iA)
    lam, V = jnp.linalg.eigh(iA)
    return V, lam.astype(jnp.real(d).dtype)


def _cayley_from_spectrum(V, lam, tau):
    """Compute U(τ) = V · diag((1 + iτλ/2)/(1 − iτλ/2)) · V†."""
    half = jnp.asarray(0.5, dtype=lam.dtype)
    arg = tau * lam * half  # (..., nb), real
    one = jnp.asarray(1.0, dtype=V.dtype)
    iexp = (1j * arg).astype(V.dtype)
    c = (one + iexp) / (one - iexp)  # complex, |c|=1
    return (V * c[..., None, :]) @ jnp.conj(jnp.swapaxes(V, -1, -2))


def main():
    rng = np.random.default_rng(0)
    nk1, nk2, nb = 4, 4, 8

    # Skew-Hermitian d
    d_raw = rng.normal(size=(nk1, nk2, nb, nb)) + 1j * rng.normal(size=(nk1, nk2, nb, nb))
    d = jnp.asarray(_skew(jnp.asarray(d_raw, dtype=jnp.complex64)))

    # Various τ values
    taus = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, -0.2]

    print(f"Cayley LU vs spectral, shape ({nk1},{nk2},{nb},{nb}), complex64")
    print(f"{'tau':>6} {'max|Δ|':>14} {'unitary err':>14}")
    V, lam = _spectral_cayley_setup(d)
    for tau in taus:
        U_lu = _cayley_lu(d, jnp.asarray(tau, dtype=jnp.float32))
        U_sp = _cayley_from_spectrum(V, lam, jnp.asarray(tau, dtype=jnp.float32))
        diff = float(jnp.max(jnp.abs(U_lu - U_sp)))
        # Check unitarity of spectral version
        UUH = U_sp @ jnp.conj(jnp.swapaxes(U_sp, -1, -2))
        eye = jnp.eye(nb, dtype=U_sp.dtype)
        unit_err = float(jnp.max(jnp.abs(UUH - eye)))
        print(f"{tau:>+6.2f} {diff:>14.3e} {unit_err:>14.3e}")

    # Also sanity-check: at tau=0, spectral gives identity exactly?
    U_sp_zero = _cayley_from_spectrum(V, lam, jnp.asarray(0.0, dtype=jnp.float32))
    eye = jnp.eye(nb, dtype=U_sp_zero.dtype)
    print(f"\n||U_spectral(0) - I||_max = {float(jnp.max(jnp.abs(U_sp_zero - eye))):.3e}")


if __name__ == "__main__":
    main()
