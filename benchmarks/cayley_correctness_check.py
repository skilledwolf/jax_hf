"""End-to-end correctness check: spectral diag matches LU-based diag.

Compares :func:`_diag_UFU_from_spectrum` against the reference computation
``diag(U(τ)† Ft U(τ))`` where U is built via the LU-based Cayley.  Tests on
Hermitian Ft and skew-Hermitian d_Q matrices with shapes that match the
solver's actual carry.
"""

import numpy as np
import jax
import jax.numpy as jnp

from jax_hf.solver import (
    _cayley_retract,
    _cayley_spectral_setup,
    _cayley_unitary_from_spectrum,
    _diag_UFU_from_spectrum,
)


def main():
    rng = np.random.default_rng(0)
    nk1, nk2, nb = 4, 4, 8
    dtype = jnp.complex64

    # Skew-Hermitian d_Q
    d_raw = rng.normal(size=(nk1, nk2, nb, nb)) + 1j * rng.normal(size=(nk1, nk2, nb, nb))
    d_raw = 0.5 * (d_raw - np.conj(d_raw.swapaxes(-1, -2)))
    d_Q = jnp.asarray(d_raw, dtype=dtype)

    # Hermitian Ft
    F_raw = rng.normal(size=(nk1, nk2, nb, nb)) + 1j * rng.normal(size=(nk1, nk2, nb, nb))
    F_raw = 0.5 * (F_raw + np.conj(F_raw.swapaxes(-1, -2)))
    Ft = jnp.asarray(F_raw, dtype=dtype)

    V_d, lam_d = _cayley_spectral_setup(d_Q)
    Ft_eig = jnp.conj(jnp.swapaxes(V_d, -2, -1)) @ Ft @ V_d

    print(f"Spectral diag(U†FtU) vs LU-based, shape ({nk1},{nk2},{nb},{nb}), {dtype}")
    print(f"{'tau':>6} {'max|Δ diag|':>15} {'max|Δ U|':>15} {'unitary err':>14}")
    for tau_val in [0.0, 0.1, 0.3, 0.5, 1.0, -0.5]:
        tau = jnp.asarray(tau_val, dtype=jnp.float32)

        # Spectral diag
        diag_sp = _diag_UFU_from_spectrum(V_d, Ft_eig, lam_d, tau)

        # Reference: build U via LU, compute U†FtU, take diag
        U_lu = _cayley_retract(d_Q, tau)
        Ft_trial_lu = jnp.conj(jnp.swapaxes(U_lu, -2, -1)) @ Ft @ U_lu
        diag_lu = jnp.real(jnp.diagonal(Ft_trial_lu, axis1=-2, axis2=-1))

        diff_diag = float(jnp.max(jnp.abs(diag_sp - diag_lu)))

        # Compare U_spectral vs U_lu
        U_sp = _cayley_unitary_from_spectrum(V_d, lam_d, tau)
        diff_U = float(jnp.max(jnp.abs(U_sp - U_lu)))

        # Spectral unitarity check
        UUH = U_sp @ jnp.conj(jnp.swapaxes(U_sp, -1, -2))
        eye = jnp.eye(nb, dtype=dtype)
        unit_err = float(jnp.max(jnp.abs(UUH - eye)))

        print(f"{tau_val:>+6.2f} {diff_diag:>15.3e} {diff_U:>15.3e} {unit_err:>14.3e}")


if __name__ == "__main__":
    main()
