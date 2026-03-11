"""Benchmark: block-diagonal QR operations (loop vs batched).

Compares three approaches for the block-diagonal operations in the QR solver:
  1. Dense  – ignore block structure, full n×n matmuls
  2. Loop   – current code: Python loop over block slices with .at[].set()
  3. Batched – reshape into (n_blocks, bs, bs), single batched QR/einsum
"""

import functools
import time

import jax
import jax.numpy as jnp

from jax_hf.variational_qr import (
    _apply_orbital_basis_update,
    _apply_right_unitary,
    _block_slices_from_sizes,
    _qr_retraction_unitary,
)


# ---------------------------------------------------------------------------
# Batched implementations for equal-sized blocks
# ---------------------------------------------------------------------------


def _extract_diag_blocks(M, bs):
    """Extract diagonal blocks: (..., n, n) -> (..., n_blocks, bs, bs)."""
    n = M.shape[-1]
    nb = n // bs
    return jnp.stack(
        [M[..., i * bs : (i + 1) * bs, i * bs : (i + 1) * bs] for i in range(nb)],
        axis=-3,
    )


def _place_diag_blocks(blocks, n):
    """Place diagonal blocks: (..., n_blocks, bs, bs) -> (..., n, n)."""
    bs = blocks.shape[-1]
    nb = blocks.shape[-3]
    batch = blocks.shape[:-3]
    out = jnp.zeros(batch + (n, n), dtype=blocks.dtype)
    for i in range(nb):
        out = out.at[..., i * bs : (i + 1) * bs, i * bs : (i + 1) * bs].set(
            blocks[..., i, :, :]
        )
    return out


def _qr_retraction_batched(G, tau, *, block_size, n):
    """Batched QR retraction for equal-sized diagonal blocks."""
    tiny = jnp.asarray(1e-30, dtype=jnp.real(G).dtype)
    tau_bc = jnp.asarray(tau, dtype=jnp.real(G).dtype)[..., None, None]
    bs = block_size

    G_blocks = _extract_diag_blocks(G, bs)  # (..., n_blocks, bs, bs)
    trial = jnp.eye(bs, dtype=G.dtype) - tau_bc[..., None, :, :] * G_blocks
    U_blocks, R_blocks = jnp.linalg.qr(trial)

    phases = jnp.diagonal(R_blocks, axis1=-2, axis2=-1)
    phase_norm = jnp.where(
        jnp.abs(phases) > tiny,
        phases / jnp.abs(phases),
        jnp.ones_like(phases),
    )
    U_blocks = U_blocks * phase_norm[..., None, :]

    return _place_diag_blocks(U_blocks, n)


def _orbital_basis_update_batched(Ft, U, *, block_size, n):
    """Batched U†FU for block-diagonal U using einsum."""
    bs = block_size
    nb = n // bs
    batch = Ft.shape[:-2]

    U_blocks = _extract_diag_blocks(U, bs)  # (..., nb, bs, bs)
    U_dag_blocks = U_blocks.conj().swapaxes(-1, -2)

    # Reshape Ft into block form: (..., nb, bs, nb, bs) -> (..., nb, nb, bs, bs)
    Ft_blocks = Ft.reshape(batch + (nb, bs, nb, bs))
    Ft_blocks = jnp.moveaxis(Ft_blocks, -2, -3)

    # U†[i] @ Ft[i,j] @ U[j]  via two einsums
    left = jnp.einsum("...iab,...ijbc->...ijac", U_dag_blocks, Ft_blocks)
    result = jnp.einsum("...ijab,...jbc->...ijac", left, U_blocks)

    # Reshape back: (..., nb, nb, bs, bs) -> (..., nb, bs, nb, bs) -> (..., n, n)
    result = jnp.moveaxis(result, -3, -2)
    return result.reshape(batch + (n, n))


def _right_unitary_batched(X, U, *, block_size, n):
    """Batched X@U for block-diagonal U using reshape + matmul."""
    bs = block_size
    nb = n // bs
    batch = X.shape[:-2]
    m = X.shape[-2]

    U_blocks = _extract_diag_blocks(U, bs)  # (..., nb, bs, bs)

    # Reshape X columns into blocks: (..., m, n) -> (..., m, nb, bs) -> (..., nb, m, bs)
    X_by_block = X.reshape(batch + (m, nb, bs))
    X_by_block = jnp.moveaxis(X_by_block, -2, -3)

    # Batched matmul: (..., nb, m, bs) @ (..., nb, bs, bs) -> (..., nb, m, bs)
    out = X_by_block @ U_blocks

    # Reshape back: (..., nb, m, bs) -> (..., m, nb, bs) -> (..., m, n)
    out = jnp.moveaxis(out, -3, -2)
    return out.reshape(batch + (m, n))


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def bench(fn, args, n_warmup=5, n_runs=200):
    """Time a pre-jitted function. Returns (mean_seconds, result)."""
    for _ in range(n_warmup):
        r = fn(*args)
        jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        r = fn(*args)
        jax.block_until_ready(r)
    return (time.perf_counter() - t0) / n_runs, r


def make_test_data(nk1, nk2, nb, bs, dtype=jnp.complex64):
    n_blocks = nb // bs
    key = jax.random.PRNGKey(42)

    # Block-diagonal skew-Hermitian generator
    G_raw = (
        jax.random.normal(key, (nk1, nk2, nb, nb))
        + 1j * jax.random.normal(jax.random.fold_in(key, 1), (nk1, nk2, nb, nb))
    ).astype(dtype)
    G = jnp.zeros_like(G_raw)
    for i in range(n_blocks):
        s = slice(i * bs, (i + 1) * bs)
        blk = G_raw[..., s, s]
        blk = 0.5 * (blk - blk.conj().swapaxes(-1, -2))
        G = G.at[..., s, s].set(blk)

    tau = jnp.full((nk1, nk2), 0.2, dtype=jnp.real(G).dtype)

    # Full Hermitian Ft (not block-diagonal)
    Ft_raw = (
        jax.random.normal(jax.random.fold_in(key, 2), (nk1, nk2, nb, nb))
        + 1j
        * jax.random.normal(jax.random.fold_in(key, 3), (nk1, nk2, nb, nb))
    ).astype(dtype)
    Ft = 0.5 * (Ft_raw + Ft_raw.conj().swapaxes(-1, -2))

    # Non-square X for right-unitary test
    m = nb + 4
    X = (
        jax.random.normal(jax.random.fold_in(key, 4), (nk1, nk2, m, nb))
        + 1j * jax.random.normal(jax.random.fold_in(key, 5), (nk1, nk2, m, nb))
    ).astype(dtype)

    return G, tau, Ft, X


def run():
    configs = [
        (16, 16, 4, 2),
        (64, 64, 4, 2),
        (128, 128, 4, 2),
        (64, 64, 8, 2),
        (64, 64, 8, 4),
        (64, 64, 12, 2),
        (64, 64, 12, 4),
        (64, 64, 16, 4),
    ]

    for nk1, nk2, nb, bs in configs:
        n_blocks = nb // bs
        block_sizes = tuple([bs] * n_blocks)
        block_slices = _block_slices_from_sizes(block_sizes, nb)

        G, tau, Ft, X = make_test_data(nk1, nk2, nb, bs)

        print(f"\n{'=' * 70}")
        print(
            f"({nk1}x{nk2}) x ({nb}x{nb}),  block_size={bs},  n_blocks={n_blocks}"
        )
        print("=" * 70)

        # --- QR retraction ---
        qr_dense = jax.jit(lambda G, tau: _qr_retraction_unitary(G, tau))
        qr_loop = jax.jit(
            lambda G, tau, _bs=block_slices: _qr_retraction_unitary(
                G, tau, block_slices=_bs
            )
        )
        qr_batch = jax.jit(
            lambda G, tau, _s=bs, _n=nb: _qr_retraction_batched(
                G, tau, block_size=_s, n=_n
            )
        )

        t_d, U_d = bench(qr_dense, (G, tau))
        t_l, U_l = bench(qr_loop, (G, tau))
        t_b, U_b = bench(qr_batch, (G, tau))

        ok_l = jnp.allclose(U_d, U_l, atol=1e-5)
        ok_b = jnp.allclose(U_d, U_b, atol=1e-5)

        print(f"\n  QR retraction:")
        print(f"    Dense:   {t_d*1e3:7.3f} ms")
        print(f"    Loop:    {t_l*1e3:7.3f} ms  ({t_d/t_l:.2f}x vs dense)  match={ok_l}")
        print(f"    Batched: {t_b*1e3:7.3f} ms  ({t_d/t_b:.2f}x vs dense)  match={ok_b}")
        print(f"    Batched vs Loop: {t_l/t_b:.2f}x")

        # --- Orbital basis update (U†FU) ---
        U = U_l  # use loop result as the unitary
        obu_dense = jax.jit(lambda Ft, U: _apply_orbital_basis_update(Ft, U))
        obu_loop = jax.jit(
            lambda Ft, U, _bs=block_slices: _apply_orbital_basis_update(
                Ft, U, block_slices=_bs
            )
        )
        obu_batch = jax.jit(
            lambda Ft, U, _s=bs, _n=nb: _orbital_basis_update_batched(
                Ft, U, block_size=_s, n=_n
            )
        )

        t_d2, Ft_d = bench(obu_dense, (Ft, U))
        t_l2, Ft_l = bench(obu_loop, (Ft, U))
        t_b2, Ft_b = bench(obu_batch, (Ft, U))

        ok_l2 = jnp.allclose(Ft_d, Ft_l, atol=1e-5)
        ok_b2 = jnp.allclose(Ft_d, Ft_b, atol=1e-5)

        print(f"\n  Orbital basis update (U†FU):")
        print(f"    Dense:   {t_d2*1e3:7.3f} ms")
        print(f"    Loop:    {t_l2*1e3:7.3f} ms  ({t_d2/t_l2:.2f}x vs dense)  match={ok_l2}")
        print(f"    Batched: {t_b2*1e3:7.3f} ms  ({t_d2/t_b2:.2f}x vs dense)  match={ok_b2}")
        print(f"    Batched vs Loop: {t_l2/t_b2:.2f}x")

        # --- Right unitary (X@U) ---
        ru_dense = jax.jit(lambda X, U: _apply_right_unitary(X, U))
        ru_loop = jax.jit(
            lambda X, U, _bs=block_slices: _apply_right_unitary(
                X, U, block_slices=_bs
            )
        )
        ru_batch = jax.jit(
            lambda X, U, _s=bs, _n=nb: _right_unitary_batched(
                X, U, block_size=_s, n=_n
            )
        )

        t_d3, X_d = bench(ru_dense, (X, U))
        t_l3, X_l = bench(ru_loop, (X, U))
        t_b3, X_b = bench(ru_batch, (X, U))

        ok_l3 = jnp.allclose(X_d, X_l, atol=1e-5)
        ok_b3 = jnp.allclose(X_d, X_b, atol=1e-5)

        print(f"\n  Right unitary (X@U):")
        print(f"    Dense:   {t_d3*1e3:7.3f} ms")
        print(f"    Loop:    {t_l3*1e3:7.3f} ms  ({t_d3/t_l3:.2f}x vs dense)  match={ok_l3}")
        print(f"    Batched: {t_b3*1e3:7.3f} ms  ({t_d3/t_b3:.2f}x vs dense)  match={ok_b3}")
        print(f"    Batched vs Loop: {t_l3/t_b3:.2f}x")


if __name__ == "__main__":
    run()
