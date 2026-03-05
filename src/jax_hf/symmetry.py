"""Symmetry-averaging projection for density / Fock matrices.

Provides ``make_project_fn`` which returns a JAX-traceable callable that
enforces point-group and (optionally) time-reversal symmetry by averaging
a matrix over the supplied symmetry operations.

The returned function has signature ``(A: jax.Array) -> jax.Array`` where
``A`` has shape ``(nk1, nk2, nb, nb)`` and can be used inside
``lax.while_loop`` without issues.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


ProjectFn = Callable[[jax.Array], jax.Array]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flip_k(
    A: jax.Array,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> jax.Array:
    """Negate specified k-axes on the discrete k-grid.

    Parameters
    ----------
    A : jax.Array, shape ``(nk1, nk2, ...)``
    k_convention : ``"flip"`` or ``"mod"``
    flip_axes : which k-axes to negate.  ``(0, 1)`` is the full
        ``k → -k`` flip; ``(0,)`` flips only the first axis
        (``kx → -kx``).
    """
    if k_convention == "flip":
        return jnp.flip(A, axis=flip_axes)
    elif k_convention == "mod":
        nk1, nk2 = A.shape[0], A.shape[1]
        i = (
            (-jnp.arange(nk1, dtype=jnp.int32)) % nk1
            if 0 in flip_axes
            else jnp.arange(nk1, dtype=jnp.int32)
        )
        j = (
            (-jnp.arange(nk2, dtype=jnp.int32)) % nk2
            if 1 in flip_axes
            else jnp.arange(nk2, dtype=jnp.int32)
        )
        return A[i[:, None], j[None, :], ...]
    else:
        raise ValueError(
            f"k_convention must be 'mod' or 'flip', got {k_convention!r}"
        )


def _sum_unitary_conj(A: jax.Array, G: jax.Array) -> jax.Array:
    """Sum ``g_i @ A @ g_i†`` over group elements (unnormalized)."""
    ng = G.shape[0]

    def body(i, acc):
        g = G[i]
        gH = jnp.conj(jnp.swapaxes(g, -1, -2))
        return acc + (g @ A) @ gH

    return lax.fori_loop(0, ng, body, jnp.zeros_like(A))


def _avg_unitary_conj(A: jax.Array, G: jax.Array) -> jax.Array:
    """Average *A* over unitary conjugations by group elements *G*.

    Parameters
    ----------
    A : jax.Array, shape ``(nk1, nk2, nb, nb)``
    G : jax.Array, shape ``(ng, nb, nb)``

    Returns
    -------
    jax.Array
        ``(1/ng) sum_i  G[i] @ A @ G[i]†``
    """
    ng = G.shape[0]
    return _sum_unitary_conj(A, G) / jnp.asarray(float(ng), dtype=A.dtype)


def _avg_combined_group(
    A: jax.Array,
    G_same: jax.Array,
    G_flip: jax.Array,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> jax.Array:
    """Average *A* over a symmetry group with same-k and flipped-k elements.

    Computes::

        (1/N) [ sum_i g_i A(k) g_i†  +  sum_j h_j A(σk) h_j† ]

    where ``σk`` negates the k-axes specified by *flip_axes* and
    N = |G_same| + |G_flip| is the total group order.
    """
    acc = _sum_unitary_conj(A, G_same)
    A_neg = _flip_k(A, k_convention, flip_axes)
    acc = acc + _sum_unitary_conj(A_neg, G_flip)
    N = G_same.shape[0] + G_flip.shape[0]
    return acc / jnp.asarray(float(N), dtype=A.dtype)


def _avg_time_reversal(
    A: jax.Array,
    U: jax.Array,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> jax.Array:
    """Average *A* with its time-reversed partner.

    Time reversal acts as  ``A(k) -> U conj(A(σk)) U†``, where ``σk``
    negates the k-axes specified by *flip_axes*.

    Parameters
    ----------
    A : shape ``(nk1, nk2, nb, nb)``
    U : shape ``(nb, nb)`` — antiunitary part of time reversal.
    k_convention : ``"flip"`` or ``"mod"``
        How to map k -> -k on the grid.
    flip_axes : which k-axes to negate (default both).
    """
    UH = jnp.conj(jnp.swapaxes(U, -1, -2))
    A_neg = _flip_k(A, k_convention, flip_axes)
    A_tr = U @ jnp.conj(A_neg) @ UH
    return 0.5 * (A + A_tr)


# ---------------------------------------------------------------------------
# SVP symmetry group construction
# ---------------------------------------------------------------------------


def make_svp_symmetry_group(
    *,
    identity: jax.Array,
    s1: jax.Array,
    s3: jax.Array,
    v_rotation: jax.Array,
    v3: jax.Array,
    outlier_sv: tuple[int, int] = (+1, +1),
) -> tuple[jax.Array, jax.Array]:
    """Build S₃ group that permutes 3 spin-valley sectors, leaving one fixed.

    The SVP (spin-valley polarized) state has three sectors with equal
    occupation and one "outlier" sector that differs.  This function returns
    the 6-element S₃ group whose conjugation action permutes the three
    equal sectors while leaving the outlier untouched.

    Because valleys K and K' are related by a spatial symmetry (e.g. C₂
    rotation), transpositions that cross valleys require a k → -k flip in
    addition to the band-space unitary.  To obtain a *closed* group, the
    k-action must be a group homomorphism S₃ → Z₂.  The only nontrivial
    such homomorphism is the **sign** (parity) map: all three transpositions
    are assigned k-flip, while the two 3-cycles (even permutations) and the
    identity act at the same k.

    Each transposition has the conditional-swap form::

        T = swap_op · P_swap + I · (I - P_swap)

    where ``P_swap`` projects onto the two sectors being exchanged and
    ``swap_op`` performs the appropriate spin/valley rotation.

    Parameters
    ----------
    identity : jax.Array, shape ``(nb, nb)``
    s1 : jax.Array, shape ``(nb, nb)``
        Spin-flip operator (Pauli-x analogue).
    s3 : jax.Array, shape ``(nb, nb)``
        Spin diagonal operator with eigenvalues ±1.
    v_rotation : jax.Array, shape ``(nb, nb)``
        The valley-exchange unitary: the representation of the spatial
        symmetry (e.g. C₂) that maps valley K → K' while properly rotating
        the orbital basis.  In the simplest single-orbital model this
        reduces to Pauli-x in valley space, but in general it includes an
        orbital rotation that aligns the momentum-space structure of the
        two valleys.
    v3 : jax.Array, shape ``(nb, nb)``
        Valley diagonal operator with eigenvalues ±1.
    outlier_sv : (int, int)
        ``(s3_eigenvalue, v3_eigenvalue)`` of the outlier sector.
        ``(+1,+1)`` = K↑, ``(-1,+1)`` = K↓,
        ``(+1,-1)`` = K'↑, ``(-1,-1)`` = K'↓.

    Returns
    -------
    same_k : jax.Array, shape ``(3, nb, nb)``
        Even permutations acting at the same k-point: ``{I, C₁, C₂}``.
    flip_k : jax.Array, shape ``(3, nb, nb)``
        Odd permutations (transpositions) acting at ``-k``:
        ``{T_AB, T_AC, T_BC}``.
    """
    so = float(outlier_sv[0])
    vo = float(outlier_sv[1])

    s1v = s1 @ v_rotation
    s3v3 = s3 @ v3

    # The three non-outlier sectors, labelled by how they differ from the
    # outlier:
    #   A = (-so, vo)  — opposite spin, same valley
    #   B = (so, -vo)  — same spin, opposite valley
    #   C = (-so, -vo) — opposite spin, opposite valley
    #
    # Transpositions (each swaps two sectors, fixes the other two):
    #   T_AB swaps A↔B  (differ in both spin+valley) → s1·v_rotation
    #   T_AC swaps A↔C  (differ in valley only)      → v_rotation
    #   T_BC swaps B↔C  (differ in spin only)        → s1
    T_AB = s1v @ (identity - so * vo * s3v3) / 2 + (identity + so * vo * s3v3) / 2
    T_AC = v_rotation @ (identity - so * s3) / 2 + (identity + so * s3) / 2
    T_BC = s1 @ (identity - vo * v3) / 2 + (identity + vo * v3) / 2

    # 3-cycles (even permutations → same_k)
    C1 = T_AB @ T_AC
    C2 = T_AC @ T_AB

    same_k = jnp.stack([identity, C1, C2], axis=0)
    flip_k = jnp.stack([T_AB, T_AC, T_BC], axis=0)
    return same_k, flip_k


def make_svp_project_fn(
    *,
    s3: jax.Array,
    v3: jax.Array,
    n_orb: int,
    outlier_sv: tuple[int, int] = (+1, +1),
    k_convention: str = "flip",
    k_flip_axes: tuple[int, ...] = (0,),
) -> ProjectFn:
    """Build a quarter-metal SVP projection that leaves the outlier untouched.

    Unlike the S₃ group average (via ``make_project_fn`` +
    ``make_svp_symmetry_group``), which unavoidably forces k-flip symmetry
    on the outlier block, this projection:

    1. Zeros all off-diagonal flavour blocks (kills coherences).
    2. Leaves the outlier diagonal block **completely untouched**.
    3. Averages the three inactive diagonal blocks with equal weight,
       using the ``kx``-inversion that relates the two valleys.

    The three inactive blocks satisfy at self-consistency::

        P_{same-valley}(k) = P_{other-valley-1}(σk) = P_{other-valley-2}(σk)

    where ``σk`` negates the k-axes given by *k_flip_axes*.

    Parameters
    ----------
    s3 : jax.Array, shape ``(nb, nb)``
        Spin diagonal operator with eigenvalues ±1.
    v3 : jax.Array, shape ``(nb, nb)``
        Valley diagonal operator with eigenvalues ±1.
    n_orb : int
        Number of orbitals per spin-valley sector.
    outlier_sv : (int, int)
        ``(s3_eigenvalue, v3_eigenvalue)`` of the outlier sector.
    k_convention : str
        ``"flip"`` or ``"mod"`` — grid convention for k-negation.
    k_flip_axes : tuple of int
        Which k-grid axes to negate for the valley exchange.
        ``(0,)`` for ``kx → -kx`` (continuum graphene convention).

    Returns
    -------
    ProjectFn
    """
    s3_np = np.asarray(s3)
    v3_np = np.asarray(v3)
    nb = s3_np.shape[0]
    n_blocks = nb // n_orb
    so, vo = float(outlier_sv[0]), float(outlier_sv[1])

    # Classify blocks by their (spin, valley) eigenvalues.
    idx_outlier = None
    idx_same_v = None       # same valley as outlier, opposite spin
    idx_other_v: list[int] = []  # other-valley blocks

    for i in range(n_blocks):
        s_val = float(np.sign(np.real(s3_np[i * n_orb, i * n_orb])))
        v_val = float(np.sign(np.real(v3_np[i * n_orb, i * n_orb])))
        if s_val == so and v_val == vo:
            idx_outlier = i
        elif v_val == vo:
            idx_same_v = i
        else:
            idx_other_v.append(i)

    if idx_outlier is None or idx_same_v is None or len(idx_other_v) != 2:
        raise ValueError(
            f"Could not identify 4 spin-valley blocks from s3/v3 "
            f"(n_orb={n_orb}, nb={nb}, outlier_sv={outlier_sv})"
        )

    def _sl(i: int) -> slice:
        return slice(i * n_orb, (i + 1) * n_orb)

    sl_same = _sl(idx_same_v)
    sl_ov0 = _sl(idx_other_v[0])
    sl_ov1 = _sl(idx_other_v[1])

    # Block-diagonal mask: 1 on diagonal flavour blocks, 0 elsewhere.
    mask_np = np.zeros((nb, nb), dtype=np.float32)
    for i in range(n_blocks):
        a, b = i * n_orb, (i + 1) * n_orb
        mask_np[a:b, a:b] = 1.0
    mask = jnp.asarray(mask_np)

    k_conv = str(k_convention)
    k_axes = tuple(k_flip_axes)

    def project(P: jax.Array) -> jax.Array:
        # 1. Zero off-diagonal flavour blocks.
        out = P * mask

        # 2. Compute the inactive-block average Q(k).
        #    P_same_v lives at the same k as the K valley;
        #    the two other-valley blocks live at kx-flipped k.
        P_same = P[..., sl_same, sl_same]
        P_ov0_flip = _flip_k(P[..., sl_ov0, sl_ov0], k_conv, k_axes)
        P_ov1_flip = _flip_k(P[..., sl_ov1, sl_ov1], k_conv, k_axes)

        Q = (P_same + P_ov0_flip + P_ov1_flip) / 3.0
        Q_flip = _flip_k(Q, k_conv, k_axes)

        # 3. Write the averaged blocks (outlier block untouched).
        out = out.at[..., sl_same, sl_same].set(Q)
        out = out.at[..., sl_ov0, sl_ov0].set(Q_flip)
        out = out.at[..., sl_ov1, sl_ov1].set(Q_flip)

        return out

    return project


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def make_project_fn(
    *,
    unitary_group: jax.Array | None = None,
    spatial_group: jax.Array | None = None,
    spatial_k_convention: str = "mod",
    spatial_k_flip_axes: tuple[int, ...] = (0, 1),
    time_reversal_U: jax.Array | None = None,
    time_reversal_k_convention: str = "mod",
    time_reversal_k_flip_axes: tuple[int, ...] = (0, 1),
) -> ProjectFn:
    """Build a symmetry-averaging projection function.

    Parameters
    ----------
    unitary_group : jax.Array, shape ``(ng, nb, nb)``, optional
        Stack of unitary matrices forming a group (or the same-k subgroup
        of a larger group).  The projection averages ``A`` over all
        conjugations ``g A g†``.
    spatial_group : jax.Array, shape ``(ns, nb, nb)``, optional
        Stack of unitary matrices whose conjugation acts on ``A(σk)``
        instead of ``A(k)``, where ``σk`` negates the k-axes given by
        *spatial_k_flip_axes*.  These represent symmetry operations that
        include a spatial transformation (e.g. ``kx → -kx`` for a valley
        exchange, or full ``k → -k`` for C₂ rotation) combined with a
        band-space unitary.

        When provided together with ``unitary_group``, the two sets form
        a single group and are averaged jointly::

            (1/N) [Σ g·A(k)·g† + Σ h·A(σk)·h†],  N = ng + ns.

        If ``spatial_group`` is given without ``unitary_group``, the
        identity is automatically included in the same-k part.
    spatial_k_convention : str
        ``"mod"`` (default) or ``"flip"`` — grid convention for the
        spatial group k-negation.
    spatial_k_flip_axes : tuple of int
        Which k-grid axes to negate for the spatial group.
        ``(0, 1)`` (default) is the full ``k → -k`` flip (e.g. C₂);
        ``(0,)`` flips only the first axis (``kx → -kx``), appropriate
        when valleys are related by ``kx``-inversion (as in continuum
        graphene models where ``H_{K'}(kx, ky) = H_K(-kx, ky)``).
    time_reversal_U : jax.Array, shape ``(nb, nb)``, optional
        Antiunitary part of time reversal.  When provided the projection
        also averages ``A(k)`` with ``U conj(A(σk)) U†``.  Applied
        *after* the unitary/spatial group averaging.
    time_reversal_k_convention : str
        ``"mod"`` (default) or ``"flip"`` — grid convention for the
        time-reversal k-negation.
    time_reversal_k_flip_axes : tuple of int
        Which k-grid axes to negate for time reversal.  Default
        ``(0, 1)`` is the full ``k → -k`` flip.

    Returns
    -------
    ProjectFn
        ``(A: jax.Array) -> jax.Array`` that can be used inside JIT /
        ``lax.while_loop``.
    """
    has_group = unitary_group is not None or spatial_group is not None
    if not has_group and time_reversal_U is None:
        return lambda A: A

    G = None if unitary_group is None else jnp.asarray(unitary_group)
    S = None if spatial_group is None else jnp.asarray(spatial_group)
    U = None if time_reversal_U is None else jnp.asarray(time_reversal_U)
    s_k_conv = str(spatial_k_convention)
    s_k_axes = tuple(spatial_k_flip_axes)
    t_k_conv = str(time_reversal_k_convention)
    t_k_axes = tuple(time_reversal_k_flip_axes)

    def project(A: jax.Array) -> jax.Array:
        out = A
        if G is not None and S is not None:
            out = _avg_combined_group(out, G, S, s_k_conv, s_k_axes)
        elif G is not None:
            out = _avg_unitary_conj(out, G)
        elif S is not None:
            # Only spatial elements provided — add identity for same-k part.
            I_mat = jnp.eye(S.shape[-1], dtype=S.dtype)[None]
            out = _avg_combined_group(out, I_mat, S, s_k_conv, s_k_axes)
        if U is not None:
            out = _avg_time_reversal(out, U, t_k_conv, t_k_axes)
        return out

    return project
