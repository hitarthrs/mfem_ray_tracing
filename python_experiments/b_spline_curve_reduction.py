"""
B-spline curve degree reduction (Piegl & Tiller, *The NURBS Book*, Sec. 5.6 / Alg. A5.11).

Pipeline:
  1. Scan the input knot vector U; at each distinct knot, extract the current span
     as a degree-p Bézier patch ``bpts`` (via knot insertion, Alg. A5.4, when needed).
  2. Degree-reduce that patch with ``bezier_degree_reduce`` (Alg. A5.8) -> ``rbpts``.
  3. If the previous knot required partial insertion (``old_r > 0``), remove the knot
     U[a] ``old_r`` times (same algebra as knot removal, Sec. 5.4) and bound the error.
  4. Append reduced controls into ``Pw``, build the reduced knot vector ``Uh``, and
     advance to the next span.

Related code: ``bezier_curve_degree_reduction.py`` (A5.8), ``element_extractor.cpp`` (A5.4).

Indexing note: several loops use book-style expressions ``Pw[i-1]``, ``Uh[i-1]`` where
``i`` is a running index in the algorithm — equivalent to the pseudocode, not off-by-one
fixes on top of 0-based NumPy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bezier_curve_degree_reduction import bezier_degree_reduce


def distance4d(p: np.ndarray, q: np.ndarray) -> float:
    """Euclidean norm of ``p - q`` (book: Distance4D).

    For rational NURBS use homogeneous (x, y, z, w); for polynomial B-splines, ``dim``
    is typically 2 or 3.
    """
    diff = np.asarray(p, dtype=float) - np.asarray(q, dtype=float)
    return float(np.linalg.norm(diff))


def DegreeReduceCurve(
    n_control_points: int,
    degree: int,
    U: np.array,
    Qw: np.array,
    tol: float = float("inf"),
):
    """
    Reduce a B-spline curve of degree ``degree`` (p) to degree ``degree - 1`` (ph).

    Parameters
    ----------
    n_control_points : int
        Number of input control points n (not counting phantom book padding).
    degree : int
        Input degree p (>= 2). Local buffers are sized as in A5.11:
        ``bpts[p+1]``, ``Nextbpts[p-1]``, ``rbpts[p]``, ``alphas[p-1]``, ``e[m]``.
    U : np.ndarray
        Knot vector of length ``n + p + 1`` (standard open knot count).
        Book ``m`` is the **index** of the last knot, ``m = len(U) - 1``.
    Qw : np.ndarray
        Input controls, shape (n, dim).
    tol : float
        Knot-removal / reduction tolerance (book: TOL). ``inf`` disables early exit.

    Returns
    -------
    (Pw, Uh, error_array) on success, or ``1`` if tolerance is exceeded.

    ``Pw`` and ``Uh`` are trimmed to the active lengths written by A5.11 (``cind``,
    ``kind``); no trailing zero padding.
    """
    dim = Qw.shape[1]
    p = int(degree)
    if p < 2:
        raise ValueError("DegreeReduceCurve requires input degree p >= 2")

    # --- Alg. A5.11 initialization (reduced degree ph = p - 1) -----------------
    ph = p - 1
    mh = ph  # last index in Uh that has been written; grows when knots are inserted

    kind = ph + 1  # next free slot in Uh for a new knot value
    r = -1  # insertion count from the *previous* knot event (-1 = none yet)
    a = p  # left knot index of the active span [U[a], U[b]]
    b = p + 1  # scan index: next knot cluster to process

    cind = 1  # next row in Pw to fill (row 0 is the fixed start point)
    mult = p  # multiplicity of knot at U[b] (updated each outer iteration)

    n = int(n_control_points)
    U = np.asarray(U, dtype=float)
    if U.shape[0] != n + p + 1:
        raise ValueError(
            f"len(U) must be n + p + 1 = {n + p + 1}, got {U.shape[0]}"
        )
    # Book m = index of last knot (not knot count); enables U[b] == U[b+1] scan safely
    m = int(U.shape[0] - 1)

    # Book indexes Qw[b-p+i] while b runs to m-1; pad tail so 0-based loads stay in-bounds
    Qw_work = np.zeros((m + 1, dim), dtype=float)
    Qw_work[:n] = np.asarray(Qw, dtype=float)
    if n > 0:
        Qw_work[n : m + 1] = Qw_work[n - 1]

    # Output buffers (book grows Pw / Uh during the scan; size for worst-case indices)
    Pw = np.zeros((n + p, dim), dtype=float)
    Pw[0] = Qw_work[0]

    uh_len = m + ph + 2
    Uh = np.zeros(uh_len, dtype=float)
    Uh[0 : ph + 1] = U[0]  # left end: ph+1 copies of U[0]

    # Local arrays (Piegl & Tiller A5.11): fixed sizes in terms of input degree p
    bpts = np.zeros((p + 1, dim), dtype=float)  # bpts[p+1]
    bpts[:] = Qw_work[: p + 1]

    next_bpts = np.zeros((p - 1, dim), dtype=float)  # Nextbpts[p-1]; use rows 0..r-1

    # rbpts[p]: p reduced CPs; one extra row so rbpts[kj+1] in knot removal stays in-bounds
    rbpts = np.zeros((p + 1, dim), dtype=float)

    alphas = np.zeros(p - 1, dtype=float)  # alphas[p-1]; active length is p-mult

    error_array = np.zeros(m + 1, dtype=float)  # e[0..m] in the book

    # --- Main loop over interior knots (same driver as knot insertion, A5.4) ----
    while b < m:
        i = b

        # Extend b across all knots equal to U[i] (cluster multiplicity)
        while b < m and U[b] == U[b + 1]:
            b += 1

        mult = b - i + 1
        mh += mult - 1  # reduced knot vector gains (mult-1) entries for this cluster

        old_r = r
        r = p - mult  # insertions needed so this span is a full Bézier; r <= p-1

        # lbz: first Bézier control index to copy from rbpts into Pw after reduction
        if old_r > 0:
            lbz = (old_r + 2) // 2
        else:
            lbz = 1

        # --- Knot insertion (A5.4) when mult < p: refine toward a Bézier piece ---
        if r > 0:
            numer = U[b] - U[a]

            # alphas[k-mult-1] for k = p, p-1, ..., mult+1 (at most p-1 entries).
            # NOTE: the loop must stop at k = mult+1 (book A5.6/A5.11); running it
            # through k = mult writes alphas[-1], which numpy wraps around to
            # alphas[p-2] and silently corrupts the insertion for interior knots
            # of multiplicity 1.
            for k in range(p, mult, -1):
                alphas[k - mult - 1] = numer / (U[a + k] - U[a])

            for j in range(1, r + 1):
                save = r - j
                s = mult + j
                for k in range(p, s - 1, -1):
                    alpha = alphas[k - s]
                    bpts[k] = alpha * bpts[k] + (1.0 - alpha) * bpts[k - 1]
                next_bpts[save] = bpts[p].copy()

        # --- Degree reduction of the current Bézier piece (A5.8) ----------------
        reduced, max_err = bezier_degree_reduce(bpts)
        rbpts[:p] = reduced
        error_array[a] += max_err
        if error_array[a] > tol:
            return 1

        # --- Remove knot U[a] old_r times (knot-removal core from A5.10) -------
        if old_r > 0:
            first = kind
            last = kind

            for k in range(0, old_r):
                i = first
                j = last
                kj = j - kind  # index into rbpts for the sliding removal window

                # Knot removal (A5.10): i++, j-- so j-i shrinks; both i++ was non-terminating
                while j - i > k:
                    alfa = (U[a] - Uh[i - 1]) / (U[b] - Uh[i - 1])
                    beta = (U[a] - Uh[j - k - 1]) / (U[b] - Uh[j - k - 1])
                    Pw[i - 1] = (Pw[i - 1] - (1.0 - alfa) * Pw[i - 2]) / alfa
                    rbpts[kj] = (rbpts[kj] - beta * rbpts[kj + 1]) / (1.0 - beta)
                    i += 1
                    j -= 1
                    kj -= 1

                # Knot-removal error bound Br at this removal pass
                if j - i < k:
                    Br = distance4d(Pw[i - 2], rbpts[kj + 1])
                else:
                    delta = (U[a] - Uh[i - 1]) / (U[b] - Uh[i - 1])
                    A = delta * rbpts[kj + 1] + (1.0 - delta) * Pw[i - 2]
                    Br = distance4d(Pw[i - 1], A)

                # Spread Br into e[L..a] (book uses K, q, L; here ii runs L..a)
                K = a + old_r - k
                q = (2 * p - k + 1) // 2
                L = int(K - q)
                for ii in range(L, a + 1):
                    error_array[ii] += Br
                    if error_array[ii] > tol:
                        return 1

                first -= 1
                last += 1

            cind = i - 1

        # --- Write knot value U[a] into Uh (interior spans only) ----------------
        if a != p:
            for _ in range(int(ph - old_r)):
                Uh[kind] = U[a]
                kind += 1

        # --- Copy reduced Bézier controls rbpts[lbz..ph] into Pw ---------------
        for i in range(lbz, ph + 1):
            Pw[cind] = rbpts[i]
            cind += 1

        # --- Advance to next span: reload bpts, update (a, b) -------------------
        if b < m:
            if r > 0:
                for i in range(0, r):
                    bpts[i] = next_bpts[i]
            for i in range(r, p + 1):
                bpts[i] = Qw_work[b - p + i]
            a = b
            b += 1
        else:
            # Right end of knot vector: clamp Uh with U[b]
            for i in range(0, ph + 1):
                Uh[kind + i] = U[b]

    # A5.11 uses fixed buffers; cind / kind are the active lengths (book counters).
    n_active = cind
    uh_active = kind + ph + 1
    return Pw[:n_active].copy(), Uh[:uh_active].copy(), error_array





