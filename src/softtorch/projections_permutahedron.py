"""Permutahedron projection algorithms with custom autograd backwards."""

import warnings
from typing import Literal

import numpy as np
import torch

from softtorch.utils import _validate_softness


try:
    from numba import njit
except ImportError:
    warnings.warn(
        "Numba could not be imported. PAV isotonic regression will fall back to "
        "pure Python and run significantly slower. Check your Numba/LLVM installation.",
        stacklevel=2,
    )

    def njit(func):
        return func


# ── Vendored isotonic solvers ─────────────────────────────────────────
# _isotonic_l2, _log_add_exp, _isotonic_kl are from:
#   google-research/fast-soft-sort
#   https://github.com/google-research/fast-soft-sort
# Modifications: renamed, reformatted.
#
# Copyright 2007-2020 The scikit-learn developers.
# Copyright 2020 Google LLC.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


@njit
def _isotonic_l2(y, sol):
    """PAV for monotone decreasing L2 isotonic regression.

    Solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
    """
    n = y.shape[0]
    target = np.arange(n)
    c = np.ones(n)
    sums = np.zeros(n)

    for i in range(n):
        sol[i] = y[i]
        sums[i] = y[i]

    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if sol[i] > sol[k]:
            i = k
            continue
        sum_y = sums[i]
        sum_c = c[i]
        while True:
            prev_y = sol[k]
            sum_y += sums[k]
            sum_c += c[k]
            k = target[k] + 1
            if k == n or prev_y > sol[k]:
                sol[i] = sum_y / sum_c
                sums[i] = sum_y
                c[i] = sum_c
                target[i] = k - 1
                target[k - 1] = i
                if i > 0:
                    i = target[i - 1]
                break

    i = 0
    while i < n:
        k = target[i] + 1
        sol[i + 1 : k] = sol[i]
        i = k


@njit
def _log_add_exp(x, y):
    larger = max(x, y)
    smaller = min(x, y)
    return larger + np.log1p(np.exp(smaller - larger))


@njit
def _isotonic_kl(y, w, sol):
    """PAV for monotone decreasing KL isotonic regression.

    Solves argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v>.
    """
    n = y.shape[0]
    target = np.arange(n)
    lse_y_ = np.zeros(n)
    lse_w_ = np.zeros(n)

    for i in range(n):
        sol[i] = y[i] - w[i]
        lse_y_[i] = y[i]
        lse_w_[i] = w[i]

    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if sol[i] > sol[k]:
            i = k
            continue
        lse_y = lse_y_[i]
        lse_w = lse_w_[i]
        while True:
            prev_y = sol[k]
            lse_y = _log_add_exp(lse_y, lse_y_[k])
            lse_w = _log_add_exp(lse_w, lse_w_[k])
            k = target[k] + 1
            if k == n or prev_y > sol[k]:
                sol[i] = lse_y - lse_w
                lse_y_[i] = lse_y
                lse_w_[i] = lse_w
                target[i] = k - 1
                target[k - 1] = i
                if i > 0:
                    i = target[i - 1]
                break

    i = 0
    while i < n:
        k = target[i] + 1
        sol[i + 1 : k] = sol[i]
        i = k


# ── p-norm isotonic solvers (q=3 and q=4) ────────────────────────────
# These extend the fast-soft-sort approach to higher-order p-norm
# regularization. Original to SoftTorch.


@njit
def _solve_block_q3(s, prefix_s, prefix_s2, start, length, sum_w):
    """Analytical solver for q=3 block: sum_block (gamma - s)|gamma - s| + sum_w = 0."""
    end = start + length
    eps = 2.2e-12  # ~float64 eps^0.75

    for k in range(start, end + 1):
        n_hi = float(k - start)
        n_lo = float(end - k)
        S_hi = prefix_s[k] - prefix_s[start]
        S_lo = prefix_s[end] - prefix_s[k]
        M2_hi = prefix_s2[k] - prefix_s2[start]
        M2_lo = prefix_s2[end] - prefix_s2[k]

        a = n_lo - n_hi
        b = 2.0 * (S_hi - S_lo)
        c = (M2_lo - M2_hi) + sum_w

        if a != 0.0:
            disc = b * b - 4.0 * a * c
            if disc < -eps:
                continue
            gamma = (-b + np.sqrt(max(disc, 0.0))) / (2.0 * a)
        elif b != 0.0:
            gamma = -c / b
        else:
            continue

        if k < end:
            lo_bound = s[k]
        else:
            lo_bound = -np.inf
        if k > start:
            hi_bound = s[k - 1]
        else:
            hi_bound = np.inf

        if gamma > lo_bound - eps and gamma <= hi_bound + eps:
            return gamma

    return (s[start] + s[min(end - 1, len(s) - 1)]) * 0.5


@njit
def _isotonic_pnorm_q3(s, w, sol):
    """PAV for q=3 (p=3/2) isotonic regression."""
    n = s.shape[0]

    prefix_s = np.zeros(n + 1)
    prefix_s2 = np.zeros(n + 1)
    for i in range(n):
        prefix_s[i + 1] = prefix_s[i] + s[i]
        prefix_s2[i + 1] = prefix_s2[i] + s[i] * s[i]

    starts = np.zeros(n, dtype=np.int64)
    lens = np.zeros(n, dtype=np.int64)
    sumw = np.zeros(n)
    gam = np.zeros(n)
    top = 0

    for i in range(n):
        starts[top] = i
        lens[top] = 1
        sumw[top] = w[i]
        gam[top] = _solve_block_q3(s, prefix_s, prefix_s2, i, 1, w[i])
        top += 1

        while top >= 2 and gam[top - 2] < gam[top - 1]:
            lens_new = lens[top - 2] + lens[top - 1]
            sumw_new = sumw[top - 2] + sumw[top - 1]
            gam[top - 2] = _solve_block_q3(
                s, prefix_s, prefix_s2, starts[top - 2], lens_new, sumw_new
            )
            lens[top - 2] = lens_new
            sumw[top - 2] = sumw_new
            top -= 1

    for j in range(top):
        for k in range(starts[j], starts[j] + lens[j]):
            sol[k] = gam[j]


@njit
def _solve_block_q4(length, sum_w, m1, m2, m3):
    """Closed-form solver for q=4 block gamma via Cardano's hyperbolic method."""
    c = m1 / length
    mu2 = m2 - 2.0 * c * m1 + length * c * c
    mu3 = m3 - 3.0 * c * m2 + 3.0 * c * c * m1 - length * c**3

    p = 3.0 * mu2 / length
    q = (sum_w - mu3) / length

    eps = 2.220446049250313e-16  # float64 eps

    if q > 0:
        sign = -1.0
    elif q < 0:
        sign = 1.0
    else:
        sign = 0.0

    if p > eps * max(abs(q), 1.0):
        sp3 = np.sqrt(p / 3.0)
        denom = 2.0 * p * sp3
        A = 3.0 * abs(q) / denom
        u = sign * 2.0 * sp3 * np.sinh(np.arcsinh(A) / 3.0)
    else:
        u = sign * abs(q) ** (1.0 / 3.0)

    return u + c


@njit
def _isotonic_pnorm_q4(s, w, sol):
    """PAV for q=4 (p=4/3) isotonic regression."""
    n = s.shape[0]

    starts = np.zeros(n, dtype=np.int64)
    lens = np.zeros(n, dtype=np.int64)
    sumw = np.zeros(n)
    m1_arr = np.zeros(n)
    m2_arr = np.zeros(n)
    m3_arr = np.zeros(n)
    gam = np.zeros(n)
    top = 0

    for i in range(n):
        si = s[i]
        starts[top] = i
        lens[top] = 1
        sumw[top] = w[i]
        m1_arr[top] = si
        m2_arr[top] = si * si
        m3_arr[top] = si * si * si
        gam[top] = _solve_block_q4(1.0, w[i], si, si * si, si * si * si)
        top += 1

        while top >= 2 and gam[top - 2] < gam[top - 1]:
            lens_new = lens[top - 2] + lens[top - 1]
            sumw_new = sumw[top - 2] + sumw[top - 1]
            m1_new = m1_arr[top - 2] + m1_arr[top - 1]
            m2_new = m2_arr[top - 2] + m2_arr[top - 1]
            m3_new = m3_arr[top - 2] + m3_arr[top - 1]
            gam[top - 2] = _solve_block_q4(
                float(lens_new), sumw_new, m1_new, m2_new, m3_new
            )
            lens[top - 2] = lens_new
            sumw[top - 2] = sumw_new
            m1_arr[top - 2] = m1_new
            m2_arr[top - 2] = m2_new
            m3_arr[top - 2] = m3_new
            top -= 1

    for j in range(top):
        for k in range(starts[j], starts[j] + lens[j]):
            sol[k] = gam[j]


def _inv_permutation(p: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.shape[0], dtype=p.dtype, device=p.device)
    return inv


def _segment_sum(
    data: torch.Tensor, ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    out = torch.zeros(num_segments, dtype=data.dtype, device=data.device)
    return out.scatter_add_(0, ids.long(), data)


# ── c0: euclidean (q=2) ──────────────────────────────────────────────


def _pav_isotonic_decreasing_pnorm_q2(y: torch.Tensor):
    """Pool Adjacent Violators for isotonic decreasing regression (L2).

    Uses Numba-JIT'd solver from fast-soft-sort when available.
    """
    n = y.shape[0]
    dtype = y.dtype
    device = y.device

    y_np = y.detach().cpu().numpy()
    sol_np = np.empty(n, dtype=y_np.dtype)
    _isotonic_l2(y_np, sol_np)

    v = torch.from_numpy(sol_np.copy()).to(dtype=dtype, device=device)

    if n <= 1:
        block_idx = torch.zeros(n, dtype=torch.int64, device=device)
        lens_t = torch.ones(max(n, 1), dtype=torch.int64, device=device)
        return v, block_idx, None, lens_t

    # Extract block structure from solution (vectorized)
    eps = max(float(torch.finfo(dtype).eps) ** 0.75, 1e-9)
    changes = (v[:-1] - v[1:]).abs() > eps
    block_idx = torch.zeros(n, dtype=torch.int64, device=device)
    block_idx[1:] = changes.cumsum(0)
    lens_t = block_idx.bincount()

    return v, block_idx, None, lens_t


class _ProjPermutahedronPnormQ2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        perm_z = torch.argsort(-z, stable=True)
        z_sorted = z[perm_z]
        inv_perm_z = _inv_permutation(perm_z)

        perm_w = torch.argsort(-w, stable=True)
        w_sorted = w[perm_w]
        inv_perm_w = _inv_permutation(perm_w)

        y = z_sorted - w_sorted
        v, block_idx, _, lens_t = _pav_isotonic_decreasing_pnorm_q2(y)

        p_sorted = z_sorted - v
        p = p_sorted[inv_perm_z]

        ctx.save_for_backward(perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens_t)
        return p

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens_t = ctx.saved_tensors
        dtype = g.dtype
        n = g.shape[0]

        g_sorted = g[perm_z]
        block_sum_g = _segment_sum(g_sorted, block_idx, n)
        Jt_g = block_sum_g[block_idx] / lens_t[block_idx].to(dtype)

        grad_z_sorted = g_sorted - Jt_g
        grad_w_sorted = Jt_g

        return grad_z_sorted[inv_perm_z], grad_w_sorted[inv_perm_w]


def _proj_permutahedron_pnorm_q2(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _ProjPermutahedronPnormQ2Fn.apply(z, w)


# ── c1: p-norm q=3 (p=3/2) ───────────────────────────────────────────


def _pav_isotonic_decreasing_pnorm_q3(s: torch.Tensor, w: torch.Tensor):
    """PAV for q=3 isotonic regression. Uses Numba JIT when available."""
    n = s.shape[0]
    device = s.device

    s_np = s.detach().cpu().to(torch.float64).numpy()
    w_np = w.detach().cpu().to(torch.float64).numpy()
    sol_np = np.empty(n, dtype=np.float64)
    _isotonic_pnorm_q3(s_np, w_np, sol_np)

    v = torch.from_numpy(sol_np.copy()).to(dtype=torch.float64, device=device)

    if n <= 1:
        block_idx = torch.zeros(n, dtype=torch.int64, device=device)
        lens_t = torch.ones(max(n, 1), dtype=torch.int64, device=device)
        return v, block_idx, lens_t

    eps = max(float(torch.finfo(torch.float64).eps) ** 0.75, 1e-9)
    changes = (v[:-1] - v[1:]).abs() > eps
    block_idx = torch.zeros(n, dtype=torch.int64, device=device)
    block_idx[1:] = changes.cumsum(0)
    lens_t = block_idx.bincount()

    return v, block_idx, lens_t


class _ProjPermutahedronPnormQ3Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        orig_dtype = z.dtype
        z64 = z.to(torch.float64)
        w64 = w.to(torch.float64)

        perm_z = torch.argsort(-z64, stable=True)
        z_sorted = z64[perm_z]
        inv_perm_z = _inv_permutation(perm_z)

        perm_w = torch.argsort(-w64, stable=True)
        w_sorted = w64[perm_w]
        inv_perm_w = _inv_permutation(perm_w)

        v, block_idx, lens_t = _pav_isotonic_decreasing_pnorm_q3(z_sorted, w_sorted)

        t = z_sorted - v
        y_sorted = t * torch.abs(t)  # t|t|

        y = y_sorted[inv_perm_z].to(orig_dtype)

        ctx.save_for_backward(
            perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens_t, t
        )
        ctx.orig_dtype = orig_dtype
        return y

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens_t, t = ctx.saved_tensors
        orig_dtype = ctx.orig_dtype
        g = g.to(torch.float64)
        n = g.shape[0]

        g_sorted_y = g[perm_z]
        g_t = g_sorted_y * (2.0 * torch.abs(t))

        weight = torch.abs(t)
        denom_block = _segment_sum(weight, block_idx, n)
        denom = denom_block[block_idx]

        sumg_block = _segment_sum(g_t, block_idx, n)
        sumg = sumg_block[block_idx]

        lens_elem = lens_t[block_idx].to(torch.float64)
        alpha = torch.where(denom > 0, weight / denom, 1.0 / lens_elem)

        jtg_s = alpha * sumg
        grad_z_sorted = g_t - jtg_s

        grad_w_sorted = torch.where(
            denom > 0, sumg / (2.0 * denom), torch.zeros_like(denom)
        )

        return (
            grad_z_sorted[inv_perm_z].to(orig_dtype),
            grad_w_sorted[inv_perm_w].to(orig_dtype),
        )


def _proj_permutahedron_pnorm_q3(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _ProjPermutahedronPnormQ3Fn.apply(z, w)


# ── c2: p-norm q=4 (p=4/3) ───────────────────────────────────────────


def _pav_isotonic_decreasing_pnorm_q4(s: torch.Tensor, w: torch.Tensor):
    """PAV for q=4 isotonic regression. Uses Numba JIT when available."""
    n = s.shape[0]
    device = s.device

    s_np = s.detach().cpu().to(torch.float64).numpy()
    w_np = w.detach().cpu().to(torch.float64).numpy()
    sol_np = np.empty(n, dtype=np.float64)
    _isotonic_pnorm_q4(s_np, w_np, sol_np)

    v = torch.from_numpy(sol_np.copy()).to(dtype=torch.float64, device=device)

    if n <= 1:
        block_idx = torch.zeros(n, dtype=torch.int64, device=device)
        lens_t = torch.ones(max(n, 1), dtype=torch.int64, device=device)
        return v, block_idx, lens_t

    eps = max(float(torch.finfo(torch.float64).eps) ** 0.75, 1e-9)
    changes = (v[:-1] - v[1:]).abs() > eps
    block_idx = torch.zeros(n, dtype=torch.int64, device=device)
    block_idx[1:] = changes.cumsum(0)
    lens_t = block_idx.bincount()

    return v, block_idx, lens_t


class _ProjPermutahedronPnormQ4Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        orig_dtype = z.dtype
        z64 = z.to(torch.float64)
        w64 = w.to(torch.float64)

        perm_z = torch.argsort(-z64, stable=True)
        z_sorted = z64[perm_z]
        inv_perm_z = _inv_permutation(perm_z)

        perm_w = torch.argsort(-w64, stable=True)
        w_sorted = w64[perm_w]
        inv_perm_w = _inv_permutation(perm_w)

        v, block_idx, lens_t = _pav_isotonic_decreasing_pnorm_q4(z_sorted, w_sorted)

        t = z_sorted - v
        y_sorted = t * (torch.abs(t) ** 2)  # t^3

        y = y_sorted[inv_perm_z].to(orig_dtype)

        ctx.save_for_backward(
            perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens_t, t
        )
        ctx.orig_dtype = orig_dtype
        return y

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens_t, t = ctx.saved_tensors
        orig_dtype = ctx.orig_dtype
        g = g.to(torch.float64)
        n = g.shape[0]

        g_sorted_y = g[perm_z]
        g_t = g_sorted_y * (3.0 * t * t)

        weight = torch.abs(t) ** 2
        denom_block = _segment_sum(weight, block_idx, n)
        denom = denom_block[block_idx]

        sumg_block = _segment_sum(g_t, block_idx, n)
        sumg = sumg_block[block_idx]

        lens_elem = lens_t[block_idx].to(torch.float64)
        alpha = torch.where(denom > 0, weight / denom, 1.0 / lens_elem)

        jtg_s = alpha * sumg
        grad_z_sorted = g_t - jtg_s

        grad_w_sorted = torch.where(
            denom > 0, sumg / (3.0 * denom), torch.zeros_like(denom)
        )

        return (
            grad_z_sorted[inv_perm_z].to(orig_dtype),
            grad_w_sorted[inv_perm_w].to(orig_dtype),
        )


def _proj_permutahedron_pnorm_q4(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _ProjPermutahedronPnormQ4Fn.apply(z, w)


# ── entropic isotonic regression (smooth mode) ────────────────────────


def _pav_isotonic_decreasing_entropic(s: torch.Tensor, w: torch.Tensor):
    """PAV for entropic isotonic regression (KL).

    Uses Numba-JIT'd solver from fast-soft-sort when available.
    """
    n = s.shape[0]
    dtype = s.dtype
    device = s.device

    s_np = s.detach().cpu().numpy()
    w_np = w.detach().cpu().numpy()
    sol_np = np.empty(n, dtype=s_np.dtype)
    _isotonic_kl(s_np, w_np, sol_np)

    v = torch.from_numpy(sol_np.copy()).to(dtype=dtype, device=device)

    if n <= 1:
        block_idx = torch.zeros(n, dtype=torch.int64, device=device)
        return v, block_idx, None, s.clone(), w.clone()

    # Extract block structure from solution (vectorized)
    eps = max(float(torch.finfo(dtype).eps) ** 0.75, 1e-9)
    changes = (v[:-1] - v[1:]).abs() > eps
    block_idx = torch.zeros(n, dtype=torch.int64, device=device)
    block_idx[1:] = changes.cumsum(0)
    num_blocks = block_idx[-1].item() + 1

    # Compute logsumexp of s and w per block (scatter-based)
    max_s = torch.full((num_blocks,), float("-inf"), dtype=dtype, device=device)
    max_w = torch.full((num_blocks,), float("-inf"), dtype=dtype, device=device)
    max_s.scatter_reduce_(0, block_idx, s, reduce="amax", include_self=True)
    max_w.scatter_reduce_(0, block_idx, w, reduce="amax", include_self=True)

    sum_exp_s = torch.zeros(num_blocks, dtype=dtype, device=device)
    sum_exp_w = torch.zeros(num_blocks, dtype=dtype, device=device)
    sum_exp_s.scatter_add_(0, block_idx, torch.exp(s - max_s[block_idx]))
    sum_exp_w.scatter_add_(0, block_idx, torch.exp(w - max_w[block_idx]))

    logS_compact = torch.log(sum_exp_s) + max_s
    logW_compact = torch.log(sum_exp_w) + max_w

    # Pad to size n for backward (save_for_backward expects consistent sizes)
    logS_t = torch.full((n,), float("-inf"), dtype=dtype, device=device)
    logW_t = torch.full((n,), float("-inf"), dtype=dtype, device=device)
    logS_t[:num_blocks] = logS_compact
    logW_t[:num_blocks] = logW_compact

    return v, block_idx, None, logS_t, logW_t


class _ProjPermutahedronEntropicFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        perm_z = torch.argsort(-z, stable=True)
        z_sorted = z[perm_z]
        inv_perm_z = _inv_permutation(perm_z)

        perm_w = torch.argsort(-w, stable=True)
        w_sorted = w[perm_w]
        inv_perm_w = _inv_permutation(perm_w)

        v, block_idx, _, logS, logW = _pav_isotonic_decreasing_entropic(
            z_sorted, w_sorted
        )
        p_sorted = z_sorted - v
        p = p_sorted[inv_perm_z]

        ctx.save_for_backward(
            perm_z,
            inv_perm_z,
            perm_w,
            inv_perm_w,
            block_idx,
            z_sorted,
            w_sorted,
            logS,
            logW,
        )
        return p

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        (
            perm_z,
            inv_perm_z,
            perm_w,
            inv_perm_w,
            block_idx,
            z_sorted,
            w_sorted,
            logS,
            logW,
        ) = ctx.saved_tensors
        n = g.shape[0]

        g_sorted = g[perm_z]

        block_sum_g = _segment_sum(g_sorted, block_idx, n)

        logS_b = logS[block_idx]
        logW_b = logW[block_idx]
        p_s = torch.exp(z_sorted - logS_b)
        q_w = torch.exp(w_sorted - logW_b)

        sum_g = block_sum_g[block_idx]
        Jt_g_s = p_s * sum_g
        Jt_g_w = (-q_w) * sum_g

        grad_z_sorted = g_sorted - Jt_g_s
        grad_w_sorted = -Jt_g_w

        return grad_z_sorted[inv_perm_z], grad_w_sorted[inv_perm_w]


def _proj_permutahedron_entropic(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _ProjPermutahedronEntropicFn.apply(z, w)


# ── dispatcher ────────────────────────────────────────────────────────


def _proj_permutahedron(
    z: torch.Tensor,  # (..., n)
    w: torch.Tensor,  # (..., n)
    softness: float | torch.Tensor = 0.1,
    mode: Literal["smooth", "c0", "c1", "c2"] = "smooth",
) -> torch.Tensor:  # (..., n)
    """Projects `z` onto the permutahedron of `w`.

    Modes:
        - ``smooth``: Entropic (log-KL) projection onto the permutahedron. Solved via isotonic regression, as described in `Fast Differentiable Sorting and Ranking <https://arxiv.org/abs/2002.08871>`_.
          Not fully C∞ due to argsort discontinuities at the boundary of sorting chambers.
        - ``c0``/``c1``/``c2``: p-norm projections (p=2, 3/2, 4/3).

    Note: The PAV isotonic regression step uses Numba JIT (``@njit``), which runs on
    CPU only. When using GPU tensors, data is transferred to CPU for the forward pass
    and back. The backward pass uses standard PyTorch ops and runs on GPU.
    """
    if z.shape != w.shape:
        raise ValueError(
            f"Shapes of z and w must match, got z.shape={z.shape} and w.shape={w.shape}"
        )
    _validate_softness(softness)
    *batch_sizes, n = z.shape
    z_batched = z.reshape(-1, n)
    w_batched = w.reshape(-1, n)
    z_batched = z_batched / softness

    if mode == "smooth":
        proj_fn = _proj_permutahedron_entropic
    elif mode == "c0":
        proj_fn = _proj_permutahedron_pnorm_q2
    elif mode == "c1":
        proj_fn = _proj_permutahedron_pnorm_q3
    elif mode == "c2":
        proj_fn = _proj_permutahedron_pnorm_q4
    else:
        raise ValueError(f"Invalid mode: {mode}")

    results = []
    for i in range(z_batched.shape[0]):
        results.append(proj_fn(z_batched[i], w_batched[i]))
    soft_values = torch.stack(results, dim=0)
    soft_values = soft_values.reshape(*batch_sizes, n)
    return soft_values
