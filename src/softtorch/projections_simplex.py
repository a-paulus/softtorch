"""Simplex projection algorithms with custom autograd backwards."""

from typing import Literal

import torch
import torch.nn.functional as F

from softtorch.utils import _canonicalize_dim, _validate_softness


# ── c0: Euclidean projection (q=2) ──────────────────────────────────


class _ProjUnitSimplexPnormQ2Fn(torch.autograd.Function):
    """Euclidean projection onto the unit simplex with custom backward.

    Projection via sorting+thresholding (Duchi et al. 2008).
    Backward projects gradient onto the tangent space of the simplex at the solution.
    """

    @staticmethod
    def forward(ctx, values: torch.Tensor) -> torch.Tensor:
        s = 1.0
        n_features = values.shape[-1]
        u, _ = torch.sort(values, dim=-1, descending=True)
        cumsum_u = torch.cumsum(u, dim=-1)
        ind = torch.arange(1, n_features + 1, device=values.device, dtype=values.dtype)
        cond = s / ind + (u - cumsum_u / ind) > 0
        idx = cond.sum(dim=-1).clamp(min=1)  # (...,)
        gather_idx = (idx - 1).long().unsqueeze(-1)
        cumsum_sel = torch.gather(cumsum_u, -1, gather_idx).squeeze(-1)
        theta = cumsum_sel / idx.to(values.dtype) - s / idx.to(values.dtype)
        result = torch.clamp(values - theta.unsqueeze(-1), min=0.0)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (result,) = ctx.saved_tensors
        supp = (result > 0).to(grad_output.dtype)
        card = supp.sum(dim=-1, keepdim=True).clamp(min=1.0)
        grad_input = supp * grad_output - (
            supp * (supp * grad_output).sum(dim=-1, keepdim=True) / card
        )
        return grad_input


def _proj_unit_simplex_pnorm_q2(values: torch.Tensor) -> torch.Tensor:
    return _ProjUnitSimplexPnormQ2Fn.apply(values)


# ── c1: p=3/2 projection (q=3) ──────────────────────────────────────


def _proj_unit_simplex_pnorm_q3_impl(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form simplex projection for p=3/2 (alpha=2) via quadratic formula.

    Returns (primal_out, theta).
    """
    n = values.shape[-1]
    dtype = values.dtype
    u, _ = torch.sort(values, dim=-1, descending=True)
    u0 = u[..., 0:1]  # (..., 1)
    u_shift = u - u0

    S = torch.cumsum(u_shift, dim=-1)
    M2 = torch.cumsum(u_shift**2, dim=-1)
    k_arr = torch.arange(1, n + 1, device=values.device, dtype=dtype)

    disc = S**2 - k_arr * (M2 - 1.0)
    theta_k = (S - torch.sqrt(torch.clamp(disc, min=0.0))) / k_arr

    cond = u_shift > theta_k
    idx = cond.sum(dim=-1).clamp(min=1)  # (...,)
    gather_idx = (idx - 1).long().unsqueeze(-1)
    theta = torch.gather(theta_k, -1, gather_idx).squeeze(-1) + u0.squeeze(-1)

    y = torch.clamp(values - theta.unsqueeze(-1), min=0.0) ** 2
    y_sum = y.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    return y / y_sum, theta


class _ProjUnitSimplexPnormQ3Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor) -> torch.Tensor:
        primal_out, theta = _proj_unit_simplex_pnorm_q3_impl(values)
        ctx.save_for_backward(values, primal_out, theta)
        return primal_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        values, primal_out, theta = ctx.saved_tensors

        supp = (primal_out > 0).to(values.dtype)
        t = torch.clamp(values - theta.unsqueeze(-1), min=0.0)
        w = t * supp  # weight propto 2*t
        w_sum = w.sum(dim=-1, keepdim=True)
        w_sum = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))

        raw_tangent = (
            2.0
            * t
            * (grad_output - (w * grad_output).sum(dim=-1, keepdim=True) / w_sum)
            * supp
        )

        sum_t2 = (t**2).sum(dim=-1, keepdim=True)
        sum_t2 = torch.where(sum_t2 > 0, sum_t2, torch.ones_like(sum_t2))
        tangent_out = (
            raw_tangent / sum_t2
            - primal_out * raw_tangent.sum(dim=-1, keepdim=True) / sum_t2
        )
        return tangent_out


def _proj_unit_simplex_pnorm_q3(values: torch.Tensor) -> torch.Tensor:
    return _ProjUnitSimplexPnormQ3Fn.apply(values)


# ── c2: p=4/3 projection (q=4) ──────────────────────────────────────


def _proj_unit_simplex_pnorm_q4_impl(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form simplex projection for p=4/3 (alpha=3) via Cardano's method.

    Returns (primal_out, theta).
    """
    n = values.shape[-1]
    dtype = values.dtype
    u, _ = torch.sort(values, dim=-1, descending=True)
    u0 = u[..., 0:1]  # (..., 1)
    u_shift = u - u0

    S = torch.cumsum(u_shift, dim=-1)
    M2 = torch.cumsum(u_shift**2, dim=-1)
    M3 = torch.cumsum(u_shift**3, dim=-1)
    k_arr = torch.arange(1, n + 1, device=values.device, dtype=dtype)

    c = S / k_arr
    mu2 = M2 - 2.0 * c * S + k_arr * c**2
    mu3 = M3 - 3.0 * c * M2 + 3.0 * c**2 * S - k_arr * c**3

    p_coeff = 3.0 * mu2 / k_arr  # >= 0
    q_coeff = (1.0 - mu3) / k_arr

    tiny = torch.finfo(dtype).tiny
    eps = torch.finfo(dtype).eps
    sp3 = torch.sqrt(torch.clamp(p_coeff / 3.0, min=0.0))
    denom = 2.0 * torch.clamp(p_coeff, min=tiny) * sp3
    A = 3.0 * torch.abs(q_coeff) / denom
    u_hyp = -torch.sign(q_coeff) * 2.0 * sp3 * torch.sinh(torch.arcsinh(A) / 3.0)
    u_cbrt = -torch.sign(q_coeff) * torch.abs(q_coeff) ** (1.0 / 3.0)
    u_root = torch.where(
        p_coeff > eps * torch.clamp(torch.abs(q_coeff), min=1.0),
        u_hyp,
        u_cbrt,
    )
    theta_k = u_root + c

    cond = u_shift > theta_k
    idx = cond.sum(dim=-1).clamp(min=1)
    gather_idx = (idx - 1).long().unsqueeze(-1)
    theta = torch.gather(theta_k, -1, gather_idx).squeeze(-1) + u0.squeeze(-1)

    y = torch.clamp(values - theta.unsqueeze(-1), min=0.0) ** 3
    y_sum = y.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    return y / y_sum, theta


class _ProjUnitSimplexPnormQ4Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor) -> torch.Tensor:
        primal_out, theta = _proj_unit_simplex_pnorm_q4_impl(values)
        ctx.save_for_backward(values, primal_out, theta)
        return primal_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        values, primal_out, theta = ctx.saved_tensors

        supp = (primal_out > 0).to(values.dtype)
        t = torch.clamp(values - theta.unsqueeze(-1), min=0.0)
        w = t**2 * supp  # weight propto 3*t^2
        w_sum = w.sum(dim=-1, keepdim=True)
        w_sum = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))

        raw_tangent = (
            3.0
            * t**2
            * (grad_output - (w * grad_output).sum(dim=-1, keepdim=True) / w_sum)
            * supp
        )

        sum_t3 = (t**3).sum(dim=-1, keepdim=True)
        sum_t3 = torch.where(sum_t3 > 0, sum_t3, torch.ones_like(sum_t3))
        tangent_out = (
            raw_tangent / sum_t3
            - primal_out * raw_tangent.sum(dim=-1, keepdim=True) / sum_t3
        )
        return tangent_out


def _proj_unit_simplex_pnorm_q4(values: torch.Tensor) -> torch.Tensor:
    return _ProjUnitSimplexPnormQ4Fn.apply(values)


# ── dispatcher ────────────────────────────────────────────────────────


def _proj_simplex(
    x: torch.Tensor,  # (..., n, ...)
    dim: int,
    softness: float = 0.1,
    mode: Literal["smooth", "c0", "c1", "c2"] = "smooth",
) -> torch.Tensor:  # (..., [n], ...)
    """Projects `x` onto the unit simplex along the specified dim.

    Solves the optimization problem along the specified dim:
        min_y <x, y> + softness * R(y)
        s.t. y >= 0, sum(y) = 1
    where R(y) is the regularizer determined by `mode`.
    """
    _validate_softness(softness)
    dim = _canonicalize_dim(dim, x.ndim)
    n = x.shape[dim]
    _x = x / softness
    if mode == "smooth":
        soft_index = F.softmax(_x, dim=dim)
    elif mode == "c0":
        _x = torch.movedim(_x, dim, -1)
        *batch_sizes, n = _x.shape
        _x = _x.reshape(-1, n)
        soft_index = _proj_unit_simplex_pnorm_q2(_x)
        soft_index = soft_index.reshape(*batch_sizes, n)
        soft_index = torch.movedim(soft_index, -1, dim)
    else:
        if mode == "c1":
            proj = _proj_unit_simplex_pnorm_q3
        elif mode == "c2":
            proj = _proj_unit_simplex_pnorm_q4
        else:
            raise ValueError(f"Invalid mode: {mode}")

        _x = torch.movedim(_x, dim, -1)
        *batch_sizes, n = _x.shape
        _x = _x.reshape(-1, n)
        soft_index = proj(_x)
        soft_index = soft_index.reshape(*batch_sizes, n)
        soft_index = torch.movedim(soft_index, -1, dim)
    return soft_index
