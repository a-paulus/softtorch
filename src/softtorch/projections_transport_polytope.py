"""Transport polytope projections (smooth via Sinkhorn, c0/c1/c2 via LBFGS)."""

from typing import Literal

import ot
import torch
import torchopt

from softtorch.utils import _validate_softness


def _proj_transport_polytope_entropic(
    C: torch.Tensor,  # (n, m)
    mu: torch.Tensor,  # (n,)
    nu: torch.Tensor,  # (m,)
    epsilon: float,
    tol: float = 1e-5,
    max_iter: int = 10000,
) -> torch.Tensor:
    """Solve entropy-regularized OT via Sinkhorn (log-domain) with implicit diff."""
    orig_dtype = C.dtype
    C = C.to(torch.float64)
    mu = mu.to(torch.float64)
    nu = nu.to(torch.float64)

    tiny = 1e-12
    mu = torch.clamp(mu, min=tiny)
    mu = mu / mu.sum()
    nu = torch.clamp(nu, min=tiny)
    nu = nu / nu.sum()

    n, m = C.shape
    epsilon = float(epsilon)

    def optimality_fn(opt_params, C, mu, nu, epsilon):
        f, g_rest = opt_params
        g = torch.cat([torch.zeros(1, dtype=C.dtype, device=C.device), g_rest])
        Z = (f[:, None] + g[None, :] - C) / epsilon
        K = torch.exp(Z)
        grad_f = mu - K.sum(dim=1)
        grad_g_rest = nu[1:] - K.sum(dim=0)[1:]
        return (grad_f, grad_g_rest)

    @torchopt.diff.implicit.custom_root(
        optimality_fn,
        argnums=1,
        solve=torchopt.linear_solve.solve_normal_cg(maxiter=50, atol=0),
    )
    def solve(opt_params_init, C, mu, nu, epsilon):
        # Forward: Sinkhorn in log-domain (numerically stable for small epsilon).
        _, log_dict = ot.bregman.sinkhorn_log(
            mu,
            nu,
            C,
            reg=epsilon,
            numItermax=max_iter,
            stopThr=tol,
            warn=False,
            log=True,
        )
        # Extract gauge-fixed dual potentials: f_i = eps*log_u_i, g_j = eps*log_v_j.
        f = epsilon * log_dict["log_u"]
        g = epsilon * log_dict["log_v"]
        g0 = g[0]
        f = f + g0  # absorb g[0] into f
        g_rest = g[1:] - g0  # gauge fix: g[0] = 0
        return (f, g_rest)

    f0 = torch.zeros(n, dtype=C.dtype, device=C.device)
    g_rest0 = torch.zeros(m - 1, dtype=C.dtype, device=C.device)

    f_star, g_rest_star = solve((f0, g_rest0), C, mu, nu, epsilon)

    g_star = torch.cat([torch.zeros(1, dtype=C.dtype, device=C.device), g_rest_star])
    Gamma = torch.exp((f_star[:, None] + g_star[None, :] - C) / epsilon)
    return Gamma.to(orig_dtype)


def _proj_transport_polytope_pnorm_lbfgs(
    C: torch.Tensor,  # (n, m)
    mu: torch.Tensor,  # (n,)
    nu: torch.Tensor,  # (m,)
    lam: float,
    p: float,
    tol: float = 1e-5,
    max_iter: int = 10000,
) -> torch.Tensor:
    """Solve p-norm regularized OT dual via LBFGS with torchopt implicit diff."""
    orig_dtype = C.dtype
    C = C.to(torch.float64)
    mu = mu.to(torch.float64)
    nu = nu.to(torch.float64)

    # Avoid exact zeros
    tiny = 1e-12
    mu = torch.clamp(mu, min=tiny)
    mu = mu / mu.sum()
    nu = torch.clamp(nu, min=tiny)
    nu = nu / nu.sum()

    n, m = C.shape
    q = p / (p - 1.0)
    lam_pow = lam ** (-(q - 1.0))

    def optimality_fn(opt_params, C, mu, nu, lam_pow, q):
        f, g_rest = opt_params
        g = torch.cat([torch.zeros(1, dtype=C.dtype, device=C.device), g_rest])
        S = f[:, None] + g[None, :] - C
        P = torch.clamp(S, min=0.0)
        Pq1 = lam_pow * P ** (q - 1.0)
        grad_f = mu - Pq1.sum(dim=1)
        grad_g_rest = nu[1:] - Pq1.sum(dim=0)[1:]
        return (grad_f, grad_g_rest)

    @torchopt.diff.implicit.custom_root(
        optimality_fn,
        argnums=1,
        solve=torchopt.linear_solve.solve_normal_cg(maxiter=50, atol=0),
    )
    def solve(opt_params_init, C, mu, nu, lam_pow, q):
        # Implicit diff handles gradients w.r.t. C via the optimality
        # conditions. The forward LBFGS closure can use detached inputs so
        # .backward() does not traverse the outer graph under create_graph=True.
        C_d = C.detach()
        mu_d = mu.detach()
        nu_d = nu.detach()

        f = opt_params_init[0].detach().clone().requires_grad_(True)
        g_rest = opt_params_init[1].detach().clone().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [f, g_rest],
            lr=1.0,
            max_iter=1,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        old_loss = torch.tensor(float("inf"), dtype=C_d.dtype, device=C_d.device)

        for _ in range(max_iter):
            old_f = f.data.clone()
            old_g_rest = g_rest.data.clone()

            def closure():
                optimizer.zero_grad()
                g = torch.cat(
                    [torch.zeros(1, dtype=C_d.dtype, device=C_d.device), g_rest]
                )
                S = f[:, None] + g[None, :] - C_d
                P = torch.clamp(S, min=0.0)
                dual = (
                    torch.dot(mu_d, f)
                    + torch.dot(nu_d, g)
                    - (lam_pow / q) * torch.sum(P**q)
                )
                neg_dual = -dual
                neg_dual.backward()
                return neg_dual

            loss = optimizer.step(closure)

            # Cauchy termination (matches optimistix): both relative parameter
            # change and relative function value change must be small.
            f_diff = (f.data - old_f).abs().max()
            g_diff = (g_rest.data - old_g_rest).abs().max()
            y_scale = tol + tol * max(f.data.abs().max(), g_rest.data.abs().max())
            y_converged = max(f_diff, g_diff) < y_scale

            loss_diff = (loss - old_loss).abs()
            f_scale = tol + tol * old_loss.abs()
            f_converged = loss_diff < f_scale

            if y_converged and f_converged:
                break

            old_loss = loss.detach()

        return (f, g_rest)

    f0 = torch.zeros(n, dtype=C.dtype, device=C.device)
    g_rest0 = torch.zeros(m - 1, dtype=C.dtype, device=C.device)

    f_star, g_rest_star = solve((f0, g_rest0), C, mu, nu, lam_pow, q)

    g_star = torch.cat([torch.zeros(1, dtype=C.dtype, device=C.device), g_rest_star])
    S = f_star[:, None] + g_star[None, :] - C
    Gamma = lam_pow * torch.clamp(S, min=0.0) ** (q - 1.0)
    return Gamma.to(orig_dtype)


def _proj_transport_polytope(
    cost: torch.Tensor,  # (..., n, m)
    mu: torch.Tensor,  # ([n],)
    nu: torch.Tensor,  # ([m],)
    softness: float = 0.1,
    mode: Literal["smooth", "c0", "c1", "c2"] = "smooth",
    sinkhorn_tol: float = 1e-5,
    sinkhorn_max_iter: int = 10000,
    lbfgs_tol: float = 1e-5,
    lbfgs_max_iter: int = 10000,
) -> torch.Tensor:  # (..., [n], m)
    """Projects a cost matrix onto the transport polytope between `mu` and `nu`.

    Solves the optimization problem:
        min_G <C, G> + softness * R(G)
        s.t. G 1_m = mu, G^T 1_n = nu, G >= 0
    where R(G) is the regularizer determined by `mode`.
    """
    _validate_softness(softness)
    *batch_sizes, n, m = cost.shape
    cost = cost.reshape(-1, n, m)  # (B, n, m)

    if mode == "smooth":

        def proj_fn(c: torch.Tensor) -> torch.Tensor:
            return _proj_transport_polytope_entropic(
                c,
                mu=mu,
                nu=nu,
                epsilon=softness,
                tol=sinkhorn_tol,
                max_iter=sinkhorn_max_iter,
            )

    elif mode in ("c0", "c1", "c2"):
        if mode == "c0":
            # Curvature of (1/2)||y||^2 at transport polytope center: R''=1
            p = 2.0
        elif mode == "c1":
            p = 3 / 2
        else:  # c2
            p = 4 / 3

        def proj_fn(c: torch.Tensor) -> torch.Tensor:
            return _proj_transport_polytope_pnorm_lbfgs(
                c,
                mu=mu,
                nu=nu,
                lam=softness,
                p=p,
                tol=lbfgs_tol,
                max_iter=lbfgs_max_iter,
            )

    else:
        raise ValueError(f"Invalid mode: {mode}")

    Gamma_list = [proj_fn(c) for c in cost]
    Gamma = torch.stack(Gamma_list, dim=0)  # (B, n, m)
    y = Gamma * n  # (B, [n], [m])
    y = y.reshape(*batch_sizes, n, m)
    return y
