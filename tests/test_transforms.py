"""Tests for higher-order derivatives."""

import numpy as np
import pytest
import torch

import softtorch as st


# ---------------------------------------------------------------------------
# Valid methods per function (mirrors test_arraywise.py)
# ---------------------------------------------------------------------------

VALUE_METHODS = ["softsort", "neuralsort", "fast_soft_sort", "ot", "sorting_network"]
ARG_METHODS = ["softsort", "neuralsort", "ot", "sorting_network"]

FUNCTION_SPECS = {
    "sort": VALUE_METHODS,
    "rank": VALUE_METHODS,
    "max": VALUE_METHODS,
    "min": VALUE_METHODS,
    "median": ["softsort", "neuralsort", "ot"],
    "quantile": VALUE_METHODS,
    "topk": VALUE_METHODS,
    "argsort": ARG_METHODS,
    "argmax": ARG_METHODS,
    "argmin": ARG_METHODS,
    "argmedian": ARG_METHODS,
    "argquantile": ARG_METHODS,
}

MODES = ["smooth", "c0", "c1", "c2"]
ELEMENTWISE_OPS = ["relu", "abs", "sign", "round", "heaviside", "clamp"]
COMPARISON_OPS = ["greater", "less", "eq", "not_equal", "isclose"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_elementwise(op, x, mode="smooth", softness=0.5):
    fn = getattr(st, op)
    if op == "clamp":
        return fn(x, 0.0, 1.0, softness=softness, mode=mode)
    return fn(x, softness=softness, mode=mode)


def _call_comparison(op, x, mode="smooth", softness=0.5):
    return getattr(st, op)(
        x, torch.tensor(0.0, dtype=x.dtype), softness=softness, mode=mode
    )


def _weighted_sum(out):
    if out.ndim == 0:
        return out
    weights = torch.arange(1, out.shape[-1] + 1, dtype=out.dtype, device=out.device)
    return (out * weights).sum()


def _axiswise_loss(op, x, method, mode, softness=0.5):
    """Scalar weighted loss from an axiswise op."""
    if op == "quantile":
        out = st.quantile(x, 0.5, dim=-1, softness=softness, mode=mode, method=method)
    elif op == "argquantile":
        out = st.argquantile(
            x, 0.5, dim=-1, softness=softness, mode=mode, method=method
        )
    elif op == "topk":
        result = st.topk(x, k=2, dim=-1, softness=softness, mode=mode, method=method)
        out = result.values
    elif op == "sort":
        result = st.sort(
            x, softness=softness, mode=mode, method=method, return_indices=False
        )
        out = result.values
    elif op in ("max", "min"):
        result = getattr(st, op)(x, dim=-1, softness=softness, mode=mode, method=method)
        out = result.values
    elif op == "median":
        result = st.median(x, dim=-1, softness=softness, mode=mode, method=method)
        out = result.values
    elif op == "argmedian":
        out = st.argmedian(x, dim=-1, softness=softness, mode=mode, method=method)
    elif op in ("argsort", "argmax", "argmin"):
        out = getattr(st, op)(x, dim=-1, softness=softness, mode=mode, method=method)
    else:
        out = getattr(st, op)(x, dim=-1, softness=softness, mode=mode, method=method)
    return _weighted_sum(out)


# ---------------------------------------------------------------------------
# Build exhaustive (op, method, mode) parametrization
# ---------------------------------------------------------------------------

_AXISWISE_CASES = []
for op, methods in FUNCTION_SPECS.items():
    for method in methods:
        for mode in MODES:
            _AXISWISE_CASES.append((op, method, mode))

_AXISWISE_IDS = [f"{op}_{method}_{mode}" for op, method, mode in _AXISWISE_CASES]


# ===================================================================
# Higher-order derivative tests
# ===================================================================


class TestHigherOrderDerivatives:
    """Verify second-order and higher derivatives via torch.autograd.grad(create_graph=True)."""

    # --- Elementwise: Hessian across all ops x modes ---

    @pytest.mark.parametrize("op", ELEMENTWISE_OPS)
    @pytest.mark.parametrize("mode", MODES)
    def test_hessian_elementwise(self, op, mode):
        """torch.func.hessian of elementwise ops produces finite results."""
        x = torch.tensor([-0.8, 0.3, -0.1, 0.6], dtype=torch.float64)
        loss = lambda x: _call_elementwise(op, x, mode=mode).sum()
        H = torch.func.hessian(loss)(x)
        assert H.shape == (4, 4)
        assert torch.all(torch.isfinite(H)), f"Hessian NaN/Inf for {op} {mode}"

    @pytest.mark.parametrize("op", ["relu", "abs", "sign"])
    @pytest.mark.parametrize("mode", MODES)
    def test_hessian_elementwise_vs_finite_diff(self, op, mode):
        """Elementwise Hessian matches finite-difference approximation."""
        x = torch.tensor([-0.8, 0.3, -0.1, 0.6], dtype=torch.float64)
        loss = lambda x: _call_elementwise(op, x, mode=mode).sum()
        H = torch.func.hessian(loss)(x)

        eps = 1e-5
        n = x.shape[0]
        H_fd = np.zeros((n, n))
        for i in range(n):
            ei = torch.zeros_like(x)
            ei[i] = eps
            gp = torch.func.grad(loss)(x + ei)
            gm = torch.func.grad(loss)(x - ei)
            H_fd[i] = ((gp - gm) / (2 * eps)).numpy()

        np.testing.assert_allclose(H.numpy(), H_fd, rtol=1e-3, atol=1e-6)

    # --- Comparison ops: Hessian ---

    @pytest.mark.parametrize("op", COMPARISON_OPS)
    @pytest.mark.parametrize("mode", MODES)
    def test_hessian_comparison(self, op, mode):
        """torch.func.hessian of comparison ops produces finite results."""
        x = torch.tensor([-0.8, 0.3, -0.1, 0.6], dtype=torch.float64)
        loss = lambda x: _call_comparison(op, x, mode=mode).sum()
        H = torch.func.hessian(loss)(x)
        assert H.shape == (4, 4)
        assert torch.all(torch.isfinite(H)), f"Hessian NaN/Inf for {op} {mode}"

    # --- Elementwise: higher-order via create_graph ---

    @pytest.mark.parametrize(
        "mode,max_order",
        [
            ("smooth", 4),  # C-infinity: arbitrary-order derivatives
            ("c0", 2),  # C0: 1st derivative is piecewise constant, 2nd is zero
            ("c1", 3),  # C1: 2nd derivative exists, 3rd is zero at boundaries
            ("c2", 3),  # C2: 3rd derivative exists
        ],
    )
    def test_higher_order_derivative_sigmoidal(self, mode, max_order):
        """Higher-order derivatives of sigmoidal up to the expected order per mode."""
        x = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        y = st.sigmoidal(x, softness=0.5, mode=mode).sum()
        g = y
        for order in range(1, max_order + 1):
            create = order < max_order
            g = torch.autograd.grad(
                g if order == 1 else g.sum(), x, create_graph=create
            )[0]
            assert torch.all(torch.isfinite(g)), f"order-{order} not finite for {mode}"

    def test_sigmoidal_c0_double_backward_preserves_graph(self):
        """Regression for issue #2: direct c0 sigmoidal supports double backward."""
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        y = st.sigmoidal(x, softness=0.1, mode="c0")
        g1 = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        g2 = torch.autograd.grad(g1.sum(), x)[0]
        assert torch.all(torch.isfinite(g2))

    def test_fourth_derivative_smooth_relu(self):
        """Smooth relu (softplus) supports 4th-order derivatives."""
        x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        y = st.relu(x, softness=0.5, mode="smooth").sum()
        g = y
        for order in range(1, 5):
            create = order < 4
            g = torch.autograd.grad(
                g if order == 1 else g.sum(), x, create_graph=create
            )[0]
            assert torch.all(torch.isfinite(g)), f"order-{order} derivative not finite"

    # --- Axiswise: Hessian via autograd.grad(create_graph=True) ---

    @pytest.mark.parametrize("op,method,mode", _AXISWISE_CASES, ids=_AXISWISE_IDS)
    def test_hessian_axiswise(self, op, method, mode):
        """Second derivative via create_graph=True for every (op, method, mode)."""
        x = torch.tensor([0.3, -0.5, 1.2, 0.1], dtype=torch.float64, requires_grad=True)
        y = _axiswise_loss(op, x, method, mode)
        g = torch.autograd.grad(y, x, create_graph=True)[0]
        assert torch.all(torch.isfinite(g)), (
            f"1st grad NaN/Inf for {op} {method} {mode}"
        )
        h = torch.autograd.grad(g.sum(), x)[0]
        assert torch.all(torch.isfinite(h)), (
            f"2nd grad NaN/Inf for {op} {method} {mode}"
        )
