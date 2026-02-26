from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import softtorch as st

from . import common


SOFTNESS = common.STABILITY_SOFTNESS


# ---------------------------------------------------------------------------
# grad_replace
# ---------------------------------------------------------------------------


def test_grad_replace_scalar():
    """grad_replace uses backward branch for gradients but forward value."""

    @st.grad_replace
    def fn(x, *, forward: bool):
        return x if forward else 2.0 * x

    x = torch.tensor(3.0, requires_grad=True)
    val = fn(x)
    assert float(val) == 3.0
    (grad,) = torch.autograd.grad(val, x)
    assert float(grad) == 2.0


# ---------------------------------------------------------------------------
# Elementwise ST functions
# ---------------------------------------------------------------------------

ELEMENTWISE_ST_CASES = [
    {
        "name": "abs_st",
        "st_fn": st.abs_st,
        "hard_fn": lambda x: torch.abs(x),
        "soft_fn": lambda x, **kw: st.abs(x, **kw),
        "args": lambda: (torch.tensor([-1.0, 0.5, -0.3, 0.8]),),
        "extra_kwargs": {},
    },
    {
        "name": "relu_st",
        "st_fn": st.relu_st,
        "hard_fn": lambda x: torch.relu(x),
        "soft_fn": lambda x, **kw: st.relu(x, **kw),
        "args": lambda: (torch.tensor([-1.0, 0.5, -0.3, 0.8]),),
        "extra_kwargs": {},
    },
    {
        "name": "clamp_st",
        "st_fn": st.clamp_st,
        "hard_fn": lambda x: torch.clamp(x, -0.25, 0.25),
        "soft_fn": lambda x, **kw: st.clamp(x, -0.25, 0.25, **kw),
        "args": lambda: (torch.tensor([-1.0, 0.5, -0.3, 0.8]),),
        "extra_kwargs": {"a": -0.25, "b": 0.25},
    },
    {
        "name": "sign_st",
        "st_fn": st.sign_st,
        "hard_fn": lambda x: torch.sign(x).float(),
        "soft_fn": lambda x, **kw: st.sign(x, **kw),
        "args": lambda: (torch.tensor([-1.0, 0.5, -0.3, 0.8]),),
        "extra_kwargs": {},
    },
    {
        "name": "round_st",
        "st_fn": st.round_st,
        "hard_fn": lambda x: torch.round(x),
        "soft_fn": lambda x, **kw: st.round(x, **kw),
        "args": lambda: (torch.tensor([-0.7, 0.3, 1.5, -1.2]),),
        "extra_kwargs": {},
    },
    {
        "name": "heaviside_st",
        "st_fn": st.heaviside_st,
        "hard_fn": lambda x: torch.where(x < 0.0, 0.0, torch.where(x > 0.0, 1.0, 0.5)).float(),
        "soft_fn": lambda x, **kw: st.heaviside(x, **kw),
        "args": lambda: (torch.tensor([-1.0, 0.5, -0.3, 0.8]),),
        "extra_kwargs": {},
    },
]


@pytest.mark.parametrize("case", ELEMENTWISE_ST_CASES, ids=lambda c: c["name"])
def test_st_elementwise_forward(case):
    """Forward value of ST function must equal the hard function output."""
    args = case["args"]()
    x = args[0]

    if case["name"] == "clamp_st":
        st_out = case["st_fn"](
            x,
            case["extra_kwargs"]["a"],
            case["extra_kwargs"]["b"],
            softness=SOFTNESS,
            mode="smooth",
        )
    else:
        st_out = case["st_fn"](x, softness=SOFTNESS, mode="smooth")

    expected = case["hard_fn"](x)
    np.testing.assert_allclose(
        st_out.detach().numpy(), expected.detach().numpy(), atol=1e-5
    )


@pytest.mark.parametrize("case", ELEMENTWISE_ST_CASES, ids=lambda c: c["name"])
def test_st_elementwise_gradient(case):
    """Gradient of ST function must match soft function gradient and be finite."""
    args = case["args"]()
    x = args[0]

    if case["name"] == "clamp_st":

        def loss_st(z):
            return case["st_fn"](
                z,
                case["extra_kwargs"]["a"],
                case["extra_kwargs"]["b"],
                softness=SOFTNESS,
                mode="smooth",
            ).sum()

        def loss_soft(z):
            return case["soft_fn"](z, softness=SOFTNESS, mode="smooth").sum()
    else:

        def loss_st(z):
            return case["st_fn"](z, softness=SOFTNESS, mode="smooth").sum()

        def loss_soft(z):
            return case["soft_fn"](z, softness=SOFTNESS, mode="smooth").sum()

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    (grad_st,) = torch.autograd.grad(loss_st(x_st), x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft(x_soft), x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg=case["name"])


# ---------------------------------------------------------------------------
# Comparison ST functions
# ---------------------------------------------------------------------------

COMPARISON_ST_CASES = [
    {
        "name": "greater_st",
        "st_fn": st.greater_st,
        "hard_fn": lambda x, y: torch.gt(x, y).float(),
        "soft_fn": st.greater,
    },
    {
        "name": "greater_equal_st",
        "st_fn": st.greater_equal_st,
        "hard_fn": lambda x, y: torch.ge(x, y).float(),
        "soft_fn": st.greater_equal,
    },
    {
        "name": "less_st",
        "st_fn": st.less_st,
        "hard_fn": lambda x, y: torch.lt(x, y).float(),
        "soft_fn": st.less,
    },
    {
        "name": "less_equal_st",
        "st_fn": st.less_equal_st,
        "hard_fn": lambda x, y: torch.le(x, y).float(),
        "soft_fn": st.less_equal,
    },
    {
        "name": "eq_st",
        "st_fn": st.eq_st,
        "hard_fn": lambda x, y: torch.eq(x, y).float(),
        "soft_fn": st.eq,
    },
    {
        "name": "not_equal_st",
        "st_fn": st.not_equal_st,
        "hard_fn": lambda x, y: torch.ne(x, y).float(),
        "soft_fn": st.not_equal,
    },
    {
        "name": "isclose_st",
        "st_fn": st.isclose_st,
        "hard_fn": lambda x, y: torch.isclose(x, y).float(),
        "soft_fn": st.isclose,
    },
]


@pytest.mark.parametrize("case", COMPARISON_ST_CASES, ids=lambda c: c["name"])
def test_st_comparison_forward(case):
    """Forward of comparison ST must equal hard comparison."""
    x = torch.tensor([-1.0, 0.5, 0.0, 0.8])
    y = torch.tensor([0.1, 0.2, 0.0, 0.9])
    st_out = case["st_fn"](x, y, softness=SOFTNESS, mode="smooth")
    expected = case["hard_fn"](x, y)
    np.testing.assert_allclose(
        st_out.detach().numpy(), expected.detach().numpy(), atol=1e-5
    )


@pytest.mark.parametrize("case", COMPARISON_ST_CASES, ids=lambda c: c["name"])
def test_st_comparison_gradient(case):
    """Gradient of comparison ST must match soft gradient and be finite."""
    x = torch.tensor([-1.0, 0.5, 0.0, 0.8])
    y = torch.tensor([0.1, 0.2, 0.0, 0.9])

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = case["st_fn"](x_st, y, softness=SOFTNESS, mode="smooth").sum()
    loss_soft = case["soft_fn"](x_soft, y, softness=SOFTNESS, mode="smooth").sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg=case["name"])


# ---------------------------------------------------------------------------
# Array ST functions
# ---------------------------------------------------------------------------


def _make_vec():
    return torch.tensor([0.1, 0.4, -0.2, 0.3])


def test_st_argmax_forward_and_grad():
    """argmax_st returns hard one-hot but soft gradients."""
    x = _make_vec()
    weights = torch.arange(4.0)

    hard_forward = st.argmax_st(x, softness=SOFTNESS, mode="smooth")
    expected_forward = F.one_hot(torch.argmax(x), num_classes=x.shape[0]).float()
    np.testing.assert_allclose(
        hard_forward.detach().numpy(), expected_forward.numpy()
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = torch.dot(st.argmax_st(x_st, softness=SOFTNESS, mode="smooth"), weights)
    loss_soft = torch.dot(st.argmax(x_soft, softness=SOFTNESS, mode="smooth"), weights)

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argmax_st")


def test_st_argmin_forward_and_grad():
    """argmin_st returns hard one-hot but soft gradients."""
    x = _make_vec()
    weights = torch.arange(4.0)

    hard_forward = st.argmin_st(x, softness=SOFTNESS, mode="smooth")
    expected_forward = F.one_hot(torch.argmin(x), num_classes=x.shape[0]).float()
    np.testing.assert_allclose(
        hard_forward.detach().numpy(), expected_forward.numpy()
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = torch.dot(st.argmin_st(x_st, softness=SOFTNESS, mode="smooth"), weights)
    loss_soft = torch.dot(st.argmin(x_soft, softness=SOFTNESS, mode="smooth"), weights)

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argmin_st")


def test_st_max_forward_and_grad():
    """max_st returns hard max but soft gradients."""
    x = _make_vec()

    hard_forward = st.max_st(x, softness=SOFTNESS, mode="smooth")
    expected = torch.max(x)
    np.testing.assert_allclose(
        hard_forward.detach().numpy(), expected.numpy(), atol=1e-5
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = st.max_st(x_st, softness=SOFTNESS, mode="smooth").sum()
    loss_soft = st.max(x_soft, softness=SOFTNESS, mode="smooth").sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="max_st")


def test_st_min_forward_and_grad():
    """min_st returns hard min but soft gradients."""
    x = _make_vec()

    hard_forward = st.min_st(x, softness=SOFTNESS, mode="smooth")
    expected = torch.min(x)
    np.testing.assert_allclose(
        hard_forward.detach().numpy(), expected.numpy(), atol=1e-5
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = st.min_st(x_st, softness=SOFTNESS, mode="smooth").sum()
    loss_soft = st.min(x_soft, softness=SOFTNESS, mode="smooth").sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="min_st")


def test_st_sort_forward_and_grad():
    """sort_st returns hard sorted but soft gradients."""
    x = _make_vec()

    hard_forward = st.sort_st(x, softness=SOFTNESS, mode="smooth")
    expected = torch.sort(x)
    np.testing.assert_allclose(
        hard_forward.values.detach().numpy(), expected.values.numpy(), atol=1e-5
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = st.sort_st(x_st, softness=SOFTNESS, mode="smooth").values.sum()
    loss_soft = st.sort(x_soft, softness=SOFTNESS, mode="smooth").values.sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="sort_st")


def test_st_argsort_forward_and_grad():
    """argsort_st returns hard permutation but soft gradients."""
    x = _make_vec()
    weights = torch.arange(4.0)

    hard_forward = st.argsort_st(x, softness=SOFTNESS, mode="smooth")
    expected = F.one_hot(torch.argsort(x), num_classes=x.shape[0]).float()
    np.testing.assert_allclose(
        hard_forward.detach().numpy(), expected.numpy(), atol=1e-5
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = (st.argsort_st(x_st, softness=SOFTNESS, mode="smooth") @ weights).sum()
    loss_soft = (st.argsort(x_soft, softness=SOFTNESS, mode="smooth") @ weights).sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argsort_st")


def test_st_rank_forward_and_grad():
    """rank_st returns hard rank but soft gradients."""
    x = _make_vec()

    hard_forward = st.rank_st(x, softness=SOFTNESS, mode="smooth")
    expected = st.rank(x, mode="hard")
    np.testing.assert_allclose(
        hard_forward.detach().numpy(), expected.detach().numpy(), atol=1e-5
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = st.rank_st(x_st, softness=SOFTNESS, mode="smooth").sum()
    loss_soft = st.rank(x_soft, softness=SOFTNESS, mode="smooth").sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="rank_st")


def test_st_median_forward_and_grad():
    """median_st returns hard median but soft gradients."""
    x = torch.tensor([0.1, 0.4, -0.2, 0.3, 0.5])

    hard_forward = st.median_st(x, softness=SOFTNESS, mode="smooth")
    expected = torch.median(x)
    np.testing.assert_allclose(
        hard_forward.detach().numpy(), expected.numpy(), atol=1e-5
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = st.median_st(x_st, softness=SOFTNESS, mode="smooth").sum()
    loss_soft = st.median(x_soft, softness=SOFTNESS, mode="smooth").sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="median_st")


def test_st_argmedian_forward_and_grad():
    """argmedian_st returns hard argmedian but soft gradients."""
    x = torch.tensor([0.1, 0.4, -0.2, 0.3, 0.5])
    weights = torch.arange(5.0)

    hard_forward = st.argmedian_st(x, softness=SOFTNESS, mode="smooth")
    expected = st.argmedian(x, mode="hard")
    np.testing.assert_allclose(
        hard_forward.detach().numpy(), expected.detach().numpy(), atol=1e-5
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = torch.dot(
        st.argmedian_st(x_st, softness=SOFTNESS, mode="smooth"), weights
    )
    loss_soft = torch.dot(
        st.argmedian(x_soft, softness=SOFTNESS, mode="smooth"), weights
    )

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argmedian_st")


def test_st_quantile_forward_and_grad():
    """quantile_st returns hard quantile but soft gradients."""
    x = torch.tensor([0.1, 0.4, -0.2, 0.3, 0.5])
    q = 0.75

    hard_forward = st.quantile_st(x, q, softness=SOFTNESS, mode="smooth")
    expected = torch.quantile(x, q)
    np.testing.assert_allclose(
        hard_forward.detach().numpy(), expected.numpy(), atol=1e-5
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = st.quantile_st(x_st, q, softness=SOFTNESS, mode="smooth").sum()
    loss_soft = st.quantile(x_soft, q, softness=SOFTNESS, mode="smooth").sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="quantile_st")


def test_st_argquantile_forward_and_grad():
    """argquantile_st returns hard argquantile but soft gradients."""
    x = torch.tensor([0.1, 0.4, -0.2, 0.3, 0.5])
    q = 0.75
    weights = torch.arange(5.0)

    hard_forward = st.argquantile_st(x, q, softness=SOFTNESS, mode="smooth")
    expected = st.argquantile(x, q, mode="hard")
    np.testing.assert_allclose(
        hard_forward.detach().numpy(), expected.detach().numpy(), atol=1e-5
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = torch.dot(
        st.argquantile_st(x_st, q, softness=SOFTNESS, mode="smooth"), weights
    )
    loss_soft = torch.dot(
        st.argquantile(x_soft, q, softness=SOFTNESS, mode="smooth"), weights
    )

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argquantile_st")


def test_st_topk_forward_and_grad():
    """topk_st supports tuple outputs and straight-through gradients."""
    x = torch.tensor([0.5, -0.1, 0.3, 0.8])

    hard_out = st.topk_st(x, k=3, softness=SOFTNESS, mode="smooth")
    expected_out = st.topk(x, k=3, mode="hard")
    np.testing.assert_allclose(
        hard_out.values.detach().numpy(), expected_out.values.detach().numpy()
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = st.topk_st(x_st, k=3, softness=SOFTNESS, mode="smooth").values.sum()
    loss_soft = st.topk(x_soft, k=3, mode="smooth", softness=SOFTNESS).values.sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="topk_st")


# ---------------------------------------------------------------------------
# st() generic wrapper
# ---------------------------------------------------------------------------


def test_st_generic_wrapper_forward():
    """st() wrapper: forward uses hard mode, backward uses soft mode."""
    x = torch.tensor([0.1, 0.4, -0.2, 0.3])

    max_st_via_wrapper = st.st(st.max)(x, softness=SOFTNESS, mode="smooth")
    expected = torch.max(x)
    np.testing.assert_allclose(
        max_st_via_wrapper.detach().numpy(), expected.numpy(), atol=1e-5
    )


def test_st_generic_wrapper_gradient():
    """st() wrapper gradient matches the soft gradient, not hard gradient."""
    x = torch.tensor([0.1, 0.4, -0.2, 0.3])

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = st.st(st.max)(x_st, softness=SOFTNESS, mode="smooth").sum()
    loss_soft = st.max(x_soft, softness=SOFTNESS, mode="smooth").sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="st() wrapper")


def test_st_generic_wrapper_sort():
    """st() wrapper works on sort: forward is hard sorted, backward is soft."""
    x = torch.tensor([0.3, 0.1, 0.4, -0.2])

    st_sorted = st.st(st.sort)(x, softness=SOFTNESS, mode="smooth")
    expected = torch.sort(x)
    np.testing.assert_allclose(
        st_sorted.values.detach().numpy(), expected.values.numpy(), atol=1e-5
    )

    x_st = x.clone().requires_grad_(True)
    x_soft = x.clone().requires_grad_(True)

    loss_st = st.st(st.sort)(x_st, softness=SOFTNESS, mode="smooth").values.sum()
    loss_soft = st.sort(x_soft, softness=SOFTNESS, mode="smooth").values.sum()

    (grad_st,) = torch.autograd.grad(loss_st, x_st)
    (grad_soft,) = torch.autograd.grad(loss_soft, x_soft)

    np.testing.assert_allclose(
        grad_st.numpy(), grad_soft.numpy(), atol=1e-5, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# st() mode passing variants
# ---------------------------------------------------------------------------


def test_st_no_mode_param_default():
    """st() on a function without mode param defaults to smooth backward."""
    x = torch.tensor([-0.5, 0.5, 1.5])

    @st.st
    def my_relu_prod(x, y, **kwargs):
        return st.relu(x, **kwargs) * st.relu(y, **kwargs)

    y = torch.tensor([1.0, 2.0, 0.5])
    result = my_relu_prod(x, y)
    expected = torch.relu(x) * torch.relu(y)
    np.testing.assert_allclose(
        result.detach().numpy(), expected.numpy(), atol=1e-5
    )


def test_st_no_mode_param_gradient():
    """st() without mode param gives non-zero gradient at hard-zero points."""

    @st.st
    def my_relu_prod(x, y, **kwargs):
        return st.relu(x, **kwargs) * st.relu(y, **kwargs)

    x = torch.tensor(-0.5, requires_grad=True)
    y = torch.tensor(2.0)

    # Forward is hard: relu(-0.5) * relu(2.0) = 0
    result = my_relu_prod(x, y)
    np.testing.assert_allclose(float(result), 0.0, atol=1e-5)

    # But gradient should be non-zero (from smooth backward)
    (grad,) = torch.autograd.grad(result, x)
    assert float(grad) != 0.0
    common.assert_finite(grad, msg="st() no mode param grad")


def test_st_no_mode_param_override_mode():
    """st() without mode param allows overriding mode via kwarg."""
    x = torch.tensor([-0.5, 0.5, 1.5])

    @st.st
    def my_abs(x, **kwargs):
        return st.abs(x, **kwargs)

    # c0 mode should also work
    result_c0 = my_abs(x, mode="c0")
    expected = torch.abs(x)
    np.testing.assert_allclose(
        result_c0.detach().numpy(), expected.numpy(), atol=1e-5
    )

    # Gradient with c0 mode
    x_c0 = x.clone().requires_grad_(True)
    (grad_c0,) = torch.autograd.grad(my_abs(x_c0, mode="c0").sum(), x_c0)
    common.assert_finite(grad_c0, msg="st() no mode c0 grad")

    # Gradient with smooth mode (explicit override, same as default)
    x_sm = x.clone().requires_grad_(True)
    (grad_smooth,) = torch.autograd.grad(my_abs(x_sm, mode="smooth").sum(), x_sm)
    common.assert_finite(grad_smooth, msg="st() no mode smooth grad")


def test_st_no_mode_param_matches_explicit_mode():
    """st() without mode param gives same results as explicit mode='smooth'."""
    x = torch.tensor([-0.5, 0.5, 1.5])

    @st.st
    def abs_no_mode(x, **kwargs):
        return st.abs(x, **kwargs)

    @st.st
    def abs_with_mode(x, mode="smooth", **kwargs):
        return st.abs(x, mode=mode, **kwargs)

    result_no = abs_no_mode(x, softness=SOFTNESS)
    result_with = abs_with_mode(x, softness=SOFTNESS)
    np.testing.assert_allclose(
        result_no.detach().numpy(), result_with.detach().numpy()
    )

    x_no = x.clone().requires_grad_(True)
    x_with = x.clone().requires_grad_(True)
    (grad_no,) = torch.autograd.grad(abs_no_mode(x_no, softness=SOFTNESS).sum(), x_no)
    (grad_with,) = torch.autograd.grad(
        abs_with_mode(x_with, softness=SOFTNESS).sum(), x_with
    )
    np.testing.assert_allclose(grad_no.numpy(), grad_with.numpy(), atol=1e-10)


def test_st_explicit_mode_positional():
    """st() with explicit mode param supports passing mode positionally."""
    x = torch.tensor([-0.5, 0.5, 1.5])

    @st.st
    def my_abs(x, mode="smooth", **kwargs):
        return st.abs(x, mode=mode, **kwargs)

    # mode as kwarg
    result_kw = my_abs(x, mode="c0", softness=SOFTNESS)
    # mode as positional
    result_pos = my_abs(x, "c0", softness=SOFTNESS)
    np.testing.assert_allclose(
        result_kw.detach().numpy(), result_pos.detach().numpy()
    )

    x_kw = x.clone().requires_grad_(True)
    x_pos = x.clone().requires_grad_(True)
    (grad_kw,) = torch.autograd.grad(
        my_abs(x_kw, mode="c0", softness=SOFTNESS).sum(), x_kw
    )
    (grad_pos,) = torch.autograd.grad(
        my_abs(x_pos, "c0", softness=SOFTNESS).sum(), x_pos
    )
    np.testing.assert_allclose(grad_kw.numpy(), grad_pos.numpy(), atol=1e-10)


def test_st_explicit_mode_nondefault():
    """st() respects a non-default mode in the function signature."""
    x = torch.tensor([-0.5, 0.5, 1.5])

    @st.st
    def my_abs_c0(x, mode="c0", **kwargs):
        return st.abs(x, mode=mode, **kwargs)

    # Should use c0 by default (not smooth)
    x_default = x.clone().requires_grad_(True)
    (grad_default,) = torch.autograd.grad(
        my_abs_c0(x_default, softness=SOFTNESS).sum(), x_default
    )

    @st.st
    def my_abs_smooth(x, mode="smooth", **kwargs):
        return st.abs(x, mode=mode, **kwargs)

    x_smooth = x.clone().requires_grad_(True)
    (grad_smooth,) = torch.autograd.grad(
        my_abs_smooth(x_smooth, softness=SOFTNESS).sum(), x_smooth
    )

    # c0 and smooth have different gradients
    assert not np.allclose(grad_default.numpy(), grad_smooth.numpy(), atol=1e-3)
