from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import softtorch as st

from . import common


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_selection_error_paths():
    x = torch.arange(6.0).reshape(2, 3)
    bad_soft = torch.ones((2, 3))
    with pytest.raises(Exception):
        st.take_along_dim(x, bad_soft, dim=0)  # rank mismatch

    with pytest.raises(Exception):
        st.take(x, torch.ones((2, 2, 3)), dim=None)

    with pytest.raises(ValueError):
        st.index_select(x, torch.ones((4,)), dim=1)  # dim size mismatch

    with pytest.raises(Exception):
        st.argmax(x, dim=5)


# ---------------------------------------------------------------------------
# Hard paths
# ---------------------------------------------------------------------------


def test_selection_helpers_hard_paths():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    one_hot = F.one_hot(torch.tensor([[0, 2], [1, 0]]), x.shape[1]).to(torch.float)
    out = st.take_along_dim(x, one_hot, dim=1)
    expected = torch.take_along_dim(x, torch.tensor([[0, 2], [1, 0]]), dim=1)
    common.assert_allclose(out, expected)

    flat_idx = F.one_hot(torch.tensor([0, 3]), x.numel()).to(torch.float)
    taken = st.take(x, flat_idx, dim=None)
    expected_take = torch.take(x, torch.tensor([0, 3]))
    common.assert_allclose(taken, expected_take)

    idx = F.one_hot(torch.tensor(1), 3).to(torch.float)
    dyn = st.index_select(x, idx, dim=1, keepdim=True)
    expected_idx = torch.index_select(x, 1, torch.tensor([1]))
    common.assert_allclose(dyn, expected_idx)


# ---------------------------------------------------------------------------
# Parametrized take_along_dim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=str)
@pytest.mark.parametrize(
    "shape, dim",
    [((4,), 0), ((2, 3), 0), ((2, 3), 1), ((2, 3), -1)],
)
def test_take_along_dim_parametrized(dtype, shape, dim):
    x = common.make_tensor(shape, dtype)
    n = x.shape[dim]
    idx_int = torch.zeros(shape, dtype=torch.long)
    idx_one_hot = F.one_hot(idx_int, n).to(dtype)

    out = st.take_along_dim(x, idx_one_hot, dim=dim)
    expected = torch.take_along_dim(x, idx_int, dim=dim)
    common.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Parametrized index_select
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=str)
@pytest.mark.parametrize("idx_val", [0, 1, 2])
def test_index_select_parametrized(dtype, idx_val):
    x = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=dtype)
    idx = F.one_hot(torch.tensor(idx_val), x.shape[1]).to(dtype)
    out = st.index_select(x, idx, dim=1, keepdim=True)
    expected = torch.index_select(x, 1, torch.tensor([idx_val]))
    common.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Parametrized narrow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start_val", [0, 1])
def test_narrow_parametrized(start_val):
    x = torch.arange(6.0)
    start = F.one_hot(torch.tensor(start_val), x.shape[0]).to(torch.float)
    out = st.narrow(x, start, length=3, dim=0)
    expected = torch.narrow(x, 0, start_val, 3)
    common.assert_allclose(out, expected, tol=1e-5)


# ---------------------------------------------------------------------------
# Soft interpolation
# ---------------------------------------------------------------------------


def test_take_along_dim_soft_interpolation():
    x = torch.tensor([10.0, 20.0, 30.0])
    soft_idx = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    out = st.take_along_dim(x, soft_idx, dim=0)
    expected = torch.tensor([20.0])
    common.assert_allclose(out, expected, tol=1e-5)


def test_index_select_soft_interpolation():
    """Non-one-hot soft index interpolates along dimension."""
    x = torch.tensor([[10.0], [30.0]])
    soft_idx = torch.tensor([0.5, 0.5])  # equal weight on both rows
    out = st.index_select(x, soft_idx, dim=0)
    expected = torch.tensor([[20.0]])  # 0.5*10 + 0.5*30
    common.assert_allclose(out, expected, tol=1e-5)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_take_along_dim_gradient():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    logits = torch.tensor([[[0.0, 5.0, 0.0]], [[5.0, 0.0, 0.0]]])
    soft_idx = torch.softmax(logits, dim=-1)
    out = st.take_along_dim(x, soft_idx, dim=1)
    out.sum().backward()
    common.assert_finite(x.grad, msg="take_along_dim gradient wrt x")


def test_take_along_dim_gradient_wrt_index():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    logits = torch.tensor([[[0.0, 5.0, 0.0]], [[5.0, 0.0, 0.0]]], requires_grad=True)

    soft_idx = torch.softmax(logits, dim=-1)
    out = st.take_along_dim(x, soft_idx, dim=1)
    out.sum().backward()
    common.assert_finite(logits.grad, msg="take_along_dim gradient wrt index")


def test_index_select_gradient():
    x = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], requires_grad=True)
    idx = F.one_hot(torch.tensor(1), x.shape[1]).to(torch.float)
    out = st.index_select(x, idx, dim=1)
    out.sum().backward()
    common.assert_finite(x.grad, msg="index_select gradient")


def test_narrow_gradient():
    x = torch.arange(6.0, requires_grad=True, dtype=torch.float)
    start = F.one_hot(torch.tensor(1), x.shape[0]).to(torch.float)
    out = st.narrow(x, start, length=3, dim=0)
    out.sum().backward()
    common.assert_finite(x.grad, msg="narrow gradient")


def test_where_gradient():
    condition = torch.tensor([0.8, 0.2, 0.5])

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.zeros(3)
    out_x = st.where(condition, x, y)
    out_x.sum().backward()
    common.assert_finite(x.grad, msg="where gradient wrt x")
    np.testing.assert_allclose(x.grad.numpy(), condition.numpy(), atol=1e-5)

    x2 = torch.zeros(3)
    y2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out_y = st.where(condition, x2, y2)
    out_y.sum().backward()
    common.assert_finite(y2.grad, msg="where gradient wrt y")
    np.testing.assert_allclose(y2.grad.numpy(), (1.0 - condition).numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient vs finite differences
# ---------------------------------------------------------------------------


def test_take_along_dim_grad_vs_finite_diff():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    soft_idx = torch.softmax(
        torch.tensor([[[0.0, 5.0, 0.0]], [[5.0, 0.0, 0.0]]], dtype=torch.float64),
        dim=-1,
    )

    def loss(arr):
        return st.take_along_dim(arr, soft_idx, dim=1).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg="take_along_dim")


def test_where_grad_vs_finite_diff():
    condition = torch.tensor([0.8, 0.2, 0.5], dtype=torch.float64)
    y = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)

    def loss(arr):
        return st.where(condition, arr, y).sum()

    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    common.assert_grad_matches_finite_diff(loss, x, msg="where")


# ---------------------------------------------------------------------------
# End-to-end differentiable pipeline test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_end_to_end_argmax_take(mode):
    """End-to-end pipeline: argmax -> take_along_dim is differentiable."""
    x = torch.tensor(
        [[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]], dtype=torch.float64, requires_grad=True
    )

    def pipeline(z):
        soft_idx = st.argmax(z, dim=-1, keepdim=True, mode=mode, softness=1.0)
        selected = st.take_along_dim(z, soft_idx, dim=-1)
        return selected.sum()

    # Forward pass produces finite output
    out = pipeline(x)
    common.assert_finite(out, msg=f"pipeline output mode={mode}")

    # Backward pass produces finite gradients
    out.backward()
    common.assert_finite(x.grad, msg=f"pipeline gradient mode={mode}")

    # Gradient vs finite differences
    x_no_grad = x.detach().clone()
    common.assert_grad_matches_finite_diff(pipeline, x_no_grad, msg=f"pipeline mode={mode}")
