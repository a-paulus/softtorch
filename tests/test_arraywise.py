from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import softtorch as st

from . import common


FLOAT_DTYPES = (torch.float32, torch.float64)
NEAR_HARD_SOFTNESS = common.NEAR_HARD_SOFTNESS
STABILITY_SOFTNESS = common.STABILITY_SOFTNESS

SHAPES = [(4,), (2, 3)]
DIMS = [None, -1, 0]
MODES = ("smooth", "c0", "c1", "c2", "hard", "_hard")
SOFTNESSES = [NEAR_HARD_SOFTNESS, STABILITY_SOFTNESS]
KEEPDIMS = [False, True]

VALUE_METHODS = ["softsort", "neuralsort", "fast_soft_sort", "ot"]
SORT_VALUE_METHODS = VALUE_METHODS + ["sorting_network"]
ARG_METHODS = ["softsort", "neuralsort", "ot"]


def _valid_dim(shape, dim):
    if dim is None:
        return True
    return -len(shape) <= dim < len(shape)


def _skip_unsupported(method, mode):
    """Skip unsupported method+mode combinations."""
    if method == "smooth_sort":
        pytest.skip("smooth_sort not supported in SoftTorch (ESP+LBFGS is SoftJAX-only)")
    if method == "fast_soft_sort" and mode in ("hard", "_hard"):
        return  # hard modes bypass projection, always fine


def _build_fn_method_params(fn_names, method_map):
    params = []
    for fn_name in fn_names:
        for method in method_map.get(fn_name, ARG_METHODS):
            params.append((fn_name, method))
    return params


FUNCTION_METHODS = {
    "max": SORT_VALUE_METHODS,
    "min": SORT_VALUE_METHODS,
    "argmax": ARG_METHODS + ["sorting_network"],
    "argmin": ARG_METHODS + ["sorting_network"],
    "sort": SORT_VALUE_METHODS,
    "argsort": ARG_METHODS + ["sorting_network"],
    "rank": SORT_VALUE_METHODS,
    "median": ARG_METHODS,
    "argmedian": ARG_METHODS + ["sorting_network"],
    "quantile": SORT_VALUE_METHODS,
    "argquantile": ARG_METHODS + ["sorting_network"],
}


# ---------------------------------------------------------------------------
# max / min / argmax / argmin parametric sweep
# ---------------------------------------------------------------------------

_MAX_MIN_PARAMS = _build_fn_method_params(
    ["max", "argmax", "min", "argmin"], FUNCTION_METHODS
)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("keepdim", KEEPDIMS)
@pytest.mark.parametrize(
    "fn_name, method", _MAX_MIN_PARAMS, ids=[f"{fn}-{m}" for fn, m in _MAX_MIN_PARAMS]
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
def test_max_min(dtype, shape, dim, keepdim, fn_name, method, mode, softness):
    _skip_unsupported(method, mode)
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")

    x = common.make_tensor(shape, dtype)
    fn = getattr(st, fn_name)
    ot_kwargs = common.ot_kwargs_for_method(method, softness)

    # argmax/argmin don't have keepdim=True with dim=None (returns flat)
    kwargs = dict(
        dim=dim,
        keepdim=keepdim,
        softness=softness,
        mode=mode,
        method=method,
        ot_kwargs=ot_kwargs,
    )
    # max/min with dim=None don't take keepdim
    if dim is None and fn_name in ("max", "min"):
        kwargs.pop("keepdim")

    out = fn(x, **kwargs)

    if fn_name in ("max", "min") and dim is not None:
        assert not torch.any(torch.isnan(out.values)), f"NaN in values of {fn_name}"
        if out.indices is not None:
            common.assert_simplex(out.indices, atol=common.TOLERANCE)
    elif "arg" in fn_name:
        assert not torch.any(torch.isnan(out)), f"NaN in output of {fn_name}"
        common.assert_simplex(out, atol=common.TOLERANCE)
    else:
        assert not torch.any(torch.isnan(out)), f"NaN in output of {fn_name}"

    if softness == NEAR_HARD_SOFTNESS:
        hard_kwargs = dict(dim=dim, keepdim=keepdim, mode="hard")
        if dim is None and fn_name in ("max", "min"):
            hard_kwargs.pop("keepdim")
        out_hard = fn(x, **hard_kwargs)

        if method == "ot":
            retry_kwargs = dict(dim=dim, keepdim=keepdim, softness=softness, mode=mode)
            if dim is None and fn_name in ("max", "min"):
                retry_kwargs.pop("keepdim")
            if fn_name in ("max", "min") and dim is not None:
                def _check(result):
                    common.assert_allclose(result.values, out_hard.values, tol=common.TOLERANCE)
                out = common.call_with_ot_retry(
                    fn, x, method=method, _check_fn=_check, **retry_kwargs,
                )
            else:
                out = common.call_with_ot_retry(
                    fn, x, method=method, _expected=out_hard, **retry_kwargs,
                )

        if fn_name in ("max", "min") and dim is not None:
            common.assert_allclose(out.values, out_hard.values, tol=common.TOLERANCE)
        elif "arg" in fn_name:
            common.assert_allclose(out, out_hard, tol=common.TOLERANCE)
        else:
            common.assert_allclose(out, out_hard, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# sort / argsort / rank parametric sweep
# ---------------------------------------------------------------------------

_SORT_RANK_PARAMS = _build_fn_method_params(
    ["sort", "argsort", "rank"], FUNCTION_METHODS
)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize(
    "fn_name, method",
    _SORT_RANK_PARAMS,
    ids=[f"{fn}-{m}" for fn, m in _SORT_RANK_PARAMS],
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
def test_sort_rank(dtype, shape, dim, descending, fn_name, method, mode, softness):
    _skip_unsupported(method, mode)
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    # sort/argsort/rank don't support dim=None in PyTorch
    if dim is None:
        pytest.skip("dim=None not supported for sort/argsort/rank in SoftTorch")

    x = common.make_tensor(shape, dtype)
    fn = getattr(st, fn_name)
    ot_kwargs = common.ot_kwargs_for_method(method, softness)

    out = fn(
        x,
        dim=dim,
        descending=descending,
        softness=softness,
        mode=mode,
        method=method,
        ot_kwargs=ot_kwargs,
    )

    if fn_name == "sort":
        assert not torch.any(torch.isnan(out.values)), "NaN in sort values"
        if out.indices is not None:
            common.assert_simplex(out.indices, atol=common.TOLERANCE)
    elif fn_name == "argsort":
        assert not torch.any(torch.isnan(out)), "NaN in argsort"
        common.assert_simplex(out, atol=common.TOLERANCE)
    else:
        assert not torch.any(torch.isnan(out)), "NaN in rank"

    if softness == NEAR_HARD_SOFTNESS:
        out_hard = fn(x, dim=dim, descending=descending, mode="hard")
        if method == "ot":
            if fn_name == "sort":
                def _check(result):
                    common.assert_allclose(result.values, out_hard.values, tol=common.TOLERANCE)
                out = common.call_with_ot_retry(
                    fn, x, dim=dim, descending=descending,
                    softness=softness, mode=mode, method=method,
                    _check_fn=_check,
                )
            else:
                out = common.call_with_ot_retry(
                    fn, x, dim=dim, descending=descending,
                    softness=softness, mode=mode, method=method,
                    _expected=out_hard,
                )
        if fn_name == "sort":
            common.assert_allclose(out.values, out_hard.values, tol=common.TOLERANCE)
        else:
            common.assert_allclose(out, out_hard, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# median / argmedian parametric sweep
# ---------------------------------------------------------------------------

_MEDIAN_PARAMS = _build_fn_method_params(["median", "argmedian"], FUNCTION_METHODS)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("keepdim", KEEPDIMS)
@pytest.mark.parametrize(
    "fn_name, method", _MEDIAN_PARAMS, ids=[f"{fn}-{m}" for fn, m in _MEDIAN_PARAMS]
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
def test_median(dtype, shape, dim, keepdim, fn_name, method, mode, softness):
    _skip_unsupported(method, mode)
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")

    x = common.make_tensor(shape, dtype)
    fn = getattr(st, fn_name)
    ot_kwargs = common.ot_kwargs_for_method(method, softness)

    kwargs = dict(
        dim=dim,
        keepdim=keepdim,
        softness=softness,
        mode=mode,
        method=method,
        ot_kwargs=ot_kwargs,
    )
    if dim is None and fn_name == "median":
        kwargs.pop("keepdim")

    out = fn(x, **kwargs)

    if fn_name == "median" and dim is not None:
        assert not torch.any(torch.isnan(out.values))
        common.assert_simplex(out.indices, atol=common.TOLERANCE)
    elif "arg" in fn_name:
        assert not torch.any(torch.isnan(out))
        common.assert_simplex(out, atol=common.TOLERANCE)
    else:
        assert not torch.any(torch.isnan(out))

    if softness == NEAR_HARD_SOFTNESS:
        hard_kwargs = dict(dim=dim, keepdim=keepdim, mode="hard")
        if dim is None and fn_name == "median":
            hard_kwargs.pop("keepdim")
        out_hard = fn(x, **hard_kwargs)

        if method == "ot":
            retry_kwargs = dict(dim=dim, keepdim=keepdim, softness=softness, mode=mode)
            if dim is None and fn_name == "median":
                retry_kwargs.pop("keepdim")
            if fn_name == "median" and dim is not None:
                def _check(result):
                    common.assert_allclose(result.values, out_hard.values, tol=common.TOLERANCE)
                out = common.call_with_ot_retry(
                    fn, x, method=method, _check_fn=_check, **retry_kwargs,
                )
            else:
                out = common.call_with_ot_retry(
                    fn, x, method=method, _expected=out_hard, **retry_kwargs,
                )

        if fn_name == "median" and dim is not None:
            common.assert_allclose(out.values, out_hard.values, tol=common.TOLERANCE)
        elif "arg" in fn_name:
            common.assert_allclose(out, out_hard, tol=common.TOLERANCE)
        else:
            common.assert_allclose(out, out_hard, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# quantile / argquantile parametric sweep
# ---------------------------------------------------------------------------

_QUANTILE_PARAMS = _build_fn_method_params(["quantile", "argquantile"], FUNCTION_METHODS)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("keepdim", KEEPDIMS)
@pytest.mark.parametrize(
    "fn_name, method",
    _QUANTILE_PARAMS,
    ids=[f"{fn}-{m}" for fn, m in _QUANTILE_PARAMS],
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_quantile(
    dtype, shape, dim, keepdim, fn_name, method, mode, softness, interpolation, q
):
    _skip_unsupported(method, mode)
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")

    x = common.make_tensor(shape, dtype)
    fn = getattr(st, fn_name)
    ot_kwargs = common.ot_kwargs_for_method(method, softness)

    kwargs = dict(
        q=q,
        dim=dim,
        keepdim=keepdim,
        softness=softness,
        mode=mode,
        method=method,
        interpolation=interpolation,
        ot_kwargs=ot_kwargs,
    )

    out = fn(x, **kwargs)
    if isinstance(out, torch.Tensor):
        assert not torch.any(torch.isnan(out))
    if "arg" in fn_name:
        common.assert_simplex(out, atol=common.TOLERANCE)

    if softness == NEAR_HARD_SOFTNESS:
        hard_kwargs = dict(
            q=q, dim=dim, keepdim=keepdim, mode="hard", interpolation=interpolation
        )
        out_hard = fn(x, **hard_kwargs)
        if method == "ot":
            out = common.call_with_ot_retry(
                fn, x, q, dim=dim, keepdim=keepdim,
                softness=softness, mode=mode, method=method,
                interpolation=interpolation, _expected=out_hard,
            )
        common.assert_allclose(out, out_hard, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# top_k parametric sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
@pytest.mark.parametrize("method", SORT_VALUE_METHODS)
def test_top_k(dtype, shape, k, dim, mode, softness, method):
    _skip_unsupported(method, mode)
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    if k > shape[dim]:
        pytest.skip(f"k={k} exceeds dim size {shape[dim]}")

    x = common.make_tensor(shape, dtype)
    ot_kwargs = common.ot_kwargs_for_method(method, softness)
    out = st.topk(
        x,
        k=k,
        dim=dim,
        mode=mode,
        method=method,
        softness=softness,
        ot_kwargs=ot_kwargs,
    )
    vals = out.values
    soft_idx = out.indices

    assert not torch.any(torch.isnan(vals))
    if soft_idx is not None:
        assert not torch.any(torch.isnan(soft_idx))
        common.assert_simplex(soft_idx, atol=common.TOLERANCE)

    if softness == NEAR_HARD_SOFTNESS:
        hard_out = st.topk(x, k=k, dim=dim, mode="hard")
        if method == "ot":
            def _check(result):
                common.assert_allclose(result.values, hard_out.values, tol=common.TOLERANCE)
                if result.indices is not None:
                    common.assert_allclose(result.indices, hard_out.indices, tol=common.TOLERANCE)
            out = common.call_with_ot_retry(
                st.topk, x, k=k, dim=dim,
                softness=softness, mode=mode, method=method,
                _check_fn=_check,
            )
            vals = out.values
        common.assert_allclose(vals, hard_out.values, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# all / any
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES)
def test_all_any(dtype, shape):
    ones = torch.ones(shape, dtype=dtype)
    zeros = torch.zeros(shape, dtype=dtype)

    dims = [-1]
    if len(shape) > 1:
        dims.append(0)

    for dim in dims:
        for out in (
            st.all(ones, dim=dim),
            st.all(zeros, dim=dim),
            st.any(ones, dim=dim),
            st.any(zeros, dim=dim),
        ):
            assert not torch.any(torch.isnan(out))
            assert torch.all(out >= -common.TOLERANCE)
            assert torch.all(out <= 1.0 + common.TOLERANCE)

    probs = common.make_tensor(shape, dtype, softbool=True)
    for dim in dims:
        out_all = st.all(probs, dim=dim)
        out_any = st.any(probs, dim=dim)
        assert out_all.ndim == probs.ndim - 1
        assert out_any.ndim == probs.ndim - 1
        common.assert_softbool(out_all)
        common.assert_softbool(out_any)
        assert np.all(out_all.numpy() <= out_any.numpy() + 1e-6)


def test_all_any_torch_parity():
    x_bool = torch.tensor([1.0, 1.0, 1.0])
    assert float(st.all(x_bool, dim=-1)) == pytest.approx(1.0, abs=1e-5)
    assert float(st.any(x_bool, dim=-1)) == pytest.approx(1.0, abs=1e-5)

    x_mixed = torch.tensor([1.0, 0.0, 1.0])
    assert float(st.all(x_mixed, dim=-1)) == pytest.approx(0.0, abs=1e-5)
    assert float(st.any(x_mixed, dim=-1)) == pytest.approx(1.0, abs=1e-5)

    x_zeros = torch.tensor([0.0, 0.0, 0.0])
    assert float(st.all(x_zeros, dim=-1)) == pytest.approx(0.0, abs=1e-5)
    assert float(st.any(x_zeros, dim=-1)) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# PyTorch parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 2)])
@pytest.mark.parametrize("dim", [None, -1, 0])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
def test_max_torch_parity(shape, dim, keepdim):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    if dim is None:
        soft_out = st.max(x, dim=None, mode="hard")
        torch_out = torch.max(x)
        common.assert_torch_parity(soft_out, torch_out, msg="max dim=None")
    else:
        soft_out = st.max(x, dim=dim, keepdim=keepdim, mode="hard")
        torch_out = torch.max(x, dim=dim, keepdim=keepdim)
        common.assert_torch_parity(soft_out.values, torch_out.values, msg="max values")


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 2)])
@pytest.mark.parametrize("dim", [None, -1, 0])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
def test_min_torch_parity(shape, dim, keepdim):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    if dim is None:
        soft_out = st.min(x, dim=None, mode="hard")
        torch_out = torch.min(x)
        common.assert_torch_parity(soft_out, torch_out, msg="min dim=None")
    else:
        soft_out = st.min(x, dim=dim, keepdim=keepdim, mode="hard")
        torch_out = torch.min(x, dim=dim, keepdim=keepdim)
        common.assert_torch_parity(soft_out.values, torch_out.values, msg="min values")


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
def test_argmax_torch_parity(shape, dim):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    soft_out = st.argmax(x, dim=dim, mode="hard")
    torch_idx = torch.argmax(x, dim=dim)
    expected = F.one_hot(torch_idx, x.shape[dim]).to(torch.float64)
    common.assert_torch_parity(soft_out, expected, msg="argmax")


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
def test_argmin_torch_parity(shape, dim):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    soft_out = st.argmin(x, dim=dim, mode="hard")
    torch_idx = torch.argmin(x, dim=dim)
    expected = F.one_hot(torch_idx, x.shape[dim]).to(torch.float64)
    common.assert_torch_parity(soft_out, expected, msg="argmin")


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_torch_parity(shape, dim, descending):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    soft_out = st.sort(x, dim=dim, descending=descending, mode="hard")
    torch_out = torch.sort(x, dim=dim, descending=descending)
    common.assert_torch_parity(soft_out.values, torch_out.values, msg="sort")


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("descending", [False, True])
def test_argsort_torch_parity(shape, dim, descending):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    soft_out = st.argsort(x, dim=dim, descending=descending, mode="hard")
    torch_idx = torch.argsort(x, dim=dim, descending=descending)
    expected = F.one_hot(torch_idx, x.shape[dim]).to(torch.float64)
    common.assert_torch_parity(soft_out, expected, msg="argsort")


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
def test_median_torch_parity(shape, dim, keepdim):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    soft_out = st.median(x, dim=dim, keepdim=keepdim, mode="hard")
    torch_out = torch.median(x, dim=dim, keepdim=keepdim)
    common.assert_torch_parity(soft_out.values, torch_out.values, msg="median")


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_quantile_torch_parity(shape, dim, keepdim, q):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    soft_out = st.quantile(x, q, dim=dim, keepdim=keepdim, mode="hard")
    torch_out = torch.quantile(x, q, dim=dim, keepdim=keepdim)
    common.assert_torch_parity(soft_out, torch_out, msg=f"quantile q={q}")


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("k", [1, 2])
def test_top_k_torch_parity(shape, dim, k):
    if k > shape[dim]:
        pytest.skip(f"k={k} exceeds dim size {shape[dim]}")
    x = common.make_tensor(shape, torch.float64)
    soft_out = st.topk(x, k=k, dim=dim, mode="hard")
    torch_out = torch.topk(x, k=k, dim=dim)
    common.assert_torch_parity(soft_out.values, torch_out.values, msg="top_k values")


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
def test_argmin_shape(shape, dim, keepdim):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    soft_out = st.argmin(x, dim=dim, keepdim=keepdim, mode="hard")
    torch_idx = torch.argmin(x, dim=dim, keepdim=keepdim)
    assert soft_out.shape[:-1] == torch_idx.shape
    assert soft_out.shape[-1] == x.shape[dim]


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
def test_argmedian_shape(shape, dim, keepdim):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    soft_out = st.argmedian(x, dim=dim, keepdim=keepdim, mode="hard")
    median_out = st.median(x, dim=dim, keepdim=keepdim, mode="hard")
    assert soft_out.shape[:-1] == median_out.values.shape
    assert soft_out.shape[-1] == x.shape[dim]


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
@pytest.mark.parametrize("q", [0.25, 0.75])
def test_argquantile_shape(shape, dim, keepdim, q):
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    soft_out = st.argquantile(x, q, dim=dim, keepdim=keepdim, mode="hard")
    torch_out = torch.quantile(x, q, dim=dim, keepdim=keepdim)
    assert soft_out.shape[:-1] == torch_out.shape
    assert soft_out.shape[-1] == x.shape[dim]


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("k", [1, 2])
def test_top_k_output_shapes(shape, k):
    x = common.make_tensor(shape, torch.float64)
    if k > shape[-1]:
        pytest.skip(f"k={k} exceeds last dim size {shape[-1]}")
    soft_out = st.topk(x, k=k, mode="hard")
    torch_out = torch.topk(x, k=k)
    assert soft_out.values.shape == torch_out.values.shape
    assert soft_out.indices.shape[:-1] == torch_out.indices.shape
    assert soft_out.indices.shape[-1] == x.shape[-1]


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
def test_rank_hard_shape(shape, dim):
    """st.rank(x, mode='hard') has correct shape and produces valid rank values."""
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    out = st.rank(x, dim=dim, mode="hard")
    assert out.shape == x.shape, f"Unexpected rank shape {out.shape}"
    common.assert_finite(out, msg="rank hard")


@pytest.mark.parametrize("fn_name", ["sort", "rank"])
@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
def test_sort_rank_output_shape(fn_name, shape, dim):
    """sort/rank output shape must match input shape."""
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    fn = getattr(st, fn_name)
    out = fn(x, dim=dim, mode="hard")
    if fn_name == "sort":
        assert out.values.shape == x.shape
    else:
        assert out.shape == x.shape


@pytest.mark.parametrize(
    "fn_name", ["median", "quantile"],
)
@pytest.mark.parametrize("keepdim", KEEPDIMS)
@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("dim", [-1, 0])
def test_value_reduction_output_shape(fn_name, keepdim, shape, dim):
    """Value reduction output shape must match torch equivalent."""
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    fn = getattr(st, fn_name)
    if fn_name == "quantile":
        out = fn(x, 0.5, dim=dim, keepdim=keepdim, mode="hard")
        expected = torch.quantile(x, 0.5, dim=dim, keepdim=keepdim)
    else:
        out = fn(x, dim=dim, keepdim=keepdim, mode="hard")
        expected = torch.median(x, dim=dim, keepdim=keepdim)
    if fn_name == "quantile":
        assert out.shape == expected.shape
    else:
        assert out.values.shape == expected.values.shape


@pytest.mark.parametrize("fn_name", ["max", "min"])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
def test_max_min_output_shape(fn_name, keepdim):
    """Verify output shape for max/min matches torch equivalent."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    fn = getattr(st, fn_name)
    torch_fn = getattr(torch, fn_name)
    # dim=None
    out = fn(x, dim=None, mode="hard")
    expected = torch_fn(x)
    assert out.shape == expected.shape
    # dim=0, 1, -1
    for dim in [0, 1, -1]:
        out = fn(x, dim=dim, keepdim=keepdim, mode="hard")
        expected = torch_fn(x, dim=dim, keepdim=keepdim)
        assert out.values.shape == expected.values.shape


# ---------------------------------------------------------------------------
# Gradient finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_arraywise_gradient_finite(fn_name, mode):
    x = common.gradient_input((4,), torch.float32)
    fn = getattr(st, fn_name)

    if fn_name in ("max", "min", "sort", "median"):
        out = fn(x, dim=-1, mode=mode, softness=1.0)
        loss = out.values.sum()
    else:
        out = fn(x, dim=-1, mode=mode, softness=1.0)
        loss = out.sum()
    loss.backward()
    common.assert_finite(x.grad, msg=f"{fn_name} mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_quantile_gradient_finite(mode):
    x = common.gradient_input((5,), torch.float32)
    out = st.quantile(x, 0.5, dim=-1, mode=mode, softness=1.0)
    out.sum().backward()
    common.assert_finite(x.grad, msg=f"quantile mode={mode}")


@pytest.mark.parametrize("fn_name", ["argmax", "argmin", "argsort", "argmedian"])
@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_arg_gradient_finite(fn_name, mode):
    x = common.gradient_input((4,), torch.float32)
    weights = torch.arange(4.0)
    fn = getattr(st, fn_name)

    out = fn(x, dim=-1, mode=mode, softness=1.0)
    loss = (out * weights).sum()
    loss.backward()
    common.assert_finite(x.grad, msg=f"{fn_name} mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_argquantile_gradient_finite(mode):
    x = common.gradient_input((5,), torch.float32)
    weights = torch.arange(5.0)
    out = st.argquantile(x, 0.5, dim=-1, mode=mode, softness=1.0)
    loss = (out * weights).sum()
    loss.backward()
    common.assert_finite(x.grad, msg=f"argquantile mode={mode}")


# ---------------------------------------------------------------------------
# Gradient vs finite differences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_value_grad_vs_finite_diff(fn_name, mode):
    x = common.make_tensor((5,), torch.float64)
    fn = getattr(st, fn_name)

    if fn_name in ("max", "min", "sort", "median"):

        def loss(z):
            return fn(z, dim=-1, mode=mode, softness=1.0).values.sum()

    else:

        def loss(z):
            return fn(z, dim=-1, mode=mode, softness=1.0).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"{fn_name} mode={mode}")


@pytest.mark.parametrize("fn_name", ["argmax", "argmin", "argsort", "argmedian"])
@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_arg_grad_vs_finite_diff(fn_name, mode):
    x = common.make_tensor((5,), torch.float64)
    weights = torch.arange(5.0, dtype=torch.float64)
    fn = getattr(st, fn_name)

    def loss(z):
        out = fn(z, dim=-1, mode=mode, softness=1.0)
        return (out * weights).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"{fn_name} mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_quantile_grad_vs_finite_diff(mode):
    x = common.make_tensor((5,), torch.float64)

    def loss(z):
        return st.quantile(z, 0.5, dim=-1, mode=mode, softness=1.0).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"quantile mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_argquantile_grad_vs_finite_diff(mode):
    x = common.make_tensor((5,), torch.float64)
    weights = torch.arange(5.0, dtype=torch.float64)

    def loss(z):
        out = st.argquantile(z, 0.5, dim=-1, mode=mode, softness=1.0)
        return (out * weights).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"argquantile mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_top_k_grad_vs_finite_diff(mode):
    x = common.make_tensor((5,), torch.float64)

    def loss(z):
        return st.topk(z, k=3, dim=-1, mode=mode, softness=1.0).values.sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"top_k mode={mode}")


# ---------------------------------------------------------------------------
# OT validity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 10])
@pytest.mark.parametrize("fn_name", ["argmax", "argmin"])
def test_ot_produces_simplex(n, fn_name):
    """OT (transport polytope): verify it produces valid output."""
    torch.manual_seed(42)
    x = torch.randn(n, dtype=torch.float64)
    fn = getattr(st, fn_name)
    soft = fn(x, dim=-1, mode="smooth", method="ot", softness=10.0)
    common.assert_simplex(soft, atol=1e-4)
    common.assert_finite(soft, msg=f"OT {fn_name} n={n}")


# ---------------------------------------------------------------------------
# fast_soft_sort + smooth mode (entropic PAV)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank"])
def test_fast_soft_sort_smooth_mode(fn_name):
    """fast_soft_sort+smooth mode works (entropic PAV)."""
    x = common.make_tensor((4,), torch.float64)
    fn = getattr(st, fn_name)
    out = fn(x, dim=-1, mode="smooth", method="fast_soft_sort", softness=1.0)
    if fn_name in ("max", "min", "sort"):
        values = out.values
        assert not torch.any(torch.isnan(values)), f"NaN in {fn_name} fast_soft_sort+smooth values"
        common.assert_finite(values, msg=f"{fn_name} fast_soft_sort+smooth")
    else:
        assert not torch.any(torch.isnan(out)), f"NaN in {fn_name} fast_soft_sort+smooth"
        common.assert_finite(out, msg=f"{fn_name} fast_soft_sort+smooth")


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank"])
def test_fast_soft_sort_smooth_gradient_finite(fn_name):
    """Gradients through fast_soft_sort+smooth mode must be finite."""
    x = common.gradient_input((4,), torch.float32)
    fn = getattr(st, fn_name)

    if fn_name in ("max", "min", "sort"):
        out = fn(x, dim=-1, mode="smooth", method="fast_soft_sort", softness=1.0)
        loss = out.values.sum()
    else:
        out = fn(x, dim=-1, mode="smooth", method="fast_soft_sort", softness=1.0)
        loss = out.sum()
    loss.backward()
    common.assert_finite(x.grad, msg=f"{fn_name} fast_soft_sort+smooth gradient")


# ---------------------------------------------------------------------------
# Error path tests (issue 15)
# ---------------------------------------------------------------------------


def test_invalid_mode():
    x = torch.tensor([3.0, 1.0, 2.0])
    with pytest.raises((ValueError, KeyError)):
        st.sort(x, mode="invalid")
    with pytest.raises((ValueError, KeyError)):
        st.argmax(x, mode="invalid")


def test_invalid_method():
    x = torch.tensor([3.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        st.sort(x, method="invalid")
    with pytest.raises(ValueError):
        st.argsort(x, method="invalid")


def test_softness_validation():
    x = torch.tensor([3.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="softness must be positive"):
        st.argmax(x, softness=0.0)
    with pytest.raises(ValueError, match="softness must be positive"):
        st.sort(x, method="fast_soft_sort", mode="c0", softness=-1.0)
    with pytest.raises(ValueError, match="softness must be positive"):
        st.greater(x, torch.tensor(2.0), softness=0.0)


def test_scalar_tensor_softness_validation():
    x = torch.tensor([3.0, 1.0, 2.0])
    softness = torch.tensor(0.5)

    out = st.argmax(x, softness=softness)
    common.assert_finite(out, msg="argmax scalar tensor softness")
    common.assert_finite(
        st.sigmoidal(x, softness=softness), msg="sigmoidal scalar tensor softness"
    )
    common.assert_finite(
        st.sort(x, softness=softness, method="sorting_network").values,
        msg="sorting_network scalar tensor softness",
    )
    common.assert_finite(
        st.sort(x, softness=softness, method="fast_soft_sort").values,
        msg="fast_soft_sort scalar tensor softness",
    )
    common.assert_finite(
        st.sort(x, softness=softness, method="ot", return_indices=False).values,
        msg="ot scalar tensor softness",
    )

    with pytest.raises(ValueError, match="softness must be positive"):
        st.argmax(x, softness=torch.tensor(0.0))
    with pytest.raises(ValueError, match="softness must be a scalar"):
        st.argmax(x, softness=torch.tensor([0.5, 1.0]))


def _weighted_last_dim_sum(x):
    weights = torch.arange(1, x.shape[-1] + 1, dtype=x.dtype, device=x.device)
    return (x * weights).sum()


def _assert_learnable_softness(loss_fn, init=0.5):
    softness = torch.nn.Parameter(torch.tensor(init, dtype=torch.float64))
    loss = loss_fn(softness)

    (grad,) = torch.autograd.grad(loss, softness)
    assert torch.isfinite(grad)
    assert grad.abs() > 1e-8


@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_elementwise_learnable_scalar_tensor_softness(mode):
    x = torch.tensor([-0.4, 0.1, 0.8, 1.3], dtype=torch.float64)

    _assert_learnable_softness(
        lambda softness: _weighted_last_dim_sum(
            st.sigmoidal(x, softness=softness, mode=mode)
        )
    )


@pytest.mark.parametrize("method", ("softsort", "neuralsort"))
@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_argsort_learnable_scalar_tensor_softness(method, mode):
    x = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64)

    def loss_fn(softness):
        soft_perm = st.argsort(
            x,
            dim=-1,
            softness=softness,
            mode=mode,
            method=method,
        )
        soft_sorted = torch.bmm(soft_perm, x.unsqueeze(-1)).squeeze(-1)
        return _weighted_last_dim_sum(soft_sorted)

    _assert_learnable_softness(loss_fn)


@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_sorting_network_learnable_scalar_tensor_softness(mode):
    x = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64)

    _assert_learnable_softness(
        lambda softness: _weighted_last_dim_sum(
            st.sort(x, dim=-1, softness=softness, mode=mode, method="sorting_network").values
        )
    )


@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_fast_soft_sort_learnable_scalar_tensor_softness(mode):
    x = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64)

    _assert_learnable_softness(
        lambda softness: _weighted_last_dim_sum(
            st.sort(
                x,
                dim=-1,
                softness=softness,
                mode=mode,
                method="fast_soft_sort",
                return_indices=False,
            ).values
        ),
        init=5.0,
    )


@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_ot_rejects_learnable_scalar_tensor_softness(mode):
    x = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64)
    softness = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

    with pytest.raises(
        ValueError, match="learnable tensor softness is not supported for OT"
    ):
        st.sort(
            x,
            dim=-1,
            softness=softness,
            mode=mode,
            method="ot",
            return_indices=False,
        )


def test_topk_k_validation():
    x = torch.tensor([3.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="k must be positive"):
        st.topk(x, k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        st.topk(x, k=-1)


def test_narrow_length_validation():
    x = torch.tensor([3.0, 1.0, 2.0])
    soft_start = torch.tensor([0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="length"):
        st.narrow(x, soft_start=soft_start, length=0, dim=0)
    with pytest.raises(ValueError, match="length"):
        st.narrow(x, soft_start=soft_start, length=4, dim=0)


def test_median_unsupported_method():
    x = torch.tensor([3.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="not supported"):
        st.median(x, dim=0, method="fast_soft_sort", mode="c0")
    with pytest.raises(ValueError, match="not supported"):
        st.median(x, dim=0, method="sorting_network", mode="c0")


def test_quantile_return_argquantile_unsupported():
    x = torch.tensor([3.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="not supported"):
        st.quantile(
            x, q=0.5, return_argquantile=True, method="fast_soft_sort", mode="c0"
        )


# ---------------------------------------------------------------------------
# Single-element (n=1) edge cases (issue 17)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
def test_single_element_value_ops(fn_name):
    """Value ops work correctly with a single-element input."""
    x = torch.tensor([42.0])
    fn = getattr(st, fn_name)
    out = fn(x, dim=0, mode="smooth", softness=1.0)
    if fn_name in ("max", "min", "sort", "median"):
        common.assert_finite(out.values, msg=f"{fn_name} n=1")
    else:
        common.assert_finite(out, msg=f"{fn_name} n=1")


@pytest.mark.parametrize("fn_name", ["argmax", "argmin", "argsort", "argmedian"])
def test_single_element_arg_ops(fn_name):
    """Arg ops return [1.0] for single-element input."""
    x = torch.tensor([42.0])
    fn = getattr(st, fn_name)
    out = fn(x, dim=0, mode="smooth", softness=1.0)
    common.assert_finite(out, msg=f"{fn_name} n=1")
    common.assert_simplex(out, atol=1e-4)


def test_single_element_topk():
    x = torch.tensor([42.0])
    out = st.topk(x, k=1, dim=0, mode="smooth", softness=1.0)
    common.assert_finite(out.values, msg="topk n=1")


def test_single_element_quantile():
    x = torch.tensor([42.0])
    out = st.quantile(x, q=0.5, dim=0, mode="smooth", softness=1.0)
    common.assert_finite(out, msg="quantile n=1")


# ---------------------------------------------------------------------------
# NaN input propagation (issue 18)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["sort", "max", "min", "rank"])
def test_nan_propagation(fn_name):
    """NaN inputs should propagate through soft operators."""
    x = torch.tensor([1.0, float("nan"), 3.0])
    fn = getattr(st, fn_name)
    out = fn(x, dim=0, mode="c0", softness=1.0)
    vals = out.values if isinstance(out, tuple) else out
    assert torch.any(torch.isnan(vals)), f"NaN should propagate through {fn_name}"


@pytest.mark.parametrize(
    "fn_name", ["sort", "argsort", "rank", "max", "min", "median", "topk"]
)
def test_smooth_sort_raises(fn_name):
    """smooth_sort method should raise NotImplementedError in SoftTorch."""
    x = torch.tensor([3.0, 1.0, 2.0])
    fn = getattr(st, fn_name)
    kwargs = {"dim": -1, "mode": "smooth", "softness": 0.1, "method": "smooth_sort"}
    if fn_name == "topk":
        kwargs["k"] = 2
    with pytest.raises((NotImplementedError, ValueError)):
        fn(x, **kwargs)


# ---------------------------------------------------------------------------
# gated_grad parameter (issue 19)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name", ["max", "min", "sort", "topk", "quantile", "median"]
)
def test_gated_grad(fn_name):
    """gated_grad=True and gated_grad=False both produce finite gradients."""
    x_true = common.gradient_input((5,), torch.float64)
    x_false = x_true.detach().clone().requires_grad_(True)
    fn = getattr(st, fn_name)

    has_values = fn_name in ("max", "min", "sort", "topk", "median")
    kwargs = {"dim": -1, "mode": "smooth", "softness": 1.0}
    if fn_name == "topk":
        kwargs["k"] = 2
    elif fn_name == "quantile":
        kwargs["q"] = 0.5

    out_true = fn(x_true, **kwargs, gated_grad=True)
    vals_true = out_true.values if has_values else out_true
    vals_true.sum().backward()
    common.assert_finite(x_true.grad, msg=f"{fn_name} gated_grad=True")

    out_false = fn(x_false, **kwargs, gated_grad=False)
    vals_false = out_false.values if has_values else out_false
    vals_false.sum().backward()
    common.assert_finite(x_false.grad, msg=f"{fn_name} gated_grad=False")


@pytest.mark.parametrize(
    "case_name, call",
    [
        (
            "max",
            lambda x, **kwargs: st.max(
                x, dim=-1, method="softsort", mode="smooth", **kwargs
            ).indices,
        ),
        (
            "min",
            lambda x, **kwargs: st.min(
                x, dim=-1, method="softsort", mode="smooth", **kwargs
            ).indices,
        ),
        (
            "sort",
            lambda x, **kwargs: st.sort(
                x, dim=-1, method="softsort", mode="smooth", **kwargs
            ).indices,
        ),
        (
            "quantile",
            lambda x, **kwargs: st.quantile(
                x,
                q=0.5,
                dim=-1,
                method="softsort",
                mode="smooth",
                return_argquantile=True,
                **kwargs,
            )[1],
        ),
        (
            "median",
            lambda x, **kwargs: st.median(
                x, dim=-1, method="softsort", mode="smooth", **kwargs
            ).indices,
        ),
        (
            "topk",
            lambda x, **kwargs: st.topk(
                x, k=2, dim=-1, method="softsort", mode="smooth", **kwargs
            ).indices,
        ),
    ],
)
@pytest.mark.parametrize(
    "return_log_probs, log_prob_eps",
    [(False, None), (True, None), (True, 1e-3)],
)
def test_gated_grad_false_preserves_returned_index_gradients(
    case_name, call, return_log_probs, log_prob_eps
):
    x = common.gradient_input((5,), torch.float64)
    kwargs = {"return_log_probs": return_log_probs}
    if log_prob_eps is not None:
        kwargs["log_prob_eps"] = log_prob_eps

    indices = call(x, softness=1.0, gated_grad=False, **kwargs)
    weights = torch.arange(
        indices.numel(), dtype=indices.dtype, device=indices.device
    ).reshape_as(indices)
    grad = torch.autograd.grad(torch.sum(indices * weights), x)[0]

    common.assert_finite(grad, msg=f"{case_name} returned indices gated_grad=False")
    assert torch.any(grad != 0)


# ---------------------------------------------------------------------------
# return_indices parameter for sort (issue 19)
# ---------------------------------------------------------------------------


def test_sort_return_indices():
    """sort(return_indices=False) skips soft permutation matrix."""
    x = torch.tensor([3.0, 1.0, 2.0])
    out_with = st.sort(x, dim=0, mode="smooth", softness=1.0, return_indices=True)
    assert out_with.indices is not None
    out_without = st.sort(x, dim=0, mode="smooth", softness=1.0, return_indices=False)
    assert out_without.indices is None
    common.assert_allclose(out_with.values, out_without.values, tol=1e-10)


def test_sort_return_indices_hard():
    """sort(return_indices=False, mode='hard') skips one-hot allocation."""
    x = torch.tensor([3.0, 1.0, 2.0])
    out = st.sort(x, dim=0, mode="hard", return_indices=False)
    assert out.indices is None
    expected = torch.sort(x, dim=0).values
    common.assert_allclose(out.values, expected, tol=1e-10)


# ---------------------------------------------------------------------------
# return_argquantile parameter for quantile (issue 19)
# ---------------------------------------------------------------------------


def test_quantile_return_argquantile():
    """quantile with return_argquantile=True returns both value and index."""
    x = torch.tensor([3.0, 1.0, 2.0, 4.0])
    val, idx = st.quantile(
        x, q=0.5, dim=0, return_argquantile=True, softness=1.0
    )
    common.assert_finite(val, msg="quantile value")
    common.assert_finite(idx, msg="quantile argquantile")
    common.assert_simplex(idx, atol=1e-4)


# ---------------------------------------------------------------------------
# return_log_probs parameter for SoftIndex outputs
# ---------------------------------------------------------------------------


def _assert_log_probs_match_probs(log_probs, probs, msg):
    assert not torch.any(torch.isnan(log_probs)), f"NaN in {msg}"
    common.assert_allclose(torch.exp(log_probs), probs, tol=1e-6)


def _manual_log_prob_eps(probs, eps):
    probs = torch.clamp(probs, min=eps, max=1.0)
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return torch.log(probs)


_LOG_PROB_INDEX_CASES = [
    (
        "argmax",
        lambda x, method, mode, **kwargs: st.argmax(
            x, dim=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "argmin",
        lambda x, method, mode, **kwargs: st.argmin(
            x, dim=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "argsort",
        lambda x, method, mode, **kwargs: st.argsort(
            x, dim=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "argquantile",
        lambda x, method, mode, **kwargs: st.argquantile(
            x, q=0.35, dim=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "argmedian",
        lambda x, method, mode, **kwargs: st.argmedian(
            x, dim=-1, method=method, mode=mode, **kwargs
        ),
    ),
]


@pytest.mark.parametrize(
    "case_name, call",
    _LOG_PROB_INDEX_CASES,
    ids=[case_name for case_name, _ in _LOG_PROB_INDEX_CASES],
)
@pytest.mark.parametrize("method", ("softsort", "neuralsort", "ot", "sorting_network"))
@pytest.mark.parametrize("mode", ("hard", "smooth", "c0", "c1", "c2"))
def test_return_log_probs_softindex_outputs(case_name, call, method, mode):
    x = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64)

    probs = call(x, method, mode)
    log_probs = call(x, method, mode, return_log_probs=True)

    _assert_log_probs_match_probs(log_probs, probs, f"{case_name}/{method}/{mode}")


_LOG_PROB_VALUE_CASES = [
    (
        "max",
        lambda x, method, mode, **kwargs: st.max(
            x, dim=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "min",
        lambda x, method, mode, **kwargs: st.min(
            x, dim=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "sort",
        lambda x, method, mode, **kwargs: st.sort(
            x, dim=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "quantile",
        lambda x, method, mode, **kwargs: st.quantile(
            x,
            q=0.35,
            dim=-1,
            method=method,
            mode=mode,
            return_argquantile=True,
            **kwargs,
        ),
    ),
    (
        "median",
        lambda x, method, mode, **kwargs: st.median(
            x, dim=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "topk",
        lambda x, method, mode, **kwargs: st.topk(
            x, k=2, dim=-1, method=method, mode=mode, **kwargs
        ),
    ),
]


def _values_and_indices(out):
    if isinstance(out, tuple) and not hasattr(out, "values"):
        return out
    return out.values, out.indices


@pytest.mark.parametrize(
    "case_name, call",
    _LOG_PROB_VALUE_CASES,
    ids=[case_name for case_name, _ in _LOG_PROB_VALUE_CASES],
)
@pytest.mark.parametrize("method", ("softsort", "neuralsort", "ot"))
@pytest.mark.parametrize("mode", ("hard", "smooth", "c0", "c1", "c2"))
def test_return_log_probs_value_outputs(case_name, call, method, mode):
    x = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64)

    values, probs = _values_and_indices(call(x, method, mode))
    log_values, log_probs = _values_and_indices(
        call(x, method, mode, return_log_probs=True)
    )

    common.assert_allclose(log_values, values, tol=1e-6)
    _assert_log_probs_match_probs(log_probs, probs, f"{case_name}/{method}/{mode}")


@pytest.mark.parametrize(
    "case_name, call",
    _LOG_PROB_INDEX_CASES,
    ids=[case_name for case_name, _ in _LOG_PROB_INDEX_CASES],
)
def test_return_log_probs_log_prob_eps_index_outputs_match_manual(case_name, call):
    x = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64)
    eps = 1e-3

    probs = call(x, "softsort", "c0", softness=0.01)
    log_probs = call(
        x,
        "softsort",
        "c0",
        softness=0.01,
        return_log_probs=True,
        log_prob_eps=eps,
    )
    expected = _manual_log_prob_eps(probs, eps)

    assert torch.all(torch.isfinite(log_probs)), f"nonfinite log_probs in {case_name}"
    common.assert_allclose(log_probs, expected, tol=1e-6)
    common.assert_allclose(
        torch.sum(torch.exp(log_probs), dim=-1),
        torch.ones_like(log_probs[..., 0]),
        tol=1e-6,
    )


@pytest.mark.parametrize(
    "case_name, call",
    _LOG_PROB_VALUE_CASES,
    ids=[case_name for case_name, _ in _LOG_PROB_VALUE_CASES],
)
def test_return_log_probs_log_prob_eps_value_outputs_match_manual(case_name, call):
    x = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64)
    eps = 1e-3

    values, probs = _values_and_indices(call(x, "softsort", "c0", softness=0.01))
    log_values, log_probs = _values_and_indices(
        call(
            x,
            "softsort",
            "c0",
            softness=0.01,
            return_log_probs=True,
            log_prob_eps=eps,
        )
    )
    expected = _manual_log_prob_eps(probs, eps)

    common.assert_allclose(log_values, values, tol=1e-6)
    assert torch.all(torch.isfinite(log_probs)), f"nonfinite log_probs in {case_name}"
    common.assert_allclose(log_probs, expected, tol=1e-6)


def test_return_log_probs_log_prob_eps_matches_manual_ot():
    x = torch.tensor([[0.2, 1.7, -0.5]], dtype=torch.float64)
    eps = 1e-3

    probs = st.argsort(x, dim=-1, method="ot", mode="c1", softness=1.0)
    log_probs = st.argsort(
        x,
        dim=-1,
        method="ot",
        mode="c1",
        softness=1.0,
        return_log_probs=True,
        log_prob_eps=eps,
    )
    expected = _manual_log_prob_eps(probs, eps)

    assert torch.all(torch.isfinite(log_probs)), "nonfinite OT log_prob_eps"
    common.assert_allclose(log_probs, expected, tol=1e-5)


def test_return_log_probs_sparse_mode_nan_to_num_does_not_fix_gradients():
    x = torch.tensor([0.2, 1.7, -0.5, 0.9], dtype=torch.float64).requires_grad_()
    weights = torch.arange(1, 5, dtype=x.dtype)

    log_probs = st.argmax(
        x,
        dim=0,
        mode="c0",
        softness=0.01,
        return_log_probs=True,
    )
    loss = (torch.nan_to_num(log_probs, neginf=-1e9) * weights).sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.all(torch.isfinite(x.grad))


@pytest.mark.parametrize("method", ("softsort", "neuralsort"))
@pytest.mark.parametrize("mode", ("c0", "c1", "c2"))
def test_return_log_probs_log_prob_eps_has_finite_sparse_gradients(method, mode):
    x = torch.tensor([0.2, 1.7, -0.5, 0.9], dtype=torch.float64).requires_grad_()
    weights = torch.arange(1, 5, dtype=x.dtype)

    log_probs = st.argmax(
        x,
        dim=0,
        method=method,
        mode=mode,
        softness=0.01,
        return_log_probs=True,
        log_prob_eps=1e-12,
    )
    loss = (log_probs * weights).sum()
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad)), f"nonfinite gradient in {method}/{mode}"


def test_return_log_probs_log_prob_eps_requires_return_log_probs():
    x = torch.tensor([0.2, 1.7, -0.5, 0.9], dtype=torch.float64)

    with pytest.raises(ValueError, match="return_log_probs=True"):
        st.argmax(x, dim=0, log_prob_eps=1e-3)

    with pytest.raises(ValueError, match="return_log_probs=True"):
        st.topk(x, k=2, dim=0, log_prob_eps=1e-3)


@pytest.mark.parametrize("eps", (0.0, -1.0, 1.5))
def test_return_log_probs_log_prob_eps_validates_range(eps):
    x = torch.tensor([0.2, 1.7, -0.5, 0.9], dtype=torch.float64)

    with pytest.raises(ValueError, match="log_prob_eps must be in"):
        st.argmax(x, dim=0, return_log_probs=True, log_prob_eps=eps)


@pytest.mark.parametrize("method", ("softsort", "neuralsort", "ot", "sorting_network"))
@pytest.mark.parametrize("mode", ("smooth", "c0"))
def test_return_log_probs_nonlast_dim_keepdim_and_endpoints(method, mode):
    x = torch.tensor(
        [[0.2, 1.7, -0.5, 0.9], [2.0, -1.0, 0.3, 0.7]], dtype=torch.float64
    )
    cases = [
        ("argmax", lambda **kwargs: st.argmax(x, dim=0, keepdim=True, **kwargs)),
        ("argmin", lambda **kwargs: st.argmin(x, dim=0, keepdim=False, **kwargs)),
        ("argsort", lambda **kwargs: st.argsort(x, dim=0, **kwargs)),
        (
            "argquantile_q0",
            lambda **kwargs: st.argquantile(x, q=0.0, dim=0, keepdim=True, **kwargs),
        ),
        (
            "argquantile_q1",
            lambda **kwargs: st.argquantile(x, q=1.0, dim=0, keepdim=False, **kwargs),
        ),
        (
            "argmedian",
            lambda **kwargs: st.argmedian(x, dim=0, keepdim=True, **kwargs),
        ),
    ]

    for case_name, call in cases:
        probs = call(method=method, mode=mode, softness=1.0, standardize=False)
        log_probs = call(
            method=method,
            mode=mode,
            softness=1.0,
            standardize=False,
            return_log_probs=True,
        )
        _assert_log_probs_match_probs(log_probs, probs, f"{case_name}/{method}/{mode}")


@pytest.mark.parametrize(
    "case_name, call",
    [
        (
            "argmax",
            lambda x, method: st.argmax(
                x, dim=-1, method=method, mode="smooth", return_log_probs=True
            ),
        ),
        (
            "argmin",
            lambda x, method: st.argmin(
                x, dim=-1, method=method, mode="smooth", return_log_probs=True
            ),
        ),
        (
            "argsort",
            lambda x, method: st.argsort(
                x, dim=-1, method=method, mode="smooth", return_log_probs=True
            ),
        ),
        (
            "argquantile",
            lambda x, method: st.argquantile(
                x, q=0.35, dim=-1, method=method, mode="smooth", return_log_probs=True
            ),
        ),
        (
            "argmedian",
            lambda x, method: st.argmedian(
                x, dim=-1, method=method, mode="smooth", return_log_probs=True
            ),
        ),
    ],
)
@pytest.mark.parametrize("method", ("softsort", "neuralsort", "ot", "sorting_network"))
def test_return_log_probs_smooth_outputs_have_finite_gradients(case_name, call, method):
    x = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64).requires_grad_()

    log_probs = call(x, method)
    weights = torch.arange(
        1, log_probs.shape[-1] + 1, dtype=log_probs.dtype, device=log_probs.device
    )
    loss = (log_probs * weights).sum()
    loss.backward()

    assert torch.all(torch.isfinite(log_probs)), f"nonfinite log_probs in {case_name}"
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad)), f"nonfinite gradient in {case_name}"


@pytest.mark.parametrize("fn_name", ("max", "sort", "topk"))
@pytest.mark.parametrize("method", ("fast_soft_sort", "sorting_network"))
@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_return_log_probs_preserves_none_indices(fn_name, method, mode):
    x = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64)
    fn = getattr(st, fn_name)
    kwargs = dict(dim=-1, method=method, mode=mode)
    if fn_name == "topk":
        kwargs["k"] = 2

    out = fn(x, **kwargs)
    log_out = fn(x, **kwargs, return_log_probs=True)

    assert out.indices is None
    assert log_out.indices is None
    common.assert_allclose(log_out.values, out.values, tol=1e-6)


def test_return_log_probs_hard_zeros_are_negative_infinity():
    x = torch.tensor([3.0, 1.0, 2.0])

    probs = st.argsort(x, dim=0, mode="hard")
    log_probs = st.argsort(x, dim=0, mode="hard", return_log_probs=True)

    _assert_log_probs_match_probs(log_probs, probs, "argsort hard")
    assert torch.all(log_probs[probs == 1] == 0)
    assert torch.all(torch.isneginf(log_probs[probs == 0]))


def test_return_log_probs_smooth_simplex_avoids_softmax_underflow():
    x = torch.tensor([1000.0, 0.0, -1000.0], dtype=torch.float32)

    probs = st.argmax(
        x,
        dim=0,
        mode="smooth",
        method="softsort",
        softness=1.0,
        standardize=False,
    )
    log_probs = st.argmax(
        x,
        dim=0,
        mode="smooth",
        method="softsort",
        softness=1.0,
        standardize=False,
        return_log_probs=True,
    )

    assert torch.isneginf(torch.log(probs)).any()
    assert torch.all(torch.isfinite(log_probs))
    _assert_log_probs_match_probs(log_probs, probs, "argmax smooth softsort")


@pytest.mark.parametrize(
    "case_name, call",
    [
        (
            "max",
            lambda x, **kwargs: st.max(
                x, dim=0, method="softsort", mode="smooth", **kwargs
            ).indices,
        ),
        (
            "min",
            lambda x, **kwargs: st.min(
                x, dim=0, method="softsort", mode="smooth", **kwargs
            ).indices,
        ),
        (
            "sort",
            lambda x, **kwargs: st.sort(
                x, dim=0, method="softsort", mode="smooth", **kwargs
            ).indices,
        ),
        (
            "quantile",
            lambda x, **kwargs: st.quantile(
                x,
                q=0.5,
                dim=0,
                method="softsort",
                mode="smooth",
                return_argquantile=True,
                **kwargs,
            )[1],
        ),
        (
            "median",
            lambda x, **kwargs: st.median(
                x, dim=0, method="softsort", mode="smooth", **kwargs
            ).indices,
        ),
        (
            "topk",
            lambda x, **kwargs: st.topk(
                x, k=2, dim=0, method="softsort", mode="smooth", **kwargs
            ).indices,
        ),
    ],
)
def test_return_log_probs_value_outputs_avoid_softmax_underflow(case_name, call):
    x = torch.tensor([1000.0, 0.0, -1000.0], dtype=torch.float32)

    probs = call(x, softness=1.0, standardize=False)
    log_probs = call(x, softness=1.0, standardize=False, return_log_probs=True)

    assert torch.isneginf(torch.log(probs)).any()
    assert torch.all(torch.isfinite(log_probs))
    _assert_log_probs_match_probs(log_probs, probs, case_name)


@pytest.mark.parametrize(
    "case_name, call",
    [
        (
            "max",
            lambda x, **kwargs: st.max(
                x, dim=-1, method="softsort", mode="smooth", **kwargs
            ),
        ),
        (
            "min",
            lambda x, **kwargs: st.min(
                x, dim=-1, method="softsort", mode="smooth", **kwargs
            ),
        ),
        (
            "sort",
            lambda x, **kwargs: st.sort(
                x, dim=-1, method="softsort", mode="smooth", **kwargs
            ),
        ),
        (
            "quantile",
            lambda x, **kwargs: st.quantile(
                x,
                q=0.35,
                dim=-1,
                method="softsort",
                mode="smooth",
                return_argquantile=True,
                **kwargs,
            ),
        ),
        (
            "median",
            lambda x, **kwargs: st.median(
                x, dim=-1, method="softsort", mode="smooth", **kwargs
            ),
        ),
        (
            "topk",
            lambda x, **kwargs: st.topk(
                x, k=2, dim=-1, method="softsort", mode="smooth", **kwargs
            ),
        ),
    ],
)
def test_return_log_probs_preserves_value_gradients(case_name, call):
    x0 = torch.tensor([[0.2, 1.7, -0.5, 0.9]], dtype=torch.float64)

    def value_and_grad(return_log_probs, log_prob_eps=None):
        x = x0.clone().requires_grad_(True)
        out = call(
            x,
            softness=1.0,
            standardize=False,
            return_log_probs=return_log_probs,
            log_prob_eps=log_prob_eps,
        )
        values, _ = _values_and_indices(out)
        values.sum().backward()
        return values.detach(), x.grad

    values, grad = value_and_grad(False)
    log_values, log_grad = value_and_grad(True)
    safe_log_values, safe_log_grad = value_and_grad(True, log_prob_eps=1e-3)

    common.assert_allclose(log_values, values, tol=1e-6)
    common.assert_allclose(log_grad, grad, tol=1e-6)
    common.assert_allclose(safe_log_values, values, tol=1e-6)
    common.assert_allclose(safe_log_grad, grad, tol=1e-6)


@pytest.mark.parametrize(
    "case_name, st_call, hard_call",
    [
        (
            "argmax_st",
            lambda x, mode: st.argmax_st(
                x, dim=0, mode=mode, return_log_probs=True
            ),
            lambda x: st.argmax(x, dim=0, mode="hard"),
        ),
        (
            "argsort_st",
            lambda x, mode: st.argsort_st(
                x, dim=0, mode=mode, return_log_probs=True
            ),
            lambda x: st.argsort(x, dim=0, mode="hard"),
        ),
        (
            "argquantile_st",
            lambda x, mode: st.argquantile_st(
                x, q=0.35, dim=0, mode=mode, return_log_probs=True
            ),
            lambda x: st.argquantile(x, q=0.35, dim=0, mode="hard"),
        ),
    ],
)
@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_return_log_probs_st_outputs_handle_negative_infinity(
    case_name, st_call, hard_call, mode
):
    x = torch.tensor([0.2, 1.7, -0.5, 0.9], dtype=torch.float64)

    log_probs = st_call(x, mode)

    assert not torch.any(torch.isnan(log_probs)), f"NaN in {case_name}/{mode}"
    _assert_log_probs_match_probs(log_probs, hard_call(x), f"{case_name}/{mode}")


@pytest.mark.parametrize(
    "case_name, st_call, hard_call",
    [
        (
            "max_st",
            lambda x, mode: st.max_st(
                x, dim=0, mode=mode, return_log_probs=True
            ).indices,
            lambda x: st.max(x, dim=0, mode="hard").indices,
        ),
        (
            "min_st",
            lambda x, mode: st.min_st(
                x, dim=0, mode=mode, return_log_probs=True
            ).indices,
            lambda x: st.min(x, dim=0, mode="hard").indices,
        ),
        (
            "sort_st",
            lambda x, mode: st.sort_st(
                x, dim=0, mode=mode, return_log_probs=True
            ).indices,
            lambda x: st.sort(x, dim=0, mode="hard").indices,
        ),
        (
            "quantile_st",
            lambda x, mode: st.quantile_st(
                x,
                q=0.35,
                dim=0,
                mode=mode,
                return_argquantile=True,
                return_log_probs=True,
            )[1],
            lambda x: st.quantile(
                x, q=0.35, dim=0, mode="hard", return_argquantile=True
            )[1],
        ),
        (
            "median_st",
            lambda x, mode: st.median_st(
                x, dim=0, mode=mode, return_log_probs=True
            ).indices,
            lambda x: st.median(x, dim=0, mode="hard").indices,
        ),
        (
            "topk_st",
            lambda x, mode: st.topk_st(
                x, k=2, dim=0, mode=mode, return_log_probs=True
            ).indices,
            lambda x: st.topk(x, k=2, dim=0, mode="hard").indices,
        ),
    ],
)
@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_return_log_probs_st_namedtuple_outputs_handle_negative_infinity(
    case_name, st_call, hard_call, mode
):
    x = torch.tensor([0.2, 1.7, -0.5, 0.9], dtype=torch.float64)

    log_probs = st_call(x, mode)

    assert not torch.any(torch.isnan(log_probs)), f"NaN in {case_name}/{mode}"
    _assert_log_probs_match_probs(log_probs, hard_call(x), f"{case_name}/{mode}")


@pytest.mark.parametrize("keepdim", (False, True))
def test_return_log_probs_dim_none_shapes(keepdim):
    x = torch.tensor([[0.2, 1.7], [-0.5, 0.9]], dtype=torch.float64)

    argmax_probs = st.argmax(x, dim=None, keepdim=keepdim)
    argmax_log_probs = st.argmax(
        x, dim=None, keepdim=keepdim, return_log_probs=True
    )
    _assert_log_probs_match_probs(argmax_log_probs, argmax_probs, "argmax dim=None")

    argquantile_probs = st.argquantile(x, q=0.35, dim=None, keepdim=keepdim)
    argquantile_log_probs = st.argquantile(
        x, q=0.35, dim=None, keepdim=keepdim, return_log_probs=True
    )
    _assert_log_probs_match_probs(
        argquantile_log_probs, argquantile_probs, "argquantile dim=None"
    )
    assert argmax_log_probs.shape == argmax_probs.shape
    assert argquantile_log_probs.shape == argquantile_probs.shape


def test_quantile_vector_q_return_log_probs():
    x = common.make_tensor((6,), torch.float64)
    q_vec = _VECTOR_Q

    _, probs = st.quantile(
        x, q_vec, dim=0, mode="smooth", softness=1.0, return_argquantile=True
    )
    _, log_probs = st.quantile(
        x,
        q_vec,
        dim=0,
        mode="smooth",
        softness=1.0,
        return_argquantile=True,
        return_log_probs=True,
    )

    _assert_log_probs_match_probs(log_probs, probs, "quantile vector q")


def test_quantile_vector_q_return_log_probs_log_prob_eps_matches_manual():
    x = common.make_tensor((6,), torch.float64)
    q_vec = _VECTOR_Q
    eps = 1e-3

    _, probs = st.quantile(
        x, q_vec, dim=0, mode="c0", softness=0.01, return_argquantile=True
    )
    _, log_probs = st.quantile(
        x,
        q_vec,
        dim=0,
        mode="c0",
        softness=0.01,
        return_argquantile=True,
        return_log_probs=True,
        log_prob_eps=eps,
    )
    expected = _manual_log_prob_eps(probs, eps)

    assert torch.all(torch.isfinite(log_probs)), "nonfinite quantile vector q log_probs"
    common.assert_allclose(log_probs, expected, tol=1e-6)


# ---------------------------------------------------------------------------
# Vector q support for quantile / argquantile
# ---------------------------------------------------------------------------

_VECTOR_Q = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
_ARGQUANTILE_METHODS = ["neuralsort", "softsort", "ot", "sorting_network"]
_QUANTILE_METHODS = _ARGQUANTILE_METHODS + ["fast_soft_sort"]


@pytest.mark.parametrize("shape", [(5,), (2, 4)])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
@pytest.mark.parametrize("method", _ARGQUANTILE_METHODS)
def test_argquantile_vector_q_shape(shape, dim, keepdim, method):
    """argquantile with vector q: output shape matches torch.quantile shape + [n]."""
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    out = st.argquantile(
        x, _VECTOR_Q, dim=dim, keepdim=keepdim, mode="hard", method=method,
    )
    torch_out = torch.quantile(x, _VECTOR_Q, dim=dim, keepdim=keepdim)
    assert out.shape[:-1] == torch_out.shape, (
        f"shape mismatch: {out.shape[:-1]} vs {torch_out.shape}"
    )
    assert out.shape[-1] == x.shape[dim]


@pytest.mark.parametrize("shape", [(5,), (2, 4)])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
@pytest.mark.parametrize("method", _QUANTILE_METHODS)
def test_quantile_vector_q_shape(shape, dim, keepdim, method):
    """quantile with vector q: output shape matches torch.quantile."""
    if not _valid_dim(shape, dim):
        pytest.skip(f"dim {dim} invalid for shape {shape}")
    x = common.make_tensor(shape, torch.float64)
    out = st.quantile(
        x, _VECTOR_Q, dim=dim, keepdim=keepdim, mode="hard", method=method,
    )
    torch_out = torch.quantile(x, _VECTOR_Q, dim=dim, keepdim=keepdim)
    assert out.shape == torch_out.shape, (
        f"shape mismatch: {out.shape} vs {torch_out.shape}"
    )


@pytest.mark.parametrize("method", _ARGQUANTILE_METHODS)
def test_argquantile_vector_q_consistency(method):
    """Vector-q result matches stacked scalar-q results."""
    x = common.make_tensor((6,), torch.float64)
    q_vec = _VECTOR_Q
    vec_out = st.argquantile(x, q_vec, dim=0, mode="smooth", method=method, softness=1.0)
    for i, qi in enumerate(q_vec):
        scalar_out = st.argquantile(
            x, qi.item(), dim=0, mode="smooth", method=method, softness=1.0,
        )
        np.testing.assert_allclose(
            vec_out[i].detach().numpy(), scalar_out.detach().numpy(), atol=1e-6,
            err_msg=f"argquantile vector vs scalar mismatch at q={qi}, method={method}",
        )


@pytest.mark.parametrize("method", _QUANTILE_METHODS)
def test_quantile_vector_q_consistency(method):
    """Vector-q result matches stacked scalar-q results."""
    x = common.make_tensor((6,), torch.float64)
    q_vec = _VECTOR_Q
    vec_out = st.quantile(x, q_vec, dim=0, mode="smooth", method=method, softness=1.0)
    for i, qi in enumerate(q_vec):
        scalar_out = st.quantile(
            x, qi.item(), dim=0, mode="smooth", method=method, softness=1.0,
        )
        np.testing.assert_allclose(
            vec_out[i].detach().numpy(), scalar_out.detach().numpy(), atol=1e-6,
            err_msg=f"quantile vector vs scalar mismatch at q={qi}, method={method}",
        )


@pytest.mark.parametrize("method", _ARGQUANTILE_METHODS)
def test_argquantile_vector_q_simplex(method):
    """argquantile with vector q: each row is a valid SoftIndex."""
    x = common.make_tensor((6,), torch.float64)
    out = st.argquantile(
        x, _VECTOR_Q, dim=0, mode="smooth", method=method, softness=1.0,
    )
    for i in range(len(_VECTOR_Q)):
        common.assert_simplex(out[i], atol=common.TOLERANCE)


def test_quantile_vector_q_monotonicity():
    """quantile values should be non-decreasing in q."""
    x = common.make_tensor((8,), torch.float64)
    q_fine = torch.linspace(0.0, 1.0, 5)
    out = st.quantile(x, q_fine, dim=0, mode="smooth", softness=1.0)
    diffs = torch.diff(out)
    assert torch.all(diffs >= -1e-6), "quantile values should be non-decreasing in q"


@pytest.mark.parametrize("shape", [(5,), (2, 4)])
@pytest.mark.parametrize("keepdim", KEEPDIMS)
def test_quantile_vector_q_hard_parity(shape, keepdim):
    """mode='hard' with vector q matches torch.quantile."""
    x = common.make_tensor(shape, torch.float64)
    q_vec = _VECTOR_Q
    soft_out = st.quantile(x, q_vec, dim=-1, keepdim=keepdim, mode="hard")
    torch_out = torch.quantile(x, q_vec, dim=-1, keepdim=keepdim)
    common.assert_torch_parity(soft_out, torch_out, msg="quantile vector q hard parity")


def test_quantile_vector_q_dim_none():
    """Vector q with dim=None flattens input before computing quantiles."""
    x = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
    q_vec = torch.tensor([0.25, 0.75])
    out = st.quantile(x, q_vec, dim=None, mode="smooth", softness=1.0)
    assert out.shape == (2,), f"expected (2,) for dim=None, got {out.shape}"
    out_hard = st.quantile(x, q_vec, dim=None, mode="hard")
    torch_out = torch.quantile(x, q_vec, dim=None)
    common.assert_torch_parity(out_hard, torch_out, msg="quantile vector q dim=None")


def test_argquantile_vector_q_dim_none():
    """argquantile with vector q and dim=None flattens input."""
    x = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
    q_vec = torch.tensor([0.25, 0.75])
    out = st.argquantile(x, q_vec, dim=None, mode="hard")
    torch_out = torch.quantile(x, q_vec, dim=None)
    assert out.shape[:-1] == torch_out.shape
    assert out.shape[-1] == x.numel()


def test_argquantile_vector_q_2d_rejection():
    """2D q should raise ValueError."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="q must be scalar or 1-D"):
        st.argquantile(x, torch.tensor([[0.25, 0.75]]), dim=0)


def test_quantile_vector_q_2d_rejection():
    """2D q should raise ValueError."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="q must be scalar or 1-D"):
        st.quantile(x, torch.tensor([[0.25, 0.75]]), dim=0)


def test_quantile_vector_q_gradient():
    """Gradients through vector-q quantile must be finite."""
    x = common.gradient_input((6,), torch.float32)
    q_vec = torch.tensor([0.25, 0.5, 0.75])
    out = st.quantile(x, q_vec, dim=-1, mode="smooth", softness=1.0)
    out.sum().backward()
    common.assert_finite(x.grad, msg="quantile vector q gradient")


def test_argquantile_vector_q_gradient():
    """Gradients through vector-q argquantile (via weighted sum) must be finite."""
    x = common.gradient_input((6,), torch.float32)
    weights = torch.arange(6.0)
    q_vec = torch.tensor([0.25, 0.75])
    out = st.argquantile(x, q_vec, dim=-1, mode="smooth", softness=1.0)
    loss = (out * weights).sum()
    loss.backward()
    common.assert_finite(x.grad, msg="argquantile vector q gradient")


def test_quantile_vector_q_return_argquantile():
    """quantile with return_argquantile=True works with vector q."""
    x = common.make_tensor((6,), torch.float64)
    q_vec = _VECTOR_Q
    val, idx = st.quantile(
        x, q_vec, dim=0, mode="smooth", softness=1.0, return_argquantile=True,
    )
    assert val.shape == (3,), f"expected val shape (3,), got {val.shape}"
    assert idx.shape == (3, 6), f"expected idx shape (3, 6), got {idx.shape}"
    for i in range(len(q_vec)):
        common.assert_simplex(idx[i], atol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# Device placement tests (CPU + GPU)
# ---------------------------------------------------------------------------


def _available_devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return tuple(devs)


DEVICES = _available_devices()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
def test_device_value_ops(device, fn_name):
    """Value arraywise ops produce correct results on each device."""
    x = common.make_tensor((4,), torch.float64).to(device)
    fn = getattr(st, fn_name)
    out_soft = fn(x, dim=-1, mode="smooth", softness=1.0)
    out_hard = fn(x, dim=-1, mode="hard")
    if fn_name in ("max", "min", "sort", "median"):
        common.assert_finite(out_soft.values, msg=f"{fn_name} on {device}")
        common.assert_finite(out_hard.values, msg=f"{fn_name} hard on {device}")
    else:
        common.assert_finite(out_soft, msg=f"{fn_name} on {device}")
        common.assert_finite(out_hard, msg=f"{fn_name} hard on {device}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn_name", ["argmax", "argmin", "argsort", "argmedian"])
def test_device_arg_ops(device, fn_name):
    """Arg arraywise ops produce correct results on each device."""
    x = common.make_tensor((4,), torch.float64).to(device)
    fn = getattr(st, fn_name)
    out = fn(x, dim=-1, mode="smooth", softness=1.0)
    common.assert_finite(out, msg=f"{fn_name} on {device}")
    common.assert_simplex(out, atol=common.TOLERANCE)


@pytest.mark.parametrize("device", DEVICES)
def test_device_gradient(device):
    """Gradients work on each device."""
    x = common.make_tensor((4,), torch.float32).to(device).requires_grad_(True)
    out = st.sort(x, dim=-1, mode="smooth", softness=1.0)
    out.values.sum().backward()
    common.assert_finite(x.grad, msg=f"sort gradient on {device}")
