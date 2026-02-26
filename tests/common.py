import numpy as np
import torch

import softtorch as st


# Suite-wide constants
NEAR_HARD_SOFTNESS = 1e-3
STABILITY_SOFTNESS = 100.0
TOLERANCE = 1e-2
SHAPES = {
    "vector": (4,),
    "matrix": (2, 3),
    "tensor": (4, 2, 3),
}

# Mode constants
MODES_ELEMENTWISE = ("smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm")
MODES_ARRAYWISE = ("smooth", "c0", "c1", "c2", "hard", "_hard")


# ---------------------------------------------------------------------------
# Tensor factories
# ---------------------------------------------------------------------------


def _base_values(shape, offset=0.0):
    size = int(np.prod(shape))
    data = np.linspace(-1.0, 1.0, num=size, dtype=np.float64)
    return data.reshape(shape) + offset


def make_tensor(shape, dtype=torch.float32, *, offset=0.0, softbool=False):
    values = _base_values(shape, offset=offset)
    if softbool:
        values = np.asarray(st.sigmoidal(torch.tensor(values)).numpy())
    t = torch.tensor(values, dtype=dtype)
    return t


def pair_tensors(shape, dtype=torch.float32, *, delta=0.2):
    x = make_tensor(shape, dtype, offset=0.0)
    y = make_tensor(shape, dtype, offset=delta)
    return x, y


def gradient_input(shape, dtype=torch.float32):
    base = _base_values(shape)
    return torch.tensor(base, dtype=dtype, requires_grad=True)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_allclose(actual, expected, tol=TOLERANCE, err_msg=""):
    np.testing.assert_allclose(
        _to_numpy(actual), _to_numpy(expected), rtol=tol, atol=tol, err_msg=err_msg
    )


def assert_softbool(values, atol=1e-6):
    arr = _to_numpy(values)
    assert np.all(arr >= -atol), f"SoftBool entries must be >= 0 (min={arr.min():.2e})"
    assert np.all(arr <= 1.0 + atol), f"SoftBool entries must be <= 1 (max={arr.max():.2e})"


def assert_simplex(values, dim=-1, atol=1e-6):
    arr = _to_numpy(values)
    assert arr.ndim >= 1, "SoftIndex outputs must have at least one dimension"
    assert np.all(arr >= -atol), "SoftIndex entries must be non-negative"
    sums = np.sum(arr, axis=dim)
    np.testing.assert_allclose(sums, 1.0, atol=atol)


def assert_finite(values, msg=""):
    arr = _to_numpy(values)
    assert np.all(np.isfinite(arr)), f"Non-finite values found. {msg}"


def assert_torch_parity(soft_output, torch_output, tol=TOLERANCE, msg=""):
    assert_allclose(
        soft_output, torch_output, tol=tol, err_msg=f"PyTorch parity mismatch. {msg}"
    )


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------


def assert_grad_matches_finite_diff(
    loss_fn, x, eps=1e-5, rtol=1e-3, atol=1e-4, msg=""
):
    x64 = x.detach().to(torch.float64).requires_grad_(True)
    loss = loss_fn(x64)
    (analytic_grad,) = torch.autograd.grad(loss, x64)
    analytic_np = analytic_grad.detach().numpy()
    assert np.all(np.isfinite(analytic_np)), f"Non-finite analytic gradient. {msg}"

    # Central finite differences
    x_np = x64.detach().numpy().copy()
    fd_grad = np.zeros_like(x_np)
    for idx in np.ndindex(x_np.shape):
        x_plus = x_np.copy()
        x_minus = x_np.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        fp = float(loss_fn(torch.tensor(x_plus, dtype=torch.float64)))
        fm = float(loss_fn(torch.tensor(x_minus, dtype=torch.float64)))
        fd_grad[idx] = (fp - fm) / (2 * eps)

    np.testing.assert_allclose(
        analytic_np,
        fd_grad,
        rtol=rtol,
        atol=atol,
        err_msg=f"Gradient vs finite diff mismatch. {msg}",
    )


# ---------------------------------------------------------------------------
# OT retry
# ---------------------------------------------------------------------------

OT_SOLVER_TOLS = [
    {"lbfgs_tol": 1e-5, "lbfgs_max_iter": 10000, "sinkhorn_tol": 1e-3, "sinkhorn_max_iter": 10000},
    {"lbfgs_tol": 1e-7, "lbfgs_max_iter": 20000, "sinkhorn_tol": 1e-5, "sinkhorn_max_iter": 20000},
    {"lbfgs_tol": 1e-9, "lbfgs_max_iter": 50000, "sinkhorn_tol": 1e-7, "sinkhorn_max_iter": 50000},
]


def ot_kwargs_for_method(method, softness):
    if method == "ot" and softness <= NEAR_HARD_SOFTNESS:
        return {"lbfgs_tol": 1e-7, "lbfgs_max_iter": 10000, "sinkhorn_tol": 1e-5, "sinkhorn_max_iter": 10000}
    return {}


def _any_nan(out):
    if isinstance(out, tuple):
        return any(_any_nan(o) for o in out)
    if out is None:
        return False
    return bool(torch.any(torch.isnan(out)).item()) if isinstance(out, torch.Tensor) else bool(np.any(np.isnan(np.asarray(out))))


def call_with_ot_retry(fn, *args, method="", tol=TOLERANCE, **kwargs):
    expected = kwargs.pop("_expected", None)
    check_fn = kwargs.pop("_check_fn", None)

    # Re-inject method into kwargs so fn receives it
    kwargs["method"] = method

    if method != "ot":
        return fn(*args, **kwargs)

    out = None
    for solver_kwargs in OT_SOLVER_TOLS:
        kwargs["ot_kwargs"] = solver_kwargs
        out = fn(*args, **kwargs)
        if _any_nan(out):
            continue
        if expected is not None:
            try:
                assert_allclose(out, expected, tol=tol)
                return out
            except AssertionError:
                continue
        if check_fn is not None:
            try:
                check_fn(out)
                return out
            except AssertionError:
                continue
        return out

    return out


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
