from __future__ import annotations

import numpy as np
import pytest
import torch

import softtorch as st

from . import common


SHAPES = tuple(common.SHAPES.items())
FLOAT_DTYPES = (torch.float32, torch.float64)


def _ids_from_shape(item):
    return item[0]


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

ELEMENTWISE_CASES = (
    {
        "name": "relu",
        "soft_fn": st.relu,
        "hard_fn": lambda x, **_: torch.relu(x),
        "modes": {"hard", "smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"},
        "kwargs": {},
    },
    {
        "name": "abs",
        "soft_fn": st.abs,
        "hard_fn": lambda x, **_: torch.abs(x),
        "modes": {"hard", "smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"},
        "kwargs": {},
    },
    {
        "name": "sign",
        "soft_fn": st.sign,
        "hard_fn": lambda x, **_: torch.sign(x).to(torch.float),
        "modes": {"hard", "smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"},
        "kwargs": {},
    },
    {
        "name": "round",
        "soft_fn": st.round,
        "hard_fn": lambda x, **_: torch.round(x),
        "modes": {"hard", "smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"},
        "kwargs": {},
    },
    {
        "name": "clamp",
        "soft_fn": st.clamp,
        "hard_fn": lambda x, *, a, b: torch.clamp(x, a, b),
        "modes": {"hard", "smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"},
        "kwargs": {"a": -0.25, "b": 0.25},
    },
)


COMPARISON_CASES = (
    {
        "name": "greater",
        "soft_fn": st.greater,
        "hard_fn": lambda x, y: torch.gt(x, y),
    },
    {
        "name": "greater_equal",
        "soft_fn": st.greater_equal,
        "hard_fn": lambda x, y: torch.ge(x, y),
    },
    {
        "name": "less",
        "soft_fn": st.less,
        "hard_fn": lambda x, y: torch.lt(x, y),
    },
    {
        "name": "less_equal",
        "soft_fn": st.less_equal,
        "hard_fn": lambda x, y: torch.le(x, y),
    },
    {
        "name": "eq",
        "soft_fn": st.eq,
        "hard_fn": lambda x, y: torch.eq(x, y),
    },
    {
        "name": "not_equal",
        "soft_fn": st.not_equal,
        "hard_fn": lambda x, y: torch.ne(x, y),
    },
    {
        "name": "isclose",
        "soft_fn": st.isclose,
        "hard_fn": lambda x, y: torch.isclose(x, y),
    },
)


# ---------------------------------------------------------------------------
# Elementwise: mode / softness / shape sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ELEMENTWISE_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
@pytest.mark.parametrize(
    "softness", (common.NEAR_HARD_SOFTNESS, common.STABILITY_SOFTNESS), ids=str
)
def test_elementwise_float_modes(case, shape_name, shape, dtype, softness):
    x = common.make_tensor(shape, dtype)
    kwargs = dict(case["kwargs"])
    out_hard = case["hard_fn"](x, **kwargs)

    for mode in case["modes"]:
        if mode == "hard":
            out = case["soft_fn"](x, mode="hard", **kwargs)
        else:
            out = case["soft_fn"](x, mode=mode, softness=softness, **kwargs)
            assert out.dtype in {torch.float32, torch.float64}

        assert out.shape == out_hard.shape
        assert not torch.any(torch.isnan(out))

        if softness == common.NEAR_HARD_SOFTNESS:
            common.assert_allclose(out, out_hard)


@pytest.mark.parametrize("case", ELEMENTWISE_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
def test_elementwise_torch_parity(case, shape_name, shape, dtype):
    x = common.make_tensor(shape, dtype)
    kwargs = dict(case["kwargs"])
    soft_out = case["soft_fn"](x, mode="hard", **kwargs)
    torch_out = case["hard_fn"](x, **kwargs)
    common.assert_torch_parity(soft_out, torch_out, msg=case["name"])


@pytest.mark.parametrize("case", ELEMENTWISE_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
def test_elementwise_int_hard_parity(case, shape_name, shape):
    values = common._base_values(shape)
    x = torch.tensor(values, dtype=torch.int32)
    kwargs = dict(case["kwargs"])
    out_hard = case["hard_fn"](x, **kwargs)
    out = case["soft_fn"](x, mode="hard", **kwargs)
    common.assert_allclose(out, out_hard.float())
    assert out.shape == out_hard.shape
    assert out.dtype in {torch.float32, torch.float64}


# ---------------------------------------------------------------------------
# Gradient finiteness for elementwise ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ELEMENTWISE_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_elementwise_gradient_finite(case, mode):
    x = common.gradient_input((5,), torch.float32)
    kwargs = dict(case["kwargs"])

    out = case["soft_fn"](x, mode=mode, softness=1.0, **kwargs)
    loss = out.sum()
    loss.backward()
    common.assert_finite(x.grad, msg=f"{case['name']} mode={mode}")


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_heaviside_gradient_finite(mode):
    x = common.gradient_input((5,), torch.float32)
    out = st.heaviside(x, mode=mode, softness=1.0)
    out.sum().backward()
    common.assert_finite(x.grad, msg=f"heaviside mode={mode}")


# ---------------------------------------------------------------------------
# Heaviside boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_heaviside_sign_boundaries(dtype):
    vec = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype)
    hv = st.heaviside(vec, mode="hard")
    common.assert_allclose(hv, torch.tensor([0.0, 0.5, 1.0], dtype=dtype))

    sg = st.sign(vec, mode="hard")
    common.assert_allclose(sg, torch.tensor([-1.0, 0.0, 1.0], dtype=dtype))

    soft_mid = st.sign(vec, mode="smooth", softness=common.NEAR_HARD_SOFTNESS)
    assert -0.01 < float(soft_mid[1]) < 0.01
    assert not torch.any(torch.isnan(soft_mid))


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_heaviside_torch_parity(dtype):
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=dtype)
    soft_out = st.heaviside(x, mode="hard")
    torch_out = torch.heaviside(x, torch.tensor(0.5, dtype=dtype))
    common.assert_torch_parity(soft_out, torch_out, msg="heaviside")


# ---------------------------------------------------------------------------
# Comparison / SoftBool tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", COMPARISON_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
@pytest.mark.parametrize(
    "softness", (common.NEAR_HARD_SOFTNESS, common.STABILITY_SOFTNESS), ids=str
)
def test_softbool(case, shape_name, shape, dtype, softness):
    x, y = common.pair_tensors(shape, dtype)
    out_hard = case["hard_fn"](x, y)

    for mode in common.MODES_ELEMENTWISE:
        out = case["soft_fn"](x, y, mode=mode, softness=softness)
        assert out.dtype in {torch.float32, torch.float64}
        assert out.shape == out_hard.shape
        assert not torch.any(torch.isnan(out))
        common.assert_softbool(out)

        if softness == common.NEAR_HARD_SOFTNESS:
            common.assert_allclose(
                out, out_hard, err_msg=f"{case['name']} near-hard mismatch"
            )

    out_hard_mode = case["soft_fn"](x, y, mode="hard")
    assert out_hard_mode.shape == out_hard.shape
    common.assert_allclose(
        out_hard_mode, out_hard, err_msg=f"{case['name']} hard mismatch"
    )


@pytest.mark.parametrize("case", COMPARISON_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
def test_comparison_torch_parity(case, shape_name, shape, dtype):
    x, y = common.pair_tensors(shape, dtype)
    soft_out = case["soft_fn"](x, y, mode="hard")
    torch_out = case["hard_fn"](x, y).to(torch.float)
    common.assert_torch_parity(soft_out, torch_out, msg=case["name"])


@pytest.mark.parametrize("case", COMPARISON_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_comparison_gradient_finite(case, mode):
    x = common.gradient_input((5,), torch.float32)
    y = common.make_tensor((5,), torch.float32, offset=0.1)

    out = case["soft_fn"](x, y, mode=mode, softness=1.0)
    loss = out.sum()
    loss.backward()
    common.assert_finite(x.grad, msg=f"{case['name']} mode={mode}")


@pytest.mark.parametrize("case", COMPARISON_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_comparison_gradient_finite_wrt_y(case, mode):
    """Gradients through soft comparisons wrt y must be finite."""
    x = common.make_tensor((5,), torch.float32)
    y = common.gradient_input((5,), torch.float32)

    out = case["soft_fn"](x, y, mode=mode, softness=1.0)
    loss = out.sum()
    loss.backward()
    common.assert_finite(y.grad, msg=f"{case['name']} wrt y mode={mode}")


# ---------------------------------------------------------------------------
# Where
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
def test_where_matches_mask(dtype, shape_name, shape):
    x, y = common.pair_tensors(shape, dtype)
    condition = st.greater(x, y, softness=common.NEAR_HARD_SOFTNESS)
    out = st.where(condition, x, y)
    mask = condition > 0.5
    expected = torch.where(mask, x, y)
    common.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Logical ops
# ---------------------------------------------------------------------------


def test_logical_ops_range_and_shapes():
    x = torch.tensor([0.1, 0.5, 0.9])
    y = torch.tensor([0.9, 0.2, 0.4])
    ops = [
        st.logical_not(x),
        st.logical_and(x, y),
        st.logical_or(x, y),
        st.logical_xor(x, y),
    ]
    for out in ops:
        assert out.shape == x.shape
        common.assert_softbool(out)


def test_logical_ops_truth_table():
    zero = torch.tensor([0.0])
    one = torch.tensor([1.0])

    common.assert_allclose(st.logical_not(zero), one, tol=1e-5)
    common.assert_allclose(st.logical_not(one), zero, tol=1e-5)

    common.assert_allclose(st.logical_and(zero, zero), zero, tol=1e-5)
    common.assert_allclose(st.logical_and(zero, one), zero, tol=1e-5)
    common.assert_allclose(st.logical_and(one, zero), zero, tol=1e-5)
    common.assert_allclose(st.logical_and(one, one), one, tol=1e-5)

    common.assert_allclose(st.logical_or(zero, zero), zero, tol=1e-5)
    common.assert_allclose(st.logical_or(zero, one), one, tol=1e-5)
    common.assert_allclose(st.logical_or(one, zero), one, tol=1e-5)
    common.assert_allclose(st.logical_or(one, one), one, tol=1e-5)

    common.assert_allclose(st.logical_xor(zero, zero), zero, tol=1e-5)
    common.assert_allclose(st.logical_xor(zero, one), one, tol=1e-5)
    common.assert_allclose(st.logical_xor(one, zero), one, tol=1e-5)
    common.assert_allclose(st.logical_xor(one, one), zero, tol=1e-5)


@pytest.mark.parametrize("use_geometric_mean", [False, True])
def test_logical_ops_geometric_mean(use_geometric_mean):
    x = torch.tensor([0.1, 0.5, 0.9])
    y = torch.tensor([0.9, 0.2, 0.4])
    for fn in [st.logical_and, st.logical_or, st.logical_xor]:
        out = fn(x, y, use_geometric_mean=use_geometric_mean)
        common.assert_softbool(out)
        assert out.shape == x.shape


@pytest.mark.parametrize("use_geometric_mean", [False, True])
def test_logical_ops_geometric_mean_truth_table(use_geometric_mean):
    zero = torch.tensor([0.0])
    one = torch.tensor([1.0])
    # geometric mean with epsilon can't reach exactly 0, use looser tolerance
    tol = 2e-5 if use_geometric_mean else 1e-5

    common.assert_allclose(
        st.logical_and(one, one, use_geometric_mean=use_geometric_mean), one, tol=tol
    )
    common.assert_allclose(
        st.logical_and(one, zero, use_geometric_mean=use_geometric_mean), zero, tol=tol
    )
    common.assert_allclose(
        st.logical_or(zero, one, use_geometric_mean=use_geometric_mean), one, tol=tol
    )
    common.assert_allclose(
        st.logical_or(zero, zero, use_geometric_mean=use_geometric_mean), zero, tol=tol
    )


@pytest.mark.parametrize("use_geometric_mean", [False, True])
def test_all_any_geometric_mean(use_geometric_mean):
    x = torch.tensor([0.8, 0.9, 1.0])
    out_all = st.all(x, dim=-1, use_geometric_mean=use_geometric_mean)
    out_any = st.any(x, dim=-1, use_geometric_mean=use_geometric_mean)
    common.assert_softbool(out_all.unsqueeze(0))
    common.assert_softbool(out_any.unsqueeze(0))
    common.assert_finite(out_all.unsqueeze(0), msg="all geometric_mean")
    common.assert_finite(out_any.unsqueeze(0), msg="any geometric_mean")


# ---------------------------------------------------------------------------
# Sigmoidal and softrelu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_sigmoidal_bounds(mode):
    x = common.make_tensor((10,), torch.float32)
    out = st.sigmoidal(x, mode=mode, softness=1.0)
    common.assert_softbool(out)
    common.assert_finite(out, msg=f"sigmoidal mode={mode}")


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_sigmoidal_gradient_finite(mode):
    x = common.gradient_input((5,), torch.float32)
    out = st.sigmoidal(x, mode=mode, softness=1.0)
    out.sum().backward()
    common.assert_finite(x.grad, msg=f"sigmoidal mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0", "_c1_pnorm", "_c2_pnorm"])
@pytest.mark.parametrize("softness", [0.1, 1.0, 5.0])
def test_sigmoidal_equals_argmax(mode, softness):
    """sigmoidal(x, s) must equal argmax([0, x], softness=s', standardize=False)[1].

    For smooth: s' = s. For piecewise modes: s' = 5s (piecewise sigmoidal has a
    built-in 1/5 scaling to match smooth's effective transition width).
    """
    x = torch.tensor([-2.0, -0.5, 0.0, 0.3, 1.0, 3.0], dtype=torch.float64)
    expected = st.sigmoidal(x, mode=mode, softness=softness)
    # Piecewise modes have a /5 factor, so argmax needs 5*softness.
    # c1_pnorm/c2_pnorm in sigmoidal correspond to c1/c2 in argmax.
    argmax_softness = softness if mode == "smooth" else 5.0 * softness
    argmax_mode = {"_c1_pnorm": "c1", "_c2_pnorm": "c2"}.get(mode, mode)
    pairs = torch.stack([torch.zeros_like(x), x], dim=-1)
    argmax_outs = []
    for i in range(pairs.shape[0]):
        p = pairs[i]
        out = st.argmax(
            p, dim=0, softness=argmax_softness, mode=argmax_mode, standardize=False
        )
        argmax_outs.append(out[1].item())
    actual = torch.tensor(argmax_outs, dtype=torch.float64)
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"sigmoidal != argmax([0,x])[1] for mode={mode} softness={softness}",
    )


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
@pytest.mark.parametrize("gated", [False, True])
def test_softrelu_non_negative(mode, gated):
    x = common.make_tensor((10,), torch.float32)
    out = st.softrelu(x, mode=mode, softness=1.0, gated=gated)
    if not gated:
        assert np.all(out.detach().numpy() >= -1e-7), (
            f"softrelu negative for mode={mode} gated={gated}"
        )
    common.assert_finite(out, msg=f"softrelu mode={mode} gated={gated}")


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
@pytest.mark.parametrize("gated", [False, True])
def test_softrelu_gradient_finite(mode, gated):
    x = common.gradient_input((5,), torch.float32)
    out = st.softrelu(x, mode=mode, softness=1.0, gated=gated)
    out.sum().backward()
    common.assert_finite(x.grad, msg=f"softrelu mode={mode} gated={gated}")


@pytest.mark.parametrize("mode", ["smooth", "c0", "_c1_pnorm", "_c2_pnorm"])
@pytest.mark.parametrize("softness", [0.1, 1.0, 5.0])
def test_softrelu_gated_equals_max(mode, softness):
    """softrelu(x, gated=True) must equal max([0, x], softness=s', standardize=False).

    For smooth: s' = s. For piecewise modes: s' = 5s (piecewise has a built-in
    1/5 scaling to match smooth's effective transition width).
    """
    x = torch.tensor([-2.0, -0.5, 0.0, 0.3, 1.0, 3.0], dtype=torch.float64)
    expected = st.softrelu(x, mode=mode, softness=softness, gated=True)
    # Piecewise modes have a /5 factor, so max needs 5*softness.
    # c1_pnorm/c2_pnorm in softrelu correspond to c1/c2 in max.
    max_softness = softness if mode == "smooth" else 5.0 * softness
    max_mode = {"_c1_pnorm": "c1", "_c2_pnorm": "c2"}.get(mode, mode)
    pairs = torch.stack([torch.zeros_like(x), x], dim=-1)
    max_outs = []
    for i in range(pairs.shape[0]):
        p = pairs[i]
        out = st.max(p, dim=0, softness=max_softness, mode=max_mode, standardize=False)
        max_outs.append(out.values.item())
    actual = torch.tensor(max_outs, dtype=torch.float64)
    np.testing.assert_allclose(
        actual.numpy(),
        expected.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"softrelu(gated=True) != max([0,x]) for mode={mode} softness={softness}",
    )


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
@pytest.mark.parametrize("softness", [0.1, 1.0, 5.0])
def test_softrelu_ungated_is_integral_of_sigmoidal(mode, softness):
    x = torch.tensor([-1.5, -0.3, 0.0, 0.4, 1.2], dtype=torch.float64, requires_grad=True)

    grad_softrelu = []
    for i in range(x.shape[0]):
        xi = x[i].detach().clone().requires_grad_(True)
        val = st.softrelu(xi, mode=mode, softness=softness, gated=False)
        val.backward()
        grad_softrelu.append(xi.grad.item())

    sigmoidal_vals = st.sigmoidal(x.detach(), mode=mode, softness=softness)
    np.testing.assert_allclose(
        np.array(grad_softrelu),
        sigmoidal_vals.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"d/dx softrelu != sigmoidal for mode={mode} softness={softness}",
    )


# ---------------------------------------------------------------------------
# c1_pnorm / c2_pnorm correspondence with simplex projections
# ---------------------------------------------------------------------------


def test_c1_pnorm_sigmoidal_matches_simplex_projection():
    """sigmoidal(x, mode='c1_pnorm') must equal the 2D p=3/2 simplex projection.

    Piecewise sigmoidal includes a 1/5 scaling factor, so the projection input
    is x/(5*softness) rather than x/softness.
    """
    from softtorch.projections_simplex import _proj_unit_simplex_pnorm_q3

    xs = torch.linspace(-3.0, 3.0, 50, dtype=torch.float64)
    for softness in [0.1, 1.0, 5.0]:
        sig_vals = st.sigmoidal(xs, softness=softness, mode="_c1_pnorm")
        proj_vals = []
        for x_scalar in xs:
            v = torch.tensor(
                [x_scalar / (5.0 * softness), 0.0], dtype=torch.float64
            ).unsqueeze(0)
            proj = _proj_unit_simplex_pnorm_q3(v)
            proj_vals.append(float(proj[0, 0]))
        proj_vals = torch.tensor(proj_vals, dtype=torch.float64)
        np.testing.assert_allclose(
            sig_vals.numpy(),
            proj_vals.numpy(),
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"sigmoidal c1_pnorm != 2D p=3/2 projection, softness={softness}",
        )


def test_c2_pnorm_sigmoidal_matches_simplex_projection():
    """sigmoidal(x, mode='c2_pnorm') must equal the 2D p=4/3 simplex projection.

    Piecewise sigmoidal includes a 1/5 scaling factor, so the projection input
    is x/(5*softness) rather than x/softness.
    """
    from softtorch.projections_simplex import _proj_unit_simplex_pnorm_q4

    xs = torch.linspace(-3.0, 3.0, 50, dtype=torch.float64)
    for softness in [0.1, 1.0, 5.0]:
        sig_vals = st.sigmoidal(xs, softness=softness, mode="_c2_pnorm")
        proj_vals = []
        for x_scalar in xs:
            v = torch.tensor(
                [x_scalar / (5.0 * softness), 0.0], dtype=torch.float64
            ).unsqueeze(0)
            proj = _proj_unit_simplex_pnorm_q4(v)
            proj_vals.append(float(proj[0, 0]))
        proj_vals = torch.tensor(proj_vals, dtype=torch.float64)
        np.testing.assert_allclose(
            sig_vals.numpy(),
            proj_vals.numpy(),
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"sigmoidal c2_pnorm != 2D p=4/3 projection, softness={softness}",
        )


@pytest.mark.parametrize("mode", ["_c1_pnorm", "_c2_pnorm"])
def test_pnorm_sigmoidal_grad_vs_finite_diff(mode):
    x = torch.tensor([-0.7, -0.3, 0.0, 0.3, 0.7], dtype=torch.float64)

    def loss(z):
        return st.sigmoidal(z, softness=1.0, mode=mode).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"sigmoidal {mode}")


@pytest.mark.parametrize("mode", ["_c1_pnorm", "_c2_pnorm"])
def test_pnorm_softrelu_grad_vs_finite_diff(mode):
    x = torch.tensor([-0.7, -0.3, 0.0, 0.3, 0.7], dtype=torch.float64)

    def loss(z):
        return st.softrelu(z, softness=1.0, mode=mode).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"softrelu {mode}")


@pytest.mark.parametrize("mode", ["_c1_pnorm", "_c2_pnorm"])
def test_pnorm_softrelu_integral_grad_vs_finite_diff(mode):
    x = torch.tensor([-0.7, -0.3, 0.0, 0.3, 0.7], dtype=torch.float64)

    def loss(z):
        return st.softrelu(z, softness=1.0, mode=mode, gated=False).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"softrelu {mode} integral")


@pytest.mark.parametrize("mode", ["_c1_pnorm", "_c2_pnorm"])
def test_pnorm_softrelu_gated_matches_x_times_sigmoidal(mode):
    xs = torch.linspace(-3.0, 3.0, 50, dtype=torch.float64)
    for softness in [0.1, 1.0, 5.0]:
        relu_vals = st.softrelu(xs, softness=softness, mode=mode, gated=True)
        sig_vals = xs * st.sigmoidal(xs, softness=softness, mode=mode)
        np.testing.assert_allclose(
            relu_vals.detach().numpy(),
            sig_vals.detach().numpy(),
            atol=1e-10,
            err_msg=f"gated softrelu {mode} != x*sigmoidal, softness={softness}",
        )


@pytest.mark.parametrize("mode", ["_c1_pnorm", "_c2_pnorm"])
@pytest.mark.parametrize("softness", [0.1, 1.0, 5.0])
def test_pnorm_softrelu_integral_derivative_is_sigmoidal(mode, softness):
    x = torch.linspace(-0.45, 0.45, 20, dtype=torch.float64)

    grad_softrelu = []
    for i in range(x.shape[0]):
        xi = x[i].detach().clone().requires_grad_(True)
        val = st.softrelu(xi, mode=mode, softness=softness, gated=False)
        val.backward()
        grad_softrelu.append(xi.grad.item())

    sigmoidal_vals = st.sigmoidal(x.detach(), mode=mode, softness=softness)
    np.testing.assert_allclose(
        np.array(grad_softrelu),
        sigmoidal_vals.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"d/dx softrelu != sigmoidal for {mode} softness={softness}",
    )


# ---------------------------------------------------------------------------
# Gradient vs finite differences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_abs_grad_vs_finite_diff(mode):
    x = torch.tensor([-0.8, 0.3, -0.1, 0.6], dtype=torch.float64)

    def loss(z):
        return st.abs(z, softness=1.0, mode=mode).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"abs mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_relu_grad_vs_finite_diff(mode):
    x = torch.tensor([-0.8, 0.3, -0.1, 0.6], dtype=torch.float64)

    def loss(z):
        return st.relu(z, softness=1.0, mode=mode).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"relu mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_clamp_grad_vs_finite_diff(mode):
    x = torch.tensor([-0.8, 0.3, -0.1, 0.6], dtype=torch.float64)

    def loss(z):
        return st.clamp(z, -0.25, 0.25, softness=1.0, mode=mode).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"clamp mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_greater_grad_vs_finite_diff(mode):
    x = torch.tensor([-0.5, 0.3, 0.0, 0.7], dtype=torch.float64)
    y = torch.tensor([0.1, 0.2, 0.0, 0.9], dtype=torch.float64)

    def loss(z):
        return st.greater(z, y, softness=1.0, mode=mode).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"greater mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_sign_grad_vs_finite_diff(mode):
    x = torch.tensor([-0.8, 0.3, -0.1, 0.6], dtype=torch.float64)

    def loss(z):
        return st.sign(z, softness=1.0, mode=mode).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"sign mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_round_grad_vs_finite_diff(mode):
    x = torch.tensor([-0.7, 0.3, 1.5, -1.2], dtype=torch.float64)

    def loss(z):
        return st.round(z, softness=0.1, mode=mode).sum()

    common.assert_grad_matches_finite_diff(loss, x, msg=f"round mode={mode}")


# ---------------------------------------------------------------------------
# relu/clamp gated parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_relu_gated(mode):
    """relu with gated=True produces finite output and gradient."""
    x = common.gradient_input((5,), torch.float32)
    out = st.relu(x, mode=mode, softness=1.0, gated=True)
    common.assert_finite(out, msg=f"relu gated mode={mode}")
    assert out.shape == x.shape

    out.sum().backward()
    common.assert_finite(x.grad, msg=f"relu gated grad mode={mode}")


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_clamp_gated(mode):
    """clamp with gated=True produces finite output and gradient."""
    x = common.gradient_input((5,), torch.float32)
    out = st.clamp(x, -0.25, 0.25, mode=mode, softness=1.0, gated=True)
    common.assert_finite(out, msg=f"clamp gated mode={mode}")
    assert out.shape == x.shape

    out.sum().backward()
    common.assert_finite(x.grad, msg=f"clamp gated grad mode={mode}")


# ---------------------------------------------------------------------------
# round neighbor_radius parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("neighbor_radius", [1, 3, 5, 10])
@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_round_neighbor_radius(neighbor_radius, mode):
    """round with different neighbor_radius produces finite output."""
    x = common.gradient_input((5,), torch.float32)
    out = st.round(x, mode=mode, softness=1.0, neighbor_radius=neighbor_radius)
    common.assert_finite(
        out, msg=f"round neighbor_radius={neighbor_radius} mode={mode}"
    )
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# isclose rtol/atol parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rtol", [1e-5, 1e-3, 1e-1])
@pytest.mark.parametrize("atol", [1e-8, 1e-5, 1e-2])
@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_isclose_rtol_atol(rtol, atol, mode):
    """isclose with different rtol/atol produces finite output with correct shape."""
    x = torch.tensor([1.0, 1.0001, 1.1, 2.0], dtype=torch.float32)
    y = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    out = st.isclose(x, y, mode=mode, softness=1.0, rtol=rtol, atol=atol)
    common.assert_finite(out, msg=f"isclose rtol={rtol} atol={atol} mode={mode}")
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# c1_pnorm/c2_pnorm softrelu matches max projection
# ---------------------------------------------------------------------------


def test_c1_pnorm_softrelu_matches_max_projection():
    """gated softrelu(x, mode='c1_pnorm') must equal x * sigmoidal(x)."""
    xs = torch.linspace(-3.0, 3.0, 50, dtype=torch.float64)
    for softness in [0.1, 1.0, 5.0]:
        relu_vals = st.softrelu(xs, softness=softness, mode="_c1_pnorm", gated=True)
        sig_vals = st.sigmoidal(xs, softness=softness, mode="_c1_pnorm")
        expected = xs * sig_vals
        np.testing.assert_allclose(
            relu_vals.detach().numpy(),
            expected.detach().numpy(),
            rtol=1e-5,
            err_msg=f"softrelu(gated=True) c1_pnorm != x*sigmoidal, softness={softness}",
        )


def test_c2_pnorm_softrelu_matches_max_projection():
    """gated softrelu(x, mode='c2_pnorm') must equal x * sigmoidal(x)."""
    xs = torch.linspace(-3.0, 3.0, 50, dtype=torch.float64)
    for softness in [0.1, 1.0, 5.0]:
        relu_vals = st.softrelu(xs, softness=softness, mode="_c2_pnorm", gated=True)
        sig_vals = st.sigmoidal(xs, softness=softness, mode="_c2_pnorm")
        expected = xs * sig_vals
        np.testing.assert_allclose(
            relu_vals.detach().numpy(),
            expected.detach().numpy(),
            rtol=1e-5,
            err_msg=f"softrelu(gated=True) c2_pnorm != x*sigmoidal, softness={softness}",
        )
