"""Gradient correctness tests: autodiff vs finite differences.

Exhaustively tests all (op, method, mode) combinations to ensure
analytic gradients match central finite differences in float64.
"""

import pytest
import torch

import softtorch as st

from . import common


# ---------------------------------------------------------------------------
# Methods per function (same as test_arraywise.py)
# ---------------------------------------------------------------------------

VALUE_METHODS = ["softsort", "neuralsort", "fast_soft_sort", "ot", "sorting_network"]
ARG_METHODS = ["softsort", "neuralsort", "ot", "sorting_network"]

FUNCTION_METHODS = {
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


# ---------------------------------------------------------------------------
# Loss functions (scalar output for autograd)
# ---------------------------------------------------------------------------

# Weights for non-trivial loss; avoids constant-sum outputs (e.g. sum(rank)
# is constant for OT since the transport plan is bistochastic).
W5 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
W2 = torch.tensor([1.0, 2.0], dtype=torch.float64)


def _make_loss(op, method, mode, softness=1.0):
    """Build a scalar loss function for the given (op, method, mode)."""

    def loss(x):
        kwargs = dict(softness=softness, mode=mode, method=method)
        if op == "quantile":
            return (W5 * st.quantile(x, 0.5, dim=-1, **kwargs)).sum()
        elif op == "argquantile":
            out = st.argquantile(x, 0.5, dim=-1, **kwargs)
            return (W5 * out).sum()
        elif op == "topk":
            result = st.topk(x, k=2, dim=-1, **kwargs)
            return (W2 * result.values).sum()
        elif op == "sort":
            result = st.sort(x, **kwargs, return_indices=False)
            return (W5 * result.values).sum()
        elif op in ("max", "min"):
            result = getattr(st, op)(x, dim=-1, **kwargs)
            return (
                (W5 * result.values).sum() if result.values.ndim > 0 else result.values
            )
        elif op == "median":
            result = st.median(x, dim=-1, **kwargs)
            return (
                (W5 * result.values).sum() if result.values.ndim > 0 else result.values
            )
        elif op in ("argmax", "argmin"):
            out = getattr(st, op)(x, **kwargs)
            return (W5 * out).sum()
        elif op == "argmedian":
            out = st.argmedian(x, **kwargs)
            return (W5 * out).sum()
        elif op == "argsort":
            out = st.argsort(x, **kwargs)
            return (W5[None, :] * out).sum()
        elif op == "rank":
            out = st.rank(x, **kwargs)
            return (W5 * out).sum()
        else:
            out = getattr(st, op)(x, **kwargs)
            return (W5 * out).sum()

    return loss


# ---------------------------------------------------------------------------
# Build exhaustive parametrization
# ---------------------------------------------------------------------------

_CASES = []
for op, methods in FUNCTION_METHODS.items():
    for method in methods:
        for mode in MODES:
            _CASES.append((op, method, mode))

_IDS = [f"{op}_{method}_{mode}" for op, method, mode in _CASES]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op,method,mode", _CASES, ids=_IDS)
def test_grad_vs_finite_diff(op, method, mode):
    """Analytic gradient matches finite differences for all (op, method, mode)."""
    # Non-uniform spacing avoids degenerate finite-diff behavior with standardized
    # ops (uniform spacing means a shift perturbation barely changes the standardized
    # values, making finite diff noisy while autodiff correctly gives ~0).
    x = torch.tensor([-0.8, -0.1, 0.3, 0.5, 1.2], dtype=torch.float64)
    loss = _make_loss(op, method, mode)
    # OT c0 (L2) transport plan is Gamma_ij = (1/tau)*max(f_i+g_j-C_ij, 0), which
    # is C0 but not C1 at support boundaries. The implicit function theorem requires
    # the dual Hessian, which involves d/dy max(S,0) = Heaviside (discontinuous).
    # This makes the implicit diff linear system ill-conditioned, causing inherently
    # lower gradient accuracy. Higher smoothness modes (c1/c2/smooth) don't have
    # this issue because their P^(q-1) terms are differentiable.
    if method == "ot" and mode == "c0":
        # Worse in softtorch than softjax due to torchopt CG solver differences
        common.assert_grad_matches_finite_diff(
            loss,
            x,
            rtol=1.0,
            atol=1.0,
            msg=f"{op} {method} {mode}",
        )
    elif method == "ot":
        common.assert_grad_matches_finite_diff(
            loss,
            x,
            rtol=5e-2,
            atol=5e-2,
            msg=f"{op} {method} {mode}",
        )
    else:
        common.assert_grad_matches_finite_diff(loss, x, msg=f"{op} {method} {mode}")
