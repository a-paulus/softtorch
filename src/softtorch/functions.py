# functions.py

import builtins
from typing import Literal

import torch
import torch.nn.functional as F

from softtorch.projections_permutahedron import _proj_permutahedron
from softtorch.projections_simplex import _proj_simplex
from softtorch.projections_transport_polytope import _proj_transport_polytope
from softtorch.sorting_network import (
    _argsort_via_sorting_network,
    _sort_via_sorting_network,
)
from softtorch.utils import (
    _canonicalize_dim,
    _ensure_float,
    _polyval,
    _quantile_k_a,
    _standardize_and_squash,
    _unsquash_and_destandardize,
    _validate_softness,
)


SoftBool = torch.Tensor  # probability in [0, 1]
SoftIndex = torch.Tensor  # probabilities summing to 1 along the last dim


# Selection operators


def where(condition: SoftBool, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes a soft version of [torch.where](https://pytorch.org/docs/stable/generated/torch.where.html) as `x * condition + y * (1.0 - condition)`.

    **Arguments:**
    - `condition`: SoftBool condition Tensor, same shape as `x` and `y`.
    - `x`: First input Tensor, same shape as `condition`.
    - `y`: Second input Tensor, same shape as `condition`.

    **Returns:**

    Tensor of the same shape as `x` and `y`, interpolating between `x` and `y` according to `condition` in [0, 1].
    """
    return x * condition + y * (1.0 - condition)


def take_along_dim(
    x: torch.Tensor,  # (..., n, ...)
    soft_index: SoftIndex,  # (..., k, ..., [n])
    dim: int | None = None,
) -> torch.Tensor:  # (..., k, ...)
    """Performs a soft version of [torch.take_along_dim](https://pytorch.org/docs/stable/generated/torch.take_along_dim.html) via a weighted dot product.

    **Arguments:**

    - `x`: Input Tensor of shape (..., n, ...).
    - `soft_index`: A SoftIndex of shape (..., k, ..., [n]) (positive Tensor which sums to 1 over the last dimension). If dim is None, must be two-dimensional. If dim is not None, must have x.ndim + 1 == soft_index.ndim, and x must be broadcast-compatible with soft_index along dimensions other than dim.
    - `dim`: Dim along which to apply the soft index. If None, the input is flattened before applying the soft indices.

    **Returns:**

    Tensor of shape (..., k, ...), representing the result after soft selection along the specified dim.
    """
    if dim is None:
        x = torch.flatten(x)
        dim = 0
    else:
        dim = _canonicalize_dim(dim, x.ndim)
    if x.ndim + 1 != soft_index.ndim:
        raise ValueError(
            f"Input x and soft_index must have compatible dimensions, "
            f"but got x.ndim={x.ndim} and soft_index.ndim={soft_index.ndim}. "
            f"Should be x.ndim + 1 == soft_index.ndim."
        )
    x = torch.movedim(x, dim, -1)  # (..., ..., n)
    soft_index = torch.movedim(soft_index, dim, -2)  # (..., ..., k, [n])
    dotprod = torch.einsum("...n,...kn->...k", x, soft_index)  # (..., ..., k)
    dotprod = torch.movedim(dotprod, -1, dim)  # (..., k, ...)
    return dotprod


def take(
    x: torch.Tensor,  # (..., n, ...)
    soft_index: SoftIndex,  # (k, [n])
    dim: int | None = None,
) -> torch.Tensor:  # (..., k, ...)
    """Performs a soft version of [torch.take](https://pytorch.org/docs/stable/generated/torch.take.html) via a weighted dot product.

    **Arguments:**

    - `x`: Input Tensor of shape (..., n, ...).
    - `soft_index`: A SoftIndex of shape (k, [n]) (positive Tensor which sums to 1 over the last dimension).
    - `dim`: Dim along which to apply the soft index. If None, the input is flattened before applying the soft indices.

    **Returns:**

    Tensor of shape (..., k, ...) after soft selection.
    """
    if soft_index.ndim != 2:
        raise ValueError(
            f"soft_index must be of shape (k, [n]), but got shape {soft_index.shape}."
        )
    if dim is None:
        x = torch.flatten(x)
        dim = 0
    else:
        dim = _canonicalize_dim(dim, x.ndim)
        x = torch.movedim(x, dim, -1)  # (..., ..., n)
    soft_index = torch.reshape(
        soft_index, (1,) * (x.ndim - 1) + soft_index.shape
    )  # (1..., 1..., k, [n])
    x = torch.unsqueeze(x, dim=dim)  # (..., 1, ..., n)
    soft_index = torch.movedim(soft_index, -2, dim)  # (1..., k, 1..., [n])
    y = torch.sum(x * soft_index, dim=-1)  # (..., k, ...)
    return y


def index_select(
    x: torch.Tensor,  # (..., n, ...)
    soft_index: SoftIndex,  # ([n],)
    dim: int,
    keepdim: bool = True,
) -> torch.Tensor:  # (..., {1}, ...)
    """Performs a soft version of [torch.index_select](https://pytorch.org/docs/stable/generated/torch.index_select.html) via a weighted dot product.

    **Arguments:**

    - `x`: Input Tensor of shape (..., n, ...).
    - `soft_index`: A SoftIndex of shape ([n],) (positive Tensor which sums to 1 over the last dimension).
    - `dim`: Dim along which to apply the soft index.
    - `keepdim`: If True, keeps the reduced dimension as a singleton {1}.

    **Returns:**

    Tensor after soft indexing, shape (..., {1}, ...).
    """
    dim = _canonicalize_dim(dim, x.ndim)
    if x.shape[dim] != soft_index.shape[0]:
        raise ValueError(
            f"Dimension mismatch between x and soft_index along dim {dim}: "
            f"x.shape[{dim}]={x.shape[dim]} vs soft_index.shape[0]={soft_index.shape[0]}"
        )
    x = torch.movedim(x, dim, -1)  # (..., ..., n)
    x_reshaped = x.reshape(-1, x.shape[-1])  # (B, n)
    dotprod = torch.sum(x_reshaped * soft_index[None, :], dim=-1)  # (B,)
    y = dotprod.reshape(x.shape[:-1])  # (..., ...)
    if keepdim:
        y = torch.unsqueeze(y, dim=dim)  # (..., 1, ...)
    return y


def narrow(
    x: torch.Tensor,  # (..., n, ...)
    soft_start: SoftIndex,  # ([n],)
    length: int,
    dim: int = 0,
) -> torch.Tensor:  # (..., length, ...)
    """Performs a soft version of [torch.narrow](https://pytorch.org/docs/stable/generated/torch.narrow.html) via a weighted dot product.

    **Arguments:**

    - `x`: Input Tensor of shape (..., n, ...).
    - `soft_start`: A SoftIndex of shape ([n],) (positive Tensor which sums to 1 over the last dimension).
    - `length`: Length of the slice to extract.
    - `dim`: Dim along which to apply the soft slice.

    **Returns:**

    Tensor of shape (..., length, ...) after soft slicing.
    """
    dim = _canonicalize_dim(dim, x.ndim)
    if not (x.shape[dim] >= length > 0):
        raise ValueError(
            f"length must satisfy 0 < length <= x.shape[dim], "
            f"got length={length}, x.shape[dim]={x.shape[dim]}"
        )

    x_last = torch.movedim(x, dim, -1)  # (..., ..., n)

    y_stack = torch.stack(
        [
            torch.einsum(
                "...n,n->...",
                torch.roll(x_last, shifts=-i, dims=-1),
                soft_start,
            )
            for i in range(length)
        ],
        dim=0,
    )  # (length, ...)
    y_last = torch.movedim(y_stack, 0, -1)  # (..., length)
    y = torch.movedim(y_last, -1, dim)  # (..., length, ...)
    return y


# Tensor-valued operators


def argmax(
    x: torch.Tensor,  # (..., n, ...)
    dim: int | None = None,
    keepdim: bool = False,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Performs a soft version of [torch.argmax](https://pytorch.org/docs/stable/generated/torch.argmax.html) of `x` along the specified dim.

    **Arguments:**

    - `x`: Input Tensor of shape (..., n, ...).
    - `dim`: The dimension along which to compute the argmax. If None, the input Tensor is flattened before computing the argmax.
    - `keepdim`: If True, keeps the reduced dimension as a singleton {1}.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns the result of torch.argmax with a one-hot encoding of the indices.
        - `smooth`: C∞ smooth (entropy-based). Soften unit simplex projection via an entropic regularizer, computed in closed-form via a softmax operation.
        - `c0`: C0 continuous (L2-based). Soften unit simplex projection via a quadratic regularizer, computed via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
        - `c1`: C1 differentiable (p=3/2 p-norm). Computed in closed form via quadratic formula.
        - `c2`: C2 twice differentiable (p=4/3 p-norm). Computed in closed form via Cardano's method.
    - `method`: Method to compute the soft argmax. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Computes the max element via optimal transport projection onto a 2-point support.
        - `softsort`: Computes the max element of the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038). Reduces to projecting `x` onto the unit simplex.
        - `neuralsort`: Computes the max element of the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850).
    - `standardize`: If True, standardizes and squashes the input `x` along the specified dim before applying the softargmax operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.

    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.


    **Returns:**

    A SoftIndex of shape (..., {1}, ..., [n]) (positive Tensor which sums to 1 over the last dimension).
    Represents the probability of an index corresponding to the argmax along the specified dim.
    """

    if mode == "hard" or mode == "_hard":
        indices = torch.argmax(x, dim=dim, keepdim=keepdim)
        num_classes = x.shape[dim] if dim is not None else x.numel()
        soft_index = F.one_hot(indices, num_classes=num_classes).to(x.dtype)
    else:
        x = _ensure_float(x)
        if dim is None:
            num_dims = x.ndim
            x = torch.flatten(x)
            _dim = 0
        else:
            _dim = _canonicalize_dim(dim, x.ndim)

        if standardize:
            x = _standardize_and_squash(x, dim=_dim)

        x_last = torch.movedim(x, _dim, -1)  # (..., ..., n)
        *batch_dims, n = x_last.shape
        if method == "softsort":
            soft_index = _proj_simplex(
                x_last, dim=-1, softness=softness, mode=mode
            )  # (..., ..., [n])
        elif method == "neuralsort":
            A = abs(
                x_last[..., :, None] - x_last[..., None, :],
                mode=mode,
                softness=softness,
            )  # (..., ..., n, n)
            A_sum = torch.sum(A, dim=-1)  # (..., ..., n)
            z = (n - 1) * x_last - A_sum  # (..., ..., n)
            soft_index = _proj_simplex(
                z, dim=-1, softness=softness, mode=mode
            )  # (..., ..., [n])
        elif method == "ot":
            anchors = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)  # (2,)
            anchors = anchors.expand(*batch_dims, 2)  # (..., ..., 2)
            cost = (
                x_last[..., :, None] - anchors[..., None, :]
            ) ** 2  # (..., ..., n, 2)

            mu = torch.ones((n,), device=x.device, dtype=x.dtype) / n  # ([n],)
            nu = torch.tensor(
                [(n - 1) / n, 1 / n], device=x.device, dtype=x.dtype
            )  # ([2],)

            if ot_kwargs is None:
                ot_kwargs = {}
            out = _proj_transport_polytope(
                cost=cost,
                mu=mu,
                nu=nu,
                softness=softness,
                mode=mode,
                **ot_kwargs,
            )  # (..., [n], 2)
            soft_index = out[..., 1]  # (..., ..., [n])
        elif method == "sorting_network":
            P = _argsort_via_sorting_network(
                x_last, softness, mode, descending=True, standardized=standardize
            )  # (..., ..., n, [n])
            soft_index = P[..., 0, :]  # (..., ..., [n])
        else:
            raise ValueError(f"Invalid method: {method}")

        if keepdim:
            if dim is None:
                soft_index = soft_index.reshape(
                    *(1,) * num_dims, n
                )  # (1..., 1, 1..., [n])
            else:
                soft_index = soft_index.unsqueeze(dim=_dim)  # (..., {1}, ..., [n])
    return soft_index


def max(
    x: torch.Tensor,  # (..., n, ...)
    dim: int | None = None,
    keepdim: bool = False,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> torch.Tensor | torch.return_types.max[torch.Tensor, SoftIndex]:
    """Performs a soft version of [torch.max](https://pytorch.org/docs/stable/generated/torch.max.html) of `x` along the specified dim.

    For methods other than `fast_soft_sort` and `sorting_network`, implemented as [`softtorch.argmax`][] followed by [`softtorch.take_along_dim`][], see respective documentations for details.
    For `fast_soft_sort` and `sorting_network`, uses [`softtorch.sort`][] to compute soft sorted values and retrieves the maximum as the first element. See [`softtorch.sort`][] for method details.

    **Extra Arguments:**

    - `gated_grad`: If `False`, stops the gradient flow through the soft index. True gives gated 'SiLU-style' gradients, while False gives integrated 'Softplus-style' gradients.

    **Returns:**

    - If `dim` is None (default):  Scalar tensor representing the soft maximum of the flattened `x`.
    - If `dim` is specified: Namedtuple containing two fields:
        - `values`: Tensor of shape (..., {1}, ...) representing the soft maximum of `x`  along the specified dim.
        - `indices`: SoftIndex of shape (..., {1}, ..., [n]) (positive Tensor which sums to 1 over the last dimension). Represents the soft  indices of the maximum values.
    """

    if dim is None:
        num_dims = x.ndim
        x = torch.flatten(x)
        _dim = 0
    else:
        _dim = _canonicalize_dim(dim, x.ndim)

    if mode == "hard":
        if dim is None:
            return torch.max(x, dim=_dim, keepdim=False).values
        else:
            ret = torch.max(x, dim=dim, keepdim=keepdim)
            values = ret.values  # (..., {1}, ...)
            indices = ret.indices  # (..., {1}, ...)
            soft_index = F.one_hot(indices, num_classes=x.shape[dim]).to(
                x.dtype
            )  # (..., {1}, ..., [n])
            return torch.return_types.max((values, soft_index))
    elif method in ("fast_soft_sort", "smooth_sort", "sorting_network"):
        soft_sorted = sort(
            x,
            dim=_dim,
            descending=True,
            softness=softness,
            standardize=standardize,
            mode=mode,
            method=method,
        )
        values = soft_sorted.values.select(_dim, 0)
        if dim is None:
            if keepdim:
                values = values.reshape(*(1,) * num_dims)
            return values
        if keepdim:
            values = values.unsqueeze(_dim)
        return torch.return_types.max((values, None))
    else:
        soft_index = argmax(
            x,
            dim=dim,
            keepdim=True,
            softness=softness,
            mode=mode,
            method=method,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
        )  # (..., 1, ..., [n])
        if not gated_grad:
            soft_index = soft_index.detach()
        values = take_along_dim(x, soft_index, dim=dim)  # (..., 1, ...)
        if dim is None:
            values = values.reshape(*(1,) * num_dims)  # (1..., 1, 1...)
        if dim is None or not keepdim:
            # dim=None: always squeeze to scalar (matching torch.max behavior).
            # dim specified + keepdim=False: squeeze the reduced dimension.
            values = torch.squeeze(values, dim=_dim)  # (..., ...)
            soft_index = torch.squeeze(soft_index, dim=_dim)  # (..., ..., [n])

        if dim is None:
            return values
        else:
            return torch.return_types.max(
                (values, soft_index)
            )  # (..., {1}, ...), (..., {1}, ..., [n])


def argmin(
    x: torch.Tensor,  # (..., n, ...)
    dim: int | None = None,
    keepdim: bool = False,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Performs a soft version of [torch.argmin](https://pytorch.org/docs/stable/generated/torch.argmin.html) of `x` along the specified dim.
    Implemented as [`softtorch.argmax`][] on `-x`, see respective documentation for details.
    """
    return argmax(
        -x,
        dim=dim,
        mode=mode,
        method=method,
        softness=softness,
        keepdim=keepdim,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
    )


def min(
    x: torch.Tensor,  # (..., n, ...)
    dim: int | None = None,
    keepdim: bool = False,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> torch.Tensor | torch.return_types.min[torch.Tensor, SoftIndex]:
    """Performs a soft version of [torch.min](https://pytorch.org/docs/stable/generated/torch.min.html) of `x` along the specified dim.
    Implemented via [`softtorch.max`][] on `-x`, see respective documentation for details.
    """
    if dim is None:
        return -max(
            -x,
            dim=dim,
            softness=softness,
            mode=mode,
            method=method,
            keepdim=keepdim,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
            gated_grad=gated_grad,
        )
    else:
        values, soft_index = max(
            -x,
            dim=dim,
            softness=softness,
            mode=mode,
            method=method,
            keepdim=keepdim,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
            gated_grad=gated_grad,
        )
        return torch.return_types.min(
            (-values, soft_index)
        )  # (..., {1}, ...), (..., {1}, ..., [n])


def argsort(
    x: torch.Tensor,  # (..., n, ...)
    dim: int | None = -1,
    descending: bool = False,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> SoftIndex:  # (..., n, ..., [n])
    """Performs a soft version of [torch.argsort](https://pytorch.org/docs/stable/generated/torch.argsort.html) of `x` along the specified dim.

    **Arguments:**

    - `x`: Input Tensor of shape (..., n, ...).
    - `dim`: The dim along which to compute the argsort operation. If None, uses the last dimension.
    - `descending`: If True, sorts in descending order.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns the result of torch.argsort with a one-hot encoding of the indices.
        - `smooth`: Soften projections via an entropic regularizer.
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via a softmax operation.
            - For optimal transport (`ot` method), transport plan is computed via Sinkhorn iterations (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)).
        - `c0`: Soften projections via a quadratic regularizer.
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
            - For optimal transport (`ot` method), transport plan is computed via LBFGS (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`/`c2`: C1 differentiable / C2 twice differentiable. Similar to `c0`, but using p-norm regularizers with p=3/2 and p=4/3, respectively.
    - `method`: Method to compute the soft argsort. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Uses the approach in [Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885).
            Intuition: The sorted elements are selected by specifying n "anchors" and then transporting the ith-largest value to the ith-largest anchor.
            Note: `sinkhorn_max_iter` and `sinkhorn_tol` can be passed via `ot_kwargs` to control convergence.
        - `softsort`: Computes the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            Note: Can introduce gradient discontinuities when elements in `x` are not unique, but is much faster than OT-based method.
        - `neuralsort`: Computes the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850).
    - `standardize`: If True, standardizes and squashes the input `x` along the specified dim before applying the softargsort operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.

    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.

    **Returns:**

    A SoftIndex of shape (..., n, ..., [n]) (positive Tensor which sums to 1 over the last dimension).
    The elements in (..., i, ..., [n]) represent a distribution over values in x for the ith smallest element along the specified dim.
    """
    if dim is None:
        dim = -1
    dim = _canonicalize_dim(dim, x.ndim)

    if mode == "hard" or mode == "_hard":
        indices = torch.argsort(x, dim=dim, descending=descending)  # (..., n, ...)
        num_classes = x.shape[dim]
        soft_index = F.one_hot(indices, num_classes=num_classes).to(
            x.dtype
        )  # (..., n, ..., [n])
    else:
        x = _ensure_float(x)
        if standardize:
            x = _standardize_and_squash(x, dim=dim)
        x_last = torch.movedim(x, dim, -1)  # (..., ..., n)
        *batch_dims, n = x_last.shape

        if method == "softsort":
            anchors = torch.sort(
                x_last, dim=-1, descending=descending
            ).values  # (..., ..., n)
            abs_diff = torch.abs(
                anchors[..., :, None] - x_last[..., None, :]
            )  # (..., ..., n, n)
            soft_index = _proj_simplex(
                -abs_diff, dim=-1, softness=softness, mode=mode
            )  # (..., ..., n, [n])
        elif method == "neuralsort":
            A = abs(
                x_last[..., :, None] - x_last[..., None, :],
                mode=mode,
                softness=softness,
            )  # (..., ..., n, n)
            A_sum = torch.sum(A, dim=-1)  # (..., ..., n)
            i = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)  # (n,)
            if descending:
                i = torch.flip(i, dims=(0,))  # (n,)
            coef = n + 1 - 2 * i  # (n,)
            coef = coef.broadcast_to(*batch_dims, n)  # (..., ..., n)
            z = -(
                coef[..., :, None] * x_last[..., None, :] + A_sum[..., None, :]
            )  # (..., ..., n, n)
            soft_index = _proj_simplex(
                z, dim=-1, softness=softness, mode=mode
            )  # (..., ..., n, [n])
        elif method == "ot":
            anchors = (
                torch.linspace(0, n, n, dtype=x.dtype, device=x.device) / n
            )  # (n,)
            if descending:
                anchors = torch.flip(anchors, dims=(0,))  # (n,)
            anchors = anchors.expand(*batch_dims, n)  # (..., ..., n)

            cost = (
                x_last[..., :, None] - anchors[..., None, :]
            ) ** 2  # (..., ..., n, n)

            mu = torch.ones((n,), dtype=x.dtype, device=x.device) / n  # ([n],)
            nu = torch.ones((n,), dtype=x.dtype, device=x.device) / n  # ([n],)

            if ot_kwargs is None:
                ot_kwargs = {}
            out = _proj_transport_polytope(
                cost=cost, mu=mu, nu=nu, softness=softness, mode=mode, **ot_kwargs
            )  # (..., ..., [n], n)
            soft_index = torch.movedim(out, -1, -2)  # (..., ..., n, [n])
        elif method == "sorting_network":
            soft_index = _argsort_via_sorting_network(
                x_last, softness, mode, descending, standardized=standardize
            )  # (..., ..., n, [n])
        else:
            raise ValueError(f"Invalid method: {method}")

        soft_index = torch.movedim(soft_index, -2, dim)  # (..., n, ..., [n])
    return soft_index


def sort(
    x: torch.Tensor,  # (..., n, ...)
    dim: int | None = -1,
    descending: bool = False,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
    return_indices: bool = True,
) -> torch.return_types.sort[torch.Tensor, SoftIndex]:
    """Performs a soft version of [torch.sort](https://pytorch.org/docs/stable/generated/torch.sort.html) of `x` along the specified dim.

    Most methods go through [`softtorch.argsort`][] + [`softtorch.take_along_dim`][] to produce soft sorted values.
    The exceptions (`fast_soft_sort`, `smooth_sort`, `sorting_network`) bypass soft indices and compute values directly:

    - `fast_soft_sort`: permutahedron projection via PAV isotonic regression ([Blondel et al., 2020](https://arxiv.org/abs/2002.08871)). In `smooth` mode, uses an entropic (log-KL) variant that is piecewise smooth but not C∞ (discontinuities at argsort chamber boundaries). The PAV step uses Numba JIT (CPU-only); GPU tensors are transferred to CPU for the forward pass.
    - `smooth_sort`: not available in SoftTorch, see `softjax.sort` for the SoftJAX-only implementation.
    - `sorting_network`: soft bitonic sorting network ([Petersen et al., 2021](https://arxiv.org/abs/2105.04019)).

    **Extra Arguments:**

    - `gated_grad`: If `False`, stops the gradient flow through the soft index. True gives gated 'SiLU-style' gradients, while False gives integrated 'Softplus-style' gradients.
    - `return_indices`: If `False`, skips computation of the soft index (indices will be `None`). This avoids the O(n²) memory cost of materializing the n×n soft permutation matrix.

    **Returns:**

    - Namedtuple containing two fields:
        - `values`: Soft sorted values of `x`, shape (..., n, ...).
        - `indices`: SoftIndex of shape (..., n, ..., [n]) (positive Tensor which sums
            to 1 over the last dimension). Represents the soft indices of the sorted
            values. `None` if `return_indices=False`, or when using `fast_soft_sort`
            or `sorting_network` methods.
    """
    if dim is None:
        dim = -1
    dim = _canonicalize_dim(dim, x.ndim)

    if mode == "hard" or (
        mode == "_hard" and method in ("fast_soft_sort", "smooth_sort", "sorting_network")
    ):
        out = torch.sort(x, dim=dim, descending=descending)
        values = out.values  # (..., n, ...)
        if return_indices:
            indices = out.indices  # (..., n, ...)
            soft_index = F.one_hot(indices, num_classes=x.shape[dim]).to(
                x.dtype
            )  # (..., n, ..., [n])
        else:
            soft_index = None
    else:
        x = _ensure_float(x)
        if method == "sorting_network":
            if standardize:
                x, mean, std = _standardize_and_squash(x, dim=dim, return_mean_std=True)
            x_last = torch.movedim(x, dim, -1)
            soft_values = _sort_via_sorting_network(
                x_last,
                softness=softness,
                mode=mode,
                descending=descending,
                standardized=standardize,
            )
            soft_values = torch.movedim(soft_values, -1, dim)
            if standardize:
                soft_values = _unsquash_and_destandardize(soft_values, mean, std)
            values = soft_values
            soft_index = None
        elif method == "fast_soft_sort":
            if standardize:
                x, mean, std = _standardize_and_squash(x, dim=dim, return_mean_std=True)
            x_last = torch.movedim(x, dim, -1)  # (..., ..., n)
            *batch_dims, n = x_last.shape
            w = x_last
            anchors = torch.arange(n, dtype=x.dtype, device=x.device) / (
                n - 1 if n > 1 else 1
            )
            anchors = anchors.expand(*batch_dims, n)
            soft_values = _proj_permutahedron(anchors, w, softness=softness, mode=mode)
            soft_values = torch.movedim(soft_values, -1, dim)
            if descending:
                soft_values = torch.flip(soft_values, dims=(dim,))
            if standardize:
                soft_values = _unsquash_and_destandardize(soft_values, mean, std)
            values = soft_values
            soft_index = None
        elif method == "smooth_sort":
            raise NotImplementedError(
                "smooth_sort is not supported in SoftTorch. "
                "Use method='fast_soft_sort' with mode='smooth' instead, or use SoftJAX."
            )
        else:
            soft_index = argsort(
                x=x,
                dim=dim,
                descending=descending,
                softness=softness,
                mode=mode,
                method=method,
                standardize=standardize,
                ot_kwargs=ot_kwargs,
            )  # (..., n, ..., [n])
            if not gated_grad:
                soft_index = soft_index.detach()
            values = take_along_dim(x, soft_index, dim=dim)
            if not return_indices:
                soft_index = None

    ret = torch.return_types.sort(
        (values, soft_index)
    )  # (..., n, ...), (..., n, ..., [n])
    return ret


def argquantile(
    x: torch.Tensor,  # (..., n, ...)
    q: torch.Tensor | float,  # scalar or (k,)
    dim: int | None = None,
    keepdim: bool = False,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "neuralsort",
    interpolation: Literal[
        "linear", "lower", "higher", "nearest", "midpoint"
    ] = "linear",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Performs a soft version of [torch.quantile](https://pytorch.org/docs/stable/generated/torch.quantile.html)
    of `x` along the specified dim.

    **Arguments:**

    - `x`: Input Tensor of shape (..., n, ...).
    - `q`: Scalar quantile or 1-D Tensor of quantiles in [0, 1]. When a 1-D tensor of length k is passed, the q dimension is prepended to the output shape.
    - `dim`: The dim along which to compute the argquantile. If None, the input Tensor is flattened before computing the argquantile.
    - `keepdim`: If True, keeps the reduced dimension as a singleton {1}.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns a one/two-hot encoding of the indices corresponding to the torch.quantile definitions.
        - `smooth`: Soften projections via an entropic regularizer.
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via a softmax operation.
            - For optimal transport (`ot` method), transport plan is computed via Sinkhorn iterations (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)).
        - `c0`: Soften projections via a quadratic regularizer.
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
            - For optimal transport (`ot` method), transport plan is computed via LBFGS (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`/`c2`: C1 differentiable / C2 twice differentiable. Similar to `c0`, but using p-norm regularizers with p=3/2 and p=4/3, respectively.
    - `method`: Method to compute the soft argquantile. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Uses a variation of the soft quantile approach in [Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885), which is adapted to converge to the standard quantile definitions for small softness. Depending on the interpolation, either a lower and upper quantile are computed and combined, or just a single quantile is computed.
            Intuition: The sorted elements are selected by specifying 4 or 3 "anchors" and then transporting the upper/lower quantile values to the appropriate anchors.
            Note: `sinkhorn_max_iter` and `sinkhorn_tol` can be passed via `ot_kwargs` to control convergence.
        - `softsort`: Computes the upper and lower quantiles via the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            Note: Can introduce gradient discontinuities when elements in `x` are not unique, but is much faster than OT-based method.
        - `neuralsort`: Computes the upper and lower quantiles via the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850).
    - `interpolation`: Method to compute the quantile, following the options in [torch.quantile](https://pytorch.org/docs/stable/generated/torch.quantile.html).
    - `standardize`: If True, standardizes and squashes the input `x` along the specified dim before applying the softargquantile operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.

    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.

    **Returns:**

    A SoftIndex of shape (..., {1}, ..., [n]) for scalar q, or (k, ..., {1}, ..., [n]) for vector q of length k (q dimension prepended). Positive Tensor which sums to 1 over the last dimension. It represents a distribution over values in x being the q-quantile along the specified dim.
    """
    q_t = torch.as_tensor(q)
    if q_t.ndim > 1:
        raise ValueError(
            f"q must be scalar or 1-D, got q with shape {tuple(q_t.shape)}"
        )
    if q_t.ndim == 1:
        results = [
            argquantile(
                x,
                q=qi.item(),
                dim=dim,
                keepdim=keepdim,
                softness=softness,
                mode=mode,
                method=method,
                interpolation=interpolation,
                standardize=standardize,
                ot_kwargs=ot_kwargs,
            )
            for qi in q_t
        ]
        return torch.stack(results, dim=0)

    orig_dim_is_none = dim is None
    if dim is None:
        num_dims = x.ndim
        x = torch.flatten(x)
        dim = 0
    else:
        dim = _canonicalize_dim(dim, x.ndim)

    if mode not in ("hard", "_hard"):
        x = _ensure_float(x)
    if standardize and mode not in ("hard", "_hard"):
        x = _standardize_and_squash(x, dim=dim)

    x_last = torch.movedim(x, dim, -1)  # (..., ..., n)
    *batch_dims, n = x_last.shape

    q = torch.tensor(q, dtype=x.dtype, device=x.device)
    q = torch.clamp(q, 0.0, 1.0)
    k, a, take_next = _quantile_k_a(q, n, interpolation)
    a_b = torch.unsqueeze(a, dim=-1)  # (..., ..., 1)
    kp1 = torch.minimum(k + 1, torch.tensor(n - 1, dtype=k.dtype, device=k.device))

    if mode == "hard" or mode == "_hard":
        indices = torch.argsort(x_last, dim=-1, descending=False)  # (..., ..., n)
        if take_next:
            indices = torch.stack(
                [indices[..., k], indices[..., kp1]], dim=-1
            )  # (..., ..., 2)
            soft_index = F.one_hot(indices, num_classes=n)  # (..., ..., 2, [n])
            soft_index = (1.0 - a_b) * soft_index[..., 0, :] + a_b * soft_index[
                ..., 1, :
            ]  # (..., ..., [n])
        else:
            indices = indices[..., k]  # (..., ...)
            soft_index = F.one_hot(indices, num_classes=n)  # (..., ..., [n])
        soft_index = soft_index.to(x.dtype)
    else:
        if method == "softsort":
            x_sorted = torch.sort(
                x_last, dim=-1, descending=False
            ).values  # (..., ..., n)
            if take_next:
                anchors = torch.stack(
                    [x_sorted[..., k], x_sorted[..., kp1]], dim=-1
                )  # (..., ..., 2)
                abs_diff = torch.abs(
                    anchors[..., :, None] - x_last[..., None, :]
                )  # (..., ..., 2, n)
                proj = _proj_simplex(
                    -abs_diff, dim=-1, softness=softness, mode=mode
                )  # (..., ..., 2, [n])
                idx_k = proj[..., 0, :]  # (..., ..., [n])
                idx_kp1 = proj[..., 1, :]  # (..., ..., [n])
                soft_index = (1.0 - a_b) * idx_k + a_b * idx_kp1  # (..., ..., [n])
            else:
                anchors = x_sorted[..., k, None]  # (..., ..., 1)
                abs_diff = torch.abs(
                    anchors[..., :, None] - x_last[..., None, :]
                )  # (..., ..., 1, n)
                soft_index = _proj_simplex(
                    -abs_diff, dim=-1, softness=softness, mode=mode
                )[..., 0, :]  # (..., ..., [n])
        elif method == "neuralsort":
            A = abs(
                x_last[..., :, None] - x_last[..., None, :],
                mode=mode,
                softness=softness,
            )  # (..., n, n)
            A_sum = torch.sum(A, dim=-1)  # (..., n)
            if take_next:
                i = torch.stack([k + 1, k + 2])  # (2,)
                coef = n + 1 - 2 * i  # (2,)
                coef = torch.broadcast_to(coef, (*batch_dims, 2))  # (..., 2)
                z = -(
                    coef[..., :, None] * x_last[..., None, :] + A_sum[..., None, :]
                )  # (..., 2, n)
                proj = _proj_simplex(
                    z, dim=-1, softness=softness, mode=mode
                )  # (..., ..., 2, [n])
                idx_k = proj[..., 0, :]  # (..., ..., [n])
                idx_k1 = proj[..., 1, :]  # (..., ..., [n])
                soft_index = (1.0 - a_b) * idx_k + a_b * idx_k1  # (..., ..., [n])
            else:
                coef = (n + 1 - 2 * (k + 1)).unsqueeze(0)  # (1,)
                coef = torch.broadcast_to(coef, (*batch_dims, 1))  # (..., 1)
                z = -(
                    coef[..., :, None] * x_last[..., None, :] + A_sum[..., None, :]
                )  # (..., 1, n)
                soft_index = _proj_simplex(z, dim=-1, softness=softness, mode=mode)[
                    ..., 0, :
                ]  # (..., ..., [n])
        elif method == "ot":
            if take_next:
                mu = torch.ones((n,), dtype=x.dtype, device=x.device) / n  # ([n],)
                nu = torch.stack(
                    [k / n, torch.ones_like(k) / n, (kp1 - k) / n, (n - kp1 - 1) / n],
                ).to(dtype=x.dtype, device=x.device)  # (4,)

                anchors = torch.tensor(
                    [0.0, 1 / 3, 2 / 3, 1.0], dtype=x.dtype, device=x.device
                )
                anchors = torch.broadcast_to(anchors, (*batch_dims, 4))  # (..., ..., 4)

                cost = (
                    x_last[..., :, None] - anchors[..., None, :]
                ) ** 2  # (..., ..., n, 4)

                if ot_kwargs is None:
                    ot_kwargs = {}
                out = _proj_transport_polytope(
                    cost=cost,
                    mu=mu,
                    nu=nu,
                    softness=softness,
                    mode=mode,
                    **ot_kwargs,
                )  # (..., ..., [n], 4)

                soft_index = torch.swapaxes(out, -2, -1)  # (..., ..., 4, [n])
                idx_k = soft_index[..., 1, :]  # (...,  ..., [n])
                idx_k1 = soft_index[..., 2, :]  # (..., ..., [n])
                soft_index = (1.0 - a_b) * idx_k + a_b * idx_k1  # (..., ..., [n])
            else:
                mu = torch.ones((n,), dtype=x.dtype, device=x.device) / n  # ([n],)
                nu = torch.stack(
                    [k / n, torch.ones_like(k) / n, (n - k - 1) / n],
                ).to(dtype=x.dtype, device=x.device)  # (3,)

                anchors = torch.tensor([0.0, 0.5, 1.0], dtype=x.dtype, device=x.device)
                anchors = torch.broadcast_to(anchors, (*batch_dims, 3))  # (..., ..., 3)

                cost = (
                    x_last[..., :, None] - anchors[..., None, :]
                ) ** 2  # (..., ..., n, 3)

                if ot_kwargs is None:
                    ot_kwargs = {}
                out = _proj_transport_polytope(
                    cost=cost,
                    mu=mu,
                    nu=nu,
                    softness=softness,
                    mode=mode,
                    **ot_kwargs,
                )  # (..., ..., [n], 3)

                soft_index = torch.swapaxes(out, -2, -1)  # (..., ..., 3, [n])
                idx_k = soft_index[..., 1, :]  # (...,  ..., [n])
                soft_index = idx_k  # (..., ..., [n]))
        elif method == "sorting_network":
            P = _argsort_via_sorting_network(
                x_last, softness, mode, descending=False, standardized=standardize
            )  # (..., ..., n, [n])
            if take_next:
                soft_index = (1.0 - a_b) * P[..., k, :] + a_b * P[..., kp1, :]
            else:
                soft_index = P[..., k, :]  # (..., ..., [n])
        else:
            raise ValueError(f"Invalid method: {method}")

    if keepdim:
        if orig_dim_is_none:
            soft_index = soft_index.reshape(*(1,) * num_dims, n)
        else:
            soft_index = torch.unsqueeze(soft_index, dim=dim)  # (..., {1}, ..., [n])

    return soft_index


def quantile(
    x: torch.Tensor,  # (..., n, ...)
    q: torch.Tensor | float,  # quantile in [0, 1]
    dim: int | None = None,
    keepdim: bool = False,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "neuralsort",
    interpolation: Literal[
        "linear", "lower", "higher", "nearest", "midpoint"
    ] = "linear",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    return_argquantile: bool = False,
    gated_grad: bool = True,
) -> torch.Tensor:  # (..., {1}, ...)
    """Performs a soft version of [torch.quantile](https://pytorch.org/docs/stable/generated/torch.quantile.html) of `x` along the specified dim.

    For methods other than `fast_soft_sort` and `sorting_network`, implemented as [`softtorch.argquantile`][] followed by [`softtorch.take_along_dim`][], see respective documentations for details.
    For `fast_soft_sort` and `sorting_network`, uses [`softtorch.sort`][] to compute soft sorted values, then retrieves the quantile as a combination of the appropriate elements depending on the interpolation method. See [`softtorch.sort`][] for method details.

    **Extra Arguments:**

    - `gated_grad`: If `False`, stops the gradient flow through the soft index. True gives gated 'SiLU-style' gradients, while False gives integrated 'Softplus-style' gradients.

    **Returns:**

    Tensor of shape (..., {1}, ...) for scalar q, or (k, ..., {1}, ...) for vector q of length k (q dimension prepended). Represents the soft q-quantile of `x` along the specified dim.
    """

    if mode == "hard":
        quantile_val = torch.quantile(
            x, q=q, dim=dim, keepdim=keepdim, interpolation=interpolation
        )  # (..., {1}, ...)
        if return_argquantile:
            soft_index = argquantile(
                x,
                q=q,
                dim=dim,
                keepdim=keepdim,
                mode="hard",
                interpolation=interpolation,
            )
            return quantile_val, soft_index
    else:
        q_t_check = torch.as_tensor(q)
        if q_t_check.ndim > 1:
            raise ValueError(
                f"q must be scalar or 1-D, got q with shape {tuple(q_t_check.shape)}"
            )
        if q_t_check.ndim == 1:
            results = [
                quantile(
                    x,
                    q=qi.item(),
                    dim=dim,
                    keepdim=keepdim,
                    softness=softness,
                    mode=mode,
                    method=method,
                    interpolation=interpolation,
                    standardize=standardize,
                    ot_kwargs=ot_kwargs,
                    return_argquantile=return_argquantile,
                    gated_grad=gated_grad,
                )
                for qi in q_t_check
            ]
            if return_argquantile:
                vals = torch.stack([r[0] for r in results], dim=0)
                idxs = torch.stack([r[1] for r in results], dim=0)
                return vals, idxs
            return torch.stack(results, dim=0)

        x = _ensure_float(x)
        if dim is None:
            num_dims = x.ndim
            x = torch.flatten(x)
            _dim = 0
        else:
            _dim = _canonicalize_dim(dim, x.ndim)

        if method in ("fast_soft_sort", "smooth_sort", "sorting_network"):
            if return_argquantile:
                raise ValueError(
                    f"return_argquantile=True is not supported with method='{method}' "
                    "because it only produces sorted values, not soft indices"
                )
            soft_sorted = sort(
                x,
                dim=_dim,
                descending=False,
                softness=softness,
                standardize=standardize,
                mode=mode,
                method=method,
            )
            n = x.shape[_dim]
            q_t = torch.tensor(q, dtype=x.dtype, device=x.device)
            q_t = torch.clamp(q_t, 0.0, 1.0)
            k, a, take_next = _quantile_k_a(q_t, n, interpolation)
            quantile_val = soft_sorted.values.select(_dim, int(k))
            if take_next:
                kp1 = builtins.min(int(k) + 1, n - 1)
                next_val = soft_sorted.values.select(_dim, kp1)
                quantile_val = (1.0 - a) * quantile_val + a * next_val
            if dim is None:
                if keepdim:
                    quantile_val = quantile_val.reshape(*(1,) * num_dims)
            elif keepdim:
                quantile_val = quantile_val.unsqueeze(_dim)
        else:
            soft_index = argquantile(
                x,
                q=q,
                dim=_dim,
                keepdim=True,
                softness=softness,
                mode=mode,
                method=method,
                interpolation=interpolation,
                standardize=standardize,
                ot_kwargs=ot_kwargs,
            )  # (..., 1, ..., [n])
            if not gated_grad:
                soft_index = soft_index.detach()
            quantile_val = take_along_dim(x, soft_index, dim=_dim)  # (..., 1, ...)
            if not keepdim:
                soft_index = torch.squeeze(soft_index, dim=_dim)  # (..., ..., [n])
                quantile_val = torch.squeeze(quantile_val, dim=_dim)  # (..., ...)
            else:
                if dim is None:
                    quantile_val = quantile_val.reshape(
                        *(1,) * num_dims
                    )  # (1..., 1, 1...)
            if return_argquantile:
                return quantile_val, soft_index
    return quantile_val


def argmedian(
    x: torch.Tensor,  # (..., n, ...)
    dim: int | None = None,
    keepdim: bool = False,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Computes the soft argmedian of `x` along the specified dim.
    Implemented as [`softtorch.argquantile`][] with q=0.5, see respective documentation for details.
    """
    return argquantile(
        x,
        q=0.5,
        dim=dim,
        keepdim=keepdim,
        softness=softness,
        mode=mode,
        method=method,
        interpolation="lower",  # same as torch.median
        standardize=standardize,
        ot_kwargs=ot_kwargs,
    )


def median(
    x: torch.Tensor,  # (..., n, ...)
    dim: int | None = None,
    keepdim: bool = False,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> torch.Tensor | torch.return_types.median[torch.Tensor, SoftIndex]:
    """Performs a soft version of [torch.median](https://pytorch.org/docs/stable/generated/torch.median.html) of `x` along the specified dim.
    Implemented as [`softtorch.quantile`][] with q=0.5, see respective documentation for details.
    """
    if dim is None:
        if keepdim:
            raise ValueError("keepdim=True is not supported when dim=None")
    if method in ("fast_soft_sort", "smooth_sort", "sorting_network") and dim is not None:
        raise ValueError(
            f"method='{method}' is not supported for median with dim != None "
            "because it only produces sorted values, not soft indices needed "
            "for the (values, indices) return type. Use method='neuralsort' or "
            "'softsort' instead, or pass dim=None to get a scalar result."
        )
    if mode == "hard":
        if dim is None:
            return torch.median(x)  # ({1},)
        else:
            out = torch.median(x, dim=dim, keepdim=keepdim)
            median_val = out.values  # (..., {1}, ...)
            indices = out.indices  # (..., {1}, ...)
            soft_index = F.one_hot(indices, num_classes=x.shape[dim]).to(
                x.dtype
            )  # (..., {1}, ..., [n])
            ret = torch.return_types.median(
                (median_val, soft_index)
            )  # (..., {1}, ...), (..., {1}, ..., [n])
            return ret
    else:
        if method in ("fast_soft_sort", "smooth_sort", "sorting_network"):
            # dim=None path: return scalar value only (no indices needed)
            quantile_val = quantile(
                x,
                q=0.5,
                dim=dim,
                keepdim=keepdim,
                softness=softness,
                mode=mode,
                method=method,
                interpolation="lower",
                standardize=standardize,
                ot_kwargs=ot_kwargs,
                return_argquantile=False,
                gated_grad=gated_grad,
            )
            return quantile_val
        quantile_val, soft_index = quantile(
            x,
            q=0.5,
            dim=dim,
            keepdim=keepdim,
            softness=softness,
            mode=mode,
            method=method,
            interpolation="lower",  # same as torch.median
            standardize=standardize,
            ot_kwargs=ot_kwargs,
            return_argquantile=True,
            gated_grad=gated_grad,
        )  # (..., ...), (..., ..., [n])
        if dim is None:
            return quantile_val  # ({1},)
        else:
            ret = torch.return_types.median(
                (quantile_val, soft_index)
            )  # (..., {1}, ...), (..., {1}, ..., [n])
            return ret


def _argtopk(
    x: torch.Tensor,  # (..., n, ...)
    k: int,
    dim: int | None = None,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> SoftIndex:  # (..., k, ...), (..., k, ..., [n])
    """Performs a soft version of argtopk of `x` along the specified dim.

    **Arguments:**

    - `x`: Input Tensor of shape (..., n, ...).
    - `k`: The number of top elements to select.
    - `dim`: The dim along which to compute the topk operation. If dim is None, the last dimension of the input is chosen.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns the result of torch.topk with a one-hot encoding of the indices.
        - `smooth`: Soften projections via an entropic regularizer.
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via a softmax operation.
            - For optimal transport (`ot` method), transport plan is computed via Sinkhorn iterations (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)).
        - `c0`: Soften projections via a quadratic regularizer.
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
            - For optimal transport (`ot` method), transport plan is computed via LBFGS (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`/`c2`: C1 differentiable / C2 twice differentiable. Similar to `c0`, but using p-norm regularizers with p=3/2 and p=4/3, respectively.
    - `method`: Method to compute the soft argtopk. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Uses the approach in [Differentiable Top-k with Optimal Transport](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf).
            Intuition: The top-k elements are selected by specifying k+1 "anchors" and then transporting the top_k values to the top k anchors, and the remaining (n-k) values to the last anchor.
            Note: `sinkhorn_max_iter` and `sinkhorn_tol` can be passed via `ot_kwargs` to control convergence.
        - `softsort`: Computes the top-k elements of the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            Note: Can introduce gradient discontinuities when elements in `x` are not unique, but is much faster than OT-based method.
        - `neuralsort`: Computes the top-k elements of the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850).
    - `standardize`: If True, standardizes and squashes the input `x` along the specified dim before applying the softtopk operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.

    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.

    **Returns:**

    SoftIndex of shape (..., k, ..., [n]) (positive Tensor which sums to 1 over the last dimension). Represents the soft indices of the top-k values.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")
    if dim is None:
        dim = -1
    dim = _canonicalize_dim(dim, x.ndim)
    if k > x.shape[dim]:
        raise ValueError(f"k={k} exceeds dimension size {x.shape[dim]} along dim={dim}")

    if mode == "hard" or mode == "_hard":
        out = torch.topk(x, k=k, dim=dim)
        indices = out.indices  # (..., k, ...)
        soft_index = F.one_hot(indices, num_classes=x.shape[dim]).to(
            x.dtype
        )  # (..., k, ..., [n])
    else:
        x = _ensure_float(x)
        x_last = torch.movedim(x, dim, -1)  # (..., ..., n)
        if standardize:
            x_last = _standardize_and_squash(x_last, dim=-1)
        *batch_dims, n = x_last.shape
        if method == "softsort":
            anchors = torch.topk(x_last, k=k, dim=-1).values  # (..., ..., k)
            abs_diff = torch.abs(
                anchors[..., :, None] - x_last[..., None, :]
            )  # (..., ..., k, n)
            soft_index = _proj_simplex(
                -abs_diff, dim=-1, softness=softness, mode=mode
            )  # (..., ..., k, [n])
        elif method == "neuralsort":
            A = abs(
                x_last[..., :, None] - x_last[..., None, :],
                mode=mode,
                softness=softness,
            )  # (..., ..., n, n)
            A_sum = torch.sum(A, dim=-1)  # (..., ..., n)

            i = torch.arange(n - k + 1, n + 1, dtype=x.dtype, device=x.device)  # (k,)
            i = torch.flip(i, dims=(0,))  # (k,)

            coef = n + 1 - 2 * i  # (k,)
            coef = coef.expand(*batch_dims, k)  # (..., ..., k)

            z = -(
                coef[..., :, None] * x_last[..., None, :] + A_sum[..., None, :]
            )  # (..., ..., k, n)
            soft_index = _proj_simplex(
                z, dim=-1, softness=softness, mode=mode
            )  # (..., ..., k, [n])
        elif method == "ot":
            if k == n:
                soft_index = argsort(
                    x=x_last,
                    dim=-1,
                    descending=True,
                    softness=softness,
                    mode=mode,
                    method=method,
                    standardize=False,
                    ot_kwargs=ot_kwargs,
                )  # (..., ..., k, [n])
            else:
                anchors = (
                    torch.linspace(0, k, k + 1, dtype=x.dtype, device=x.device) / k
                )  # (k+1,)
                anchors = torch.flip(anchors, dims=(0,))  # (k+1,)
                anchors = torch.broadcast_to(
                    anchors, (*batch_dims, k + 1)
                )  # (..., ..., k+1)

                cost = (
                    x_last[..., :, None] - anchors[..., None, :]
                ) ** 2  # (..., ..., n, k+1)

                mu = torch.ones((n,), dtype=x.dtype, device=x.device) / n  # ([n],)
                nu = torch.cat(
                    [
                        torch.ones(k, dtype=x.dtype, device=x.device) / n,
                        torch.tensor((n - k) / n, dtype=x.dtype, device=x.device)[None],
                    ]
                )  # ([k+1],)

                if ot_kwargs is None:
                    ot_kwargs = {}
                out = _proj_transport_polytope(
                    cost=cost, mu=mu, nu=nu, softness=softness, mode=mode, **ot_kwargs
                )  # (..., ..., [n], k+1)
                soft_index = torch.movedim(out, -2, -1)  # (..., ..., k+1, [n])
                soft_index = soft_index[..., :k, :]  # (..., ..., k, [n])
        elif method == "sorting_network":
            P = _argsort_via_sorting_network(
                x_last, softness, mode, descending=True, standardized=standardize
            )  # (..., ..., n, [n])
            soft_index = P[..., :k, :]  # (..., ..., k, [n])
        else:
            raise ValueError(f"Invalid method: {method}")

        soft_index = torch.movedim(soft_index, -2, dim)  # (..., k, ..., [n])
    return soft_index


def topk(
    x: torch.Tensor,  # (..., n, ...)
    k: int,
    dim: int | None = None,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> torch.return_types.topk[
    torch.Tensor, SoftIndex
]:  # (..., k, ...), (..., k, ..., [n])
    """Performs a soft version of [torch.topk](https://pytorch.org/docs/stable/generated/torch.topk.html)

    **Arguments:**

    - `x`: Input Tensor of shape (..., n, ...).
    - `k`: The number of top elements to select.
    - `dim`: The dim along which to compute the topk operation. If dim is None, the last dimension of the input is chosen.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns the result of torch.topk with a one-hot encoding of the indices.
        - `smooth`: Soften projections via an entropic regularizer.
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via a softmax operation.
            - For optimal transport (`ot` method), transport plan is computed via Sinkhorn iterations (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)).
        - `c0`: Soften projections via a quadratic regularizer.
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
            - For optimal transport (`ot` method), transport plan is computed via LBFGS (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`/`c2`: C1 differentiable / C2 twice differentiable. Similar to `c0`, but using p-norm regularizers with p=3/2 and p=4/3, respectively.
    - `method`: Method to compute the soft topk. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Uses the approach in [Differentiable Top-k with Optimal Transport](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf).
            Intuition: The top-k elements are selected by specifying k+1 "anchors" and then transporting the top_k values to the top k anchors, and the remaining (n-k) values to the last anchor.
            Note: `sinkhorn_max_iter` and `sinkhorn_tol` can be passed via `ot_kwargs` to control convergence.
        - `softsort`: Computes the top-k elements of the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            Note: Can introduce gradient discontinuities when elements in `x` are not unique, but is much faster than OT-based method.
        - `neuralsort`: Computes the top-k elements of the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850).
        - `fast_soft_sort`: Uses the `FastSoftSort` operator from [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871) to directly compute the soft sorted values, via projection onto the permutahedron. The projection is solved via a PAV algorithm as proposed in [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871). The top-k values are then retrieved by taking the first k values from the soft sorted output.
            Note: This method does not return the soft indices, only the soft values. The PAV step uses Numba JIT (CPU-only); GPU tensors are transferred to CPU for the forward pass.
        - `sorting_network`: Uses a soft bitonic sorting network as proposed in [Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision](https://arxiv.org/abs/2105.04019), replacing hard compare-and-swap with soft versions. The top-k values are retrieved by taking the first k values from the soft sorted output.
            Note: This method does not return the soft indices, only the soft values.
    - `standardize`: If True, standardizes and squashes the input `x` along the specified dim before applying the softtopk operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.

    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.
    - `gated_grad`: If `False`, stops the gradient flow through the soft index. True gives gated 'SiLU-style' gradients, while False gives integrated 'Softplus-style' gradients.

    **Returns:**

    - Namedtuple containing two fields:
        - `values`: Top-k values of `x`, shape (..., k, ...).
        - `indices`: SoftIndex of shape (..., k, ..., [n]) (positive Tensor which sums to 1 over the last dimension). Represents the soft indices of the top-k values.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")
    if dim is None:
        dim = -1
    dim = _canonicalize_dim(dim, x.ndim)
    if k > x.shape[dim]:
        raise ValueError(f"k={k} exceeds dimension size {x.shape[dim]} along dim={dim}")

    if mode == "hard":
        out = torch.topk(x, k=k, dim=dim)
        values = out.values  # (..., k, ...)
        indices = out.indices  # (..., k, ...)
        soft_index = F.one_hot(indices, num_classes=x.shape[dim]).to(
            x.dtype
        )  # (..., k, ..., [n])
    elif method in ("fast_soft_sort", "smooth_sort", "sorting_network"):
        sorted_out = sort(
            x,
            dim=dim,
            descending=True,
            softness=softness,
            mode=mode,
            method=method,
            standardize=standardize,
        )
        values = torch.narrow(sorted_out.values, dim, 0, k)  # (..., k, ...)
        soft_index = None
    else:
        soft_index = _argtopk(
            x=x,
            k=k,
            dim=dim,
            softness=softness,
            mode=mode,
            method=method,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
        )  # (..., k, ..., [n])
        if not gated_grad:
            soft_index_tmp = soft_index.detach()
        else:
            soft_index_tmp = soft_index
        values = take_along_dim(x, soft_index_tmp, dim=dim)  # (..., k, ...)
    return torch.return_types.topk((values, soft_index))


def rank(
    x: torch.Tensor,  # (..., n, ...)
    dim: int | None = None,
    softness: float = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"] = "neuralsort",
    descending: bool = True,
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> torch.Tensor:  # (..., n, ...)
    """Computes the soft rankings of `x` along the specified dim.

    **Arguments:**

    - `x`: Input Tensor of shape (..., n, ...).
    - `dim`: The dim along which to compute the ranking operation. If None, the input
        Tensor is flattened before computing the ranking.
    - `descending`: If True, larger inputs receive smaller ranks (best rank is 0). If
        False, ranks increase with the input values.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns rank computed as two torch.argsort calls.
        - `smooth`: Soften projections via an entropic regularizer.
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via a softmax operation.
            - For optimal transport (`ot` method), transport plan is computed via Sinkhorn iterations (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)).
        - `c0`: Soften projections via a quadratic regularizer.
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
            - For optimal transport (`ot` method), transport plan is computed via LBFGS (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`/`c2`: C1 differentiable / C2 twice differentiable. Similar to `c0`, but using p-norm regularizers with p=3/2 and p=4/3, respectively.
    - `method`: Method to compute the soft rank. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Uses the approach in [Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885).
            Intuition: Run an OT procedure as in [`softtorch.argsort`][], then transport the sorted ranks (0, 1, ..., n-1) back to the ranks of the original values by using the transpose of the transport plan.
            Note: `sinkhorn_max_iter` and `sinkhorn_tol` can be passed via `ot_kwargs` to control convergence.
        - `softsort`: Adapts the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038) for rank by using a single **column**-wise projection instead of row-wise projection.
            Note: Can introduce gradient discontinuities when elements in `x` are not unique, but is much faster than OT-based method.
        - `neuralsort`: Adapts the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850) for rank by renormalizing over columns after the row-wise projection.
        - `fast_soft_sort`: Uses the `FastSoftSort` operator from [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871) to directly compute the soft ranks, via projection onto the permutahedron. The projection is solved via a PAV algorithm as proposed in [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871).
            Note: The PAV step uses Numba JIT (CPU-only); GPU tensors are transferred to CPU for the forward pass.
    - `standardize`: If True, standardizes and squashes the input `x` along the specified dim before applying the softrank operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.

    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.

    **Returns:**

    A positive Tensor of shape (..., n, ...) with values in [1, n].
    The elements in (..., i, ...) represent the soft rank of the ith element along the specified dim.
    """
    if dim is None:
        x = torch.flatten(x)
        dim = 0
    else:
        dim = _canonicalize_dim(dim, x.ndim)

    x = _ensure_float(x)
    if mode == "hard" or mode == "_hard":
        indices = torch.argsort(x, dim=dim, descending=descending)  # (..., n, ...)
        indices = indices.to(x.dtype)
        ranks = torch.argsort(indices, dim=dim, descending=False) + 1  # (..., n, ...)
        ranks = ranks.to(x.dtype)
    else:
        x_last = torch.movedim(x, dim, -1)  # (..., ..., n)
        *batch_dims, n = x_last.shape

        if standardize:
            x_last = _standardize_and_squash(x_last, dim=-1)

        if method == "fast_soft_sort":
            w = -x_last if descending else x_last
            scale = n - 1 if n > 1 else 1
            anchors = torch.arange(1, n + 1, dtype=x.dtype, device=x.device) / scale
            anchors = anchors.expand(*batch_dims, n)
            ranks = _proj_permutahedron(w, anchors, softness=softness, mode=mode)
            ranks = ranks * scale
            ranks = torch.movedim(ranks, -1, dim)
        elif method == "smooth_sort":
            raise NotImplementedError(
                "smooth_sort is not supported in SoftTorch. "
                "Use method='fast_soft_sort' with mode='smooth' instead, or use SoftJAX."
            )
        else:
            if method == "softsort":
                anchors = torch.sort(
                    x_last, dim=-1, descending=descending
                ).values  # (..., ..., n)
                abs_diff = torch.abs(
                    x_last[..., :, None] - anchors[..., None, :]
                )  # (..., ..., n, n)
                soft_index = _proj_simplex(
                    -abs_diff, dim=-1, softness=softness, mode=mode
                )  # (..., ..., n, [n])
            elif method == "neuralsort":
                A = abs(
                    x_last[..., :, None] - x_last[..., None, :],
                    mode=mode,
                    softness=softness,
                )  # (..., ..., n, n)
                A_sum = torch.sum(A, dim=-1)  # (..., ..., n)

                i = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)  # (n,)
                if descending:
                    i = torch.flip(i, dims=(0,))  # (n,)

                coef = n + 1 - 2 * i  # (n,)
                coef = coef.expand(*batch_dims, n)  # (..., ..., n)

                z = -(
                    coef[..., None, :] * x_last[..., :, None] + A_sum[..., :, None]
                )  # (..., ..., n, n)
                soft_index = _proj_simplex(
                    z, dim=-2, softness=softness, mode=mode
                )  # (..., ..., [n], n)
                # additional normalization compared to argsort version
                soft_index = soft_index / torch.clamp(
                    torch.sum(soft_index, dim=-1, keepdim=True), min=1e-10
                )  # (..., ..., n, [n])
            elif method == "ot":
                anchors = (
                    torch.linspace(0, n, n, dtype=x.dtype, device=x.device) / n
                )  # (n,)
                if descending:
                    anchors = torch.flip(anchors, dims=(0,))  # (n,)
                anchors = anchors.broadcast_to(*batch_dims, n)  # (..., ..., n)

                cost = (
                    x_last[..., :, None] - anchors[..., None, :]
                ) ** 2.0  # (..., ..., n, n)

                mu = torch.ones((n,), dtype=x.dtype, device=x.device) / n  # ([n],)
                nu = torch.ones((n,), dtype=x.dtype, device=x.device) / n  # ([n],)

                if ot_kwargs is None:
                    ot_kwargs = {}
                out = _proj_transport_polytope(
                    cost=cost, mu=mu, nu=nu, softness=softness, mode=mode, **ot_kwargs
                )  # (..., ..., [n], n)
                soft_index = out / torch.clamp(
                    torch.sum(out, dim=-1, keepdim=True), min=1e-10
                )  # (..., ..., n, [n])
            elif method == "sorting_network":
                P = _argsort_via_sorting_network(
                    x_last, softness, mode, descending=descending, standardized=standardize
                )  # (..., ..., n, [n])
                # Transpose: P[sorted_pos, elem] → soft_index[elem, sorted_pos]
                soft_index = torch.swapaxes(P, -2, -1)  # (..., ..., n, [n])
                soft_index = soft_index / torch.clamp(
                    torch.sum(soft_index, dim=-1, keepdim=True), min=1e-10
                )  # (..., ..., n, [n])
            else:
                raise ValueError(f"Invalid method: {method}")

            nums = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)  # (n,)
            nums = nums.broadcast_to(*batch_dims, n)  # (..., ..., n)
            nums = torch.movedim(nums, -1, dim)  # (..., n, ...)
            soft_rank_indices = torch.movedim(soft_index, -2, dim)  # (..., n, ..., [n])
            ranks = take_along_dim(nums, soft_rank_indices, dim=dim)  # (..., n, ...)
    return ranks


# Elementwise operators


def sigmoidal(
    x: torch.Tensor,
    softness: float = 0.1,
    mode: Literal["smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"] = "smooth",
) -> SoftBool:
    """Sigmoidal functions defining a characteristic S-shaped curve.

    **Arguments:**

    - `x`: Input Tensor.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Choice of smoothing family for the surrogate step.
        - `smooth`: Smooth sigmoidal based on the logistic function.
        - `c0`: Continuous sigmoidal based on a piecewise quadratic polynomial.
        - `c1`: Differentiable sigmoidal based on a piecewise cubic polynomial.
        - `c2`: Twice differentiable sigmoidal based on a piecewise quintic polynomial.

    **Returns:**

    SoftBool of same shape as `x` (Tensor with values in [0, 1]).
    """
    _validate_softness(softness)
    x = x / softness
    if mode == "smooth":  # smooth
        # closed form of argmax([0,x], softness=softness, mode="smooth", standardize=False)[1]
        y = torch.sigmoid(x)
    else:
        # Piecewise modes are defined on [-1, 1]. Scale by 1/5 so the effective
        # transition region [-5s, 5s] matches smooth's range (sigmoid(±5) ≈ 0/1).
        x = x / 5.0
        if mode == "c0":  # continuous
            # closed form of argmax([0,x], softness=5*softness, mode="c0", standardize=False)[1]
            # The leading zero is a no-op term that keeps the graph alive for
            # higher-order derivatives.
            y = _polyval(
                torch.tensor([0.0, 0.5, 0.5], device=x.device, dtype=x.dtype), x
            )
            y = torch.where(
                x < -1.0,
                torch.zeros_like(x),
                torch.where(x < 1.0, y, torch.ones_like(x)),
            )
        elif mode == "c1":  # differentiable
            # C1 Hermite smoothstep (f=0/1 and f'=0 at boundaries)
            y = _polyval(
                torch.tensor(
                    [-0.25, 0.0, 0.75, 0.5], device=x.device, dtype=x.dtype
                ),
                x,
            )
            y = torch.where(
                x < -1.0,
                torch.zeros_like(x),
                torch.where(x < 1.0, y, torch.ones_like(x)),
            )
        elif mode == "_c1_pnorm":  # differentiable, p-norm based
            # closed form of argmax([0,x], softness=5*softness, mode="c1", standardize=False)[1]
            # y = (x + sqrt(2 - x^2))^2 / 4 on [-1, 1]
            y = (
                torch.square(
                    x + torch.sqrt(torch.clamp(2.0 - torch.square(x), min=0.0))
                )
                / 4.0
            )
            y = torch.where(
                x < -1.0,
                torch.zeros_like(x),
                torch.where(x < 1.0, y, torch.ones_like(x)),
            )
        elif mode == "c2":  # twice differentiable
            # C2 Hermite smoothstep (f=0/1, f'=0, and f''=0 at boundaries)
            y = _polyval(
                torch.tensor(
                    [0.1875, 0.0, -0.625, 0.0, 0.9375, 0.5],
                    device=x.device,
                    dtype=x.dtype,
                ),
                x,
            )
            y = torch.where(
                x < -1.0,
                torch.zeros_like(x),
                torch.where(x < 1.0, y, torch.ones_like(x)),
            )
        elif mode == "_c2_pnorm":  # twice differentiable, p-norm based
            # closed form of argmax([0,x], softness=5*softness, mode="c2", standardize=False)[1]
            # depressed cubic: t^3 + A*t + B = 0, where A = 3d^2/4, B = -1/2, d = x
            d = x
            p_coeff = -1.5 * d
            A = 0.75 * d**2
            B = torch.full_like(x, -0.5)
            A_safe = torch.where(A > 1e-12, A, torch.ones_like(A))
            sA3 = torch.sqrt(torch.clamp(A_safe / 3.0, min=0.0))
            arg = 3.0 * B / (2.0 * A_safe * sA3)
            t_hyp = -2.0 * sA3 * torch.sinh(torch.arcsinh(arg) / 3.0)
            t_cbrt = -torch.sign(B) * torch.abs(B) ** (1.0 / 3.0)
            t = torch.where(A > 1e-12, t_hyp, t_cbrt)
            b_val = t - p_coeff / 3.0
            y = b_val**3
            y = torch.where(
                x < -1.0,
                torch.zeros_like(x),
                torch.where(x < 1.0, y, torch.ones_like(x)),
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
    return y


def softrelu(
    x: torch.Tensor,
    softness: float = 0.1,
    mode: Literal["smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"] = "smooth",
    gated: bool = False,
) -> torch.Tensor:
    """Family of soft relaxations to ReLU.

    **Arguments:**

    - `x`: Input Tensor.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Choice of [`softtorch.sigmoidal`][] smoothing mode, options are "smooth", "c0", "c1", "_c1_pnorm", "c2", or "_c2_pnorm".
    - `gated`: If True, uses the 'gated' version `x * sigmoidal(x)`. If False, uses the integral of the sigmoidal.

    **Returns:**

    Result of applying soft elementwise ReLU to `x`.
    """
    _validate_softness(softness)
    x = x / softness
    if mode == "smooth":
        if gated:
            # x * sigmoidal(x) = max([0,x], softness=softness, mode="smooth", standardize=False)
            y = x * sigmoidal(x, softness=1.0, mode="smooth")
        else:
            # closed form integral of sigmoidal(x, mode="smooth")
            y = F.softplus(x)
    else:
        # Piecewise modes: scale by 1/5 to match smooth's transition width.
        # Non-gated: softrelu = 5*G(u) where G is the antiderivative on [-1,1].
        # Gated: softrelu = x*sigmoidal(x), sigmoidal handles the 1/5 internally.
        u = x / 5.0
        if mode == "c0":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="c0")
            else:
                y = 5.0 * _polyval(
                    torch.tensor([0.25, 0.5, 0.25], device=x.device, dtype=x.dtype),
                    u,
                )
                y = torch.where(
                    u < -1.0,
                    torch.zeros_like(x),
                    torch.where(u < 1.0, y, x),
                )
        elif mode == "c1":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="c1")
            else:
                y = 5.0 * _polyval(
                    torch.tensor(
                        [-0.0625, 0.0, 0.375, 0.5, 0.1875],
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    u,
                )
                y = torch.where(
                    u < -1.0,
                    torch.zeros_like(x),
                    torch.where(u < 1.0, y, x),
                )
        elif mode == "_c1_pnorm":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="_c1_pnorm")
            else:
                # F(u) = u/2 - (2 - u^2)^(3/2) / 6 + 2/3
                inside = 2.0 - u**2
                y = 5.0 * (
                    u / 2.0
                    - inside * torch.sqrt(torch.clamp(inside, min=0.0)) / 6.0
                    + 2.0 / 3.0
                )
                y = torch.where(
                    u < -1.0,
                    torch.zeros_like(x),
                    torch.where(u < 1.0, y, x),
                )
        elif mode == "c2":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="c2")
            else:
                y = 5.0 * _polyval(
                    torch.tensor(
                        [0.03125, 0.0, -0.15625, 0.0, 0.46875, 0.5, 0.15625],
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    u,
                )
                y = torch.where(
                    u < -1.0,
                    torch.zeros_like(x),
                    torch.where(u < 1.0, y, x),
                )
        elif mode == "_c2_pnorm":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="_c2_pnorm")
            else:
                # F(u) = 0.75 + u*g(u) - 0.75*((1-g(u))^(4/3) + g(u)^(4/3))
                y2 = sigmoidal(x, softness=1.0, mode="_c2_pnorm")
                safe_y2 = torch.clamp(y2, 0.0, 1.0)
                y = 5.0 * (
                    0.75
                    + u * safe_y2
                    - 0.75
                    * ((1.0 - safe_y2) ** (4.0 / 3.0) + safe_y2 ** (4.0 / 3.0))
                )
                y = torch.where(
                    u < -1.0,
                    torch.zeros_like(x),
                    torch.where(u < 1.0, y, x),
                )
        else:
            raise ValueError(f"Unknown mode '{mode}' for softrelu.")
    y = y * softness
    return y


def heaviside(
    x: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
) -> SoftBool:
    """Performs a soft version of [torch.heaviside](https://pytorch.org/docs/stable/generated/torch.heaviside.html)(x,0.5).

    **Arguments:**

    - `x`: Input Tensor of any shape.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact Heaviside step. Otherwise uses [`softtorch.sigmoidal`][]-based "smooth", "c0", "c1", "c2" relaxations.

    **Returns:**

    SoftBool of same shape as `x` (Tensor with values in [0, 1]), relaxing the elementwise Heaviside step function.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.heaviside(x, torch.tensor(0.5, device=x.device, dtype=x.dtype))
    else:
        return sigmoidal(x, softness=softness, mode=mode)


def round(
    x: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    neighbor_radius: int = 5,
) -> torch.Tensor:
    """Performs a soft version of [torch.round](https://pytorch.org/docs/stable/generated/torch.round.html).

    **Arguments:**

    - `x`: Input Tensor of any shape.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", applies `torch.round`. Otherwise uses a sigmoidal-based relaxation based on the algorithm described in [Smooth Approximations of the Rounding Function](https://arxiv.org/pdf/2504.19026v1).
        Supports the [`softtorch.sigmoidal`][] modes "smooth", "c0", "c1", "c2".
    - `neighbor_radius`: Number of neighbors on each side of the floor value to consider for the soft rounding.

    **Returns:**

    Result of applying soft elementwise rounding to `x`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.round(x)
    else:
        center = (torch.floor(x)).detach()
        offsets = torch.arange(
            -neighbor_radius, neighbor_radius + 1, dtype=x.dtype, device=x.device
        )  # (M,)
        n = center[..., None] + offsets  # (..., M)

        w_left = sigmoidal(x[..., None] - (n - 0.5), softness=softness, mode=mode)
        w_right = sigmoidal(x[..., None] - (n + 0.5), softness=softness, mode=mode)
        return torch.sum(n * (w_left - w_right), dim=-1)


def sign(
    x: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
) -> torch.Tensor:
    """Performs a soft version of [torch.sign](https://pytorch.org/docs/stable/generated/torch.sign.html).

    **Arguments:**

    - `x`: Input Tensor of any shape.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns `torch.sign`. Otherwise uses [`softtorch.sigmoidal`][]-based "smooth", "c0", "c1", "c2" relaxations.

    **Returns:**

    Result of applying soft elementwise sign to `x`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.sign(x).to(x.dtype)
    else:
        return sigmoidal(x, mode=mode, softness=softness) * 2.0 - 1.0


def abs(
    x: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
) -> torch.Tensor:
    """Performs a soft version of [torch.abs](https://pytorch.org/docs/stable/generated/torch.abs.html).

    **Arguments:**

    - `x`: Input Tensor of any shape.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Projection mode. "hard" returns the exact absolute value. Otherwise uses [`softtorch.sigmoidal`][]-based "smooth", "c0", "c1", "c2" relaxations.

    **Returns:**

    Result of applying soft elementwise absolute value to `x`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.abs(x)
    else:
        return x * sign(x, softness=softness, mode=mode)


def relu(
    x: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    gated: bool = False,
) -> torch.Tensor:
    """Performs a soft version of [torch.relu](https://pytorch.org/docs/stable/generated/torch.relu.html).

    **Arguments:**

    - `x`: Input Tensor of any shape.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", applies `torch.relu`. Otherwise uses [`softtorch.softrelu`][] with "smooth", "c0", "c1", "c2" relaxations.
    - `gated`: See [`softtorch.softrelu`][] documentation.

    **Returns:**

    Result of applying soft elementwise ReLU to `x`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.relu(x)
    else:
        return softrelu(x, mode=mode, softness=softness, gated=gated)


def clamp(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    gated: bool = False,
) -> torch.Tensor:
    """Performs a soft version of [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html).

    **Arguments:**

    - `x`: Input Tensor of any shape.
    - `a`: Lower bound scalar.
    - `b`: Upper bound scalar.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", applies `torch.clamp`. Otherwise uses [`softtorch.softrelu`][]-based "smooth", "c0", "c1", "c2" relaxations.
    - `gated`: See [`softtorch.softrelu`][] documentation.

    **Returns:**

    Result of applying soft elementwise clamping to `x`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.clamp(x, a, b)
    else:
        tmp1 = softrelu(x - a, mode=mode, softness=softness, gated=gated)
        tmp2 = softrelu(x - b, mode=mode, softness=softness, gated=gated)
        return a + tmp1 - tmp2


# Comparison operators


def greater(
    x: torch.Tensor,
    y: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x > y`.
    Uses a Heaviside relaxation so the output approaches 0 at equality.

    **Arguments:**

    - `x`: First input Tensor.
    - `y`: Second input Tensor, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, greater returns 0 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Tensor with values in [0, 1]), relaxing the elementwise `x > y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.gt(x, y).to(x.dtype)
    else:
        return sigmoidal(x - y - epsilon, softness=softness, mode=mode)


def greater_equal(
    x: torch.Tensor,
    y: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x >= y`.
    Uses a Heaviside relaxation so the output approaches 1 at equality.

    **Arguments:**

    - `x`: First input Tensor.
    - `y`: Second input Tensor, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, greater_equal returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Tensor with values in [0, 1]), relaxing the elementwise `x >= y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.ge(x, y).to(x.dtype)
    else:
        return sigmoidal(x - y + epsilon, softness=softness, mode=mode)


def less(
    x: torch.Tensor,
    y: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x < y`.
    Uses a Heaviside relaxation so the output approaches 0 at equality.

    **Arguments:**

    - `x`: First input Tensor.
    - `y`: Second input Tensor, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside ("smooth", "c0", "c1", or "c2"). Defaults to "smooth".
    - `epsilon`: Small offset so that as softness->0, less returns 0 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Tensor with values in [0, 1]), relaxing the elementwise `x < y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.lt(x, y).to(x.dtype)
    else:
        return logical_not(
            greater_equal(x, y, softness=softness, mode=mode, epsilon=epsilon)
        )


def less_equal(
    x: torch.Tensor,
    y: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x <= y`.
    Uses a Heaviside relaxation so the output approaches 1 at equality.

    **Arguments:**

    - `x`: First input Tensor.
    - `y`: Second input Tensor, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, less_equal returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Tensor with values in [0, 1]), relaxing the elementwise `x <= y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.le(x, y).to(x.dtype)
    else:
        return logical_not(greater(x, y, softness=softness, mode=mode, epsilon=epsilon))


def eq(
    x: torch.Tensor,
    y: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x == y`.
    Implemented as a soft `abs(x - y) <= 0` comparison.

    **Arguments:**

    - `x`: First input Tensor.
    - `y`: Second input Tensor, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, eq returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Tensor with values in [0, 1]), relaxing the elementwise `x == y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.eq(x, y).to(x.dtype)
    else:
        diff = abs(x - y, softness=softness, mode=mode)
        # diff >= 0, so less_equal(diff, 0) (a sigmoid on a non-positive input)
        # is in [0, 0.5], and the 2x scaling keeps the result in [0, 1].
        return 2.0 * less_equal(
            diff, torch.zeros_like(diff), mode=mode, softness=softness, epsilon=epsilon
        )


def not_equal(
    x: torch.Tensor,
    y: torch.Tensor,
    softness: float = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x != y`.
    Implemented as a soft `abs(x - y) > 0` comparison.

    **Arguments:**

    - `x`: First input Tensor.
    - `y`: Second input Tensor, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, not_equal returns 0 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Tensor with values in [0, 1]), relaxing the elementwise `x != y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.ne(x, y).to(x.dtype)
    else:
        diff = abs(x - y, softness=softness, mode=mode)
        # diff >= 0, so greater(diff, 0) (a sigmoid on a non-negative input)
        # is in [0.5, 1], and the 2x-1 scaling keeps the result in [0, 1].
        tmp = greater(
            diff, torch.zeros_like(diff), mode=mode, softness=softness, epsilon=epsilon
        )
        return 2.0 * tmp - 1.0


def isclose(
    x: torch.Tensor,
    y: torch.Tensor,
    softness: float = 0.1,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to `torch.isclose` for elementwise comparison.
    Implemented as a soft `abs(x - y) <= atol + rtol * abs(y)` comparison.

    **Arguments:**

    - `x`: First input Tensor.
    - `y`: Second input Tensor, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `rtol`: Relative tolerance. Defaults to 1e-5.
    - `atol`: Absolute tolerance. Defaults to 1e-8.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, isclose returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Tensor with values in [0, 1]), relaxing the elementwise `isclose(x, y)`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return torch.isclose(x, y, atol=atol, rtol=rtol).to(x.dtype)
    else:
        diff = abs(x - y, softness=softness, mode=mode)
        y_abs = abs(y, softness=softness, mode=mode)
        # diff >= 0 and atol + rtol * y_abs >= 0, so less_equal(diff, tol)
        # is in [0, 0.5] when diff > tol, and the 2x scaling keeps the result in [0, 1].
        return 2.0 * less_equal(
            diff,
            atol + rtol * y_abs,
            mode=mode,
            softness=softness,
            epsilon=epsilon,
        )


# Logical operators


def logical_not(x: SoftBool) -> SoftBool:
    """Computes soft elementwise logical NOT of a SoftBool Tensor.
    Fuzzy logic implemented as `1.0 - x`.

    **Arguments:**
    - `x`: SoftBool input Tensor.

    **Returns:**

    SoftBool of same shape as `x` (Tensor with values in [0, 1]), relaxing the elementwise logical NOT.
    """
    return 1.0 - x


def all(
    x: SoftBool, dim: int = -1, epsilon: float = 1e-10, use_geometric_mean: bool = False
) -> SoftBool:
    """Computes soft elementwise logical AND across a specified dim.
    Fuzzy logic implemented as the geometric mean along the dim.

    **Arguments:**
    - `x`: SoftBool input Tensor.
    - `dim`: Axis along which to compute the logical AND. Default is -1 (last dim).
    - `epsilon`: Minimum value for numerical stability inside the log.
    - `use_geometric_mean`: If True, uses the geometric mean to compute the soft AND. Otherwise, the product is used.

    **Returns:**

    SoftBool (Tensor with values in [0, 1]) with the specified dim reduced, relaxing
    the logical ALL along that dim.
    """
    if use_geometric_mean:
        return torch.exp(torch.mean(torch.log(torch.clamp(x, min=epsilon)), dim=dim))
    else:
        return torch.prod(x, dim=dim)


def any(x: SoftBool, dim: int = -1, use_geometric_mean: bool = False) -> SoftBool:
    """Computes soft elementwise logical OR across a specified dim.
    Fuzzy logic implemented as `1.0 - all(logical_not(x), dim=dim)`.

    **Arguments:**
    - `x`: SoftBool input Tensor.
    - `dim`: Axis along which to compute the logical OR. Default is -1 (last dim).
    - `use_geometric_mean`: If True, uses the geometric mean to compute the soft AND inside the logical NOT. Otherwise, the product is used.

    **Returns:**

    SoftBool (Tensor with values in [0, 1]) with the specified dim reduced, relaxing
    the logical ANY along that dim.
    """
    return logical_not(
        all(logical_not(x), dim=dim, use_geometric_mean=use_geometric_mean)
    )


def logical_and(x: SoftBool, y: SoftBool, use_geometric_mean: bool = False) -> SoftBool:
    """Computes soft elementwise logical AND between two SoftBool Tensors.
    Fuzzy logic implemented as `all(stack([x, y], dim=-1), dim=-1)`.

    **Arguments:**

    - `x`: First SoftBool input Tensor.
    - `y`: Second SoftBool input Tensor.
    - `use_geometric_mean`: If True, uses the geometric mean to compute the soft AND. Otherwise, the product is used.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Tensor with values in [0, 1]), relaxing the elementwise logical AND.
    """
    return all(
        torch.stack([x, y], dim=-1), dim=-1, use_geometric_mean=use_geometric_mean
    )


def logical_or(x: SoftBool, y: SoftBool, use_geometric_mean: bool = False) -> SoftBool:
    """Computes soft elementwise logical OR between two SoftBool Tensors.
    Fuzzy logic implemented as `any(stack([x, y], dim=-1), dim=-1)`.

    **Arguments:**
    - `x`: First SoftBool input Tensor.
    - `y`: Second SoftBool input Tensor.
    - `use_geometric_mean`: If True, uses the geometric mean to compute the soft AND inside the logical NOT. Otherwise, the product is used.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Tensor with values in [0, 1]), relaxing the elementwise logical OR.
    """
    return any(
        torch.stack([x, y], dim=-1), dim=-1, use_geometric_mean=use_geometric_mean
    )


def logical_xor(x: SoftBool, y: SoftBool, use_geometric_mean: bool = False) -> SoftBool:
    """Computes soft elementwise logical XOR between two SoftBool Tensors.

    **Arguments:**
    - `x`: First SoftBool input Tensor.
    - `y`: Second SoftBool input Tensor.
    - `use_geometric_mean`: If True, uses the geometric mean to compute the soft ANDs inside the logical OR. Otherwise, the product is used.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Tensor with values in [0, 1]), relaxing the elementwise logical XOR.
    """
    tmp1 = logical_and(x, logical_not(y), use_geometric_mean=use_geometric_mean)
    tmp2 = logical_and(logical_not(x), y, use_geometric_mean=use_geometric_mean)
    return logical_or(tmp1, tmp2, use_geometric_mean=use_geometric_mean)
