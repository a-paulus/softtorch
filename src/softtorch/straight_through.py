import functools
import inspect
from collections.abc import Callable

import torch

import softtorch


def grad_replace(fn: Callable) -> Callable:
    """This decorator calls the decorated function twice: once with `forward=True` and once with `forward=False`.
    It returns the output from the forward pass, but uses the output from the backward pass to compute gradients.

    **Arguments:**

    - `fn`: The function to be wrapped. It should accept a `forward` argument that specifies which computation to perform depending on forward or backward pass.

    **Returns:**

    A wrapped function that behaves like the `forward=True` version during the forward pass, but computes gradients using the `forward=False` version during the backward pass.
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        fw_y = fn(*args, **kwargs, forward=True)
        bw_y = fn(*args, **kwargs, forward=False)
        return torch.utils._pytree.tree_map(
            lambda f, b: f if f is None or b is None else (f - b).detach() + b,
            fw_y,
            bw_y,
        )

    return wrapped


def st(fn: Callable) -> Callable:
    """This decorator calls the decorated function twice: once with `mode="hard"` and once with the specified `mode`.
    It returns the output from the hard forward pass, but uses the output from the soft backward pass to compute gradients.

    **Arguments:**

    - `fn`: The function to be wrapped. It may accept a `mode` parameter. If `fn` has no `mode` parameter, it defaults to `"smooth"` and `mode` is passed through via `**kwargs`.

    **Returns:**

    A wrapped function that behaves like the `mode="hard"` version during the forward pass, but computes gradients using the specified `mode` and `softness` during the backward pass.
    """
    sig = inspect.signature(fn)
    mode_param = sig.parameters.get("mode")
    if mode_param is not None:
        mode_default = mode_param.default
        mode_idx = list(sig.parameters.keys()).index("mode")
    else:
        mode_default = "smooth"
        mode_idx = None

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if mode_idx is not None and len(args) > mode_idx:
            # mode was passed positionally — extract it
            mode = args[mode_idx]
            args = args[:mode_idx] + args[mode_idx + 1:]
        else:
            mode = kwargs.pop("mode", mode_default)
        fw_y = fn(*args, **kwargs, mode="hard")
        bw_y = fn(*args, **kwargs, mode=mode)
        return torch.utils._pytree.tree_map(
            lambda f, b: f if f is None or b is None else (f - b).detach() + b,
            fw_y,
            bw_y,
        )

    return wrapped


_st_cache = {}


def _cached_st(fn):
    """Return a cached st() wrapper, avoiding repeated inspect.signature() calls."""
    if fn not in _st_cache:
        _st_cache[fn] = st(fn)
    return _st_cache[fn]


def abs_st(*args, **kwargs):
    """Straight-through version of [`softtorch.abs`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.abs)`.

    Returns the hard `abs` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.abs)(*args, **kwargs)


def argmax_st(*args, **kwargs):
    """Straight-through version of [`softtorch.argmax`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.argmax)`.

    Returns the hard `argmax` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.argmax)(*args, **kwargs)


def argmedian_st(*args, **kwargs):
    """Straight-through version of [`softtorch.argmedian`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.argmedian)`.

    Returns the hard `argmedian` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.argmedian)(*args, **kwargs)


def argmin_st(*args, **kwargs):
    """Straight-through version of [`softtorch.argmin`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.argmin)`.

    Returns the hard `argmin` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.argmin)(*args, **kwargs)


def argquantile_st(*args, **kwargs):
    """Straight-through version of [`softtorch.argquantile`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.argquantile)`.

    Returns the hard `argquantile` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.argquantile)(*args, **kwargs)


def argsort_st(*args, **kwargs):
    """Straight-through version of [`softtorch.argsort`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.argsort)`.

    Returns the hard `argsort` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.argsort)(*args, **kwargs)


def clamp_st(*args, **kwargs):
    """Straight-through version of [`softtorch.clamp`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.clamp)`.

    Returns the hard `clamp` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.clamp)(*args, **kwargs)


def eq_st(*args, **kwargs):
    """Straight-through version of [`softtorch.eq`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.eq)`.

    Returns the hard `eq` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.eq)(*args, **kwargs)


def greater_st(*args, **kwargs):
    """Straight-through version of [`softtorch.greater`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.greater)`.

    Returns the hard `greater` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.greater)(*args, **kwargs)


def greater_equal_st(*args, **kwargs):
    """Straight-through version of [`softtorch.greater_equal`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.greater_equal)`.

    Returns the hard `greater_equal` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.greater_equal)(*args, **kwargs)


def heaviside_st(*args, **kwargs):
    """Straight-through version of [`softtorch.heaviside`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.heaviside)`.

    Returns the hard `heaviside` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.heaviside)(*args, **kwargs)


def isclose_st(*args, **kwargs):
    """Straight-through version of [`softtorch.isclose`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.isclose)`.

    Returns the hard `isclose` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.isclose)(*args, **kwargs)


def less_st(*args, **kwargs):
    """Straight-through version of [`softtorch.less`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.less)`.

    Returns the hard `less` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.less)(*args, **kwargs)


def less_equal_st(*args, **kwargs):
    """Straight-through version of [`softtorch.less_equal`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.less_equal)`.

    Returns the hard `less_equal` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.less_equal)(*args, **kwargs)


def max_st(*args, **kwargs):
    """Straight-through version of [`softtorch.max`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.max)`.

    Returns the hard `max` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.max)(*args, **kwargs)


def median_st(*args, **kwargs):
    """Straight-through version of [`softtorch.median`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.median)`.

    Returns the hard `median` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.median)(*args, **kwargs)


def min_st(*args, **kwargs):
    """Straight-through version of [`softtorch.min`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.min)`.

    Returns the hard `min` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.min)(*args, **kwargs)


def not_equal_st(*args, **kwargs):
    """Straight-through version of [`softtorch.not_equal`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.not_equal)`.

    Returns the hard `not_equal` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.not_equal)(*args, **kwargs)


def quantile_st(*args, **kwargs):
    """Straight-through version of [`softtorch.quantile`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.quantile)`.

    Returns the hard `quantile` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.quantile)(*args, **kwargs)


def rank_st(*args, **kwargs):
    """Straight-through version of [`softtorch.rank`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.rank)`.

    Returns the hard `rank` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.rank)(*args, **kwargs)


def relu_st(*args, **kwargs):
    """Straight-through version of [`softtorch.relu`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.relu)`.

    Returns the hard `relu` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.relu)(*args, **kwargs)


def round_st(*args, **kwargs):
    """Straight-through version of [`softtorch.round`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.round)`.

    Returns the hard `round` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.round)(*args, **kwargs)


def sign_st(*args, **kwargs):
    """Straight-through version of [`softtorch.sign`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.sign)`.

    Returns the hard `sign` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.sign)(*args, **kwargs)


def sort_st(*args, **kwargs):
    """Straight-through version of [`softtorch.sort`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.sort)`.

    Returns the hard `sort` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.sort)(*args, **kwargs)


def topk_st(*args, **kwargs):
    """Straight-through version of [`softtorch.topk`][].
    Implemented using the [`softtorch.st`][] decorator as `st(softtorch.topk)`.

    Returns the hard `topk` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(softtorch.topk)(*args, **kwargs)
