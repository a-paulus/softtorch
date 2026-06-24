from typing import Literal

import torch


def _validate_softness(softness: float | torch.Tensor) -> None:
    """Raise ValueError if softness is not a positive scalar."""
    if isinstance(softness, torch.Tensor):
        if softness.ndim != 0:
            raise ValueError(
                f"softness must be a scalar, got tensor with shape {tuple(softness.shape)}"
            )
        if softness.detach() <= 0:
            raise ValueError(f"softness must be positive, got {softness}")
        return
    if softness <= 0:
        raise ValueError(f"softness must be positive, got {softness}")


def _ensure_float(x: torch.Tensor) -> torch.Tensor:
    """Cast to default float dtype if not already floating point."""
    return x if x.is_floating_point() else x.to(torch.get_default_dtype())


def _quantile_k_a(
    q: torch.Tensor,
    n: int,
    interpolation: Literal["linear", "lower", "higher", "nearest", "midpoint"],
):
    # p in [0, n-1]
    p = q * (n - 1)

    if interpolation == "linear":
        k = torch.floor(p).to(torch.int32)
        a = p - k  # in [0,1)
        take_next = True
    elif interpolation == "lower":
        k = torch.floor(p).to(torch.int32)
        k = torch.clamp(k, 0, n - 1)
        a = torch.zeros_like(p)
        take_next = False
    elif interpolation == "higher":
        k = torch.ceil(p).to(torch.int32)
        k = torch.clamp(k, 0, n - 1)
        a = torch.zeros_like(p)
        take_next = False
    elif interpolation == "nearest":
        # Banker's rounding (round half to even) to match torch.quantile convention
        k = torch.round(p).to(torch.int32)
        k = torch.clamp(k, 0, n - 1)
        a = torch.zeros_like(p)
        take_next = False
    elif interpolation == "midpoint":
        k = torch.floor(p).to(torch.int32)
        a = torch.full_like(p, 0.5)
        # Special-case when p is an integer: midpoint should equal that exact order stat
        is_int = torch.isclose(p, torch.round(p))
        a = torch.where(is_int, torch.tensor(0.0, device=p.device, dtype=p.dtype), a)
        take_next = True
    else:
        raise ValueError(f"Unknown interpolation={interpolation}")
    return k, a, take_next


def _standardize_and_squash(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
    temperature: float = 1.0,
    return_mean_std: bool = False,
) -> torch.Tensor:
    """
    1) standardize: (x - mean) / std  along `dim`
    2) squash: map to (0,1) (or approx [0,1]) with a smooth function
    """
    mean = torch.mean(x, dim=dim, keepdim=True)
    var = torch.mean((x - mean) ** 2, dim=dim, keepdim=True)
    std = torch.sqrt(var + eps)

    z = (x - mean) / std
    z = z / temperature
    z = torch.sigmoid(z)
    if return_mean_std:
        return z, mean, std
    else:
        return z


def _unsquash_and_destandardize(
    y: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Inverse of _standardize_and_squash.
    1) unsquash: map from (0,1) back to R with logit
    2) destandardize: y * std + mean along `dim`
    """
    safe_eps = max(eps, 10 * torch.finfo(y.dtype).eps)
    y = torch.clamp(y, safe_eps, 1.0 - safe_eps)
    z = torch.log(y / (1.0 - y))
    z = z * temperature
    x = z * std + mean
    return x



def _canonicalize_dim(dim: int | None, num_dims: int) -> int:
    if dim is None:
        raise ValueError("dim must be specified")
    if not -num_dims <= dim < num_dims:
        raise ValueError(
            f"dim {dim} is out of bounds for array of dimension {num_dims}"
        )
    if dim < 0:
        dim += num_dims
    return dim


def _polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x) + coeffs[0]
    for c in coeffs[1:]:
        y = y * x + c
    return y
