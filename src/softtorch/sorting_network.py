import torch


def _soft_compare_and_swap(
    a: torch.Tensor, b: torch.Tensor, softness: float | torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (soft_min, soft_max, sigma) via sigmoidal mixing."""
    from softtorch.functions import sigmoidal

    sigma = sigmoidal(a - b, softness=softness, mode=mode)
    soft_min = sigma * b + (1.0 - sigma) * a
    soft_max = sigma * a + (1.0 - sigma) * b
    return soft_min, soft_max, sigma


def _bitonic_sort_ascending(x: torch.Tensor, softness: float | torch.Tensor, mode: str) -> torch.Tensor:
    """Ascending bitonic sort.  Last dim must have power-of-2 length."""
    n = x.shape[-1]
    num_phases = (n.bit_length() - 1) if n > 1 else 0

    for phase in range(num_phases):
        for sub_step in range(phase + 1):
            d = 1 << (phase - sub_step)
            indices = torch.arange(n, device=x.device)
            partner = indices ^ d
            block_size = 1 << (phase + 1)
            ascending_block = (indices & block_size) == 0

            soft_min, soft_max, _ = _soft_compare_and_swap(
                x, x[..., partner], softness, mode
            )
            x = torch.where(
                ascending_block,
                torch.where(indices < partner, soft_min, soft_max),
                torch.where(indices < partner, soft_max, soft_min),
            )

    return x


def _sort_via_sorting_network(
    x: torch.Tensor,
    softness: float | torch.Tensor,
    mode: str,
    descending: bool,
    standardized: bool = False,
) -> torch.Tensor:
    """Sort along the last axis using a soft bitonic sorting network."""
    *batch_shape, n = x.shape

    n_padded = 1 << (n - 1).bit_length() if n > 1 else 2
    if n_padded > n:
        # Pad with 1.0 when input is in (0,1) from sigmoid; else finite sentinel.
        pad_val = 1.0 if standardized else x.max().item() + 1.0
        pad_width = (0, n_padded - n)  # pad last dim on the right
        x = torch.nn.functional.pad(x, pad_width, value=pad_val)

    # Sort ascending so padding ends up at the tail; flip after truncation.
    if batch_shape:
        x_flat = x.reshape(-1, n_padded)
        rows = []
        for i in range(x_flat.shape[0]):
            rows.append(_bitonic_sort_ascending(x_flat[i], softness, mode))
        sorted_x = torch.stack(rows).reshape(*batch_shape, n_padded)
    else:
        sorted_x = _bitonic_sort_ascending(x, softness, mode)

    if n_padded > n:
        sorted_x = sorted_x[..., :n]
    if descending:
        sorted_x = torch.flip(sorted_x, dims=(-1,))

    return sorted_x


def _bitonic_argsort_ascending(
    x: torch.Tensor, softness: float | torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D ascending bitonic sort with permutation tracking.

    Returns (sorted_x, P) where P is an (n, n) soft permutation matrix:
    P[sorted_pos, original_elem] = probability that original_elem ends up at sorted_pos.
    """
    n = x.shape[-1]
    P = torch.eye(n, device=x.device, dtype=x.dtype)
    num_phases = (n.bit_length() - 1) if n > 1 else 0

    for phase in range(num_phases):
        for sub_step in range(phase + 1):
            d = 1 << (phase - sub_step)
            indices = torch.arange(n, device=x.device)
            partner = indices ^ d
            block_size = 1 << (phase + 1)
            ascending_block = (indices & block_size) == 0

            soft_min, soft_max, sigma = _soft_compare_and_swap(
                x, x[..., partner], softness, mode
            )
            x = torch.where(
                ascending_block,
                torch.where(indices < partner, soft_min, soft_max),
                torch.where(indices < partner, soft_max, soft_min),
            )

            # Positions receiving soft_min take σ from partner's row;
            # positions receiving soft_max take (1-σ) from partner's row.
            gets_min = (ascending_block & (indices < partner)) | (
                ~ascending_block & (indices >= partner)
            )
            mix = torch.where(gets_min, sigma, 1.0 - sigma)  # (n,)

            P_partner = P[partner]
            P = (1.0 - mix[:, None]) * P + mix[:, None] * P_partner

    return x, P


def _argsort_via_sorting_network(
    x: torch.Tensor,
    softness: float | torch.Tensor,
    mode: str,
    descending: bool,
    standardized: bool = False,
) -> torch.Tensor:
    """Argsort along the last axis using a soft bitonic sorting network.

    Returns a soft permutation matrix P of shape (..., n, n) where P[..., sorted_pos, original_elem] is the probability that original_elem ends up at sorted_pos.
    """
    *batch_shape, n = x.shape

    n_padded = 1 << (n - 1).bit_length() if n > 1 else 2
    if n_padded > n:
        pad_val = 1.0 if standardized else x.max().item() + 1.0
        pad_width = (0, n_padded - n)
        x = torch.nn.functional.pad(x, pad_width, value=pad_val)

    if batch_shape:
        x_flat = x.reshape(-1, n_padded)
        P_rows = []
        for i in range(x_flat.shape[0]):
            _, P_i = _bitonic_argsort_ascending(x_flat[i], softness, mode)
            P_rows.append(P_i)
        P = torch.stack(P_rows).reshape(*batch_shape, n_padded, n_padded)
    else:
        _, P = _bitonic_argsort_ascending(x, softness, mode)

    # Strip padding rows and columns, renormalize
    if n_padded > n:
        P = P[..., :n, :n]
        P = P / torch.clamp(torch.sum(P, dim=-1, keepdim=True), min=1e-10)

    if descending:
        P = torch.flip(P, dims=(-2,))

    return P
