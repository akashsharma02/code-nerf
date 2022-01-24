import torch


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod


def widened_sigmoid(x, eps=0.001):
    return torch.sigmoid(x) * (1 + 2 * eps) - eps


def shifted_softplus(x):
    return torch.nn.functional.softplus(x-1)


def volume_render(
    radiance_field,
    depth_values,
    ray_directions,
):
    dists = depth_values[..., 1:] - depth_values[..., :-1]
    # Add distance from far-limit to infinity to retain shape (64 samples or 128 samples)
    dists = torch.cat((dists, torch.full_like(dists[..., :1], 1e10)), dim=-1)

    delta = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    # sigma_a = torch.nn.functional.softplus(radiance_field[..., 3] + noise)
    sigma_a = shifted_softplus(radiance_field[..., 3])
    sigma_delta = sigma_a * delta

    rgb = widened_sigmoid(radiance_field[..., :3])
    transmittance = torch.exp(-torch.cat([
        torch.zeros_like(sigma_delta[..., :1]),
        torch.cumsum(sigma_delta[..., :-1], axis=-1)
    ], dim=-1))
    alpha = 1.0 - torch.exp(-sigma_delta)
    weights = alpha * transmittance

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    return rgb_map, disp_map, acc_map, weights, depth_map
