from typing import Literal, Tuple
import numpy as np
import torch
import torch.nn as nn


def widened_sigmoid(x, eps=0.001):
    return torch.sigmoid(x) * (1 + 2 * eps) - eps


def shifted_softplus(x):
    return torch.nn.functional.softplus(x-1)


def volume_render(
    rgb_points, sigma,
    depth_values,
    ray_directions,
):
    dists = depth_values[..., 1:] - depth_values[..., :-1]
    # Add distance from far-limit to infinity to retain shape (64 samples or 128 samples)
    dists = torch.cat((dists, torch.full_like(dists[..., :1], 1e10)), dim=-1)

    delta = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    sigma_a = shifted_softplus(sigma.squeeze())
    sigma_delta = sigma_a * delta

    rgb_rays = widened_sigmoid(rgb_points)
    transmittance = torch.exp(-torch.cat([
        torch.zeros_like(sigma_delta[..., :1]),
        torch.cumsum(sigma_delta[..., :-1], axis=-1)
    ], dim=-1))
    alpha = 1.0 - torch.exp(-sigma_delta)
    weights = alpha * transmittance

    rgb_map = (weights[..., None] * rgb_rays).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    return rgb_map, disp_map, acc_map, weights, depth_map


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Gaussian(nn.Module):
    def __init__(self, mean=0, std=1.):
        super().__init__()
        self.mean, self.std = mean, std

    def forward(self, x):
        return torch.exp((-(x - self.mean) ** 2)/(2 * self.std ** 2))


class SirenLinear(nn.Module):
    def __init__(self, in_features, out_features, w0=1.):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            self.linear.weight.uniform_(-np.sqrt(6 / in_features) / w0, np.sqrt(6 / in_features) / w0)
        self.sine = Sine()

    def forward(self, x):
        x = self.linear(x)
        x = self.sine(x)
        return x


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        if isinstance(self.activation, Sine):
            with torch.no_grad():
                self.layer.weight.uniform_(-np.sqrt(6 / input_dim) / self.activation.w0, np.sqrt(6 / input_dim) / self.activation.w0)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.expand_as(x)
        phase_shift = phase_shift.expand_as(x)
        return self.activation(freq * x + phase_shift)


class ResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.linear1 = nn.Linear(in_features, out_features, bias=bias)
        self.linear2 = nn.Linear(out_features, out_features, bias=bias)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x_init):
        x = self.relu1(self.linear1(x_init))
        x = x_init + self.linear2(x)
        x = self.relu2(x)
        return x


class PositionalEncoder(object):
    """Positionally encode the input vector through fourier basis with given frequency bands"""

    def __init__(self, num_freq: int, log_sampling: bool, include_input: bool) -> None:
        assert num_freq > 0, "Number of frequency samples should be a positive integer"
        self.num_freq = num_freq
        self.log_sampling = log_sampling
        self.include_input = include_input

        self.frequency_bands = None
        if self.log_sampling:
            self.frequency_bands = 2.0 ** torch.linspace(0.0, self.num_freq - 1, self.num_freq)
        else:
            self.frequency_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (self.num_freq - 1), self.num_freq)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """

        :function:
            tensor: torch.Tensor [:, num_dim] Tensor to be positionally embedded
        :returns:
            tensor: torch.Tensor [:, num_dim*num_freq*2] Positionally embedded feature vector

        """
        encoding = [tensor] if self.include_input else []
        for i, freq in enumerate(self.frequency_bands):
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


class PointSampler(object):

    """Sample 3D points along the given rays"""

    def __init__(self,
                 num_samples: int,
                 near: float,
                 far: float,
                 spacing_mode: Literal["lindisp", "lindepth"],
                 perturb: bool):

        assert near >= 0 and far > near, "Near and far ranges should be positive values, and far > near"
        assert num_samples > 0, "Number of samples must be greater than 0"

        self.num_samples = num_samples
        self.near = near
        self.far = far
        self.spacing_mode = spacing_mode
        self.perturb = perturb

        self.t_vals = torch.linspace(0.0, 1.0, self.num_samples)
        if self.spacing_mode == "lindisp":
            self.z_vals = self.near * (1.0 - self.t_vals) + self.far * self.t_vals
        else:
            self.z_vals = 1.0 / (1.0 / self.near * (1.0 - self.t_vals) + 1.0 / self.far * self.t_vals)

        self.mids = 0.5 * (self.z_vals[..., 1:] + self.z_vals[..., :-1])
        self.upper = torch.cat((self.mids, self.z_vals[..., -1:]), dim=-1)
        self.lower = torch.cat((self.z_vals[..., :1], self.mids), dim=-1)

    def sample_uniform(self, rays: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Uniform sample points according to spacing mode along the ray

        :function:
            ro: [num_random_rays, 3] ray_origins
            rd: [num_random_rays, 3] ray_directions
        :returns:
            pts: [num_random_rays*num_samples, 3] pts alongs the ray
            z_vals: [num_random_rays, num_samples, 3] z_vals along the ray

        """
        ray_origins, ray_directions = rays[..., :3], rays[..., 3:]
        num_random_rays = ray_origins.shape[-2]
        if self.perturb:
            upper = self.upper.expand(num_random_rays, self.num_samples)
            lower = self.lower.expand(num_random_rays, self.num_samples)
            t_rand = torch.rand_like(upper, dtype=ray_origins.dtype)
            z_vals = lower + (upper - lower) * t_rand
        else:
            z_vals = self.z_vals.expand(num_random_rays, self.num_samples)
        z_vals = z_vals.to(rays.device)
        assert z_vals.shape == torch.Size([num_random_rays, self.num_samples]), "Incorrect shape of depth samples z_vals"
        pts = ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
        return pts, z_vals

    def normalize(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Normalize the points to be in range: -1 to 1 for Siren
        """
        min_point = torch.Tensor([self.near, self.near, self.near]).to(pts.device)
        max_point = torch.Tensor([self.far, self.far, self.far]).to(pts.device)
        pts = pts - min_point[None, None, :]
        pts = pts / max_point[None, None, :]
        pts = (pts - 0.5) * 2
        return pts
