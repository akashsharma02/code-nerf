from typing import Literal, Tuple
import torch
import torchvision
import torch.nn as nn


class ShapeTexturecode(nn.Module):
    def __init__(
        self,
        num_codes,
        code_size=128,
    ):
        super(ShapeTexturecode, self).__init__()
        self.num_codes = num_codes
        self.code_size = code_size

        self.shape_code = nn.code(self.num_codes, self.code_size)
        self.texture_code = nn.code(self.num_codes, self.code_size)

    def forward(self, object_ids: torch.Tensor):
        z_s = self.shape_code(object_ids)
        z_t = self.texture_code(object_ids)
        return z_s, z_t

    def get_all_codes(self, device: torch.cuda.Device):
        all_idx = torch.arange(0, self.num_codes, dtype=int, device=device)
        z_s = self.shape_code(all_idx)
        z_t = self.texture_code(all_idx)
        return z_s, z_t


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


# class RaySampler(object):
#
#     """RaySampler samples rays for a given image size and intrinsics """
#
#     def __init__(self, num_samples: int, height: int, width: int, intrinsics: Union[torch.Tensor, np.ndarray]):
#         """ Prepares a ray bundle for a given image size and intrinsics
#
#         :Function: intrinsics: torch.Tensor 4x4
#
#         """
#         assert height > 0 and width > 0, "Height and width must be positive integers"
#         assert num_samples > 0 and num_samples <= height * width, "Sample size must be a positive number less than height * width"
#
#         self.height = height
#         self.width = width
#         self.num_samples = num_samples
#
#         if isinstance(intrinsics, np.ndarray):
#             intrinsics = torch.from_numpy(intrinsics)
#         assert intrinsics.shape == torch.Size(
#             [1, 4, 4]), "Incorrect intrinsics shape"
#         self.intrinsics = intrinsics
#         dtype = self.intrinsics.dtype
#         device = self.intrinsics.device
#         self.focal_length = self.intrinsics[..., 0, 0]
#         self.cx = self.intrinsics[..., 0, 2]
#         self.cy = self.intrinsics[..., 1, 2]
#
#         ii, jj = torch.meshgrid(
#             torch.arange(
#                 width, dtype=dtype, device=device
#             ),
#             torch.arange(
#                 height, dtype=dtype, device=device
#             ),
#             indexing='xy'
#         )
#         self.directions = torch.stack(
#             [
#                 (ii - self.cx) / self.focal_length,
#                 -(jj - self.cy) / self.focal_length,
#                 -torch.ones_like(ii),
#             ],
#             dim=-1,
#         )
#
#     def sample(self, ray_bundle: Optional[torch.Tensor] = None, world_T_camera: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
#         """
#         Rotate the bundle of rays given the camera pose and return a random subset of rays
#
#         :function:
#             world_T_camera: [batch, 4, 4] torch.Tensor camera pose (SE3)
#         :returns:
#             ray origins: torch.Tensor [batch_size*num_samples, 3]
#             ray directions: torch.Tensor [batch_size*num_samples, 3]
#             select_inds: np.ndarray [batch_size*num_samples]
#
#         """
#         batch_size = None
#         if ray_bundle is None:
#             assert world_T_camera is None, "world_T_camera pose required when ray_bundle is not supplied"
#             batch_size = world_T_camera.shape[0]
#             ray_bundle = self.get_bundle(world_T_camera)
#         else:
#             batch_size = ray_bundle.origins.shape[0]
#
#         ray_origins, ray_directions = ray_bundle.origins.flatten(1, 2), ray_bundle.directions.flatten(1, 2)
#
#         select_inds = []
#         pixel_range = np.arange(0, ray_origins.shape[-2])
#         for _ in range(batch_size):
#             select_inds.append(np.random.permutation(pixel_range)[:self.num_samples])
#         select_inds = np.asarray(select_inds)
#
#         ray_origins = [ray_origins[i, select_inds[i], :] for i in range(batch_size)]
#         ray_directions = [ray_directions[i, select_inds[i], :] for i in range(batch_size)]
#         ray_origins = torch.cat(ray_origins, dim=0)
#         ray_directions = torch.cat(ray_directions, dim=0)
#
#         return Rays(ray_origins, ray_directions), select_inds
#
#     def get_bundle(self, world_T_camera: torch.Tensor):
#         """
#             Rotate the bundle of rays given the camera pose
#
#         :function:
#             world_T_camera: 4x4 torch.Tensor camera pose (SE3)
#         :returns:
#             ray origins: torch.Tensor [batch, H, W, 3]
#             ray directions: torch.Tensor [batch, H, W, 3]
#
#         """
#         directions = self.directions[..., None].to(world_T_camera.device)
#         ray_directions = torch.einsum('hwij, bji->bhwj', directions, world_T_camera[..., :3, :3]).contiguous()
#         ray_origins = world_T_camera[..., :3, -1][:, None, None, :].expand(ray_directions.shape)
#         return Rays(ray_origins, ray_directions)
#

class NeRFDecoderNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        code_size: int = 128,
        num_encoding_xyz: int = 6,
        log_sampling: bool = True,
        include_input: bool = True,
    ):
        super(NeRFDecoderNet, self).__init__()
        self.hidden_size = hidden_size
        self.code_size = code_size

        include_input_xyz = 3 if include_input else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_xyz
        self.xyz_encoder = PositionalEncoder(num_encoding_xyz, log_sampling, True)

        self.layer_xyz1 = nn.Linear(self.dim_xyz, self.hidden_size)
        self.layer_xyz2 = nn.Linear(self.hidden_size + self.code_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size + self.code_size, self.code_size + 1)

        self.shape_code_layer1 = nn.Linear(self.code_size, self.code_size)
        self.shape_code_layer2 = nn.Linear(self.code_size, self.code_size)
        self.texture_code_layer1 = nn.Linear(self.code_size, self.code_size)
        self.texture_code_layer2 = nn.Linear(self.code_size, self.code_size)

        self.layer_dir1 = nn.Linear(self.code_size, self.hidden_size)
        self.layer_dir2 = nn.Linear(self.hidden_size + self.code_size, self.hidden_size)

        self.fc_rgb = nn.Linear(self.hidden_size + self.code_size, 3)

        self.activation = nn.functional.relu

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, xyz: torch.Tensor):
        """ Forward function for NeRF Model

        :function:
            z_s: Shape Latent code [num_samples x code_size]
            z_t: Texture Latent code [num_samples x code_size]
            x: torch.Tensor [num_samples: dim_xyz + dim_dir]
        :returns: TODO

        """
        xyz = self.xyz_encoder.forward(xyz)

        z_s_out = self.activation(self.shape_code_layer1(z_s))
        z_s_out2 = self.activation(self.shape_code_layer2(z_s))

        z_t_out = self.activation(self.texture_code_layer1(z_t))
        z_t_out2 = self.activation(self.texture_code_layer2(z_t))

        xyz_out = self.activation(self.layer_xyz1(xyz))
        xyz_out = torch.cat((xyz_out, z_s_out), dim=-1)
        xyz_out = self.activation(self.layer_xyz2(xyz_out))
        xyz_out = torch.cat((xyz_out, z_s_out2), dim=-1)

        feat = self.fc_out(xyz_out)

        sigma, feat = feat[..., :1], feat[..., 1:]

        view_out = self.activation(self.layer_dir1(feat))
        view_out = torch.cat((view_out, z_t_out), dim=-1)
        view_out = self.activation(self.layer_dir2(view_out))
        view_out = torch.cat((view_out, z_t_out2), dim=-1)
        rgb = self.fc_rgb(view_out)

        return rgb, sigma


class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class DisentangledNeRFModel(nn.Module):
    def __init__(
        self,
        hidden_size=128,
        code_size=128,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True
    ):
        super(DisentangledNeRFModel, self).__init__()
        self.hidden_size = hidden_size
        self.code_size = code_size

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

        self.layer_xyz1 = nn.Linear(self.dim_xyz, self.hidden_size)
        self.layer_xyz2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.shape_code_layer1 = nn.Linear(self.code_size, self.shape_code_size)
        self.shape_code_layer2 = nn.Linear(self.code_size, self.shape_code_size)
        self.fc_out = nn.Linear(self.hidden_size + self.code_size, self.shape_code_size + 1)

        self.layer_rgb1 = nn.Linear(self.dim_dir + self.code_size, self.hidden_size)
        self.layer_rgb2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.texture_code_layer1 = nn.Linear(self.code_size + self.code_size, self.code_size)
        self.texture_code_layer2 = nn.Linear(self.code_size, self.code_size)

        self.fc_rgb = nn.Linear(self.hidden_size + self.code_size, 3)

        self.activation = nn.functional.relu
        # self.activation = Sine

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, x: torch.Tensor):
        """ Forward function for NeRF Model

        :function:
            z_s: Shape Latent code [num_samples x code_size]
            z_t: Texture Latent code [num_samples x code_size]
            x: torch.Tensor [num_samples: dim_xyz + dim_dir]
        :returns: TODO

        """

        xyz = x[..., : self.dim_xyz]
        view = x[..., self.dim_xyz:]

        xyz_out = self.activation(self.layer_xyz1(xyz))
        xyz_out = self.activation(self.layer_xyz2(xyz_out))

        z_s_out = self.activation(self.shape_code_layer1(z_s))
        z_s_out = self.activation(self.shape_code_layer2(z_s_out))

        shape_feat_in = torch.cat((xyz_out, z_s_out), dim=-1)
        feat = self.fc_out(shape_feat_in)
        sigma, feat = feat[..., :1], feat[..., 1:]

        dir_in = torch.cat((feat, view), dim=-1)
        dir_out = self.activation(self.layer_rgb1(dir_in))
        dir_out = self.activation(self.layer_rgb2(dir_out))

        z_t_in = torch.cat((feat, z_t), dim=-1)
        z_t_out = self.activation(self.texture_code_layer1(z_t_in))
        z_t_out = self.activation(self.texture_code_layer2(z_t_out))

        view_out = torch.cat((dir_out, z_t_out), dim=-1)
        rgb = self.fc_rgb(view_out)

        return torch.cat((rgb, sigma), dim=-1)


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.latent_size = latent_size
        self.fc_shape = nn.Linear(512, latent_size)
        self.fc_texture = nn.Linear(512, latent_size)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        shape_code = self.fc_shape(x)
        texture_code = self.fc_texture(x)

        return shape_code, texture_code


class GenerativeNeRF(nn.Module):
    def __init__(self,
                 point_sampler: PointSampler,
                 code_size: int = 128,
                 decoder_hidden_size: int = 128,
                 num_encoding_xyz=6,
                 log_sampling=True,
                 include_input=True,
                 ) -> None:

        super(GenerativeNeRF, self).__init__()
        self.encoder = ImageEncoder(
            backbone="resnet34",
            pretrained=True,
            latent_size=code_size
        )
        self.point_sampler = point_sampler
        self.decoder = NeRFDecoderNet(
            decoder_hidden_size,
            code_size,
            num_encoding_xyz,
            log_sampling,
            include_input
        )

    def forward(self, x: torch.Tensor, rays: torch.Tensor):
        """
        Forward function for GenerativeNeRF model
        Args:
            x: torch.Tensor [N, C, H, W] input image
            pts: torch.Tensor [batch_size, 3]
            viewdirs: torch.Tensor [batch_size, 3]
        """
        shape_code, texture_code = self.encoder(x)
        # rays, select_inds = self.ray_sampler.sample(world_T_camera)
        pts, z_vals = self.point_sampler.sample_uniform(rays)
        num_rays, num_points = rays.shape[0], pts.shape[1]
        shape_code = shape_code[:, None, :].expand(num_rays, num_points, -1)
        texture_code = texture_code[:, None, :].expand(num_rays, num_points, -1)
        rgb_per_xyz, sigma_per_xyz = self.decoder(shape_code, texture_code, pts)
        ray_directions = rays[..., 3:]
        rgb_per_ray, disparity_per_ray, _, _, depth_per_ray = volume_render(rgb_per_xyz, sigma_per_xyz, z_vals, ray_directions)
        return rgb_per_ray, depth_per_ray


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
