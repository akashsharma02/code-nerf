from typing import Literal, Tuple
import torch
import torchvision
import torch.nn as nn
from . import ResidualLayer, SirenLinear, FiLMLayer, PositionalEncoder, PointSampler, volume_render, Sine, Gaussian


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


class FiLMLinearNeRFDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        code_size: int = 128,
        num_cond_layers: int = 6,
        num_encoding_xyz: int = 6,
        log_sampling: bool = True,
        include_input: bool = True
    ):
        super(FiLMLinearNeRFDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.code_size = code_size
        self.num_cond_layers = num_cond_layers

        include_input_xyz = 3 if include_input else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_xyz
        self.xyz_encoder = PositionalEncoder(num_encoding_xyz, log_sampling, True)

        self.layer_xyz1 = FiLMLayer(self.dim_xyz, self.hidden_size, torch.nn.functional.relu)
        self.layer_xyz2 = FiLMLayer(self.hidden_size, self.hidden_size, torch.nn.functional.relu)
        self.layer_xyz3 = FiLMLayer(self.hidden_size, self.hidden_size, torch.nn.functional.relu)
        self.layer_xyz4 = FiLMLayer(self.hidden_size, self.hidden_size, torch.nn.functional.relu)
        self.layer_xyz5 = FiLMLayer(self.hidden_size, self.hidden_size, torch.nn.functional.relu)

        self.layer_rgb = FiLMLayer(self.hidden_size, self.hidden_size, torch.nn.functional.relu)
        self.fc_sigma = nn.Linear(self.hidden_size, 1)
        self.fc_rgb = nn.Linear(self.hidden_size, 3)

    def forward(self, freq: torch.Tensor, phase_shift: torch.Tensor, xyz: torch.Tensor):
        """ Forward function for NeRF Model

        :function:
            z_s: Shape Latent code [num_samples x code_size]
            z_t: Texture Latent code [num_samples x code_size]
            x: torch.Tensor [num_samples: dim_xyz + dim_dir]
        :returns: TODO

        """
        xyz = self.xyz_encoder.forward(xyz)
        freq_list, phase_shift_list = [], []
        for i in range(self.num_cond_layers):
            freq_list.append(freq[..., i * self.hidden_size: (i+1) * self.hidden_size])
            phase_shift_list.append(phase_shift[..., i * self.hidden_size: (i+1) * self.hidden_size])

        xyz_out = self.layer_xyz1(xyz, freq_list[0], phase_shift_list[0])
        xyz_out = self.layer_xyz2(xyz_out, freq_list[0], phase_shift_list[0])
        xyz_out = self.layer_xyz3(xyz_out, freq_list[0], phase_shift_list[0])
        xyz_out = self.layer_xyz4(xyz_out, freq_list[0], phase_shift_list[0])
        xyz_out = self.layer_xyz5(xyz_out, freq_list[0], phase_shift_list[0])

        sigma = self.fc_sigma(xyz_out)

        view_out = self.layer_rgb(xyz_out, freq_list[1], phase_shift_list[1])
        rgb = self.fc_rgb(view_out)

        return rgb, sigma


class FiLMNeRFDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        code_size: int = 128,
        num_cond_layers: int = 6
    ):
        super(FiLMNeRFDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.code_size = code_size
        self.num_cond_layers = num_cond_layers

        self.layer_xyz1 = FiLMLayer(3, self.hidden_size, Sine(30.0))
        self.layer_xyz2 = FiLMLayer(self.hidden_size, self.hidden_size, Sine())
        self.layer_xyz3 = FiLMLayer(self.hidden_size, self.hidden_size, Sine())
        self.layer_xyz4 = FiLMLayer(self.hidden_size, self.hidden_size, Sine())
        self.layer_xyz5 = FiLMLayer(self.hidden_size, self.hidden_size, Sine())

        self.layer_rgb = FiLMLayer(self.hidden_size, self.hidden_size, Sine())
        self.fc_sigma = nn.Linear(self.hidden_size, 1)
        self.fc_rgb = nn.Linear(self.hidden_size, 3)

    def forward(self, freq: torch.Tensor, phase_shift: torch.Tensor, xyz: torch.Tensor):
        """ Forward function for NeRF Model

        :function:
            z_s: Shape Latent code [num_samples x code_size]
            z_t: Texture Latent code [num_samples x code_size]
            x: torch.Tensor [num_samples: dim_xyz + dim_dir]
        :returns: TODO

        """

        freq_list, phase_shift_list = [], []
        for i in range(self.num_cond_layers):
            freq_list.append(freq[..., i * self.hidden_size: (i+1) * self.hidden_size])
            phase_shift_list.append(phase_shift[..., i * self.hidden_size: (i+1) * self.hidden_size])

        xyz_out = self.layer_xyz1(xyz, freq_list[0], phase_shift_list[0])
        xyz_out = self.layer_xyz2(xyz_out, freq_list[0], phase_shift_list[0])
        xyz_out = self.layer_xyz3(xyz_out, freq_list[0], phase_shift_list[0])
        xyz_out = self.layer_xyz4(xyz_out, freq_list[0], phase_shift_list[0])
        xyz_out = self.layer_xyz5(xyz_out, freq_list[0], phase_shift_list[0])

        sigma = self.fc_sigma(xyz_out)

        view_out = self.layer_rgb(xyz_out, freq_list[1], phase_shift_list[1])
        rgb = self.fc_rgb(view_out)

        return rgb, sigma


class SirenNeRFDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        code_size: int = 128,
        num_encoding_xyz: int = 6,
        log_sampling: bool = True,
        include_input: bool = True,
    ):
        super(SirenNeRFDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.code_size = code_size

        self.layer_xyz1 = SirenLinear(3, self.hidden_size)
        self.layer_xyz2 = SirenLinear(self.hidden_size + self.code_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size + self.code_size, self.code_size + 1)

        self.layer_dir1 = SirenLinear(self.code_size, self.hidden_size)
        self.layer_dir2 = SirenLinear(self.hidden_size + self.code_size, self.hidden_size)

        self.fc_rgb = nn.Linear(self.hidden_size + self.code_size, 3)

        # self.activation = nn.functional.relu

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, xyz: torch.Tensor):
        """ Forward function for NeRF Model

        :function:
            z_s: Shape Latent code [num_samples x code_size]
            z_t: Texture Latent code [num_samples x code_size]
            x: torch.Tensor [num_samples: dim_xyz + dim_dir]
        :returns: TODO

        """

        xyz_out = self.layer_xyz1(xyz)
        xyz_out = torch.cat((xyz_out, z_s), dim=-1)
        xyz_out = self.layer_xyz2(xyz_out)
        xyz_out = torch.cat((xyz_out, z_s), dim=-1)

        feat = self.fc_out(xyz_out)

        sigma, feat = feat[..., :1], feat[..., 1:]

        view_out = self.layer_dir1(feat)
        view_out = torch.cat((view_out, z_t), dim=-1)
        view_out = self.layer_dir2(view_out)
        view_out = torch.cat((view_out, z_t), dim=-1)
        rgb = self.fc_rgb(view_out)

        return rgb, sigma


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
        self.model.fc = nn.Sequential()
        self.fc_shape = nn.Linear(512, latent_size)
        self.fc_texture = nn.Linear(512, latent_size)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = self.model(x)
        shape_code = self.fc_shape(x)
        texture_code = self.fc_texture(x)

        return shape_code, texture_code


class ImageEncoderStyle(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128, num_cond_layers=4):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        # Remove the final classification layer
        self.model.fc = nn.Sequential()
        self.cond_dim = latent_size*2*num_cond_layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.cond_dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualLayer(self.cond_dim, self.cond_dim)
        )

    def forward(self, inp):
        """
        For extracting ResNet's features.
        :param inp image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = self.model(inp)
        x = self.fc(x)

        freq = x[..., :x.shape[-1]//2]
        phase_shift = x[..., x.shape[-1]//2:]

        return freq, phase_shift  # freq and phase_shift = [..., latent_size*num_cond_layers]


class GenerativeNeRF(nn.Module):
    def __init__(self,
                 point_sampler: PointSampler,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 ) -> None:

        super(GenerativeNeRF, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.point_sampler = point_sampler

    def forward(self, x: torch.Tensor, rays: torch.Tensor):
        """
        Forward function for GenerativeNeRF model
        Args:
            x: torch.Tensor [N, C, H, W] input image
            pts: torch.Tensor [batch_size, 3]
            viewdirs: torch.Tensor [batch_size, 3]
        """
        shape_code, texture_code = self.encoder(x)

        pts, z_vals = self.point_sampler.sample_uniform(rays)
        num_rays, num_points = rays.shape[0], pts.shape[1]
        shape_code = shape_code[:, None, :].expand(num_rays, num_points, -1)
        texture_code = texture_code[:, None, :].expand(num_rays, num_points, -1)
        rgb_per_xyz, sigma_per_xyz = self.decoder(shape_code, texture_code, pts)
        ray_directions = rays[..., 3:]
        rgb_per_ray, disparity_per_ray, _, _, depth_per_ray = volume_render(rgb_per_xyz, sigma_per_xyz, z_vals, ray_directions)
        return rgb_per_ray, depth_per_ray


class GenerativeFiLMNeRF(nn.Module):
    def __init__(self,
                 point_sampler: PointSampler,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 ) -> None:

        super(GenerativeFiLMNeRF, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.point_sampler = point_sampler

    def forward(self, x: torch.Tensor, rays: torch.Tensor):
        """
        Forward function for GenerativeNeRF model
        Args:
            x: torch.Tensor [N, C, H, W] input image
            pts: torch.Tensor [batch_size, 3]
            viewdirs: torch.Tensor [batch_size, 3]
        """
        freq, phase_shift = self.encoder(x)
        pts, z_vals = self.point_sampler.sample_uniform(rays)
        ray_directions = rays[..., 3:]
        rgb_per_xyz, sigma_per_xyz = self.decoder(freq, phase_shift, pts)
        rgb_per_ray, disparity_per_ray, _, _, depth_per_ray = volume_render(rgb_per_xyz, sigma_per_xyz, z_vals, ray_directions)
        return rgb_per_ray, depth_per_ray
