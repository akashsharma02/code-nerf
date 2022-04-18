from typing import NamedTuple
import torch
import torchvision
import torch.nn as nn


class Rays(NamedTuple):
    origins: torch.Tensor
    directions: torch.Tensor


class FlexibleNeRFModel(nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_ids=[4],
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
    ):
        super(FlexibleNeRFModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.skip_connect_ids = skip_connect_ids
        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.use_viewdirs = use_viewdirs

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = 0
        if use_viewdirs:
            self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

        self.layer1 = nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = nn.ModuleList()
        for i in range(self.num_layers - 1):
            if i in self.skip_connect_ids:
                self.layers_xyz.append(nn.Linear(self.dim_xyz + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(nn.Linear(hidden_size, hidden_size))

        if self.use_viewdirs:
            self.fc_feat = nn.Linear(hidden_size, hidden_size)
            self.layers_dir = nn.ModuleList()
            self.layers_dir.append(nn.Linear(self.dim_dir + hidden_size, hidden_size // 2))

            self.fc_alpha = nn.Linear(hidden_size, 1)
            self.fc_rgb = nn.Linear(hidden_size // 2, 3)
        else:
            self.fc_out = nn.Linear(hidden_size, 4)

        self.relu = nn.functional.relu

    def forward(self, x: torch.Tensor):
        """ Forward function for NeRF Model

        :function:
            x: torch.Tensor [sample_size: dim_xyz + dim_dir]
        :returns: TODO

        """
        xyz = x[..., : self.dim_xyz]
        out = self.relu(self.layer1(xyz))
        for i, layer_xyz in enumerate(self.layers_xyz):
            if i in self.skip_connect_ids:
                out = torch.cat((out, xyz), dim=-1)
            out = self.relu(layer_xyz(out))

        if self.use_viewdirs:
            view = x[..., self.dim_xyz:]
            feat = self.relu(self.fc_feat(out))
            sigma = self.fc_alpha(feat)
            out = torch.cat((feat, view), dim=-1)
            for layer_dir in self.layers_dir:
                out = self.relu(layer_dir(out))
            rgb = self.fc_rgb(out)
            return torch.cat((rgb, sigma), dim=-1)
        else:
            return self.fc_out(out)


class ShapeTexturecode(nn.Module):
    def __init__(
        self,
        num_codes,
        code_size=128,
        code_size=128,
    ):
        super(ShapeTexturecode, self).__init__()
        self.num_codes = num_codes
        self.code_size = shape_code_size
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


class PositionalEncoder(nn.Module):
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


class NeRFDecoderNet(nn.Module):
    def __init__(
        self,
            hidden_size: int = 128,
            code_size: int = 128,
            num_encoding_xyz: int = 6,
            num_encoding_dir: int = 4,
            log_sampling: bool = True,
            include_input: bool = True,
    ):
        super(NeRFDecoderNet, self).__init__()
        self.hidden_size = hidden_size
        self.code_size = code_size

        include_input_xyz = 3 if include_input else 0
        include_input_dir = 3 if include_input else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_dir
        self.xyz_encoder = PositionalEncoder(num_encoding_xyz, log_sampling, True)
        self.view_encoder = PositionalEncoder(num_encoding_dir, log_sampling, True)

        self.layer_xyz1 = nn.Linear(self.dim_xyz, self.hidden_size)
        self.layer_xyz2 = nn.Linear(self.hidden_size + self.code_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size + self.code_size, self.shape_code_size + 1)

        self.shape_code_layer1 = nn.Linear(self.code_size, self.shape_code_size)
        self.shape_code_layer2 = nn.Linear(self.code_size, self.shape_code_size)
        self.texture_code_layer1 = nn.Linear(self.code_size, self.code_size)
        self.texture_code_layer2 = nn.Linear(self.code_size, self.code_size)

        self.layer_dir1 = nn.Linear(self.dim_dir + self.code_size, self.hidden_size)
        self.layer_dir2 = nn.Linear(self.hidden_size + self.code_size, self.hidden_size)

        self.fc_rgb = nn.Linear(self.hidden_size + self.code_size, 3)

        self.activation = nn.functional.relu

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, xyz: torch.Tensor, viewdirs: torch.Tensor):
        """ Forward function for NeRF Model

        :function:
            z_s: Shape Latent code [sample_size x code_size]
            z_t: Texture Latent code [sample_size x code_size]
            x: torch.Tensor [sample_size: dim_xyz + dim_dir]
        :returns: TODO

        """
        xyz = self.xyz_encoder(xyz)
        viewdirs = self.view_encoder(viewdirs)

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

        view_in = torch.cat((feat, viewdirs), dim=-1)
        view_out = self.activation(self.layer_dir1(view_in))
        view_out = torch.cat((view_out, z_t_out), dim=-1)
        view_out = self.activation(self.layer_dir2(view_out))
        view_out = torch.cat((view_out, z_t_out2), dim=-1)
        rgb = self.fc_rgb(view_out)

        return torch.cat((rgb, sigma), dim=-1)


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
            z_s: Shape Latent code [sample_size x code_size]
            z_t: Texture Latent code [sample_size x code_size]
            x: torch.Tensor [sample_size: dim_xyz + dim_dir]
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
        self.model.fc = nn.Sequential()
        self.latent_size = latent_size
        self.fc_shape = nn.Linear(512, latent_size)
        self.fc_texture = nn.Linear(512, latent_size)

    # def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
    #     """
    #     Params ignored (compatibility)
    #     :param uv (B, N, 2) only used for shape
    #     :return latent vector (B, L, N)
    #     """
    #     return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
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
                 code_size: int = 128,
                 decoder_hidden_size: int = 128,
                 num_encoding_xyz=6,
                 num_encoding_dir=4,
                 log_sampling=True,
                 include_input=True,
                 ) -> None:

        super(GenerativeNeRF, self).__init__()
        self.encoder = ImageEncoder(
            backbone="resnet34",
            pretrained=True,
            latent_size=code_size
        )
        self.decoder = NeRFDecoderNet(
            decoder_hidden_size,
            code_size,
            num_encoding_xyz,
            num_encoding_dir,
            log_sampling,
            include_input
        )

    def forward(self, x: torch.Tensor, pts: torch.Tensor, viewdirs: torch.Tensor):
        """
        Forward function for GenerativeNeRF model
        Args:
            x: torch.Tensor [N, C, H, W] input image
            pts: torch.Tensor [batch_size, 3]
            viewdirs: torch.Tensor [batch_size, 3]
        """
        shape_code, texture_code = self.encoder(x)
        rgb, sigma = self.decoder(shape_code, texture_code, pts, viewdirs)
        return rgb, sigma
