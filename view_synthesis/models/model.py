from typing import Tuple
import torch


class FlexibleNeRFModel(torch.nn.Module):
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

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            if i in self.skip_connect_ids:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        if self.use_viewdirs:
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
            self.layers_dir = torch.nn.ModuleList()
            self.layers_dir.append(torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2))

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu

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


def get_params_tensor(model, is_distributed):
    if is_distributed:
        shape_params, texture_params = model.module.get_params_tensor()
    else:
        shape_params, texture_params = model.get_params_tensor()
    return shape_params, texture_params


class ShapeTextureEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings,
        shape_code_size=128,
        texture_code_size=128,
    ):
        super(ShapeTextureEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.shape_code_size = shape_code_size
        self.texture_code_size = texture_code_size

        self.shape_embedding = torch.nn.Embedding(self.num_embeddings, self.shape_code_size)
        self.texture_embedding = torch.nn.Embedding(self.num_embeddings, self.texture_code_size)

    def forward(self, object_ids: torch.Tensor):
        z_s = self.shape_embedding(object_ids)
        z_t = self.texture_embedding(object_ids)
        return z_s, z_t

    def get_all_embeddings(self, device: torch.cuda.Device):
        all_idx = torch.arange(0, self.num_embeddings, dtype=int, device=device)
        z_s = self.shape_embedding(all_idx)
        z_t = self.texture_embedding(all_idx)
        return z_s, z_t

    def get_params_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        shape_params, texture_params = None, None
        for name, params in self.named_parameters():
            if 'shape' in name:
                shape_params = torch.cat([x.view(-1) for x in params.data])
            if 'texture' in name:
                texture_params = torch.cat([x.view(-1) for x in params.data])
        return shape_params, texture_params


class CodeNeRFModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size=128,
        num_embeddings=1,
        shape_code_size=128,
        texture_code_size=128,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
    ):
        super(CodeNeRFModel, self).__init__()
        self.hidden_size = hidden_size
        self.shape_code_size = shape_code_size
        self.texture_code_size = texture_code_size

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

        self.layer_xyz1 = torch.nn.Linear(self.dim_xyz, self.hidden_size)
        self.layer_xyz2 = torch.nn.Linear(self.hidden_size + self.shape_code_size, self.hidden_size)
        self.fc_out = torch.nn.Linear(self.hidden_size + self.shape_code_size, self.shape_code_size + 1)

        self.shape_code_layer1 = torch.nn.Linear(self.shape_code_size, self.shape_code_size)
        self.shape_code_layer2 = torch.nn.Linear(self.shape_code_size, self.shape_code_size)
        self.texture_code_layer1 = torch.nn.Linear(self.shape_code_size, self.shape_code_size)

        self.layer_dir1 = torch.nn.Linear(self.dim_dir + self.shape_code_size, self.hidden_size)
        self.layer_dir2 = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.fc_rgb = torch.nn.Linear(self.hidden_size + self.texture_code_size, 3)

        self.activation = torch.nn.functional.relu

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, x: torch.Tensor):
        """ Forward function for NeRF Model

        :function:
            z_s: Shape Latent Embedding [sample_size x shape_code_size]
            z_t: Texture Latent Embedding [sample_size x texture_code_size]
            x: torch.Tensor [sample_size: dim_xyz + dim_dir]
        :returns: TODO

        """

        xyz = x[..., : self.dim_xyz]
        view = x[..., self.dim_xyz:]

        z_s_out = self.activation(self.shape_code_layer1(z_s))
        z_s_out2 = self.activation(self.shape_code_layer2(z_s))

        z_t_out = self.activation(self.texture_code_layer1(z_t))

        xyz_out = self.activation(self.layer_xyz1(xyz))
        xyz_out = torch.cat((xyz_out, z_s_out), dim=-1)
        xyz_out = self.activation(self.layer_xyz2(xyz_out))
        xyz_out = torch.cat((xyz_out, z_s_out2), dim=-1)

        feat = self.fc_out(xyz_out)

        sigma, feat = feat[..., :1], feat[..., 1:]

        view_in = torch.cat((feat, view), dim=-1)
        view_out = self.activation(self.layer_dir1(view_in))
        view_out = self.activation(self.layer_dir2(view_out))
        view_out = torch.cat((view_out, z_t_out), dim=-1)
        rgb = self.fc_rgb(view_out)

        return torch.cat((rgb, sigma), dim=-1)
