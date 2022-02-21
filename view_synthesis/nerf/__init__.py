from typing import List, Union, Tuple, OrderedDict
from numpy.typing import DTypeLike
from .ray_sampler import RaySampler
from .point_sampler import PointSampler
from .position_embed import PositionalEmbedder
from .volumetric_render import volume_render

import torch
import numpy as np
import torch.distributed as dist
from ..cfgnode import CfgNode
from ..utils import util


def prepare_samplers(cfg: CfgNode,
                     height: int,
                     width: int,
                     intrinsics: np.ndarray,
                     datatype: DTypeLike,
                     device: torch.cuda.Device
                     ) -> Tuple[RaySampler, PointSampler]:
    """
    Load the RaySampler and PointSampler objects
    """
    ray_sampler = RaySampler(height,
                             width,
                             intrinsics,
                             sample_size=cfg.nerf.ray_sampler.num_random_rays,
                             device=device)
    point_sampler = PointSampler(cfg.nerf.point_sampler.num_coarse,
                                 cfg.nerf.point_sampler.num_fine,
                                 cfg.nerf.point_sampler.near_limit,
                                 cfg.nerf.point_sampler.far_limit,
                                 spacing_mode=cfg.nerf.point_sampler.spacing_mode,
                                 perturb=cfg.nerf.point_sampler.perturb,
                                 dtype=datatype,
                                 device=device)
    return ray_sampler, point_sampler


def prepare_embedders(cfg: CfgNode,
                      datatype: DTypeLike,
                      device: torch.cuda.Device
                      ) -> Tuple[PositionalEmbedder, Union[PositionalEmbedder, None]]:
    """
    Load the embedding functions for points and viewdirs from config file

    Args:
        cfg: CfgNode object
        datatype: numpy datatype (float32 or float16)
        device: torch cuda Device ID
    Returns:
        embedding objects, that can positionally embed (with fourier frequencies) xyz and directions

    """
    embedder_xyz = PositionalEmbedder(num_freq=cfg.nerf.embedder.num_encoding_fn_xyz,
                                      log_sampling=cfg.nerf.embedder.log_sampling_xyz,
                                      include_input=cfg.nerf.embedder.include_input_xyz,
                                      dtype=datatype,
                                      device=device)

    embedder_dir = None
    if cfg.nerf.embedder.use_viewdirs:
        embedder_dir = PositionalEmbedder(num_freq=cfg.nerf.embedder.num_encoding_fn_dir,
                                          log_sampling=cfg.nerf.embedder.log_sampling_dir,
                                          include_input=cfg.nerf.embedder.include_input_dir,
                                          dtype=datatype,
                                          device=device)

    return embedder_xyz, embedder_dir


def predict_radiance_and_render(rays: Tuple[torch.Tensor, torch.Tensor],
                                point_sampler: PointSampler,
                                embedders: List[Union[PositionalEmbedder, None]],
                                coarse_model: torch.nn.Module,
                                fine_model: torch.nn.Module,
                                latent_embedding: Tuple[torch.Tensor, torch.Tensor]
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    (ro, rd) = rays
    pts_coarse, z_vals_coarse = point_sampler.sample_uniform(ro, rd)
    radiance_field_coarse = forward_pass(coarse_model, embedders, rd, pts_coarse, latent_embedding)
    (rgb_coarse, _, _, weights, _) = volume_render(radiance_field_coarse, z_vals_coarse, rd)

    # Pass through nerf_fine model
    pts_fine, z_vals_fine = point_sampler.sample_pdf(ro, rd, weights[..., 1:-1], z_vals_coarse)
    radiance_field_fine = forward_pass(fine_model, embedders, rd, pts_fine, latent_embedding)
    (rgb_fine, _, _, _, _) = volume_render(radiance_field_fine, z_vals_fine, rd)

    return rgb_coarse, rgb_fine


def forward_pass(model: torch.nn.Module,
                 embedders: List[Union[PositionalEmbedder, None]],
                 rd: Union[torch.Tensor, None],
                 pts: torch.Tensor,
                 object_embedding: Tuple[torch.Tensor, torch.Tensor]
                 ) -> torch.Tensor:
    """
    One forward pass through NeRF given a batch of rays:
        1. embed sampled points
        2. MLP model forward

    Args:
        model: NeRF MLP model
        embedders: List of embedding functions
        rd: torch.Tensor ray directions  chunksize x 3 | None, if none embedder_dir is not used
        pts: torch.Tensor chunksize x num_samples x 3
        object_embedding: torch.Tensor containing indexed latent codes

    Returns:
        Radiance field: (chunksize x num_samples) x 4

    """
    num_rays, num_samples = pts.shape[0], pts.shape[1]
    [shape_embedding, texture_embedding] = object_embedding
    shape_embedding, texture_embedding = shape_embedding[:, None, :].expand(-1, num_samples, -1), texture_embedding[:, None, :].expand(-1, num_samples, -1)
    shape_embedding, texture_embedding = shape_embedding.reshape(-1, shape_embedding.shape[-1]), texture_embedding.reshape(-1, texture_embedding.shape[-1])
    pts_flat = pts.reshape(-1, pts.shape[-1])

    embedder_xyz, embedder_dir = embedders
    embedded = embedder_xyz.embed(pts_flat)
    if rd is not None:
        viewdirs = rd
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        input_dirs = viewdirs.repeat([1, num_samples, 1])
        input_dirs_flat = input_dirs.reshape(-1, input_dirs.shape[-1])
        embedded_dirs = embedder_dir.embed(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    radiance_field = model(shape_embedding, texture_embedding, embedded)
    radiance_field = radiance_field.reshape([num_rays, num_samples, radiance_field.shape[-1]])
    return radiance_field


def parallel_image_render(cfg: CfgNode,
                          pose: torch.Tensor,
                          object_embedding: Tuple[torch.Tensor, torch.Tensor],
                          models: "OrderedDict[torch.nn.Module, torch.nn.Module]",
                          samplers: Tuple[RaySampler, PointSampler],
                          embedders: List[Union[PositionalEmbedder, None]],
                          device: torch.cuda.Device
                          ) -> torch.Tensor:
    """
    Parallely render images on multiple GPUs for validation

    Args:
        cfg: CfgNode object
        pose: Camera pose at which to render novel view (batch x 4 x 4)
        object_embedding: Optimized shape and texture embedding to render image with
        models: NN models for forward pass
        samplers:
        embedders:
        device:
    Returns:
        rendered image: torch.Tensor height*width x 4
    """

    rank = 0
    if cfg.is_distributed:
        rank = dist.get_rank()

    for _, model in models.items():
        model.eval()

    with torch.no_grad():
        ray_sampler, point_sampler = samplers
        ray_origins, ray_directions = ray_sampler.get_bundle(tform_cam2world=pose)
        ro, rd = ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3)
        num_rays = ro.shape[0]

        [shape_embedding, texture_embedding] = object_embedding
        shape_embedding = shape_embedding.expand(num_rays, -1)
        texture_embedding = texture_embedding.expand(num_rays, -1)

        batchsize_per_process = torch.full([cfg.gpus], (num_rays / cfg.gpus), dtype=int)
        padding = num_rays - torch.sum(batchsize_per_process)
        batchsize_per_process[-1] = num_rays - torch.sum(batchsize_per_process[: -1])
        assert torch.sum(batchsize_per_process) == num_rays, "Mismatch in batchsize per process and total number of rays"

        padding_per_process = torch.zeros([cfg.gpus], dtype=int)
        if padding > 0:
            padding_per_process[: -1] = padding
        assert padding + batchsize_per_process[0] == batchsize_per_process[-1], "Incorrect calculation of padding"

        # Only use the split of the rays for the current process
        ro_batch = torch.split(ro, batchsize_per_process.tolist())[rank].to(device)
        rd_batch = torch.split(rd, batchsize_per_process.tolist())[rank].to(device)
        shape_embedding_batch = torch.split(shape_embedding, batchsize_per_process.tolist())[rank].to(device)
        texture_embedding_batch = torch.split(texture_embedding, batchsize_per_process.tolist())[rank].to(device)

        # Minibatch the rays allocated to current process
        ro_minibatches = util.get_minibatches(ro_batch, cfg.nerf.validation.chunksize)
        rd_minibatches = util.get_minibatches(rd_batch, cfg.nerf.validation.chunksize)
        shape_embedding_minibatches = util.get_minibatches(shape_embedding_batch, cfg.nerf.validation.chunksize)
        texture_embedding_minibatches = util.get_minibatches(texture_embedding_batch, cfg.nerf.validation.chunksize)

        rgb_batches = []
        for ro, rd, z_s, z_t in zip(ro_minibatches, rd_minibatches, shape_embedding_minibatches, texture_embedding_minibatches):
            # Pass through NeRF model
            latent_embedding, rays = (z_s, z_t), (ro, rd)
            _, rgb_fine = predict_radiance_and_render(rays, point_sampler, embedders, models["nerf_coarse"], models["nerf_fine"], latent_embedding)
            rgb_batches.append(rgb_fine)
        rgb_batches = torch.cat(rgb_batches, dim=0)

        if not cfg.is_distributed:
            return rgb_batches

        # Pad image chunks to get equal chunksize for all_gather/gather
        padded_rgb = torch.zeros((padding_per_process[rank], rgb_batches.shape[-1]),
                                 dtype=rgb_batches.dtype,
                                 device=rgb_batches.device)
        rgb_batches = torch.cat([rgb_batches, padded_rgb], dim=0)
        all_rgb_batches = [torch.zeros_like(rgb_batches) for _ in range(cfg.gpus)]
        torch.distributed.all_gather(all_rgb_batches, rgb_batches)

        if util.is_main_process(cfg.is_distributed):
            for i, size in enumerate(batchsize_per_process):
                all_rgb_batches[i] = all_rgb_batches[i][: size, ...]
            all_rgb_batches = torch.cat(all_rgb_batches, dim=0)

            return all_rgb_batches
        else:
            return None
