from typing import List, Tuple, Union
from types import FunctionType
import os
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from itertools import islice

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from view_synthesis.cfgnode import CfgNode
from view_synthesis import utils
import view_synthesis.nerf as nerf
from view_synthesis.nerf import RaySampler, PointSampler, PositionalEmbedder
torch.set_printoptions(sci_mode=False)


def pose_spherical(theta: torch.Tensor, phi: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """
    Generates a camera pose viewing the object at origin, where the camera lies on a S^2 sphere facing the object.

    Args:
        theta: azimuth angle
        phi: elevation angle
        rho: radius of the sphere
    Returns:
        T_c2w: SE3 transformation from camera to world (4x4 matrix)
    """
    c2w = torch.eye(n=4, device=theta.device)
    c2w[0, 0], c2w[1, 0] = -torch.sin(phi), torch.cos(phi)
    c2w[0, 1], c2w[1, 1], c2w[2, 1] = -torch.sin(theta) * torch.cos(phi), -torch.sin(theta) * torch.sin(phi), torch.cos(theta)
    c2w[0, 2], c2w[1, 2], c2w[2, 2] = torch.cos(theta) * torch.cos(phi), torch.cos(theta) * torch.sin(phi), torch.sin(theta)
    c2w[0, 3], c2w[1, 3], c2w[2, 3] = rho * torch.cos(theta) * torch.cos(phi), rho * torch.cos(theta) * torch.sin(phi), rho * torch.sin(theta)
    return c2w


def eval(rank: int, cfg: CfgNode) -> None:
    """
    Implements the test-time optimization method that optimizes the latent code and camera parameters.
    """
    # Seed experiment for repeatability (Each process should sample different rays)
    seed = (rank + 1) + cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set device and logdir_path
    logdir_path, writer = None, None
    if utils.is_main_process(cfg.is_distributed):
        logdir_path = utils.prepare_experiment(cfg)
        writer = SummaryWriter(logdir_path)

    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)

    # Load Data
    dataloader, dataset = utils.prepare_dataloader("val", cfg)
    _, train_dataset = utils.prepare_dataloader("train", cfg)

    # Prepare Model, Optimizer, and load checkpoint
    models = utils.prepare_models(cfg, train_dataset.num_objects)
    optimizer, scheduler = utils.prepare_optimizer(cfg, models)
    _ = utils.load_checkpoint(cfg, models, optimizer)

    # Prepare RaySampler
    first_data_sample = next(iter(dataloader))
    (height, width), intrinsic, datatype = first_data_sample["color"][0].shape[:2], first_data_sample["intrinsic"][0], first_data_sample["intrinsic"][0].dtype

    # Prepare RaySampler and PointSampler
    samplers = nerf.prepare_samplers(cfg, height, width, intrinsic, datatype, device)
    # Prepare Positional Embedding functions
    embedders = nerf.prepare_embedders(cfg, datatype, device)

    total_load_iterations = cfg.experiment.iterations // cfg.dataset.val_batch_size
    for iteration in range(0, total_load_iterations):
        validate(cfg, iteration, dataloader, models, samplers, embedders, writer, device)


def validate(cfg: CfgNode,
             iteration: int,
             dataloader: torch.utils.data.DataLoader,
             models: "OrderedDict[torch.nn.Module, torch.nn.Module]",
             samplers: Tuple[RaySampler, PointSampler],
             embedders: List[Union[PositionalEmbedder, None]],
             writer: SummaryWriter,
             device: torch.cuda.Device
             ) -> Tuple[float, float, float]:
    """
    Validation loop for Code-NeRF
    1. Run a test-time optimization to estimate the best latent code with fixed weights.
    2. Use the optimized latent codes to render a novel view.

    Args:
        Self-explanatory
    Returns:
        None

    """
    ray_sampler, point_sampler = samplers
    if cfg.is_distributed:
        dataloader.sampler.set_epoch(iteration)

    # Load data independently in all processes as a list of tuples
    # Required since broadcast_object_list requires that each process provides an object list of same size
    val_iterator = iter(dataloader)
    val_data = next(islice(val_iterator, 5, None))

    # Broadcast validation data in rank 0 to all the processes
    if cfg.is_distributed:
        val_data = list(val_data.items())
        torch.distributed.broadcast_object_list(val_data, 0)
        val_data = dict(val_data)

    for key, val in val_data.items():
        if torch.is_tensor(val):
            val_data[key] = val_data[key].to(device, non_blocking=True)

    if cfg.is_distributed:
        all_shape_embedding, all_texture_embedding = models["embedding"].module.get_all_embeddings(device=device)
    else:
        all_shape_embedding, all_texture_embedding = models["embedding"].get_all_embeddings(device=device)

    shape_embedding = all_shape_embedding.mean(dim=0, keepdim=True).clone().detach().requires_grad_(True)
    texture_embedding = all_texture_embedding.mean(dim=0, keepdim=True).clone().detach().requires_grad_(True)

    theta = torch.Tensor([1.57]).to(device).requires_grad_(True)
    phi = torch.Tensor([0]).to(device).requires_grad_(True)
    rho = torch.Tensor([1.30]).to(device).requires_grad_(True)

    optimizer = getattr(torch.optim, cfg.optimizer.val_type)([
        {'params': [shape_embedding, texture_embedding]},
        {'params': [theta, phi], 'lr': cfg.optimizer.angle_lr},
        {'params': [rho], 'lr': cfg.optimizer.radius_lr},
    ], lr=cfg.optimizer.val_lr,
    )

    for val_iter in range(0, cfg.experiment.val_iterations):
        val_then = time.time()
        for _, model in models.items():
            model.train()

        cam_pose = pose_spherical(theta, phi, rho)[None, :]
        ro, rd, select_inds = ray_sampler.sample(tform_cam2world=cam_pose)
        target_pixels = val_data["color"].flatten(1, 2)
        target_pixels = target_pixels[..., select_inds, :].squeeze()
        z_s, z_t = shape_embedding.expand(ro.shape[0], -1), texture_embedding.expand(ro.shape[0], -1)

        # Pass through NeRF
        latent_embedding, rays = (z_s, z_t), (ro, rd)
        rgb_coarse, rgb_fine = nerf.predict_radiance_and_render(
            rays, point_sampler, embedders, models["nerf_coarse"], models["nerf_fine"], latent_embedding)

        # Compute losses
        nerf_loss_coarse = torch.nn.functional.mse_loss(rgb_coarse[..., :3], target_pixels[..., :3])
        nerf_loss_fine = torch.nn.functional.mse_loss(rgb_fine[..., :3], target_pixels[..., :3])
        psnr = utils.mse2psnr(nerf_loss_fine.item())
        embedding_regularization = cfg.experiment.regularizer_lambda * (torch.norm(z_s, p=2) + torch.norm(z_t, p=2))
        tform_cam2gt = torch.matmul(torch.inverse(val_data["pose"]), cam_pose)
        pose_error = torch.norm(utils.SE3.Log(tform_cam2gt), p=2)
        # TODO: Should not add pose_error...... (assumes available ground truth)
        loss = nerf_loss_coarse + nerf_loss_fine + embedding_regularization  # + 0.01 * pose_error

        # Backprop and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if utils.is_main_process(cfg.is_distributed):
            if val_iter % cfg.experiment.val_print_every == 0 or val_iter == cfg.experiment.val_iterations-1:
                losses_dict = {"nerf_loss_coarse": nerf_loss_coarse.item(),
                               "nerf_loss_fine": nerf_loss_fine.item(),
                               "embedding_loss": embedding_regularization.item(),
                               "pose_error": pose_error,
                               "total_loss": loss.item(),
                               "psnr": psnr}
                log_iter = iteration*cfg.experiment.val_iterations + val_iter
                log_string = utils.log_losses(writer, "val-optim", log_iter, time.time()-val_then, losses_dict)
                print(log_string)

                render_then = time.time()
                rgb = nerf.parallel_image_render(cfg,
                                                 cam_pose,
                                                 [shape_embedding, texture_embedding],
                                                 models,
                                                 (ray_sampler, point_sampler),
                                                 embedders,
                                                 device)

                assert rgb is not None, "Main process must contain rgb"

                target_pixels = val_data["color"].view(-1, 4)
                loss = torch.nn.functional.mse_loss(rgb[..., : 3], target_pixels[..., : 3])
                psnr = utils.mse2psnr(loss.item())

                target_rgb = target_pixels.reshape(list(val_data["color"].shape[: -1]) + [4])
                rgb = rgb.reshape(list(val_data["color"].shape[: -1]) + [rgb.shape[-1]])

                render_losses_dict = {"loss": loss, "psnr": psnr}
                log_string = utils.log_losses(writer, "val", log_iter, time.time()-render_then, render_losses_dict)
                writer.add_images("val/rgb_image", rgb[..., : 3], log_iter, dataformats='NHWC')
                writer.add_images("val/target_image", target_rgb[..., : 3], log_iter, dataformats='NHWC')
                print(log_string)


def init_process(rank: int, fn: FunctionType, cfg: CfgNode, backend: str = "gloo"):
    """TODO: Docstring for init_process.

    :function: TODO
    :returns: TODO

    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank,  world_size=cfg.gpus)
    fn(rank, cfg)
    torch.distributed.destroy_process_group()


def main(cfg: CfgNode):
    """ Main function setting up the training loop

    :function: TODO
    :returns: TODO

    """
    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _, device_ids = utils.prepare_device(cfg.gpus, cfg.is_distributed)

    if len(device_ids) > 1 and configargs.is_distributed:
        # TODO: Setup DataDistributedParallel
        print(f"Using {len(device_ids)} GPUs for training")
        mp.spawn(init_process, args=(eval, cfg, "nccl"),
                 nprocs=cfg.gpus, join=True)
    else:
        eval(0, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        required=True,
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='Number of gpus per node')
    parser.add_argument("--distributed", action='store_true', dest="is_distributed",
                        help="Run the models in DataDistributedParallel")
    configargs = parser.parse_args()

    # Read config file.
    cfg = CfgNode(vars(configargs), new_allowed=True)
    cfg.merge_from_file(configargs.config)

    main(cfg)
