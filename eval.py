from typing import List, Tuple, Union
from types import FunctionType
import os
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from view_synthesis.cfgnode import CfgNode
from view_synthesis import utils
import view_synthesis.nerf as nerf
from view_synthesis.nerf import RaySampler, PointSampler, PositionalEmbedder


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

    # Prepare Model, Optimizer, and load checkpoint
    models = utils.prepare_models(cfg, dataset.num_objects)
    optimizer, scheduler = utils.prepare_optimizer(cfg, models)
    start_iter = utils.load_checkpoint(cfg, models, optimizer)

    # Prepare RaySampler
    first_data_sample = next(iter(dataloader))
    (height, width), intrinsic, datatype = first_data_sample["color"][0].shape[:2], first_data_sample["intrinsic"][0], first_data_sample["intrinsic"][0].dtype

    # Prepare RaySampler and PointSampler
    samplers = nerf.prepare_samplers(cfg, height, width, intrinsic, datatype, device)
    # Prepare Positional Embedding functions
    embedders = nerf.prepare_embedders(cfg, datatype, device)

    total_load_iterations = cfg.experiment.iterations // cfg.dataset.train_batch_size
    for iteration in range(start_iter, total_load_iterations):
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
    val_data = next(iter(dataloader))

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

    # TODO: Camera pose optimization
    optimizer = getattr(torch.optim, cfg.optimizer.val_type)(
        [shape_embedding, texture_embedding],
        lr=cfg.optimizer.val_lr,
    )

    for val_iter in range(0, cfg.experiment.val_iterations):
        val_then = time.time()
        for _, model in models.items():
            model.eval()

        ro_batch, rd_batch, select_inds = ray_sampler.sample(tform_cam2world=val_data["pose"])
        tgt_pixel_batch = val_data["color"].flatten(1, 2)
        shape_embedding_batch, texture_embedding_batch = shape_embedding.expand(ro_batch.shape[0], -1), texture_embedding.expand(ro_batch.shape[0], -1)
        tgt_pixel_batch = [tgt_pixel_batch[k, select_inds[k], :] for k in range(cfg.dataset.val_batch_size)]
        tgt_pixel_batch = torch.cat(tgt_pixel_batch, dim=0)

        msg = "Chunksize needs to atleast be less than to the number of rays sampled from a single image"
        assert cfg.nerf.validation.chunksize <= cfg.nerf.ray_sampler.num_random_rays * cfg.dataset.val_batch_size, msg
        ro_minibatches, rd_minibatches = utils.get_minibatches(
            ro_batch, cfg.nerf.validation.chunksize), utils.get_minibatches(rd_batch, cfg.nerf.validation.chunksize)
        tgt_pixel_minibatches = utils.get_minibatches(tgt_pixel_batch, cfg.nerf.validation.chunksize)
        z_s_minibatches = utils.get_minibatches(shape_embedding_batch, cfg.nerf.validation.chunksize)
        z_t_minibatches = utils.get_minibatches(texture_embedding_batch, cfg.nerf.validation.chunksize)

        num_batches = len(ro_minibatches)
        msg = "Mismatch in batch length of ray origins, ray directions and target pixels"
        assert num_batches == len(rd_minibatches) == len(tgt_pixel_minibatches), msg

        for j, (ro, rd, target_pixels, z_s, z_t) in enumerate(zip(ro_minibatches, rd_minibatches, tgt_pixel_minibatches, z_s_minibatches, z_t_minibatches)):

            # Pass through NeRF
            latent_embedding, rays = (z_s, z_t), (ro, rd)
            rgb_coarse, rgb_fine = nerf.predict_radiance_and_render(
                rays, point_sampler, embedders, models["nerf_coarse"], models["nerf_fine"], latent_embedding)

            # Compute losses
            nerf_loss_coarse = torch.nn.functional.mse_loss(rgb_coarse[..., :3], target_pixels[..., :3])
            nerf_loss_fine = torch.nn.functional.mse_loss(rgb_fine[..., :3], target_pixels[..., :3])
            psnr = utils.mse2psnr(nerf_loss_fine.item())
            embedding_regularization = cfg.experiment.regularizer_lambda * (torch.norm(z_s, p=2) + torch.norm(z_t, p=2))
            loss = nerf_loss_coarse + nerf_loss_fine + embedding_regularization

            # Backprop and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if utils.is_main_process(cfg.is_distributed):
            if val_iter > 0 and val_iter % cfg.experiment.val_print_every == 0:
                losses_dict = {"nerf_loss_coarse": nerf_loss_coarse.item(),
                               "nerf_loss_fine": nerf_loss_fine.item(),
                               "embedding_loss": embedding_regularization.item(),
                               "total_loss": loss.item(),
                               "psnr": psnr}
                log_string = utils.log_losses(writer, "val-optim", iteration+val_iter, time.time()-val_then, losses_dict)
                writer.add_images("val_optimization/target_image", val_data["color"][..., :3], iteration+val_iter, dataformats='NHWC')
                print(log_string)

    render_then = time.time()
    rgb = nerf.parallel_image_render(cfg,
                                     val_data["pose"],
                                     [shape_embedding, texture_embedding],
                                     models,
                                     (ray_sampler, point_sampler),
                                     embedders,
                                     device)

    if utils.is_main_process(cfg.is_distributed):
        assert rgb is not None, "Main process must contain rgb"

        target_pixels = val_data["color"].view(-1, 4)
        loss = torch.nn.functional.mse_loss(rgb[..., : 3], target_pixels[..., : 3])
        psnr = utils.mse2psnr(loss.item())

        target_rgb = target_pixels.reshape(list(val_data["color"].shape[: -1]) + [4])
        rgb = rgb.reshape(list(val_data["color"].shape[: -1]) + [rgb.shape[-1]])

        render_losses_dict = {"loss": loss, "psnr": psnr}
        log_string = utils.log_losses(writer, "val", iteration, time.time()-render_then, render_losses_dict)
        writer.add_images("val/rgb_image", rgb[..., : 3], iteration, dataformats='NHWC')
        writer.add_images("val/target_image", target_rgb[..., : 3], iteration, dataformats='NHWC')
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
