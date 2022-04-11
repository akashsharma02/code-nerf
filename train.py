from types import FunctionType
import os
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from view_synthesis.cfgnode import CfgNode
import view_synthesis.models as network_arch
import view_synthesis.utils as utils
import view_synthesis.nerf as nerf
from eval import validate


def train(rank: int, cfg: CfgNode) -> None:
    """
    Main training loop for the model

    :function:
        rank: Process rank. 0 == main process
    :returns: TODO

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
    train_dataloader, train_dataset = utils.prepare_dataloader("train", cfg)
    val_dataloader, val_dataset = utils.prepare_dataloader("val", cfg)

    # Prepare Model, Optimizer, and load checkpoint
    models = utils.prepare_models(cfg, train_dataset.num_objects)
    optimizer, scheduler = utils.prepare_optimizer(cfg, models)
    start_iter = utils.load_checkpoint(cfg, models, optimizer)

    first_data_sample = next(iter(train_dataloader))
    (height, width), intrinsic, datatype = first_data_sample["color"][0].shape[:2], first_data_sample["intrinsic"][0], first_data_sample["intrinsic"][0].dtype

    # Prepare RaySampler and PointSampler
    ray_sampler, point_sampler = nerf.prepare_samplers(cfg, height, width, intrinsic, datatype, device)

    # Prepare Positional Embedding functions
    embedders = nerf.prepare_embedders(cfg, datatype, device)

    i = 0
    total_load_iterations = cfg.experiment.iterations // cfg.dataset.train_batch_size
    for iteration in range(start_iter, total_load_iterations):

        for _, model in models.items():
            model.train()

        if cfg.is_distributed:
            train_dataloader.sampler.set_epoch(iteration)

        train_data = next(iter(train_dataloader))
        for key, value in train_data.items():
            if torch.is_tensor(value):
                train_data[key] = train_data[key].to(device, non_blocking=True)

        # Load a sample of rays and target pixels from data
        ro_batch, rd_batch, select_inds = ray_sampler.sample(tform_cam2world=train_data["pose"])
        tgt_pixel_batch = train_data["color"].flatten(1, 2)
        tgt_object_ids_batch = train_data["object_id"][:, None].expand(-1, cfg.nerf.ray_sampler.num_random_rays)
        tgt_pixel_batch = [tgt_pixel_batch[k, select_inds[k], :] for k in range(cfg.dataset.train_batch_size)]
        tgt_pixel_batch, tgt_object_ids_batch = torch.cat(tgt_pixel_batch, dim=0), tgt_object_ids_batch.reshape(-1)

        msg = "Chunksize needs to atleast be less than to the number of rays sampled from a single image"
        assert cfg.nerf.train.chunksize <= cfg.nerf.ray_sampler.num_random_rays * cfg.dataset.train_batch_size, msg
        ro_minibatches, rd_minibatches = utils.get_minibatches(ro_batch, cfg.nerf.train.chunksize), utils.get_minibatches(rd_batch, cfg.nerf.train.chunksize)
        tgt_pixel_minibatches = utils.get_minibatches(tgt_pixel_batch, cfg.nerf.train.chunksize)
        tgt_object_ids_minibatches = utils.get_minibatches(tgt_object_ids_batch, cfg.nerf.train.chunksize)

        num_batches = len(ro_minibatches)
        msg = "Mismatch in batch length of ray origins, ray directions and target pixels"
        assert num_batches == len(rd_minibatches) == len(tgt_pixel_minibatches) == len(tgt_object_ids_minibatches), msg

        for j, (ro, rd, target_object_ids, target_pixels) in enumerate(zip(ro_minibatches, rd_minibatches, tgt_object_ids_minibatches, tgt_pixel_minibatches)):
            then = time.time()

            # Pass through NeRF
            target_object_embedding, rays = models["embedding"](target_object_ids), (ro, rd)
            rgb_coarse, rgb_fine = nerf.predict_radiance_and_render(rays,
                                                                    point_sampler,
                                                                    embedders,
                                                                    models["nerf_coarse"],
                                                                    models["nerf_fine"] if "nerf_fine" in models else None,
                                                                    target_object_embedding)
            # Compute losses
            nerf_loss_coarse = torch.nn.functional.mse_loss(rgb_coarse[..., :3], target_pixels[..., :3])
            nerf_loss_fine = 0.0
            psnr = utils.mse2psnr(nerf_loss_coarse.item())
            if rgb_fine:
                nerf_loss_fine = torch.nn.functional.mse_loss(rgb_fine[..., :3], target_pixels[..., :3])
                psnr = utils.mse2psnr(nerf_loss_fine.item())
            shape_params, texture_params = network_arch.get_params_tensor(models["embedding"], cfg.is_distributed)
            embedding_regularization = cfg.experiment.regularizer_lambda * (torch.norm(shape_params, p=2) + torch.norm(texture_params, p=2))
            loss = nerf_loss_coarse + nerf_loss_fine + embedding_regularization

            # Backprop and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            i = iteration * num_batches + j
            if utils.is_main_process(cfg.is_distributed) and i > 0:
                if i % cfg.experiment.print_every == 0:
                    losses_dict = {"nerf_loss_coarse": nerf_loss_coarse.item()}
                    if rgb_fine:
                        losses_dict["nerf_loss_fine"] = nerf_loss_fine.item()
                    losses_dict["embedding_loss"] = embedding_regularization.item()
                    losses_dict["total_loss"] = loss.item()
                    losses_dict["psnr"] = psnr
                    log_string = utils.log_losses(writer, "train", i, time.time()-then, losses_dict, scheduler.get_last_lr()[0])
                    writer.add_images("train/target_image", train_data["color"][..., :3], i, dataformats='NHWC')
                    print(log_string)

                if (i % cfg.experiment.save_every == 0 or i == cfg.experiment.iterations - 1):
                    checkpoint_dict = {
                        "iter": iteration,
                        "model_nerf_coarse_state_dict": models["nerf_coarse"].state_dict()}
                    if "nerf_fine" in models:
                        checkpoint_dict["model_nerf_fine_state_dict"] = models["nerf_fine"].state_dict()
                    checkpoint_dict["model_embedding_state_dict"] = models["embedding"].state_dict()
                    checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()
                    torch.save(checkpoint_dict, logdir_path / f"checkpoint{i:5d}.ckpt")
                    print("================== Saved Checkpoint =================")

            # Parallel rendering of image for Validation
            if (i > 0 and i % cfg.experiment.validate_every == 0):
                validate(cfg, i, val_dataloader, models, [ray_sampler, point_sampler], embedders, writer, device)


def init_process(rank: int, fn: FunctionType, cfg: CfgNode, backend: str = "gloo"):
    """TODO: Docstring for init_process.

    :function: TODO
    :returns: TODO

    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
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
        mp.spawn(init_process, args=(train, cfg, "nccl"),
                 nprocs=cfg.gpus, join=True)
    else:
        train(0, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
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
