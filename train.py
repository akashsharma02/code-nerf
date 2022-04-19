from typing import Callable
import os
import time
from pathlib import Path

import hydra
import omegaconf
from dotenv import load_dotenv
from omegaconf import DictConfig
import numpy as np
import torch
import torch.multiprocessing as mp

import view_synthesis.models as nerf
import view_synthesis.utils as utils
from view_synthesis.utils import rank_zero_only
import torchmetrics

load_dotenv()

log = utils.get_logger(__name__)


def train(cfg: DictConfig) -> None:
    """
    Main training loop for the model

    :function:
        rank: Process rank. 0 == main process
    :returns: TODO

    """
    # Seed experiment for repeatability (Each process should sample different rays)
    device = torch.device('cuda', cfg.gpu)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = False  # Allow PyTorch to internally use tf32 for convolutions

    seed = cfg.experiment.randomseed * cfg.world_size + cfg.rank
    np.random.seed(seed)
    torch.manual_seed(seed)

    writer = utils.prepare_experiment(cfg)

    # Load Data
    log.info(f"Instantiating datamodule: {cfg.datamodule._target_}")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(rank=cfg.rank, shuffle=True, seed=seed)

    # Load model
    log.info(f"Instantiating model: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)

    # Load RaySampler
    sample = next(iter(datamodule.train_iterator()))
    height, width = sample["color"].shape[-2:]
    intrinsic = sample["intrinsic"][0]
    log.info(f"Instantiating Ray sampler: {cfg.ray_sampler._target_}")
    ray_sampler = hydra.utils.instantiate(cfg.ray_sampler, height=height, width=width, intrinsics=intrinsic)

    # Create optimizer
    log.info(f"Instantiating optimizer: {cfg.optim._target_}")
    optimization_params = [
        {"params": list(model.encoder.parameters()), 'lr': cfg.experiment.encoder_lr},
        {"params": list(model.decoder.parameters()), 'lr': cfg.experiment.decoder_lr}
    ]
    optimizer = hydra.utils.instantiate(cfg.optim, optimization_params)

    train_iter = 0
    if cfg.checkpoint_dir:
        train_iter, model, optimizer = utils.load_checkpoint(cfg, model, optimizer)

    model.train()
    model.to(device)
    for train_iter in range(train_iter, cfg.experiment.iterations):

        then = time.time()
        batch = next(datamodule.train_iterator())
        batch = utils.dict_to_device(batch, device)

        target_image = batch["color"][:, :3, ...]
        target_pose = batch["pose"][0]

        # Load a bundle of rays
        ray_bundle = ray_sampler.get_bundle(target_pose)
        for k in range(cfg.experiment.runs_per_image):
            optimizer.zero_grad()

            # Sample a bunch of rays
            rays, selected_ray_idxs = ray_sampler.sample(ray_bundle)

            # Predict rgb and depth
            rgb_per_ray, depth_per_ray = model(target_image, rays, selected_ray_idxs, target_pose)

            target_pixels = target_image.flatten(2, 3)
            target_pixels = target_pixels[..., selected_ray_idxs.squeeze()].squeeze().transpose(0, 1)

            loss = torch.nn.functional.mse_loss(rgb_per_ray, target_pixels)
            psnr = torchmetrics.functional.peak_signal_noise_ratio(rgb_per_ray, target_pixels)
            loss.backward()
            optimizer.step()

        if train_iter != 0 and train_iter % cfg.experiment.print_freq == 0:
            log_string = utils.log_losses(writer, "train", train_iter, time.time()-then, losses={"loss": loss.item(), "psnr": psnr})
            log.info(log_string)

        if train_iter != 0 and train_iter % cfg.experiment.validation_freq == 0:
            val_target_image, rgb_image, depth_image = validate(cfg, datamodule, model, ray_sampler, device)

            val_target_image = val_target_image.squeeze().permute(1, 2, 0)
            mse = torchmetrics.functional.mean_squared_error(rgb_image, val_target_image)
            psnr = torchmetrics.functional.peak_signal_noise_ratio(rgb_image, val_target_image)
            utils.log_losses(writer, "val", train_iter, time.time() - then, losses={"mse": mse, "psnr": psnr})
            writer.add_image("val/target_image", val_target_image, train_iter, dataformats="HWC")
            writer.add_image("val/rgb_image", rgb_image, train_iter, dataformats="HWC")
            writer.add_image("val/depth_image", depth_image, train_iter, dataformats="HWC")

        if train_iter != 0 and train_iter % cfg.experiment.save_freq == 0:
            ckpt_dict = {
                "iter": train_iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            torch.save(ckpt_dict, Path(writer.log_dir) / f"checkpoint{train_iter}.ckpt")


def validate(cfg: DictConfig,
             datamodule: Callable,
             model: torch.nn.Module,
             ray_sampler: nerf.RaySampler,
             device: torch.device
             ):
    batch = next(datamodule.val_iterator())
    batch = utils.dict_to_device(batch, device)
    target_image = batch["color"][:, :3, ...]
    im_height, im_width = target_image.shape[-2], target_image.shape[-1]
    target_pose = batch["pose"][0]

    ray_bundle = ray_sampler.get_bundle(world_T_camera=target_pose)
    ray_origins, ray_directions = ray_bundle.origins.reshape(-1, 3), ray_bundle.directions.reshape(-1, 3)
    ray_origins = utils.batchify(ray_origins, cfg.ray_sampler.num_samples)
    ray_directions = utils.batchify(ray_directions, cfg.ray_sampler.num_samples)

    rgb_per_image, depth_per_image = [], []
    with torch.no_grad():
        for i, (ro, rd) in enumerate(zip(ray_origins, ray_directions)):
            start_idx = i*cfg.ray_sampler.num_samples
            selected_ray_idxs = np.arange(start_idx, start_idx + ro.shape[0])
            rgb_per_ray, depth_per_ray = model(target_image, nerf.Rays(ro, rd), selected_ray_idxs, target_pose)
            rgb_per_image.append(rgb_per_ray)
            depth_per_image.append(depth_per_ray)

    rgb_per_image = torch.cat(rgb_per_image, dim=0)
    depth_per_image = torch.cat(depth_per_image, dim=0)

    rgb_image = rgb_per_image.reshape([im_height, im_width, -1])
    depth_image = depth_per_image.reshape([im_height, im_width, -1])

    return target_image, rgb_image, depth_image


def init_distributed_mode(rank, args):
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        args.world_size = args.gpus
    args.rank = rank
    args.gpu = rank

    if 'RANK' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.gpu = int(os.environ['LOCAL_RANK'])

    if args.world_size == 1:
        return

    if 'MASTER_ADDR' in os.environ:
        args.dist_url = 'tcp://{}:{}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])

    print(f'gpu={args.gpu}, rank={args.rank}, world_size={args.world_size}')
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    with omegaconf.open_dict(args):
        args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)

    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()


def subprocess_fn(rank: int, args: DictConfig):
    """TODO: Docstring for init_process.

    :function: TODO
    :returns: TODO

    """
    init_distributed_mode(rank, args)
    train(args)


def setup_training_kwargs(cfg: DictConfig):
    """
    """
    with omegaconf.open_dict(cfg):
        cfg.rank = 0
        cfg.gpu = 0
        cfg.gpus = torch.cuda.device_count() if cfg.gpus is None else cfg.gpus
        cfg.distributed = True if cfg.distributed and cfg.gpus > 1 else False
        cfg.world_size = 1


@ hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """ Main function setting up the training loop

    :function: TODO
    :returns: TODO

    """
    setup_training_kwargs(cfg)
    log.info(f"Working dir: {os.getcwd()}")
    log.info('Launching processes...')
    if cfg.distributed:
        mp.spawn(subprocess_fn, args=(cfg,), nprocs=cfg.gpus)
    else:
        subprocess_fn(rank=0, args=cfg)


if __name__ == "__main__":
    main()
