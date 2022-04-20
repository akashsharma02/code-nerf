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

import src.models as nerf
import src.utils as utils
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
    # TODO: Remove this later
    torch.backends.cudnn.enabled = False    # Improves training speed.
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
    val_iterator = iter(datamodule.val_dataloader())
    for train_iter, batch in enumerate(datamodule.train_dataloader()):

        then = time.time()
        rays = batch["rays"].to(device)
        target_rgb = batch["rgb"].to(device)
        target_image = batch["rgb_image"].to(device)
        batchsize = rays.shape[0]
        loss_per_batch = 0.0
        for batchnum in range(0,  batchsize):

            ray_batch = utils.batchify(rays[batchnum], cfg.num_rays)
            target_rgb_batch = utils.batchify(target_rgb[batchnum], cfg.num_rays)
            target_image_batch = target_image[batchnum]
            height, width = target_image_batch.shape[-2], target_image_batch.shape[-1]

            loss_per_image, pred_image = [], []
            for i, (ray, rgb) in enumerate(zip(ray_batch, target_rgb_batch)):
                optimizer.zero_grad()
                pred_rgb, pred_depth = model(target_image_batch[None, ...], ray)
                loss = torch.nn.functional.mse_loss(pred_rgb, rgb)
                loss_per_image.append(loss.item())
                pred_image.append(pred_rgb)
                loss.backward()
                optimizer.step()

            pred_image = torch.cat(pred_image, dim=0)
            loss_per_image = np.mean(loss_per_image)
            pred_image = pred_image.reshape([height, width, -1])
            writer.add_image("train/rgb_image", pred_image, train_iter, dataformats="HWC")
            writer.add_image("train/target", target_image_batch, train_iter, dataformats="CHW")
            loss_per_batch += loss_per_image

        loss_per_batch = loss_per_batch/batchsize
        log_string = utils.log_losses(writer, "train", train_iter, time.time()-then,
                                      losses={"avg_loss": loss_per_batch, "avg_psnr": utils.mse2psnr(loss_per_batch)})
        log.info(log_string)

        val_then = time.time()
        val_target_image, rgb_image, depth_image = validate(cfg, val_iterator, model, device)
        val_target_image = val_target_image.squeeze().permute(1, 2, 0)
        mse = torchmetrics.functional.mean_squared_error(rgb_image, val_target_image)
        psnr = torchmetrics.functional.peak_signal_noise_ratio(rgb_image, val_target_image)
        utils.log_losses(writer, "val", train_iter, time.time() - then, losses={"mse": mse, "psnr": psnr})
        writer.add_image("val/target_image", val_target_image, train_iter, dataformats="HWC")
        writer.add_image("val/rgb_image", rgb_image, train_iter, dataformats="HWC")
        writer.add_image("val/depth_image", depth_image, train_iter, dataformats="HWC")
        val_log_string = utils.log_losses(writer, "val", train_iter, time.time() - val_then,
                                          losses={"avg_loss": mse, "avg_psnr": psnr})
        log.info(val_log_string)

        if train_iter != 0 and train_iter % cfg.experiment.save_freq == 0:
            ckpt_dict = {
                "iter": train_iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            torch.save(ckpt_dict, Path(writer.log_dir) / f"checkpoint{train_iter}.ckpt")


def validate(cfg: DictConfig,
             iterator: Callable,
             model: torch.nn.Module,
             device: torch.device
             ):

    batch = next(iterator)

    rays = batch["rays"].to(device)
    target_image = batch["rgb_image"].to(device)

    im_height, im_width = target_image.shape[-2], target_image.shape[-1]
    ray_batch = utils.batchify(rays[0], cfg.num_rays)

    rgb_per_image, depth_per_image = [], []
    with torch.no_grad():
        for i, rays in enumerate(ray_batch):
            rgb_per_ray, depth_per_ray = model(target_image, rays)
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
