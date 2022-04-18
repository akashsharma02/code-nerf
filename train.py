import os
import time

import hydra
import omegaconf
from dotenv import load_dotenv
from omegaconf import DictConfig
import numpy as np
import torch

import view_synthesis.models as network_arch
import view_synthesis.utils as utils
from view_synthesis.utils import rank_zero_only
import view_synthesis.nerf as nerf
from eval import validate

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

    writer = rank_zero_only(utils.prepare_experiment(cfg))

    # Load Data
    log.info(f"Instantiating datamodule: {cfg.datamodule._target_}")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    sample = next(iter(datamodule.train_dataloader()))
    height, width = sample["color"].shape[-2:]
    intrinsic = sample["intrinsic"][0]

    # Load model
    log.info(f"Instantiating model: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model, ray_sampler={"height": height, "width": width, "intrinsics": intrinsic})

    # Create optimizer
    log.info(f"Instantiating optimizer: {cfg.optim._target_}")
    optimization_params = [
        {"params": list(model.encoder.parameters()), 'lr': cfg.experiment.encoder_lr},
        {"params": list(model.decoder.parameters()), 'lr': cfg.experiment.decoder_lr}
    ]
    # optimizable_params = [model.]
    optimizer = hydra.utils.instantiate(cfg.optim, optimization_params)

    model.train()
    model.to(device)
    for batch in datamodule.train_dataloader():

        batch = utils.dict_to_device(batch, device)
        optimizer.zero_grad()

        target_image = batch["color"][:, :3, ...]
        target_pose = batch["pose"][0]

        rgb, sigma = model(target_image, target_pose)

        # # Load a sample of rays and target pixels from data
        # ro_batch, rd_batch, select_inds = ray_sampler.sample(tform_cam2world=train_data["pose"])
        # tgt_pixel_batch = train_data["color"].flatten(1, 2)
        # tgt_object_ids_batch = train_data["object_id"][:, None].expand(-1, cfg.nerf.ray_sampler.num_random_rays)
        # tgt_pixel_batch = [tgt_pixel_batch[k, select_inds[k], :] for k in range(cfg.dataset.train_batch_size)]
        # tgt_pixel_batch, tgt_object_ids_batch = torch.cat(tgt_pixel_batch, dim=0), tgt_object_ids_batch.reshape(-1)
        #
        # msg = "Chunksize needs to atleast be less than to the number of rays sampled from a single image"
        # assert cfg.nerf.train.chunksize <= cfg.nerf.ray_sampler.num_random_rays * cfg.dataset.train_batch_size, msg
        # ro_minibatches, rd_minibatches = utils.get_minibatches(ro_batch, cfg.nerf.train.chunksize), utils.get_minibatches(rd_batch, cfg.nerf.train.chunksize)
        # tgt_pixel_minibatches = utils.get_minibatches(tgt_pixel_batch, cfg.nerf.train.chunksize)
        # tgt_object_ids_minibatches = utils.get_minibatches(tgt_object_ids_batch, cfg.nerf.train.chunksize)
        #
        # num_batches = len(ro_minibatches)
        # msg = "Mismatch in batch length of ray origins, ray directions and target pixels"
        # assert num_batches == len(rd_minibatches) == len(tgt_pixel_minibatches) == len(tgt_object_ids_minibatches), msg
        #
        # for j, (ro, rd, target_object_ids, target_pixels) in enumerate(zip(ro_minibatches, rd_minibatches, tgt_object_ids_minibatches, tgt_pixel_minibatches)):
        #     then = time.time()
        #
        #     # Pass through NeRF
        #     target_object_embedding, rays = models["embedding"](target_object_ids), (ro, rd)
        #     rgb_coarse, rgb_fine = nerf.predict_radiance_and_render(rays,
        #                                                             point_sampler,
        #                                                             embedders,
        #                                                             models["nerf_coarse"],
        #                                                             models["nerf_fine"] if "nerf_fine" in models else None,
        #                                                             target_object_embedding)
        #     # Compute losses
        #     nerf_loss_coarse = torch.nn.functional.mse_loss(rgb_coarse[..., :3], target_pixels[..., :3])
        #     nerf_loss_fine = 0.0
        #     psnr = utils.mse2psnr(nerf_loss_coarse.item())
        #     if rgb_fine is not None:
        #         nerf_loss_fine = torch.nn.functional.mse_loss(rgb_fine[..., :3], target_pixels[..., :3])
        #         psnr = utils.mse2psnr(nerf_loss_fine.item())
        #     shape_params, texture_params = network_arch.get_params_tensor(models["embedding"], cfg.is_distributed)
        #     embedding_regularization = cfg.experiment.regularizer_lambda * (torch.norm(shape_params, p=2) + torch.norm(texture_params, p=2))
        #     loss = nerf_loss_coarse + nerf_loss_fine + embedding_regularization
        #
        #     # Backprop and optimizer step
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()
        #
        # i = iteration * num_batches + j
        # if utils.is_main_process(cfg.is_distributed) and i > 0:
        #     if i % cfg.experiment.print_every == 0:
        #         losses_dict = {"nerf_loss_coarse": nerf_loss_coarse.item()}
        #         if rgb_fine is not None:
        #             losses_dict["nerf_loss_fine"] = nerf_loss_fine.item()
        #         losses_dict["embedding_loss"] = embedding_regularization.item()
        #         losses_dict["total_loss"] = loss.item()
        #         losses_dict["psnr"] = psnr
        #         log_string = utils.log_losses(writer, "train", i, time.time()-then, losses_dict, scheduler.get_last_lr()[0])
        #         writer.add_images("train/target_image", train_data["color"][..., :3], i, dataformats='NHWC')
        #         print(log_string)
        #
        #     if (i % cfg.experiment.save_every == 0 or i == cfg.experiment.iterations - 1):
        #         checkpoint_dict = {
        #             "iter": iteration,
        #             "model_nerf_coarse_state_dict": models["nerf_coarse"].state_dict()}
        #         if "nerf_fine" in models:
        #             checkpoint_dict["model_nerf_fine_state_dict"] = models["nerf_fine"].state_dict()
        #         checkpoint_dict["model_embedding_state_dict"] = models["embedding"].state_dict()
        #         checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()
        #         torch.save(checkpoint_dict, logdir_path / f"checkpoint{i:5d}.ckpt")
        #         print("================== Saved Checkpoint =================")
        #
        # # Parallel rendering of image for Validation
        # if (i > 0 and i % cfg.experiment.validate_every == 0):
        #     validate(cfg, i, val_dataloader, models, [ray_sampler, point_sampler], embedders, writer, device)


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


@hydra.main(config_path="conf", config_name="config")
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
