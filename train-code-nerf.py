from typing import List, Tuple, Union, Literal
from numpy.typing import DTypeLike
from types import FunctionType
import os
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as ddp

from view_synthesis.cfgnode import CfgNode
import view_synthesis.datasets as datasets
import view_synthesis.models as network_arch
from view_synthesis.utils import prepare_device, is_main_process, prepare_experiment, mse2psnr, get_minibatches
from view_synthesis.nerf import RaySampler, PointSampler, volume_render, PositionalEmbedder


def prepare_dataloader(cfg: CfgNode
                       ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.Dataset]:
    """
    Prepare the dataloader considering DataDistributedParallel

    Args:
        cfg: CfgNode from YAML file
    Returns:
        train_dataloader: train subset dataloader
        val_dataloader: validation subset dataloader
        dataset: Complete dataset to access dataset properties

    """
    if cfg.dataset.type == "SRNDataset":
        train_dataset = datasets.SRNDataset(
            path=cfg.dataset.basedir,
            stage="train",
            image_size=(cfg.dataset.image_size, cfg.dataset.image_size),
            world_scale=cfg.dataset.world_scale
        )
        val_dataset = datasets.SRNDataset(
            path=cfg.dataset.basedir,
            stage="val",
            image_size=(cfg.dataset.image_size, cfg.dataset.image_size),
            world_scale=cfg.dataset.world_scale
        )
    else:
        dataset = getattr(datasets, cfg.dataset.type)(
            cfg.dataset.basedir,
            cfg.dataset.resolution_level)
        train_size = int(len(dataset) * 0.75)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=cfg.experiment.iterations
    )

    val_sampler = torch.utils.data.RandomSampler(
        val_dataset,
        replacement=True,
        num_samples=cfg.experiment.iterations
    )

    if cfg.is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            drop_last=False
        )

        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            drop_last=False
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.dataset.train_batch_size, shuffle=False, num_workers=0, sampler=train_sampler, pin_memory=False)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.dataset.val_batch_size, shuffle=False, num_workers=0, sampler=val_sampler, pin_memory=False)

    return train_dataloader, val_dataloader, train_dataset


def prepare_models(cfg: CfgNode,
                   num_objects: int
                   ) -> "OrderedDict[torch.nn.Module, Union[torch.nn.Module, None]]":
    """
    Prepare the torch models

    Args:
        rank: Process rank. 0 == main process
    Return:
        models: Dict containing embedding model and NeRF model
        latent_codes: List containing latent codes for
                      each object style in a semantic category defined by the dataset

    """
    rank = 0
    if cfg.is_distributed:
        rank = dist.get_rank()

    models = OrderedDict()
    models['embedding'] = network_arch.ShapeTextureEmbedding(
        num_embeddings=num_objects,
        shape_code_size=cfg.models.embedding.shape_code_size,
        texture_code_size=cfg.models.embedding.texture_code_size,
    ).to(rank)

    models['nerf'] = getattr(network_arch, cfg.models.nerf.type)(
        hidden_size=cfg.models.nerf.hidden_size,
        num_embeddings=num_objects,
        shape_code_size=cfg.models.embedding.shape_code_size,
        texture_code_size=cfg.models.embedding.texture_code_size,
        num_encoding_fn_xyz=cfg.nerf.embedder.num_encoding_fn_xyz,
        include_input_xyz=cfg.nerf.embedder.include_input_xyz,
        num_encoding_fn_dir=cfg.nerf.embedder.num_encoding_fn_dir,
        include_input_dir=cfg.nerf.embedder.include_input_dir,
    ).to(rank)

    if cfg.is_distributed:
        models["embedding"] = ddp(models["embedding"], device_ids=[rank], output_device=rank)
        models["nerf"] = ddp(models['nerf'], device_ids=[rank], output_device=rank)

    return models


def prepare_optimizer(cfg: CfgNode, models: "OrderedDict[str, torch.nn.Module]") -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """ Load the optimizer and learning schedule according to the configuration

    :function: TODO
    :returns: TODO

    """
    trainable_params = []
    for model_name, model in models.items():
        trainable_params += list(model.parameters())

    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_params,
        lr=cfg.optimizer.lr
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cfg.optimizer.scheduler_gamma ** (
            epoch / cfg.optimizer.scheduler_step_size)
    )

    return optimizer, scheduler


def load_checkpoint(cfg: CfgNode, models: "OrderedDict[str, torch.nn.Module]", optimizer: torch.optim.Optimizer) -> int:
    """TODO: Docstring for load_checkpoint.

    :function: TODO
    :returns: TODO

    """
    start_iter = 0

    checkpoint_file = Path(cfg.load_checkpoint)
    if checkpoint_file.exists() and checkpoint_file.is_file() and checkpoint_file.suffix == ".ckpt":
        rank = 0
        if cfg.is_distributed:
            rank = dist.get_rank()
            map_location = {"cuda:0": f"cuda:{rank}"}
            checkpoint = torch.load(
                cfg.load_checkpoint, map_location=map_location)
            # Ensure that all loading by all processes is done before any process has started saving models
            torch.distributed.barrier()
        else:
            checkpoint = torch.load(cfg.load_checkpoint)
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint, "module")

        for model_name, model in models.items():
            model.load_state_dict(
                checkpoint[f"model_{model_name}_state_dict"])

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["start_iter"]

    return start_iter


def prepare_embedders(cfg: CfgNode, datatype: DTypeLike, device: torch.cuda.Device) -> Tuple[PositionalEmbedder, Union[PositionalEmbedder, None]]:
    """Load the embedding functions for points and viewdirs from config file

    :function: TODO
    :returns: TODO

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


def nerf_forward_pass(model: torch.nn.Module,
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
    if is_main_process(cfg.is_distributed):
        logdir_path = prepare_experiment(cfg)
        writer = SummaryWriter(logdir_path)

    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)

    # Load Data
    train_dataloader, val_dataloader, dataset = prepare_dataloader(cfg)

    # Prepare Model, Optimizer, and load checkpoint
    models = prepare_models(cfg, dataset.num_objects)
    optimizer, scheduler = prepare_optimizer(cfg, models)
    start_iter = load_checkpoint(cfg, models, optimizer)

    # Prepare RaySampler
    first_data_sample = next(iter(train_dataloader))
    (height, width), intrinsic, datatype = first_data_sample["color"][0].shape[:2], first_data_sample["intrinsic"][0], first_data_sample["intrinsic"][0].dtype

    ray_sampler = RaySampler(height,
                             width,
                             intrinsic,
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

    # Prepare Positional Embedding functions
    embedders = prepare_embedders(cfg, datatype, device)

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
        tgt_object_ids_batch = train_data["object_id"].repeat(1, cfg.nerf.ray_sampler.num_random_rays)
        tgt_pixel_batch = [tgt_pixel_batch[k, select_inds[k], :] for k in range(cfg.dataset.train_batch_size)]
        tgt_pixel_batch, tgt_object_ids_batch = torch.cat(tgt_pixel_batch, dim=0), tgt_object_ids_batch.view(-1)

        msg = "Chunksize needs to atleast be less than to the number of rays sampled from a single image"
        assert cfg.nerf.train.chunksize <= cfg.nerf.ray_sampler.num_random_rays * cfg.dataset.train_batch_size, msg
        ro_minibatches, rd_minibatches = get_minibatches(ro_batch, cfg.nerf.train.chunksize), get_minibatches(rd_batch, cfg.nerf.train.chunksize)
        tgt_pixel_minibatches = get_minibatches(tgt_pixel_batch, cfg.nerf.train.chunksize)
        tgt_object_ids_minibatches = get_minibatches(tgt_object_ids_batch, cfg.nerf.train.chunksize)

        num_batches = len(ro_minibatches)
        msg = "Mismatch in batch length of ray origins, ray directions and target pixels"
        assert num_batches == len(rd_minibatches) == len(tgt_pixel_minibatches) == len(tgt_object_ids_minibatches), msg

        for j, (ro, rd, target_object_ids, target_pixels) in enumerate(zip(ro_minibatches, rd_minibatches, tgt_object_ids_minibatches, tgt_pixel_minibatches)):
            then = time.time()

            # Pass through nerf model
            pts, z_vals = point_sampler.sample_uniform(ro, rd)
            target_object_embedding = models["embedding"](target_object_ids)
            radiance_field = nerf_forward_pass(models["nerf"], embedders, rd, pts, target_object_embedding)
            (rgb, _, _, weights, _) = volume_render(radiance_field,
                                                    z_vals,
                                                    rd,
                                                    radiance_field_noise_std=cfg.nerf.train.radiance_field_noise_std,
                                                    white_background=cfg.nerf.white_background)

            nerf_loss = torch.nn.functional.mse_loss(rgb[..., :3], target_pixels[..., :3])
            embedding_params = models['embedding'].parameters()
            embedding_params = torch.cat([x.view(-1) for x in embedding_params])
            embedding_regularization = cfg.experiment.regularizer_lambda * (torch.norm(embedding_params) + torch.norm(embedding_params))
            psnr = mse2psnr(nerf_loss.item())
            loss = nerf_loss + embedding_regularization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            i = iteration * num_batches + j

            if is_main_process(cfg.is_distributed) and i % cfg.experiment.print_every == 0:
                writer.add_scalar("train/nerf_loss", nerf_loss.item(), i)
                writer.add_scalar("train/embedding_loss", embedding_regularization.item(), i)
                writer.add_scalar("train/psnr", psnr, i)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], i)
                writer.add_images("train/target_image", train_data["color"][..., :3], i, dataformats='NHWC')

                log_string = f"[TRAIN] Iter: {i:>8} "
                log_string += f"Load Iter: {iteration:>8} Time taken: {time.time()-then:>4.4f} "
                log_string += f"Learning rate: {scheduler.get_last_lr()[0]:0.8f} NeRF Loss: {nerf_loss.item():>4.4f} "
                log_string += f"Regularization Loss: {embedding_regularization.item():>4.4f} "
                log_string += f"Total Loss: {loss.item():>4.4f} "
                log_string += f"PSNR: {psnr:>4.4f}"
                print(log_string)

            if is_main_process(cfg.is_distributed) and (i > 0 and i % cfg.experiment.save_every == 0 or i == cfg.experiment.iterations - 1):
                checkpoint_dict = {
                    "iter": i,
                    "model_nerf_state_dict": models["nerf"].state_dict(),
                    "model_embedding_state_dict": models["embedding"].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(checkpoint_dict, logdir_path / f"checkpoint{i:5d}.ckpt")
                print("================== Saved Checkpoint =================")

            # Parallel rendering of image for Validation
            if (i > 0 and i % cfg.experiment.validate_every == 0):
                validate(cfg, iteration, i, val_dataloader, models,
                         ray_sampler, point_sampler, embedders, writer, device)


def validate(cfg: CfgNode,
             iteration: int,
             i: int,
             dataloader: torch.utils.data.DataLoader,
             models: "OrderedDict[torch.nn.Module, torch.nn.Module]",
             ray_sampler: RaySampler,
             point_sampler: PointSampler,
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
    if cfg.is_distributed:
        dataloader.sampler.set_epoch(iteration)

    # Load data independently in all processes as a list of tuples
    val_data = next(iter(dataloader))

    # Broadcast validation data in rank 0 to all the processes
    if cfg.is_distributed:
        val_data = list(val_data.items())
        torch.distributed.broadcast_object_list(val_data, 0)
        val_data = dict(val_data)

    for key, val in val_data.items():
        if torch.is_tensor(val):
            val_data[key] = val_data[key].to(device)

    if cfg.is_distributed:
        all_shape_embedding, all_texture_embedding = models["embedding"].module.get_all_embeddings(device=device)
    else:
        all_shape_embedding, all_texture_embedding = models["embedding"].get_all_embeddings(device=device)

    shape_embedding = all_shape_embedding.mean(dim=0, keepdim=True).detach().clone()
    texture_embedding = all_texture_embedding.mean(dim=0, keepdim=True).detach().clone()
    optimizer = getattr(torch.optim, cfg.optimizer.val_type)(
        [shape_embedding, texture_embedding],
        lr=cfg.optimizer.val_lr,
    )

    for val_iter in range(0, cfg.experiment.val_iterations):
        val_then = time.time()

        ro_batch, rd_batch, select_inds = ray_sampler.sample(tform_cam2world=val_data["pose"])
        tgt_pixel_batch = val_data["color"].flatten(1, 2)
        shape_embedding_batch, texture_embedding_batch = shape_embedding.expand(ro_batch.shape[0], -1), texture_embedding.expand(ro_batch.shape[0], -1)
        tgt_pixel_batch = [tgt_pixel_batch[k, select_inds[k], :] for k in range(cfg.dataset.val_batch_size)]
        tgt_pixel_batch = torch.cat(tgt_pixel_batch, dim=0)

        msg = "Chunksize needs to atleast be less than to the number of rays sampled from a single image"
        assert cfg.nerf.validation.chunksize <= cfg.nerf.ray_sampler.num_random_rays * cfg.dataset.val_batch_size, msg
        ro_minibatches, rd_minibatches = get_minibatches(ro_batch, cfg.nerf.validation.chunksize), get_minibatches(rd_batch, cfg.nerf.validation.chunksize)
        tgt_pixel_minibatches = get_minibatches(tgt_pixel_batch, cfg.nerf.validation.chunksize)

        num_batches = len(ro_minibatches)
        msg = "Mismatch in batch length of ray origins, ray directions and target pixels"
        assert num_batches == len(rd_minibatches) == len(tgt_pixel_minibatches), msg

        for j, (ro, rd, target_pixels) in enumerate(zip(ro_minibatches, rd_minibatches, tgt_pixel_minibatches)):

            # Pass through NeRF model
            pts, z_vals = point_sampler.sample_uniform(ro, rd)
            radiance_field = nerf_forward_pass(models["nerf"], embedders, rd, pts, [shape_embedding_batch, texture_embedding_batch])
            (rgb, _, _, weights, _) = volume_render(radiance_field,
                                                    z_vals,
                                                    rd,
                                                    radiance_field_noise_std=cfg.nerf.train.radiance_field_noise_std,
                                                    white_background=cfg.nerf.white_background)

            nerf_loss = torch.nn.functional.mse_loss(rgb[..., :3], target_pixels[..., :3])
            embedding_regularization = cfg.experiment.regularizer_lambda * (torch.norm(shape_embedding) + torch.norm(texture_embedding))
            psnr = mse2psnr(nerf_loss.item())
            loss = nerf_loss + embedding_regularization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if is_main_process(cfg.is_distributed) and val_iter > 0 and val_iter % cfg.experiment.val_print_every == 0:
            writer.add_scalar("val_optimization/nerf_loss", nerf_loss.item(), val_iter)
            writer.add_scalar("val_optimization/psnr", psnr, val_iter)
            writer.add_images("val_optimization/target_image", val_data["color"][..., :3], i, dataformats='NHWC')

            log_string = f"[VAL OPTIM] ValIter: {val_iter:>8} "
            log_string += f"Time taken: {time.time()-val_then:>4.4f} "
            log_string += f"NeRF Loss: {nerf_loss.item():>4.4f} "
            log_string += f"Regularization Loss: {embedding_regularization.item():>4.4f} "
            log_string += f"Total Loss: {loss.item():>4.4f} "
            log_string += f"PSNR: {psnr:>4.4f}"
            print(log_string)

    rgb = parallel_image_render(val_data["pose"],
                                [shape_embedding, texture_embedding],
                                cfg,
                                models,
                                ray_sampler,
                                point_sampler,
                                embedders,
                                device)

    if is_main_process(cfg.is_distributed):
        assert rgb is not None, "Main process must contain rgb"
        target_pixels = val_data["color"].view(-1, 4)
        loss = torch.nn.functional.mse_loss(rgb[..., : 3], target_pixels[..., : 3])
        psnr = mse2psnr(loss.item())

        target_rgb = target_pixels.reshape(list(val_data["color"].shape[: -1]) + [4])
        rgb = rgb.reshape(list(val_data["color"].shape[: -1]) + [rgb.shape[-1]])
        writer.add_scalar("val/loss", loss, i)
        writer.add_images("val/rgb_image", rgb[..., : 3], i, dataformats='NHWC')
        writer.add_images("val/target_image", target_rgb[..., : 3], i, dataformats='NHWC')

        log_string = f"[VAL  ] Iter: {i:>8} Iteration: {iteration:>8} "
        log_string += f"Time taken: {time.time() - val_then:>4.4f} Loss: {loss.item():>4.4f} PSNR: {psnr:>4.4f}"
        print(log_string)


def parallel_image_render(pose: torch.Tensor,
                          object_embedding: Tuple[torch.Tensor, torch.Tensor],
                          cfg: CfgNode,
                          models: "OrderedDict[torch.nn.Module, torch.nn.Module]",
                          ray_sampler: RaySampler,
                          point_sampler: PointSampler,
                          embedders: List[Union[PositionalEmbedder, None]],
                          device: torch.cuda.Device):
    """
    Parallely render images on multiple GPUs for validation
    """
    rank = 0
    if cfg.is_distributed:
        rank = dist.get_rank()

    for _, model in models.items():
        model.eval()

    with torch.no_grad():
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
        ro_minibatches = get_minibatches(ro_batch, cfg.nerf.validation.chunksize)
        rd_minibatches = get_minibatches(rd_batch, cfg.nerf.validation.chunksize)
        shape_embedding_minibatches = get_minibatches(shape_embedding_batch, cfg.nerf.validation.chunksize)
        texture_embedding_minibatches = get_minibatches(texture_embedding_batch, cfg.nerf.validation.chunksize)

        rgb_batches = []
        for ro, rd, z_s, z_t in zip(ro_minibatches, rd_minibatches, shape_embedding_minibatches, texture_embedding_minibatches):

            # Pass through NeRF model
            pts, z_vals = point_sampler.sample_uniform(ro, rd)

            radiance_field = nerf_forward_pass(models["nerf"], embedders, rd, pts, [z_s, z_t])
            (rgb, _, _, weights, _) = volume_render(radiance_field,
                                                    z_vals,
                                                    rd,
                                                    radiance_field_noise_std=cfg.nerf.validation.radiance_field_noise_std,
                                                    white_background=cfg.nerf.white_background)
            rgb_batches.append(rgb)

        rgb_batches = torch.cat(rgb_batches, dim=0)

        if not cfg.is_distributed:
            return rgb_batches

        # Pad image chunks to get equal chunksize for all_gather/gather
        padded_rgb = torch.zeros((padding_per_process[rank], rgb_batches.shape[-1]),
                                 dtype=rgb_batches.dtype,
                                 device=rgb_batches.device)
        rgb = torch.cat([rgb_batches, padded_rgb], dim=0)
        all_rgb_batches = [torch.zeros_like(rgb_batches) for _ in range(cfg.gpus)]
        torch.distributed.all_gather(all_rgb_batches, rgb_batches)

        if is_main_process(cfg.is_distributed):
            for i, size in enumerate(batchsize_per_process):
                all_rgb_batches[i] = all_rgb_batches[i][: size, ...]
            all_rgb_batches = torch.cat(all_rgb_batches, dim=0)

            return all_rgb_batches
        else:
            return None


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

    _, device_ids = prepare_device(cfg.gpus, cfg.is_distributed)

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
