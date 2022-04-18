from typing import Tuple, List, Optional, OrderedDict, Literal, Union, Callable, Any, Sequence
from functools import wraps
import math
import torch
from pathlib import Path
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp

from ..cfgnode import CfgNode
from ..datasets import dataset as datasets
from ..models import model as network_arch


def prepare_device(n_gpus_to_use: int, is_distributed: bool) -> Tuple[torch.device, List[int]]:
    """ Prepare GPU device if available, and get GPU indices for DataDistributedParallel

    :function: TODO
    :returns: TODO

    """
    n_gpu = torch.cuda.device_count()
    if n_gpus_to_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpus_to_use = 0
    if n_gpus_to_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpus_to_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpus_to_use = n_gpu

    main_device = torch.device('cuda:0' if n_gpus_to_use > 0 else 'cpu')
    list_ids = list(range(n_gpus_to_use))

    return main_device, list_ids


def is_main_process(is_distributed) -> bool:
    """ Test whether process is the main worker process
    """
    return (dist.get_rank() == 0) if is_distributed else True


def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on rank 0."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if dist.is_initialized() and dist.get_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def prepare_experiment(cfg: DictConfig):
    """TODO: Docstring for prepare_experiment.

    :function: TODO
    :returns: TODO

    """
    logdir_path = Path(cfg.experiment.logdir) / cfg.experiment.id
    logdir_path.mkdir(parents=True, exist_ok=True)

    print_config(cfg, resolve=True)
    writer = SummaryWriter(logdir_path)
    return writer


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(f"Field '{field}' not found in config")

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


def prepare_dataloader(stage: Literal["train", "val"], cfg: CfgNode) -> torch.utils.data.DataLoader:
    """ Prepare the dataloader considering DataDistributedParallel

    :function:
        rank: Process rank. 0 == main process
    :returns: TODO

    """
    is_distributed = hasattr(cfg, "is_distributed") and cfg.is_distributed

    dataset = getattr(datasets, cfg.dataset.type)(
        path=cfg.dataset.basedir,
        stage=stage
    )

    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=True,
        num_samples=cfg.experiment.iterations
    )

    if is_distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            drop_last=False
        )
    batch_size = cfg.dataset.train_batch_size if stage == "train" else cfg.dataset.val_batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=sampler, pin_memory=True)

    return dataloader, dataset


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
    if hasattr(cfg, "is_distributed") and cfg.is_distributed:
        rank = dist.get_rank()

    models = OrderedDict()
    models['embedding'] = network_arch.ShapeTextureEmbedding(
        num_embeddings=num_objects,
        shape_code_size=cfg.models.embedding.shape_code_size,
        texture_code_size=cfg.models.embedding.texture_code_size,
    ).to(rank)

    models['nerf_coarse'] = getattr(network_arch, cfg.models.nerf_coarse.type)(
        hidden_size=cfg.models.nerf_coarse.hidden_size,
        shape_code_size=cfg.models.embedding.shape_code_size,
        texture_code_size=cfg.models.embedding.texture_code_size,
        num_encoding_fn_xyz=cfg.nerf.embedder.num_encoding_fn_xyz,
        include_input_xyz=cfg.nerf.embedder.include_input_xyz,
        num_encoding_fn_dir=cfg.nerf.embedder.num_encoding_fn_dir,
        include_input_dir=cfg.nerf.embedder.include_input_dir,
    ).to(rank)
    if hasattr(cfg.models, "nerf_fine"):
        models['nerf_fine'] = getattr(network_arch, cfg.models.nerf_fine.type)(
            hidden_size=cfg.models.nerf_fine.hidden_size,
            shape_code_size=cfg.models.embedding.shape_code_size,
            texture_code_size=cfg.models.embedding.texture_code_size,
            num_encoding_fn_xyz=cfg.nerf.embedder.num_encoding_fn_xyz,
            include_input_xyz=cfg.nerf.embedder.include_input_xyz,
            num_encoding_fn_dir=cfg.nerf.embedder.num_encoding_fn_dir,
            include_input_dir=cfg.nerf.embedder.include_input_dir,
        ).to(rank)

    if hasattr(cfg, "is_distributed") and cfg.is_distributed:
        models["embedding"] = ddp(models["embedding"], device_ids=[rank], output_device=rank)
        models["nerf_coarse"] = ddp(models['nerf_coarse'], device_ids=[rank], output_device=rank)
        if hasattr(cfg.models, "nerf_fine"):
            models["nerf_fine"] = ddp(models['nerf_fine'], device_ids=[rank], output_device=rank)

    return models


def prepare_optimizer(cfg: CfgNode,
                      models: "OrderedDict[str, torch.nn.Module]"
                      ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """ Load the optimizer and learning schedule according to the configuration

    Args:
        cfg: CfgNode object
        models: torch.nn.Module objects whose parameters need to be optimized
    Return: TODO

    """
    var_list = [{'params': models['embedding'].parameters(), 'lr': cfg.optimizer.embedding_lr}]

    var_list += [{'params': models['nerf_coarse'].parameters()}]
    if 'nerf_fine' in models:
        var_list += [{'params': models['nerf_fine'].parameters()}]
    optimizer = getattr(torch.optim, cfg.optimizer.type)(var_list, lr=cfg.optimizer.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cfg.optimizer.scheduler_gamma ** (
            epoch / cfg.optimizer.scheduler_step_size)
    )
    return optimizer, scheduler


def load_checkpoint(cfg: CfgNode,
                    models: "OrderedDict[str, torch.nn.Module]",
                    optimizer: torch.optim.Optimizer
                    ) -> int:
    """
    Load checkpoint given a checkpoint file and initialize the starting iteration

    Args:
        cfg: CfgNode object
        models: torch.nn.Module objects whose parameters are to be loaded
        optimizer: optimizer whose parameters are to be loaded
    Return:
        start iteration

    """
    start_iter = 0
    is_distributed = hasattr(cfg, "is_distributed") and cfg.is_distributed
    rank = 0 if not is_distributed else dist.get_rank()

    checkpoint_file = Path(cfg.load_checkpoint)
    if checkpoint_file.exists() and checkpoint_file.is_file() and checkpoint_file.suffix == ".ckpt":
        map_location = {"cuda:0": f"cuda:{rank}"}
        checkpoint = torch.load(cfg.load_checkpoint, map_location=map_location)
        # Ensure that all loading by all processes is done before any process has started saving models
        if is_distributed:
            torch.distributed.barrier()

        for model_name, model in models.items():
            if not is_distributed:
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                    checkpoint[f"model_{model_name}_state_dict"], "module.")
            model.load_state_dict(
                checkpoint[f"model_{model_name}_state_dict"])

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    return start_iter


def mse2psnr(mse_val: float) -> float:
    """
    Calculate PSNR from MSE

    :function: TODO
    :returns: TODO

    """
    # For numerical stability, avoid a zero mse loss.
    if mse_val == 0:
        mse_val = 1e-5
    return -10.0 * math.log10(mse_val)


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i: i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def log_losses(writer: SummaryWriter,
               mode: Literal["train", "val", "val-optim"],
               i: int,
               time_taken: float,
               losses: "OrderedDict[float, float, float, float, float]",
               learning_rate: Optional[float] = None) -> str:
    """
    Log losses to Tensorboard and console output
    """
    log_string = ""
    if mode == "train":
        log_string += f"[TRAIN ] Iter: {i:>8} "
    elif mode == "val":
        log_string += f"[VAL   ] Iter: {i:>8} "
    else:
        log_string += f"[VALOPT] Iter: {i:>8} "
    log_string += f"Time taken: {time_taken:>4.4f} "

    if learning_rate:
        log_string += f"Learning rate: {learning_rate:0.8f} "
        writer.add_scalar("train/learning_rate", learning_rate, i)

    for key, val in losses.items():
        writer.add_scalar(f"{mode}/{key}", val, i)
        log_string += f"{key}: {val:>4.4f} "

    return log_string
