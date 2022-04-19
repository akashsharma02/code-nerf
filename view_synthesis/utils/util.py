from typing import Tuple, List, Optional, OrderedDict, Literal, Union, Callable, Any, Sequence, Dict
import logging
from functools import wraps
import math
import torch
from pathlib import Path
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on rank 0."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()):
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


@rank_zero_only
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
        # "callbacks",
        "logger",
        # "trainer",
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


def dict_to_device(batch: Dict, device: torch.device) -> Dict:
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = batch[k].to(device)
    return batch


def batchify(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i: i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


@rank_zero_only
def log_losses(writer: SummaryWriter,
               mode: Literal["train", "val", "val-optim"],
               i: int,
               time_taken: float,
               losses: Dict,
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


def load_checkpoint(cfg: DictConfig,
                    model: torch.nn.Module,
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
    checkpoint_file = Path(cfg.checkpoint_dir)
    if checkpoint_file.exists() and checkpoint_file.is_file() and checkpoint_file.suffix == ".ckpt":
        checkpoint = torch.load(cfg.load_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"])
        train_iter = checkpoint["iter"]

    return train_iter, model, optimizer
