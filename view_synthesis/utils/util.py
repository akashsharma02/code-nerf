from typing import Tuple, List, Optional
import math
import torch
from pathlib import Path
import torch.distributed as dist

from view_synthesis.cfgnode import CfgNode
import view_synthesis.datasets as datasets


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
    # TODO: Remove this, (using only for debugging)
    # if n_gpus_to_use < 2 and is_distributed == True:
    #     print(f"Warning: Setting up DataDistributedParallel is forbidden with 1 GPU.")
    #     is_distributed = False

    main_device = torch.device('cuda:0' if n_gpus_to_use > 0 else 'cpu')
    list_ids = list(range(n_gpus_to_use))

    return main_device, list_ids


def is_main_process(is_distributed) -> bool:
    """ Test whether process is the main worker process
    """
    return (dist.get_rank() == 0) if is_distributed else True


def prepare_experiment(cfg: CfgNode):
    """TODO: Docstring for prepare_experiment.

    :function: TODO
    :returns: TODO

    """
    logdir_path = Path(cfg.experiment.logdir) / cfg.experiment.id
    logdir_path.mkdir(parents=True, exist_ok=True)
    with open(Path(logdir_path) / "config.yml", "w") as f:
        f.write(cfg.dump())

    return logdir_path


def prepare_dataloader(cfg: CfgNode) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Prepare the dataloader considering DataDistributedParallel

    :function:
        rank: Process rank. 0 == main process
    :returns: TODO

    """
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

    return train_dataloader, val_dataloader


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
