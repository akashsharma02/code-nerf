import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


class DataModule(object):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.replacewith_dist_sampler = True if dist.is_initialized() else False

        self.data_train, self.data_val = None, None

    def setup(self, rank=0, num_replicas=1, seed=0, iterations=1):

        if self.replacewith_dist_sampler:
            self.train_sampler = torch.utils.data.DistributedSampler(
                self.data_train,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                drop_last=False
            )
            self.val_sampler = torch.utils.data.DistributedSampler(
                self.data_val,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                drop_last=False
            )

        self.train_loader = DataLoader(
            dataset=self.data_train,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        # Only one validation batch
        self.val_loader = DataLoader(
            dataset=self.data_val,
            sampler=self.val_sampler,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


def compute_ray_directions(intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Computes the set of ray directions given image intrinsics

    Args:
        height: int
        width: int
        intrinsics: np.ndarray 4x4

    Returns:
        meshgrid containing [H, W, 3] ray directions
    """
    assert intrinsics.shape == torch.Size(
        [4, 4]), "Incorrect intrinsics shape"

    focal_length = intrinsics[0, 0]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    ii, jj = torch.meshgrid(torch.arange(2*cx, dtype=intrinsics.dtype), torch.arange(2*cy, dtype=intrinsics.dtype), indexing='xy')
    return torch.stack(
        [
            (ii - cx) / focal_length,
            -(jj - cy) / focal_length,
            -torch.ones_like(ii),
        ],
        dim=-1
    )


def compute_rays(ray_directions: torch.Tensor, world_T_camera: torch.Tensor):
    """
        Rotate the bundle of rays given the camera pose

    :function:
        world_T_camera: 4x4 np.ndarray camera pose (SE3)
    :returns:
        ray origins: torch.Tensor [H, W, 3]
        ray directions: torch.Tensor [H, W, 3]

    """
    rotated_ray_directions = ray_directions @ world_T_camera[:3, :3].T
    ray_origins = world_T_camera[:3, -1][None, None, :].expand(rotated_ray_directions.shape)
    return ray_origins.view(-1, 3), rotated_ray_directions.view(-1, 3)
