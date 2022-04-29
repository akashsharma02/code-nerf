from typing import Literal, Dict, Optional, Callable

from pathlib import Path

import numpy as np
import imageio
import torch
import torchvision.transforms as transforms
from . import InfiniteSampler, DataModule, compute_ray_directions, compute_rays


class CARLADataModule(DataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False
    ):
        super(CARLADataModule, self).__init__(data_dir, batch_size, num_workers, pin_memory, shuffle)

        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.data_train = CARLADataset(path=self.data_dir, stage="train", transform=self.transforms)
        self.data_val = CARLADataset(path=self.data_dir, stage="val", transform=self.transforms)

    def setup(self, rank=0, num_replicas=1, seed=0):
        self.train_sampler = InfiniteSampler(self.data_train, rank=rank, num_replicas=num_replicas, shuffle=self.shuffle, seed=seed)
        self.val_sampler = InfiniteSampler(self.data_val, rank=rank, num_replicas=num_replicas, shuffle=self.shuffle, seed=seed)
        super(CARLADataModule, self).setup(rank=rank, num_replicas=num_replicas, seed=seed)


class CARLADataset(torch.utils.data.Dataset):
    """
    Dataset rendered from CARLA (GRAF: Schwarz et al. 2020)
    """

    def __init__(
        self, path: str,
        stage: Literal["train", "val"] = "train",
        transform: Optional[Callable] = None
    ):
        """
        Args:
            stage: train | val
            image_size: result image size (resizes if different)
            world_scale: amount to scale entire world by
        """

        super(CARLADataset, self).__init__()
        self.root_dir = Path(path)
        self.rgb_dir = self.root_dir / "rgb"
        self.pose_dir = self.root_dir / "poses"

        intrinsic = np.load(self.root_dir / "intrinsics.npy")
        self.transform = transforms.ToTensor()
        self.intrinsic = np.eye(4)
        self.intrinsic[:3, :3] = intrinsic
        # We divide by 2, since we reduce the resolution of target image by 2
        self.intrinsic[:2, :3] = self.intrinsic[:2, :3]/4
        self.intrinsic = self.transform(self.intrinsic.astype(np.float32))[0]

        self.dataset_name = "carla-cars"
        self.stage = stage
        self.num_objects = 18
        self.ray_directions = compute_ray_directions(self.intrinsic)

        print(f"Loading CARLA dataset {self.root_dir} name: {self.dataset_name}-{self.stage}")

        rgb_fnames = [f for f in self.rgb_dir.iterdir() if f.is_file and f.suffix in [".png", ".jpg"]][:2]
        pose_fnames = [f for f in self.pose_dir.iterdir() if f.is_file and f.suffix == ".npy"][:2]

        rgb_fnames = np.asarray(sorted(rgb_fnames, key=lambda x: int(x.stem)))
        pose_fnames = np.asarray(sorted(pose_fnames, key=lambda x: x.stem.replace('_extrinsics', '')))

        assert len(rgb_fnames) == len(pose_fnames), "The number of pose files do not match number of rgb images"

        total_length = len(rgb_fnames)

        train_size = int(total_length * 0.75)

        idxs = np.random.permutation(total_length)
        train_idxs, val_idxs = idxs[:train_size], idxs[train_size:]
        if self.stage == "train":
            self.rgb_fnames = rgb_fnames[train_idxs]
            self.pose_fnames = pose_fnames[train_idxs]
            self.rgb_frames = [imageio.imread(rgb_filename) for rgb_filename in self.rgb_fnames]
            self.poses = [np.load(pose_filename) for pose_filename in self.pose_fnames]
        elif self.stage == "val":
            self.rgb_fnames = rgb_fnames[val_idxs]
            self.pose_fnames = pose_fnames[val_idxs]
            self.rgb_frames = [imageio.imread(rgb_filename) for rgb_filename in self.rgb_fnames]
            self.poses = [np.load(pose_filename) for pose_filename in self.pose_fnames]

        self.length = len(self.rgb_fnames)

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> Dict:
        """
        Get Item given index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb = self.rgb_frames[idx]
        pose = np.eye(4)
        pose[:3, :4] = self.poses[idx]

        rgb = rgb / 255.0

        rgb_image = rgb.astype(np.float32)[..., :3]
        pose = pose.astype(np.float32)
        transform = transforms.Resize(128)
        rgb_image = transform(self.transform(rgb_image))
        pose = self.transform(pose)

        ray_origins, ray_directions = compute_rays(self.ray_directions, pose[0])

        rays = torch.cat([ray_origins, ray_directions], dim=-1)
        rgb = rgb_image.reshape([3, -1]).T
        sample = {
            'rays': rays,
            'rgb': rgb,
            'rgb_image': rgb_image,
        }

        return sample
