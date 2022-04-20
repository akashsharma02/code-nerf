from typing import Literal, Dict, Optional, Callable

from pathlib import Path

import numpy as np
import imageio
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from . import InfiniteSampler, compute_ray_directions, compute_rays


class SRNDataset(torch.utils.data.Dataset):
    """
    Dataset from Scene Representation Networks (V. Sitzmann et al 2020)
    """

    def __init__(
        self,
        path: str,
        stage: Literal["train", "val"] = "train",
    ):
        """
        Args:
           path: Root directory for the dataset
           stage: train | val | test
           transform: torchvision.Transforms
        """
        super(SRNDataset, self).__init__()
        self.base_path = Path(path)
        self.dataset_name = self.base_path.stem.split("_")[-1]
        self.base_path = self.base_path / f"{self.dataset_name}_{stage}"

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert self.base_path.exists(), f"{self.base_path} does not exist"

        is_chair = "chair" in self.dataset_name
        if is_chair and stage == "train":
            tmp = self.base_path / "chairs_2.0_train"
            if tmp.exists():
                self.base_path = tmp

        self.intrinsic = sorted(self.base_path.glob("*/intrinsics.txt"))
        self.num_objects = len(self.intrinsic)

        self.rgb_all_filenames, self.pose_all_filenames = [], []
        for index, intrinsic_path in enumerate(self.intrinsic):
            rgb_directory = intrinsic_path.parent / "rgb"
            pose_directory = intrinsic_path.parent / "pose"
            rgb_files = sorted([(index, path) for path in rgb_directory.iterdir()])
            pose_files = sorted([(index, path) for path in pose_directory.iterdir()])
            self.rgb_all_filenames.extend(rgb_files)
            self.pose_all_filenames.extend(pose_files)

        assert len(self.rgb_all_filenames) == len(self.pose_all_filenames)
        self.num_views = len(self.rgb_all_filenames) // self.num_objects
        self.transform = transforms.ToTensor()
        print(f"Total number of objects in the set: {self.num_objects}")
        print(f"Total number of views per object: {self.num_views}")

    def __len__(self):
        return len(self.rgb_all_filenames)

    def __getitem__(self, index):
        object_index, rgb_filename = self.rgb_all_filenames[index]
        _, pose_filename = self.pose_all_filenames[index]
        intrinsic_filename = self.intrinsic[object_index]

        with Path(intrinsic_filename).open() as intrinsic_file:
            intrinsic_lines = intrinsic_file.readlines()
            focal, cx, cy, _ = map(float, intrinsic_lines[0].split())
            height, width = map(int, intrinsic_lines[-1].split())

        rgb_image = np.asarray(imageio.imread(rgb_filename))
        rgb_image = rgb_image / 255.0

        crop_height, crop_width = height//8, width//8
        rgb_image = rgb_image[crop_width:width-crop_width, crop_height:height-crop_height, ...]

        pose = np.loadtxt(pose_filename).reshape(4, 4)

        intrinsic = np.eye(4)
        intrinsic[0, 0], intrinsic[1, 1] = focal, focal
        intrinsic[0, 2], intrinsic[1, 2] = cx-crop_width, cy-crop_height

        intrinsic = intrinsic.astype(np.float32)
        rgb_image = rgb_image.astype(np.float32)
        rgb_image = rgb_image[..., :3] * rgb_image[..., -1:] + (1 - rgb_image[..., -1:])

        pose = pose.astype(np.float32)

        intrinsic = self.transform(intrinsic)
        rgb_image = self.transform(rgb_image)
        pose = self.transform(pose)

        ray_directions = compute_ray_directions(intrinsic[0])
        ray_origins, ray_directions = compute_rays(ray_directions, pose[0])
        rays = torch.cat([ray_origins, ray_directions], dim=-1)
        target_rgb = rgb_image.view(3, -1).T

        sample = {
            "rays": rays,
            "rgb": target_rgb,
            "rgb_image": rgb_image,
        }
        return sample


class SRNDataModule(object):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.data_train = SRNDataset(path=self.data_dir, stage="train")
        self.data_val = SRNDataset(path=self.data_dir, stage="val")

    def setup(self, rank=0, num_replicas=1, shuffle=True, seed=0):
        self.train_sampler = InfiniteSampler(self.data_train, rank=rank, num_replicas=num_replicas, shuffle=self.shuffle, seed=seed)
        self.val_sampler = InfiniteSampler(self.data_val, rank=rank, num_replicas=num_replicas, shuffle=self.shuffle, seed=seed)

        self.train_loader = DataLoader(
            dataset=self.data_train,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # shuffle=self.shuffle
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            sampler=self.val_sampler,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # shuffle=self.shuffle,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
