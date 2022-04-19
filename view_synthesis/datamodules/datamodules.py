from typing import Literal, Dict, Optional, Callable

from pathlib import Path

import numpy as np
import imageio
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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


class SRNDataset(torch.utils.data.Dataset):
    """
    Dataset from Scene Representation Networks (V. Sitzmann et al 2020)
    """

    def __init__(
        self,
        path: str,
        stage: Literal["train", "val"] = "train",
        transform: Optional[Callable] = None
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
        self.transform = transform
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
        mask_image = (rgb_image != 255).all(axis=-1)[..., None]
        rgb_image = rgb_image / 255.0
        mask_image = mask_image / 255.0

        crop_height, crop_width = height//8, width//8
        rgb_image = rgb_image[crop_width:width-crop_width, crop_height:height-crop_height, ...]
        mask_image = mask_image[crop_width:width-crop_width, crop_height:height-crop_height, ...]

        pose = np.loadtxt(pose_filename).reshape(4, 4)
        pose = pose @ np.diag([1, -1, -1, 1])

        intrinsic = np.eye(4)
        intrinsic[0, 0], intrinsic[1, 1] = focal, focal
        intrinsic[0, 2], intrinsic[1, 2] = cx-crop_width, cy-crop_height

        intrinsic = intrinsic.astype(np.float32)
        rgb_image = rgb_image.astype(np.float32)
        mask_image = mask_image.astype(np.float32)
        pose = pose.astype(np.float32)
        if self.transform is not None:
            intrinsic = self.transform(intrinsic)
            rgb_image = self.transform(rgb_image)
            mask_image = self.transform(mask_image)
            pose = self.transform(pose)

        sample = {
            "object_id": object_index,
            "intrinsic": intrinsic,
            "color": rgb_image,
            "mask": mask_image,
            "pose": pose,
        }
        return sample


class SRNDataModule(object):
    def __init__(
        self,
        data_dir: str = "data/",
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.data_train = SRNDataset(path=self.data_dir, stage="train", transform=self.transforms)
        self.data_val = SRNDataset(path=self.data_dir, stage="val", transform=self.transforms)

    def setup(self, rank=0, num_replicas=1, shuffle=True, seed=0):
        self.train_sampler = InfiniteSampler(self.data_train, rank=rank, num_replicas=num_replicas, shuffle=self.shuffle, seed=seed)
        self.val_sampler = InfiniteSampler(self.data_val, rank=rank, num_replicas=num_replicas, shuffle=self.shuffle, seed=seed)

        self.train_loader = DataLoader(
            dataset=self.data_train,
            sampler=self.train_sampler,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            sampler=self.val_sampler,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)

    def train_iterator(self):
        return self.train_iter

    def val_iterator(self):
        return self.val_iter


class CARLADataset(torch.utils.data.Dataset):
    """
    Dataset rendered from CARLA (GRAF: Schwarz et al. 2020)
    """

    def __init__(
        self, path: str,
        stage: Literal["train", "val"] = "train",
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
        self.intrinsic = np.eye(4)
        self.intrinsic[:3, :3] = intrinsic
        self.dataset_name = "carla-cars"
        self.stage = stage
        self.num_objects = 18

        print(f"Loading CARLA dataset {self.root_dir} name: {self.dataset_name}-{self.stage}")

        rgb_fnames = [f for f in self.rgb_dir.iterdir() if f.is_file and f.suffix in [".png", ".jpg"]]
        pose_fnames = [f for f in self.pose_dir.iterdir() if f.is_file and f.suffix == ".npy"]

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

        sample = {'color': torch.from_numpy(rgb).double(),
                  'pose': torch.from_numpy(pose).double(),
                  'intrinsic': torch.from_numpy(self.intrinsic).double()}

        return sample
