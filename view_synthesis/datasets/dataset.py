from typing import List, Union, Dict, Literal, Tuple

from pathlib import Path
import matplotlib.pyplot as plt
import json

import numpy as np
import imageio
import torch


class SRNDataset(torch.utils.data.Dataset):
    """
    Dataset from Scene Representation Networks (V. Sitzmann et al 2020)
    """

    def __init__(
        self, path: str,
        stage: Literal["train", "val", "test"] = "train",
        image_size: Tuple[int, int] = (128, 128),
        world_scale=1.0,
    ):
        """
        Args:
            stage: train | val | test
            image_size: result image size (resizes if different)
            world_scale: amount to scale entire world by
        """
        super(SRNDataset, self).__init__()
        self.base_path = Path(path)
        self.dataset_name = self.base_path.stem.split("_")[-1]
        if stage == "train":
            self.base_path = self.base_path / f"{self.dataset_name}_val"
        elif stage == "val":
            self.base_path = self.base_path / f"{self.dataset_name}_train"
        else:
            self.base_path = self.base_path / f"{self.dataset_name}_{stage}"

        self.image_size = image_size
        self.world_scale = world_scale

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert self.base_path.exists(), f"{self.base_path} does not exist"

        is_chair = "chair" in self.dataset_name
        if is_chair and stage == "train":
            # Ugly thing from SRN's public dataset
            tmp = self.base_path / "chairs_2.0_train"
            if tmp.exists():
                self.base_path = tmp

        if stage == "train":
            self.intrinsic = sorted(self.base_path.glob("*/intrinsics.txt"))[:30]
            self.num_objects = len(self.intrinsic)
        # TODO: Remove this
        elif stage == "val":
            self.intrinsic = sorted(self.base_path.glob("*/intrinsics.txt"))[:1]
            self.num_objects = len(self.intrinsic)
        else:
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
        mask_image = (rgb_image != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
        rgb_image = rgb_image / 255.0
        mask_image = mask_image / 255.0

        crop_height, crop_width = height//8, width//8
        rgb_image = rgb_image[crop_width:width-crop_width, crop_height:height-crop_height, ...]
        mask_image = mask_image[crop_width:width-crop_width, crop_height:height-crop_height, ...]

        pose = np.loadtxt(pose_filename).reshape(4, 4)
        pose = pose @ np.diag([1, -1, -1, 1])

        # if rgb_image.shape[-2:] != self.image_size:
        #     scale = self.image_size[0] / rgb_image.shape[-2]
        #     focal *= scale
        #     cx *= scale
        #     cy *= scale

        if self.world_scale != 1.0:
            focal *= self.world_scale
            pose[:3, 3] *= self.world_scale

        intrinsic = np.eye(4)
        intrinsic[0, 0], intrinsic[1, 1] = focal, focal
        intrinsic[0, 2], intrinsic[1, 2] = cx-crop_width, cy-crop_height

        sample = {
            "object_id": object_index,
            "intrinsic": intrinsic.astype(np.float32),
            "color": rgb_image.astype(np.float32),
            "mask": mask_image.astype(np.float32),
            "pose": pose.astype(np.float32),
        }
        return sample


# if __name__ == "__main__":
    # Test Dataset
    # car_interior_dataset = CarInteriorDataset(
    #     root_dir="/home/fyusion/Documents/datasets/bmw-simulated", resolution_level=2)
    #
    # print(f"Length of the dataset: {len(car_interior_dataset)}")
    #
    # train_size = int(len(car_interior_dataset) * 0.75)
    # test_size = len(car_interior_dataset) - train_size
    #
    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     car_interior_dataset, [train_size, test_size])
    #
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=1, shuffle=True, num_workers=4)
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=1, shuffle=True, num_workers=4)
    #
    # for i, sample in enumerate(train_dataloader):
    #     sample = car_interior_dataset[i]
    #
    #     Path("./dataset_test").mkdir(parents=True, exist_ok=True)
    #     plt.imsave(f"./dataset_test/sample_color_{i}.png", sample['color'])
    #     plt.imsave(f"./dataset_test/sample_depth_{i}.png", sample['depth'])
    #     plt.imsave(f"./dataset_test/sample_normal_{i}.png", sample['normal'])
    #
    #     print(f"Pose of sample {i}:\n {sample['pose']}")
    #     print(f"Intrinsic of sample {i}:\n {sample['intrinsic']}")
    #
    #     if i > 10:
    #         break
    #
    # for i, sample in enumerate(test_dataloader):
    #     sample = car_interior_dataset[i]
    #
    #     Path("./dataset_test").mkdir(parents=True, exist_ok=True)
    #     plt.imsave(
    #         f"./dataset_test/test_sample_color_{i}.png", sample['color'])
    #     plt.imsave(
    #         f"./dataset_test/test_sample_depth_{i}.png", sample['depth'])
    #     plt.imsave(
    #         f"./dataset_test/test_sample_normal_{i}.png", sample['normal'])
    #
    #     print(f"Pose of sample {i}:\n {sample['pose']}")
    #     print(f"Intrinsic of sample {i}:\n {sample['intrinsic']}")
    #
    #     if i > 10:
    #         break
