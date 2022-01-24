from typing import List, Union, Dict, Literal, Tuple

from pathlib import Path
import matplotlib.pyplot as plt
import json

import numpy as np
import imageio
import cv2
import torch


class CarInteriorDataset(torch.utils.data.Dataset):

    """Docstring for CarInteriorDataset. """

    def __init__(self, root_dir: str, resolution_level: int = 1, max_depth: float = 5):
        """
        Args:
            root_dir: Path to the root directory containing the data
            resolution_level: resolution level to resize the images

        """
        self.root_dir = Path(root_dir)
        self.color_img_dir = self.root_dir / "color"
        self.depth_img_dir = self.root_dir / "depth"
        self.normal_img_dir = self.root_dir / "normals"
        self.pose_dir = self.root_dir / "poses"
        self.max_depth = max_depth

        self.intrinsic = np.loadtxt(
            self.root_dir / "intrinsics.txt").reshape(4, 4)

        self.color_img_fnames = np.array([f for f in self.color_img_dir.iterdir(
        ) if f.is_file() and f.suffix in [".png", ".jpg"]])
        self.depth_img_fnames = np.array([f for f in self.depth_img_dir.iterdir(
        ) if f.is_file() and f.suffix in [".png", ".jpg"]])
        self.normal_img_fnames = np.array([f for f in self.normal_img_dir.iterdir(
        ) if f.is_file() and f.suffix in [".png", ".jpg"]])
        self.pose_fnames = np.array(
            [f for f in self.pose_dir.iterdir() if f.is_file() and f.suffix == ".txt"])

        msg = "Number of images between different folders are inconsistent"
        assert len(self.color_img_fnames) == len(self.depth_img_fnames) == len(
            self.normal_img_fnames) == len(self.pose_fnames), msg

        self.length = len(self.color_img_fnames)

        assert resolution_level > 0, "Resolution level needs to be a positive integer"
        self.resolution_level = resolution_level

    def __len__(self):
        """ Returns length of the dataset

        :function: TODO
        :returns: int

        """
        return self.length

    def __getitem__(self, idx: Union[int, List[int], torch.Tensor, List[torch.Tensor]]) -> Dict:
        """TODO: Docstring for __get_item__.

        :function: TODO
        :returns: TODO

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        color_img_fname = self.color_img_fnames[idx]
        depth_img_fname = self.depth_img_fnames[idx]
        normal_img_fname = self.normal_img_fnames[idx]
        pose_fname = self.pose_fnames[idx]

        color_img = imageio.imread(color_img_fname)
        depth_img = cv2.imread(str(depth_img_fname), cv2.IMREAD_UNCHANGED)
        normal_img = imageio.imread(normal_img_fname)
        pose = np.loadtxt(pose_fname).reshape(4, 4)

        color_img = color_img / 255.0
        depth_img = depth_img * self.max_depth / 65535.0
        depth_img = depth_img[..., 0][..., None]
        normal_img = normal_img / 255.0

        sample = {'color': color_img.astype(np.float32),
                  'depth': depth_img.astype(np.float32),
                  'normal': normal_img.astype(np.float32),
                  'pose': pose.astype(np.float32),
                  'intrinsic': np.copy(self.intrinsic).astype(np.float32)}

        if self.resolution_level != 1:
            H, W = color_img.shape[:2]
            H, W = H // self.resolution_level, W // self.resolution_level

            sample['intrinsic'][:2, :3] = sample['intrinsic'][:2,
                                                              :3] // self.resolution_level
            sample['color'] = cv2.resize(
                sample['color'], (W, H), interpolation=cv2.INTER_AREA)
            sample['depth'] = cv2.resize(
                sample['depth'], (W, H), interpolation=cv2.INTER_NEAREST)
            sample['normal'] = cv2.resize(
                sample['normal'], (W, H), interpolation=cv2.INTER_NEAREST)

        return sample


class BlenderNeRFDataset(torch.utils.data.Dataset):

    """Docstring for BlenderNeRFDataset. """

    def __init__(self, root_dir: str, resolution_level: int = 1, mode: Literal["train", "test", "val"] = "train"):
        """
        Args:
            root_dir: Path to the root directory containing the data
            resolution_level: resolution level to resize the images

        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / mode
        self.pose_fname = self.root_dir / f"transforms_{mode}.json"
        assert self.img_dir.exists() and self.img_dir.is_dir(
        ), "Incorrect mode, should be either train, test, or val"
        assert self.pose_fname.exists() and self.pose_fname.is_file(
        ), "Pose file name transforms_{mode}.json does not exist"

        metadata = None
        with open(self.pose_fname, "r") as fp:
            metadata = json.load(fp)

        self.img_fnames, self.poses = [], []
        for frame in metadata["frames"]:
            self.img_fnames.append(
                self.root_dir / (frame["file_path"] + ".png"))
            self.poses.append(np.array(frame["transform_matrix"]))

        self.img_fnames = np.array(self.img_fnames)
        self.poses = np.array(self.poses)

        height, width = imageio.imread(self.img_fnames[0]).shape[:2]
        camera_angle_x = metadata["camera_angle_x"]
        focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
        self.intrinsic = np.array([[focal, 0, width/2.0, 0],
                                   [0, focal, height/2.0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

        msg = "Inconsistent number of images or poses in the dataset folder"
        assert len(self.img_fnames) == len(self.poses), msg

        self.length = len(self.img_fnames)

        assert resolution_level > 0, "Resolution level needs to be a positive integer"
        self.resolution_level = resolution_level

    def __len__(self):
        """ Returns length of the dataset

        :function: TODO
        :returns: int

        """
        return self.length

    def __getitem__(self, idx: Union[int, List[int], torch.Tensor, List[torch.Tensor]]) -> Dict:
        """TODO:
        Get a sample from the dataset given an index or index list

        :function: TODO
        :returns: TODO

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        color_img_fname = self.img_fnames[idx]
        pose = self.poses[idx].reshape(4, 4)

        assert np.isclose(np.linalg.det(
            pose[:3, :3]), 1), "Incorrect rotation does not determinant = 1"

        color_img = imageio.imread(color_img_fname)
        color_img = (color_img / 255.0)

        sample = {'color': color_img.astype(np.float32), 'pose': pose.astype(np.float32), 'intrinsic': np.copy(self.intrinsic).astype(np.float32)}

        if self.resolution_level != 1:
            H, W = color_img.shape[:2]
            H, W = H // self.resolution_level, W // self.resolution_level

            sample['intrinsic'][:2, :3] = sample['intrinsic'][:2,
                                                              :3] // self.resolution_level
            sample['color'] = cv2.resize(
                sample['color'], (W, H), interpolation=cv2.INTER_AREA)

        return sample


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
        # if stage == "train":
        #     self.base_path = self.base_path / f"{self.dataset_name}_val"
        # elif stage == "val":
        #     self.base_path = self.base_path / f"{self.dataset_name}_train"
        # else:
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
            self.intrinsic = sorted(self.base_path.glob("*/intrinsics.txt"))[:1]
            self.num_objects = len(self.intrinsic)
        # TODO: Remove this
        elif stage == "val":
            self.intrinsic = sorted(self.base_path.glob("*/intrinsics.txt"))[:1]
            self.num_objects = len(self.intrinsic)
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


if __name__ == "__main__":
    # Test Dataset
    car_interior_dataset = CarInteriorDataset(
        root_dir="/home/fyusion/Documents/datasets/bmw-simulated", resolution_level=2)

    print(f"Length of the dataset: {len(car_interior_dataset)}")

    train_size = int(len(car_interior_dataset) * 0.75)
    test_size = len(car_interior_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        car_interior_dataset, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_dataloader):
        sample = car_interior_dataset[i]

        Path("./dataset_test").mkdir(parents=True, exist_ok=True)
        plt.imsave(f"./dataset_test/sample_color_{i}.png", sample['color'])
        plt.imsave(f"./dataset_test/sample_depth_{i}.png", sample['depth'])
        plt.imsave(f"./dataset_test/sample_normal_{i}.png", sample['normal'])

        print(f"Pose of sample {i}:\n {sample['pose']}")
        print(f"Intrinsic of sample {i}:\n {sample['intrinsic']}")

        if i > 10:
            break

    for i, sample in enumerate(test_dataloader):
        sample = car_interior_dataset[i]

        Path("./dataset_test").mkdir(parents=True, exist_ok=True)
        plt.imsave(
            f"./dataset_test/test_sample_color_{i}.png", sample['color'])
        plt.imsave(
            f"./dataset_test/test_sample_depth_{i}.png", sample['depth'])
        plt.imsave(
            f"./dataset_test/test_sample_normal_{i}.png", sample['normal'])

        print(f"Pose of sample {i}:\n {sample['pose']}")
        print(f"Intrinsic of sample {i}:\n {sample['intrinsic']}")

        if i > 10:
            break
