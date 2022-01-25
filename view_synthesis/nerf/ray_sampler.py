from typing import Tuple, Union
import numpy as np
import torch


class RaySampler(object):

    """RaySampler samples rays for a given image size and intrinsics """

    def __init__(self, height: int, width: int, intrinsics: Union[torch.Tensor, np.ndarray], sample_size: int, device: torch.cuda.Device):
        """ Prepares a ray bundle for a given image size and intrinsics

        :Function: intrinsics: torch.Tensor 4x4

        """
        assert height > 0 and width > 0, "Height and width must be positive integers"
        assert sample_size > 0 and sample_size <= height * width, "Sample size must be a positive number less than height * width"

        self.height = height
        self.width = width
        self.sample_size = sample_size
        self.device = device

        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics)
        assert intrinsics.shape == torch.Size(
            [4, 4]), "Incorrect intrinsics shape"
        self.intrinsics = intrinsics
        self.intrinsics = self.intrinsics.to(device)
        self.focal_length = self.intrinsics[..., 0, 0]
        self.cx = self.intrinsics[..., 0, 2]
        self.cy = self.intrinsics[..., 1, 2]

        ii, jj = meshgrid_xy(
            torch.arange(
                width, dtype=self.intrinsics.dtype, device=self.device
            ),
            torch.arange(
                height, dtype=self.intrinsics.dtype, device=self.device
            ),
        )
        self.directions = torch.stack(
            [
                (ii - self.cx) / self.focal_length,
                -(jj - self.cy) / self.focal_length,
                -torch.ones_like(ii),
            ],
            dim=-1,
        )

    def sample(self, tform_cam2world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Rotate the bundle of rays given the camera pose and return a random subset of rays

        :function:
            tform_cam2world: [batch, 4, 4] torch.Tensor camera pose (SE3)
        :returns:
            ray origins: torch.Tensor [batch_size*sample_size, 3]
            ray directions: torch.Tensor [batch_size*sample_size, 3]
            select_inds: np.ndarray [batch_size*sample_size]

        """
        # TODO: Returns same sample across batch
        batch_size = tform_cam2world.shape[0]

        ray_origins, ray_directions = self.get_bundle(tform_cam2world)
        ray_origins, ray_directions = ray_origins.flatten(1, 2), ray_directions.flatten(1, 2)

        select_inds = []
        pixel_range = np.arange(0, ray_origins.shape[-2])
        for _ in range(batch_size):
            select_inds.append(np.random.permutation(pixel_range)[:self.sample_size])
        select_inds = np.asarray(select_inds)

        ray_origins = [ray_origins[i, select_inds[i], :] for i in range(batch_size)]
        ray_directions = [ray_directions[i, select_inds[i], :] for i in range(batch_size)]
        ray_origins = torch.cat(ray_origins, dim=0)
        ray_directions = torch.cat(ray_directions, dim=0)

        return ray_origins, ray_directions, select_inds

    def get_bundle(self, tform_cam2world: torch.Tensor):
        """
            Rotate the bundle of rays given the camera pose

        :function:
            tform_cam2world: 4x4 torch.Tensor camera pose (SE3)
        :returns:
            ray origins: torch.Tensor [batch, H, W, 3]
            ray directions: torch.Tensor [batch, H, W, 3]

        """
        directions = self.directions[..., None]

        ray_directions = torch.einsum('hwij, bji->bhwj', directions, tform_cam2world[..., :3, :3]).contiguous()
        ray_origins = tform_cam2world[..., :3, -1][:, None, None, :].expand(ray_directions.shape)
        return ray_origins, ray_directions


def meshgrid_xy(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


if __name__ == "__main__":
    # Manual testing of RaySampler and PointSampler
    import argparse
    parser = argparse.ArgumentParser("Test RaySampler")
    parser.add_argument("--dataset-dir", type=str,
                        help="Directory for the BlenderNeRF dataset root", required=True)
    parser.add_argument("--num-random-rays", type=int,
                        help="Number of random rays to sample", required=True)

    args = parser.parse_args()
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    from view_synthesis.datasets.dataset import BlenderNeRFDataset
    dataset = BlenderNeRFDataset(args.dataset_dir, resolution_level=32, mode="val")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0)

    first_data_sample = next(iter(dataloader))

    (height,
     width), intrinsic = first_data_sample["color"].shape[1:-1], first_data_sample["intrinsic"]

    print(f" Height: {height}, Width: {width}, Intrinsics:\n {intrinsic}")
    target_image_pixels = first_data_sample["color"].flatten(1, 2)
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    ray_sampler = RaySampler(height, width, intrinsic[0], sample_size=args.num_random_rays, device=device)

    print(f"Ray sampler directions: {ray_sampler.directions.T.shape}")
    pose = first_data_sample["pose"].to(device)
    ro, rd = ray_sampler.get_bundle(pose)
    ray_origins, ray_directions, select_inds = ray_sampler.sample(
        tform_cam2world=pose)

    print(
        f"Ray bundle shape: {ray_origins.shape}, {ray_directions.shape}, {select_inds.shape}")
    print(f"Ray origins:\n {ray_origins}")
    print(f"Ray directions:\n {ray_directions}")
    target_pixels = [target_image_pixels[i, select_inds[i], :] for i in range(2)]

    print(f"Target pixels for 1st batch: \n {target_image_pixels[0, select_inds[0], :]}")
    print(f"Target pixels for 2nd batch: \n {target_image_pixels[1, select_inds[1], :]}")

    target_pixels = torch.cat(target_pixels, dim=0)
    print(f"Target pixels: {target_pixels}, {target_pixels.shape}")

    print(f"select indices:\n {select_inds}")
    print(f"Pose origin:\n {pose[:, :3, 3]}")
    print(f"Pose:\n {pose}")
