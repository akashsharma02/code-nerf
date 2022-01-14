from typing import Tuple, Union, Literal
from numpy.typing import DTypeLike
import numpy as np
import torch


class RaySampler(object):

    """RaySampler samples rays for a given image size and intrinsics """

    def __init__(self, height: int, width: int, intrinsics: Union[torch.Tensor, np.ndarray], sample_size: int, device: torch.cuda.Device):
        """ Prepares a ray bundle for a given image size and intrinsics

        :Function: intrinsics: torch.Tensor 4x4

        """
        assert height > 0 and width > 0, "Height and width must be positive integers"
        assert sample_size > 0 and sample_size < height * width, "Sample size must be a positive number less than height * width"

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

        ii, jj = torch.meshgrid(
            torch.arange(
                width, dtype=self.intrinsics.dtype, device=self.device
            ),
            torch.arange(
                height, dtype=self.intrinsics.dtype, device=self.device
            ),
            indexing='xy'
        )
        self.directions = torch.stack(
            [
                (ii - width * 0.5) / self.focal_length,
                -(jj - height * 0.5) / self.focal_length,
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
        # ro, rd = ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3)

        select_inds = np.array([np.random.choice(ray_origins.shape[-2], self.sample_size, replace=False) for _ in range(batch_size)])

        # print(select_inds, select_inds.shape)
        ray_origins = [ray_origins[i, select_inds[i], :] for i in range(batch_size)]
        ray_directions = [ray_directions[i, select_inds[i], :] for i in range(batch_size)]
        ray_origins = torch.cat(ray_origins, dim=0).reshape(-1, 3)
        ray_directions = torch.cat(ray_directions, dim=0).reshape(-1, 3)

        # select_inds = np.random.choice(
        #     ro.shape[-2], size=(
        #         self.sample_size*batch_size), replace=False
        # )

        # ray_origins = ro[select_inds, :]
        # ray_directions = rd[select_inds, :]

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
        ray_directions = torch.tensordot(
            tform_cam2world[..., :3, :3],
            self.directions.T,
            dims=([2], [0])
        ).transpose(1, -1).contiguous()

        ray_origins = tform_cam2world[..., :3, -1][:, None, None, :].expand(ray_directions.shape)
        return ray_origins, ray_directions


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
        dataset, batch_size=1, shuffle=True, num_workers=0)

    first_data_sample = next(iter(dataloader))

    (height,
     width), intrinsic = first_data_sample["color"].shape[1:-1], first_data_sample["intrinsic"]

    print(f" Height: {height}, Width: {width}, Intrinsics:\n {intrinsic}")

    device = None
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    ray_sampler = RaySampler(height, width, intrinsic[0], sample_size=args.num_random_rays, device=device)
    pose = first_data_sample["pose"].to(device)
    ro, rd = ray_sampler.get_bundle(pose)
    ray_origins, ray_directions, select_inds = ray_sampler.sample(
        tform_cam2world=pose)

    print(
        f"Ray bundle shape: {ray_origins.shape}, {ray_directions.shape}, {select_inds.shape}")
    print(f"Ray origins:\n {ray_origins}")
    print(f"Ray directions:\n {ray_directions}")
    print(f"select indices:\n {select_inds}")
    print(f"Pose origin:\n {pose[:, :3, 3]}")
    print(f"Pose:\n {pose}")
