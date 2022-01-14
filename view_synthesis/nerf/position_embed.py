from numpy.typing import DTypeLike
import torch
import numpy as np


class PositionalEmbedder(object):

    """Positionally encode the input vector through fourier basis with given frequency bands"""

    def __init__(self, num_freq: int, log_sampling: bool, include_input: bool, dtype: DTypeLike, device: torch.cuda.Device) -> None:
        assert num_freq > 0, "Number of frequency samples should be a positive integer"
        self.num_freq = num_freq
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.dtype = dtype
        self.device = device

        self.frequency_bands = None
        if self.log_sampling:
            self.frequency_bands = 2.0 ** torch.linspace(
                0.0,
                self.num_freq - 1,
                self.num_freq,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (self.num_freq - 1),
                self.num_freq,
                dtype=self.dtype,
                device=self.device,
            )

    def embed(self, tensor: torch.Tensor) -> torch.Tensor:
        """

        :function:
            tensor: torch.Tensor [:, num_dim] Tensor to be positionally embedded
        :returns:
            tensor: torch.Tensor [:, num_dim*num_freq*2] Positionally embedded feature vector

        """
        encoding = [tensor] if self.include_input else []

        for i, freq in enumerate(self.frequency_bands):
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


if __name__ == "__main__":
    # TODO: Test positional embedding
    pass
