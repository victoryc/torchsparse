import numpy as np
import torch

__all__ = ['KernelRegion']


def get_kernel_offsets(kernel_size,
                       stride: int = 1,
                       dilation: int = 1) -> torch.Tensor:
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 3

    offsets = [
        np.arange(-kernel_size[k] // 2 + 1, kernel_size[k] // 2 + 1) * stride *
        dilation for k in range(3)
    ]

    # to be compatible with minkowskiengine
    if np.prod(kernel_size) % 2 == 1:
        offsets = [[x, y, z] for z in offsets[2] for y in offsets[1]
                   for x in offsets[0]]
    else:
        offsets = [[x, y, z] for x in offsets[0] for y in offsets[1]
                   for z in offsets[2]]

    offsets = torch.tensor(offsets, dtype=torch.int)
    return offsets


class KernelRegion:
    def __init__(self,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3

        kernel_volume = np.prod(kernel_size)
        if kernel_volume % 2 == 0:
            # even
            region_type = 0
        else:
            # odd
            region_type = 1

        self.region_type = region_type

        x_offset = (
            np.arange(-kernel_size[0] // 2 + 1, kernel_size[0] // 2 + 1) *
            stride * dilation).tolist()
        y_offset = (
            np.arange(-kernel_size[1] // 2 + 1, kernel_size[1] // 2 + 1) *
            stride * dilation).tolist()
        z_offset = (
            np.arange(-kernel_size[2] // 2 + 1, kernel_size[2] // 2 + 1) *
            stride * dilation).tolist()

        if self.region_type == 1:
            kernel_offset = [[x, y, z] for z in z_offset for y in y_offset
                             for x in x_offset]
        else:
            kernel_offset = [[x, y, z] for x in x_offset for y in y_offset
                             for z in z_offset]
        kernel_offset = np.array(kernel_offset)
        self.kernel_offset = torch.from_numpy(kernel_offset).int()

    def get_kernel_offset(self):
        return self.kernel_offset
