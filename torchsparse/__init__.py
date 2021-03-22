import torch

from .point_tensor import *
from .tensor import SparseTensor

__version__ = '1.2.0'


def cat(input_list, dim=1):
    assert len(input_list) > 0
    inputs = input_list[0]
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s
    output_tensor = SparseTensor(
        torch.cat([inputs.F for inputs in input_list], 1), coords, cur_stride)
    output_tensor.cmaps = inputs.cmaps
    output_tensor.kmaps = inputs.kmaps
    return output_tensor
