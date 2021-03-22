from torch.nn import functional as F

from ...tensor import SparseTensor

__all__ = ['relu', 'leaky_relu']


def relu(inputs: SparseTensor, inplace: bool = True) -> SparseTensor:
    coords, feats, stride = inputs.coords, inputs.feats, inputs.stride
    feats = F.relu(feats, inplace=inplace)
    outputs = SparseTensor(coords=coords, feats=feats, stride=stride)
    outputs.cmaps = inputs.cmaps
    outputs.kmaps = inputs.kmaps
    return outputs


def leaky_relu(inputs: SparseTensor,
               negative_slope: float = 0.1,
               inplace: bool = True) -> SparseTensor:
    coords, feats, stride = inputs.coords, inputs.feats, inputs.stride
    feats = F.leaky_relu(feats, inplace=inplace, negative_slope=negative_slope)
    outputs = SparseTensor(coords=coords, feats=feats, stride=stride)
    outputs.cmaps = inputs.cmaps
    outputs.kmaps = inputs.kmaps
    return outputs
