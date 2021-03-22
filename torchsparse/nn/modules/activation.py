from torch import nn

from ...tensor import SparseTensor

__all__ = ['ReLU', 'LeakyReLU']


class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__(inplace=inplace)

    def forward(self, inputs):
        coords, feats, stride = inputs.coords, inputs.feats, inputs.stride
        feats = super().forward(feats)
        outputs = SparseTensor(coords=coords, feats=feats, stride=stride)
        outputs.cmaps = inputs.cmaps
        outputs.kmaps = inputs.kmaps
        return outputs


class LeakyReLU(nn.LeakyReLU):
    def __init__(self,
                 negative_slope: float = 0.1,
                 inplace: bool = True) -> None:
        super().__init__(negative_slope=negative_slope, inplace=inplace)

    def forward(self, inputs):
        coords, feats, stride = inputs.coords, inputs.feats, inputs.stride
        feats = super().forward(feats)
        outputs = SparseTensor(coords=coords, feats=feats, stride=stride)
        outputs.cmaps = inputs.cmaps
        outputs.kmaps = inputs.kmaps
        return outputs
