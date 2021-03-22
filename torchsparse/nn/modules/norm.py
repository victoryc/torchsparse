from typing import List, Union

import torch
from torch import nn

from ...tensor import SparseTensor

__all__ = ['BatchNorm', 'LayerNorm']


class BatchNorm(nn.BatchNorm1d):
    def __init__(self,
                 num_features: int,
                 *,
                 eps: float = 1e-5,
                 momentum: float = 0.1) -> None:
        super().__init__(num_features=num_features, eps=eps, momentum=momentum)

    def forward(self, inputs):
        coords, feats, stride = inputs.coords, inputs.feats, inputs.stride
        feats = super().forward(feats)
        outputs = SparseTensor(coords=coords, feats=feats, stride=stride)
        outputs.cmaps = inputs.cmaps
        outputs.kmaps = inputs.kmaps
        return outputs


class LayerNorm(nn.LayerNorm):
    def __init__(self,
                 normalized_shape: Union[int, List[int], torch.Size],
                 *,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape,
                         eps=eps,
                         elementwise_affine=elementwise_affine)

    def forward(self, inputs):
        coords, feats, stride = inputs.coords, inputs.feats, inputs.stride
        feats = super().forward(feats)
        outputs = SparseTensor(coords=coords, feats=feats, stride=stride)
        outputs.cmaps = inputs.cmaps
        outputs.kmaps = inputs.kmaps
        return outputs
