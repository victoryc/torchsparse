from torch import nn

from ... import SparseTensor
from .. import functional as F

__all__ = ['GlobalAveragePooling', 'GlobalMaxPooling']


class GlobalAveragePooling(nn.Module):
    def forward(self, inputs: SparseTensor) -> SparseTensor:
        return F.global_avg_pool(inputs)


class GlobalMaxPooling(nn.Module):
    def forward(self, inputs: SparseTensor) -> SparseTensor:
        return F.global_max_pool(inputs)
