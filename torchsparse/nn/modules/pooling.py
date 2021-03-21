from torch import nn

from .. import functional as F


class GlobalAveragePooling(nn.Module):
    def forward(self, inputs):
        return F.global_avg_pool(inputs)


class GlobalMaxPooling(nn.Module):
    def forward(self, inputs):
        return F.global_max_pool(inputs)
