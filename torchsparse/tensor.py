import torch

__all__ = ['SparseTensor']


class SparseTensor:
    def __init__(self,
                 feats: torch.Tensor,
                 coords: torch.Tensor,
                 stride: int = 1) -> None:
        self.feats = feats
        self.coords = coords
        self.stride = stride
        self.cmaps = {}
        self.kmaps = {}

    @property
    def F(self):
        return self.feats

    @property
    def C(self):
        return self.coords

    @property
    def s(self):
        return self.stride

    def check(self):
        if self.stride not in self.cmaps:
            self.cmaps[self.stride] = self.coords

    def cuda(self):
        self.feats = self.feats.cuda()
        self.coords = self.coords.cuda()
        return self

    def detach(self):
        self.feats = self.feats.detach()
        self.coords = self.coords.detach()
        return self

    def to(self, device, non_blocking=True):
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        self.coords = self.coords.to(device, non_blocking=non_blocking)
        return self

    def __add__(self, other):
        tensor = SparseTensor(self.feats + other.F, self.coords, self.stride)
        tensor.cmaps = self.cmaps
        tensor.kmaps = self.kmaps
        return tensor
