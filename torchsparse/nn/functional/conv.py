import itertools
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchsparse_backend
from torch.autograd import Function

from ... import SparseTensor
from .. import functional as F

__all__ = ['conv3d']


class SparseConv(Function):
    @staticmethod
    def forward(ctx,
                feats,
                kernel,
                neighbor_map,
                neighbor_offset,
                sizes,
                transpose: bool = False):
        feats = feats.contiguous()
        kernel = kernel.contiguous()
        if not transpose:
            out = torch.zeros(sizes[1], kernel.size(-1), device=feats.device)
        else:
            # tbd: ensure the original, upsampled size to be the same.
            out = torch.zeros(sizes[0], kernel.size(-1), device=feats.device)

        if feats.device.type == 'cuda':
            torchsparse_backend.sparseconv_forward(feats, out, kernel,
                                                   neighbor_map,
                                                   neighbor_offset, transpose)
        else:
            # use the native pytorch XLA APIs for the TPU.
            cur_st = 0
            for kernel_idx in range(kernel.shape[0]):
                cur_ed = cur_st + neighbor_offset[kernel_idx]
                in_map = neighbor_map[cur_st:cur_ed, 0].long()
                out_map = neighbor_map[cur_st:cur_ed, 1].long()
                cur_st += neighbor_offset[kernel_idx]

                if transpose:
                    in_map, out_map = out_map, in_map
                # gather
                cur_feat = feats[in_map]
                # gemm
                cur_feat = torch.mm(cur_feat, kernel[kernel_idx])
                # scatter
                out[out_map] += cur_feat

        ctx.for_backwards = (feats, kernel, neighbor_map, neighbor_offset,
                             transpose)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, kernel, neighbor_map, neighbor_offset, transpose = ctx.for_backwards
        K, c_in, c_out = kernel.size()
        N_in = features.size(0)
        grad_features = torch.zeros(N_in, c_in, device=features.device)
        grad_kernel = torch.zeros(K, c_in, c_out, device=kernel.device)

        if features.device.type == 'cuda':
            torchsparse_backend.sparseconv_backward(features, grad_features,
                                                    grad_out.contiguous(),
                                                    kernel, grad_kernel,
                                                    neighbor_map,
                                                    neighbor_offset, transpose)
        else:
            raise NotImplementedError
        return grad_features, grad_kernel, None, None, None, None


sparse_conv = SparseConv.apply


def conv3d(inputs: SparseTensor,
           kernel: torch.Tensor,
           kernel_size: Union[int, Tuple[int, int, int]],
           bias: Optional[torch.Tensor] = None,
           stride: int = 1,
           dilation: int = 1,
           transpose: bool = False) -> SparseTensor:
    coords, feats = inputs.coords, inputs.feats

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 3
    assert isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 3, \
        'kernel_size must be either an integer or a triple of integers'
    kernel_volume = int(np.prod(kernel_size))

    if kernel_volume == 1 and stride == 1 and dilation == 1:
        feats = feats.matmul(kernel)
        if bias is not None:
            feats += bias
        outputs = SparseTensor(coords=coords,
                               feats=feats,
                               stride=inputs.stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.coord_maps[outputs.stride] = outputs.coords
        outputs.kernel_maps = inputs.kernel_maps
        return outputs

    if not transpose:
        kernel_map = inputs.kernel_maps.get(
            (inputs.stride, kernel_size, stride, dilation))

        if kernel_map is None:
            offsets = [
                np.arange(-kernel_size[k] // 2 + 1, kernel_size[k] // 2 + 1) *
                stride * dilation for k in range(3)
            ]
            # to be compatible with minkowskiengine
            if kernel_volume % 2 == 1:
                offsets = itertools.product(offsets[2], offsets[1], offsets[0])
            else:
                offsets = itertools.product(offsets[0], offsets[1], offsets[2])
            offsets = torch.tensor(list(offsets),
                                   dtype=torch.int,
                                   device=feats.device)

            references = F.sphash(coords)
            if stride > 1:
                coords = F.spdownsample(coords, stride * inputs.stride)
            queries = F.sphash(coords, offsets)

            idx_query = F.sphashquery(queries, references)
            idx_query = list(F.convert_neighbor_map(idx_query))
            idx_query[1] = idx_query[1].cpu()
            kernel_map = idx_query + [(feats.shape[0], coords.shape[0])]

        feats = sparse_conv(feats, kernel, kernel_map[0], kernel_map[1],
                            kernel_map[2], transpose)
        if bias is not None:
            feats += bias

        outputs = SparseTensor(coords=coords,
                               feats=feats,
                               stride=inputs.stride * stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.coord_maps[outputs.stride] = outputs.coords
        outputs.kernel_maps = inputs.kernel_maps
        outputs.kernel_maps[(inputs.stride, kernel_size, stride,
                             dilation)] = kernel_map
    else:
        # do upsample
        original_stride = int(inputs.stride / stride)
        kernel_map = inputs.kernel_maps.get(
            (original_stride, kernel_size, stride, dilation))
        feats = sparse_conv(feats, kernel, kernel_map[0], kernel_map[1],
                            kernel_map[2], transpose)
        if bias is not None:
            feats += bias
        outputs = SparseTensor(coords=inputs.coord_maps[original_stride],
                               feats=feats,
                               stride=original_stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.coord_maps[outputs.stride] = outputs.coords
        outputs.kernel_maps = inputs.kernel_maps

    return outputs
