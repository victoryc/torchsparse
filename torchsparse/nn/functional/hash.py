from typing import Optional

import torch
import torchsparse_backend

__all__ = ['hash_build', 'hash_query']


def hash_build(coords: torch.Tensor, offsets: Optional[torch.Tensor] = None):
    coords = coords.int().contiguous()
    if offsets is not None:
        offsets = offsets.int().contiguous()

    if offsets is None:
        if coords.device.type == 'cuda':
            return torchsparse_backend.hash_build_cuda(coords)
        elif coords.device.type == 'cpu':
            return torchsparse_backend.hash_build_cpu(coords)
        else:
            return torchsparse_backend.hash_build_cpu(coords.cpu()).to(
                coords.device)
    else:
        if 'cuda' in str(coords.device):
            return torchsparse_backend.hash_build_kernel_cuda(
                coords.contiguous(), offsets.contiguous())
        elif 'cpu' in str(coords.device):
            return torchsparse_backend.hash_build_kernel_cpu(
                coords.int().contiguous(),
                offsets.int().contiguous())
        else:
            device = coords.device
            return torchsparse_backend.hash_build_kernel_cpu(
                coords.int().contiguous().cpu(),
                offsets.int().contiguous().cpu()).to(device)


def hash_query(queries: torch.Tensor,
               references: torch.Tensor) -> torch.Tensor:
    queries = queries.contiguous()
    references = references.contiguous()

    sizes = queries.size()
    queries = queries.view(-1)

    indices = torch.arange(len(references),
                           device=queries.device,
                           dtype=torch.long)

    if queries.device.type == 'cuda':
        outputs = torchsparse_backend.hash_query_cuda(queries, references,
                                                      indices)
    elif queries.device.type == 'cpu':
        outputs = torchsparse_backend.hash_query_cpu(queries, references,
                                                     indices)
    else:
        outputs = torchsparse_backend.hash_query_cpu(queries.cpu(),
                                                     references.cpu(),
                                                     indices.cpu())
        outputs = outputs.to(queries.device)

    outputs = (outputs - 1).view(*sizes)
    return outputs
