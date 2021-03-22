import torchsparse_backend

__all__ = ['spcount']


def spcount(idx, num):
    if 'cuda' in str(idx.device):
        outs = torchsparse_backend.count_forward(idx.contiguous(), num)
    else:
        outs = torchsparse_backend.cpu_count_forward(idx.contiguous(), num)
    return outs
