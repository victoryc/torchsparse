import torch
import torchsparse_backend
from torch.autograd import Function

__all__ = ['voxelize', 'devoxelize']


class Voxelize(Function):
    @staticmethod
    def forward(ctx, feat, idx, cnt):
        out = torchsparse_backend.voxelize_forward_cuda(
            feat.float().contiguous(),
            idx.int().contiguous(), cnt)
        ctx.for_backwards = (idx.int().contiguous(), cnt, feat.shape[0])
        return out

    @staticmethod
    def backward(ctx, top_grad):
        idx, cnt, N = ctx.for_backwards
        bottom_grad = torchsparse_backend.voxelize_backward_cuda(
            top_grad.float().contiguous(), idx, cnt, N)
        return bottom_grad, None, None


class Devoxelize(Function):
    @staticmethod
    def forward(ctx, feat, indices, weights):
        if 'cuda' in str(feat.device):
            out = torchsparse_backend.devoxelize_forward_cuda(
                feat.contiguous(),
                indices.contiguous().int(), weights.contiguous())
        else:
            out = torchsparse_backend.devoxelize_forward_cpu(
                feat.contiguous(),
                indices.contiguous().int(), weights.contiguous())

        ctx.for_backwards = (indices.contiguous().int(), weights,
                             feat.shape[0])

        return out

    @staticmethod
    def backward(ctx, grad_out):
        indices, weights, n = ctx.for_backwards

        if 'cuda' in str(grad_out.device):
            grad_features = torchsparse_backend.devoxelize_backward_cuda(
                grad_out.contiguous(), indices, weights, n)
        else:
            grad_features = torchsparse_backend.devoxelize_backward_cpu(
                grad_out.contiguous(), indices, weights, n)

        return grad_features, None, None


def voxelize(feat, idx, cnt):
    return Voxelize.apply(feat, idx, cnt)


def devoxelize(feat, indices, weights):
    return Devoxelize.apply(feat, indices, weights)


def calc_ti_weights(pc, idx_query, scale=1.0):
    with torch.no_grad():
        pc_grid = pc
        if scale != 1.:
            pc_floor = torch.floor(pc / scale) * scale
        else:
            pc_floor = torch.floor(pc)
        pc_ceil = pc_floor + scale

        x = pc_grid[:, 0].view(-1, 1)
        y = pc_grid[:, 1].view(-1, 1)
        z = pc_grid[:, 2].view(-1, 1)
        xf = pc_floor[:, 0].view(-1, 1)
        yf = pc_floor[:, 1].view(-1, 1)
        zf = pc_floor[:, 2].view(-1, 1)
        xc = pc_ceil[:, 0].view(-1, 1)
        yc = pc_ceil[:, 1].view(-1, 1)
        zc = pc_ceil[:, 2].view(-1, 1)

        xf, yf, zf = xf.float(), yf.float(), zf.float()
        xc, yc, zc = xc.float(), yc.float(), zc.float()

        w000 = (xc - x) * (yc - y) * (zc - z)
        w001 = (xc - x) * (yc - y) * (z - zf)
        w010 = (xc - x) * (y - yf) * (zc - z)
        w011 = (xc - x) * (y - yf) * (z - zf)
        w100 = (x - xf) * (yc - y) * (zc - z)
        w101 = (x - xf) * (yc - y) * (z - zf)
        w110 = (x - xf) * (y - yf) * (zc - z)
        w111 = (x - xf) * (y - yf) * (z - zf)

        all_weights = torch.cat(
            [w000, w001, w010, w011, w100, w101, w110, w111],
            1).transpose(1, 0).contiguous()
        if scale != 1:
            all_weights /= scale ** 3
        all_weights[idx_query == -1] = 0
        all_weights /= all_weights.sum(0) + 1e-8
    return all_weights
