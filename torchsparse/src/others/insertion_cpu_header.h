#ifndef _SPARSE_INSERT_CPU
#define _SPARSE_INSERT_CPU
#include <torch/torch.h>
#include <vector>


at::Tensor voxelize_forward_cpu(
    const at::Tensor inputs,
    const at::Tensor idx,
    const at::Tensor counts
);
at::Tensor voxelize_backward_cpu(
    const at::Tensor top_grad,
    const at::Tensor idx,
    const at::Tensor counts,
    const int N
);
#endif