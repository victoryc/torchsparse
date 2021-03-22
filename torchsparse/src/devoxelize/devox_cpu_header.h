#ifndef _SPARSE_DEVOXELIZE_CPU
#define _SPARSE_DEVOXELIZE_CPU
#include <torch/torch.h>
#include <vector>

at::Tensor devoxelize_forward_cpu(
    const at::Tensor feat,
    const at::Tensor indices,
    const at::Tensor weight);
at::Tensor devoxelize_backward_cpu(
    const at::Tensor top_grad,
    const at::Tensor indices,
    const at::Tensor weight,
    int n);

#endif