#ifndef _SPARSE_DEVOXELIZE
#define _SPARSE_DEVOXELIZE
#include <torch/torch.h>
#include <vector>

void devoxelize_wrapper(int N, int c, const int *indices, const float *weight, const float *feat, float *out);
void devoxelize_grad_wrapper(int N, int n, int c, const int *indices, const float *weight, const float *top_grad, float *bottom_grad);
at::Tensor devoxelize_forward_cuda(
    const at::Tensor feat,
    const at::Tensor indices,
    const at::Tensor weight);
at::Tensor devoxelize_backward_cuda(
    const at::Tensor top_grad,
    const at::Tensor indices,
    const at::Tensor weight,
    int n);
#endif