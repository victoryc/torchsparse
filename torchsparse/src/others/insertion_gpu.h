#ifndef _SPARSE_INSERT
#define _SPARSE_INSERT
#include <torch/torch.h>
#include <vector>


void insertion_wrapper(int N, int c, int s, const float * data, const int * idx, const int * counts, float * out);
void insertion_grad_wrapper(int N, int c, int s, const float * top_grad, const int * idx, const int * counts, float * bottom_grad);

at::Tensor voxelize_forward_cuda(
    const at::Tensor inputs,
    const at::Tensor idx,
    const at::Tensor counts
);
at::Tensor voxelize_backward_cuda(
    const at::Tensor top_grad,
    const at::Tensor idx,
    const at::Tensor counts,
    const int N
);
#endif