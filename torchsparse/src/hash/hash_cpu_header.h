#ifndef _SPARSE_HASH_CPU
#define _SPARSE_HASH_CPU
#include <torch/torch.h>
#include <vector>

void cpu_hash_wrapper(int N, const int * data, long * out);
void cpu_kernel_hash_wrapper(int N, int K, const int * data, const int *kernel_offset, long int * out);

at::Tensor hash_build_cpu(
    const at::Tensor idx
);
at::Tensor hash_build_kernel_cpu(
    const at::Tensor idx,
    const at::Tensor kernel_offset
);

#endif