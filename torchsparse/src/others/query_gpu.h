#ifndef _SPARSE_QUERY
#define _SPARSE_QUERY
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <iostream>

at::Tensor hash_query_cuda(
    const at::Tensor hash_query,
    const at::Tensor hash_target,
    const at::Tensor idx_target
);
#endif