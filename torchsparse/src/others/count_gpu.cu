#include <cmath>
#include <stdio.h>
#include <stdlib.h>

__global__ void count_kernel(int N, const int *__restrict__ data,
                             int *__restrict__ out) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    if (data[i] >= 0)
      atomicAdd(&out[data[i]], 1);
  }
}

void count_wrapper(int N, const int *data, int *out) {
  count_kernel<<<ceil((double)N / 512), 512>>>(N, data, out);
}
