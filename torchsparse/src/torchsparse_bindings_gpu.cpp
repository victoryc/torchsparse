#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "convolution/convolution_cpu_header.h"
#include "hash/hash_cpu_header.h"
#include "others/insertion_cpu_header.h"
#include "others/query_cpu_header.h"
#include "convolution/convolution_gpu.h"
#include "hash/hash_gpu.h"
#include "devoxelize/devoxelize_cpu.h"
#include "devoxelize/devoxelize_cuda.h"
#include "others/count_gpu.h"
#include "others/insertion_gpu.h"
#include "others/insertion_cpu_header.h"
#include "others/query_gpu.h"
#include "others/count_cpu_header.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparseconv_forward_cpu", &sparseconv_forward_cpu, "point cloud convolution forward (CPU)");
    m.def("sparseconv_forward_cuda", &sparseconv_forward_cuda, "point cloud convolution forward (CUDA)");
    m.def("sparseconv_backward_cpu", &sparseconv_backward_cpu, "point cloud convolution backward (CPU)");
    m.def("sparseconv_backward_cuda", &sparseconv_backward_cuda, "point cloud convolution backward (CUDA)");
    m.def("hash_build_cpu", &hash_build_cpu, "Hashing forward (CPU)");
    m.def("hash_build_cuda", &hash_build_cuda, "Hashing forward (CUDA)");
    m.def("hash_build_kernel_cpu", &hash_build_kernel_cpu, "Kernel Hashing forward (CPU)");
    m.def("hash_build_kernel_cuda", &hash_build_kernel_cuda, "Kernel Hashing forward (CUDA)");
    m.def("hash_query_cpu", &hash_query_cpu, "hash query forward (CPU)");
    m.def("hash_query_cuda", &hash_query_cuda, "hash query forward (CUDA)");
    m.def("voxelize_forward_cpu", &voxelize_forward_cpu, "Insertion forward (CPU)");
    m.def("voxelize_forward_cuda", &voxelize_forward_cuda, "Insertion forward (CUDA)");
    m.def("voxelize_backward_cpu", &voxelize_backward_cpu, "Insertion backward (CPU)");
    m.def("voxelize_backward_cuda", &voxelize_backward_cuda, "Insertion backward (CUDA)");
    m.def("devoxelize_forward_cpu", &devoxelize_forward_cpu, "Devoxelization forward (CPU)");
    m.def("devoxelize_forward_cuda", &devoxelize_forward_cuda, "Devoxelization forward (CUDA)");
    m.def("devoxelize_backward_cpu", &devoxelize_backward_cpu, "Devoxelization backward (CPU)");
    m.def("devoxelize_backward_cuda", &devoxelize_backward_cuda, "Devoxelization backward (CUDA)");
    m.def("count_forward", &count_forward, "Counting forward (CUDA)");
    m.def("cpu_count_forward", &cpu_count_forward, "count forward (CPU)");
}
