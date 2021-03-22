#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "convolution/convolution_cpu_header.h"
#include "hash/hash_cpu_header.h"
#include "devoxelize/devox_cpu_header.h"
#include "others/insertion_cpu_header.h"
#include "others/query_cpu_header.h"
#include "others/count_cpu_header.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparseconv_forward_cpu", &sparseconv_forward_cpu, "point cloud convolution forward (CPU)");
    m.def("sparseconv_backward_cpu", &sparseconv_backward_cpu, "point cloud convolution backward (CPU)");
    m.def("hash_build_cpu", &hash_build_cpu, "Hashing forward (CPU)");
    m.def("cpu_kernel_hash_forward", &cpu_kernel_hash_forward, "Kernel Hashing forward (CPU)");
    m.def("voxelize_forward_cpu", &voxelize_forward_cpu, "Insertion forward (CPU)");
    m.def("voxelize_backward_cpu", &voxelize_backward_cpu, "Insertion backward (CPU)");
    m.def("devoxelize_forward_cpu", &devoxelize_forward_cpu, "Devoxelization forward (CPU)");
    m.def("devoxelize_backward_cpu", &devoxelize_backward_cpu, "Devoxelization backward (CPU)");
    m.def("hash_query_cpu", &hash_query_cpu, "hash query forward (CPU)");
    m.def("cpu_count_forward", &cpu_count_forward, "count forward (CPU)");
}
