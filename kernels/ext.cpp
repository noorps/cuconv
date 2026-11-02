// PyTorch binding. The kernels are plain CUDA; this file only moves tensors
// in and out so the benchmark can hand the same inputs to cuDNN and to each
// stage inside one process.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "common.cuh"

static ConvShape make_shape(const torch::Tensor& x, const torch::Tensor& w, int stride, int pad) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "tensors must be on the GPU");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "NCHW contiguous only");
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4, "x must be NCHW and w must be [Co, Ci, K, K]");
    TORCH_CHECK(w.size(2) == w.size(3), "square filters only");
    TORCH_CHECK(w.size(1) == x.size(1), "input channels must match");
    ConvShape s;
    s.N = (int)x.size(0); s.Ci = (int)x.size(1); s.H = (int)x.size(2); s.W = (int)x.size(3);
    s.Co = (int)w.size(0); s.K = (int)w.size(2); s.S = stride; s.P = pad;
    s.OH = (s.H + 2 * s.P - s.K) / s.S + 1;
    s.OW = (s.W + 2 * s.P - s.K) / s.S + 1;
    TORCH_CHECK((s.K == 3 || s.K == 5 || s.K == 7) && (s.S == 1 || s.S == 2),
                "built for K in {3,5,7} and stride in {1,2}");
    return s;
}

static torch::Tensor conv2d(const torch::Tensor& x, const torch::Tensor& w, int stride, int pad,
                            const std::string& stage) {
    ConvShape s = make_shape(x, w, stride, pad);
    auto y = torch::empty({s.N, s.Co, s.OH, s.OW}, x.options());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const float* xp = x.data_ptr<float>();
    const float* wp = w.data_ptr<float>();
    float* yp = y.data_ptr<float>();
    if (stage == "naive") launch_naive(xp, wp, yp, s, st);
    else if (stage == "smem") launch_smem(xp, wp, yp, s, st);
    else if (stage == "regtile") launch_regtile(xp, wp, yp, s, st);
    else if (stage == "vec") {
        auto wt = torch::empty_like(w);
        launch_vec(xp, wp, yp, wt.data_ptr<float>(), s, st);
    }
    else TORCH_CHECK(false, "unknown stage: ", stage);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

static py::dict kernel_info(const std::string& stage, int K, int S) {
    KernelInfo i;
    if (stage == "naive") i = info_naive(K, S);
    else if (stage == "smem") i = info_smem(K, S);
    else if (stage == "regtile") i = info_regtile(K, S);
    else if (stage == "vec") i = info_vec(K, S);
    else TORCH_CHECK(false, "unknown stage: ", stage);
    py::dict d;
    d["regs_per_thread"] = i.regs_per_thread;
    d["static_smem_bytes"] = i.static_smem_bytes;
    d["threads_per_block"] = i.threads_per_block;
    d["max_blocks_per_sm"] = i.max_blocks_per_sm;
    d["max_warps_per_sm"] = i.max_warps_per_sm;
    d["theoretical_occupancy"] = i.occupancy;
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &conv2d, "direct conv2d, NCHW fp32", py::arg("x"), py::arg("w"),
          py::arg("stride"), py::arg("pad"), py::arg("stage"));
    m.def("kernel_info", &kernel_info, "registers, shared memory and occupancy for a stage");
    m.def("stages", []() { return std::vector<std::string>{"naive", "smem", "regtile", "vec"}; });
}
