// Shared declarations for every kernel stage.
//
// Layout is NCHW, fp32, cross-correlation (what cuDNN and PyTorch call a
// convolution). Every stage exposes the same two entry points:
//
//   launch_<stage>(...)  runs the kernel on a stream
//   info_<stage>(K, S)   reports registers, shared memory and the
//                        occupancy the driver predicts for that instance
//
// K is the filter width, S the stride, P the padding. Each stage is
// templated on K and S so the inner loops unroll fully; the host side
// dispatches the instances this repo benchmarks (K in {3, 5, 7}, S in {1, 2}).
#pragma once
#include <cuda_runtime.h>

struct ConvShape {
    int N, Ci, H, W;    // input
    int Co, K, S, P;    // filter, stride, padding
    int OH, OW;         // output
};

struct KernelInfo {
    int regs_per_thread;
    int static_smem_bytes;
    int threads_per_block;
    int max_blocks_per_sm;
    int max_warps_per_sm;      // theoretical occupancy, in warps
    float occupancy;           // theoretical occupancy, 0..1
};

template <typename Kern>
static KernelInfo query_info(Kern kernel, int threads_per_block) {
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, kernel);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads_per_block, 0);
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    KernelInfo info;
    info.regs_per_thread = a.numRegs;
    info.static_smem_bytes = (int)a.sharedSizeBytes;
    info.threads_per_block = threads_per_block;
    info.max_blocks_per_sm = blocks;
    info.max_warps_per_sm = blocks * threads_per_block / 32;
    info.occupancy = (float)info.max_warps_per_sm / (float)(p.maxThreadsPerMultiProcessor / 32);
    return info;
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// One entry point per stage, implemented in kernels/<stage>.cu.
void launch_naive(const float* x, const float* w, float* y, ConvShape s, cudaStream_t st);
void launch_smem(const float* x, const float* w, float* y, ConvShape s, cudaStream_t st);
void launch_regtile(const float* x, const float* w, float* y, ConvShape s, cudaStream_t st);
KernelInfo info_naive(int K, int S);
KernelInfo info_smem(int K, int S);
KernelInfo info_regtile(int K, int S);

// Instantiate launch/info for every (K, S) pair this repo uses. Each stage
// file defines LAUNCH_BODY(K, S) and INFO_BODY(K, S) and expands this.
#define FOR_EACH_INSTANCE(X) \
    X(3, 1) X(5, 1) X(7, 1) \
    X(3, 2) X(5, 2) X(7, 2)
