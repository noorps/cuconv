// Stage 2: shared memory tiling.
//
// A block computes a 16x16 tile of one output channel for one image. For each
// input channel it loads the matching input tile plus its halo into shared
// memory once, loads the KxK filter, and every thread then reads its window
// from shared memory instead of global. The tile is reloaded for every output
// channel, which is the cost stage 3 removes.
#include "common.cuh"

static constexpr int TILE = 16;
static constexpr int SMEM_THREADS = TILE * TILE;

template <int K, int S>
__global__ void __launch_bounds__(SMEM_THREADS)
conv_smem(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
          int Ci, int H, int W, int Co, int P, int OH, int OW) {
    constexpr int IN_T = (TILE - 1) * S + K;   // input tile edge including halo
    __shared__ float xs[IN_T * IN_T];
    __shared__ float ws[K * K];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TILE + tx;
    const int co = blockIdx.z % Co;
    const int n = blockIdx.z / Co;
    const int oh = blockIdx.y * TILE + ty;
    const int ow = blockIdx.x * TILE + tx;
    const int ih0 = blockIdx.y * TILE * S - P;   // top-left input coordinate of the tile
    const int iw0 = blockIdx.x * TILE * S - P;

    const float* xn = x + (size_t)n * Ci * H * W;
    const float* wc = w + (size_t)co * Ci * K * K;

    float acc = 0.f;
    for (int ci = 0; ci < Ci; ++ci) {
        const float* xc = xn + (size_t)ci * H * W;
        for (int i = tid; i < IN_T * IN_T; i += SMEM_THREADS) {
            const int r = i / IN_T;
            const int c = i - r * IN_T;
            const int ih = ih0 + r, iw = iw0 + c;
            xs[i] = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? xc[ih * W + iw] : 0.f;
        }
        if (tid < K * K) ws[tid] = wc[ci * K * K + tid];
        __syncthreads();
#pragma unroll
        for (int kh = 0; kh < K; ++kh)
#pragma unroll
            for (int kw = 0; kw < K; ++kw)
                acc += xs[(ty * S + kh) * IN_T + tx * S + kw] * ws[kh * K + kw];
        __syncthreads();
    }
    if (oh < OH && ow < OW)
        y[(((size_t)n * Co + co) * OH + oh) * OW + ow] = acc;
}

template <int K, int S>
static void launch_smem_t(const float* x, const float* w, float* y, ConvShape s, cudaStream_t st) {
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(s.OW, TILE), ceil_div(s.OH, TILE), s.N * s.Co);
    conv_smem<K, S><<<grid, block, 0, st>>>(x, w, y, s.Ci, s.H, s.W, s.Co, s.P, s.OH, s.OW);
}

void launch_smem(const float* x, const float* w, float* y, ConvShape s, cudaStream_t st) {
#define X(KK, SS) if (s.K == KK && s.S == SS) { launch_smem_t<KK, SS>(x, w, y, s, st); return; }
    FOR_EACH_INSTANCE(X)
#undef X
}

KernelInfo info_smem(int K, int S) {
#define X(KK, SS) if (K == KK && S == SS) return query_info(conv_smem<KK, SS>, SMEM_THREADS);
    FOR_EACH_INSTANCE(X)
#undef X
    return KernelInfo{};
}
