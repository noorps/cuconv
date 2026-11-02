// Stage 1: naive direct convolution.
//
// One thread per output element. Every input and weight value is read from
// global memory inside the innermost loop; nothing is cached by hand. This is
// the floor every later stage is measured against.
#include "common.cuh"

template <int K, int S>
__global__ void conv_naive(const float* __restrict__ x, const float* __restrict__ w,
                           float* __restrict__ y, int Ci, int H, int W, int Co, int P,
                           int OH, int OW) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int co = blockIdx.y;
    const int n = blockIdx.z;
    if (idx >= OH * OW) return;
    const int oh = idx / OW;
    const int ow = idx - oh * OW;

    const float* xn = x + (size_t)n * Ci * H * W;
    const float* wc = w + (size_t)co * Ci * K * K;
    const int ih0 = oh * S - P;
    const int iw0 = ow * S - P;

    float acc = 0.f;
    for (int ci = 0; ci < Ci; ++ci) {
        const float* xc = xn + (size_t)ci * H * W;
        const float* wk = wc + ci * K * K;
#pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            const int ih = ih0 + kh;
            if (ih < 0 || ih >= H) continue;
#pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                const int iw = iw0 + kw;
                if (iw < 0 || iw >= W) continue;
                acc += xc[ih * W + iw] * wk[kh * K + kw];
            }
        }
    }
    y[(((size_t)n * Co + co) * OH + oh) * OW + ow] = acc;
}

static constexpr int NAIVE_THREADS = 256;

template <int K, int S>
static void launch_naive_t(const float* x, const float* w, float* y, ConvShape s, cudaStream_t st) {
    dim3 block(NAIVE_THREADS);
    dim3 grid(ceil_div(s.OH * s.OW, NAIVE_THREADS), s.Co, s.N);
    conv_naive<K, S><<<grid, block, 0, st>>>(x, w, y, s.Ci, s.H, s.W, s.Co, s.P, s.OH, s.OW);
}

void launch_naive(const float* x, const float* w, float* y, ConvShape s, cudaStream_t st) {
#define X(KK, SS) if (s.K == KK && s.S == SS) { launch_naive_t<KK, SS>(x, w, y, s, st); return; }
    FOR_EACH_INSTANCE(X)
#undef X
}

KernelInfo info_naive(int K, int S) {
#define X(KK, SS) if (K == KK && S == SS) return query_info(conv_naive<KK, SS>, NAIVE_THREADS);
    FOR_EACH_INSTANCE(X)
#undef X
    return KernelInfo{};
}
