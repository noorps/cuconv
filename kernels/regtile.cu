// Stage 3: register tiling with output-channel blocking.
//
// A block computes CO_T output channels over an OH_T x OW_T spatial tile, so
// the input tile loaded into shared memory is reused CO_T times instead of
// once. Each thread owns a TM x TN micro-tile (TM output channels, TN
// consecutive columns) held in registers. Per input channel and filter row it
// pulls one strip of input into registers and reuses every value TN times
// across the columns and TM times across the channels, so most multiply-adds
// read registers rather than shared memory.
//
// Block tile: CO_T=16 channels x OH_T=8 rows x OW_T=32 columns.
// Threads: 256 = 4 channel groups x 8 rows x 8 column groups.
// The shared row stride is padded to an odd number so the eight column groups
// and four rows a warp touches land in distinct banks.
#include "common.cuh"

static constexpr int CO_T = 16;
static constexpr int OH_T = 8;
static constexpr int OW_T = 32;
static constexpr int TM = 4;                       // channels per thread
static constexpr int TN = 4;                       // columns per thread
static constexpr int REG_THREADS = (CO_T / TM) * OH_T * (OW_T / TN);   // 256

template <int K, int S>
__global__ void __launch_bounds__(REG_THREADS)
conv_regtile(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
             int Ci, int H, int W, int Co, int P, int OH, int OW) {
    constexpr int IN_H = (OH_T - 1) * S + K;
    constexpr int IN_W = (OW_T - 1) * S + K;
    constexpr int IN_WP = IN_W | 1;                // odd row stride: conflict-free reads
    constexpr int STRIP = (TN - 1) * S + K;        // input values one thread needs per row
    __shared__ float xs[IN_H * IN_WP];
    __shared__ float ws[CO_T * K * K];

    const int tid = threadIdx.x;
    const int cg = tid / (OH_T * (OW_T / TN));                // channel group 0..3
    const int r = (tid / (OW_T / TN)) % OH_T;                 // row within tile 0..7
    const int cq = tid % (OW_T / TN);                         // column group 0..7

    const int n = blockIdx.z / ceil_div(Co, CO_T);
    const int co0 = (blockIdx.z % ceil_div(Co, CO_T)) * CO_T;
    const int oh0 = blockIdx.y * OH_T;
    const int ow0 = blockIdx.x * OW_T;
    const int ih0 = oh0 * S - P;
    const int iw0 = ow0 * S - P;

    const float* xn = x + (size_t)n * Ci * H * W;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    for (int ci = 0; ci < Ci; ++ci) {
        const float* xc = xn + (size_t)ci * H * W;
        for (int i = tid; i < IN_H * IN_W; i += REG_THREADS) {
            const int rr = i / IN_W;
            const int cc = i - rr * IN_W;
            const int ih = ih0 + rr, iw = iw0 + cc;
            xs[rr * IN_WP + cc] = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? xc[ih * W + iw] : 0.f;
        }
        for (int i = tid; i < CO_T * K * K; i += REG_THREADS) {
            const int cl = i / (K * K);
            const int co = co0 + cl;
            ws[i] = (co < Co) ? w[((size_t)co * Ci + ci) * K * K + (i - cl * K * K)] : 0.f;
        }
        __syncthreads();

#pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            float xr[STRIP];
            const float* row = xs + (r * S + kh) * IN_WP + cq * TN * S;
#pragma unroll
            for (int t = 0; t < STRIP; ++t) xr[t] = row[t];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float* wrow = ws + (cg * TM + i) * K * K + kh * K;
#pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    const float wv = wrow[kw];
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] += xr[j * S + kw] * wv;
                }
            }
        }
        __syncthreads();
    }

    const int oh = oh0 + r;
    if (oh >= OH) return;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int co = co0 + cg * TM + i;
        if (co >= Co) break;
        float* yrow = y + (((size_t)n * Co + co) * OH + oh) * OW;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int ow = ow0 + cq * TN + j;
            if (ow < OW) yrow[ow] = acc[i][j];
        }
    }
}

template <int K, int S>
static void launch_regtile_t(const float* x, const float* w, float* y, ConvShape s, cudaStream_t st) {
    dim3 block(REG_THREADS);
    dim3 grid(ceil_div(s.OW, OW_T), ceil_div(s.OH, OH_T), s.N * ceil_div(s.Co, CO_T));
    conv_regtile<K, S><<<grid, block, 0, st>>>(x, w, y, s.Ci, s.H, s.W, s.Co, s.P, s.OH, s.OW);
}

void launch_regtile(const float* x, const float* w, float* y, ConvShape s, cudaStream_t st) {
#define X(K, S) if (s.K == K && s.S == S) { launch_regtile_t<K, S>(x, w, y, s, st); return; }
    FOR_EACH_INSTANCE(X)
#undef X
}

KernelInfo info_regtile(int K, int S) {
#define X(KK, SS) if (K == KK && S == SS) return query_info(conv_regtile<KK, SS>, REG_THREADS);
    FOR_EACH_INSTANCE(X)
#undef X
    return KernelInfo{};
}
