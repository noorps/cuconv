// Stage 4: what the profiler asked for.
//
// Nsight Compute on stage 3 said: FMA pipe 82% busy, DRAM under 6%, bank
// conflicts negligible, but only 44-62% of executed instructions were FFMAs.
// The rest were shared-memory loads (one LDS per weight, one per input value)
// and the per-channel fill-and-barrier. So this stage cuts instructions, not
// bytes:
//
//   - weights are transposed once to [Ci][K*K][Co] so a thread's four output
//     channels sit together and come in as one float4 load (LDS.128);
//   - the input strip a thread needs per filter row is read as float4s from
//     a row stride that is a multiple of four;
//   - CI_T input channels are staged per barrier instead of one, so the
//     fill loop and the two __syncthreads() are amortised over four channels.
//
// Same block tile as stage 3 (16 channels x 8 rows x 32 columns, 256
// threads, 4x4 micro-tile per thread). The transposed-weight buffer is a
// workspace the caller provides; its transform kernel is launched here and
// is inside the timed region.
#include "common.cuh"

static constexpr int CO_T = 16;
static constexpr int OH_T = 8;
static constexpr int OW_T = 32;
static constexpr int TM = 4;
static constexpr int TN = 4;
static constexpr int VEC_THREADS = (CO_T / TM) * OH_T * (OW_T / TN);   // 256

// w[Co][Ci][K][K] -> wt[Ci][K*K][Co]
__global__ void transpose_weights(const float* __restrict__ w, float* __restrict__ wt,
                                  int Co, int Ci, int KK) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Co * Ci * KK) return;
    const int kk = i % KK;
    const int ci = (i / KK) % Ci;
    const int co = i / (KK * Ci);
    wt[((size_t)ci * KK + kk) * Co + co] = w[i];
}

template <int K, int S>
__global__ void __launch_bounds__(VEC_THREADS, 3)
conv_vec(const float* __restrict__ x, const float* __restrict__ wt, float* __restrict__ y,
         int Ci, int H, int W, int Co, int P, int OH, int OW) {
    constexpr int CI_T = (S == 1) ? 4 : 1;         // channels staged per barrier
    constexpr int IN_H = (OH_T - 1) * S + K;
    constexpr int IN_W = (OW_T - 1) * S + K;
    constexpr int STRIP = (TN - 1) * S + K;        // input floats a thread needs per row
    constexpr int STRIP4 = (STRIP + 3) / 4;        // float4 loads that cover the strip
    constexpr int NEED = (OW_T / TN - 1) * TN * S + 4 * STRIP4;   // widest column a strip touches
    constexpr int IN_WP = ((IN_W > NEED ? IN_W : NEED) + 3) & ~3; // row stride, multiple of 4
    constexpr int XS_CH = IN_H * IN_WP;
    constexpr int WS_CH = K * K * CO_T;            // per channel: [kh*K+kw][co]
    __shared__ __align__(16) float xs[CI_T * XS_CH];
    __shared__ __align__(16) float ws[CI_T * WS_CH];

    const int tid = threadIdx.x;
    const int cg = tid / (OH_T * (OW_T / TN));
    const int r = (tid / (OW_T / TN)) % OH_T;
    const int cq = tid % (OW_T / TN);

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

    for (int ci0 = 0; ci0 < Ci; ci0 += CI_T) {
#pragma unroll
        for (int cl = 0; cl < CI_T; ++cl) {
            const int ci = ci0 + cl;
            const float* xc = xn + (size_t)ci * H * W;
            for (int i = tid; i < IN_H * IN_W; i += VEC_THREADS) {
                const int rr = i / IN_W;
                const int cc = i - rr * IN_W;
                const int ih = ih0 + rr, iw = iw0 + cc;
                const bool ok = ci < Ci && ih >= 0 && ih < H && iw >= 0 && iw < W;
                xs[cl * XS_CH + rr * IN_WP + cc] = ok ? xc[ih * W + iw] : 0.f;
            }
            for (int i = tid; i < WS_CH; i += VEC_THREADS) {
                const int kk = i / CO_T;
                const int co = co0 + (i - kk * CO_T);
                ws[cl * WS_CH + i] = (ci < Ci && co < Co) ? wt[((size_t)ci * K * K + kk) * Co + co] : 0.f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int cl = 0; cl < CI_T; ++cl) {
            const float* xsc = xs + cl * XS_CH;
            const float* wsc = ws + cl * WS_CH + cg * TM;
#pragma unroll
            for (int kh = 0; kh < K; ++kh) {
                float xr[4 * STRIP4];
                const float4* row = reinterpret_cast<const float4*>(xsc + (r * S + kh) * IN_WP + cq * TN * S);
#pragma unroll
                for (int t = 0; t < STRIP4; ++t) {
                    const float4 v = row[t];
                    xr[4 * t] = v.x; xr[4 * t + 1] = v.y; xr[4 * t + 2] = v.z; xr[4 * t + 3] = v.w;
                }
#pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    const float4 wv = *reinterpret_cast<const float4*>(wsc + (kh * K + kw) * CO_T);
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const float xv = xr[j * S + kw];
                        acc[0][j] += xv * wv.x;
                        acc[1][j] += xv * wv.y;
                        acc[2][j] += xv * wv.z;
                        acc[3][j] += xv * wv.w;
                    }
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
static void launch_vec_t(const float* x, const float* w, float* y, float* wt, ConvShape s, cudaStream_t st) {
    const int total = s.Co * s.Ci * s.K * s.K;
    transpose_weights<<<ceil_div(total, 256), 256, 0, st>>>(w, wt, s.Co, s.Ci, s.K * s.K);
    dim3 block(VEC_THREADS);
    dim3 grid(ceil_div(s.OW, OW_T), ceil_div(s.OH, OH_T), s.N * ceil_div(s.Co, CO_T));
    conv_vec<K, S><<<grid, block, 0, st>>>(x, wt, y, s.Ci, s.H, s.W, s.Co, s.P, s.OH, s.OW);
}

void launch_vec(const float* x, const float* w, float* y, float* wt, ConvShape s, cudaStream_t st) {
#define X(KK, SS) if (s.K == KK && s.S == SS) { launch_vec_t<KK, SS>(x, w, y, wt, s, st); return; }
    FOR_EACH_INSTANCE(X)
#undef X
}

KernelInfo info_vec(int K, int S) {
#define X(KK, SS) if (K == KK && S == SS) return query_info(conv_vec<KK, SS>, VEC_THREADS);
    FOR_EACH_INSTANCE(X)
#undef X
    return KernelInfo{};
}
