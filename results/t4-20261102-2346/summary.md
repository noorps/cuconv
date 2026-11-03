# results: t4-20261102-2346


## device

| | |
|---|---|
| GPU | Tesla T4, 40 SMs, sm_75, 14.56 GiB |
| driver / CUDA runtime / nvcc | 580.82.07 / 13.0 / Cuda compilation tools, release 13.0, V13.0.88 |
| PyTorch / cuDNN | 2.11.0+cu130 / 92700 |
| max SM / memory clock | 1590 / 5001 MHz |
| power limit | 70.00 W |
| peak fp32 at max clock | 8140.8 GFLOP/s |
| peak DRAM bandwidth | 320.1 GB/s (256-bit) |
| recorded | 2026-11-02 23:46:13 UTC |

Timing: CUDA events, 10 warm-up launches discarded, 7 repeats of 20 launches each; the table shows the median per-launch time across repeats and the min-max range of GFLOP/s across repeats in brackets. Clock is the median SM clock sampled every 5 ms during the timed loop, with its min-max; W is median board power.

## throughput

| shape | stage | ms | GFLOP/s [min-max] | % cuDNN | % peak at clock | effective GB/s | SM MHz | W |
|---|---|---|---|---|---|---|---|---|
| 3->64 @224 K3 N1 | cuDNN | 0.140 | 1,242 [1,020-1,257] | 100% | 19% | 96 | 1305 [1305-1305] | 51 |
|  | 1 naive | 0.248 | 698 [587-702] | 56% | 10% | 54 | 1365 [1215-1455] | 51 |
|  | 2 smem tile | 0.390 | 445 [397-453] | 36% | 6% | 35 | 1380 [1245-1410] | 89 |
|  | 3 register tile | 0.109 | 1,592 [1,580-1,599] | 128% | 23% | 123 | 1350 [1350-1350] | 65 |
|  | 4 float4 + prefetch | 0.112 | 1,544 [1,530-1,552] | 124% | 20% | 120 | 1545 [1530-1560] | 65 |
| 64->128 @56 K3 N1 | cuDNN | 0.178 | 2,592 [2,588-2,611] | 100% | 32% | 15 | 1575 [1575-1575] | 56 |
|  | 1 naive | 0.678 | 682 [637-774] | 26% | 10% | 4 | 1290 [1260-1530] | 81 |
|  | 2 smem tile | 1.195 | 387 [387-413] | 15% | 5% | 2 | 1410 [1410-1515] | 70 |
|  | 3 register tile | 0.148 | 3,128 [3,114-3,202] | 121% | 40% | 18 | 1515 [1500-1530] | 68 |
|  | 4 float4 + prefetch | 0.127 | 3,654 [3,609-3,680] | 141% | 47% | 21 | 1522 [1515-1530] | 72 |
| 128->256 @28 K3 N1 | cuDNN | 0.173 | 2,676 [2,639-2,688] | 100% | 34% | 14 | 1545 [1305-1560] | 72 |
|  | 1 naive | 0.687 | 673 [644-717] | 25% | 9% | 3 | 1395 [1335-1485] | 59 |
|  | 2 smem tile | 1.178 | 393 [391-409] | 15% | 5% | 2 | 1440 [1440-1515] | 73 |
|  | 3 register tile | 0.185 | 2,497 [2,441-2,499] | 93% | 32% | 13 | 1545 [1515-1545] | 69 |
|  | 4 float4 + prefetch | 0.179 | 2,583 [2,437-2,594] | 96% | 34% | 13 | 1500 [1410-1500] | 46 |
| 3->64 @224 K3 N32 | cuDNN | 6.204 | 894 [894-902] | 100% | 28% | 69 | 615 [600-765] | 68 |
|  | 1 naive | 10.414 | 533 [533-533] | 60% | 10% | 41 | 1095 [1080-1095] | 68 |
|  | 2 smem tile | 13.539 | 410 [408-413] | 46% | 6% | 32 | 1260 [1245-1275] | 69 |
|  | 3 register tile | 3.763 | 1,475 [1,474-1,695] | 165% | 40% | 114 | 720 [720-1185] | 68 |
|  | 4 float4 + prefetch | 3.837 | 1,446 [1,416-1,645] | 162% | 47% | 112 | 600 [600-1275] | 68 |
| 64->128 @56 K3 N32 | cuDNN | 2.371 | 6,241 [6,207-7,152] | 100% | 159% | 33 | 765 [750-1125] | 68 |
|  | 1 naive | 22.763 | 650 [646-654] | 10% | 10% | 3 | 1245 [1230-1305] | 68 |
|  | 2 smem tile | 37.894 | 390 [388-392] | 6% | 6% | 2 | 1380 [1365-1395] | 69 |
|  | 3 register tile | 5.451 | 2,715 [2,657-2,735] | 43% | 50% | 14 | 1065 [1005-1110] | 67 |
|  | 4 float4 + prefetch | 4.854 | 3,049 [2,983-3,085] | 49% | 58% | 16 | 1020 [975-1110] | 67 |
| 128->256 @28 K3 N32 | cuDNN | 1.767 | 8,376 [8,290-9,511] | 100% | 206% | 22 | 795 [780-1095] | 68 |
|  | 1 naive | 22.887 | 647 [644-649] | 8% | 10% | 2 | 1260 [1245-1275] | 68 |
|  | 2 smem tile | 37.558 | 394 [394-395] | 5% | 6% | 1 | 1365 [1365-1380] | 69 |
|  | 3 register tile | 5.816 | 2,544 [2,486-2,566] | 30% | 46% | 7 | 1080 [1005-1095] | 67 |
|  | 4 float4 + prefetch | 5.228 | 2,831 [2,809-2,852] | 34% | 52% | 8 | 1065 [1035-1065] | 67 |
| 3->64 @224 K5 N1 | cuDNN | 0.194 | 2,482 [1,760-2,489] | 100% | 34% | 69 | 1410 [990-1410] | 67 |
|  | 1 naive | 0.792 | 608 [595-628] | 25% | 10% | 17 | 1185 [1170-1230] | 51 |
|  | 2 smem tile | 0.834 | 577 [564-636] | 23% | 8% | 16 | 1328 [1305-1485] | 74 |
|  | 3 register tile | 0.117 | 4,103 [4,014-4,226] | 165% | 55% | 115 | 1455 [1410-1500] | 84 |
|  | 4 float4 + prefetch | 0.120 | 4,021 [4,006-4,062] | 162% | 56% | 112 | 1410 [1410-1410] | 94 |
| 64->128 @56 K5 N1 | cuDNN | 0.510 | 2,520 [2,499-2,710] | 100% | 38% | 6 | 1305 [1305-1410] | 66 |
|  | 1 naive | 1.951 | 658 [637-738] | 26% | 10% | 2 | 1275 [1230-1500] | 65 |
|  | 2 smem tile | 2.601 | 494 [489-504] | 20% | 7% | 1 | 1425 [1410-1470] | 69 |
|  | 3 register tile | 0.307 | 4,188 [3,437-4,197] | 166% | 55% | 11 | 1500 [1230-1500] | 65 |
|  | 4 float4 + prefetch | 0.315 | 4,074 [3,600-4,233] | 162% | 62% | 10 | 1282 [1125-1320] | 57 |
| 128->256 @28 K5 N1 | cuDNN | 0.609 | 2,108 [2,100-2,207] | 100% | 31% | 7 | 1335 [1335-1410] | 72 |
|  | 1 naive | 2.015 | 638 [628-693] | 30% | 10% | 2 | 1290 [1275-1470] | 66 |
|  | 2 smem tile | 2.627 | 489 [483-505] | 23% | 7% | 2 | 1425 [1410-1515] | 69 |
|  | 3 register tile | 0.494 | 2,602 [2,227-2,860] | 123% | 39% | 9 | 1305 [1170-1515] | 67 |
|  | 4 float4 + prefetch | 0.461 | 2,786 [2,274-3,068] | 132% | 43% | 10 | 1275 [1065-1440] | 72 |
| 3->64 @224 K5 N32 | cuDNN | 11.323 | 1,361 [1,358-1,382] | 100% | 37% | 38 | 720 [705-855] | 68 |
|  | 1 naive | 26.272 | 587 [586-592] | 43% | 10% | 16 | 1140 [1110-1230] | 67 |
|  | 2 smem tile | 26.941 | 572 [570-578] | 42% | 9% | 16 | 1305 [1290-1305] | 68 |
|  | 3 register tile | 6.794 | 2,269 [2,120-2,269] | 167% | 64% | 63 | 690 [525-870] | 67 |
|  | 4 float4 + prefetch | 7.028 | 2,193 [2,131-2,197] | 161% | 71% | 61 | 600 [555-690] | 67 |
| 64->128 @56 K5 N32 | cuDNN | 3.085 | 13,322 [13,221-14,352] | 100% | 361% | 25 | 720 [705-915] | 67 |
|  | 1 naive | 64.496 | 637 [635-640] | 5% | 10% | 1 | 1200 [1185-1290] | 68 |
|  | 2 smem tile | 85.107 | 483 [481-487] | 4% | 7% | 1 | 1365 [1350-1380] | 68 |
|  | 3 register tile | 12.531 | 3,280 [3,190-3,293] | 25% | 67% | 6 | 960 [795-960] | 66 |
|  | 4 float4 + prefetch | 11.486 | 3,579 [3,574-3,582] | 27% | 76% | 7 | 922 [900-930] | 66 |
| 128->256 @28 K5 N32 | cuDNN | 3.580 | 11,481 [11,288-14,024] | 100% | 383% | 12 | 585 [570-1410] | 67 |
|  | 1 naive | 67.805 | 606 [605-612] | 5% | 10% | 1 | 1170 [1140-1185] | 66 |
|  | 2 smem tile | 87.110 | 472 [471-477] | 4% | 7% | 0 | 1335 [1320-1335] | 67 |
|  | 3 register tile | 14.827 | 2,772 [2,733-2,775] | 24% | 59% | 3 | 915 [870-945] | 66 |
|  | 4 float4 + prefetch | 14.088 | 2,918 [2,906-2,941] | 25% | 67% | 3 | 855 [855-915] | 66 |

'% peak at clock' is GFLOP/s over 40 SMs x 64 lanes x 2 x the median SM clock sampled during that configuration. 'effective GB/s' is compulsory traffic (x + w + y once) over time, a lower bound on what moved, not a measurement.


## correctness against cuDNN

Tolerance 0.001 absolute. Worst case over every shape, including resnet18 conv1 (7x7 stride 2) and two odd-sized shapes with partial tiles on every edge.

| stage | worst max abs err | worst max rel err | pass |
|---|---|---|---|
| 1 naive | 5.424e-05 | 9.896e-06 | yes |
| 2 smem tile | 5.424e-05 | 9.896e-06 | yes |
| 3 register tile | 5.424e-05 | 9.896e-06 | yes |
| 4 float4 + prefetch | 5.424e-05 | 9.896e-06 | yes |

Per shape:

| stage | shape | max abs err | max rel err | mean abs err |
|---|---|---|---|---|
| naive | 3->64 @224 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | 3->64 @224 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | 3->64 @224 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | 3->64 @224 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| naive | 64->128 @56 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | 64->128 @56 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | 64->128 @56 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | 64->128 @56 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| naive | 128->256 @28 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | 128->256 @28 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | 128->256 @28 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | 128->256 @28 K3 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| naive | 3->64 @224 K3 N32 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | 3->64 @224 K3 N32 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | 3->64 @224 K3 N32 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | 3->64 @224 K3 N32 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| naive | 64->128 @56 K3 N32 | 4.208e-05 | 7.328e-06 | 1.077e-06 |
| smem | 64->128 @56 K3 N32 | 4.208e-05 | 7.328e-06 | 1.077e-06 |
| regtile | 64->128 @56 K3 N32 | 4.208e-05 | 7.328e-06 | 1.077e-06 |
| vec | 64->128 @56 K3 N32 | 4.208e-05 | 7.328e-06 | 1.077e-06 |
| naive | 128->256 @28 K3 N32 | 5.424e-05 | 9.896e-06 | 1.444e-06 |
| smem | 128->256 @28 K3 N32 | 5.424e-05 | 9.896e-06 | 1.444e-06 |
| regtile | 128->256 @28 K3 N32 | 5.424e-05 | 9.896e-06 | 1.444e-06 |
| vec | 128->256 @28 K3 N32 | 5.424e-05 | 9.896e-06 | 1.444e-06 |
| naive | 3->64 @224 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | 3->64 @224 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | 3->64 @224 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | 3->64 @224 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| naive | 64->128 @56 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | 64->128 @56 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | 64->128 @56 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | 64->128 @56 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| naive | 128->256 @28 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | 128->256 @28 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | 128->256 @28 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | 128->256 @28 K5 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| naive | 3->64 @224 K5 N32 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | 3->64 @224 K5 N32 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | 3->64 @224 K5 N32 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | 3->64 @224 K5 N32 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| naive | 64->128 @56 K5 N32 | 9.537e-06 | 1.908e-06 | 5.350e-07 |
| smem | 64->128 @56 K5 N32 | 9.537e-06 | 1.908e-06 | 5.350e-07 |
| regtile | 64->128 @56 K5 N32 | 9.537e-06 | 1.908e-06 | 5.350e-07 |
| vec | 64->128 @56 K5 N32 | 9.537e-06 | 1.908e-06 | 5.350e-07 |
| naive | 128->256 @28 K5 N32 | 1.216e-05 | 2.304e-06 | 7.056e-07 |
| smem | 128->256 @28 K5 N32 | 1.216e-05 | 2.304e-06 | 7.056e-07 |
| regtile | 128->256 @28 K5 N32 | 1.216e-05 | 2.304e-06 | 7.056e-07 |
| vec | 128->256 @28 K5 N32 | 1.216e-05 | 2.304e-06 | 7.056e-07 |
| naive | resnet18.conv1 K7 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | resnet18.conv1 K7 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | resnet18.conv1 K7 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | resnet18.conv1 K7 N1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| naive | odd 5->7 @37x53 K3 N2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | odd 5->7 @37x53 K3 N2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | odd 5->7 @37x53 K3 N2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | odd 5->7 @37x53 K3 N2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| naive | odd 5->7 @37x53 s2 K5 N2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| smem | odd 5->7 @37x53 s2 K5 N2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| regtile | odd 5->7 @37x53 s2 K5 N2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vec | odd 5->7 @37x53 s2 K5 N2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

## profiler

cuDNN's kernels per shape (torch.profiler):

| shape | kernels |
|---|---|
| 3->64 @224 K3 N1 | implicit_convolve_sgemm |
| 64->128 @56 K3 N1 | implicit_convolve_sgemm |
| 128->256 @28 K3 N1 | implicit_convolve_sgemm |
| 3->64 @224 K3 N32 | implicit_convolve_sgemm |
| 64->128 @56 K3 N32 | cudnn::winograd_nonfused::winogradForwardData4x4, cudnn::winograd_nonfused::winogradForwardFilter4x4, volta_sgemm_128x64_nn, cudnn::winograd_nonfused::winogradForwardOutput4x4 |
| 128->256 @28 K3 N32 | cudnn::winograd_nonfused::winogradForwardData4x4, cudnn::winograd_nonfused::winogradForwardFilter4x4, volta_sgemm_128x64_nn, cudnn::winograd_nonfused::winogradForwardOutput4x4 |
| 3->64 @224 K5 N1 | implicit_convolve_sgemm |
| 64->128 @56 K5 N1 | implicit_convolve_sgemm |
| 128->256 @28 K5 N1 | implicit_convolve_sgemm |
| 3->64 @224 K5 N32 | implicit_convolve_sgemm |
| 64->128 @56 K5 N32 | fft2d_r2c_32x32, volta_gcgemm_64x32_nt, fft2d_c2r_32x32, fft2d_r2c_32x32, volta_gcgemm_64x32_nt, fft2d_c2r_32x32 |
| 128->256 @28 K5 N32 | fft2d_r2c_32x32, volta_gcgemm_64x32_nt, fft2d_c2r_32x32 |

Occupancy API (cudaOccupancyMaxActiveBlocksPerMultiprocessor), stride-1 instances:

| stage | K | regs/thread | static smem B | threads | blocks/SM | theoretical occupancy |
|---|---|---|---|---|---|---|
| 1 naive | 3 | 20 | 0 | 256 | 4 | 1.00 |
| 1 naive | 5 | 32 | 0 | 256 | 4 | 1.00 |
| 1 naive | 7 | 60 | 0 | 256 | 4 | 1.00 |
| 2 smem tile | 3 | 44 | 1332 | 256 | 4 | 1.00 |
| 2 smem tile | 5 | 48 | 1700 | 256 | 4 | 1.00 |
| 2 smem tile | 7 | 52 | 2132 | 256 | 4 | 1.00 |
| 3 register tile | 3 | 76 | 1976 | 256 | 3 | 0.75 |
| 3 register tile | 5 | 78 | 3376 | 256 | 3 | 0.75 |
| 3 register tile | 7 | 64 | 5320 | 256 | 4 | 1.00 |
| 4 float4 + prefetch | 3 | 78 | 8064 | 256 | 3 | 0.75 |
| 4 float4 + prefetch | 5 | 79 | 13312 | 256 | 3 | 0.75 |
| 4 float4 + prefetch | 7 | 79 | 10752 | 256 | 3 | 0.75 |

Nsight Compute Version 2025.3.1.0 (build 36398880) (public-release), hardware counters, one launch per stage per shape. Achieved occupancy = sm__warps_active % of peak; FMA pipe = sm__pipe_fma_cycles_active %; DRAM/L1 = % of peak throughput; bank conflicts are shared-memory conflicts per shared wavefront; FFMA share = FFMA warp instructions the shape needs over warp instructions executed; stalls are the top three smsp__average_warps_issue_stalled_* ratios (warps stalled per issue-active cycle).

| shape | stage | us | achieved occ | FMA pipe | SM | DRAM | L1 | bank conf/wavefront | FFMA share | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|
| 3->64 @224 K3 N1 | 1 naive | 590 | 91% | 23% | 81% | 8% | 90% | 0.00 | 0.09 | long_scoreboard 4.1, lg_throttle 3.5, not_selected 1.9 |
| 3->64 @224 K3 N1 | 2 smem tile | 889 | 97% | 24% | 46% | 6% | 88% | 0.35 | 0.08 | mio_throttle 5.1, barrier 4.9, long_scoreboard 2.5 |
| 3->64 @224 K3 N1 | 3 register tile | 154 | 70% | 48% | 46% | 49% | 71% | 0.03 | 0.45 | barrier 2.3, mio_throttle 1.7, wait 1.5 |
| 3->64 @224 K3 N1 | 4 float4 + prefetch | 137 | 70% | 49% | 47% | 61% | 88% | 0.15 | 0.52 | mio_throttle 3.7, lg_throttle 2.5, wait 1.5 |
| 64->128 @56 K3 N1 | 1 naive | 1541 | 89% | 11% | 80% | 1% | 93% | 0.00 | 0.17 | lg_throttle 10.9, long_scoreboard 6.9, wait 2.2 |
| 64->128 @56 K3 N1 | 2 smem tile | 2809 | 98% | 15% | 47% | 0% | 89% | 0.36 | 0.09 | barrier 8.6, mio_throttle 6.7, long_scoreboard 3.2 |
| 64->128 @56 K3 N1 | 3 register tile | 361 | 60% | 56% | 53% | 4% | 65% | 0.03 | 0.50 | barrier 2.1, wait 1.5, long_scoreboard 1.3 |
| 64->128 @56 K3 N1 | 4 float4 + prefetch | 301 | 54% | 68% | 61% | 5% | 84% | 0.15 | 0.61 | mio_throttle 2.5, not_selected 1.5, barrier 1.3 |
| 128->256 @28 K3 N1 | 1 naive | 1627 | 79% | 10% | 76% | 1% | 86% | 0.00 | 0.17 | lg_throttle 8.9, long_scoreboard 7.2, wait 2.2 |
| 128->256 @28 K3 N1 | 2 smem tile | 2825 | 96% | 15% | 47% | 0% | 88% | 0.36 | 0.09 | barrier 7.8, mio_throttle 6.9, long_scoreboard 3.1 |
| 128->256 @28 K3 N1 | 3 register tile | 470 | 40% | 50% | 46% | 2% | 55% | 0.03 | 0.44 | barrier 1.5, wait 1.5, long_scoreboard 1.0 |
| 128->256 @28 K3 N1 | 4 float4 + prefetch | 388 | 36% | 69% | 54% | 3% | 72% | 0.15 | 0.54 | wait 1.2, not_selected 1.1, mio_throttle 1.0 |
| 3->64 @224 K3 N32 | 1 naive | 18705 | 92% | 23% | 82% | 9% | 91% | 0.00 | 0.09 | long_scoreboard 4.1, lg_throttle 3.5, not_selected 1.9 |
| 3->64 @224 K3 N32 | 2 smem tile | 28225 | 97% | 24% | 47% | 6% | 89% | 0.35 | 0.08 | mio_throttle 5.1, barrier 4.9, long_scoreboard 2.5 |
| 3->64 @224 K3 N32 | 3 register tile | 4538 | 73% | 50% | 50% | 55% | 77% | 0.03 | 0.45 | barrier 2.5, mio_throttle 1.8, wait 1.5 |
| 3->64 @224 K3 N32 | 4 float4 + prefetch | 4009 | 73% | 52% | 52% | 70% | 97% | 0.15 | 0.52 | mio_throttle 3.6, lg_throttle 2.6, wait 1.5 |
| 64->128 @56 K3 N32 | 1 naive | 47859 | 90% | 11% | 82% | 1% | 96% | 0.00 | 0.17 | lg_throttle 10.9, long_scoreboard 6.9, wait 2.2 |
| 64->128 @56 K3 N32 | 2 smem tile | 87709 | 100% | 15% | 49% | 0% | 91% | 0.36 | 0.09 | barrier 8.8, mio_throttle 6.8, long_scoreboard 3.3 |
| 64->128 @56 K3 N32 | 3 register tile | 9542 | 74% | 64% | 64% | 5% | 79% | 0.03 | 0.50 | barrier 2.4, wait 1.5, not_selected 1.4 |
| 64->128 @56 K3 N32 | 4 float4 + prefetch | 8348 | 74% | 71% | 71% | 6% | 97% | 0.15 | 0.61 | mio_throttle 3.5, barrier 2.3, not_selected 1.9 |
| 128->256 @28 K3 N32 | 1 naive | 48034 | 82% | 11% | 83% | 0% | 94% | 0.00 | 0.17 | lg_throttle 9.2, long_scoreboard 7.0, wait 2.2 |
| 128->256 @28 K3 N32 | 2 smem tile | 86326 | 100% | 15% | 49% | 0% | 92% | 0.36 | 0.09 | barrier 8.0, mio_throttle 7.1, long_scoreboard 3.1 |
| 128->256 @28 K3 N32 | 3 register tile | 10488 | 74% | 67% | 66% | 2% | 79% | 0.03 | 0.44 | barrier 2.3, wait 1.5, not_selected 1.4 |
| 128->256 @28 K3 N32 | 4 float4 + prefetch | 9323 | 74% | 74% | 73% | 3% | 97% | 0.15 | 0.54 | mio_throttle 3.2, barrier 2.3, not_selected 2.0 |
| 3->64 @224 K5 N1 | 1 naive | 1582 | 94% | 16% | 82% | 3% | 95% | 0.00 | 0.15 | lg_throttle 12.6, long_scoreboard 3.0, not_selected 2.5 |
| 3->64 @224 K5 N1 | 2 smem tile | 1878 | 98% | 17% | 50% | 3% | 98% | 0.38 | 0.17 | mio_throttle 16.1, barrier 4.9, long_scoreboard 2.2 |
| 3->64 @224 K5 N1 | 3 register tile | 266 | 71% | 67% | 65% | 29% | 77% | 0.01 | 0.62 | not_selected 2.2, math_pipe_throttle 1.9, barrier 1.4 |
| 3->64 @224 K5 N1 | 4 float4 + prefetch | 237 | 70% | 73% | 71% | 37% | 86% | 0.03 | 0.71 | not_selected 2.5, math_pipe_throttle 2.2, mio_throttle 1.4 |
| 64->128 @56 K5 N1 | 1 naive | 4185 | 87% | 11% | 81% | 0% | 93% | 0.00 | 0.22 | lg_throttle 18.8, long_scoreboard 4.5, wait 2.6 |
| 64->128 @56 K5 N1 | 2 smem tile | 6310 | 98% | 13% | 50% | 0% | 97% | 0.38 | 0.17 | mio_throttle 18.9, barrier 8.8, long_scoreboard 2.6 |
| 64->128 @56 K5 N1 | 3 register tile | 755 | 55% | 71% | 67% | 2% | 69% | 0.01 | 0.61 | not_selected 1.8, math_pipe_throttle 1.5, wait 1.2 |
| 64->128 @56 K5 N1 | 4 float4 + prefetch | 628 | 50% | 86% | 80% | 3% | 80% | 0.00 | 0.71 | not_selected 2.2, math_pipe_throttle 2.1, wait 1.1 |
| 128->256 @28 K5 N1 | 1 naive | 4337 | 78% | 10% | 78% | 1% | 90% | 0.00 | 0.22 | lg_throttle 16.3, long_scoreboard 4.5, wait 2.6 |
| 128->256 @28 K5 N1 | 2 smem tile | 6335 | 96% | 12% | 50% | 0% | 96% | 0.38 | 0.17 | mio_throttle 18.4, barrier 8.8, long_scoreboard 2.5 |
| 128->256 @28 K5 N1 | 3 register tile | 1081 | 37% | 62% | 53% | 2% | 54% | 0.01 | 0.54 | wait 1.2, not_selected 1.2, math_pipe_throttle 1.0 |
| 128->256 @28 K5 N1 | 4 float4 + prefetch | 840 | 35% | 85% | 68% | 3% | 67% | 0.00 | 0.63 | not_selected 1.4, math_pipe_throttle 1.3, wait 1.0 |
| 3->64 @224 K5 N32 | 1 naive | 50448 | 94% | 16% | 82% | 3% | 95% | 0.00 | 0.15 | lg_throttle 12.4, long_scoreboard 3.0, not_selected 2.4 |
| 3->64 @224 K5 N32 | 2 smem tile | 59727 | 98% | 17% | 50% | 3% | 99% | 0.38 | 0.17 | mio_throttle 16.1, barrier 4.9, long_scoreboard 2.2 |
| 3->64 @224 K5 N32 | 3 register tile | 7952 | 74% | 69% | 69% | 33% | 82% | 0.01 | 0.62 | not_selected 2.3, math_pipe_throttle 2.0, barrier 1.5 |
| 3->64 @224 K5 N32 | 4 float4 + prefetch | 7199 | 74% | 75% | 75% | 41% | 91% | 0.03 | 0.71 | not_selected 2.7, math_pipe_throttle 2.4, mio_throttle 1.3 |
| 64->128 @56 K5 N32 | 1 naive | 131675 | 85% | 11% | 82% | 0% | 95% | 0.00 | 0.22 | lg_throttle 17.6, long_scoreboard 5.0, wait 2.6 |
| 64->128 @56 K5 N32 | 2 smem tile | 197772 | 100% | 13% | 51% | 0% | 99% | 0.38 | 0.17 | mio_throttle 19.3, barrier 8.9, long_scoreboard 2.7 |
| 64->128 @56 K5 N32 | 3 register tile | 19958 | 74% | 81% | 81% | 3% | 83% | 0.01 | 0.61 | not_selected 2.4, math_pipe_throttle 2.0, barrier 1.4 |
| 64->128 @56 K5 N32 | 4 float4 + prefetch | 17951 | 74% | 90% | 89% | 3% | 89% | 0.00 | 0.71 | not_selected 3.5, math_pipe_throttle 3.3, wait 1.1 |
| 128->256 @28 K5 N32 | 1 naive | 133846 | 79% | 10% | 81% | 1% | 95% | 0.00 | 0.22 | lg_throttle 16.4, long_scoreboard 4.9, wait 2.6 |
| 128->256 @28 K5 N32 | 2 smem tile | 197283 | 100% | 13% | 51% | 0% | 99% | 0.38 | 0.17 | mio_throttle 19.1, barrier 9.2, long_scoreboard 2.5 |
| 128->256 @28 K5 N32 | 3 register tile | 22839 | 74% | 82% | 81% | 2% | 81% | 0.01 | 0.54 | not_selected 2.4, math_pipe_throttle 2.0, barrier 1.3 |
| 128->256 @28 K5 N32 | 4 float4 + prefetch | 20628 | 74% | 90% | 89% | 3% | 87% | 0.00 | 0.63 | not_selected 3.3, math_pipe_throttle 3.1, barrier 1.6 |

cuDNN's own kernels under the same counters:

| shape | kernel | us | achieved occ | FMA pipe | SM | DRAM |
|---|---|---|---|---|---|---|
| 3->64 @224 K3 N1 | implicit_convolve_sgemm | 222 | 69% | 49% | 48% | 50% |
| 64->128 @56 K3 N1 | implicit_convolve_sgemm | 424 | 31% | 59% | 42% | 4% |
| 128->256 @28 K3 N1 | implicit_convolve_sgemm | 448 | 25% | 48% | 47% | 3% |
| 3->64 @224 K3 N32 | implicit_convolve_sgemm | 6513 | 73% | 52% | 52% | 56% |
| 64->128 @56 K3 N32 | winograd_nonfused::winogradForwardData4x4 | 436 | 45% | 21% | 28% | 73% |
| 64->128 @56 K3 N32 | winograd_nonfused::winogradForwardFilter4x4 | 9 | 24% | 14% | 15% | 25% |
| 64->128 @56 K3 N32 | volta_sgemm_128x64_nn | 1569 | 49% | 81% | 80% | 42% |
| 64->128 @56 K3 N32 | winograd_nonfused::winogradForwardOutput4x4 | 705 | 72% | 25% | 28% | 89% |
| 128->256 @28 K3 N32 | winograd_nonfused::winogradForwardData4x4 | 220 | 45% | 24% | 30% | 72% |
| 128->256 @28 K3 N32 | winograd_nonfused::winogradForwardFilter4x4 | 23 | 58% | 19% | 27% | 80% |
| 128->256 @28 K3 N32 | volta_sgemm_128x64_nn | 1498 | 49% | 89% | 88% | 24% |
| 128->256 @28 K3 N32 | winograd_nonfused::winogradForwardOutput4x4 | 412 | 72% | 22% | 25% | 87% |
| 3->64 @224 K5 N1 | implicit_convolve_sgemm | 485 | 92% | 53% | 52% | 23% |
| 64->128 @56 K5 N1 | implicit_convolve_sgemm | 1118 | 31% | 62% | 44% | 2% |
| 128->256 @28 K5 N1 | implicit_convolve_sgemm | 1227 | 25% | 49% | 48% | 3% |
| 3->64 @224 K5 N32 | implicit_convolve_sgemm | 13963 | 74% | 56% | 56% | 26% |
| 64->128 @56 K5 N32 | fft2d_r2c_32x32 | 266 | 48% | 34% | 50% | 50% |
| 64->128 @56 K5 N32 | fft2d_r2c_32x32 | 141 | 49% | 22% | 28% | 50% |
| 64->128 @56 K5 N32 | volta_gcgemm_64x32_nt | 484 | 37% | 65% | 63% | 47% |
| 64->128 @56 K5 N32 | fft2d_c2r_32x32 | 305 | 47% | 18% | 23% | 50% |
| 64->128 @56 K5 N32 | fft2d_r2c_32x32 | 134 | 49% | 22% | 29% | 50% |
| 64->128 @56 K5 N32 | volta_gcgemm_64x32_nt | 484 | 37% | 65% | 63% | 47% |
| 64->128 @56 K5 N32 | fft2d_c2r_32x32 | 327 | 48% | 18% | 23% | 49% |
| 64->128 @56 K5 N32 | fft2d_r2c_32x32 | 126 | 49% | 22% | 28% | 53% |
| 64->128 @56 K5 N32 | volta_gcgemm_64x32_nt | 485 | 37% | 65% | 62% | 47% |
| 64->128 @56 K5 N32 | fft2d_c2r_32x32 | 323 | 47% | 17% | 22% | 51% |
| 64->128 @56 K5 N32 | fft2d_r2c_32x32 | 134 | 49% | 22% | 27% | 51% |
| 64->128 @56 K5 N32 | volta_gcgemm_64x32_nt | 485 | 37% | 65% | 63% | 47% |
| 64->128 @56 K5 N32 | fft2d_c2r_32x32 | 298 | 48% | 18% | 24% | 50% |
| 128->256 @28 K5 N32 | fft2d_r2c_32x32 | 970 | 49% | 36% | 54% | 57% |
| 128->256 @28 K5 N32 | fft2d_r2c_32x32 | 235 | 49% | 24% | 33% | 53% |
| 128->256 @28 K5 N32 | volta_gcgemm_64x32_nt | 1715 | 37% | 70% | 69% | 42% |
| 128->256 @28 K5 N32 | fft2d_c2r_32x32 | 438 | 49% | 25% | 33% | 56% |

## feature maps


resnet18 conv1 over the photograph, stage `vec` against cuDNN: max abs err 0.000e+00 over 802,816 values; against an fp64 CPU reference, mine 5.804e-06 and cuDNN 5.804e-06. Image: Chelsea, scikit-image sample data, CC0 (Stefan van der Walt).

