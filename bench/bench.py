"""Benchmark: GFLOP/s per stage per shape, against cuDNN in the same session.

Timing is CUDA events around a batch of launches. Each configuration is warmed
up (those launches are discarded), then timed REPEATS times; every repeat is
ITERS back-to-back launches and reports the mean per-launch time. The table
shows the median across repeats with the min-max range, so one lucky or
unlucky repeat cannot hide. cuDNN is reached through torch.nn.functional.conv2d
with cudnn.benchmark=True, so it is cuDNN's own fastest algorithm for each
shape, not a fixed one.
"""
import argparse
import statistics

import torch

import common

WARMUP = 10
ITERS = 20
REPEATS = 7


def time_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    per_launch_ms = []
    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            fn()
        end.record()
        torch.cuda.synchronize()
        per_launch_ms.append(start.elapsed_time(end) / ITERS)
    return per_launch_ms


def summarize(ms_list, s):
    med = statistics.median(ms_list)
    fl = common.flops(s)
    mb = common.min_bytes(s)
    return dict(ms_median=med, ms_min=min(ms_list), ms_max=max(ms_list), ms_all=ms_list,
                gflops=fl / med / 1e6, gflops_min=fl / max(ms_list) / 1e6,
                gflops_max=fl / min(ms_list) / 1e6,
                effective_gbps=mb / med / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="latest")
    ap.add_argument("--stages", default=None)
    args = ap.parse_args()

    common.setup_torch()
    ext = common.build()
    stages = args.stages.split(",") if args.stages else list(ext.stages())
    device = common.device_record()
    peak_gf = device.get("peak_fp32_gflops")
    peak_bw = device.get("peak_dram_gbps")

    rows = []
    clocks = {"before": common.clocks_now()}
    for s in common.shapes():
        x, w = common.make_inputs(s)
        label = common.shape_label(s)
        cud = summarize(time_fn(lambda: common.cudnn_conv(x, w, s)), s)
        cud.update(stage="cudnn", shape=label, **{k: s[k] for k in ("N", "Ci", "Co", "H", "W", "K", "S")},
                   flops=common.flops(s), min_bytes=common.min_bytes(s), pct_cudnn=100.0,
                   pct_peak_fp32=(100 * cud["gflops"] / peak_gf) if peak_gf else None,
                   pct_peak_dram=(100 * cud["effective_gbps"] / peak_bw) if peak_bw else None,
                   clocks=common.clocks_now())
        rows.append(cud)
        print(f"{label:<26}{'cudnn':<10}{cud['ms_median']:>9.3f} ms  {cud['gflops']:>8.1f} GFLOP/s"
              f"  [{cud['gflops_min']:.1f}-{cud['gflops_max']:.1f}]")
        for st in stages:
            r = summarize(time_fn(lambda: ext.conv2d(x, w, s["S"], s["P"], st)), s)
            info = ext.kernel_info(st, s["K"], s["S"])
            r.update(stage=st, shape=label, **{k: s[k] for k in ("N", "Ci", "Co", "H", "W", "K", "S")},
                     flops=common.flops(s), min_bytes=common.min_bytes(s),
                     pct_cudnn=100 * r["gflops"] / cud["gflops"],
                     pct_peak_fp32=(100 * r["gflops"] / peak_gf) if peak_gf else None,
                     pct_peak_dram=(100 * r["effective_gbps"] / peak_bw) if peak_bw else None,
                     kernel=dict(info), clocks=common.clocks_now())
            rows.append(r)
            print(f"{'':<26}{st:<10}{r['ms_median']:>9.3f} ms  {r['gflops']:>8.1f} GFLOP/s"
                  f"  [{r['gflops_min']:.1f}-{r['gflops_max']:.1f}]  {r['pct_cudnn']:5.1f}% of cuDNN"
                  f"  occ {info['theoretical_occupancy']:.2f}  regs {info['regs_per_thread']}")
        del x, w
        torch.cuda.empty_cache()
    clocks["after"] = common.clocks_now()

    out = dict(device=device, method=dict(warmup=WARMUP, iters=ITERS, repeats=REPEATS,
               timer="cuda events", stat="median across repeats, min-max in brackets"),
               clocks=clocks, rows=rows)
    common.dump(out, f"{common.result_dir(args.label)}/bench.json")


if __name__ == "__main__":
    main()
