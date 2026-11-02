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


class ClockSampler:
    """Samples SM clock and power draw every few ms on a thread while a
    configuration is being timed. The T4 is power capped at 70 W and drops its
    clock under sustained fp32 load, so a GFLOP/s number without the clock it
    ran at is not comparable to anything."""

    def __init__(self):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        import threading
        self.clocks, self.watts = [], []
        self.stop = threading.Event()

        def run():
            while not self.stop.is_set():
                try:
                    self.clocks.append(self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM))
                    self.watts.append(self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0)
                except Exception:  # noqa: BLE001
                    pass
                self.stop.wait(0.005)
        self.t = threading.Thread(target=run, daemon=True)
        if self.ok:
            self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        if self.ok:
            self.t.join()

    def summary(self):
        if not self.clocks:
            return dict(available=False)
        return dict(available=True, samples=len(self.clocks),
                    sm_mhz_median=statistics.median(self.clocks), sm_mhz_min=min(self.clocks),
                    sm_mhz_max=max(self.clocks), watts_median=statistics.median(self.watts),
                    watts_max=max(self.watts))


def time_fn(fn, sampler):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    per_launch_ms = []
    with sampler:
        for _ in range(REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(ITERS):
                fn()
            end.record()
            torch.cuda.synchronize()
            per_launch_ms.append(start.elapsed_time(end) / ITERS)
    return per_launch_ms, sampler.summary()


def summarize(ms_list, clock, s, device):
    med = statistics.median(ms_list)
    fl = common.flops(s)
    mb = common.min_bytes(s)
    out = dict(ms_median=med, ms_min=min(ms_list), ms_max=max(ms_list), ms_all=ms_list,
               gflops=fl / med / 1e6, gflops_min=fl / max(ms_list) / 1e6,
               gflops_max=fl / min(ms_list) / 1e6,
               effective_gbps=mb / med / 1e6, clock_during=clock)
    # peak fp32 at the clock the card actually held while this ran
    if clock.get("available"):
        lanes = 64 if int(device["compute_capability"].split(".")[0]) >= 7 else 128
        peak_here = device["sm_count"] * lanes * 2 * clock["sm_mhz_median"] / 1e3
        out["peak_fp32_gflops_at_clock"] = peak_here
        out["pct_peak_fp32_at_clock"] = 100 * out["gflops"] / peak_here
    return out


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
        sampler = ClockSampler()
        cud = summarize(*time_fn(lambda: common.cudnn_conv(x, w, s), sampler), s, device)
        cud.update(stage="cudnn", shape=label, name=s["name"], **{k: s[k] for k in ("N", "Ci", "Co", "H", "W", "K", "S")},
                   flops=common.flops(s), min_bytes=common.min_bytes(s), pct_cudnn=100.0,
                   pct_peak_fp32=(100 * cud["gflops"] / peak_gf) if peak_gf else None,
                   pct_peak_dram=(100 * cud["effective_gbps"] / peak_bw) if peak_bw else None,
                   clocks=common.clocks_now())
        rows.append(cud)
        print(f"{label:<26}{'cudnn':<10}{cud['ms_median']:>9.3f} ms  {cud['gflops']:>8.1f} GFLOP/s"
              f"  [{cud['gflops_min']:.1f}-{cud['gflops_max']:.1f}]"
              f"  clk {cud['clock_during'].get('sm_mhz_median', '?')} MHz {cud['clock_during'].get('watts_median', '?')} W")
        for st in stages:
            r = summarize(*time_fn(lambda: ext.conv2d(x, w, s["S"], s["P"], st), sampler), s, device)
            info = ext.kernel_info(st, s["K"], s["S"])
            r.update(stage=st, shape=label, name=s["name"], **{k: s[k] for k in ("N", "Ci", "Co", "H", "W", "K", "S")},
                     flops=common.flops(s), min_bytes=common.min_bytes(s),
                     pct_cudnn=100 * r["gflops"] / cud["gflops"],
                     pct_peak_fp32=(100 * r["gflops"] / peak_gf) if peak_gf else None,
                     pct_peak_dram=(100 * r["effective_gbps"] / peak_bw) if peak_bw else None,
                     kernel=dict(info), clocks=common.clocks_now())
            rows.append(r)
            print(f"{'':<26}{st:<10}{r['ms_median']:>9.3f} ms  {r['gflops']:>8.1f} GFLOP/s"
                  f"  [{r['gflops_min']:.1f}-{r['gflops_max']:.1f}]  {r['pct_cudnn']:5.1f}% of cuDNN"
                  f"  occ {info['theoretical_occupancy']:.2f}  regs {info['regs_per_thread']}"
                  f"  clk {r['clock_during'].get('sm_mhz_median', '?')} MHz {r['clock_during'].get('watts_median', '?')} W")
        del x, w
        torch.cuda.empty_cache()
    clocks["after"] = common.clocks_now()

    out = dict(device=device, method=dict(warmup=WARMUP, iters=ITERS, repeats=REPEATS,
               timer="cuda events", stat="median across repeats, min-max in brackets"),
               clocks=clocks, rows=rows)
    common.dump(out, f"{common.result_dir(args.label)}/bench.json")


if __name__ == "__main__":
    main()
