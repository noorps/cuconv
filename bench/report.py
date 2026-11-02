"""Render results/<label>/*.json into results/<label>/summary.md.

Standalone (no torch) so it also runs on a machine without a GPU, against a
results folder copied from the Colab run.
"""
import argparse
import json
import os

STAGE_NAMES = {"cudnn": "cuDNN", "naive": "1 naive", "smem": "2 smem tile", "regtile": "3 register tile",
               "vec": "4 float4 + prefetch"}


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def short_kernel(n):
    return n.split("<")[0].split("(")[0].replace("void ", "")


def fmt_clock(r):
    c = r.get("clock_during") or {}
    if c.get("available"):
        return f"{c['sm_mhz_median']:.0f} [{c['sm_mhz_min']}-{c['sm_mhz_max']}]", f"{c['watts_median']:.0f}"
    return str((r.get("clocks") or {}).get("sm_mhz", "?")), "?"


def bench_tables(b):
    out = []
    dev = b["device"]
    smi = dev.get("nvidia_smi", {})
    out.append("## device\n")
    out.append("| | |\n|---|---|")
    out.append(f"| GPU | {dev['gpu']}, {dev['sm_count']} SMs, sm_{dev['compute_capability'].replace('.', '')}, {dev['total_memory_gb']} GiB |")
    out.append(f"| driver / CUDA runtime / nvcc | {smi.get('driver_version')} / {dev['cuda_runtime']} / {dev.get('nvcc')} |")
    out.append(f"| PyTorch / cuDNN | {dev['torch']} / {dev['cudnn']} |")
    out.append(f"| max SM / memory clock | {smi.get('clocks.max.sm')} / {smi.get('clocks.max.mem')} MHz |")
    out.append(f"| power limit | {smi.get('power.limit')} W |")
    out.append(f"| peak fp32 at max clock | {dev.get('peak_fp32_gflops')} GFLOP/s |")
    out.append(f"| peak DRAM bandwidth | {dev.get('peak_dram_gbps')} GB/s ({dev.get('memory_bus_width_bits')}-bit) |")
    out.append(f"| recorded | {dev.get('timestamp')} |")
    m = b["method"]
    out.append(f"\nTiming: CUDA events, {m['warmup']} warm-up launches discarded, {m['repeats']} repeats of "
               f"{m['iters']} launches each; the table shows the median per-launch time across repeats and the "
               f"min-max range of GFLOP/s across repeats in brackets. Clock is the median SM clock sampled every "
               f"5 ms during the timed loop, with its min-max; W is median board power.\n")
    shapes = []
    for r in b["rows"]:
        if r["shape"] not in shapes:
            shapes.append(r["shape"])
    out.append("## throughput\n")
    out.append("| shape | stage | ms | GFLOP/s [min-max] | % cuDNN | % peak at clock | effective GB/s | SM MHz | W |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for sh in shapes:
        for r in [x for x in b["rows"] if x["shape"] == sh]:
            clk, w = fmt_clock(r)
            pk = f"{r['pct_peak_fp32_at_clock']:.0f}%" if "pct_peak_fp32_at_clock" in r else "-"
            out.append(f"| {sh if r['stage'] == 'cudnn' else ''} | {STAGE_NAMES.get(r['stage'], r['stage'])} | "
                       f"{r['ms_median']:.3f} | {r['gflops']:,.0f} [{r['gflops_min']:,.0f}-{r['gflops_max']:,.0f}] | "
                       f"{r['pct_cudnn']:.0f}% | {pk} | {r['effective_gbps']:.0f} | {clk} | {w} |")
    out.append("\n'% peak at clock' is GFLOP/s over 40 SMs x 64 lanes x 2 x the median SM clock sampled during that "
               "configuration. 'effective GB/s' is compulsory traffic (x + w + y once) over time, a lower bound on "
               "what moved, not a measurement.\n")
    return "\n".join(out)


def correctness_table(c):
    out = ["## correctness against cuDNN\n", f"Tolerance {c['atol']:g} absolute. Worst case over every shape, "
           "including resnet18 conv1 (7x7 stride 2) and two odd-sized shapes with partial tiles on every edge.\n",
           "| stage | worst max abs err | worst max rel err | pass |", "|---|---|---|---|"]
    for st, v in c["summary"].items():
        out.append(f"| {STAGE_NAMES.get(st, st)} | {v['max_abs_err']:.3e} | {v['max_rel_err']:.3e} | {'yes' if v['all_ok'] else 'NO'} |")
    out.append("\nPer shape:\n")
    out.append("| stage | shape | max abs err | max rel err | mean abs err |")
    out.append("|---|---|---|---|---|")
    for r in c["rows"]:
        out.append(f"| {r['stage']} | {r['shape']} | {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} | {r['mean_abs_err']:.3e} |")
    return "\n".join(out)


def profile_tables(p, b):
    out = ["## profiler\n"]
    out.append("cuDNN's kernels per shape (torch.profiler):\n")
    out.append("| shape | kernels |\n|---|---|")
    for sh, names in p["cudnn_kernels"].items():
        short = []
        for n in names:
            n = short_kernel(n)
            if not short or short[-1] != n:
                short.append(n)
        out.append(f"| {sh} | {', '.join(short)} |")
    out.append("\nOccupancy API (cudaOccupancyMaxActiveBlocksPerMultiprocessor), stride-1 instances:\n")
    out.append("| stage | K | regs/thread | static smem B | threads | blocks/SM | theoretical occupancy |")
    out.append("|---|---|---|---|---|---|---|")
    for k, v in p["occupancy_api"].items():
        if "S1" in k:
            st, kk, _ = k.split()
            out.append(f"| {STAGE_NAMES.get(st, st)} | {kk[1:]} | {v['regs_per_thread']} | {v['static_smem_bytes']} | "
                       f"{v['threads_per_block']} | {v['max_blocks_per_sm']} | {v['theoretical_occupancy']:.2f} |")
    nc = p.get("ncu", {})
    if not nc.get("ok"):
        out.append(f"\nNsight Compute did not run: `{nc.get('error', '')[:400]}`\n")
        return "\n".join(out)
    out.append(f"\nNsight Compute {nc.get('version')}, hardware counters, one launch per stage per shape. "
               "Achieved occupancy = sm__warps_active % of peak; FMA pipe = sm__pipe_fma_cycles_active %; "
               "DRAM/L1 = % of peak throughput; bank conflicts are shared-memory conflicts per shared wavefront; "
               "FFMA share = FFMA warp instructions the shape needs over warp instructions executed; stalls are "
               "the top three smsp__average_warps_issue_stalled_* ratios (warps stalled per issue-active cycle).\n")
    out.append("| shape | stage | us | achieved occ | FMA pipe | SM | DRAM | L1 | bank conf/wavefront | FFMA share | top stalls |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    flops = {r["shape"]: r["flops"] for r in b["rows"]}
    for k in nc["kernels"]:
        if k["stage"] == "cudnn" or ":" in k["stage"]:
            continue
        m = k["metrics"]
        g = lambda key: m.get(key, 0)  # noqa: E731
        inst = g("smsp__inst_executed.sum") or 1
        wf = g("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum") or 1
        stalls = sorted(((g(s), s.split("stalled_")[1].split("_per")[0]) for s in m if "stalled" in s), reverse=True)[:3]
        out.append(f"| {k['shape']} | {STAGE_NAMES.get(k['stage'], k['stage'])} | {g('gpu__time_duration.sum') / 1e3:.0f} | "
                   f"{g('sm__warps_active.avg.pct_of_peak_sustained_active'):.0f}% | "
                   f"{g('sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active'):.0f}% | "
                   f"{g('sm__throughput.avg.pct_of_peak_sustained_elapsed'):.0f}% | "
                   f"{g('dram__throughput.avg.pct_of_peak_sustained_elapsed'):.0f}% | "
                   f"{g('l1tex__throughput.avg.pct_of_peak_sustained_active'):.0f}% | "
                   f"{g('l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum') / wf:.2f} | "
                   f"{flops[k['shape']] / 2 / 32 / inst:.2f} | "
                   + ", ".join(f"{n} {v:.1f}" for v, n in stalls) + " |")
    out.append("\ncuDNN's own kernels under the same counters:\n")
    out.append("| shape | kernel | us | achieved occ | FMA pipe | SM | DRAM |\n|---|---|---|---|---|---|---|")
    for k in nc["kernels"]:
        if k["stage"] != "cudnn":
            continue
        m = k["metrics"]
        g = lambda key: m.get(key, 0)  # noqa: E731
        out.append(f"| {k['shape']} | {short_kernel(k['kernel'])} | "
                   f"{g('gpu__time_duration.sum') / 1e3:.0f} | {g('sm__warps_active.avg.pct_of_peak_sustained_active'):.0f}% | "
                   f"{g('sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active'):.0f}% | "
                   f"{g('sm__throughput.avg.pct_of_peak_sustained_elapsed'):.0f}% | "
                   f"{g('dram__throughput.avg.pct_of_peak_sustained_elapsed'):.0f}% |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="latest")
    ap.add_argument("--dir", default=None, help="results folder (default results/<label>)")
    args = ap.parse_args()
    d = args.dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", args.label)
    b, c, p, f = load(d, "bench.json"), load(d, "correctness.json"), load(d, "profile.json"), load(d, "featuremaps.json")
    parts = [f"# results: {os.path.basename(d)}\n"]
    if b:
        parts.append(bench_tables(b))
    if c:
        parts.append(correctness_table(c))
    if p and b:
        parts.append(profile_tables(p, b))
    if f:
        parts.append("## feature maps\n")
        parts.append(f"resnet18 conv1 over the photograph, stage `{f['stage']}` against cuDNN: max abs err "
                     f"{f['max_abs_err']:.3e} over {f['n_values']:,} values; against an fp64 CPU reference, mine "
                     f"{f.get('max_abs_err_vs_fp64', float('nan')):.3e} and cuDNN {f.get('cudnn_max_abs_err_vs_fp64', float('nan')):.3e}. "
                     f"Image: {f['image']}.\n")
    out = os.path.join(d, "summary.md")
    with open(out, "w") as fh:
        fh.write("\n\n".join(parts) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
