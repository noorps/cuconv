"""Profiler pass.

Two parts. First, torch.profiler records which kernels cuDNN actually launched
for each shape (cuDNN picks per shape, and the name says whether it is an
implicit GEMM, FFT or Winograd). Second, Nsight Compute, if it runs here:
occupancy, DRAM and shared-memory throughput, bank conflicts, and the warp
stall breakdown for every stage on the batch-1 shapes. If ncu cannot read the
hardware counters the error is recorded verbatim and the benchmark's
occupancy-API numbers are all this repo has.
"""
import argparse
import csv
import io
import os
import shutil
import subprocess
import sys

import torch

import common

# Metrics by base name; the suffix is the roll-up. The list is filtered
# against `ncu --query-metrics` so an unknown name does not sink the run.
METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",         # achieved occupancy
    "sm__maximum_warps_per_active_cycle_pct",                     # theoretical occupancy
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks",
    "launch__waves_per_multiprocessor",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",                                           # L2 traffic
    "l1tex__t_bytes.sum",                                         # L1 traffic
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "smsp__inst_executed.sum",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
]


def find_ncu():
    p = shutil.which("ncu")
    if p:
        return p
    import glob
    hits = (glob.glob("/usr/local/cuda/bin/ncu") + glob.glob("/usr/local/cuda*/bin/ncu")
            + glob.glob("/opt/nvidia/nsight-compute/*/ncu") + glob.glob("/opt/nvidia/nsight-compute/ncu"))
    return hits[0] if hits else None


def available_metrics(ncu):
    try:
        out = subprocess.check_output([ncu, "--query-metrics"], text=True, timeout=120)
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if parts and "__" in parts[0]:
            names.add(parts[0])
    return names, None


def cudnn_kernels(ext, stages):
    """Kernel names torch.profiler sees for cuDNN, per shape."""
    from torch.profiler import ProfilerActivity, profile
    out = {}
    for s in common.shapes():
        x, w = common.make_inputs(s)
        for _ in range(3):
            common.cudnn_conv(x, w, s)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            common.cudnn_conv(x, w, s)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if str(e.device_type).endswith("CUDA")]
        out[common.shape_label(s)] = names
        del x, w
    torch.cuda.empty_cache()
    return out


def run_ncu(ncu, metrics, stages, batches, log_path):
    cmd = [ncu, "--csv", "--target-processes", "all", "--profile-from-start", "off",
           "--metrics", ",".join(metrics),
           sys.executable, os.path.join(common.ROOT, "bench", "ncu_target.py"),
           "--stages", ",".join(stages), "--batches", batches]
    print("running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True, cwd=common.ROOT)
    with open(log_path, "w") as f:
        f.write("# stdout\n" + p.stdout + "\n# stderr\n" + p.stderr)
    return p


def parse_ncu_csv(text):
    # ncu prints banner lines before the CSV header; skip to the header
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return []
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    kernels = {}
    for r in rows:
        key = (int(r["ID"]), r["Kernel Name"])
        k = kernels.setdefault(key, dict(id=int(r["ID"]), kernel=r["Kernel Name"],
                                         grid=r.get("Grid Size"), block=r.get("Block Size"), metrics={}))
        val = r["Metric Value"].replace(",", "")
        try:
            val = float(val)
        except ValueError:
            pass
        k["metrics"][r["Metric Name"]] = val
    return [kernels[k] for k in sorted(kernels)]


def attribute(kernels, stages, batches):
    """ncu lists launches in order. For every shape, ncu_target emits cuDNN's
    launches and then each stage in order; the last stage's main kernel
    closes the group. Inside a group, our kernels are named by stage
    (conv_<stage>, plus stage 4's transpose_weights helper); anything else is
    cuDNN's."""
    shapes = [s for s in common.shapes() if s["N"] in {int(b) for b in batches.split(",")}]
    groups, cur = [], []
    for k in kernels:
        cur.append(k)
        if f"conv_{stages[-1]}" in k["kernel"]:
            groups.append(cur)
            cur = []
    out = []
    for s, group in zip(shapes, groups):
        label = common.shape_label(s)
        for k in group:
            stage = "cudnn"
            for st in stages:
                if f"conv_{st}" in k["kernel"]:
                    stage = st
            if "transpose_weights" in k["kernel"]:
                stage = "vec:transpose"
            out.append(dict(shape=label, stage=stage, **k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="latest")
    ap.add_argument("--stages", default=None)
    ap.add_argument("--batches", default="1,32")
    args = ap.parse_args()
    common.setup_torch()
    ext = common.build()
    stages = args.stages.split(",") if args.stages else list(ext.stages())
    outdir = common.result_dir(args.label)

    result = dict(device=common.device_record(), cudnn_kernels=cudnn_kernels(ext, stages),
                  occupancy_api={})
    for st in stages:
        for K in (3, 5, 7):
            for S in (1, 2):
                result["occupancy_api"][f"{st} K{K} S{S}"] = dict(ext.kernel_info(st, K, S))
    for label, names in result["cudnn_kernels"].items():
        print(f"cuDNN on {label}: {names}")

    ncu = find_ncu()
    if not ncu:
        result["ncu"] = dict(ok=False, error="ncu binary not found")
        print("ncu not found")
    else:
        avail, err = available_metrics(ncu)
        if avail is None:
            metrics = METRICS
        else:
            metrics = [m for m in METRICS if m.split(".")[0] in avail]
            dropped = [m for m in METRICS if m not in metrics]
            if dropped:
                print("metrics not on this ncu/GPU:", dropped)
        p = run_ncu(ncu, metrics, stages, args.batches, os.path.join(outdir, "ncu.log"))
        kernels = parse_ncu_csv(p.stdout)
        if p.returncode != 0 or not kernels:
            msg = (p.stderr + p.stdout)[-3000:]
            result["ncu"] = dict(ok=False, returncode=p.returncode, error=msg, binary=ncu)
            print("ncu failed:\n", msg)
        else:
            try:
                ver = subprocess.check_output([ncu, "--version"], text=True).strip().splitlines()[-1]
            except Exception:  # noqa: BLE001
                ver = None
            result["ncu"] = dict(ok=True, binary=ncu, version=ver, metrics=metrics,
                                 kernels=attribute(kernels, stages, args.batches))
            print(f"ncu profiled {len(kernels)} launches")
            for k in result["ncu"]["kernels"]:
                m = k["metrics"]
                print(f"{k['shape']:<26}{k['stage']:<10}"
                      f" occ {m.get('sm__warps_active.avg.pct_of_peak_sustained_active', '?')}"
                      f" dram% {m.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', '?')}"
                      f" sm% {m.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', '?')}"
                      f" conflicts {m.get('l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum', '?')}")
    common.dump(result, os.path.join(outdir, "profile.json"))


if __name__ == "__main__":
    main()
