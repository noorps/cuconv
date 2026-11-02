"""Shared pieces: building the extension, the shape list, GPU facts.

Everything that produces a number imports from here so the shapes, the FLOP
count and the device record are the same across the correctness check, the
benchmark and the profiler.
"""
import json
import os
import subprocess
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS = os.path.join(ROOT, "kernels")
RESULTS = os.path.join(ROOT, "results")

# Shapes that look like real CNN layers: (name, N, Ci, Co, H, W). Each is run
# with a 3x3 and a 5x5 filter, stride 1, "same" padding.
LAYERS = [
    ("3->64 @224", 3, 64, 224, 224),
    ("64->128 @56", 64, 128, 56, 56),
    ("128->256 @28", 128, 256, 28, 28),
]
BATCHES = [1, 32]
KSIZES = [3, 5]


def shapes():
    out = []
    for K in KSIZES:
        for N in BATCHES:
            for name, Ci, Co, H, W in LAYERS:
                out.append(dict(name=name, N=N, Ci=Ci, Co=Co, H=H, W=W, K=K, S=1, P=K // 2))
    return out


def out_hw(s):
    OH = (s["H"] + 2 * s["P"] - s["K"]) // s["S"] + 1
    OW = (s["W"] + 2 * s["P"] - s["K"]) // s["S"] + 1
    return OH, OW


def flops(s):
    """Multiply-adds counted as two operations, the convention cuDNN papers use."""
    OH, OW = out_hw(s)
    return 2.0 * s["N"] * s["Co"] * OH * OW * s["Ci"] * s["K"] * s["K"]


def min_bytes(s):
    """Compulsory traffic: read x and w once, write y once. A lower bound, not
    what the kernel actually moves."""
    OH, OW = out_hw(s)
    x = s["N"] * s["Ci"] * s["H"] * s["W"]
    w = s["Co"] * s["Ci"] * s["K"] * s["K"]
    y = s["N"] * s["Co"] * OH * OW
    return 4.0 * (x + w + y)


def shape_label(s):
    return f"{s['name']} K{s['K']} N{s['N']}"


def build(verbose=False):
    from torch.utils.cpp_extension import load
    sources = [os.path.join(KERNELS, f) for f in sorted(os.listdir(KERNELS))
               if f.endswith(".cu") or f.endswith(".cpp")]
    build_dir = os.path.join(ROOT, "build")
    os.makedirs(build_dir, exist_ok=True)
    return load(name="cuconv", sources=sources, build_directory=build_dir,
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=verbose)


def _smi(fields):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip().split("\n")[0]
        vals = [v.strip() for v in out.split(",")]
        return dict(zip(fields, vals))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def clocks_now():
    """Current SM and memory clock in MHz, from nvidia-smi."""
    d = _smi(["clocks.sm", "clocks.mem"])
    try:
        return {"sm_mhz": int(d["clocks.sm"]), "mem_mhz": int(d["clocks.mem"])}
    except Exception:  # noqa: BLE001
        return {"sm_mhz": None, "mem_mhz": None, "error": d.get("error")}


def device_record():
    """Everything a reader needs to interpret a number from this session."""
    p = torch.cuda.get_device_properties(0)
    smi = _smi(["name", "driver_version", "clocks.max.sm", "clocks.max.mem", "clocks.sm",
                "clocks.mem", "memory.total", "pstate", "power.limit", "temperature.gpu"])
    bus_width = None
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(h)
    except Exception:  # noqa: BLE001
        pass
    try:
        nvcc = subprocess.check_output(["nvcc", "--version"], text=True).strip().split("\n")[-2]
    except Exception:  # noqa: BLE001
        nvcc = None
    rec = {
        "gpu": p.name,
        "sm_count": p.multi_processor_count,
        "compute_capability": f"{p.major}.{p.minor}",
        "total_memory_gb": round(p.total_memory / 2**30, 2),
        "max_threads_per_sm": p.max_threads_per_multi_processor,
        "memory_bus_width_bits": bus_width,
        "nvidia_smi": smi,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvcc": nvcc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    # Peak fp32: 64 FMA lanes per SM on sm_75 (and sm_8x), two flops each,
    # at the maximum SM clock nvidia-smi reports.
    try:
        max_sm_mhz = int(smi["clocks.max.sm"])
        lanes = 64 if p.major >= 7 else 128
        rec["peak_fp32_gflops"] = round(p.multi_processor_count * lanes * 2 * max_sm_mhz / 1e3, 1)
    except Exception:  # noqa: BLE001
        rec["peak_fp32_gflops"] = None
    # Peak DRAM bandwidth from the bus width and max memory clock (DDR, so x2).
    try:
        max_mem_mhz = int(smi["clocks.max.mem"])
        rec["peak_dram_gbps"] = round(bus_width / 8 * max_mem_mhz * 1e6 * 2 / 1e9, 1)
    except Exception:  # noqa: BLE001
        rec["peak_dram_gbps"] = None
    return rec


def setup_torch():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True       # let cuDNN pick its fastest algorithm
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False     # fp32 means fp32 (no TF32 on T4 anyway)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)


def make_inputs(s, device="cuda"):
    x = torch.randn(s["N"], s["Ci"], s["H"], s["W"], device=device, dtype=torch.float32)
    w = torch.randn(s["Co"], s["Ci"], s["K"], s["K"], device=device, dtype=torch.float32)
    w *= (1.0 / (s["Ci"] * s["K"] * s["K"])) ** 0.5      # Kaiming-ish scale so outputs are O(1)
    return x, w


def cudnn_conv(x, w, s):
    return torch.nn.functional.conv2d(x, w, stride=s["S"], padding=s["P"])


def result_dir(label):
    d = os.path.join(RESULTS, label)
    os.makedirs(d, exist_ok=True)
    return d


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {os.path.relpath(path, ROOT)}")
