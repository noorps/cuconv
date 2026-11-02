"""The performance chart: GFLOP/s by stage and shape, cuDNN drawn on it.

One panel per (filter size, batch). Bars are the stages in order, one blue
stepping darker per stage; cuDNN for the same shape is the dark dashed line
with a marker above each group. Numbers come straight from bench.json, which
sits next to the image so the site can redraw it.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import common  # noqa: E402

# Ordinal blue ramp, light to dark, one step per stage (validated ordinal palette).
STAGE_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
STAGE_NAMES = {"naive": "1 naive", "smem": "2 shared memory", "regtile": "3 register tile"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CUDNN = "#0b0b0b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="latest")
    args = ap.parse_args()
    outdir = common.result_dir(args.label)
    with open(os.path.join(outdir, "bench.json")) as f:
        data = json.load(f)
    rows = data["rows"]
    dev = data["device"]
    stages = []
    for r in rows:
        if r["stage"] != "cudnn" and r["stage"] not in stages:
            stages.append(r["stage"])
    ks = sorted({r["K"] for r in rows})
    ns = sorted({r["N"] for r in rows})
    layer_names = [l[0] for l in common.LAYERS]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK})
    fig, axes = plt.subplots(len(ks), len(ns), figsize=(13, 8.2), sharey=True, dpi=200,
                             facecolor=SURFACE)
    axes = axes.reshape(len(ks), len(ns))
    ymax = max(r["gflops_max"] for r in rows) * 1.12
    width = 0.8 / len(stages)

    for i, K in enumerate(ks):
        for j, N in enumerate(ns):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            sub = [r for r in rows if r["K"] == K and r["N"] == N]
            xs = list(range(len(layer_names)))
            cud = []
            for gi, name in enumerate(layer_names):
                group = [r for r in sub if r["name"] == name]
                c = next(r for r in group if r["stage"] == "cudnn")
                cud.append(c["gflops"])
                for si, st in enumerate(stages):
                    r = next(r for r in group if r["stage"] == st)
                    xpos = gi - 0.4 + width * (si + 0.5)
                    ax.bar(xpos, r["gflops"], width=width * 0.92, color=STAGE_COLORS[si],
                           edgecolor=SURFACE, linewidth=1.0, zorder=3)
                    # spread across repeats as a thin whisker
                    ax.plot([xpos, xpos], [r["gflops_min"], r["gflops_max"]], color=INK, lw=0.8,
                            zorder=4, solid_capstyle="butt")
                    if si == len(stages) - 1:
                        ax.text(xpos, r["gflops_max"] + ymax * 0.012, f"{r['pct_cudnn']:.0f}%",
                                ha="center", va="bottom", fontsize=8.5, color=INK2)
            ax.plot(xs, cud, color=CUDNN, lw=1.6, ls=(0, (4, 3)), marker="D", ms=6,
                    markerfacecolor=SURFACE, markeredgewidth=1.6, zorder=5)
            for gi, v in enumerate(cud):
                ax.text(gi + 0.42, v, f"{v:,.0f}", ha="left", va="center", fontsize=8.5, color=INK2)
            ax.set_xticks(xs)
            ax.set_xticklabels(layer_names)
            ax.set_xlim(-0.6, len(layer_names) - 0.4 + 0.35)
            ax.set_ylim(0, ymax)
            ax.set_title(f"{K}x{K} filter, batch {N}", loc="left", fontsize=11.5, color=INK, pad=8)
            ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", length=0)
            if j == 0:
                ax.set_ylabel("GFLOP/s")

    handles = [Patch(facecolor=STAGE_COLORS[i], label=STAGE_NAMES.get(st, st)) for i, st in enumerate(stages)]
    handles.append(Line2D([0], [0], color=CUDNN, lw=1.6, ls=(0, (4, 3)), marker="D", ms=6,
                          markerfacecolor=SURFACE, label="cuDNN (via PyTorch)"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.985), ncol=len(handles),
               frameon=False, fontsize=10)
    smi = dev.get("nvidia_smi", {})
    fig.suptitle("Direct 2D convolution, fp32 NCHW: my kernels against cuDNN", x=0.012, ha="left",
                 fontsize=15, fontweight="bold", color=INK, y=0.985)
    fig.text(0.012, 0.945,
             f"{dev['gpu']}, CUDA {dev['cuda_runtime']}, cuDNN {dev['cudnn']}, driver {smi.get('driver_version', '?')}. "
             f"Bars: median of {data['method']['repeats']} repeats, whisker = min to max. "
             f"Label = best stage as % of cuDNN. Peak fp32 {dev.get('peak_fp32_gflops', '?'):,} GFLOP/s.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "svg"):
        p = os.path.join(outdir, f"gflops.{ext}")
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        print("wrote", os.path.relpath(p, common.ROOT))


if __name__ == "__main__":
    main()
