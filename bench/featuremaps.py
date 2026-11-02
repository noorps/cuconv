"""Feature maps: resnet18's first conv layer, my kernel against cuDNN.

Loads conv1 (64 filters, 3 channels, 7x7, stride 2, pad 3) from torchvision's
pretrained resnet18, runs it over a photograph with my kernel and with cuDNN,
and renders both sets of 64 output channels side by side. The max error between
the two sits under the grids. The photograph is "Chelsea" from scikit-image's
sample data, released under CC0 by Stefan van der Walt.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_photo():
    from skimage import data
    from skimage.transform import resize
    img = data.chelsea()                      # (300, 451, 3) uint8, CC0
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    out = resize(crop, (224, 224), anti_aliasing=True)   # float64 in 0..1
    credit = "Chelsea, scikit-image sample data, CC0 (Stefan van der Walt)"
    try:
        import skimage
        lic = os.path.join(os.path.dirname(skimage.__file__), "data", "README.txt")
        if os.path.exists(lic):
            with open(lic) as f:
                line = next((l.strip() for l in f if "chelsea" in l.lower()), None)
            if line:
                credit += f" [{line}]"
    except Exception:  # noqa: BLE001
        pass
    return out.astype(np.float32), credit


def grid(ax, maps, title):
    """64 channels as an 8x8 grid, each channel scaled by its own range."""
    n, h, w = maps.shape
    gap = 2
    canvas = np.full((8 * h + 7 * gap, 8 * w + 7 * gap), np.nan, dtype=np.float32)
    for k in range(n):
        m = maps[k]
        lo, hi = m.min(), m.max()
        m = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
        r, c = divmod(k, 8)
        canvas[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = m
    cmap = matplotlib.colormaps["gray"].copy()
    cmap.set_bad(SURFACE)
    ax.imshow(canvas, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=8)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="latest")
    ap.add_argument("--stage", default=None, help="which of my kernels to draw; default the last stage")
    args = ap.parse_args()
    common.setup_torch()
    ext = common.build()
    stage = args.stage or list(ext.stages())[-1]
    outdir = common.result_dir(args.label)

    from torchvision.models import ResNet18_Weights, resnet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    w = model.conv1.weight.detach().float().cuda().contiguous()      # (64, 3, 7, 7)

    photo, credit = load_photo()
    x = torch.from_numpy(photo).permute(2, 0, 1)
    x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
    x = x[None].float().cuda().contiguous()                          # (1, 3, 224, 224)

    s = dict(S=2, P=3)
    ref = common.cudnn_conv(x, w, s)
    mine = ext.conv2d(x, w, 2, 3, stage)
    torch.cuda.synchronize()
    diff = (mine - ref).abs()
    max_abs = diff.max().item()
    ref_max = ref.abs().max().item()
    # fp64 ground truth on the CPU, so "identical to cuDNN" is not the only evidence
    truth = torch.nn.functional.conv2d(x.double().cpu(), w.double().cpu(), stride=2, padding=3)
    mine_vs_truth = (mine.double().cpu() - truth).abs().max().item()
    cudnn_vs_truth = (ref.double().cpu() - truth).abs().max().item()
    stats = dict(stage=stage, max_abs_err=max_abs, max_rel_err=max_abs / ref_max, ref_max=ref_max,
                 mean_abs_err=diff.mean().item(), n_values=ref.numel(),
                 max_abs_err_vs_fp64=mine_vs_truth, cudnn_max_abs_err_vs_fp64=cudnn_vs_truth,
                 output_shape=list(ref.shape), weights="torchvision resnet18 IMAGENET1K_V1 conv1",
                 image=credit, device=common.device_record())
    print(f"{stage} vs cuDNN on resnet18.conv1: max abs {max_abs:.3e}, max rel {stats['max_rel_err']:.3e}"
          f" over {ref.numel():,} values; vs fp64: mine {mine_vs_truth:.3e}, cuDNN {cudnn_vs_truth:.3e}")

    m = mine[0].cpu().numpy()
    r = ref[0].cpu().numpy()
    fig = plt.figure(figsize=(13, 7.6), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.6, 2.6], wspace=0.06, left=0.02, right=0.98,
                          top=0.86, bottom=0.11)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(np.clip(photo, 0, 1))
    ax0.set_title("input, 224x224", loc="left", fontsize=12, color=INK, pad=8)
    ax0.axis("off")
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    name = {"naive": "stage 1, naive", "smem": "stage 2, shared memory",
            "regtile": "stage 3, register tile",
            "vec": "stage 4, float4 smem + channel batch"}.get(stage, stage)
    grid(ax1, m, f"my kernel ({name.split(',')[0]}), 64 channels")
    grid(ax2, r, "cuDNN, 64 channels")
    fig.suptitle("resnet18 conv1 (7x7, stride 2) over a photograph: my kernel vs cuDNN", x=0.02,
                 ha="left", fontsize=15, fontweight="bold", color=INK, y=0.975)
    fig.text(0.02, 0.925, f"Pretrained ImageNet weights from torchvision. {stats['device']['gpu']}, "
             f"CUDA {stats['device']['cuda_runtime']}, fp32. Each channel scaled to its own range.",
             fontsize=9.5, color=INK2)
    fig.text(0.5, 0.065, f"max |mine - cuDNN| = {max_abs:.2e}  (relative to max |cuDNN| = {ref_max:.2f}: "
             f"{stats['max_rel_err']:.2e}), over {ref.numel():,} output values",
             ha="center", fontsize=12.5, color=INK)
    fig.text(0.5, 0.035, f"against an fp64 reference on the CPU: max |mine - fp64| = {mine_vs_truth:.2e}, "
             f"max |cuDNN - fp64| = {cudnn_vs_truth:.2e}", ha="center", fontsize=10.5, color=INK2)
    fig.text(0.98, 0.012, f"photo: {credit.split(' [')[0]}", ha="right", fontsize=8.5, color=INK2)
    for e in ("png", "svg"):
        p = os.path.join(outdir, f"featuremaps.{e}")
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        print("wrote", os.path.relpath(p, common.ROOT))
    common.dump(stats, os.path.join(outdir, "featuremaps.json"))


if __name__ == "__main__":
    main()
