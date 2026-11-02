"""What ncu runs: one launch of every stage on every benchmark shape, then one
cuDNN call per shape so its kernels show up in the same report. No timing
here; ncu replays each launch itself."""
import argparse

import torch

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=None)
    ap.add_argument("--batches", default="1", help="comma list of batch sizes to include")
    args = ap.parse_args()
    common.setup_torch()
    ext = common.build()
    stages = args.stages.split(",") if args.stages else list(ext.stages())
    batches = {int(b) for b in args.batches.split(",")}
    for s in common.shapes():
        if s["N"] not in batches:
            continue
        x, w = common.make_inputs(s)
        # cuDNN first, with a warm-up so autotuning does not land in the report
        for _ in range(3):
            common.cudnn_conv(x, w, s)
        torch.cuda.synchronize()
        print("SHAPE", common.shape_label(s), flush=True)
        common.cudnn_conv(x, w, s)
        torch.cuda.synchronize()
        for st in stages:
            ext.conv2d(x, w, s["S"], s["P"], st)
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
