"""Correctness: every stage against cuDNN on the same inputs.

Reports max absolute error and max error relative to the largest output
magnitude, per stage and shape, plus the resnet18 conv1 shape (7x7, stride 2)
that the feature-map figure uses. A kernel that fails the tolerance makes the
script exit non-zero so the one-command run stops here.
"""
import argparse
import sys

import torch

import common

ATOL = 1e-3   # loose bound: fp32 sums of up to 6400 terms in a different order


def check_shape(ext, stages, s):
    x, w = common.make_inputs(s)
    ref = common.cudnn_conv(x, w, s)
    ref_max = ref.abs().max().item()
    rows = []
    for st in stages:
        y = ext.conv2d(x, w, s["S"], s["P"], st)
        torch.cuda.synchronize()
        diff = (y - ref).abs()
        max_abs = diff.max().item()
        rows.append(dict(stage=st, shape=common.shape_label(s), N=s["N"], Ci=s["Ci"], Co=s["Co"],
                         H=s["H"], W=s["W"], K=s["K"], S=s["S"], max_abs_err=max_abs,
                         max_rel_err=max_abs / ref_max, ref_max=ref_max,
                         mean_abs_err=diff.mean().item(), ok=bool(max_abs <= ATOL)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="latest")
    ap.add_argument("--stages", default=None, help="comma list, default all built stages")
    args = ap.parse_args()

    common.setup_torch()
    ext = common.build()
    stages = args.stages.split(",") if args.stages else list(ext.stages())

    shapes = common.shapes() + [
        # resnet18 conv1: 7x7, stride 2, pad 3, used by the feature-map figure
        dict(name="resnet18.conv1", N=1, Ci=3, Co=64, H=224, W=224, K=7, S=2, P=3),
        # odd sizes so partial tiles on every edge get exercised
        dict(name="odd 5->7 @37x53", N=2, Ci=5, Co=7, H=37, W=53, K=3, S=1, P=1),
        dict(name="odd 5->7 @37x53 s2", N=2, Ci=5, Co=7, H=37, W=53, K=5, S=2, P=2),
    ]
    rows = []
    for s in shapes:
        rows += check_shape(ext, stages, s)

    print(f"{'stage':<10}{'shape':<26}{'max abs':>12}{'max rel':>12}{'mean abs':>12}  ok")
    for r in rows:
        print(f"{r['stage']:<10}{r['shape']:<26}{r['max_abs_err']:>12.3e}{r['max_rel_err']:>12.3e}"
              f"{r['mean_abs_err']:>12.3e}  {'yes' if r['ok'] else 'NO'}")

    summary = {}
    for st in stages:
        mine = [r for r in rows if r["stage"] == st]
        summary[st] = dict(max_abs_err=max(r["max_abs_err"] for r in mine),
                           max_rel_err=max(r["max_rel_err"] for r in mine),
                           all_ok=all(r["ok"] for r in mine))
    print()
    for st, v in summary.items():
        print(f"{st:<10} worst max abs {v['max_abs_err']:.3e}  worst max rel {v['max_rel_err']:.3e}"
              f"  {'PASS' if v['all_ok'] else 'FAIL'}")

    out = dict(device=common.device_record(), atol=ATOL, rows=rows, summary=summary)
    common.dump(out, f"{common.result_dir(args.label)}/correctness.json")
    if not all(v["all_ok"] for v in summary.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
