"""Controlled latency / throughput / memory benchmark (batch size 1).

Each model is run on a fixed image (a KITTI frame, 1242x375, and a COCO image)
with warm-up, then timed with CUDA synchronisation around each call. Both
fp32 and fp16 are measured where the framework supports it.

Usage:
    python src/latency.py --models tv:fasterrcnn_resnet50_fpn_v2 ul:yolo11n.pt ... --iters 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from detectors import DEVICE, build_detector, cuda_sync

ROOT = Path(__file__).resolve().parents[1]


def bench(det, img, warmup, iters):
    for _ in range(warmup):
        det.predict(img)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        cuda_sync(); t0 = time.perf_counter()
        det.predict(img)
        cuda_sync(); times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return dict(mean_ms=float(t.mean()), median_ms=float(np.median(t)), std_ms=float(t.std()),
                p90_ms=float(np.percentile(t, 90)), fps=float(1000 / np.median(t)),
                peak_mem_alloc_MB=float(torch.cuda.max_memory_allocated() / 2**20),
                peak_mem_reserved_MB=float(torch.cuda.max_memory_reserved() / 2**20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--kitti-img", default=str(ROOT / "data/kitti/training/image_2/000008.png"))
    ap.add_argument("--coco-img", default=str(ROOT / "data/coco/val2017/000000000139.jpg"))
    a = ap.parse_args()

    images = {"kitti_1242x375": Image.open(a.kitti_img).convert("RGB"),
              "coco_640x426": Image.open(a.coco_img).convert("RGB")}
    out_path = ROOT / "results/latency.json"
    results = json.load(open(out_path)) if out_path.exists() else {}
    for spec in a.models:
        for fp16 in (False, True):
            det = build_detector(spec, score_thr=0.25, fp16=fp16)
            key = det.name + ("_fp16" if fp16 else "_fp32")
            results[key] = dict(model=det.info(), gpu=torch.cuda.get_device_name(0),
                                n_params_M=float(sum(p.numel() for p in det.parameters()) / 1e6),
                                iters=a.iters, warmup=a.warmup)
            for img_name, img in images.items():
                r = bench(det, img, a.warmup, a.iters)
                results[key][img_name] = r
                print(f"{key:40s} {img_name:16s} median {r['median_ms']:7.2f} ms  "
                      f"fps {r['fps']:6.1f}  peak alloc {r['peak_mem_alloc_MB']:7.0f} MB", flush=True)
            del det; torch.cuda.empty_cache()
            json.dump(results, open(out_path, "w"), indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
