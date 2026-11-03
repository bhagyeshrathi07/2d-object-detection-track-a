"""Run a detector over a COCO-format dataset and score it with pycocotools.

Raw detections (in COCO category ids) are cached in results/raw/ so that the
same predictions can be re-scored under several taxonomies without re-running
the network.

Usage:
    python src/evaluate.py --model tv:fasterrcnn_resnet50_fpn_v2 --dataset coco
    python src/evaluate.py --model ul:yolo11m.pt --dataset kitti --taxonomy driving3 driving5
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectors import DEVICE, build_detector, timed_predict
from taxonomy import coco_id_to_unified

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "coco": dict(images=ROOT / "data/coco/val2017",
                 ann=ROOT / "data/coco/annotations/instances_val2017.json",
                 split="COCO 2017 val (5,000 images)"),
    "kitti": dict(images=ROOT / "data/kitti/training/image_2",
                  ann=ROOT / "data/kitti/annotations/kitti_driving3.json",   # for image list only
                  split="KITTI 2D object detection 'training' set (7,481 labeled frames); no training done"),
}


def run_inference(det, ds, max_images=None, log_every=500):
    coco = COCO(str(ds["ann"]))
    img_ids = sorted(coco.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]
    dets, lat = [], []
    torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None
    # warm-up (shape-dependent kernels, cudnn autotune)
    for iid in img_ids[:5]:
        det.predict(Image.open(ds["images"] / coco.imgs[iid]["file_name"]).convert("RGB"))
    t_start = time.perf_counter()
    for n, iid in enumerate(img_ids):
        img = Image.open(ds["images"] / coco.imgs[iid]["file_name"]).convert("RGB")
        (boxes, scores, labels), ms = timed_predict(det, img)
        lat.append(ms)
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(float, b)
            dets.append({"image_id": int(iid), "category_id": int(l),
                         "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                         "score": round(float(s), 5)})
        if (n + 1) % log_every == 0:
            print(f"  {n + 1}/{len(img_ids)}  mean latency {np.mean(lat):.1f} ms", flush=True)
    wall = time.perf_counter() - t_start
    lat = np.array(lat)
    timing = dict(n_images=len(img_ids), latency_ms_mean=float(lat.mean()),
                  latency_ms_median=float(np.median(lat)), latency_ms_p90=float(np.percentile(lat, 90)),
                  fps_from_median=float(1000.0 / np.median(lat)), wall_s=wall,
                  peak_gpu_mem_MB=float(torch.cuda.max_memory_allocated() / 2**20) if DEVICE == "cuda" else None,
                  peak_gpu_reserved_MB=float(torch.cuda.max_memory_reserved() / 2**20) if DEVICE == "cuda" else None)
    return dets, img_ids, timing


def score(ann_file: Path, dets: list, img_ids: list, taxonomy: str | None):
    gt = COCO(str(ann_file))
    if taxonomy:
        mapped = []
        for d in dets:
            c = coco_id_to_unified(taxonomy, d["category_id"])
            if c is not None:
                mapped.append({**d, "category_id": c})
        dets = mapped
    if not dets:
        return {"note": "no detections"}
    dt = gt.loadRes(dets)
    E = COCOeval(gt, dt, "bbox")
    E.params.imgIds = img_ids
    E.evaluate(); E.accumulate(); E.summarize()
    names = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]
    res = {k: float(v) for k, v in zip(names, E.stats)}
    # per-class AP (area=all, maxDets=100), averaged over IoU 0.5:0.95 and over recall
    prec = E.eval["precision"]  # [T, R, K, A, M]
    per_class = {}
    for k, cid in enumerate(E.params.catIds):
        p = prec[:, :, k, 0, -1]
        p50 = prec[0, :, k, 0, -1]
        ap = float(np.mean(p[p > -1])) if (p > -1).any() else float("nan")
        ap50 = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else float("nan")
        n_gt = len(gt.getAnnIds(catIds=[cid], imgIds=img_ids, iscrowd=False))
        per_class[gt.cats[cid]["name"]] = dict(AP=ap, AP50=ap50, n_gt=n_gt)
    res["per_class"] = per_class
    res["n_detections_scored"] = len(dets)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--taxonomy", nargs="*", default=None,
                    help="for kitti: one or more of driving3 driving5 (default both)")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--score-thr", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640, help="ultralytics letterbox size")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--rescore-only", action="store_true", help="reuse cached raw detections")
    a = ap.parse_args()

    ds = DATASETS[a.dataset]
    det = None
    tag = a.model.split(":")[1].replace(".pt", "")
    if a.model.startswith("ulkitti2"):
        tag += "_kittiprompts2"
    elif a.model.startswith("ulkitti"):
        tag += "_kittiprompts"
    if a.fp16:
        tag += "_fp16"
    if a.model.startswith("ul") and a.imgsz != 640:
        tag += f"_{a.imgsz}"
    raw_dir = ROOT / "results/raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{tag}__{a.dataset}.json"

    if a.rescore_only and raw_path.exists():
        cache = json.load(open(raw_path))
        dets, img_ids, timing, info = cache["detections"], cache["img_ids"], cache["timing"], cache["model"]
    else:
        det = build_detector(a.model, score_thr=a.score_thr, fp16=a.fp16, imgsz=a.imgsz)
        info = det.info()
        print("model:", json.dumps(info, indent=1))
        dets, img_ids, timing = run_inference(det, ds, a.max_images)
        json.dump({"model": info, "timing": timing, "img_ids": img_ids, "detections": dets}, open(raw_path, "w"))
        print(f"cached {len(dets)} raw detections -> {raw_path}")

    taxonomies = [None] if a.dataset == "coco" else (a.taxonomy or ["driving3", "driving5"])
    metrics = {}
    for tax in taxonomies:
        ann = ds["ann"] if tax is None else ROOT / f"data/kitti/annotations/kitti_{tax}.json"
        print(f"\n=== {tag} on {a.dataset} [{tax or 'coco80'}] ===")
        metrics[tax or "coco80"] = score(ann, dets, img_ids, tax)

    out = dict(model=info, dataset=a.dataset, split=ds["split"], n_images=len(img_ids),
               timing=timing, metrics=metrics,
               hardware=dict(gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
                             cpu=platform.processor(), os=platform.platform(), python=sys.version.split()[0],
                             torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version()),
               command=" ".join(sys.argv))
    out_dir = ROOT / "results/metrics"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}__{a.dataset}.json"
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
