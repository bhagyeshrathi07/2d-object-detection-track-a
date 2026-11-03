"""Why is KITTI 'cyclist' AP ~0 for every COCO-vocabulary detector?

KITTI draws ONE box around rider + bicycle. COCO detectors emit a 'bicycle' box
(bike only, roughly the lower half of the KITTI box) and a separate 'person'
box (the rider). This script (CPU only, re-uses results/raw/*__kitti.json)

  1. measures, for every Cyclist GT box, the best IoU reached by any bicycle /
     motorcycle detection, by any person detection, and by their union;
  2. evaluates a simple post-hoc "rider + bike fusion" rule: a bicycle or
     motorcycle detection and a person detection that overlap (IoU > 0.1,
     person top above bike top, x-overlap > 50 % of the narrower box) are
     merged into one 'cyclist' detection with the union box and score
     sqrt(s_bike * s_person); the person detection is consumed so that it
     no longer counts as a false-positive pedestrian.

Writes results/cyclist_fusion.json and results/cyclist_fusion.md.
"""
from __future__ import annotations

import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from taxonomy import coco_id_to_unified, unified_cat_id

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data/kitti/annotations/kitti_driving3.json"
BIKE, PERSON = {2, 4}, 1  # COCO ids: bicycle, motorcycle, person


def xywh2xyxy(b):
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=float)


def iou(a, B):
    """a: (4,), B: (N,4) in xyxy."""
    if len(B) == 0:
        return np.zeros(0)
    ix1 = np.maximum(a[0], B[:, 0]); iy1 = np.maximum(a[1], B[:, 1])
    ix2 = np.minimum(a[2], B[:, 2]); iy2 = np.minimum(a[3], B[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1]); area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def fuse(dets_img, min_score=0.05):
    """dets_img: list of raw COCO-id detections for one image -> list of driving3 detections."""
    bikes = [d for d in dets_img if d["category_id"] in BIKE and d["score"] >= min_score]
    persons = [d for d in dets_img if d["category_id"] == PERSON and d["score"] >= min_score]
    others = [d for d in dets_img if d["category_id"] not in BIKE and d["category_id"] != PERSON]
    low = [d for d in dets_img if (d["category_id"] in BIKE or d["category_id"] == PERSON) and d["score"] < min_score]
    P = np.array([xywh2xyxy(p["bbox"]) for p in persons]) if persons else np.zeros((0, 4))
    used = np.zeros(len(persons), bool)
    out = []
    for b in sorted(bikes, key=lambda d: -d["score"]):
        bb = xywh2xyxy(b["bbox"])
        best, best_j = 0.0, -1
        if len(P):
            ious = iou(bb, P)
            for j in np.argsort(-ious):
                if used[j] or ious[j] <= 0.1:
                    break
                p = P[j]
                xov = max(0, min(bb[2], p[2]) - max(bb[0], p[0])) / max(1e-6, min(bb[2] - bb[0], p[2] - p[0]))
                if p[1] < bb[1] and xov > 0.5:
                    best, best_j = ious[j], j; break
        if best_j >= 0:
            used[best_j] = True
            p = P[best_j]
            u = [min(bb[0], p[0]), min(bb[1], p[1]), max(bb[2], p[2]), max(bb[3], p[3])]
            out.append({"image_id": b["image_id"], "category_id": unified_cat_id("driving3", "cyclist"),
                        "bbox": [u[0], u[1], u[2] - u[0], u[3] - u[1]],
                        "score": float(np.sqrt(b["score"] * persons[best_j]["score"]))})
        else:
            out.append({**b, "category_id": unified_cat_id("driving3", "cyclist")})
    for j, p in enumerate(persons):
        if not used[j]:
            out.append({**p, "category_id": unified_cat_id("driving3", "person")})
    for d in others + low:
        c = coco_id_to_unified("driving3", d["category_id"])
        if c is not None:
            out.append({**d, "category_id": c})
    return out


def evaluate(gt: COCO, dets, img_ids):
    with redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox"); E.params.imgIds = img_ids
        E.evaluate(); E.accumulate(); E.summarize()
    prec = E.eval["precision"]; per = {}
    for k, cid in enumerate(E.params.catIds):
        p = prec[:, :, k, 0, -1]; p50 = prec[0, :, k, 0, -1]
        per[gt.cats[cid]["name"]] = dict(AP=float(p[p > -1].mean()), AP50=float(p50[p50 > -1].mean()))
    return dict(mAP=float(E.stats[0]), AP50=float(E.stats[1]), per_class=per)


def main():
    with redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN))
    cyc_id = unified_cat_id("driving3", "cyclist")
    results, L = {}, []
    for raw in sorted((ROOT / "results/raw").glob("*__kitti.json")):
        tag = raw.stem.split("__")[0]
        if "kittiprompts" in tag:
            continue
        cache = json.load(open(raw)); img_ids = cache["img_ids"]
        by_img = defaultdict(list)
        for d in cache["detections"]:
            by_img[d["image_id"]].append(d)
        # ---- 1. IoU diagnostic ------------------------------------------
        best_bike, best_person, best_union = [], [], []
        for iid in img_ids:
            anns = [a for a in gt.loadAnns(gt.getAnnIds(imgIds=[iid], catIds=[cyc_id], iscrowd=False))]
            if not anns:
                continue
            ds = by_img.get(iid, [])
            Bk = np.array([xywh2xyxy(d["bbox"]) for d in ds if d["category_id"] in BIKE and d["score"] > 0.3]).reshape(-1, 4)
            Pe = np.array([xywh2xyxy(d["bbox"]) for d in ds if d["category_id"] == PERSON and d["score"] > 0.3]).reshape(-1, 4)
            Fu = np.array([xywh2xyxy(d["bbox"]) for d in fuse(ds, 0.3) if d["category_id"] == cyc_id]).reshape(-1, 4)
            for a in anns:
                g = xywh2xyxy(a["bbox"])
                best_bike.append(iou(g, Bk).max() if len(Bk) else 0.0)
                best_person.append(iou(g, Pe).max() if len(Pe) else 0.0)
                best_union.append(iou(g, Fu).max() if len(Fu) else 0.0)
        diag = {k: dict(median_iou=float(np.median(v)), frac_iou_ge_05=float(np.mean(np.array(v) >= 0.5)),
                        frac_iou_ge_03=float(np.mean(np.array(v) >= 0.3)))
                for k, v in (("bicycle_box", best_bike), ("person_box", best_person), ("fused_box", best_union))}
        # ---- 2. baseline vs fused scoring -------------------------------
        base_dets = []
        for d in cache["detections"]:
            c = coco_id_to_unified("driving3", d["category_id"])
            if c is not None:
                base_dets.append({**d, "category_id": c})
        fused_dets = [d for iid in img_ids for d in fuse(by_img.get(iid, []))]
        base, fused = evaluate(gt, base_dets, img_ids), evaluate(gt, fused_dets, img_ids)
        results[tag] = dict(diag=diag, baseline=base, fused=fused)
        print(f"{tag:28s} best IoU w/ cyclist GT (median): bike {diag['bicycle_box']['median_iou']:.2f} person {diag['person_box']['median_iou']:.2f} "
              f"fused {diag['fused_box']['median_iou']:.2f} | cyclist AP {base['per_class']['cyclist']['AP']:.3f}->{fused['per_class']['cyclist']['AP']:.3f} "
              f"person AP {base['per_class']['person']['AP']:.3f}->{fused['per_class']['person']['AP']:.3f} mAP {base['mAP']:.3f}->{fused['mAP']:.3f}", flush=True)
    json.dump(results, open(ROOT / "results/cyclist_fusion.json", "w"), indent=1)
    L.append("## Cyclist box geometry: best IoU of a detection with each KITTI Cyclist GT box (score > 0.3)\n")
    L.append("| Model | bicycle box: median IoU | ≥0.5 | person box: median IoU | ≥0.5 | fused rider+bike box: median IoU | ≥0.5 |")
    L.append("|---|---|---|---|---|---|---|")
    for t, r in results.items():
        d = r["diag"]
        L.append(f"| {t} | {d['bicycle_box']['median_iou']:.2f} | {d['bicycle_box']['frac_iou_ge_05']:.0%} | {d['person_box']['median_iou']:.2f} | "
                 f"{d['person_box']['frac_iou_ge_05']:.0%} | {d['fused_box']['median_iou']:.2f} | {d['fused_box']['frac_iou_ge_05']:.0%} |")
    L.append("\n## driving3 scores with post-hoc rider+bike fusion\n")
    L.append("| Model | mAP | mAP fused | cyclist AP | fused | cyclist AP50 | fused | person AP | fused | vehicle AP |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for t, r in results.items():
        b, f = r["baseline"], r["fused"]
        L.append(f"| {t} | {b['mAP']:.3f} | {f['mAP']:.3f} | {b['per_class']['cyclist']['AP']:.3f} | {f['per_class']['cyclist']['AP']:.3f} | "
                 f"{b['per_class']['cyclist']['AP50']:.3f} | {f['per_class']['cyclist']['AP50']:.3f} | {b['per_class']['person']['AP']:.3f} | "
                 f"{f['per_class']['person']['AP']:.3f} | {f['per_class']['vehicle']['AP']:.3f} |")
    (ROOT / "results/cyclist_fusion.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
