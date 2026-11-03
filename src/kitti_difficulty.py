"""Extra KITTI analyses that re-score the cached raw detections (no GPU needed).

1. KITTI difficulty breakdown (driving3 taxonomy). KITTI defines
       Easy:     bbox height >= 40 px, occlusion 0,    truncation <= 15 %
       Moderate: bbox height >= 25 px, occlusion <= 1, truncation <= 30 %
       Hard:     bbox height >= 25 px, occlusion <= 2, truncation <= 50 %
   Following the official KITTI protocol, GT boxes that do not satisfy the
   level are turned into ignore regions (iscrowd=1) rather than deleted, so
   detections on them are neither TP nor FP.

2. Rider/person taxonomy mismatch. KITTI boxes a cyclist (rider + bike) as one
   object, COCO detectors emit a separate 'person' box for the rider, which is
   a false positive for the 'person' class. Re-scoring with every Cyclist GT
   box additionally registered as a *person* ignore region shows how much of
   the 'person' AP gap is due to this labelling convention.

Writes results/kitti_extra.json and prints a markdown table.
"""
from __future__ import annotations

import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from taxonomy import coco_id_to_unified

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data/kitti/annotations/kitti_driving3.json"
LEVELS = {"easy": (40, 0, 0.15), "moderate": (25, 1, 0.30), "hard": (25, 2, 0.50)}


def evaluate(gt_dict, dets, img_ids):
    gt = COCO(); gt.dataset = copy.deepcopy(gt_dict);
    with redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox"); E.params.imgIds = img_ids
        E.evaluate(); E.accumulate(); E.summarize()
    prec = E.eval["precision"]
    per = {}
    for k, cid in enumerate(E.params.catIds):
        p = prec[:, :, k, 0, -1]; p50 = prec[0, :, k, 0, -1]
        per[gt.cats[cid]["name"]] = dict(AP=float(p[p > -1].mean()) if (p > -1).any() else float("nan"),
                                         AP50=float(p50[p50 > -1].mean()) if (p50 > -1).any() else float("nan"))
    return dict(mAP=float(E.stats[0]), AP50=float(E.stats[1]), per_class=per)


def main():
    base = json.load(open(ANN))
    results = {}
    for raw in sorted((ROOT / "results/raw").glob("*__kitti.json")):
        tag = raw.stem.split("__")[0]
        cache = json.load(open(raw))
        img_ids = cache["img_ids"]
        dets = []
        for d in cache["detections"]:
            c = coco_id_to_unified("driving3", d["category_id"])
            if c is not None:
                dets.append({**d, "category_id": c})
        out = {}
        # --- difficulty levels -------------------------------------------
        for lvl, (min_h, max_occ, max_trunc) in LEVELS.items():
            g = copy.deepcopy(base)
            for a in g["annotations"]:
                if a["iscrowd"]:
                    continue
                h = a["bbox"][3]
                if h < min_h or a["occluded"] > max_occ or a["truncated"] > max_trunc:
                    a["iscrowd"] = 1
            n_valid = sum(1 for a in g["annotations"] if not a["iscrowd"])
            out[lvl] = evaluate(g, dets, img_ids); out[lvl]["n_gt"] = n_valid
        out["all"] = evaluate(base, dets, img_ids)
        # --- rider-as-person ignore --------------------------------------
        g = copy.deepcopy(base)
        person_id = next(c["id"] for c in g["categories"] if c["name"] == "person")
        cyc_id = next(c["id"] for c in g["categories"] if c["name"] == "cyclist")
        nid = max(a["id"] for a in g["annotations"]) + 1
        for a in list(g["annotations"]):
            if a["category_id"] == cyc_id and not a["iscrowd"]:
                g["annotations"].append({**a, "id": nid, "category_id": person_id, "iscrowd": 1}); nid += 1
        out["rider_ignored"] = evaluate(g, dets, img_ids)
        results[tag] = out
        print(f"{tag:32s} all {out['all']['mAP']:.3f} | easy {out['easy']['mAP']:.3f} mod {out['moderate']['mAP']:.3f} "
              f"hard {out['hard']['mAP']:.3f} | person AP {out['all']['per_class']['person']['AP']:.3f} -> "
              f"{out['rider_ignored']['per_class']['person']['AP']:.3f} with rider boxes ignored", flush=True)
    json.dump(results, open(ROOT / "results/kitti_extra.json", "w"), indent=1)

    L = ["\n## KITTI difficulty breakdown (driving3 mAP@[.5:.95], GT outside the level treated as ignore)\n",
         "| Model | all | Easy | Moderate | Hard | vehicle E/M/H AP50 | person E/M/H AP50 | cyclist E/M/H AP50 |", "|---|---|---|---|---|---|---|---|"]
    for t, o in results.items():
        f3 = lambda c: "/".join(f"{o[l]['per_class'][c]['AP50']:.2f}" for l in ("easy", "moderate", "hard"))
        L.append(f"| {t} | {o['all']['mAP']:.3f} | {o['easy']['mAP']:.3f} | {o['moderate']['mAP']:.3f} | {o['hard']['mAP']:.3f} | {f3('vehicle')} | {f3('person')} | {f3('cyclist')} |")
    L += ["\n## Rider-as-person mismatch (driving3)\n", "| Model | person AP | person AP, rider boxes ignored | person AP50 | person AP50, rider ignored |", "|---|---|---|---|---|"]
    for t, o in results.items():
        a, b = o["all"]["per_class"]["person"], o["rider_ignored"]["per_class"]["person"]
        L.append(f"| {t} | {a['AP']:.3f} | {b['AP']:.3f} | {a['AP50']:.3f} | {b['AP50']:.3f} |")
    (ROOT / "results/kitti_extra.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
