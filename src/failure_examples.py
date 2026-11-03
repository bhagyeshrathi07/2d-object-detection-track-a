"""Qualitative failure examples for the report: GT vs predictions on three KITTI frames.

Picks, deterministically, from the cached raw detections of one model:
  (a) a frame with a KITTI Cyclist where the detector emits a separate 'person'
      and 'bicycle' box (box-definition mismatch);
  (b) a frame with many small pedestrians where several are missed (scale/occlusion);
  (c) a frame with a KITTI Truck that the detector labels 'car' or 'bus'
      (class-confusion; only matters for driving5).

GT boxes are drawn green (dashed when they are ignore regions), predictions with
score >= --thr in a per-class colour with their COCO class name and score.
Writes results/figures/failure_examples.png and prints the chosen frame ids.

Usage:
    python src/failure_examples.py --model rtdetr-l --thr 0.3
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from taxonomy import COCO_ID2NAME

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data/kitti/training/image_2"
RAW_ANN = ROOT / "data/kitti/annotations/kitti_raw.json"
COLORS = {"person": (255, 80, 80), "bicycle": (80, 140, 255), "motorcycle": (80, 140, 255),
          "car": (255, 200, 0), "truck": (255, 120, 255), "bus": (0, 220, 220), "train": (0, 220, 220)}
GT_COLOR = (0, 230, 0)


def xyxy(b):
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], float)


def iou(a, B):
    if len(B) == 0:
        return np.zeros(0)
    ix1 = np.maximum(a[0], B[:, 0]); iy1 = np.maximum(a[1], B[:, 1])
    ix2 = np.minimum(a[2], B[:, 2]); iy2 = np.minimum(a[3], B[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


def load(model, thr):
    raw = json.load(open(ROOT / f"results/raw/{model}__kitti.json"))
    dets = defaultdict(list)
    for d in raw["detections"]:
        if d["score"] >= thr:
            dets[d["image_id"]].append(d)
    gt_json = json.load(open(RAW_ANN))
    cat = {c["id"]: c["name"] for c in gt_json["categories"]}
    gts = defaultdict(list)
    for a in gt_json["annotations"]:
        gts[a["image_id"]].append({**a, "name": cat[a["category_id"]]})
    return dets, gts, raw["img_ids"]


def pick_cyclist(dets, gts, ids):
    """Cyclist GT with a person box (IoU>=0.4) AND a bicycle box (IoU>=0.15) but no single box with IoU>=0.5."""
    best = None
    for iid in ids:
        cyc = [g for g in gts[iid] if g["name"] == "Cyclist" and g["bbox"][3] >= 60]
        if not cyc:
            continue
        P = np.array([xyxy(d["bbox"]) for d in dets[iid] if d["category_id"] == 1]).reshape(-1, 4)
        B = np.array([xyxy(d["bbox"]) for d in dets[iid] if d["category_id"] in (2, 4)]).reshape(-1, 4)
        for g in cyc:
            gb = xyxy(g["bbox"])
            ip = iou(gb, P).max() if len(P) else 0
            ib = iou(gb, B).max() if len(B) else 0
            if ip >= 0.4 and ib >= 0.15 and max(ip, ib) < 0.5:
                score = g["bbox"][3]  # prefer big, clearly visible cyclists
                if best is None or score > best[0]:
                    best = (score, iid, dict(person_iou=ip, bike_iou=ib, h=g["bbox"][3]))
    return best


def _ped_stats(dets, gts, iid):
    ped = [g for g in gts[iid] if g["name"] == "Pedestrian"]
    P = np.array([xyxy(d["bbox"]) for d in dets[iid] if d["category_id"] == 1]).reshape(-1, 4)
    hit = sum(1 for g in ped if len(P) and iou(xyxy(g["bbox"]), P).max() >= 0.5)
    hs = [g["bbox"][3] for g in ped]
    return ped, dict(n=len(ped), hit=hit, miss=len(ped) - hit,
                     median_h=float(np.median(hs)) if hs else 0.0, min_h=float(min(hs)) if hs else 0.0)


def pick_small_pedestrians(dets, gts, ids):
    """Scale failure: >=3 pedestrians, all under 50 px high, at least half missed."""
    best = None
    for iid in ids:
        ped, s = _ped_stats(dets, gts, iid)
        if s["n"] < 3 or max((g["bbox"][3] for g in ped), default=0) >= 50:
            continue
        if s["miss"] * 2 >= s["n"]:
            key = (s["miss"], -s["median_h"])
            if best is None or key > best[0]:
                best = (key, iid, s)
    return best


def pick_crowd(dets, gts, ids):
    """Occlusion failure: >=6 pedestrians that are NOT small (median >= 100 px), at least a third missed."""
    best = None
    for iid in ids:
        ped, s = _ped_stats(dets, gts, iid)
        if s["n"] < 6 or s["median_h"] < 100:
            continue
        if s["miss"] >= max(2, s["n"] // 3):
            key = (s["miss"], s["n"])
            if best is None or key > best[0]:
                best = (key, iid, s)
    return best


def pick_truck(dets, gts, ids):
    """KITTI Truck (height>=50) whose best-overlapping detection (IoU>=0.5) is 'car', with no truck/bus box.

    Note: KITTI has no Bus class (buses are labelled Truck) and driving5 maps COCO bus -> truck,
    so a 'bus' prediction on a KITTI Truck is scored correctly and is NOT a failure."""
    best = None
    for iid in ids:
        trucks = [g for g in gts[iid] if g["name"] == "Truck" and g["bbox"][3] >= 50]
        if not trucks:
            continue
        ds = dets[iid]
        if not ds:
            continue
        D = np.array([xyxy(d["bbox"]) for d in ds])
        for g in trucks:
            ious = iou(xyxy(g["bbox"]), D)
            j = int(ious.argmax())
            name = COCO_ID2NAME[ds[j]["category_id"]]
            if ious[j] >= 0.5 and name == "car":
                has_truck = any(COCO_ID2NAME[d["category_id"]] in ("truck", "bus") and ious[k] >= 0.5
                                for k, d in enumerate(ds))
                if has_truck:
                    continue
                key = g["bbox"][3]
                if best is None or key > best[0]:
                    best = (key, iid, dict(pred=name, score=ds[j]["score"], iou=float(ious[j]), h=g["bbox"][3]))
    return best


def draw(iid, dets, gts, thr, title, font):
    im = Image.open(IMG_DIR / f"{iid:06d}.png").convert("RGB")
    dr = ImageDraw.Draw(im)
    for g in gts[iid]:
        x1, y1, x2, y2 = xyxy(g["bbox"])
        if g["name"] in ("DontCare", "Misc"):
            for x in np.arange(x1, x2, 8):
                dr.line([(x, y1), (min(x + 4, x2), y1)], fill=GT_COLOR, width=1)
                dr.line([(x, y2), (min(x + 4, x2), y2)], fill=GT_COLOR, width=1)
            continue
        dr.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=3)
        dr.text((x1 + 2, max(0, y1 - 13)), g["name"], fill=GT_COLOR, font=font)
    for d in sorted(dets[iid], key=lambda d: d["score"]):
        name = COCO_ID2NAME.get(d["category_id"], "?")
        if name not in COLORS:
            continue
        x1, y1, x2, y2 = xyxy(d["bbox"])
        c = COLORS[name]
        dr.rectangle([x1, y1, x2, y2], outline=c, width=2)
        dr.text((x1 + 2, min(im.height - 12, y2 + 1)), f"{name} {d['score']:.2f}", fill=c, font=font)
    # title bar
    bar = Image.new("RGB", (im.width, 22), (20, 20, 20))
    ImageDraw.Draw(bar).text((6, 4), title, fill=(255, 255, 255), font=font)
    out = Image.new("RGB", (im.width, im.height + 22))
    out.paste(bar, (0, 0)); out.paste(im, (0, 22))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rtdetr-l")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default=str(ROOT / "results/figures/failure_examples.png"))
    a = ap.parse_args()
    dets, gts, ids = load(a.model, a.thr)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    picks = {}
    c = pick_cyclist(dets, gts, ids)
    s = pick_small_pedestrians(dets, gts, ids)
    p = pick_crowd(dets, gts, ids)
    t = pick_truck(dets, gts, ids)
    panels = []
    if c:
        _, iid, info = c; picks["cyclist"] = dict(frame=iid, **info)
        panels.append(draw(iid, dets, gts, a.thr, f"(a) frame {iid:06d}: KITTI 'Cyclist' = one box around rider+bike; "
                           f"{a.model} emits person (IoU {info['person_iou']:.2f}) + bicycle (IoU {info['bike_iou']:.2f}) -> no match at IoU 0.5", font))
    if s:
        _, iid, info = s; picks["small_pedestrians"] = dict(frame=iid, **info)
        panels.append(draw(iid, dets, gts, a.thr, f"(b) frame {iid:06d}: {info['n']} distant pedestrians, all < 50 px high (median {info['median_h']:.0f} px); "
                           f"{info['hit']} found, {info['miss']} missed at score >= {a.thr} (scale)", font))
    if p:
        _, iid, info = p; picks["crowd"] = dict(frame=iid, **info)
        panels.append(draw(iid, dets, gts, a.thr, f"(c) frame {iid:06d}: {info['n']} pedestrians in a group (median height {info['median_h']:.0f} px); "
                           f"{info['hit']} found, {info['miss']} missed at score >= {a.thr} (occlusion); parked bicycles -> 'cyclist' false positives", font))
    if t:
        _, iid, info = t; picks["truck"] = dict(frame=iid, **info)
        panels.append(draw(iid, dets, gts, a.thr, f"(d) frame {iid:06d}: KITTI 'Truck' ({info['h']:.0f} px high) detected only as '{info['pred']}' "
                           f"({info['score']:.2f}, IoU {info['iou']:.2f}) -> wrong class under driving5", font))
    if not panels:
        raise SystemExit("no examples found")
    W = max(p.width for p in panels); H = sum(p.height for p in panels) + 4 * (len(panels) - 1)
    sheet = Image.new("RGB", (W, H), (255, 255, 255)); y = 0
    for pnl in panels:
        sheet.paste(pnl, (0, y)); y += pnl.height + 4
    # legend
    leg = Image.new("RGB", (W, 20), (255, 255, 255)); d = ImageDraw.Draw(leg); x = 6
    for name, col in [("GT (KITTI)", GT_COLOR), ("GT ignore (DontCare/Misc)", GT_COLOR)] + [(k, v) for k, v in COLORS.items() if k in ("person", "bicycle", "car", "truck", "bus")]:
        d.rectangle([x, 4, x + 12, 16], outline=col, width=2); d.text((x + 16, 3), name, fill=(0, 0, 0), font=font); x += 16 + 7 * len(name) + 18
    final = Image.new("RGB", (W, H + 20), (255, 255, 255)); final.paste(sheet, (0, 0)); final.paste(leg, (0, H))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    final.save(a.out)
    print(json.dumps(picks, indent=1))
    print("wrote", a.out, final.size)


if __name__ == "__main__":
    main()
