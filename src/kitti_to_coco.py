"""Convert KITTI 2D object-detection labels into COCO-format JSON files.

One JSON per taxonomy preset is produced (see taxonomy.py). The raw KITTI
classes are also kept in a "kitti_raw" JSON (9 classes) for reference.

Usage:
    python src/kitti_to_coco.py --root data/kitti --out data/kitti/annotations --split data/kitti/ImageSets/val.txt
"""
import argparse
import json
import os
from pathlib import Path

from PIL import Image

from taxonomy import IGNORE, KITTI_CLASSES, TAXONOMIES


def read_kitti_label(path: Path):
    objs = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            x1, y1, x2, y2 = map(float, p[4:8])
            objs.append({
                "type": p[0],
                "truncated": float(p[1]),
                "occluded": int(float(p[2])),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            })
    return objs


def build(root: Path, out_dir: Path, max_images: int | None = None, split: Path | None = None):
    img_dir = root / "training" / "image_2"
    lbl_dir = root / "training" / "label_2"
    if split:
        ids = sorted(l.strip() for l in open(split) if l.strip())
        print(f"using split file {split}")
    else:
        ids = sorted(p.stem for p in lbl_dir.glob("*.txt"))
    if max_images:
        ids = ids[:max_images]
    print(f"{len(ids)} KITTI training frames")

    images, per_frame = [], {}
    for n, sid in enumerate(ids):
        fn = f"{sid}.png"
        with Image.open(img_dir / fn) as im:
            w, h = im.size
        images.append({"id": int(sid), "file_name": fn, "width": w, "height": h})
        per_frame[int(sid)] = read_kitti_label(lbl_dir / f"{sid}.txt")
        if n % 1000 == 0:
            print(f"  read {n}/{len(ids)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- raw 9-class file -------------------------------------------------
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(KITTI_CLASSES)]
    anns, aid = [], 1
    for img_id, objs in per_frame.items():
        for o in objs:
            anns.append({
                "id": aid, "image_id": img_id,
                "category_id": KITTI_CLASSES.index(o["type"]) + 1,
                "bbox": o["bbox"], "area": o["bbox"][2] * o["bbox"][3],
                "iscrowd": 1 if o["type"] == "DontCare" else 0,
                "truncated": o["truncated"], "occluded": o["occluded"],
            })
            aid += 1
    _dump(out_dir / "kitti_raw.json", images, anns, cats)

    # ---- unified taxonomies ----------------------------------------------
    for tax_name, tax in TAXONOMIES.items():
        cats = [{"id": i + 1, "name": c} for i, c in enumerate(tax["classes"])]
        anns, aid, n_ign = [], 1, 0
        for img_id, objs in per_frame.items():
            for o in objs:
                target = tax["kitti"].get(o["type"], IGNORE)
                base = {"image_id": img_id, "bbox": o["bbox"],
                        "area": o["bbox"][2] * o["bbox"][3],
                        "truncated": o["truncated"], "occluded": o["occluded"]}
                if target == IGNORE:
                    # ignore region: one crowd box per unified class
                    for c in cats:
                        anns.append({**base, "id": aid, "category_id": c["id"], "iscrowd": 1})
                        aid += 1
                    n_ign += 1
                else:
                    anns.append({**base, "id": aid, "iscrowd": 0,
                                 "category_id": tax["classes"].index(target) + 1})
                    aid += 1
        _dump(out_dir / f"kitti_{tax_name}.json", images, anns, cats)
        print(f"  {tax_name}: {len(anns)} annotations ({n_ign} ignore regions x {len(cats)} classes)")


def _dump(path, images, anns, cats):
    with open(path, "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)
    print(f"wrote {path}: {len(images)} images, {len(anns)} annotations, {len(cats)} categories")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/kitti")
    ap.add_argument("--out", default="data/kitti/annotations")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--split", default="data/kitti/ImageSets/val.txt",
                    help="text file with frame ids to include ('' = all label files)")
    a = ap.parse_args()
    build(Path(a.root), Path(a.out), a.max_images, Path(a.split) if a.split else None)
