"""Aggregate results/metrics/*.json and results/latency.json into tables + figures.

Writes results/summary.md and results/figures/*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results/metrics"
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["fasterrcnn_resnet50_fpn_v2", "retinanet_resnet50_fpn_v2", "fcos_resnet50_fpn",
         "yolo11n", "yolo11n_1280", "yolo11s", "yolo11m", "yolo11m_1280", "rtdetr-l", "rtdetr-l_1280",
         "yolov8s-worldv2", "yolov8s-worldv2_kittiprompts", "yolov8s-worldv2_kittiprompts2"]
PRETTY = {"fasterrcnn_resnet50_fpn_v2": "Faster R-CNN R50-FPN v2", "retinanet_resnet50_fpn_v2": "RetinaNet R50-FPN v2",
          "fcos_resnet50_fpn": "FCOS R50-FPN", "yolo11n": "YOLO11n", "yolo11s": "YOLO11s", "yolo11m": "YOLO11m",
          "yolo11n_1280": "YOLO11n @1280", "yolo11m_1280": "YOLO11m @1280", "rtdetr-l_1280": "RT-DETR-L @1280",
          "rtdetr-l": "RT-DETR-L", "yolov8s-worldv2": "YOLO-World v2-S (COCO vocab)",
          "yolov8s-worldv2_kittiprompts": "YOLO-World v2-S (KITTI prompts: pedestrian)",
          "yolov8s-worldv2_kittiprompts2": "YOLO-World v2-S (KITTI prompts: person)"}


def load():
    runs = {}
    for p in sorted(MET.glob("*.json")):
        tag, ds = p.stem.split("__")
        runs.setdefault(tag, {})[ds] = json.load(open(p))
    return runs


def f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def key(tag):
    base = tag.split("_fp16")[0]
    return (ORDER.index(base) if base in ORDER else 99, tag)


def main():
    runs = load()
    lat = json.load(open(ROOT / "results/latency.json")) if (ROOT / "results/latency.json").exists() else {}
    tags = sorted(runs, key=key)
    L = ["# Results summary\n"]

    # ---------------- COCO table ----------------
    L.append("## COCO val2017 (80 classes, 5,000 images)\n")
    L.append("| Model | mAP@[.5:.95] | AP50 | AP75 | AP_S | AP_M | AP_L | AR_100 | median ms/img | FPS | peak GPU MB |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tags:
        r = runs[t].get("coco")
        if not r: continue
        m, tm = r["metrics"]["coco80"], r["timing"]
        L.append(f"| {PRETTY.get(t, t)} | {f(m['mAP'])} | {f(m['AP50'])} | {f(m['AP75'])} | {f(m['AP_small'])} | {f(m['AP_medium'])} | "
                 f"{f(m['AP_large'])} | {f(m['AR_100'])} | {f(tm['latency_ms_median'],1)} | {f(tm['fps_from_median'],1)} | {f(tm['peak_gpu_mem_MB'],0)} |")

    # ---------------- KITTI tables ----------------
    for tax in ("driving3", "driving5"):
        L.append(f"\n## KITTI (zero-shot, taxonomy `{tax}`)\n")
        classes = None
        for t in tags:
            r = runs[t].get("kitti")
            if r and tax in r["metrics"] and "per_class" in r["metrics"][tax]:
                classes = list(r["metrics"][tax]["per_class"]); break
        if not classes: continue
        hdr = "| Model | mAP@[.5:.95] | AP50 | AP75 | AP_S | AP_M | AP_L | AR_100 | " + " | ".join(f"AP {c}" for c in classes) + " | " + " | ".join(f"AP50 {c}" for c in classes) + " | median ms/img | FPS |"
        L.append(hdr); L.append("|" + "---|" * (hdr.count("|") - 1))
        for t in tags:
            r = runs[t].get("kitti")
            if not r or tax not in r["metrics"] or "per_class" not in r["metrics"][tax]: continue
            m, tm = r["metrics"][tax], r["timing"]
            pc = m["per_class"]
            L.append(f"| {PRETTY.get(t, t)} | {f(m['mAP'])} | {f(m['AP50'])} | {f(m['AP75'])} | {f(m['AP_small'])} | {f(m['AP_medium'])} | {f(m['AP_large'])} | {f(m['AR_100'])} | "
                     + " | ".join(f(pc[c]["AP"]) for c in classes) + " | " + " | ".join(f(pc[c]["AP50"]) for c in classes)
                     + f" | {f(tm['latency_ms_median'],1)} | {f(tm['fps_from_median'],1)} |")
        L.append("\nGT counts: " + ", ".join(f"{c}={pc[c]['n_gt']}" for c in classes))

    # ---------------- COCO -> KITTI degradation on the shared classes --------
    L.append("\n## Domain shift: COCO per-class AP on the driving classes vs KITTI\n")
    L.append("| Model | COCO car | COCO truck | COCO bus | COCO person | COCO bicycle | KITTI vehicle | KITTI person | KITTI cyclist |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for t in tags:
        c, k = runs[t].get("coco"), runs[t].get("kitti")
        if not c or not k or "driving3" not in k["metrics"]: continue
        pc, pk = c["metrics"]["coco80"]["per_class"], k["metrics"]["driving3"]["per_class"]
        L.append(f"| {PRETTY.get(t, t)} | " + " | ".join(f(pc[n]["AP"]) for n in ("car", "truck", "bus", "person", "bicycle"))
                 + " | " + " | ".join(f(pk[n]["AP"]) for n in ("vehicle", "person", "cyclist")) + " |")

    # ---------------- latency table ----------------
    if lat:
        L.append("\n## Controlled latency benchmark (batch 1, 100 timed iterations after 20 warm-up)\n")
        L.append("| Model | precision | params (M) | KITTI 1242x375 median ms | FPS | COCO 640x426 median ms | FPS | peak alloc MB | peak reserved MB |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for k in sorted(lat, key=key):
            r = lat[k]; base = k.rsplit("_fp", 1)[0]
            a, b = r["kitti_1242x375"], r["coco_640x426"]
            L.append(f"| {PRETTY.get(base, base)} | {r['model']['precision']} | {f(r['n_params_M'],1)} | {f(a['median_ms'],1)} | {f(a['fps'],1)} | "
                     f"{f(b['median_ms'],1)} | {f(b['fps'],1)} | {f(a['peak_mem_alloc_MB'],0)} | {f(a['peak_mem_reserved_MB'],0)} |")

    # ---------------- reproducibility block ----------------
    L.append("\n## Run configuration per experiment\n")
    L.append("| Model | dataset | checkpoint | input resolution | score thr | NMS IoU | max det | precision | framework | command |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in tags:
        for ds, r in runs[t].items():
            mi = r["model"]
            L.append(f"| {PRETTY.get(t, t)} | {ds} ({r['n_images']} imgs) | {mi['checkpoint']} | {mi['input_resolution']} | {mi['score_thr']} | "
                     f"{mi.get('nms_thr')} | {mi.get('max_det')} | {mi.get('precision')} | {mi['framework']} | `{r['command']}` |")
    hw = next(iter(next(iter(runs.values())).values()))["hardware"]
    L.append(f"\nHardware: {hw['gpu']}, {hw['cpu']}, {hw['os']}, Python {hw['python']}, torch {hw['torch']}, CUDA {hw['cuda']}, cuDNN {hw['cudnn']}")

    (ROOT / "results/summary.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote results/summary.md")

    # ================= figures =================
    names = [t for t in tags if "coco" in runs[t] and "kitti" in runs[t] and "driving3" in runs[t]["kitti"]["metrics"]]
    if names:
        coco_map = [runs[t]["coco"]["metrics"]["coco80"]["mAP"] for t in names]
        kit_map = [runs[t]["kitti"]["metrics"]["driving3"]["mAP"] for t in names]
        x = np.arange(len(names)); w = 0.38
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x - w/2, coco_map, w, label="COCO val2017 mAP (80 cls)")
        ax.bar(x + w/2, kit_map, w, label="KITTI zero-shot mAP (driving3)")
        ax.set_xticks(x); ax.set_xticklabels([PRETTY.get(t, t) for t in names], rotation=20, ha="right")
        ax.set_ylabel("mAP@[.5:.95]"); ax.legend(); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(FIG / "map_coco_vs_kitti.png", dpi=150); plt.close(fig)

        # per-class KITTI
        cls = ["vehicle", "person", "cyclist"]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, c in enumerate(cls):
            ax.bar(x + (i - 1) * 0.27, [runs[t]["kitti"]["metrics"]["driving3"]["per_class"][c]["AP"] for t in names], 0.27, label=c)
        ax.set_xticks(x); ax.set_xticklabels([PRETTY.get(t, t) for t in names], rotation=20, ha="right")
        ax.set_ylabel("KITTI AP@[.5:.95]"); ax.legend(); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(FIG / "kitti_per_class.png", dpi=150); plt.close(fig)

        # scale breakdown COCO vs KITTI
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, ds, tax, title in ((axes[0], "coco", "coco80", "COCO val2017"), (axes[1], "kitti", "driving3", "KITTI (driving3)")):
            for i, s in enumerate(("AP_small", "AP_medium", "AP_large")):
                ax.bar(x + (i - 1) * 0.27, [runs[t][ds]["metrics"][tax][s] for t in names], 0.27, label=s)
            ax.set_xticks(x); ax.set_xticklabels([PRETTY.get(t, t) for t in names], rotation=25, ha="right")
            ax.set_title(title); ax.grid(axis="y", alpha=.3)
        axes[0].set_ylabel("AP@[.5:.95]"); axes[0].legend()
        fig.tight_layout(); fig.savefig(FIG / "scale_breakdown.png", dpi=150); plt.close(fig)

    # speed / accuracy
    if lat:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bases = sorted({k.rsplit("_fp", 1)[0] for k in lat}, key=key)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for ci, base in enumerate(bases):
            if base not in runs or "coco" not in runs[base]: continue
            mp = runs[base]["coco"]["metrics"]["coco80"]["mAP"]
            pts = {}
            for prec in ("fp32", "fp16"):
                r = lat.get(f"{base}_{prec}")
                if not r: continue
                ms = r["kitti_1242x375"]["median_ms"]; pts[prec] = ms
                ax.scatter(ms, mp, s=60 + r["kitti_1242x375"]["peak_mem_alloc_MB"] / 8, marker="o" if prec == "fp32" else "^",
                           color=colors[ci % 10], alpha=.85, edgecolor="k", linewidth=.5)
            if "fp32" in pts and "fp16" in pts:
                ax.plot([pts["fp32"], pts["fp16"]], [mp, mp], color=colors[ci % 10], lw=1, alpha=.6)
            ax.annotate(PRETTY.get(base, base), (max(pts.values()), mp), fontsize=8, xytext=(6, -3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("median latency per KITTI frame, ms (log scale)"); ax.set_ylabel("COCO val2017 mAP@[.5:.95]")
        ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
        ax.set_title("Speed vs accuracy on RTX 5060 (circle = fp32, triangle = fp16, size ~ peak GPU memory)"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(FIG / "speed_accuracy.png", dpi=150); plt.close(fig)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
