# Assignment 1 — 2D Object Detection, Track A (experimental evaluation)

Zero-shot evaluation of seven COCO-pretrained detectors (five detector families)
on COCO val2017 and the KITTI 2D detection val split, on an RTX 5060 (8 GB).

Repository: <https://github.com/bhagyeshrathi07/2d-object-detection-track-a>
(code, report, per-run metrics, figures and the unedited console logs in
`results/logs/`). Excluded from the repo: `data/`, model weights, and the
1.2 GB `results/raw/` detection cache, which exceeds GitHub's file-size limit.

* **[REPORT.md](REPORT.md)** — the write-up: setup, label mapping, results, analysis.
* `results/summary.md` — all result tables (auto-generated).
* `results/kitti_extra.md` — KITTI Easy/Moderate/Hard breakdown and rider-as-person analysis.
* `results/cyclist_fusion.md` — why cyclist AP is ~0 and a rider+bike fusion fix.
* `results/figures/*.png` — plots; `failure_examples.png` is the qualitative failure figure.
* `results/metrics/<model>__<dataset>.json` — one file per experiment with every
  metric, per-class AP, timing, hardware, versions and the exact command.
* `results/logs/*.log` — console logs of the job chains that produced the results
  (progress bars stripped): evidence of execution.
* `results/raw/` — cached raw detections (re-scorable without a GPU).
* `results/latency.json` — controlled latency / FPS / memory benchmark.

## Code (`src/`)

| File | Purpose |
|---|---|
| `taxonomy.py` | COCO class table and the `driving3` / `driving5` KITTI↔COCO mappings |
| `kitti_to_coco.py` | KITTI label files → COCO-format JSON (one per taxonomy, DontCare → ignore regions) |
| `fetch_kitti_subset.py` | downloads only the val-split frames from the 12.5 GB KITTI zip via HTTP range requests |
| `verify_images.py` | removes truncated PNGs so the fetcher can re-download them |
| `detectors.py` | unified wrappers: torchvision (Faster R-CNN, RetinaNet, FCOS), Ultralytics (YOLO11, RT-DETR, YOLO-World) |
| `evaluate.py` | runs a detector over a dataset, caches raw detections, scores with pycocotools |
| `latency.py` | batch-1 latency / FPS / memory benchmark, fp32 and fp16 |
| `kitti_difficulty.py` | KITTI difficulty levels + rider-ignore re-scoring of cached detections |
| `cyclist_fusion.py` | IoU diagnostic for the Cyclist class + post-hoc rider/bike fusion |
| `failure_examples.py` | draws GT vs predictions on automatically chosen KITTI frames (cyclist split, small pedestrians, crowd) |
| `make_report.py` | builds `results/summary.md` and the figures |
| `wsl_run.sh` | helper: run any `src/` script with the Linux venv from the Windows project folder |
| `run_coco.sh`, `run_kitti.sh`, `run_rest.sh`, `run_extra.sh` | the job chains, for Linux / WSL (the `.ps1` files are the original Windows versions that produced the results) |
| `build_pdf.py` | renders `REPORT.md` to `REPORT.pdf` through headless Chrome/Edge (works natively on Linux or from WSL via the Windows browser) |

## Reproduce (WSL 2 Ubuntu; project files stay on the Windows drive)

The repo lives under `/mnt/c/Users/<you>/Documents/...` and the Python environment lives on the
Linux filesystem (`~/venvs/assignment1`), so nothing is installed on the Windows side.

```bash
cd /mnt/c/Users/bhagy/Documents/IA-systems/Assignment_1
# one-time: uv (no sudo needed)  ->  curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 ~/venvs/assignment1
PY=~/venvs/assignment1/bin/python
uv pip install --python $PY torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $PY ultralytics pycocotools matplotlib pandas transformers timm ftfy regex markdown-it-py remotezip
uv pip install --python $PY https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
# COCO: unzip val2017.zip + annotations_trainval2017.zip into data/coco
# KITTI: unzip data_object_label_2.zip into data/kitti, put ImageSets/val.txt (OpenPCDet split) in data/kitti/ImageSets
$PY -I src/fetch_kitti_subset.py --split data/kitti/ImageSets/val.txt --out data/kitti/training/image_2 --workers 24
$PY src/kitti_to_coco.py --root data/kitti --out data/kitti/annotations --split data/kitti/ImageSets/val.txt
bash src/run_coco.sh
bash src/run_kitti.sh
$PY src/latency.py --models tv:fasterrcnn_resnet50_fpn_v2 tv:retinanet_resnet50_fpn_v2 tv:fcos_resnet50_fpn ul:yolo11n.pt ul:yolo11m.pt ul:rtdetr-l.pt ul:yolov8s-worldv2.pt
$PY src/kitti_difficulty.py
$PY src/cyclist_fusion.py
$PY src/failure_examples.py --model rtdetr-l --thr 0.3
$PY src/make_report.py
$PY src/build_pdf.py
```

The `run_*.sh` chains default to `~/venvs/assignment1/bin/python`; set `PY=...` to override.
The `.ps1` files are the original Windows/PowerShell chains that produced the submitted results.

`data/`, `downloads/`, `.venv/`, the `.pt` weights and `results/raw/` are not in the
repository (≈ 7 GB; see `.gitignore`). The weights are re-downloaded automatically
by torchvision / Ultralytics on first use.
