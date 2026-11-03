#!/usr/bin/env bash
# Waits for the KITTI fetch and the COCO chain, then runs everything remaining.
# Usage (from the project root inside WSL Ubuntu):  bash src/run_rest.sh <log-dir>
# The log dir is where the other chains drop their *_done.txt marker files.
set -u
cd "$(dirname "$0")/.." || exit 1
log=${1:?usage: run_rest.sh <log-dir>}
py=${PY:-$HOME/venvs/assignment1/bin/python}
while [ ! -f "$log/fetch_kitti_done.txt" ]; do sleep 20; done
echo "fetch done $(date -Is)"
"$py" src/verify_images.py data/kitti/training/image_2
"$py" -I src/fetch_kitti_subset.py --split data/kitti/ImageSets/val.txt --out data/kitti/training/image_2 --workers 24
"$py" src/kitti_to_coco.py --root data/kitti --out data/kitti/annotations --split data/kitti/ImageSets/val.txt
while [ ! -f "$log/run_coco_done.txt" ]; do sleep 20; done
echo "coco chain done $(date -Is)"
bash src/run_kitti.sh
"$py" src/latency.py --models tv:fasterrcnn_resnet50_fpn_v2 tv:retinanet_resnet50_fpn_v2 tv:fcos_resnet50_fpn ul:yolo11n.pt ul:yolo11m.pt ul:rtdetr-l.pt ul:yolov8s-worldv2.pt
"$py" src/make_report.py
echo "REST DONE $(date -Is)"
