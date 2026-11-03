#!/usr/bin/env bash
# Zero-shot KITTI (val split) evaluation for every model configuration, scored under driving3 and driving5.
# Run inside WSL Ubuntu:  bash src/run_kitti.sh     (venv lives on Linux; override with PY=/path/to/python)
set -u
cd "$(dirname "$0")/.." || exit 1
py=${PY:-$HOME/venvs/assignment1/bin/python}
models=(
  "tv:fasterrcnn_resnet50_fpn_v2"
  "tv:retinanet_resnet50_fpn_v2"
  "tv:fcos_resnet50_fpn"
  "ul:yolo11n.pt"
  "ul:yolo11m.pt"
  "ul:rtdetr-l.pt"
  "ul:yolov8s-worldv2.pt"
  "ulkitti:yolov8s-worldv2.pt"
)
for m in "${models[@]}"; do
  echo "===== $m  $(date -Is) ====="
  "$py" src/evaluate.py --model "$m" --dataset kitti
done
echo "ALL DONE $(date -Is)"
