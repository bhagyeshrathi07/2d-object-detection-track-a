#!/usr/bin/env bash
# Runs after run_rest.sh: higher-resolution KITTI runs + CPU re-scoring analyses, then regenerates the report.
# Usage (from the project root inside WSL Ubuntu):  bash src/run_extra.sh <log-dir>
set -u
cd "$(dirname "$0")/.." || exit 1
log=${1:?usage: run_extra.sh <log-dir>}
py=${PY:-$HOME/venvs/assignment1/bin/python}
while [ ! -f "$log/run_rest_done.txt" ]; do sleep 30; done
echo "rest done $(date -Is)"
"$py" src/evaluate.py --model ul:yolo11m.pt --dataset kitti --imgsz 1280
"$py" src/evaluate.py --model ul:rtdetr-l.pt --dataset kitti --imgsz 1280
"$py" src/evaluate.py --model ul:yolo11n.pt --dataset kitti --imgsz 1280
"$py" src/kitti_difficulty.py
"$py" src/make_report.py
echo "EXTRA DONE $(date -Is)"
