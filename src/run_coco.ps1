# Full COCO val2017 evaluation for every model configuration (sequential, GPU-bound).
$py = ".\.venv\Scripts\python.exe"
$models = @(
  "tv:fasterrcnn_resnet50_fpn_v2",
  "tv:retinanet_resnet50_fpn_v2",
  "tv:fcos_resnet50_fpn",
  "ul:yolo11n.pt",
  "ul:yolo11m.pt",
  "ul:rtdetr-l.pt",
  "ul:yolov8s-worldv2.pt"
)
foreach ($m in $models) {
  Write-Output "===== $m  $(Get-Date -Format s) ====="
  & $py src\evaluate.py --model $m --dataset coco
}
Write-Output "ALL DONE $(Get-Date -Format s)"
