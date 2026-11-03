# Zero-shot KITTI (val split) evaluation for every model configuration, scored under driving3 and driving5.
$py = ".\.venv\Scripts\python.exe"
$models = @(
  "tv:fasterrcnn_resnet50_fpn_v2",
  "tv:retinanet_resnet50_fpn_v2",
  "tv:fcos_resnet50_fpn",
  "ul:yolo11n.pt",
  "ul:yolo11m.pt",
  "ul:rtdetr-l.pt",
  "ul:yolov8s-worldv2.pt",
  "ulkitti:yolov8s-worldv2.pt"
)
foreach ($m in $models) {
  Write-Output "===== $m  $(Get-Date -Format s) ====="
  & $py src\evaluate.py --model $m --dataset kitti
}
Write-Output "ALL DONE $(Get-Date -Format s)"
