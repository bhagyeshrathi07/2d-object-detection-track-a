# Waits for the KITTI fetch and the COCO chain, then runs everything remaining.
param([string]$log)
$py = ".\.venv\Scripts\python.exe"
while (-not (Test-Path "$log\fetch_kitti_done.txt")) { Start-Sleep 20 }
Write-Output "fetch done $(Get-Date -Format s)"
& $py src\verify_images.py data\kitti\training\image_2
python -I src\fetch_kitti_subset.py --split data\kitti\ImageSets\val.txt --out data\kitti\training\image_2 --workers 24
& $py src\kitti_to_coco.py --root data\kitti --out data\kitti\annotations --split data\kitti\ImageSets\val.txt
while (-not (Test-Path "$log\run_coco_done.txt")) { Start-Sleep 20 }
Write-Output "coco chain done $(Get-Date -Format s)"
.\src\run_kitti.ps1
& $py src\latency.py --models tv:fasterrcnn_resnet50_fpn_v2 tv:retinanet_resnet50_fpn_v2 tv:fcos_resnet50_fpn ul:yolo11n.pt ul:yolo11m.pt ul:rtdetr-l.pt ul:yolov8s-worldv2.pt
& $py src\make_report.py
Write-Output "REST DONE $(Get-Date -Format s)"
