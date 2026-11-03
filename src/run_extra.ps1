# Runs after run_rest.ps1: higher-resolution KITTI runs + CPU re-scoring analyses, then regenerates the report.
param([string]$log)
$py = ".\.venv\Scripts\python.exe"
while (-not (Test-Path "$log\run_rest_done.txt")) { Start-Sleep 30 }
Write-Output "rest done $(Get-Date -Format s)"
& $py src\evaluate.py --model ul:yolo11m.pt --dataset kitti --imgsz 1280
& $py src\evaluate.py --model ul:rtdetr-l.pt --dataset kitti --imgsz 1280
& $py src\evaluate.py --model ul:yolo11n.pt --dataset kitti --imgsz 1280
& $py src\kitti_difficulty.py
& $py src\make_report.py
Write-Output "EXTRA DONE $(Get-Date -Format s)"
