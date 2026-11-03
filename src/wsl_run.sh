#!/usr/bin/env bash
# Helper: run a project script with the Linux venv from the Windows project folder.
#   wsl -d Ubuntu -e bash /mnt/c/Users/bhagy/Documents/IA-systems/Assignment_1/src/wsl_run.sh src/failure_examples.py --model rtdetr-l
set -euo pipefail
cd /mnt/c/Users/bhagy/Documents/IA-systems/Assignment_1
PY="${PY:-$HOME/venvs/assignment1/bin/python}"
echo "python: $($PY -c 'import sys,torch;print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())')"
exec "$PY" "$@"
