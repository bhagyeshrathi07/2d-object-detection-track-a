"""Fully decode every PNG in a folder and delete any that are truncated/corrupt
(so the resumable fetcher can re-download them)."""
import sys
from pathlib import Path

from PIL import Image

folder = Path(sys.argv[1])
bad = []
for p in sorted(folder.glob("*.png")):
    try:
        with Image.open(p) as im:
            im.load()
    except Exception as e:
        bad.append(p); print("corrupt:", p, e)
for p in bad:
    p.unlink()
print(f"checked {len(list(folder.glob('*.png'))) + len(bad)} files, removed {len(bad)}")
