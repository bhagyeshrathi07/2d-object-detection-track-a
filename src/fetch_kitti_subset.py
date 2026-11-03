"""Fetch only selected KITTI training images out of the official 12.5 GB zip
using HTTP range requests (remotezip), so the whole archive need not be
downloaded. Files already present on disk are skipped, so the script can be
re-run after a network failure. Several workers fetch in parallel.

Usage:
    python -I src/fetch_kitti_subset.py --split data/kitti/ImageSets/val.txt --out data/kitti/training/image_2
"""
import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteZip

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
_local = threading.local()


def _zip():
    if getattr(_local, "z", None) is None:
        _local.z = RemoteZip(URL)
    return _local.z


def fetch_one(args):
    i, out = args
    dst = out / f"{i}.png"
    for attempt in range(20):
        try:
            data = _zip().read(f"training/image_2/{i}.png")
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dst)
            return i, True
        except Exception as e:
            print(f"  {i}: {e!r} (attempt {attempt})", file=sys.stderr, flush=True)
            _local.z = None
            time.sleep(5)
    return i, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.split) if l.strip()]
    if a.limit:
        ids = ids[: a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (out / f"{i}.png").exists()]
    print(f"{len(ids)} frames requested, {len(todo)} still to fetch, {a.workers} workers", flush=True)
    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, ok in ex.map(fetch_one, [(i, out) for i in todo]):
            done += 1
            if not ok:
                failed.append(i)
            if done % 100 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(todo)} done, {done/el:.2f} img/s, eta {(len(todo)-done)/(done/el)/60:.1f} min", flush=True)
    print(f"finished {done - len(failed)} files in {time.time()-t0:.0f} s; failed: {failed}")


if __name__ == "__main__":
    main()
