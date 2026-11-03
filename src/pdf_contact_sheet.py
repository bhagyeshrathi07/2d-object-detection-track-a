"""Render every page of a PDF into one contact-sheet PNG and print a per-page fill estimate.

Used to check REPORT.pdf layout (page count, half-empty pages, clipped tables)
without a PDF viewer.

Usage:
    python src/pdf_contact_sheet.py REPORT.pdf --out results/REPORT_pages.png --cols 6 --dpi 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # pymupdf
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=40)
    a = ap.parse_args()

    doc = fitz.open(a.pdf)
    pages = []
    print(f"{len(doc)} pages")
    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(dpi=a.dpi)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        # fill estimate: lowest non-white row / page height
        g = im.convert("L")
        w, h = g.size
        px = g.load()
        last = 0
        for y in range(h - 1, -1, -1):
            if any(px[x, y] < 235 for x in range(0, w, 3)):
                last = y
                break
        fill = last / h
        text = page.get_text("text")
        heads = [l.strip() for l in text.splitlines() if l.strip()][:2]
        print(f"page {i:2d}: fill {fill:4.0%}  chars {len(text):5d}  images {len(page.get_images()):d}  | {' / '.join(heads)[:90]}")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0))
        d.text((4, 4), f"p{i} {fill:.0%}", fill=(200, 0, 0))
        pages.append(im)

    cols = a.cols
    rows = (len(pages) + cols - 1) // cols
    pw, ph = pages[0].size
    sheet = Image.new("RGB", (cols * (pw + 6), rows * (ph + 6)), (120, 120, 120))
    for i, im in enumerate(pages):
        sheet.paste(im, ((i % cols) * (pw + 6) + 3, (i // cols) * (ph + 6) + 3))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print("wrote", a.out, sheet.size)


if __name__ == "__main__":
    main()
