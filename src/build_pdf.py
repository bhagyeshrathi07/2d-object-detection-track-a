"""Build REPORT.pdf from REPORT.md.

Pipeline: Markdown -> styled HTML (markdown-it-py) -> PDF (headless Chrome/Edge).
Figures are embedded as base64 so the intermediate HTML is self-contained.

Usage:
    ~/venvs/assignment1/bin/python src/build_pdf.py [--md REPORT.md] [--out REPORT.pdf]   (WSL / Linux)
    .\\.venv\\Scripts\\python.exe src\\build_pdf.py [--md REPORT.md] [--out REPORT.pdf]      (Windows)
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent

BROWSERS = [
    # Linux-native browsers (WSL / Ubuntu)
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/microsoft-edge",
    # Windows browsers, reachable from WSL through the /mnt/c drive mount
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
    # Native Windows
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _is_windows_exe_under_wsl(browser: str) -> bool:
    return browser.startswith("/mnt/") and browser.lower().endswith(".exe")


def _wslpath_w(p: Path) -> str:
    """Convert a Linux path to the Windows path a Windows .exe can open (needs `wslpath`)."""
    return subprocess.run(["wslpath", "-w", str(p)], check=True, capture_output=True, text=True).stdout.strip()

# file name -> (short title for the contents list, full caption)
FIGURE_CAPTIONS = {
    "map_coco_vs_kitti.png": (
        "COCO mAP versus zero-shot KITTI mAP per model",
        "In-domain COCO val2017 mAP versus zero-shot KITTI mAP (driving3 taxonomy) for each model. "
        "The 16-point spread on COCO compresses to 4 points on KITTI.",
    ),
    "kitti_per_class.png": (
        "KITTI per-class AP (driving3)",
        "KITTI per-class AP@[.5:.95] under the driving3 taxonomy. Vehicles transfer well, "
        "persons lose most, cyclists collapse to ≈0 for every closed-set model.",
    ),
    "scale_breakdown.png": (
        "AP by object scale on COCO and KITTI",
        "AP by object scale (COCO area thresholds) on COCO and KITTI. Small-object AP is the weakest "
        "bucket everywhere and is dominated by input resolution on KITTI.",
    ),
    "speed_accuracy.png": (
        "Speed–accuracy trade-off on the RTX 5060",
        "Speed–accuracy trade-off on the RTX 5060 at batch size 1 (circle = fp32, triangle = fp16, "
        "marker size ∝ peak GPU memory). Latency axis is logarithmic.",
    ),
}

NUMERIC_CELL = re.compile(r"[\d.,/%→\-–+ ()≥≈*×]*\d[\d.,/%→\-–+ ()≥≈*×]*( ?(M|ms|FPS|MB|px))?")

CSS = r"""
@page {
  size: A4;
  margin: 18mm 17mm 18mm 17mm;
  @bottom-center {
    content: counter(page);
    font-family: "Times New Roman", Times, serif;
    font-size: 9pt;
    color: #000;
  }
}

* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  color: #000;
  font-family: "Times New Roman", Times, serif;
  font-size: 9.6pt;
  line-height: 1.22;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
p { margin: 0 0 5pt 0; text-align: justify; }
a { color: #0563C1; text-decoration: underline; word-break: break-all; }
strong { color: #000; }
code {
  font-family: "Courier New", Courier, monospace;
  font-size: 8.4pt;
}
pre {
  font-family: "Courier New", Courier, monospace;
  border: 1px solid #000;
  padding: 4pt 6pt;
  margin: 3pt 0 6pt 0;
  font-size: 7.2pt;
  line-height: 1.2;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  break-inside: avoid;
}
pre code { font-size: inherit; }
hr { display: none; }
ul, ol { margin: 0 0 5pt 0; padding-left: 18pt; }
li { margin-bottom: 1.5pt; text-align: justify; }
li > p { margin-bottom: 1.5pt; }
li ol { margin-top: 1.5pt; }
ol.refs { font-size: 8pt; columns: 2; column-gap: 12pt; padding-left: 14pt; line-height: 1.18; }
ol.refs li { break-inside: avoid; margin-bottom: 1.5pt; text-align: left; }

/* ---------- headings ---------- */
h1, h2, h3, h4 { font-family: "Times New Roman", Times, serif; color: #000; line-height: 1.2; break-after: avoid; }
h2 { font-size: 13pt; font-weight: bold; margin: 12pt 0 5pt 0; }
h2.newpage { margin-top: 12pt; }
h2.appendix { margin-top: 12pt; }
h3 { font-size: 10.8pt; font-weight: bold; margin: 8pt 0 3pt 0; }
h4 { font-size: 10pt; font-weight: bold; margin: 6pt 0 3pt 0; }

/* ---------- figures ---------- */
.figgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 3pt 8pt; margin: 3pt 0 6pt 0; }
.figgrid figure { margin: 0; }
.figgrid figcaption { font-size: 8pt; text-align: justify; }
figure { margin: 5pt 0 7pt 0; break-inside: avoid; }
figure img { width: 100%; height: auto; display: block; }
figure.single { break-inside: avoid; margin: 4pt 0 6pt 0; }
figure.single img { width: auto; max-width: 100%; max-height: 86mm; margin: 0 auto; }
figcaption { font-size: 8.4pt; font-style: italic; margin-top: 2pt; text-align: justify; }
figcaption b { font-style: normal; }

/* ---------- title block ---------- */
.cover { margin-bottom: 8pt; text-align: center; }
.cover .kicker-row { font-size: 9.6pt; }
.cover .kicker { display: inline; }
.cover .date { display: inline; }
.cover .kicker::after { content: " · "; }
.cover h1 { font-size: 16pt; font-weight: bold; margin: 6pt 0 2pt 0; }
.cover .subtitle { font-size: 12pt; margin: 0 0 4pt 0; }
.cover .author { font-size: 10.5pt; margin: 0 0 8pt 0; }
.cover .bar { display: none; }
.cover .abstract { text-align: justify; margin: 0 8mm; }
.cover .abstract p { margin-bottom: 4pt; }
.meta { margin: 4pt 8mm 0 8mm; font-size: 9pt; text-align: left; }
.meta .item { display: inline; }
.meta .item::after { content: "  "; }
.meta .label { display: inline; font-weight: bold; }
.meta .label::after { content: ": "; }
.meta .value { display: inline; }
.meta .value::after { content: ". "; }
.cover .note { margin: 5pt 8mm 0 8mm; font-size: 9pt; text-align: justify; }
.cover .note p { margin: 0; }

/* ---------- tables ---------- */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 3pt 0 6pt 0;
  font-size: 8.2pt;
  line-height: 1.18;
  break-inside: auto;
}
table.mid { font-size: 8pt; }
table.wide { font-size: 7.4pt; }
table.wide td, table.wide th { padding: 1.5pt 2pt; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { padding: 2pt 3.5pt; border: 1px solid #000; vertical-align: top; }
th { background: #d9d9d9; font-weight: bold; text-align: left; }
table.data td:not(:first-child), table.data th:not(:first-child) { text-align: center; }
table.data td:not(:first-child) { white-space: nowrap; }
td:first-child { white-space: normal; }
table.wide td:first-child { min-width: 28mm; }
td strong { font-weight: bold; }
td code, th code { font-size: 0.95em; }

/* ---------- misc ---------- */
.footnote { font-size: 8.4pt; }
.gt-note { font-size: 8.6pt; font-style: italic; }
h3 + p { margin-top: 1pt; }
"""


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "section"


def embed_image(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def replace_figure_table(md_text: str) -> tuple[str, list[str]]:
    """Replace the 2x2 image table in the figures section with <figure> blocks."""
    pattern = re.compile(r"\|\s*\|\s*\|\n\|---\|---\|\n((?:\|.*!\[\]\(.*\).*\|\n?)+)")
    m = pattern.search(md_text)
    captions: list[str] = []
    if not m:
        return md_text, captions
    paths = re.findall(r"!\[\]\(([^)]+)\)", m.group(1))
    html = []
    for i, rel in enumerate(paths, 1):
        p = ROOT / rel
        short, cap = FIGURE_CAPTIONS.get(p.name, (p.name, p.name))
        captions.append(f"Figure {i}. {short}")
        html.append(
            f'<figure id="fig-{i}"><img src="{embed_image(p)}" alt="{p.name}">'
            f"<figcaption><b>Figure {i}.</b> {cap}</figcaption></figure>"
        )
    grid = '<div class="figgrid">' + "\n".join(html) + "</div>"
    return md_text[: m.start()] + grid + "\n" + md_text[m.end():], captions


def embed_remaining_images(html: str) -> str:
    """Inline every <img src="relative/path"> that is not already a data URI (e.g. Figure 5)."""
    def repl(m: re.Match) -> str:
        rel = m.group(1)
        p = ROOT / rel
        if not p.exists():
            return m.group(0)
        return f'<img src="{embed_image(p)}"'
    return re.sub(r'<img src="(?!data:)([^"]+)"', repl, html)


def split_front_matter(md_text: str) -> tuple[str, str, list[str], str, str]:
    """Return (title, author line, abstract paragraphs, starting-point paragraph, body markdown)."""
    lines = md_text.splitlines()
    assert lines[0].startswith("# ")
    title = lines[0][2:].strip()
    body_start = next(i for i, l in enumerate(lines) if l.startswith("## "))
    front = "\n".join(lines[1:body_start]).strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", front) if p.strip() and p.strip() != "---"]
    author = next((p[len("Author:"):].strip() for p in paras if p.startswith("Author:")), "")
    abstract = [p for p in paras if not p.startswith(("Starting point", "Author:"))]
    starting = next((p for p in paras if p.startswith("Starting point")), "")
    return title, author, abstract, starting, "\n".join(lines[body_start:])


def render_markdown(md: MarkdownIt, text: str) -> tuple[str, list[tuple[int, str, str]]]:
    tokens = md.parse(text)
    headings: list[tuple[int, str, str]] = []
    first_h2 = True
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == "heading_open" and t.tag in ("h2", "h3"):
            inline = tokens[i + 1]
            txt = md.renderer.renderInline(inline.children or [], md.options, {})
            plain = re.sub(r"<[^>]+>", "", txt)
            sid = slugify(plain)
            t.attrSet("id", sid)
            if t.tag == "h2":
                if not first_h2:
                    t.attrJoin("class", "newpage")
                if plain.lower().startswith("appendix"):
                    t.attrJoin("class", "appendix")
                first_h2 = False
            headings.append((2 if t.tag == "h2" else 3, sid, txt))
        if t.type == "table_open":
            # count header cells, and decide whether this is a numeric "data" table
            n = 0
            j = i + 1
            while tokens[j].type != "thead_close":
                if tokens[j].type == "th_open":
                    n += 1
                j += 1
            body_cells = 0
            numeric = 0
            col = 0
            while tokens[j].type != "table_close":
                if tokens[j].type == "tr_open":
                    col = 0
                if tokens[j].type == "td_open":
                    col += 1
                    if col > 1:
                        body_cells += 1
                        txt = re.sub(r"<[^>]+>", "", tokens[j + 1].content).strip()
                        if NUMERIC_CELL.fullmatch(txt):
                            numeric += 1
                j += 1
            if n >= 10:
                t.attrJoin("class", "wide")
            elif n >= 7:
                t.attrJoin("class", "mid")
            if body_cells and numeric / body_cells >= 0.5:
                t.attrJoin("class", "data")
        i += 1
    return md.renderer.render(tokens, md.options, {}), headings


def build_html(md_path: Path) -> str:
    md = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable("table")
    text = md_path.read_text(encoding="utf-8")
    text, captions = replace_figure_table(text)
    title, author, abstract, starting, body = split_front_matter(text)

    body_html, headings = render_markdown(md, body)
    body_html = embed_remaining_images(body_html)
    # a stand-alone image paragraph followed by an italic caption paragraph -> <figure>
    body_html = re.sub(
        r'<p>(<img [^>]+>)</p>\s*<p><em>(.*?)</em></p>',
        r'<figure class="single">\1<figcaption>\2</figcaption></figure>',
        body_html, flags=re.S)
    # small tidy-ups: footnote + GT-count lines
    body_html = body_html.replace("<p>* torchvision / Ultralytics model cards.", '<p class="footnote">* torchvision / Ultralytics model cards.')
    body_html = re.sub(r"<p>(GT instances[^<]*)</p>", r'<p class="gt-note">\1</p>', body_html)
    # two-column, small-print reference list
    body_html = re.sub(r'(<h2 id="[^"]*references"[^>]*>.*?</h2>\s*)<ol>', r'\1<ol class="refs">', body_html, flags=re.S)

    abstract_html = "".join(f"<p>{md.renderInline(p)}</p>" for p in abstract)
    starting_html = f"<p>{md.renderInline(starting)}</p>" if starting else ""

    main_title, _, sub = title.partition(" — ")
    toc_items = "".join(
        f'<li class="l{lvl}"><a href="#{sid}">{txt}</a></li>' for lvl, sid, txt in headings
    )
    fig_items = "".join(
        f'<li class="l3"><a href="#fig-{i}">{c}</a></li>' for i, c in enumerate(captions, 1)
    )
    today = dt.datetime.fromtimestamp(md_path.stat().st_mtime).strftime("%B %Y")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head><body>
<section class="cover">
  <div class="kicker-row"><span class="kicker">Assignment 1 — Technical report</span><span class="date">{today}</span></div>
  <h1>{main_title}</h1>
  <div class="subtitle">{sub}</div>
  <div class="author">{md.renderInline(author)}</div>
  <div class="bar"></div>
  <div class="abstract">{abstract_html}</div>
  <p class="meta"><b>Models:</b> 7 COCO-pretrained configurations, 5 detector families.
  <b>Datasets:</b> COCO val2017 (5,000 images), KITTI val split (3,769 frames).
  <b>Hardware:</b> NVIDIA GeForce RTX 5060 8 GB, PyTorch 2.11 (CUDA 12.8).
  <b>Training:</b> none, pure zero-shot / domain-transfer evaluation.</p>
  <div class="note">{starting_html}</div>
</section>
{body_html}
</body></html>"""


def find_browser() -> str:
    for p in BROWSERS:
        if Path(p).exists():
            return p
    for name in ("chrome", "msedge"):
        if shutil.which(name):
            return shutil.which(name)  # type: ignore[return-value]
    sys.exit("No Chrome/Edge found for PDF printing.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(ROOT / "REPORT.md"))
    ap.add_argument("--out", default=str(ROOT / "REPORT.pdf"))
    ap.add_argument("--html", default=None, help="also keep the intermediate HTML here")
    args = ap.parse_args()

    html = build_html(Path(args.md))
    browser = find_browser()
    out = Path(args.out).resolve()

    # A Windows browser launched from WSL cannot read Linux paths, so when we are in that
    # situation every file the browser touches lives in a Windows temp dir on /mnt/c and the
    # finished PDF is copied back to the requested Linux location afterwards.
    via_windows = _is_windows_exe_under_wsl(browser)
    if via_windows:
        win_tmp_root = Path("/mnt/c/Users") / (subprocess.run(
            ["cmd.exe", "/c", "echo %USERNAME%"], capture_output=True, text=True, cwd="/mnt/c").stdout.strip()
        ) / "AppData" / "Local" / "Temp"
        tmpdir = Path(tempfile.mkdtemp(prefix="report_pdf_", dir=str(win_tmp_root)))
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="report_pdf_"))

    html_path = Path(args.html) if args.html else tmpdir / "REPORT.html"
    html_path.write_text(html, encoding="utf-8")
    if via_windows and args.html:
        # keep a copy where the browser can see it
        (tmpdir / "REPORT.html").write_text(html, encoding="utf-8")
        html_path = tmpdir / "REPORT.html"

    pdf_target = tmpdir / "REPORT.pdf" if via_windows else out
    if via_windows:
        profile_arg = _wslpath_w(tmpdir / "profile")
        pdf_arg = _wslpath_w(pdf_target)
        html_uri = "file:///" + _wslpath_w(html_path).replace("\\", "/")
    else:
        profile_arg = str(tmpdir / "profile")
        pdf_arg = str(pdf_target)
        html_uri = html_path.resolve().as_uri()

    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=10000",
        f"--user-data-dir={profile_arg}",
        f"--print-to-pdf={pdf_arg}",
        html_uri,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    if via_windows:
        shutil.copyfile(pdf_target, out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) using {Path(browser).name}")


if __name__ == "__main__":
    main()
