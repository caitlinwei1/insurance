"""Parse the Aetna PDFs into section-based chunks for RAG.

Output: data/chunks_pdf.jsonl, one JSON object per chunk (schema in README).

How it works
1. pdfplumber reads every line with its font, so we can find headings
   (bold lines at heading sizes, set per document in DOCS below).
2. pdftotext -layout gives the page text with table columns kept apart;
   runs of 3+ spaces become " | " so the LLM can still read the columns.
3. Text is cut at each heading; each piece keeps its full heading path.
4. Pieces that are too long are split at paragraph breaks; tiny ones are
   merged into the next piece under the same parent heading.

Usage:  python scripts/01_parse_pdfs.py
"""
import json
import re
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "raw" / "pdf"
OUT = ROOT / "data" / "chunks_pdf.jsonl"
PLAN_DOCS_URL = "https://www.aetnastudenthealth.com/en/plan-documents.html"

MAX_WORDS = 450   # split chunks longer than this
MIN_WORDS = 40    # merge chunks shorter than this into the next one

# heading_sizes: font size -> heading level (1 = top). Only all-bold lines count.
# mode "pages": no reliable headings, one chunk per page.
DOCS = {
    "AET-POL": dict(file="AET-POL_member_policy.pdf",
                    title="Member Policy Contract Documents 2026-2027",
                    heading_sizes={16.0: 1, 14.0: 2, 13.0: 3, 12.0: 4}),
    "AET-PDB": dict(file="AET-PDB_plan_design_benefits_summary.pdf",
                    title="Plan Design and Benefits Summary 2026-2027",
                    heading_sizes={16.0: 1, 11.0: 1, 12.0: 3, 10.0: 2}),
    "AET-SBC": dict(file="AET-SBC_summary_of_benefits_coverage.pdf",
                    title="Summary of Benefits and Coverage 2026-2027",
                    mode="pages"),
    "AET-DEN": dict(file="AET-DEN_dental_summary.pdf",
                    title="Dental Plan Design and Benefits Summary 2026-2027 (optional plan)",
                    heading_sizes={16.0: 1, 14.0: 1, 12.0: 2, 10.0: 2}),
    "AET-VIS": dict(file="AET-VIS_vision_summary.pdf",
                    title="Summary of Vision Benefits 2026-2027 (optional plan)",
                    mode="pages"),
}

# Bold lines that are table headers or fragments, not section headings
NOT_HEADING = re.compile(
    r"(in-network coverage|out-of-network coverage|coverage start date|http|www\.|"
    r"^description in-network|^covered services\b|^\(?continued\)?$|^charge\)|^year$|^policy year$)",
    re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def is_foreign(line: str) -> bool:
    """Language-assistance notices repeat one sentence in ~15 scripts; keep English only."""
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and sum(ord(c) > 0x24F for c in letters) / len(letters) > 0.3


def find_headings(pdf, sizes):
    """Return {page_index: [(normalized_text, level), ...]} in reading order."""
    out = {}
    for i, page in enumerate(pdf.pages):
        found = []
        for line in page.extract_text_lines(return_chars=True):
            chars = [c for c in line["chars"] if c["text"].strip()]
            if not chars or not all("Bold" in c["fontname"] for c in chars):
                continue
            size = round(max(c["size"] for c in chars), 1)
            text = norm(line["text"])
            if size not in sizes or NOT_HEADING.search(text):
                continue
            if not text[0].isupper() and not text[0].isdigit():
                continue  # wrapped fragment of a longer bold line
            if len(text.split()) > 14 or text.endswith((".", ",", ";")):
                continue
            found.append((text, sizes[size]))
        out[i] = found
    return out


def layout_pages(path: Path):
    raw = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                         capture_output=True, text=True, check=True).stdout
    return raw.split("\f")


def boilerplate_lines(pages):
    """Lines (digits removed) that repeat on >30% of pages = headers/footers."""
    cnt = Counter()
    for p in pages:
        cnt.update({re.sub(r"\d+", "#", norm(l)) for l in p.splitlines() if norm(l)})
    limit = max(3, 0.3 * len(pages))
    return {k for k, v in cnt.items() if v >= limit}


def clean_line(line: str) -> str:
    line = line.rstrip()
    indent = len(line) - len(line.lstrip())
    body = re.sub(r" {3,}", " | ", line.strip())
    return ("  " if indent >= 6 else "") + body


def segments(code, cfg):
    """Yield raw sections: dict(path=[...], lines=[...], pages={...})."""
    path = PDF_DIR / cfg["file"]
    pages = layout_pages(path)
    junk = boilerplate_lines(pages)
    if cfg.get("mode") == "pages":
        for i, p in enumerate(pages):
            lines = [clean_line(l) for l in p.splitlines()
                     if norm(l) and re.sub(r"\d+", "#", norm(l)) not in junk and not is_foreign(l)]
            if lines:
                yield dict(path=[f"Page {i + 1}"], lines=lines, pages={i + 1})
        return

    with pdfplumber.open(path) as pdf:
        heads = find_headings(pdf, cfg["heading_sizes"])

    stack = []  # [(level, text)]
    cur = dict(path=[], lines=[], pages=set())
    for i, p in enumerate(pages):
        todo = list(heads.get(i, []))
        for raw in p.splitlines():
            t = norm(raw)
            if not t or re.sub(r"\d+", "#", t) in junk or is_foreign(t):
                continue
            hit = next((h for h in todo if h[0] == t), None)
            if hit:
                todo.remove(hit)
                text, level = hit
                if cur["lines"]:
                    yield cur
                stack = [s for s in stack if s[0] < level] + [(level, text)]
                cur = dict(path=[s[1] for s in stack], lines=[], pages={i + 1})
                continue
            cur["lines"].append(clean_line(raw))
            cur["pages"].add(i + 1)
    if cur["lines"]:
        yield cur


def split_long(seg):
    """Split at blank-ish paragraph boundaries (bullets / short lines) to <= MAX_WORDS."""
    text = "\n".join(seg["lines"])
    if len(text.split()) <= MAX_WORDS:
        return [text]
    parts, buf = [], []
    for line in seg["lines"]:
        buf.append(line)
        words = len(" ".join(buf).split())
        boundary = line.lstrip().startswith(("•", "-")) or len(line.strip()) < 60
        if words >= MAX_WORDS * 0.8 and boundary or words >= MAX_WORDS * 1.3:
            parts.append("\n".join(buf))
            buf = []
    if buf:
        if parts and len(" ".join(buf).split()) < MIN_WORDS:
            parts[-1] += "\n" + "\n".join(buf)
        else:
            parts.append("\n".join(buf))
    return parts


def build(code, cfg):
    segs = list(segments(code, cfg))
    # merge tiny sections into the next one when they share a parent heading
    merged = []
    for s in segs:
        if merged:
            prev = merged[-1]
            small = len(" ".join(prev["lines"]).split()) < MIN_WORDS
            same_parent = prev["path"][:-1] == s["path"][:-1] or not prev["path"]
            if small and same_parent and cfg.get("mode") != "pages":
                label = prev["path"][-1] if prev["path"] else ""
                s = dict(path=s["path"],
                         lines=([f"[{label}]"] if label else []) + prev["lines"] + [f"[{s['path'][-1]}]"] + s["lines"],
                         pages=prev["pages"] | s["pages"])
                merged[-1] = s
                continue
        merged.append(s)

    chunks = []
    for s in merged:
        for part in split_long(s):
            chunks.append(dict(
                source_code=code,
                source_title=cfg["title"],
                source_type="official",
                url=PLAN_DOCS_URL,
                heading_path=" > ".join([cfg["title"]] + s["path"]),
                text=part.strip(),
                page_start=min(s["pages"]),
                page_end=max(s["pages"]),
                plan_year="2026-2027",
                date_captured=date.today().isoformat(),
            ))
    for n, c in enumerate(chunks, 1):
        c["id"] = f"{code}-{n:03d}"
    return chunks


def main():
    OUT.parent.mkdir(exist_ok=True)
    total = 0
    with OUT.open("w", encoding="utf-8") as f:
        for code, cfg in DOCS.items():
            chunks = build(code, cfg)
            total += len(chunks)
            words = [len(c["text"].split()) for c in chunks]
            print(f"{code}: {len(chunks)} chunks, words min/median/max = "
                  f"{min(words)}/{sorted(words)[len(words) // 2]}/{max(words)}")
            for c in chunks:
                f.write(json.dumps({"id": c.pop("id"), **c}, ensure_ascii=False) + "\n")
    print(f"total {total} chunks -> {OUT}")


if __name__ == "__main__":
    main()
