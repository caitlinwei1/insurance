"""Read the team's test set (raw/test_set.docx) and produce two files:

data/eval_set.jsonl   one row per test question: question, expected answer,
                      expected source codes, status, phase/section, and whether
                      the bot should answer or hand off.
data/chunks_curated.jsonl
                      the same Q&A pairs as "curated" chunks. The chatbot can
                      search them alongside the official text, but the eval
                      always runs with them switched OFF (otherwise every test
                      question would retrieve its own answer).

Usage:  python scripts/03_build_testset.py
"""
import json
import re
from datetime import date
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "raw" / "test_set.docx"
EVAL_OUT = ROOT / "data" / "eval_set.jsonl"
CUR_OUT = ROOT / "data" / "chunks_curated.jsonl"


def blocks(document):
    for el in document.element.body.iterchildren():
        tag = el.tag.split("}")[1]
        if tag == "p":
            yield Paragraph(el, document)
        elif tag == "tbl":
            yield Table(el, document)


def source_codes(cell: str):
    """'WEL-027-06, AET-PDB' -> ['WEL-027-06', 'AET-PDB']; 'Bot design' / '—' -> []."""
    return re.findall(r"\b(?:WEL|AET|GOV|PMD|GLB)(?:-[A-Z0-9]+)*\b", cell)


# The test set uses a few section labels for parts of the Aetna PDFs. Map each one to
# the document(s) that actually contain those facts, so the eval can match them.
ALIASES = {
    "AET-RXP": ["AET-RX", "AET-POL", "AET-PDB"],  # pharmacy rules: drug guide + policy + summary
    "AET-PNN": ["AET-POL", "AET-PDB"],            # telemedicine, interpreters, provider network
    "AET-FRM": ["AET-POL", "AET-PDB"],            # claim forms and deadlines (policy claim procedures)
    "AET-TRV": ["AET-PDB"],                       # "Out of Country Claims" section
    "AET-SUR": ["AET-POL"],                       # surprise bill section
}


def families(code: str):
    return ALIASES.get(code, [family(code)])


def family(code: str) -> str:
    """Page-level code used to match retrieved chunks: WEL-027-06 -> WEL-027, AET-PDB -> AET-PDB."""
    parts = code.split("-")
    if parts[0] == "WEL" and len(parts) >= 2:
        return "-".join(parts[:2])
    if parts[0] in ("GOV", "PMD") and len(parts) >= 2:
        return "-".join(parts[:2])
    return "-".join(parts[:2]) if len(parts) >= 2 else code


def main():
    d = docx.Document(SRC)
    phase = section = ""
    rows = []
    for b in blocks(d):
        if isinstance(b, Paragraph):
            style = (b.style.name or "").lower()
            text = b.text.strip()
            if not text:
                continue
            if style.startswith("heading 2") or style == "heading 2":
                phase, section = text, ""
            elif style.startswith("heading 3"):
                section = text
            continue
        header = [c.text.strip() for c in b.rows[0].cells]
        if header[:1] != ["#"]:
            continue
        for r in b.rows[1:]:
            cells = [c.text.strip() for c in r.cells]
            rec = dict(zip(header, cells))
            num = int(rec["#"])
            ans_key = "Answer" if "Answer" in rec else "Correct reply"
            q_key = "Question" if "Question" in rec else "Example question"
            codes = source_codes(rec.get("Source", ""))
            rows.append(dict(
                qid=f"Q{num:03d}", phase=phase, section=section or phase,
                question=rec[q_key], expected_answer=rec[ans_key],
                expected_sources=codes,
                expected_families=sorted({f for c in codes for f in families(c)}),
                status=rec.get("Status", ""),
                kind="handoff" if "hand off" in phase.lower() else "answer",
                source_note=rec.get("Source", "")))

    with EVAL_OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with CUR_OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            if r["status"] != "Ready" or r["kind"] != "answer" or not r["expected_sources"]:
                continue  # only confirmed answers that point to a real source
            f.write(json.dumps(dict(
                id=f"CUR-{r['qid']}", source_code="CUR", family="CUR",
                source_title="Team-verified answer (test set)",
                source_type="curated", url=None,
                heading_path=f"Curated answers > {r['phase']} > {r['section']}",
                text=f"Question: {r['question']}\nAnswer: {r['expected_answer']}",
                cites=r["expected_sources"], page_start=None, page_end=None,
                plan_year="2026-2027", date_captured=date.today().isoformat(),
            ), ensure_ascii=False) + "\n")

    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    n_cur = sum(1 for _ in CUR_OUT.open())
    print(f"{len(rows)} test questions {by_status}; {n_cur} curated chunks")


if __name__ == "__main__":
    main()
