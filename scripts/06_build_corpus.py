"""Merge every chunk file into one corpus the chatbot indexes: data/corpus.jsonl.

Sources: Aetna plan PDFs, the drug guide's explanation pages, official web pages,
and the team-verified Q&A answers from the test set (source_type "curated").
Curated answers are part of the corpus; the eval can still leave them out by
filtering on source_type, so a test question can't simply retrieve its own answer.

Usage:  python scripts/06_build_corpus.py
"""
import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
FILES = ["chunks_pdf.jsonl", "chunks_rx.jsonl", "chunks_web.jsonl", "chunks_curated.jsonl"]

rows = []
for name in FILES:
    for line in (DATA / name).open():
        c = json.loads(line)
        c.setdefault("family", c["source_code"])
        c.setdefault("cites", [])
        rows.append(c)
ids = Counter(c["id"] for c in rows)
assert all(v == 1 for v in ids.values()), "duplicate chunk ids"
with (DATA / "corpus.jsonl").open("w", encoding="utf-8") as f:
    for c in rows:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(f"{len(rows)} chunks -> data/corpus.jsonl", dict(Counter(c["source_type"] for c in rows)))
