"""Parse the 2026 Aetna Pharmacy Drug Guide (formulary) for Aetna Student Health PA.

Two outputs:
  data/formulary.jsonl  one row per drug listing: name, generic name, generic/brand,
                        tier, coverage requirements (PA, QL, ST...), category, page.
                        The chatbot looks drugs up here by name (exact/fuzzy match),
                        not by vector search: one letter can be a different drug.
  data/chunks_rx.jsonl  the guide's explanation pages (definitions, how tiers work,
                        PA/ST/QL, exceptions, mail order) as normal text chunks,
                        plus one chunk that explains the tier and requirement codes.

How the drug table is read: pdfplumber gives every word with its x/y position.
Columns are split by x (name | tier | requirements). Lines inside one listing are
~7pt apart, listings are ~25pt apart, so a vertical gap > 12pt starts a new row.
Bold lines are category headers. Italic names are generics, CAPITALS are brands.

Usage:  python scripts/05_parse_formulary.py
"""
import json
import re
import subprocess
from datetime import date
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "raw" / "pdf" / "AET-RX_drug_guide_2026.pdf"
OUT_DRUGS = ROOT / "data" / "formulary.jsonl"
OUT_CHUNKS = ROOT / "data" / "chunks_rx.jsonl"
URL = "https://www.aetnastudenthealth.com/en/plan-documents.html"
TITLE = "2026 Pharmacy Drug Guide (Advanced Control Plan, Aetna Student Health PA)"

INFO_PAGES = range(4, 10)   # definitions, how to use, FAQs, exceptions
FIRST_DRUG_PAGE = 21
TIER_X, REQ_X = 360, 425    # column boundaries (points)
ROW_GAP = 12                # vertical gap that starts a new listing
FOOTER_Y = 690              # running footer ("2026 Pharmacy Drug Guide ...") starts at y=694

TIERS = {
    "PG": "Preferred generic", "PB": "Preferred brand", "NP": "Non-preferred brand and generic",
    "PSP": "Preferred specialty", "NPSP": "Non-preferred specialty",
    "NF": "Non-formulary: not covered unless an exception request is granted",
    "CE": "Copay exception: $0 with a prescription at an in-network pharmacy (limits may apply)",
}
REQUIREMENTS = {
    "PA": "Prior authorization", "QL": "Quantity limit", "QLR": "Quantity limit based on age",
    "ST": "Step therapy", "STX": "Safer and/or more effective treatments are available",
    "AL": "Age limit", "IBC": "Indication based coverage", "N7": "Drug tier when CE does not apply",
    "N8": "Drug specific coverage", "N10": "Drug coverage for Student Health members",
    "SPC": "Select plan coverage only", "Select OTC": "Select OTC program",
}


def is_bold(w):
    return "Bold" in w["fontname"]


def lines_of(words):
    """Group words into lines by their top coordinate."""
    rows = {}
    for w in words:
        key = round(w["top"])
        hit = next((k for k in rows if abs(k - key) <= 2), key)
        rows.setdefault(hit, []).append(w)
    return [sorted(v, key=lambda w: w["x0"]) for k, v in sorted(rows.items())]


def text(ws):
    t = " ".join(w["text"] for w in ws).strip()
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", t))


def is_index_page(page_text):
    return page_text.count("....") > 5


def parse_drugs(pdf):
    drugs, category, subcategory, last_set = [], "", "", "cat"
    for pno in range(FIRST_DRUG_PAGE - 1, len(pdf.pages)):
        page = pdf.pages[pno]
        raw = page.extract_text() or ""
        if is_index_page(raw):
            break
        words = page.extract_words(extra_attrs=["fontname", "size"])
        # start below the column header ("Prescription Drug Name ... Drug Tier")
        head = [w for w in words if w["text"] == "Prescription" and is_bold(w)]
        start = head[0]["top"] + 12 if head else 0   # header spans 2 lines ("... and Limits")
        words = [w for w in words if start < w["top"] < FOOTER_Y]  # drop column header and footer
        # bold header lines vs table lines
        ls = lines_of(words)
        block, last_top, prev = [], None, []

        def flush(block):
            if not block:
                return None
            name_ws = [w for l in block for w in l if w["x1"] <= TIER_X + 5]
            tier_ws = [w for l in block for w in l if TIER_X < w["x0"] < REQ_X]
            req_ws = [w for l in block for w in l if w["x0"] >= REQ_X]
            name = text(name_ws)
            tier = text(tier_ws)
            if not name or not tier:
                return None
            italic = sum("Italic" in w["fontname"] for w in name_ws) > len(name_ws) / 2
            m = re.search(r"\(([^()]*(?:\([^()]*\))?[^()]*)\)\s*$", name) if not italic else None
            return dict(drug=name, generic_name=(m.group(1) if m else name if italic else None),
                        is_generic=italic, tier=tier, tier_label=TIERS.get(tier, tier),
                        requirements=text(req_ws), category=category,
                        subcategory=subcategory, page=pno + 1)

        for l in ls:
            if all(is_bold(w) for w in l):
                d = flush(block)
                if d:
                    drugs.append(d)
                block = []
                t = text(l)
                wrapped = last_top is not None and l[0]["top"] - last_top <= 16 and \
                    prev and all(is_bold(w) for w in prev)
                if wrapped:  # header continues on a 2nd line
                    if last_set == "cat":
                        category = f"{category} {t}"
                    else:
                        subcategory = f"{subcategory} {t}"
                elif " - " in t or t.endswith(" -") or t == "OTHER":
                    category, subcategory, last_set = t, "", "cat"
                else:
                    subcategory, last_set = t, "sub"
                last_top, prev = l[0]["top"], l
                continue
            if last_top is not None and block and l[0]["top"] - last_top > ROW_GAP:
                d = flush(block)
                if d:
                    drugs.append(d)
                block = []
            block.append(l)
            last_top, prev = l[0]["top"], l
        d = flush(block)
        if d:
            drugs.append(d)
    return drugs


def info_chunks():
    chunks = []
    for p in INFO_PAGES:
        t = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), str(PDF), "-"],
                           capture_output=True, text=True).stdout.strip()
        if not t:
            continue
        first = t.split("\n", 1)[0].strip()
        paras, buf = [], []
        for line in t.split("\n"):
            buf.append(line)
            if len(" ".join(buf).split()) > 350 and (line.endswith(".") or not line.strip()):
                paras.append("\n".join(buf))
                buf = []
        if buf:
            paras.append("\n".join(buf))
        for part in paras:
            chunks.append(dict(page_start=p, page_end=p, heading=first, text=part.strip()))
    legend = ("Drug tiers: " + "; ".join(f"{k} = {v}" for k, v in TIERS.items()) +
              ".\nCoverage requirements and limits: " +
              "; ".join(f"{k} = {v}" for k, v in REQUIREMENTS.items()) +
              ".\nGeneric drugs are listed in lowercase italics; brand-name drugs in UPPERCASE.")
    chunks.append(dict(page_start=21, page_end=21, heading="Drug tier and requirement codes", text=legend))
    out = []
    for n, c in enumerate(chunks, 1):
        out.append(dict(id=f"AET-RX-{n:03d}", source_code="AET-RX", family="AET-RX",
                        source_title=TITLE, source_type="official", url=URL,
                        heading_path=f"{TITLE} > {c['heading']}", text=c["text"],
                        page_start=c["page_start"], page_end=c["page_end"],
                        plan_year="2026", date_captured=date.today().isoformat()))
    return out


def main():
    with pdfplumber.open(PDF) as pdf:
        drugs = parse_drugs(pdf)
    with OUT_DRUGS.open("w", encoding="utf-8") as f:
        for n, d in enumerate(drugs, 1):
            f.write(json.dumps({"id": f"RX-{n:05d}", **d}, ensure_ascii=False) + "\n")
    chunks = info_chunks()
    with OUT_CHUNKS.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tiers = {}
    for d in drugs:
        tiers[d["tier"]] = tiers.get(d["tier"], 0) + 1
    print(f"{len(drugs)} drug listings, tiers {tiers}; {len(chunks)} info chunks")


if __name__ == "__main__":
    main()
