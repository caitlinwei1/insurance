"""Turn the captured web pages (raw/web/*.json) into chunks with the same schema as the PDFs.

Each raw/web/<site>.json holds {captured, site, pages:[{url, final_url, title, md}]}:
the page's main content converted to Markdown (headings, lists, tables, links)
in the browser, with navigation, headers and footers removed.

Chunking: cut at "##"/"###" headings, keep the heading path, split long pieces
at paragraph breaks, merge tiny ones into the next piece.

Output: data/chunks_web.jsonl
Usage:  python scripts/02_chunk_web.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "web"
OUT = ROOT / "data" / "chunks_web.jsonl"
MAX_WORDS, MIN_WORDS = 400, 30

# URL path -> (source code, family). "family" groups pages that the test set
# cites under one code (e.g. the Medical Care sub-pages are all WEL-010).
CODES = {
    "/insurance-requirements/penn-student-insurance-plan-psip": ("WEL-046", "WEL-046"),
    "/immunization-insurance-compliance/insurance-compliance": ("WEL-027", "WEL-027"),
    "/student-health-counseling/medical-care": ("WEL-010", "WEL-010"),
    "/appointments": ("WEL-APT", "WEL-010"),
    "/get-help-now": ("WEL-HLP", "WEL-010"),
    "/immunization-insurance-requirements/new-and-incoming-students": ("WEL-NEW", "WEL-NEW"),
    "/student-health-counseling/counseling": ("WEL-CNS", "WEL-CNS"),
    "/counseling/individual-therapy": ("WEL-THR", "WEL-CNS"),
    "/counseling/mental-health-and-suicide-prevention-resources": ("WEL-CRS", "WEL-CNS"),
    "/student-health-and-counseling": ("WEL-SHC", "WEL-010"),
    "/immunization-insurance-compliance/immunization-compliance": ("WEL-IMM", "WEL-NEW"),
    "/faqs-immunization-requirements": ("WEL-IMF", "WEL-NEW"),
    "/immunization-and-insurance-requirements": ("WEL-014", "WEL-027"),
    "/isss/healthcare/": ("GLB", "GLB"),
    "/patient-resources/policies/patient-billing": ("PMD-BIL", "PMD-BIL"),
    "/patient-resources/policies/patient-billing/faqs": ("PMD-BFQ", "PMD-BIL"),
    "/patient-resources/policies/financial-assistance": ("PMD-FIN", "PMD-FIN"),
    "/patient-resources/policies/no-surprises-act": ("PMD-NSA", "GOV-NSA"),
    "/patient-resources/information-for-patients/language-support-services": ("PMD-LNG", "PMD-LNG"),
    # CMS (U.S. government) guide to reading an EOB: the glossary has no EOB entry
    "/initiatives/your-patient-rights/medical-bill-rights/get-help/guides-resources/read-explanation-benefits":
        ("GOV-EOB", "GOV"),
}

# Lines/blocks that are page furniture, not content
DROP_LINE = re.compile(r"^(#+ )?(navigate this section|video file|print page|overview|get directions)$", re.I)
CUT_FROM = re.compile(r"^#+ (related content|more resources|sources)\s*$", re.I | re.M)


def code_for(page):
    path = re.sub(r"^https?://[^/]+", "", page["final_url"] or page["url"]).split("?")[0]
    if "/glossary/" in path:
        slug = path.rstrip("/").split("/")[-1]
        return "GOV-" + slug.upper()[:30], "GOV"
    if path in CODES:
        return CODES[path]
    if path.rstrip("/") + "/" in CODES:
        return CODES[path.rstrip("/") + "/"]
    raise KeyError(f"no source code for {path}")


def clean_md(md: str) -> str:
    m = CUT_FROM.search(md)
    if m:
        md = md[: m.start()]
    # shorten huge map/tracking URLs but keep the link text
    md = re.sub(r"\]\((https?://(www\.)?google\.[^)]+|https?://maps\.google[^)]+)\)", "](map)", md)
    md = re.sub(r"\[([^\]]+)\]\(map\)", r"\1", md)
    md = re.sub(r"\(link is external\)", "", md)
    md = re.sub(r"(\?|&)(utm_[^&)\s]+|WT\.mc_id=[^&)\s]+|sxsrf=[^&)\s]+)", "", md)
    lines = [l for l in md.split("\n") if not DROP_LINE.match(l.strip())]
    md = "\n".join(lines)
    # drop the left-hand section menu that sits before the first real heading on Wellness pages
    md = re.sub(r"^(- \[[^\n]+\]\([^)]+\)\n\n(  - [^\n]+\n\n)*)+", "", md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def sections(md, title):
    """Split markdown at ## / ### headings -> [(path list, text)]."""
    out, path, buf = [], [], []
    for block in md.split("\n\n"):
        m = re.match(r"^(#{1,6}) (.+)$", block.strip())
        if m:
            if buf:
                out.append((list(path), "\n\n".join(buf)))
                buf = []
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1 or text == title:
                continue
            path = [p for p in path if p[0] < level] + [(level, text)]
            continue
        buf.append(block)
    if buf:
        out.append((list(path), "\n\n".join(buf)))
    # keep the top section heading and the nearest heading; deeper nesting on these
    # pages is unreliable (FAQ toggles and sub-headings share levels)
    def short(pth):
        names = [p[1] for p in pth]
        return names if len(names) <= 2 else [names[0], names[-1]]
    return [(short(pth), txt) for pth, txt in out]


def split_long(text):
    if len(text.split()) <= MAX_WORDS:
        return [text]
    parts, buf = [], []
    for para in text.split("\n\n"):
        if buf and len(" ".join(buf + [para]).split()) > MAX_WORDS:
            parts.append("\n\n".join(buf))
            buf = []
        buf.append(para)
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def chunk_page(page, captured):
    code, family = code_for(page)
    title = page["title"].strip()
    md = clean_md(page["md"])
    if family == "GOV" and code != "GOV-EOB":
        # glossary pages: drop the pop-up definitions of other terms, keep one chunk per term
        md = re.sub(r"### [^\n]+\n\n[^\n]+\n\nRefer to glossary for more details\.\n*", "", md)
        secs = [([], re.sub(r"^# [^\n]+\n+", "", md).strip())]
    else:
        secs = sections(md, title)
    merged = []
    for path, text in secs:
        if merged and len(merged[-1][1].split()) < MIN_WORDS and \
                len((merged[-1][1] + text).split()) <= MAX_WORDS:
            ppath, ptext = merged.pop()
            label = f"**{ppath[-1]}**\n\n" if ppath else ""
            text = label + ptext + f"\n\n**{path[-1]}**\n\n" + text if path else ptext + "\n\n" + text
        merged.append((path, text))
    chunks = []
    for path, text in merged:
        for part in split_long(text):
            if len(part.split()) < 5:
                continue
            chunks.append(dict(
                source_code=code, family=family, source_title=title,
                source_type="official", url=page["final_url"] or page["url"],
                heading_path=" > ".join([title] + path), text=part.strip(),
                page_start=None, page_end=None, plan_year="2026-2027",
                date_captured=captured[:10]))
    return chunks


def main():
    all_chunks = []
    for f in sorted(RAW.glob("*.json")):
        d = json.loads(f.read_text())
        for page in d["pages"]:
            if not page.get("md"):
                continue
            all_chunks += chunk_page(page, d["captured"])
    # number chunks per source code
    counter = {}
    with OUT.open("w", encoding="utf-8") as out:
        for c in all_chunks:
            counter[c["source_code"]] = counter.get(c["source_code"], 0) + 1
            cid = f"{c['source_code']}-{counter[c['source_code']]:02d}"
            out.write(json.dumps({"id": cid, **c}, ensure_ascii=False) + "\n")
    words = sorted(len(c["text"].split()) for c in all_chunks)
    print(f"{len(all_chunks)} web chunks from {len(counter)} pages; words min/median/max = "
          f"{words[0]}/{words[len(words) // 2]}/{words[-1]} -> {OUT}")


if __name__ == "__main__":
    main()
