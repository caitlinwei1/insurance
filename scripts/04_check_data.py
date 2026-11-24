"""Sanity checks on the processed data. Run after scripts 01-03.

Checks
1. Every chunk has the required fields; IDs are unique.
2. Chunk sizes (words) per source.
3. Every number in plan_costs.json is backed by its quoted evidence in the cited chunk.
4. Key plan facts can be found somewhere in the official chunks.
5. Which sources the test set cites that are not in the corpus yet.

Writes data/data_report.md and exits non-zero if a hard check fails.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REQUIRED = ["id", "source_code", "source_title", "source_type", "heading_path", "text", "plan_year", "date_captured"]


def norm(s):
    return re.sub(r"[\s|]+", " ", s).strip().lower()


def load(name):
    p = DATA / name
    return [json.loads(l) for l in p.open()] if p.exists() else []


def main():
    pdf, web, cur = load("chunks_pdf.jsonl"), load("chunks_web.jsonl"), load("chunks_curated.jsonl")
    pdf += load("chunks_rx.jsonl")
    drugs = load("formulary.jsonl")
    official = pdf + web
    every = official + cur
    evals = load("eval_set.jsonl")
    fails, lines = [], ["# Data report", ""]

    # 1. schema + unique ids
    missing = [(c.get("id"), k) for c in every for k in REQUIRED if not c.get(k)]
    dupes = [k for k, v in Counter(c["id"] for c in every).items() if v > 1]
    if missing:
        fails.append(f"{len(missing)} missing fields, e.g. {missing[:3]}")
    if dupes:
        fails.append(f"duplicate ids: {dupes[:5]}")
    lines += [f"Chunks: {len(pdf)} PDF, {len(web)} web, {len(cur)} curated. "
              f"Missing fields: {len(missing)}. Duplicate IDs: {len(dupes)}.", ""]

    # 1b. formulary table
    tiers = Counter(d["tier"] for d in drugs)
    bad_tier = [d["id"] for d in drugs if d["tier"] not in ("PG", "PB", "NP", "PSP", "NPSP", "NF", "CE")]
    no_name = [d["id"] for d in drugs if not d["drug"]]
    if bad_tier or no_name:
        fails.append(f"formulary rows with bad tier {bad_tier[:5]} or no name {no_name[:5]}")
    lines += [f"Formulary: {len(drugs)} drug listings; tiers {dict(tiers)}; "
              f"{sum(1 for d in drugs if d['requirements'])} with requirements (PA/QL/ST...).", ""]

    # 2. sizes per source
    lines += ["## Chunk sizes (words)", "", "| Source | Chunks | Min | Median | Max |", "| --- | --- | --- | --- | --- |"]
    by_src = defaultdict(list)
    for c in every:
        key = c["source_code"] if c["source_code"].startswith("AET") or c["source_code"] == "CUR" else c.get("family", c["source_code"])
        by_src[key].append(len(c["text"].split()))
    for k in sorted(by_src):
        w = sorted(by_src[k])
        lines.append(f"| {k} | {len(w)} | {w[0]} | {w[len(w) // 2]} | {w[-1]} |")
    lines.append("")

    # 3. plan_costs evidence
    costs = json.loads((DATA / "plan_costs.json").read_text())
    by_id = {c["id"]: c for c in official}
    checks = []

    def walk(obj):
        if isinstance(obj, dict):
            if "source" in obj and "evidence" in obj:
                checks.append((obj["source"], obj["evidence"]))
            for v in obj.values():
                walk(v)
    walk(costs)
    for src, quote in costs["prescriptions"]["sources"].values():
        checks.append((src, quote))
    bad = [(s, q) for s, q in checks if s not in by_id or norm(q) not in norm(by_id[s]["text"])]
    if bad:
        fails.append(f"plan_costs evidence not found: {bad}")
    lines += [f"## Cost table evidence", "", f"{len(checks) - len(bad)}/{len(checks)} quotes found in their cited chunk.", ""]

    # 4. key facts present in official text
    facts = {
        "Deductible $400 / $1,500": r"\$400 per policy year.*\$1,500 per policy year",
        "Out-of-pocket max $1,500 / $4,000": r"\$1,500 per policy year.*\$4,000 per policy year",
        "Referral: Student Health first within 25 miles": r"25 miles",
        "Clinical Fee $385 per semester": r"\$385 per semester",
        "Premium $5,018": r"\$5,018",
        "Student Health phone 215-746-WELL": r"215-746-WELL",
        "Insurance Navigators 215-746-4200": r"215-746-4200",
        "Aetna Student Health 800-841-5374": r"841-5374",
        "Crisis line 988": r"\b988\b",
        "Penn Medicine billing 800-406-1177": r"800-406-1177",
    }
    lines += ["## Key facts found in official chunks", "", "| Fact | Found in |", "| --- | --- |"]
    for name, pat in facts.items():
        hits = [c["id"] for c in official if re.search(pat, c["text"].replace("\n", " "))]
        if not hits:
            fails.append(f"fact not found: {name}")
        lines.append(f"| {name} | {', '.join(hits[:4]) or '**missing**'}{' …' if len(hits) > 4 else ''} |")
    lines.append("")

    # 5. test-set sources not in corpus
    have = {c.get("family") or c["source_code"] for c in official} | {c["source_code"] for c in official}
    needed = Counter(f for r in evals for f in r["expected_families"])
    absent = {f: n for f, n in needed.items() if f not in have}
    qs_affected = [r["qid"] for r in evals if r["expected_families"] and not (set(r["expected_families"]) & have)]
    lines += ["## Test-set sources not in the corpus yet", "",
              "| Source | Cited by (questions) |", "| --- | --- |"]
    lines += [f"| {f} | {n} |" for f, n in sorted(absent.items(), key=lambda x: -x[1])]
    lines += ["", f"Questions whose cited sources are all missing: {len(qs_affected)} ({', '.join(qs_affected)})", ""]

    (DATA / "data_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    if fails:
        print("\nFAILED:\n- " + "\n- ".join(fails))
        sys.exit(1)
    print("\nAll hard checks passed.")


if __name__ == "__main__":
    main()
