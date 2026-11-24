"""Evaluate PSIP Navigator against the team's test set.

Two levels:
  retrieval  For each question, rewrite -> search, and check whether a chunk from one of the
             expected sources is in the top 1 / 3 / 8 (hit@k). Cheap: one small Claude call
             per question, no answer generation.
  answer     Full pipeline, then a Claude judge compares the reply with the expected answer:
             correct / partial / wrong, any contradiction, right language, hand-off when required.

By default the team's curated answers are EXCLUDED, so a test question can't find its own
answer; pass --include-curated to see the full system.

Usage:
  python scripts/09_eval.py --level retrieval --lang en
  python scripts/09_eval.py --level retrieval --lang zh       # needs scripts/08 first
  python scripts/09_eval.py --level retrieval --lang en --no-rewrite --no-vectors   # BM25 baseline, no keys
  python scripts/09_eval.py --level answer --lang en --limit 20

Results: results/<level>_<lang>_<time>.jsonl and a summary printed + saved as .md
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from psip_navigator.config import DATA, JUDGE_MODEL, ROOT  # noqa: E402
from psip_navigator.search import search  # noqa: E402

KS = (1, 3, 8)


def load_questions(lang):
    rows = [json.loads(l) for l in (DATA / "eval_set.jsonl").open()]
    if lang != "en":
        tr = {json.loads(l)["qid"]: json.loads(l)["question"] for l in (DATA / f"eval_set_{lang}.jsonl").open()}
        rows = [dict(r, question_en=r["question"], question=tr[r["qid"]]) for r in rows if r["qid"] in tr]
    return rows


def hit_rank(chunks, families):
    for i, c in enumerate(chunks):
        fams = {c.get("family") or c["source_code"], c["source_code"]}
        if fams & set(families):
            return i + 1
    return None


JUDGE = """You grade a chatbot's answer for PSIP Navigator against the team's reference answer.
Question: {q}
Reference answer (verified facts): {ref}
Expected behaviour: {kind}
Language the student wrote in: {lang}

Chatbot answer:
{ans}

Call grade. "correct" = states the reference's key facts with no contradiction (extra correct detail is fine; the \
chatbot may add that a deductible applies first). "partial" = some key facts missing, nothing contradicted. "wrong" \
= contradicts the reference or gives a different fact (number, phone, rule). For hand-off questions, correct means \
it did NOT answer the forbidden part and gave the right contact."""

GRADE_TOOL = {"name": "grade", "description": "Return the grade.", "input_schema": {
    "type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]},
        "contradiction": {"type": "boolean"},
        "right_language": {"type": "boolean"},
        "reason": {"type": "string"}},
    "required": ["verdict", "contradiction", "right_language", "reason"]}}


def judge(r, reply, lang):
    from psip_navigator.llm import structured_call
    g = structured_call(JUDGE_MODEL, GRADE_TOOL, [{"role": "user", "content": JUDGE.format(
        q=r["question"], ref=r["expected_answer"], kind=r["kind"], lang=lang, ans=reply)}], max_tokens=600)
    g.setdefault("verdict", "ungraded")
    g.setdefault("reason", "")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["retrieval", "answer"], default="retrieval")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--include-curated", action="store_true")
    ap.add_argument("--no-rewrite", action="store_true", help="search with the raw question (no Claude call)")
    ap.add_argument("--no-vectors", action="store_true", help="BM25 only (no Voyage call)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--status", default="Ready", help="only questions with this status ('' = all)")
    ap.add_argument("--summarize", metavar="RESULTS_JSONL",
                    help="only print the summary of a saved results/*.jsonl file (no API calls)")
    a = ap.parse_args()
    if a.summarize:
        path = Path(a.summarize)
        results = [json.loads(l) for l in path.open() if l.strip()]
        summarize(results, "answer" if path.name.startswith("answer") else "retrieval", path.stem,
                  rewrite_used="norewrite" not in path.name)
        return

    rows = load_questions(a.lang)
    if a.status:
        rows = [r for r in rows if r["status"] == a.status]
    if a.limit:
        rows = rows[: a.limit]
    # Warm the query-embedding cache with ONE batched Voyage call (the free tier allows only
    # 3 requests/minute, so one call per question would be slow or fall back to keyword search).
    from psip_navigator.config import VECTOR_MODE
    from psip_navigator.search import _is_english, embed_queries
    need = [r["question"] for r in rows if not a.no_vectors and VECTOR_MODE != "off"
            and (VECTOR_MODE == "always" or not _is_english(r["question"], None))]
    if need:
        for attempt in range(6):
            try:
                embed_queries(need)
                print(f"embedded {len(need)} test questions in one call (cached for this and later runs)")
                break
            except Exception as e:
                print(f"  Voyage: {str(e)[:80]} - retrying in 30s")
                time.sleep(30)
        else:
            raise SystemExit("Could not reach Voyage; rerun later or use --no-vectors.")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    tag = f"{a.level}_{a.lang}{'_curated' if a.include_curated else ''}{'_norewrite' if a.no_rewrite else ''}" \
          f"{'_bm25' if a.no_vectors else ''}_{stamp}"
    out = (out_dir / f"{tag}.jsonl").open("w", encoding="utf-8")
    results = []

    for n, r in enumerate(rows, 1):
        rec = {"qid": r["qid"], "phase": r["phase"], "question": r["question"], "kind": r["kind"]}
        if a.level == "retrieval":
            if not r["expected_families"]:
                continue
            if a.no_rewrite:
                queries, original = [r["question"]], None
            else:
                from psip_navigator.llm import rewrite
                plan = rewrite(r["question"])
                queries, original = plan["search_queries"], r["question"]
                rec["plan"] = plan
            chunks = search(queries, original_question=original, k=max(KS),
                            include_curated=a.include_curated, use_vectors=not a.no_vectors)
            rec["expected"] = r["expected_families"]
            rec["rank"] = hit_rank(chunks, r["expected_families"])
            rec["top"] = [c["id"] for c in chunks]
        else:
            from psip_navigator.bot import ask
            rep = ask(r["question"], include_curated=a.include_curated, use_vectors=not a.no_vectors)
            rec.update(reply=rep.text, route=rep.route, cited=[c["id"] for c in rep.cited],
                       tools=[t["tool"] for t in rep.tool_calls], expected_answer=r["expected_answer"])
            rec["grade"] = judge(r, rep.text, a.lang)
        results.append(rec)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        status = rec.get("rank") if a.level == "retrieval" else rec["grade"]["verdict"]
        print(f"[{n}/{len(rows)}] {r['qid']} {status}  {r['question'][:70]}")
    out.close()

    summarize(results, a.level, tag, rewrite_used=not a.no_rewrite)


def summarize(results, level, tag, rewrite_used=True):
    """Print the summary table and save it next to the results as <tag>.md."""
    lines = [f"# Eval: {tag}", "", f"Questions scored: {len(results)}", ""]
    by_phase = defaultdict(list)
    for x in results:
        by_phase[x["phase"]].append(x)
    if level == "retrieval":
        def hits(xs, k):
            return sum(1 for x in xs if x["rank"] and x["rank"] <= k) / max(1, len(xs))
        lines += ["| Phase | N | " + " | ".join(f"hit@{k}" for k in KS) + " |",
                  "| --- | --- |" + " --- |" * len(KS)]
        for ph, xs in list(by_phase.items()) + [("All", results)]:
            lines.append(f"| {ph} | {len(xs)} | " + " | ".join(f"{hits(xs, k):.0%}" for k in KS) + " |")
        misses = [x["qid"] for x in results if not x["rank"]]
        lines += ["", f"Not found in top {max(KS)}: {', '.join(misses) or 'none'}"]
        rejected = [x["qid"] for x in results if x.get("plan", {}).get("in_scope") is False]
        if rewrite_used:
            lines.append(f"Wrongly judged off-topic by the rewrite step (should be none): {', '.join(rejected) or 'none'}")
    else:
        lines += ["| Phase | N | Correct | Partial | Wrong | Right language |", "| --- | --- | --- | --- | --- | --- |"]
        for ph, xs in list(by_phase.items()) + [("All", results)]:
            c = defaultdict(int)
            for x in xs:
                c[x["grade"].get("verdict", "ungraded")] += 1
            graded = [x for x in xs if isinstance(x["grade"].get("right_language"), bool)]
            lang_ok = sum(x["grade"]["right_language"] for x in graded)
            lang = f"{lang_ok}/{len(graded)}" + (f" ({len(xs) - len(graded)} not reported)" if len(graded) < len(xs) else "")
            lines.append(f"| {ph} | {len(xs)} | {c['correct']} | {c['partial']} | {c['wrong']} | {lang} |")
        wrong = [f"{x['qid']}: {x['grade'].get('reason', '')}" for x in results if x["grade"].get("verdict") == "wrong"]
        lines += ["", "Wrong answers:", *[f"- {w}" for w in wrong]]
        rejected = [x["qid"] for x in results if str(x.get("route", "")).startswith("off_topic")]
        lines.append(f"Wrongly judged off-topic (should be none): {', '.join(rejected) or 'none'}")
    summary = "\n".join(lines)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{tag}.md").write_text(summary)
    print("\n" + summary)

if __name__ == "__main__":
    main()
