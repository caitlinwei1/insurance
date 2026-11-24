"""Translate the test questions into Chinese (or another language) so we can test
multilingual retrieval: the same question in two languages should find the same sources.

Writes data/eval_set_<lang>.jsonl with {qid, question}. Re-running skips questions
already translated.

Usage:  python scripts/08_translate_eval.py            # Chinese
        python scripts/08_translate_eval.py --lang ko  # Korean
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from psip_navigator.config import DATA, REWRITE_MODEL  # noqa: E402
from psip_navigator.llm import client  # noqa: E402

NAMES = {"zh": "Simplified Chinese", "ko": "Korean", "hi": "Hindi", "es": "Spanish", "ja": "Japanese"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zh")
    a = ap.parse_args()
    out = DATA / f"eval_set_{a.lang}.jsonl"
    done = {json.loads(l)["qid"] for l in out.open()} if out.exists() else set()
    rows = [json.loads(l) for l in (DATA / "eval_set.jsonl").open()]
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            if r["qid"] in done:
                continue
            resp = client().messages.create(
                model=REWRITE_MODEL, max_tokens=300,
                system=(f"Translate the student's question into natural, casual {NAMES.get(a.lang, a.lang)}, "
                        "the way an international student would actually type it to a chatbot. Keep proper "
                        "names (Penn, Student Health, Aetna, PSIP, CAPS) and drug names as they are. "
                        "Output only the translation."),
                messages=[{"role": "user", "content": r["question"]}])
            q = "".join(b.text for b in resp.content if b.type == "text").strip()
            f.write(json.dumps({"qid": r["qid"], "question": q}, ensure_ascii=False) + "\n")
            print(r["qid"], q)


if __name__ == "__main__":
    main()
