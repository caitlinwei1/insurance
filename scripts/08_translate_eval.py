"""Translate the test questions into Chinese (or another language) so we can test
multilingual retrieval: the same question in two languages should find the same sources.

Writes data/eval_set_<lang>.jsonl with {qid, question}. Re-running keeps good translations,
redoes bad ones (model notes, several lines, wrong script such as pinyin) and adds missing ones.

Usage:  python scripts/08_translate_eval.py            # Chinese
        python scripts/08_translate_eval.py --lang ko  # Korean
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from psip_navigator.config import DATA, REWRITE_MODEL  # noqa: E402
from psip_navigator.llm import structured_call  # noqa: E402

NAMES = {"zh": "Simplified Chinese", "ko": "Korean", "hi": "Hindi", "es": "Spanish", "ja": "Japanese"}
SCRIPT = {"zh": r"[一-鿿]", "ja": r"[぀-ヿ一-鿿]", "ko": r"[가-힯]",
          "hi": r"[ऀ-ॿ]"}
TOOL = {"name": "submit_translation", "description": "Return the translated question.",
        "input_schema": {"type": "object", "properties": {"translation": {"type": "string"}},
                         "required": ["translation"]}}


def good(text, lang):
    if not text or "\n" in text.strip() or re.search(r"\b(Wait|Note|translation)\b", text, re.I):
        return False
    return not SCRIPT.get(lang) or re.search(SCRIPT[lang], text) is not None


def translate(question, lang):
    system = (f"Translate the student's question into natural, casual {NAMES.get(lang, lang)}, the way an "
              "international student would type it to a chatbot, written in that language's normal script "
              "(no romanization). Keep proper names (Penn, Student Health, Aetna, PSIP, CAPS, EOB) and drug "
              "names as they are. Return only the translated question, one line, no notes.")
    for _ in range(3):
        out = structured_call(REWRITE_MODEL, TOOL, [{"role": "user", "content": question}],
                              system=system, max_tokens=300)["translation"].strip()
        if good(out, lang):
            return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zh")
    a = ap.parse_args()
    out = DATA / f"eval_set_{a.lang}.jsonl"
    existing = {}
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                existing[r["qid"]] = r["question"]
            except Exception:
                pass
    rows = [json.loads(l) for l in (DATA / "eval_set.jsonl").open()]
    result, redone = [], 0
    for r in rows:
        q = existing.get(r["qid"])
        if not good(q, a.lang):
            q = translate(r["question"], a.lang)
            redone += 1
            print(r["qid"], q)
        result.append({"qid": r["qid"], "question": q})
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in result))
    print(f"{len(result)} questions in {out.name}; {redone} translated or redone")


if __name__ == "__main__":
    main()
