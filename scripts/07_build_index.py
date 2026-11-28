"""Embed every chunk in data/corpus.jsonl with Voyage and save the vector index.

Needs VOYAGE_API_KEY in .env (and VOYAGE_BASE_URL=https://ai.mongodb.com if the key is from Atlas).
Rerun whenever corpus.jsonl changes.

Rate limits: without a payment method, Voyage's free tier allows only a few requests and
~10K tokens per minute. The corpus is ~200K tokens, so on the free tier this takes ~20-25
minutes. The script handles it: batches stay under ~8K tokens, it waits a minute when
rate-limited, slows down after the first limit, and SAVES PROGRESS after every batch, so if
you stop it (Ctrl+C) or it fails, rerunning continues where it left off.
With a payment method on file it finishes in about a minute.

Output: data/index/embeddings.npy (float32, one row per chunk) + data/index/ids.json
Usage:  python scripts/07_build_index.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from psip_navigator.config import EMBED_MODEL, INDEX_DIR  # noqa: E402
from psip_navigator.search import corpus, doc_text, voyage  # noqa: E402

MAX_BATCH_TOKENS = 8000     # stays under the free tier's 10K tokens/minute
MAX_BATCH_ITEMS = 64
SLOW_MODE_GAP = 21          # seconds between requests once rate-limited (free tier: 3/min)
PARTIAL = INDEX_DIR / "partial.npy"       # progress: vectors embedded so far (compact binary)


def est_tokens(text):
    return len(text) / 3.5 + 10


def batches(texts, start):
    i = start
    while i < len(texts):
        j, tok = i, 0
        while j < len(texts) and j - i < MAX_BATCH_ITEMS and (j == i or tok + est_tokens(texts[j]) <= MAX_BATCH_TOKENS):
            tok += est_tokens(texts[j])
            j += 1
        yield i, j
        i = j


def is_rate_limit(e):
    return "429" in str(e) or "RateLimit" in e.__class__.__name__ or "rate limit" in str(e).lower()


def main():
    docs = corpus()
    ids = [c["id"] for c in docs]
    texts = [doc_text(c) for c in docs]
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse vectors of chunks already in the current index; embed only new or changed ones.
    old = {}
    if (INDEX_DIR / "embeddings.npy").exists() and (INDEX_DIR / "ids.json").exists() and not PARTIAL.exists():
        m = np.load(INDEX_DIR / "embeddings.npy")
        old = dict(zip(json.loads((INDEX_DIR / "ids.json").read_text()), m))
    if old:
        new = [i for i in ids if i not in old]
        print(f"{len(ids) - len(new)} chunks already embedded; embedding {len(new)} new: {new[:10]}")
        if new:
            r = voyage().embed([texts[ids.index(i)] for i in new], model=EMBED_MODEL, input_type="document",
                               truncation=True)
            old.update(zip(new, np.array(r.embeddings, dtype=np.float32)))
        np.save(INDEX_DIR / "embeddings.npy", np.array([old[i] for i in ids], dtype=np.float32))
        (INDEX_DIR / "ids.json").write_text(json.dumps(ids))
        print(f"saved {len(ids)} vectors to {INDEX_DIR}")
        return

    vecs = []
    if PARTIAL.exists():
        try:  # a crash mid-save can leave a broken file; then just start over
            vecs = np.load(PARTIAL).tolist()
            print(f"resuming: {len(vecs)}/{len(texts)} already embedded")
        except Exception:
            print("progress file was damaged; starting over")
            vecs = []

    slow = False
    for i, j in batches(texts, len(vecs)):
        for attempt in range(30):
            try:
                if slow:
                    time.sleep(SLOW_MODE_GAP)
                r = voyage().embed(texts[i:j], model=EMBED_MODEL, input_type="document", truncation=True)
                vecs += r.embeddings
                break
            except Exception as e:
                if not is_rate_limit(e):
                    raise
                if not slow:
                    print("  rate-limited: switching to slow mode (free tier). This will take ~20-25 minutes;")
                    print("  you can stop with Ctrl+C and rerun later, progress is saved.")
                slow = True
                print(f"  waiting 60s (rate limit), then retrying items {i}-{j}")
                time.sleep(60)
        else:
            raise SystemExit("Still rate-limited after many retries. Rerun later; progress is saved.")
        tmp = INDEX_DIR / "partial.tmp.npy"   # write, then rename: never leaves a half-written file
        np.save(tmp, np.array(vecs, dtype=np.float32))
        os.replace(tmp, PARTIAL)
        print(f"embedded {len(vecs)}/{len(texts)}")

    np.save(INDEX_DIR / "embeddings.npy", np.array(vecs, dtype=np.float32))
    (INDEX_DIR / "ids.json").write_text(json.dumps(ids))
    PARTIAL.unlink(missing_ok=True)
    (INDEX_DIR / "partial.json").unlink(missing_ok=True)   # left over from an older version
    print(f"saved {len(vecs)} x {len(vecs[0])} vectors to {INDEX_DIR}")


if __name__ == "__main__":
    main()
