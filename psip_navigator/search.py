"""Hybrid search over data/corpus.jsonl.

- BM25 keyword search on the English queries (exact plan terms, drug names, phone numbers)
- Voyage vector search on the English queries AND on the student's original question
  (voyage-3.5 is multilingual, so the original Chinese/Korean/... wording is a backup
  in case the rewrite loses something)
- Reciprocal rank fusion (RRF) merges all ranked lists.

`include_curated=False` drops the team's test-set answers (used by the eval).
The vector part is skipped automatically if the index hasn't been built yet.
"""
import json
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from .config import CORPUS, EMBED_MODEL, INDEX_DIR, RRF_K, VECTOR_MODE, key

STOP = set("""a an and are as at be by can do does for from has have how i if in is it its me my of on or
our so that the their then there these this to was we what when where which who why will with you your
""".split())


def tokenize(text: str):
    return [t for t in re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower()) if t not in STOP]


def doc_text(c: dict) -> str:
    """What gets indexed: heading path gives the chunk its context."""
    return f"{c['heading_path']}\n{c['text']}"


@lru_cache
def corpus():
    return [json.loads(l) for l in CORPUS.open()]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.tfs))
        for i, (tf, ln) in enumerate(zip(self.tfs, self.lens)):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out[i] = s
        return out


@lru_cache
def bm25():
    return BM25([doc_text(c) for c in corpus()])


@lru_cache
def vector_index():
    """Returns (matrix, ids) or None if scripts/07_build_index.py hasn't been run."""
    emb, ids = INDEX_DIR / "embeddings.npy", INDEX_DIR / "ids.json"
    if not emb.exists():
        return None
    m = np.load(emb)
    order = json.loads(ids.read_text())
    pos = {c["id"]: i for i, c in enumerate(corpus())}
    if [pos.get(i) for i in order] != list(range(len(order))) or len(order) != len(corpus()):
        raise RuntimeError("Vector index is out of date with corpus.jsonl; rerun scripts/07_build_index.py")
    return m / np.linalg.norm(m, axis=1, keepdims=True)


class VoyageHTTP:
    """Minimal Voyage embeddings client over HTTPS (no SDK needed).

    Works with keys from the Voyage dashboard (default endpoint api.voyageai.com) and with
    keys created in MongoDB Atlas (set VOYAGE_BASE_URL=https://ai.mongodb.com in .env).
    """

    def __init__(self, api_key, base_url):
        self.api_key, self.url = api_key, base_url.rstrip("/") + "/v1/embeddings"

    def embed(self, texts, model, input_type=None, truncation=True):
        import json as _json
        import types
        import urllib.error
        import urllib.request
        body = {"input": list(texts), "model": model, "truncation": truncation}
        if input_type:
            body["input_type"] = input_type
        req = urllib.request.Request(self.url, data=_json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = _json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            hint = (" (Key from MongoDB Atlas? Set VOYAGE_BASE_URL=https://ai.mongodb.com in .env.)"
                    if e.code in (401, 403) else "")
            raise RuntimeError(f"Voyage API error {e.code}: {msg}{hint}") from e
        rows = sorted(data["data"], key=lambda d: d["index"])
        return types.SimpleNamespace(embeddings=[d["embedding"] for d in rows])


@lru_cache
def voyage():
    import os
    return VoyageHTTP(key("VOYAGE_API_KEY"), os.environ.get("VOYAGE_BASE_URL", "https://api.voyageai.com"))


QUERY_CACHE = INDEX_DIR / "query_cache.json"   # question text -> vector, so repeats cost no API call
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(QUERY_CACHE.read_text()) if QUERY_CACHE.exists() else {}
        except Exception:
            _cache = {}
    return _cache


def embed_queries(texts):
    """Embed query texts with Voyage, using the on-disk cache for anything seen before."""
    cache = _load_cache()
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        r = voyage().embed(todo, model=EMBED_MODEL, input_type="query")
        for t, v in zip(todo, r.embeddings):
            cache[t] = v
        try:
            QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            QUERY_CACHE.write_text(json.dumps(cache))
        except OSError:
            pass  # cache is only an optimisation
    m = np.array([cache[t] for t in texts], dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _rank(scores, allowed, n=50):
    idx = [i for i in np.argsort(-scores) if allowed[i] and scores[i] > 0][:n]
    return idx


def _is_english(text, language):
    if language:
        return language.lower().startswith("en")
    letters = [ch for ch in text if ch.isalpha()]
    return not letters or sum(ord(ch) < 128 for ch in letters) / len(letters) > 0.9


def search(english_queries, original_question=None, k=8, include_curated=True, use_vectors=True,
           language=None, info=None):
    """Return the top-k chunks (dicts, with a 'score' and 'matched_by') for a set of queries.

    Vector search (Voyage) is a backup for meaning the English rewrite may lose, so with
    VECTOR_MODE=auto (default) it runs only for non-English questions, and it embeds only the
    student's original question: at most one short API call per message. VECTOR_MODE=always
    also embeds English questions; VECTOR_MODE=off never calls Voyage.
    If Voyage fails (rate limit, network), search continues with BM25 only.
    `info`, if given, is filled with what happened: info["vectors"] = used / skipped / unavailable.
    """
    info = info if info is not None else {}
    docs = corpus()
    allowed = [include_curated or c["source_type"] != "curated" for c in docs]
    rankings = []  # (name, [doc indices])
    for q in english_queries:
        rankings.append((f"bm25:{q}", _rank(bm25().scores(q), allowed)))

    mode = VECTOR_MODE if use_vectors else "off"
    text = original_question or " ".join(english_queries)
    vi = vector_index() if mode != "off" else None
    if vi is None:
        info["vectors"] = "off" if mode == "off" else "skipped: index not built"
    elif mode == "auto" and _is_english(text, language):
        info["vectors"] = "skipped: English question"
    else:
        try:
            v = embed_queries([text])[0]
            rankings.append(("vec:original", _rank(vi @ v, allowed)))
            info["vectors"] = "used"
        except Exception as e:  # never fail the answer because of the embedding API
            info["vectors"] = "unavailable: " + ("rate limited" if "429" in str(e) else str(e)[:80])
    fused, why = Counter(), {}
    for name, idx in rankings:
        for r, i in enumerate(idx):
            fused[i] += 1.0 / (RRF_K + r + 1)
            why.setdefault(i, set()).add(name.split(":")[0])
    out = []
    for i, s in fused.most_common(k):
        c = dict(docs[i])
        c["score"] = round(s, 5)
        c["matched_by"] = sorted(why[i])
        out.append(c)
    return out
