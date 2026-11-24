"""Offline tests: no API keys, no network.

Claude and Voyage are replaced by small fakes so the whole pipeline (router -> rewrite ->
hybrid search -> answer with tool calls -> cited sources) runs end to end.

Run:  python -m pytest tests -q      (or: python tests/test_offline.py)
"""
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- fakes
class Block:
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


class Resp:
    def __init__(self, content, stop_reason="end_turn"):
        self.content, self.stop_reason = content, stop_reason


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        tools = [t["name"] for t in kw.get("tools", [])]
        if "submit_search_plan" in tools:  # rewrite step
            q = kw["messages"][-1]["content"]
            zh = any("一" <= ch <= "鿿" for ch in q)
            off = any(w in q.lower() for w in ("weather", "pizza", "天气"))
            return Resp([Block("tool_use", id="t0", name="submit_search_plan", input={
                "language": "zh" if zh else "en", "english_question": "How much does urgent care cost?",
                "search_queries": ["urgent care copayment", "referral requirements Student Health"],
                "drug_names": [], "crisis": False, "emergency": False, "in_scope": not off,
                "message_type": "unrelated_request" if off else "question",
                "off_topic_reply": "我只能回答 Penn 学生保险和看病相关的问题。" if off else ""})], "tool_use")
        # answer step: first round asks for the calculator, second round answers
        last = kw["messages"][-1]["content"]
        if isinstance(last, str):
            return Resp([Block("tool_use", id="t1", name="calculate_cost",
                               input={"service": "urgent_care", "allowed_amount": 200, "deductible_paid_so_far": 400})],
                        "tool_use")
        result = json.loads(last[0]["content"])
        return Resp([Block("text", text=f"Urgent care costs ${result['student_pays']:.0f} once your deductible "
                                        f"is met [AET-PDB-036]. You need a referral [WEL-027-06]. [FAKE-ID]")])


class FakeStream:
    """Mimics client.messages.stream(...): a context manager with text_stream + get_final_message()."""
    def __init__(self, resp):
        self.resp = resp

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def text_stream(self):
        for b in self.resp.content:
            if b.type == "text":
                for i in range(0, len(b.text), 7):
                    yield b.text[i:i + 7]

    def get_final_message(self):
        return self.resp


FakeMessages.stream = lambda self, **kw: FakeStream(self.create(**kw))
fake_messages = FakeMessages()
sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=lambda api_key: types.SimpleNamespace(messages=fake_messages))


def fake_embed(texts, model=None, input_type=None, truncation=True):
    vecs = []
    for t in texts:
        rng = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
        vecs.append(rng.normal(size=32).tolist())
    return types.SimpleNamespace(embeddings=vecs)



import os  # noqa: E402
os.environ.setdefault("ANTHROPIC_API_KEY", "test")
os.environ.setdefault("VOYAGE_API_KEY", "test")

from psip_navigator import config, search as S  # noqa: E402
S.voyage = lambda: types.SimpleNamespace(embed=fake_embed)   # replace the HTTP client
from psip_navigator.bot import ask, ask_stream  # noqa: E402
from psip_navigator.router import route  # noqa: E402
from psip_navigator.tools import calculate_cost, lookup_drug  # noqa: E402


# ---------------------------------------------------------------- tests
def test_router():
    assert route("I have chest pain and can't breathe").kind == "emergency"
    assert route("我不想活了").kind == "crisis"
    assert "988" in route("我不想活了").reply and "Penn" in route("我不想活了").reply
    assert route("How much is urgent care?").kind == "ok"


def test_calculator_follows_plan_order():
    # deductible not met: the first $200 all goes to the $400 deductible
    assert calculate_cost("urgent_care", 200)["student_pays"] == 200
    # deductible met: only the $35 copay
    assert calculate_cost("urgent_care", 200, deductible_paid_so_far=400)["student_pays"] == 35
    # ER: $100 copay, deductible waived
    assert calculate_cost("emergency_room", 3000)["student_pays"] == 100
    # out of network after deductible: student pays 30%
    r = calculate_cost("office_visit", 1000, network="out", deductible_paid_so_far=1500)
    assert r["breakdown"]["coinsurance"] == 300
    # no referral -> out-of-network cost sharing
    assert calculate_cost("lab_xray", 100, has_referral=False)["cost_level"] == "out-of-network"
    # mental health: $10 copay, no deductible
    assert calculate_cost("mental_health_outpatient", 150)["student_pays"] == 10
    # capped by out-of-pocket max
    assert calculate_cost("office_visit", 900, out_of_pocket_so_far=1450)["student_pays"] == 50


def test_drug_lookup():
    r = lookup_drug("adderall")
    assert r["matches"][0]["tier"] == "NF"
    assert any(g["tier"] == "PG" and g["copay"] == 20 for g in r["generic_versions"])
    fl = lookup_drug("fluoxetine")["matches"]
    assert any(m["tier"] == "PG" and m["copay"] == 20 for m in fl)      # 10/20 mg capsules: preferred generic
    assert any(m["tier"] == "NF" and m["copay"] is None for m in fl)     # 60 mg tablet: not covered


def test_bm25_finds_plan_terms():
    top = [c["id"] for c in S.search(["referral requirements Student Health 25 miles"], k=3,
                                     include_curated=False, use_vectors=False)]
    assert "WEL-027-06" in top


def test_curated_filter():
    res = S.search(["68,000 pharmacies network"], k=20, include_curated=False, use_vectors=False)
    assert all(c["source_type"] != "curated" for c in res)


def test_full_pipeline_with_vectors(tmp_path=None):
    # build a fake vector index in a temp dir
    import tempfile
    idx = Path(tmp_path or tempfile.mkdtemp())
    docs = S.corpus()
    np.save(idx / "embeddings.npy", np.array(fake_embed([S.doc_text(c) for c in docs]).embeddings, dtype=np.float32))
    (idx / "ids.json").write_text(json.dumps([c["id"] for c in docs]))
    S.INDEX_DIR = idx
    S.QUERY_CACHE = idx / "query_cache.json"
    S._cache = None
    S.vector_index.cache_clear()

    rep = ask("急诊太贵了，urgent care 要多少钱？")
    assert rep.route == "ok" and rep.plan["language"] == "zh"
    assert any("vec" in c["matched_by"] for c in rep.chunks)         # vector search ran
    assert rep.tool_calls and rep.tool_calls[0]["tool"] == "calculate_cost"
    assert "$35" in rep.text                                         # number came from the tool
    cited = [c["id"] for c in rep.cited]
    assert "FAKE-ID" not in cited                                    # only retrieved IDs are listed
    # the answer prompt contains the sources and the student's language
    system = fake_messages.calls[-1]["system"]
    assert "Reply in the student's language: zh" in system and "<source id=" in system


def test_stream_events_in_order():
    evs = list(ask_stream("How much is urgent care?"))
    kinds = [e[0] for e in evs]
    stages = [e[1] for e in evs if e[0] == "stage"]
    assert stages[:3] == ["rewrite", "search", "answer"] and "tool" in stages
    assert kinds.index("text") > kinds.index("stage") and kinds[-1] == "done"
    streamed = "".join(e[1] for e in evs if e[0] == "text")
    rep = evs[-1][1]
    assert streamed.strip() == rep.text.strip() and len([e for e in evs if e[0] == "text"]) > 3
    assert {"rewrite", "search", "answer", "total", "first_text"} <= set(rep.timings)


def test_emergency_skips_ai():
    n = len(fake_messages.calls)
    rep = ask("my friend is unconscious and not breathing")
    assert rep.route == "emergency" and len(fake_messages.calls) == n


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)


def _calls_counter():
    calls = {"n": 0}
    def embed(texts, model=None, input_type=None, truncation=True):
        calls["n"] += 1
        return fake_embed(texts)
    return calls, embed


def test_vector_policy_cache_and_fallback():
    test_full_pipeline_with_vectors()          # builds the temp index
    calls, embed = _calls_counter()
    S.voyage = lambda: types.SimpleNamespace(embed=embed)
    info = {}
    S.search(["urgent care copayment"], original_question="How much is urgent care?", language="en", info=info)
    assert info["vectors"].startswith("skipped") and calls["n"] == 0          # English: no Voyage call
    info = {}
    S.search(["urgent care copayment"], original_question="急诊多少钱？", language="zh", info=info)
    assert info["vectors"] == "used" and calls["n"] == 1                        # one call, original question only
    S.search(["urgent care copayment"], original_question="急诊多少钱？", language="zh")
    assert calls["n"] == 1                                                      # cached: no new call

    def limited(texts, **kw):
        raise RuntimeError('Voyage API error 429: {"detail":"reduced rate limits of 3 RPM"}')
    S.voyage = lambda: types.SimpleNamespace(embed=limited)
    info = {}
    res = S.search(["urgent care copayment"], original_question="新的问题：验血多少钱", language="zh", info=info)
    assert info["vectors"] == "unavailable: rate limited" and res                # still answers with BM25
    evs = list(ask_stream("新的问题：门诊多少钱"))
    assert any(e[0] == "stage" and "unavailable" in e[2] for e in evs) and evs[-1][0] == "done"
    S.voyage = lambda: types.SimpleNamespace(embed=fake_embed)


def test_off_topic_skips_search_and_answer():
    calls, embed = _calls_counter()
    S.voyage = lambda: types.SimpleNamespace(embed=embed)
    searched = {"n": 0}
    import psip_navigator.bot as B
    real = B.search
    B.search = lambda *a, **k: searched.__setitem__("n", searched["n"] + 1) or real(*a, **k)
    n_llm = len(fake_messages.calls)
    rep = ask("明天费城天气怎么样？")
    B.search = real
    assert rep.route == "off_topic:unrelated_request" and "Penn" in rep.text
    assert searched["n"] == 0 and calls["n"] == 0            # no search, no Voyage
    assert len(fake_messages.calls) == n_llm + 1             # only the rewrite call, no answer call
    assert ask("How much is urgent care?").route == "ok"      # in-scope questions unaffected
    S.voyage = lambda: types.SimpleNamespace(embed=fake_embed)
