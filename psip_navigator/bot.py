"""The full pipeline for one student message.

  1. router   fixed rules: crisis / emergency -> fixed reply, stop      (no AI)
  2. rewrite  Claude: language + English search queries + drug names    (AI: R1, R3)
  3. search   BM25 + Voyage vectors, fused                               (AI: R2)
  4. answer   Claude: cited reply in the student's language, calling
              calculate_cost / lookup_drug for every number               (tools: R4)

ask_stream() yields progress events so the chat page can show each step and stream the
answer as it is written; ask() runs the same pipeline and returns the final Reply.
"""
import re
import time
from dataclasses import dataclass, field

from . import llm, router
from .config import TOP_K
from .router import CRISIS_REPLY, looks_chinese
from .search import corpus, search

# Always given to the answer model: cost-sharing order, deductible waivers, referral exceptions,
# and the two Student Health addresses (medical care and counseling are in different buildings).
PINNED = ["AET-POL-005", "AET-POL-007", "AET-POL-010", "WEL-APT-07"]


# For requests only a person can handle (is this bill right, claim status, appeal for me): the Insurance
# Navigators' contact details and the Aetna claims line, so the answer can route the student.
HANDOFF = ["WEL-027-12", "WEL-046-02"]


def retrieve(plan, question, include_curated=True, use_vectors=True, info=None):
    """The top-TOP_K passages for a plan: hand-off contacts first when the student needs a person,
    then search results. This is exactly what the eval scores."""
    by_id = {c["id"]: c for c in corpus()}
    first = [dict(by_id[i], score=0.0, matched_by=["handoff"]) for i in HANDOFF
             if plan.get("needs_person") and i in by_id]
    found = search(plan["search_queries"], original_question=question, k=TOP_K,
                   include_curated=include_curated, use_vectors=use_vectors,
                   language=plan.get("language"), info=info)
    have = {c["id"] for c in first}
    return (first + [c for c in found if c["id"] not in have])[:TOP_K]


def pinned_chunks():
    by_id = {c["id"]: c for c in corpus()}
    return [by_id[i] for i in PINNED if i in by_id]


@dataclass
class Reply:
    text: str
    route: str = "ok"
    plan: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    cited: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)   # seconds per stage
    search_info: dict = field(default_factory=dict)


def cited_sources(text, chunks):
    """Source list for the IDs the answer actually cited (only IDs that were retrieved)."""
    by_id = {c["id"]: c for c in chunks}
    ids = []
    for i in re.findall(r"\[([A-Z]{2,4}-[A-Z0-9-]+)\]", text):
        if i in by_id and i not in ids:
            ids.append(i)
    return [by_id[i] for i in ids]


def ask_stream(question: str, history=None, include_curated=True, use_vectors=True):
    """Yields:
        ("stage", name, detail)   name in: rewrite, search, answer, tool
        ("chunks", list)          the passages given to the answer model (before the text starts)
        ("text", delta)           streamed answer text
        ("done", Reply)
    """
    t0 = time.perf_counter()
    timings = {}
    r = router.route(question)
    if r.kind != "ok":
        yield ("text", r.reply)
        yield ("done", Reply(text=r.reply, route=r.kind, timings={"total": time.perf_counter() - t0}))
        return

    yield ("stage", "rewrite", "Understanding your question")
    t = time.perf_counter()
    plan = llm.rewrite(question, history)
    timings["rewrite"] = time.perf_counter() - t

    # Off-topic or small talk: reply briefly, skip search, Voyage and the answer model.
    # Crisis/emergency flags always win; a missing in_scope field counts as in scope.
    if plan.get("in_scope") is False and not plan.get("crisis") and not plan.get("emergency"):
        reply = plan.get("off_topic_reply") or ("I can only help with the Penn Student Insurance Plan and getting "
                                                "care at Penn. For example: \"Where do I go if Student Health is closed?\"")
        timings["total"] = time.perf_counter() - t0
        yield ("text", reply)
        yield ("done", Reply(text=reply, route=f"off_topic:{plan.get('message_type', 'other')}", plan=plan,
                             timings=timings))
        return

    prefix = ""
    if plan.get("crisis"):  # the model caught something the rules missed: crisis contacts first
        lang = "zh" if looks_chinese(question) else "en"
        prefix = CRISIS_REPLY[lang] + ("\n\n" + CRISIS_REPLY["en"] if lang != "en" else "") + "\n\n---\n\n"
        yield ("text", prefix)

    yield ("stage", "search", "Searching plan documents: " + "; ".join(plan["search_queries"]))
    t = time.perf_counter()
    sinfo = {}
    chunks = retrieve(plan, question, include_curated=include_curated, use_vectors=use_vectors, info=sinfo)
    timings["search"] = time.perf_counter() - t
    # Core rules nearly every cost or "where do I go" answer depends on; always give them to the model.
    have = {c["id"] for c in chunks}
    chunks = chunks + [dict(c, score=0.0, matched_by=["pinned"]) for c in pinned_chunks() if c["id"] not in have]
    if sinfo.get("vectors", "").startswith("unavailable"):
        yield ("stage", "search", "Meaning-based search unavailable (" + sinfo["vectors"][13:] + "); using keyword search")

    yield ("stage", "answer", f"Writing the answer from {len(chunks)} sources")
    yield ("chunks", chunks)   # lets the page number citations while the answer streams
    t = time.perf_counter()
    first_token = None
    text, tool_log = "", []
    for ev in llm.answer_stream(question, plan, chunks, history):
        if ev[0] == "text":
            if first_token is None:
                first_token = time.perf_counter() - t0
            yield ev
        elif ev[0] == "tool":
            label = {"calculate_cost": "Calculating your cost", "lookup_drug": "Looking up the drug"}.get(ev[1], ev[1])
            yield ("stage", "tool", f"{label}: {ev[2].get('service') or ev[2].get('name') or ''}")
        elif ev[0] == "done":
            text, tool_log = ev[1], ev[2]
    timings["answer"] = time.perf_counter() - t
    timings["first_text"] = first_token
    timings["total"] = time.perf_counter() - t0
    yield ("done", Reply(text=prefix + text, route="crisis" if prefix else "ok", plan=plan, chunks=chunks,
                         tool_calls=tool_log, cited=cited_sources(text, chunks), timings=timings,
                         search_info=sinfo))


def ask(question: str, history=None, include_curated=True, use_vectors=True) -> Reply:
    for ev in ask_stream(question, history, include_curated, use_vectors):
        if ev[0] == "done":
            return ev[1]


def source_label(c: dict) -> str:
    page = f", p. {c['page_start']}" if c.get("page_start") else ""
    if c["source_type"] == "curated":
        return f"{c['id']}: team-verified answer (based on {', '.join(c.get('cites') or [])})"
    return f"{c['id']}: {c['source_title']}{page}"
