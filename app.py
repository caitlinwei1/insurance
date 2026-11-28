"""PSIP Navigator chat page.

Run:  streamlit run app.py
"""
import os
import time
from collections import deque

import streamlit as st

st.set_page_config(page_title="PSIP Navigator", page_icon="🩺", layout="centered")

# On Streamlit Community Cloud the API keys live in the app's Secrets; copy them into the
# environment, where psip_navigator.config reads them. Locally, .env is used instead.
try:
    for _k in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "VOYAGE_BASE_URL", "VECTOR_MODE",
               "PER_SESSION_PER_HOUR", "ALL_USERS_PER_HOUR"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # no secrets file (local run): fine
    pass

from psip_navigator.bot import ask_stream, source_label  # noqa: E402

# Usage limits, so a shared public link can't run up the API bill.
PER_SESSION_PER_HOUR = int(os.environ.get("PER_SESSION_PER_HOUR", 20))
ALL_USERS_PER_HOUR = int(os.environ.get("ALL_USERS_PER_HOUR", 200))
MAX_QUESTION_CHARS = 1000


@st.cache_resource
def _global_log():
    return deque()   # times of recent questions, shared by every visitor of this app


def _allowed():
    now = time.time()
    g = _global_log()
    while g and now - g[0] > 3600:
        g.popleft()
    mine = [t for t in st.session_state.setdefault("asked_at", []) if now - t < 3600]
    st.session_state.asked_at = mine
    if len(mine) >= PER_SESSION_PER_HOUR:
        return False, f"You've asked {PER_SESSION_PER_HOUR} questions in the last hour. Please try again a bit later."
    if len(g) >= ALL_USERS_PER_HOUR:
        return False, "PSIP Navigator is busy right now. Please try again in a few minutes."
    g.append(now)
    st.session_state.asked_at.append(now)
    return True, ""


with st.sidebar:
    st.header("PSIP Navigator")
    st.caption("Answers about the Penn Student Insurance Plan (Aetna Student Health, 2026-2027) and where to get "
               "care at Penn. Supports English or Chinese.")
    include_curated = st.toggle("Use team-verified answers", value=True,
                                help="Off = answer only from Aetna documents and official web pages.")
    show_debug = st.toggle("Show search details", value=False)
    st.divider()
    st.markdown("**Emergency:** call 911 (on campus: PennComm 215-573-3333)  \n"
                "**Student Health, 24/7:** 215-746-WELL (9355)  \n"
                "**Insurance Navigators:** 215-746-4200  \n"
                "**Aetna Student Health:** 800-841-5374")
    st.caption("Information only, not medical advice. Aetna's EOB and the plan documents are the final word.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("PSIP Navigator")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                st.markdown("\n".join(m["sources"]))

if prompt := st.chat_input("Ask in English or Chinese…", max_chars=MAX_QUESTION_CHARS):
    ok, why = _allowed()
    if not ok:
        st.warning(why)
        st.stop()
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        status = st.status("Understanding your question…", expanded=False)
        box = st.empty()
        text, rep = "", None
        try:
            for ev in ask_stream(prompt, history=history, include_curated=include_curated):
                if ev[0] == "stage":
                    status.update(label=ev[2] + "…")
                    status.write(ev[2])
                elif ev[0] == "text":
                    text += ev[1]
                    box.markdown(text + "▌")
                elif ev[0] == "done":
                    rep = ev[1]
        except RuntimeError as e:  # missing key, outdated index
            status.update(label="Something went wrong", state="error")
            st.error(str(e))
            st.stop()
        box.markdown(rep.text)
        tm = rep.timings
        status.update(label=f"Done in {tm['total']:.1f}s", state="complete", expanded=False)
        sources = []
        for c in rep.cited:
            link = f" ([link]({c['url']}))" if c.get("url") else ""
            sources.append(f"- {source_label(c)}{link}")
        if sources:
            with st.expander("Sources"):
                st.markdown("\n".join(sources))
        if show_debug and rep.route == "ok":
            with st.expander("Search details"):
                st.json({"timings_seconds": {k: round(v, 2) for k, v in tm.items() if v is not None},
                         "vector_search": rep.search_info.get("vectors"),
                         "plan": rep.plan, "retrieved": [(c["id"], c["score"], c["matched_by"]) for c in rep.chunks],
                         "tool_calls": rep.tool_calls})
    st.session_state.messages.append({"role": "assistant", "content": rep.text, "sources": sources})
