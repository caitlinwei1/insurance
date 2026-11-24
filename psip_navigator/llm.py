"""The two Claude steps: rewrite (question -> English search plan) and answer (cited reply)."""
import json
from functools import lru_cache

from .config import ANSWER_MODEL, REWRITE_MODEL, key
from .tools import TOOL_SPECS, TOOLS


@lru_cache
def client():
    import anthropic
    return anthropic.Anthropic(api_key=key("ANTHROPIC_API_KEY"))


def _text(resp):
    return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------- rewrite
REWRITE_SYSTEM = """You prepare search queries for PSIP Navigator, a chatbot that answers Penn international \
students' questions about the Penn Student Insurance Plan (PSIP, Aetna Student Health) and Penn's health care rules.

The source documents are in English and use plan language: "referral from Student Health", "in-network", \
"copayment", "policy year deductible", "outpatient lab services", "Explanation of Benefits (EOB)", \
"Clinical Fee", "out of country claims", "prior authorization", "formulary", "surprise bill".

Given the student's latest message (any language) and recent conversation, call submit_search_plan:
- language: the language the student wrote in (ISO code, e.g. "en", "zh", "ko", "hi", "es").
- english_question: a faithful English translation of what the student is asking, with context from the \
conversation filled in.
- search_queries: 1-3 short English queries in plan language, one per distinct thing to look up. Map everyday \
words to plan terms (e.g. "school clinic" -> "Student Health", "blood test" -> "outpatient lab services", \
"bill from insurance" -> "Explanation of Benefits").
- drug_names: English names of any medicines mentioned (brand or generic), else [].
- crisis / emergency: true if the student may be in crisis (suicide, self-harm) or describes an emergency symptom.
- in_scope: true if the message is about health insurance, getting medical or mental-health care, costs, bills, EOBs, claims, prescriptions, insurance terms, Penn health services, or anything else PSIP Navigator could help with, including follow-ups in the conversation. false for unrelated requests (weather, homework, coding, other topics) and for greetings or small talk with no question. When unsure, use true. Crisis or emergency is always true.
- message_type: "question" (anything in scope), "greeting", "thanks", "feedback" (complaints, frustration, \
insults, swearing, "this is wrong"), "unrelated_request" (help with something outside health insurance and care), \
or "other" (nonsense, tests like "asdf").
- off_topic_reply: only when in_scope is false, otherwise "". Write it in the student's language, 1-3 short \
sentences, like a friendly human assistant, and respond to what they actually said:
  * greeting: greet back warmly and mention 2-3 things you can help with (e.g. where to go when sick, what a \
visit or prescription costs, reading a bill or EOB).
  * thanks: a brief "you're welcome" and an offer to help with anything else about their insurance.
  * feedback: stay calm and kind, never lecture or scold, don't over-apologize. Ask what went wrong or what \
they need. If they seem unhappy with an answer, offer a person: Penn Insurance Navigators, 215-746-4200.
  * unrelated_request: say in one clause you can't help with that topic, then suggest ONE example question \
that fits the conversation so far.
  * other: briefly ask what they'd like to know about their insurance or getting care.
  Vary your wording: do not reuse the sentences or the example question from your earlier replies in the \
conversation. Don't start every reply with "抱歉"/"Sorry".
Do not answer the question."""

PLAN_TOOL = {
    "name": "submit_search_plan",
    "description": "Return the search plan for the student's message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "language": {"type": "string"},
            "english_question": {"type": "string"},
            "search_queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
            "drug_names": {"type": "array", "items": {"type": "string"}},
            "crisis": {"type": "boolean"},
            "emergency": {"type": "boolean"},
            "in_scope": {"type": "boolean"},
            "message_type": {"type": "string", "enum": ["question", "greeting", "thanks", "feedback",
                                                        "unrelated_request", "other"]},
            "off_topic_reply": {"type": "string"},
        },
        "required": ["language", "english_question", "search_queries", "drug_names", "crisis", "emergency",
                     "in_scope", "message_type", "off_topic_reply"],
    },
}


def rewrite(question: str, history=None) -> dict:
    convo = ""
    for turn in (history or [])[-4:]:
        convo += f"{turn['role']}: {turn['content'][:600]}\n"
    msg = (f"Recent conversation:\n{convo}\n" if convo else "") + f"Student's latest message:\n{question}"
    resp = client().messages.create(
        model=REWRITE_MODEL, max_tokens=500, system=REWRITE_SYSTEM,
        tools=[PLAN_TOOL], tool_choice={"type": "tool", "name": "submit_search_plan"},
        messages=[{"role": "user", "content": msg}])
    block = next(b for b in resp.content if b.type == "tool_use")
    return dict(block.input)


# ---------------------------------------------------------------- answer
ANSWER_SYSTEM = """You are PSIP Navigator, a chatbot for international students at the University of Pennsylvania \
who are on the Penn Student Insurance Plan (PSIP, Aetna Student Health, plan year Aug 1, 2026 - Jul 31, 2027).

Answer ONLY from the SOURCES provided below and the results of your tools. Rules:
1. Reply in the student's language: {language}. Keep insurance and Penn terms in English in parentheses the \
first time you use them, e.g. "免赔额 (deductible)", "转诊 (referral)", so the student recognizes them on bills \
and at the clinic. Never translate phone numbers, addresses, URLs, drug names or dollar amounts.
2. Cite every fact with the source ID in square brackets, e.g. [WEL-027-06] or [AET-PDB-036]. For the single most \
important rule, quote the English source sentence briefly. Do not cite IDs that are not in the sources.
3. Every dollar figure about what the student will pay must come from calculate_cost or lookup_drug. Never do \
arithmetic yourself. Say what you assumed (in-network, referral, deductible already met or not). Copays and the \
deductible: the plan applies the deductible FIRST unless it is waived for that service.
4. If you need several tool results, request them all in the same turn (it is faster). For medicines, call lookup_drug. If the drug is listed more than once with different tiers, say so.
5. If the sources don't answer the question, say so plainly and refer the student to the Penn Insurance \
Navigators (215-746-4200, wel-shsinsur@pobox.upenn.edu) or Aetna Student Health (800-841-5374). Never guess.
6. Out of scope, hand off instead of answering: whether a specific bill is correct (explain the EOB fields and the \
relevant rule, then refer to the Insurance Navigators); claim denials, appeals and disputes (Insurance \
Navigators); a student's own claim status (Aetna 800-841-5374 or the Aetna Health app); whether a specific \
doctor is in network (Aetna provider search, don't guess); medical advice or "is this serious?" (911, or \
Student Health 24/7 at 215-746-WELL).
7. Be short and practical: the answer first in 1-2 sentences, then what to do next. Use a short list only when \
there are steps.
8. Some sources are "Team-verified answer (test set)"; use them like any source and cite their ID.

SOURCES:
{sources}"""


def format_sources(chunks):
    parts = []
    for c in chunks:
        loc = f", p.{c['page_start']}" if c.get("page_start") else ""
        parts.append(f"<source id=\"{c['id']}\" title=\"{c['heading_path']}{loc}\">\n{c['text']}\n</source>")
    return "\n".join(parts)


FALLBACK = "Sorry, I couldn't finish that. Please contact the Insurance Navigators at 215-746-4200."


def answer_stream(question: str, plan: dict, chunks, history=None, max_tool_rounds=4):
    """Stream the answer. Yields events:
         ("text", delta)            a piece of the reply, as Claude writes it
         ("tool", name, input)      a tool is about to run (for the progress display)
         ("done", full_text, tool_log)
    Tool rounds: Claude may stop to call calculate_cost / lookup_drug; we run them and continue.
    """
    system = ANSWER_SYSTEM.format(language=plan.get("language", "en"), sources=format_sources(chunks))
    messages = [{"role": t["role"], "content": t["content"]} for t in (history or [])[-6:]]
    messages.append({"role": "user", "content": question})
    tool_log, parts = [], []
    for _ in range(max_tool_rounds + 1):
        with client().messages.stream(model=ANSWER_MODEL, max_tokens=1500, system=system,
                                      tools=TOOL_SPECS, messages=messages) as s:
            for delta in s.text_stream:
                parts.append(delta)
                yield ("text", delta)
            resp = s.get_final_message()
        if resp.stop_reason != "tool_use":
            yield ("done", "".join(parts), tool_log)
            return
        if parts and not parts[-1].endswith("\n"):  # text written before a tool call
            parts.append("\n\n")
            yield ("text", "\n\n")
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            yield ("tool", b.name, dict(b.input))
            try:
                out = TOOLS[b.name](**b.input)
            except Exception as e:  # bad arguments: tell the model, let it retry
                out = {"error": str(e)}
            tool_log.append({"tool": b.name, "input": b.input, "output": out})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
    yield ("text", FALLBACK)
    yield ("done", "".join(parts) + FALLBACK, tool_log)


def answer(question: str, plan: dict, chunks, history=None, max_tool_rounds=4):
    """Non-streaming wrapper (used by the eval)."""
    for ev in answer_stream(question, plan, chunks, history, max_tool_rounds):
        if ev[0] == "done":
            return ev[1], ev[2]
