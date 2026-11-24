"""Safety router: fixed rules that run BEFORE any AI step (design requirement R5).

If a message matches a crisis or emergency pattern, the bot replies with fixed
contacts at once, in English and the student's language, and does not try to
answer the insurance question. The rewrite step also flags these cases as a
second check, but the rules never depend on the model.
"""
import re
from dataclasses import dataclass

CRISIS = [
    r"\bkill (myself|me)\b", r"\bsuicid", r"\bend (my|it all)\b", r"\bdon'?t want to (be alive|live)\b",
    r"\bwant to die\b", r"\bself[- ]?harm", r"\bhurt(ing)? myself\b", r"\bno reason to live\b",
    r"自杀", r"想死", r"不想活", r"活不下去", r"结束(自己的)?生命", r"伤害自己", r"自残", r"轻生",
]
EMERGENCY = [
    r"\bchest pain", r"\bcan'?t breathe", r"\bdifficulty breathing", r"\bshortness of breath",
    r"\bunconscious", r"\bpassed out", r"\bnot breathing", r"\bseizure", r"\bstroke\b",
    r"\boverdos", r"\bsevere bleeding", r"\bbleeding (a lot|heavily|won'?t stop)",
    r"\banaphyla", r"\bthroat (is )?(closing|swelling)", r"\bheart attack",
    r"胸痛", r"胸口(疼|痛)", r"喘不(过|上)气", r"呼吸困难", r"昏迷", r"晕倒", r"失去意识", r"抽搐",
    r"中风", r"过量", r"大出血", r"流血不止", r"过敏性休克", r"心脏病发作",
]

CRISIS_REPLY = {
    "en": ("If you are thinking about suicide or are in emotional distress, please reach out now:\n"
           "- Call or text **988** (Suicide and Crisis Lifeline, 24/7)\n"
           "- Text **HOME** to **741741** (Crisis Text Line, 24/7)\n"
           "- Penn Student Health and Counseling: **215-746-WELL (9355)**, 24/7\n"
           "- If you are in immediate danger, call **911** (on campus: PennComm **215-573-3333**)."),
    "zh": ("如果你有自杀的想法或情绪非常痛苦，请现在就联系：\n"
           "- 拨打或发短信 **988**（Suicide and Crisis Lifeline，24 小时）\n"
           "- 发短信 **HOME** 到 **741741**（Crisis Text Line，24 小时）\n"
           "- Penn Student Health and Counseling：**215-746-WELL (9355)**，24 小时\n"
           "- 如有紧急危险，请拨 **911**（校内：PennComm **215-573-3333**）。"),
}
EMERGENCY_REPLY = {
    "en": ("This may be a medical emergency. I can't assess symptoms.\n"
           "- Call **911** now (on campus: PennComm **215-573-3333**), or go to the nearest emergency room.\n"
           "- If you're unsure, call Student Health's on-call provider: **215-746-WELL (9355)**, 24/7.\n"
           "Emergency care does not need a referral."),
    "zh": ("这可能是医疗紧急情况，我无法判断症状。\n"
           "- 请立即拨打 **911**（校内：PennComm **215-573-3333**），或前往最近的急诊室 (emergency room)。\n"
           "- 不确定时，请拨打 Student Health 值班医生电话：**215-746-WELL (9355)**，24 小时。\n"
           "急诊 (emergency) 不需要 referral。"),
}


@dataclass
class RouteResult:
    kind: str          # "crisis" | "emergency" | "ok"
    reply: str = ""


def looks_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def route(message: str) -> RouteResult:
    low = message.lower()
    lang_extra = "zh" if looks_chinese(message) else None
    for kind, pats, replies in (("crisis", CRISIS, CRISIS_REPLY), ("emergency", EMERGENCY, EMERGENCY_REPLY)):
        if any(re.search(p, low) for p in pats):
            parts = ([replies[lang_extra]] if lang_extra else []) + [replies["en"]]
            return RouteResult(kind, "\n\n".join(parts))
    return RouteResult("ok")
