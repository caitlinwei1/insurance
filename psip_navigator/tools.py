"""Deterministic tools the answer model calls (design requirement R4: no LLM arithmetic).

calculate_cost  - what a student pays for one service, from data/plan_costs.json
lookup_drug     - find a drug in the 2026 drug guide and price it by tier

Both return plain dicts with the source chunk IDs, so the answer can cite them.
"""
import json
import re
from functools import lru_cache

from .config import FORMULARY, PLAN_COSTS


@lru_cache
def plan():
    return json.loads(PLAN_COSTS.read_text())


@lru_cache
def formulary():
    return [json.loads(l) for l in FORMULARY.open()]


SERVICES = list(json.loads(PLAN_COSTS.read_text())["services"].keys()) if PLAN_COSTS.exists() else []


def calculate_cost(service: str, allowed_amount: float, network: str = "in",
                   has_referral: bool = True, deductible_paid_so_far: float = 0.0,
                   out_of_pocket_so_far: float = 0.0) -> dict:
    """Student's share for one service, following the plan's order of cost sharing:
    deductible first (unless waived for this service), then the copay, then coinsurance
    on the rest (AET-POL-005). Missing a required referral = out-of-network cost sharing
    (AET-POL-010). Capped by the remaining out-of-pocket maximum (AET-POL-008).
    """
    p = plan()
    if service not in p["services"]:
        return {"error": f"unknown service '{service}'", "valid_services": list(p["services"])}
    svc = p["services"][service]
    sources = [svc["source"], "AET-POL-005", "AET-POL-008"]
    notes = []
    level = "in_network" if network == "in" else "out_of_network"
    if network == "in" and not has_referral:
        level = "out_of_network"
        notes.append("No Student Health referral: paid at out-of-network cost sharing.")
        sources.append(p["referral_rule"]["source"])
    rule = svc[level]
    amount = max(0.0, float(allowed_amount))

    ded_total = p["deductible"]["in_network" if level == "in_network" else "out_of_network"]
    ded_left = max(0.0, ded_total - deductible_paid_so_far) if rule["deductible_applies"] else 0.0
    to_deductible = min(amount, ded_left)
    rest = amount - to_deductible
    copay = min(rule["copay"], rest)
    rest -= copay
    coinsurance = rest * (100 - rule["plan_pays_pct"]) / 100
    student = to_deductible + copay + coinsurance

    oop_max = p["out_of_pocket_max"]["in_network" if level == "in_network" else "out_of_network"]["individual"]
    oop_left = max(0.0, oop_max - out_of_pocket_so_far)
    capped = student > oop_left
    student = min(student, oop_left)
    if level == "out_of_network" and service != "emergency_room":
        notes.append("An out-of-network provider may also bill you the difference between its charge and "
                     "the allowed amount (balance billing), except where the No Surprises Act protects you.")
    if svc.get("note"):
        notes.append(svc["note"])
    if svc.get("not_covered"):
        notes.append(f"Not covered: {svc['not_covered']}.")
    return {
        "service": svc["label"], "cost_level": level.replace("_", "-"),
        "allowed_amount": round(amount, 2),
        "breakdown": {"deductible": round(to_deductible, 2), "copay": round(copay, 2),
                      "coinsurance": round(coinsurance, 2)},
        "student_pays": round(student, 2), "plan_pays": round(amount - student, 2),
        "deductible_applies": rule["deductible_applies"],
        "capped_by_out_of_pocket_max": capped, "notes": notes, "sources": sources,
        "assumption": "Uses the allowed (negotiated) amount, not the provider's list price. "
                      "Estimate only; Aetna's EOB is the final word.",
    }


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def lookup_drug(name: str, days_supply: int = 30, network: str = "in", limit: int = 8) -> dict:
    """Find a drug by brand or generic name (all words of the query must appear)."""
    q = _norm(name)
    if not q:
        return {"error": "empty drug name"}
    words = q.split()
    hits = []
    for d in formulary():
        hay = _norm(d["drug"] + " " + (d.get("generic_name") or ""))
        if all(w in hay.split() or w in hay for w in words):
            starts = hay.startswith(q)
            hits.append((0 if starts else 1, len(hay), d))
    hits.sort(key=lambda x: (x[0], x[1]))
    tiers = plan()["prescriptions"]["formulary_tiers"]
    out = []
    for _, _, d in hits[:limit]:
        t = tiers.get(d["tier"], {})
        if days_supply <= 30:
            copay = t.get("copay_30_day")
        elif network == "in":
            copay = t.get("copay_31_90_day", t.get("copay_30_day"))
        else:
            copay = "not covered (more than a 30-day supply out of network)"
        if d["tier"] == "PSP" and isinstance(copay, str):
            copay = (20 if days_supply <= 30 else 60) if d["is_generic"] else (50 if days_supply <= 30 else 150)
        out.append({"drug": d["drug"], "generic": d["is_generic"], "tier": d["tier"],
                    "tier_meaning": d["tier_label"], "requirements": d["requirements"] or None,
                    "copay": copay, "page": d["page"], "id": d["id"]})
    # For brand matches, also show the generic version's listings (often the covered option)
    generics = []
    seen = {m["id"] for m in out}
    for g in {d["generic_name"] for _, _, d in hits[:limit] if not d["is_generic"] and d.get("generic_name")}:
        gq = _norm(g)
        for d in formulary():
            if d["is_generic"] and _norm(d["drug"]).startswith(gq) and d["id"] not in seen:
                t = tiers.get(d["tier"], {})
                generics.append({"drug": d["drug"], "tier": d["tier"], "tier_meaning": d["tier_label"],
                                 "requirements": d["requirements"] or None,
                                 "copay": t.get("copay_30_day") if days_supply <= 30 else t.get("copay_31_90_day"),
                                 "page": d["page"], "id": d["id"]})
                seen.add(d["id"])
    return {
        "query": name, "matches": out, "total_matches": len(hits),
        "generic_versions": generics[:limit],
        "notes": ["Copays need no deductible.",
                  "If a drug is listed more than once with different tiers, confirm with the pharmacy or Aetna (800-841-5374).",
                  "PA = prior authorization, QL = quantity limit, ST = step therapy. The formulary can change during the year."],
        "sources": ["AET-RX", "AET-PDB-062", "AET-SBC-003"],
    }


# Tool schemas for the Claude Messages API
TOOL_SPECS = [
    {
        "name": "calculate_cost",
        "description": "Compute what a PSIP student pays for ONE medical service using the plan's real rules "
                       "(deductible, copay, coinsurance, referral rule, out-of-pocket max). Use this for EVERY "
                       "dollar figure about a medical service; never do the arithmetic yourself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "enum": SERVICES},
                "allowed_amount": {"type": "number", "description": "Allowed/negotiated amount in USD. If the student gave a bill total, use it and say it is an estimate."},
                "network": {"type": "string", "enum": ["in", "out"]},
                "has_referral": {"type": "boolean", "description": "False if a Student Health referral was required but not obtained."},
                "deductible_paid_so_far": {"type": "number"},
                "out_of_pocket_so_far": {"type": "number"},
            },
            "required": ["service", "allowed_amount"],
        },
    },
    {
        "name": "lookup_drug",
        "description": "Look up a prescription drug in the 2026 Aetna Student Health drug guide by brand or generic "
                       "name. Returns tier, requirements (PA/QL/ST) and the copay.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Drug name in English, e.g. 'sertraline' or 'Adderall'."},
                "days_supply": {"type": "integer", "description": "30 (default) or up to 90."},
                "network": {"type": "string", "enum": ["in", "out"]},
            },
            "required": ["name"],
        },
    },
]

TOOLS = {"calculate_cost": calculate_cost, "lookup_drug": lookup_drug}
