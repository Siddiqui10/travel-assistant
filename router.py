"""
router.py
----------
This is the "Agent Workflow" decision layer mentioned in the problem statement:
given a raw user message, decide whether it is a plain QA question, an itinerary
request, a budget request, or a destination comparison - and dispatch to the
right agent(s) in agents.py.

Kept as a lightweight rule + LLM hybrid classifier so it's fast and cheap:
first try obvious keyword rules, fall back to an LLM call only if ambiguous.
"""

import re
from huggingface_hub import InferenceClient

import config
import agents

INTENTS = ["itinerary", "compare", "budget", "weather", "qa"]

_keyword_rules = [
    ("itinerary", [r"\bitinerary\b", r"\bplan.*trip\b", r"\bday[- ]?wise\b", r"\bschedule\b"]),
    ("compare", [r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b", r"\bbetter\b.*\bor\b"]),
    ("budget", [r"\bbudget\b", r"\bcost\b", r"\bexpense\b", r"\bhow much\b", r"\bprice\b"]),
    ("weather", [r"\bweather\b", r"\btemperature\b", r"\bclimate\b", r"\braining?\b"]),
]


def _rule_based_intent(query: str):
    q = query.lower()
    for intent, patterns in _keyword_rules:
        if any(re.search(p, q) for p in patterns):
            return intent
    return None


def _llm_intent(query: str) -> str:
    """Fallback classifier using the Hugging Face Inference Providers API directly
    (cheap, single call, no agent/crew overhead)."""
    client = InferenceClient(provider=config.HF_PROVIDER, api_key=config.HF_API_KEY)
    prompt = (
        "Classify the travel query into exactly one label from this list: "
        "itinerary, compare, budget, weather, qa.\n"
        "Reply with only the label, nothing else.\n\n"
        f"Query: {query}"
    )
    resp = client.chat.completions.create(
        model=config.HF_MODEL_REPO,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
        max_tokens=10,
    )
    label = resp.choices[0].message.content.strip().lower()
    return label if label in INTENTS else "qa"


def classify_intent(query: str) -> str:
    intent = _rule_based_intent(query)
    if intent:
        return intent
    try:
        return _llm_intent(query)
    except Exception:
        return "qa"  # safe fallback if the LLM call itself fails


def handle_query(query: str) -> dict:
    """
    Main entry point used by the UI.
    Returns {"intent": ..., "answer": ...}
    """
    intent = classify_intent(query)

    if intent == "itinerary":
        answer = agents.run_itinerary(query)
    elif intent == "compare":
        answer = agents.run_comparison(query)
    elif intent == "budget":
        answer = agents.run_budget(query)
    elif intent == "weather":
        answer = agents.run_weather(query)
    else:
        answer = agents.run_qa(query)

    return {"intent": intent, "answer": answer}
