"""The strict output schema for the behavioural (format-adherence) experiment.

This is the thing fine-tuning is supposed to be good at and RAG cannot do:
guarantee a rigid output shape. The eval rewards EXACT adherence, so the common
prompt-only failure modes (markdown code fences, prose around the JSON, missing
or extra keys, a bad confidence value) all count as misses.
"""
from __future__ import annotations

import json

REQUIRED_KEYS = ["answer", "figure", "unit", "period", "section", "confidence"]
ALLOWED_CONFIDENCE = {"high", "medium", "low"}

FORMAT_INSTRUCTION = (
    "Answer using ONLY the provided context. Reply with a SINGLE JSON object and "
    "nothing else: no prose, no markdown, no code fences. The object must have "
    "exactly these keys: answer (string), figure (string or null), unit (string "
    "or null), period (string or null), section (string or null), confidence "
    '(one of "high", "medium", "low"). If the answer is not in the context, set '
    'answer to "Not in document", the other fields to null, and confidence to "low".'
)


def validate(raw: str) -> tuple[bool, str]:
    """Strict: the raw model output must itself be the JSON object."""
    s = (raw or "").strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False, "output is not a bare JSON object (prose or fences)"
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    if not isinstance(obj, dict):
        return False, "not an object"
    if [k for k in REQUIRED_KEYS if k not in obj]:
        return False, f"missing keys: {[k for k in REQUIRED_KEYS if k not in obj]}"
    if [k for k in obj if k not in REQUIRED_KEYS]:
        return False, f"unexpected keys: {[k for k in obj if k not in REQUIRED_KEYS]}"
    if obj.get("confidence") not in ALLOWED_CONFIDENCE:
        return False, f"bad confidence: {obj.get('confidence')!r}"
    return True, "ok"


def to_answer_text(raw: str) -> str:
    """Flatten a schema object (or raw text) into prose for the correctness judge."""
    try:
        obj = json.loads((raw or "").strip())
    except json.JSONDecodeError:
        return raw or ""
    bits = [str(obj.get("answer", "")).strip()]
    fig, unit, period = obj.get("figure"), obj.get("unit"), obj.get("period")
    if fig:
        bits.append(f"({fig}{(' ' + unit) if unit else ''}{(', ' + period) if period else ''})")
    if obj.get("section"):
        bits.append(f"[{obj['section']}]")
    return " ".join(b for b in bits if b)
