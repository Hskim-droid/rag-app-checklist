"""
Minimum viable safety filter for LLM apps.
Drop this file into your project and call check_input() before every LLM call.
"""

import re

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"you\s+are\s+now\s+(?:DAN|evil|unfiltered)",
    r"disregard\s+(all\s+)?(prior|previous|above)",
    r"forget\s+(all\s+)?your\s+(rules|instructions|guidelines)",
    r"system\s*prompt\s*[:=]",
    r"act\s+as\s+(?:if|though)\s+you\s+have\s+no\s+(restrictions|rules|limits)",
    r"\bpretend\b.*\bno\s+(rules|filters|restrictions)\b",
    r"override\s+(safety|content)\s+(filter|policy)",
    r"do\s+not\s+refuse",
    r"respond\s+without\s+(any\s+)?(restrictions|filters|limitations)",
    r"reveal\s+(your|the)\s+(system|initial)\s+(prompt|instructions)",
    r"output\s+(your|the)\s+(system|initial)\s+(prompt|instructions)",
]

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone_us": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "api_key": r"(?:sk|pk|api[_-]?key)[-_]?[a-zA-Z0-9]{20,}",
}


def check_input(text: str) -> dict:
    """Check user input for prompt injection and PII.

    Returns: {"safe": bool, "reason": str | None}
    """
    text_lower = text.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return {"safe": False, "reason": "prompt_injection"}

    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            return {"safe": False, "reason": f"pii_detected:{pii_type}"}

    return {"safe": True, "reason": None}


def mask_pii_in_output(text: str) -> str:
    """Mask any PII the LLM might leak in its response."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
    return text
