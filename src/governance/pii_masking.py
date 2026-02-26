"""PII masking utilities."""

from __future__ import annotations

import re


def mask_email(value: str) -> str:
    parts = value.split("@")
    if len(parts) != 2:
        return value
    local, domain = parts
    if len(local) <= 2:
        masked_local = "*" * len(local)
    else:
        masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
    return f"{masked_local}@{domain}"


def redact_pii(text: str) -> str:
    masked = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[REDACTED_CARD]", text)
    masked = re.sub(r"[\w\.-]+@[\w\.-]+", "[REDACTED_EMAIL]", masked)
    return masked
