"""Unit tests for PII masking helpers."""

from src.governance.pii_masking import mask_email, redact_pii


def test_mask_email():
    assert mask_email("john.doe@example.com").endswith("@example.com")


def test_redact_pii():
    text = "Card 4111-1111-1111-1111 email john@example.com"
    redacted = redact_pii(text)
    assert "REDACTED_CARD" in redacted
    assert "REDACTED_EMAIL" in redacted
