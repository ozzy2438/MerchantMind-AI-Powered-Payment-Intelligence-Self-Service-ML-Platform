"""Unit tests for security guardrails."""

import pytest

from src.agents.guardrails import OutputGuardrail, QueryGuardrail


class TestQueryGuardrail:
    def test_allows_valid_merchant_query(self):
        result = QueryGuardrail(query="What was my revenue last week?", merchant_id="M001")
        assert result.merchant_id == "M001"
        assert result.query == "What was my revenue last week?"

    def test_blocks_cross_merchant_access(self):
        with pytest.raises(ValueError, match="Cross-merchant data access denied"):
            QueryGuardrail(query="Show me all merchants revenue", merchant_id="M001")

    def test_blocks_sql_injection_drop(self):
        with pytest.raises(ValueError, match="malicious"):
            QueryGuardrail(query="'; DROP TABLE transactions; --", merchant_id="M001")

    def test_blocks_sql_injection_union(self):
        with pytest.raises(ValueError, match="malicious"):
            QueryGuardrail(query="something UNION SELECT * FROM users", merchant_id="M001")

    def test_blocks_pii_request_card_number(self):
        with pytest.raises(ValueError, match="PII"):
            QueryGuardrail(query="Show me customer card numbers", merchant_id="M001")

    def test_blocks_pii_request_email(self):
        with pytest.raises(ValueError, match="PII"):
            QueryGuardrail(query="What is the customer email for order 123?", merchant_id="M001")

    def test_allows_amount_queries(self):
        result = QueryGuardrail(query="What is my average transaction amount?", merchant_id="M001")
        assert "amount" in result.query


class TestOutputGuardrail:
    def test_blocks_credit_card_in_response(self):
        guardrail = OutputGuardrail()
        response = "The card number is 4111-1111-1111-1111"
        result = guardrail.validate_response_no_pii(response)
        assert "REDACTED" in result

    def test_allows_clean_response(self):
        guardrail = OutputGuardrail()
        response = "Your revenue last week was 12,345 AUD"
        result = guardrail.validate_response_no_pii(response)
        assert result == response

    def test_validates_sql_has_merchant_filter(self):
        guardrail = OutputGuardrail()
        sql = "SELECT SUM(amount) FROM transactions"
        with pytest.raises(ValueError, match="merchant_id filter"):
            guardrail.validate_sql_output(sql, "M001")

    def test_blocks_dangerous_sql_operations(self):
        guardrail = OutputGuardrail()
        sql = "DROP TABLE transactions WHERE merchant_id = 'M001'"
        with pytest.raises(ValueError, match="Dangerous"):
            guardrail.validate_sql_output(sql, "M001")
