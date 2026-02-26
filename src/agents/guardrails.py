"""Security guardrails for the agentic system."""

from __future__ import annotations

import re


class QueryGuardrail:
    """Validates every user query before it reaches any agent."""

    def __init__(self, query: str, merchant_id: str, user_role: str = "merchant"):
        self.query = query
        self.merchant_id = merchant_id
        self.user_role = user_role
        self._validate_query(self.query)

    @staticmethod
    def _validate_query(value: str) -> None:
        QueryGuardrail._block_cross_merchant_access(value)
        QueryGuardrail._block_sql_injection(value)
        QueryGuardrail._block_pii_requests(value)

    @staticmethod
    def _block_cross_merchant_access(value: str) -> None:
        forbidden_patterns = [
            r"all\s+merchants",
            r"other\s+merchant",
            r"every\s+merchant",
            r"merchant_id\s*(!?=|<>)",
        ]
        for pattern in forbidden_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError(
                    "Cross-merchant data access denied. "
                    "You can only query your own merchant data."
                )

    @staticmethod
    def _block_sql_injection(value: str) -> None:
        injection_patterns = [
            r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)",
            r"--\s*$",
            r"\/\*.*\*\/",
            r"UNION\s+SELECT",
            r"OR\s+1\s*=\s*1",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError("Potentially malicious query detected and blocked.")

    @staticmethod
    def _block_pii_requests(value: str) -> None:
        pii_patterns = [
            r"card\s*number",
            r"credit\s*card",
            r"cvv",
            r"customer\s*name",
            r"customer\s*email",
            r"phone\s*number",
            r"address",
            r"date\s*of\s*birth",
        ]
        for pattern in pii_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError(
                    "PII data cannot be retrieved through the query interface. "
                    "This request has been logged for compliance review."
                )


class OutputGuardrail:
    """Validates agent outputs before returning to the user."""

    def validate_sql_output(self, generated_sql: str, merchant_id: str) -> str:
        if merchant_id not in generated_sql:
            raise ValueError(
                "Generated SQL does not contain merchant_id filter. "
                "All queries must be scoped to the requesting merchant."
            )

        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "GRANT"]
        for keyword in dangerous:
            if keyword in generated_sql.upper():
                raise ValueError(f"Dangerous SQL operation blocked: {keyword}")

        return generated_sql

    def validate_response_no_pii(self, response: str) -> str:
        if re.search(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", response):
            return "[REDACTED - PII detected in response]"
        return response
