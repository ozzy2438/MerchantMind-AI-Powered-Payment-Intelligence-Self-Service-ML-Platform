"""SQL Agent: converts natural language to SQL (DuckDB local default)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from src.agents.guardrails import OutputGuardrail
from src.common.duckdb_backend import DuckDBUnavailableError, DuckDBWarehouse

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:  # pragma: no cover
    try:
        from langchain.prompts import ChatPromptTemplate
    except Exception:  # pragma: no cover
        ChatPromptTemplate = None

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None


class SQLAgent:
    SCHEMA_CONTEXT = """
    Available table in local DuckDB warehouse:

    transactions:
    - transaction_id (STRING, PK)
    - merchant_id (STRING, FK)
    - merchant_abn (STRING)
    - customer_id (STRING)  -- anonymized
    - amount_aud (FLOAT)
    - currency (STRING, always "AUD")
    - timestamp (TIMESTAMP)
    - merchant_category (STRING: hospitality/retail/health/services)
    - state (STRING: NSW/VIC/QLD/WA/SA/TAS/ACT/NT)
    - payment_terminal (STRING: tap_and_go/chip_and_pin/eftpos/manual_entry)
    - card_type (STRING: debit/credit)
    - is_fraud (BOOLEAN)
    - is_outside_business_hours (BOOLEAN)

    IMPORTANT RULES:
    1. ALWAYS filter by merchant_id = '{merchant_id}'
    2. NEVER use SELECT *
    3. NEVER access other merchants' data
    4. Use LIMIT for large result sets
    5. Prefer ANSI SQL compatible with DuckDB
    """

    def __init__(self):
        self.backend = os.getenv("QUERY_BACKEND", "duckdb").lower()
        self.output_guardrail = OutputGuardrail()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if ChatOpenAI else None
        self.warehouse = None
        if self.backend == "duckdb":
            try:
                self.warehouse = DuckDBWarehouse()
            except DuckDBUnavailableError:
                self.warehouse = None

    async def execute(self, query: str, merchant_id: str) -> dict[str, Any]:
        generated_sql = await self._generate_sql(query, merchant_id)
        validated_sql = self.output_guardrail.validate_sql_output(generated_sql, merchant_id)
        results = await self._execute_readonly(validated_sql)
        response = await self._format_results(query, results)
        return {
            "response": response,
            "sql_query": validated_sql,
            "raw_data": results,
            "confidence": 0.9,
            "source": "duckdb_local" if self.backend == "duckdb" else "warehouse_readonly",
        }

    async def _generate_sql(self, query: str, merchant_id: str) -> str:
        if not self.llm or not ChatPromptTemplate:
            safe_query = query.lower()
            if "revenue" in safe_query or "sales" in safe_query:
                return (
                    "SELECT DATE(timestamp) AS day, SUM(amount_aud) AS total_revenue "
                    f"FROM transactions WHERE merchant_id = '{merchant_id}' "
                    "GROUP BY DATE(timestamp) ORDER BY day DESC LIMIT 30"
                )
            return (
                "SELECT COUNT(*) AS txn_count "
                f"FROM transactions WHERE merchant_id = '{merchant_id}' LIMIT 1"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a SQL expert for DuckDB. Generate only SQL.\n"
                    + self.SCHEMA_CONTEXT.replace("{merchant_id}", merchant_id),
                ),
                ("human", "{question}"),
            ]
        )
        chain = prompt | self.llm
        sql_response = await chain.ainvoke({"question": query})
        return str(sql_response.content).strip().strip("```sql").strip("```")

    async def _execute_readonly(self, sql: str) -> list[dict[str, Any]]:
        normalized = self._normalize_sql_for_duckdb(sql)
        if not self._is_readonly_query(normalized):
            raise ValueError("Only read-only SELECT queries are allowed.")

        if self.backend == "duckdb" and self.warehouse:
            try:
                return self.warehouse.query_records(normalized)
            except Exception as exc:
                return [{"error": str(exc), "note": "duckdb_query_failed"}]

        return [{"note": "stubbed result set", "rows": 1}]

    async def _format_results(self, question: str, results: list[dict[str, Any]]) -> str:
        if not self.llm or not ChatPromptTemplate:
            return f"Query processed for merchant scope. Result summary: {results[:1]}"

        results_json = json.dumps(results, ensure_ascii=False, default=str)
        format_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Convert SQL results into a concise merchant-friendly response.",
                ),
                ("human", "Question: {question}\nResults: {results_json}"),
            ]
        )
        chain = format_prompt | self.llm
        formatted = await chain.ainvoke({"question": question, "results_json": results_json})
        return str(formatted.content)

    @staticmethod
    def _normalize_sql_for_duckdb(sql: str) -> str:
        normalized = sql.strip().rstrip(";")
        normalized = re.sub(
            r"\bMERCHANTMIND\.CURATED\.TRANSACTIONS\b",
            "transactions",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bTRANSACTIONS\b",
            "transactions",
            normalized,
            flags=re.IGNORECASE,
        )
        return normalized

    @staticmethod
    def _is_readonly_query(sql: str) -> bool:
        candidate = sql.strip().lower()
        if not candidate:
            return False
        if not (candidate.startswith("select") or candidate.startswith("with")):
            return False
        blocked = [" drop ", " delete ", " update ", " insert ", " alter ", " create "]
        return not any(token in f" {candidate} " for token in blocked)
