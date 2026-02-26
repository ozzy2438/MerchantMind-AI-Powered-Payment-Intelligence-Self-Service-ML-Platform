"""Main agent orchestrator that routes merchant queries."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None

try:
    import structlog
except Exception:  # pragma: no cover
    import logging

    class _CompatLogger:
        def __init__(self, name: str):
            self._logger = logging.getLogger(name)

        def info(self, event: str, **kwargs):
            self._logger.info("%s %s", event, kwargs if kwargs else "")

        def warning(self, event: str, **kwargs):
            self._logger.warning("%s %s", event, kwargs if kwargs else "")

        def error(self, event: str, **kwargs):
            self._logger.error("%s %s", event, kwargs if kwargs else "")

    class _StructlogFallback:
        @staticmethod
        def get_logger():
            return _CompatLogger("merchantmind")

    structlog = _StructlogFallback()  # type: ignore[assignment]

from src.agents.anomaly_agent import AnomalyAgent
from src.agents.guardrails import OutputGuardrail, QueryGuardrail
from src.agents.rag_agent import RAGAgent
from src.agents.sql_agent import SQLAgent
from src.observability.logging import AuditLogger

logger = structlog.get_logger()


class MerchantMindRouter:
    INTENT_CATEGORIES = {
        "analytics": {
            "keywords": [
                "revenue",
                "sales",
                "average",
                "trend",
                "compare",
                "total",
                "count",
                "growth",
                "how much",
                "how many",
            ],
            "agent": "sql",
        },
        "anomaly": {
            "keywords": [
                "suspicious",
                "fraud",
                "anomaly",
                "unusual",
                "weird",
                "alert",
                "risk",
                "investigate",
                "flagged",
            ],
            "agent": "anomaly",
        },
        "knowledge": {
            "keywords": [
                "what is",
                "how does",
                "explain",
                "best practice",
                "recommend",
                "terminal",
                "eftpos",
                "integration",
            ],
            "agent": "rag",
        },
    }

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if ChatOpenAI else None
        self.sql_agent = SQLAgent()
        self.anomaly_agent = AnomalyAgent()
        self.rag_agent = RAGAgent()
        self.output_guardrail = OutputGuardrail()
        self.audit_logger = AuditLogger()

    async def handle_query(self, query: str, merchant_id: str) -> dict[str, Any]:
        request_id = f"req_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        try:
            validated = QueryGuardrail(query=query, merchant_id=merchant_id)
        except ValueError as exc:
            await self.audit_logger.log_blocked_query(
                query=query,
                merchant_id=merchant_id,
                reason=str(exc),
            )
            return {"response": str(exc), "status": "blocked", "request_id": request_id}

        intent = await self._classify_intent(validated.query)

        logger.info("query_routed", merchant_id=merchant_id, intent=intent, request_id=request_id)

        agent_map = {"sql": self.sql_agent, "anomaly": self.anomaly_agent, "rag": self.rag_agent}
        agent = agent_map.get(intent, self.rag_agent)
        result = await agent.execute(query=validated.query, merchant_id=merchant_id)

        result["response"] = self.output_guardrail.validate_response_no_pii(result["response"])

        await self.audit_logger.log_agent_interaction(
            request_id=request_id,
            merchant_id=merchant_id,
            query=validated.query,
            intent=intent,
            response=result["response"],
            confidence=float(result.get("confidence", 0.0)),
        )

        result["request_id"] = request_id
        result["intent"] = intent
        return result

    async def _classify_intent(self, query: str) -> str:
        query_lower = query.lower()
        for intent, config in self.INTENT_CATEGORIES.items():
            if any(keyword in query_lower for keyword in config["keywords"]):
                return config["agent"]

        if not self.llm:
            return "rag"

        classification_prompt = (
            "Classify this merchant query into one category: sql, anomaly, rag. "
            "Return only the category. Query: "
            + query
        )
        response = await self.llm.ainvoke(classification_prompt)
        value = str(response.content).strip().strip('"').lower()
        if value not in {"sql", "anomaly", "rag"}:
            return "rag"
        return value
