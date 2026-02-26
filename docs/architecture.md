# MerchantMind Architecture

MerchantMind is organized into six layers:
1. Data ingestion
2. Feature and ML training
3. Agentic AI assistant
4. API and runtime
5. Infrastructure and CI/CD
6. Governance, monitoring, observability

This repository uses a canonical multi-source ingestion contract and keeps downstream layers loosely coupled to the data source.
