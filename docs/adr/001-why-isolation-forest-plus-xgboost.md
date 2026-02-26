# ADR 001: Isolation Forest + XGBoost Ensemble

## Status
Accepted

## Decision
Use an unsupervised plus supervised ensemble for anomaly detection.

## Rationale
Isolation Forest handles sparse anomalies; XGBoost leverages labels for precision.
