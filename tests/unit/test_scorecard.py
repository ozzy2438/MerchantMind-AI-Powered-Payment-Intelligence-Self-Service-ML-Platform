"""Unit tests for Responsible AI scorecard."""

from src.governance.scorecard import ResponsibleAIScorecard


def test_scorecard_runs_with_missing_model_path(tmp_path):
    scorecard = ResponsibleAIScorecard()
    report = scorecard.evaluate(
        model_path=str(tmp_path / "missing.joblib"),
        config={
            "name": "test-model",
            "guardrails": {"pii_fields": ["email"], "pii_handling": "mask"},
            "features": {"source": "feature-store"},
            "deployment": {"canary_percentage": 10},
        },
    )
    assert "overall_status" in report
    assert report["warning_count"] >= 0
