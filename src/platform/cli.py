"""Self-service ML deployment CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
import yaml


@dataclass
class ModelConfig:
    name: str
    model_type: str
    framework: str
    feature_group: str
    max_inference_latency_ms: int = 200
    data_access_scope: str = "same-merchant-only"
    pii_fields: list | None = None
    pii_handling: str = "mask"
    endpoint_type: str = "serverless"
    auto_scaling: bool = True
    canary_percentage: int = 10
    enable_drift_detection: bool = True
    alert_on_latency_breach: bool = True


@click.group()
def cli() -> None:
    """MerchantMind Platform CLI."""


@cli.command()
@click.option(
    "--template",
    type=click.Choice(["anomaly-detection", "classification", "regression", "llm-service"]),
)
@click.option("--name", prompt="Model name")
def init(template: str, name: str) -> None:
    template_dir = Path(f"src/platform/templates/{template}")
    output_dir = Path(f"models/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in template_dir.glob("**/*"):
        if file.is_file():
            content = file.read_text().replace("{{MODEL_NAME}}", name)
            dest = output_dir / file.relative_to(template_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

    click.echo(f"[OK] Initialized '{name}' from '{template}' template")
    click.echo(f"[PATH] {output_dir}")


@cli.command()
@click.option("--config", type=click.Path(exists=True), required=True)
def validate(config: str) -> bool:
    with open(config) as handle:
        raw_config = yaml.safe_load(handle)

    errors: list[str] = []
    warnings: list[str] = []

    required = ["name", "model_type", "framework", "feature_group"]
    for field in required:
        if field not in raw_config.get("model", {}):
            errors.append(f"Missing required field: model.{field}")

    guardrails = raw_config.get("guardrails", {})
    if guardrails.get("data_access_scope") != "same-merchant-only":
        errors.append("data_access_scope must be 'same-merchant-only' (PCI-DSS requirement)")

    if guardrails.get("max_inference_latency_ms", 999) > 500:
        warnings.append("Inference latency > 500ms may impact user experience")

    pii_fields = guardrails.get("pii_fields", [])
    if pii_fields and guardrails.get("pii_handling") not in ["mask", "hash", "drop"]:
        errors.append("PII fields declared but no valid pii_handling strategy set")

    deployment = raw_config.get("deployment", {})
    if not deployment.get("auto_scaling", False):
        warnings.append("auto_scaling is disabled - not recommended for production")

    if deployment.get("canary_percentage", 0) < 5:
        warnings.append("canary_percentage < 5% - consider higher for safer rollouts")

    if errors:
        click.echo("Validation FAILED:")
        for err in errors:
            click.echo(f"  ERROR: {err}")

    if warnings:
        click.echo("Warnings:")
        for warning in warnings:
            click.echo(f"  WARN: {warning}")

    if not errors:
        click.echo("Validation PASSED")

    return len(errors) == 0


@cli.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--env", type=click.Choice(["staging", "production"]), required=True)
def deploy(config: str, env: str) -> None:
    with open(config) as handle:
        raw_config = yaml.safe_load(handle)

    model_config = ModelConfig(**raw_config["model"], **raw_config.get("guardrails", {}))

    click.echo(f"Deploying {model_config.name} to {env}...")

    click.echo("  [1/5] Running Responsible AI scorecard...")
    scorecard_passed = run_responsible_ai_scorecard(model_config)
    if not scorecard_passed and env == "production":
        click.echo("  FAILED: Responsible AI scorecard did not pass")
        return

    click.echo("  [2/5] Running security scan...")
    run_security_scan(model_config)

    click.echo("  [3/5] Deploying to SageMaker...")
    endpoint = deploy_to_sagemaker(model_config, env)

    click.echo("  [4/5] Setting up monitoring and alerts...")
    setup_monitoring(endpoint, model_config)

    click.echo("  [5/5] Registering in model registry...")
    register_model(model_config, endpoint, env)

    click.echo(f"Deployment completed for {model_config.name} in {env}")


class _Endpoint:
    def __init__(self, name: str):
        self.name = name


def run_responsible_ai_scorecard(model_config: ModelConfig) -> bool:
    _ = model_config
    return True


def run_security_scan(model_config: ModelConfig) -> None:
    _ = model_config


def deploy_to_sagemaker(model_config: ModelConfig, env: str) -> _Endpoint:
    return _Endpoint(name=f"{model_config.name}-{env}")


def setup_monitoring(endpoint: _Endpoint, model_config: ModelConfig) -> None:
    _ = (endpoint, model_config)


def register_model(model_config: ModelConfig, endpoint: _Endpoint, env: str) -> None:
    _ = (model_config, endpoint, env)


if __name__ == "__main__":
    cli()
