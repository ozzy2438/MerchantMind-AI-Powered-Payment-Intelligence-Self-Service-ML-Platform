"""Central feature definitions for MerchantMind."""

from dataclasses import dataclass
from typing import List


@dataclass
class FeatureDefinition:
    name: str
    description: str
    computation: str
    window: str
    category: str


BASE_FEATURES: List[FeatureDefinition] = [
    FeatureDefinition(
        name="txn_velocity_1h",
        description="Number of transactions in the last 1 hour for this merchant",
        computation="count",
        window="1H",
        category="velocity",
    ),
    FeatureDefinition(
        name="txn_velocity_24h",
        description="Number of transactions in the last 24 hours",
        computation="count",
        window="24H",
        category="velocity",
    ),
    FeatureDefinition(
        name="avg_amount_7d",
        description="Average transaction amount over 7 days",
        computation="mean",
        window="7D",
        category="amount",
    ),
    FeatureDefinition(
        name="stddev_amount_30d",
        description="Standard deviation of amount over 30 days",
        computation="std",
        window="30D",
        category="amount",
    ),
    FeatureDefinition(
        name="amount_zscore",
        description="Z-score of current amount vs 30-day distribution",
        computation="zscore",
        window="30D",
        category="amount",
    ),
    FeatureDefinition(
        name="max_amount_7d",
        description="Maximum single transaction in 7 days",
        computation="max",
        window="7D",
        category="amount",
    ),
    FeatureDefinition(
        name="unique_customers_1d",
        description="Unique customer count in last 24 hours",
        computation="nunique",
        window="1D",
        category="behavioral",
    ),
    FeatureDefinition(
        name="pct_tap_payments_7d",
        description="Percentage of tap-and-go payments in 7 days",
        computation="ratio",
        window="7D",
        category="behavioral",
    ),
    FeatureDefinition(
        name="hour_of_day",
        description="Hour when transaction occurred (0-23)",
        computation="extract_hour",
        window="none",
        category="temporal",
    ),
    FeatureDefinition(
        name="is_outside_business_hours",
        description="Whether transaction is outside typical hours for category",
        computation="business_hours_check",
        window="none",
        category="temporal",
    ),
]

ADDITIONAL_FEATURES: List[FeatureDefinition] = [
    FeatureDefinition(
        name="pct_debit_7d",
        description="Percentage of debit card transactions in 7 days",
        computation="debit_ratio",
        window="7D",
        category="behavioral",
    ),
    FeatureDefinition(
        name="off_hours_txn_ratio_7d",
        description="Ratio of transactions outside business hours in 7 days",
        computation="off_hours_ratio",
        window="7D",
        category="temporal",
    ),
    FeatureDefinition(
        name="state_avg_amount_deviation",
        description="Merchant avg amount deviation from state average",
        computation="state_deviation",
        window="30D",
        category="contextual",
    ),
]

FEATURE_REGISTRY: List[FeatureDefinition] = BASE_FEATURES + ADDITIONAL_FEATURES
