"""Data quality checks using Great Expectations."""

from __future__ import annotations

try:
    import great_expectations as gx
except Exception:  # pragma: no cover
    gx = None


def build_transaction_expectations():
    if gx is None:
        raise RuntimeError("great_expectations is not installed")

    context = gx.get_context()
    suite = context.add_expectation_suite("transactions_quality")

    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="transaction_id"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="merchant_id"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="amount_aud"))

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="amount_aud", min_value=0.01, max_value=500_000.00
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="merchant_category",
            value_set=["hospitality", "retail", "health", "services"],
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="state", value_set=["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]
        )
    )
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="transaction_id"))

    return suite
