"""Data contract tests for transaction schema."""


class TestTransactionSchema:
    REQUIRED_COLUMNS = [
        "transaction_id",
        "merchant_id",
        "amount_aud",
        "currency",
        "timestamp",
        "merchant_category",
        "state",
        "is_fraud",
    ]

    VALID_CATEGORIES = ["hospitality", "retail", "health", "services"]
    VALID_STATES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]

    def test_all_required_columns_present(self, curated_sample):
        for col in self.REQUIRED_COLUMNS:
            assert col in curated_sample.columns, f"Missing column: {col}"

    def test_no_null_transaction_ids(self, curated_sample):
        assert curated_sample["transaction_id"].notna().all()

    def test_amounts_are_positive(self, curated_sample):
        assert (curated_sample["amount_aud"] > 0).all()

    def test_valid_categories_only(self, curated_sample):
        invalid = set(curated_sample["merchant_category"].unique()) - set(self.VALID_CATEGORIES)
        assert len(invalid) == 0, f"Invalid categories found: {invalid}"

    def test_currency_is_aud(self, curated_sample):
        assert (curated_sample["currency"] == "AUD").all()
