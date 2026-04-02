"""
test_generator.py
=================
Unit tests for the synthetic finance data generator.

Tests verify:
1. Data structure and schema (30 columns, correct dtypes, no nulls, unique IDs)
2. Distribution shapes and value ranges for every column
3. Logical relationships between variables (correlations, conditional flags)
4. Chunking behavior (id_offset, variable chunk sizes)
5. Constants consistency (PROBABILITIES sum, dict key coverage)

All tests operate exclusively on ``_generate_chunk``, which is the only public
generation function. ``generate_dataset`` is an I/O wrapper and is not tested here.

Author: Geronimo Martinez Nuñez
Course: ALDA - Algorithm Analysis & Data Analysis
"""

import pytest
import numpy as np
import pandas as pd

from src.generator import (
    _generate_chunk,
    COUNTRIES,
    PROBABILITIES,
    COUNTRY_ISO,
    COUNTRY_MEDIAN_SALARY_USD,
    COUNTRY_MIN_WAGE_USD,
    EPOCH,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df_small() -> pd.DataFrame:
    """200-row chunk used by most tests. Module-scoped so it is generated once."""
    return _generate_chunk(200)


@pytest.fixture(scope="module")
def df_large() -> pd.DataFrame:
    """1 000-row chunk used for distribution / correlation tests."""
    return _generate_chunk(1_000)


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify module-level constants are internally consistent."""

    def test_probabilities_sum_to_one(self):
        """PROBABILITIES must sum to exactly 1.0 (within float tolerance)."""
        assert abs(sum(PROBABILITIES) - 1.0) < 1e-9

    def test_probabilities_length_matches_countries(self):
        """There must be one probability per country."""
        assert len(PROBABILITIES) == len(COUNTRIES)

    def test_all_countries_have_iso_code(self):
        """Every country in COUNTRIES must have an entry in COUNTRY_ISO."""
        for country in COUNTRIES:
            assert country in COUNTRY_ISO, f"Missing ISO code for {country}"

    def test_all_countries_have_median_salary(self):
        """Every country must have a median salary defined."""
        for country in COUNTRIES:
            assert country in COUNTRY_MEDIAN_SALARY_USD, f"Missing salary for {country}"

    def test_all_countries_have_min_wage(self):
        """Every country must have a minimum wage defined."""
        for country in COUNTRIES:
            assert country in COUNTRY_MIN_WAGE_USD, f"Missing min wage for {country}"

    def test_median_salaries_positive(self):
        """All median salaries must be strictly positive."""
        for country, salary in COUNTRY_MEDIAN_SALARY_USD.items():
            assert salary > 0, f"Non-positive median salary for {country}"

    def test_min_wages_non_negative(self):
        """All minimum wages must be non-negative."""
        for country, wage in COUNTRY_MIN_WAGE_USD.items():
            assert wage >= 0, f"Negative min wage for {country}"

    def test_epoch_is_year_2000(self):
        """Reference epoch must be 2000-01-01."""
        from datetime import date
        assert EPOCH == date(2000, 1, 1)


# ---------------------------------------------------------------------------
# 2. Data structure & schema
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "client_id", "age", "country_iso",
    "monthly_salary_usd", "annual_bonus_usd", "monthly_expenses_usd",
    "debt_to_income", "savings_rate", "net_worth_usd", "industry_id",
    "account_balance_usd", "account_type_id", "join_days",
    "num_products", "is_premium_client",
    "monthly_transactions", "avg_transaction_amount_usd",
    "failed_transactions", "international_txn_pct",
    "preferred_channel_id", "last_login_days_ago",
    "credit_score", "num_loans", "loan_default_risk",
    "credit_card_limit_usd", "payment_on_time_pct",
    "has_investments", "investment_portfolio_usd",
    "monthly_investment_usd", "has_pension",
]


class TestSchema:
    """Verify column count, names, dtypes, and nullability."""

    def test_returns_dataframe(self, df_small):
        assert isinstance(df_small, pd.DataFrame)

    def test_exact_column_count(self, df_small):
        assert len(df_small.columns) == 30, (
            f"Expected 30 columns, got {len(df_small.columns)}"
        )

    def test_all_expected_columns_present(self, df_small):
        missing = [c for c in EXPECTED_COLUMNS if c not in df_small.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_extra_columns(self, df_small):
        extra = [c for c in df_small.columns if c not in EXPECTED_COLUMNS]
        assert not extra, f"Unexpected extra columns: {extra}"

    def test_no_null_values(self, df_small):
        null_counts = df_small.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].index.tolist()
        assert not cols_with_nulls, f"Columns with nulls: {cols_with_nulls}"

    def test_row_count_matches_n(self):
        for n in [1, 10, 50, 200, 500]:
            df = _generate_chunk(n)
            assert len(df) == n, f"Expected {n} rows, got {len(df)}"

    def test_integer_columns_dtype(self, df_small):
        int_cols = [
            "client_id", "age", "monthly_salary_usd", "annual_bonus_usd",
            "monthly_expenses_usd", "net_worth_usd", "industry_id",
            "account_balance_usd", "account_type_id", "join_days",
            "num_products", "is_premium_client", "monthly_transactions",
            "avg_transaction_amount_usd", "failed_transactions",
            "preferred_channel_id", "last_login_days_ago", "credit_score",
            "num_loans", "credit_card_limit_usd", "has_investments",
            "investment_portfolio_usd", "monthly_investment_usd", "has_pension",
        ]
        for col in int_cols:
            assert np.issubdtype(df_small[col].dtype, np.integer), (
                f"Column '{col}' expected integer dtype, got {df_small[col].dtype}"
            )

    def test_float_columns_dtype(self, df_small):
        float_cols = [
            "debt_to_income", "savings_rate", "international_txn_pct",
            "loan_default_risk", "payment_on_time_pct",
        ]
        for col in float_cols:
            assert np.issubdtype(df_small[col].dtype, np.floating), (
                f"Column '{col}' expected float dtype, got {df_small[col].dtype}"
            )


# ---------------------------------------------------------------------------
# 3. Identity columns
# ---------------------------------------------------------------------------

class TestIdentityColumns:
    """Verify client_id, age, and country_iso."""

    def test_client_id_unique(self, df_small):
        assert df_small["client_id"].is_unique, "Duplicate client_ids found"

    def test_client_id_sequential_from_zero(self):
        df = _generate_chunk(100, id_offset=0)
        assert list(df["client_id"]) == list(range(100))

    def test_client_id_respects_offset(self):
        df = _generate_chunk(50, id_offset=1_000)
        assert df["client_id"].min() == 1_000
        assert df["client_id"].max() == 1_049

    def test_age_range(self, df_small):
        assert df_small["age"].min() >= 18, "Age below 18"
        assert df_small["age"].max() <= 70, "Age above 70"

    def test_age_mean_near_40(self, df_large):
        mean_age = df_large["age"].mean()
        assert 35 < mean_age < 45, f"Mean age {mean_age:.1f} outside [35, 45]"

    def test_country_iso_valid_codes(self, df_small):
        valid_iso = set(COUNTRY_ISO.values())
        invalid = set(df_small["country_iso"]) - valid_iso
        assert not invalid, f"Invalid ISO codes found: {invalid}"

    def test_country_iso_two_chars(self, df_small):
        assert df_small["country_iso"].str.len().eq(2).all(), "ISO codes not 2 chars"

    def test_colombia_most_frequent(self, df_large):
        """Colombia has 25 % probability — should be the most common country."""
        most_common_iso = df_large["country_iso"].value_counts().index[0]
        assert most_common_iso == "CO", f"Most common ISO was {most_common_iso}, expected CO"


# ---------------------------------------------------------------------------
# 4. Income columns
# ---------------------------------------------------------------------------

class TestIncomeColumns:
    """Verify salary, bonus, expenses, DTI, savings, net worth, industry."""

    def test_monthly_salary_positive(self, df_small):
        assert (df_small["monthly_salary_usd"] > 0).all()

    def test_annual_bonus_non_negative(self, df_small):
        assert (df_small["annual_bonus_usd"] >= 0).all()

    def test_monthly_expenses_positive(self, df_small):
        assert (df_small["monthly_expenses_usd"] > 0).all()

    def test_debt_to_income_positive(self, df_small):
        assert (df_small["debt_to_income"] > 0).all()

    def test_savings_rate_lower_bound(self, df_large):
        """Savings rate can go slightly negative (dissaving) but not below -0.15."""
        assert df_large["savings_rate"].min() >= -0.15, (
            f"Savings rate too negative: {df_large['savings_rate'].min()}"
        )

    def test_savings_rate_upper_bound(self, df_large):
        assert df_large["savings_rate"].max() <= 1.0

    def test_net_worth_some_negative(self, df_large):
        """~15 % of clients should have negative net worth."""
        pct_negative = (df_large["net_worth_usd"] < 0).mean()
        assert 0.05 < pct_negative < 0.30, (
            f"Unexpected fraction of negative net worth: {pct_negative:.2%}"
        )

    def test_industry_id_range(self, df_small):
        assert df_small["industry_id"].min() >= 0
        assert df_small["industry_id"].max() <= 9

    def test_us_salary_higher_than_colombia(self, df_large):
        """US median salary should be significantly higher than Colombia's."""
        us = df_large[df_large["country_iso"] == "US"]["monthly_salary_usd"]
        co = df_large[df_large["country_iso"] == "CO"]["monthly_salary_usd"]
        if len(us) > 5 and len(co) > 5:
            assert us.median() > co.median() * 3, (
                "US salary not sufficiently higher than Colombia"
            )


# ---------------------------------------------------------------------------
# 5. Account columns
# ---------------------------------------------------------------------------

class TestAccountColumns:
    """Verify balance, account_type_id, join_days, num_products, is_premium."""

    def test_account_balance_non_negative(self, df_small):
        assert (df_small["account_balance_usd"] >= 0).all()

    def test_account_type_id_valid(self, df_small):
        assert df_small["account_type_id"].isin([0, 1, 2]).all()

    def test_join_days_positive(self, df_small):
        """join_days is days since 2000-01-01; must be > 0 for any open account."""
        assert (df_small["join_days"] > 0).all()

    def test_join_days_not_in_future(self, df_small):
        from datetime import date
        max_days = (date.today() - EPOCH).days
        assert df_small["join_days"].max() <= max_days, "join_days in the future"

    def test_num_products_range(self, df_small):
        assert df_small["num_products"].min() >= 1
        assert df_small["num_products"].max() <= 10

    def test_is_premium_binary(self, df_small):
        assert df_small["is_premium_client"].isin([0, 1]).all()

    def test_premium_clients_exist(self, df_large):
        """At least some clients should be premium."""
        assert df_large["is_premium_client"].sum() > 0

    def test_premium_type_skewed_to_2(self, df_large):
        """Premium clients (is_premium=1) should mostly have account_type_id=2."""
        premium = df_large[df_large["is_premium_client"] == 1]
        if len(premium) > 10:
            pct_type2 = (premium["account_type_id"] == 2).mean()
            assert pct_type2 > 0.5, f"Only {pct_type2:.0%} of premium clients have type 2"


# ---------------------------------------------------------------------------
# 6. Transaction columns
# ---------------------------------------------------------------------------

class TestTransactionColumns:
    """Verify monthly_transactions, amounts, failures, international %, channel."""

    def test_monthly_transactions_at_least_one(self, df_small):
        assert (df_small["monthly_transactions"] >= 1).all()

    def test_monthly_transactions_max(self, df_small):
        assert (df_small["monthly_transactions"] <= 100).all()

    def test_avg_transaction_amount_positive(self, df_small):
        assert (df_small["avg_transaction_amount_usd"] > 0).all()

    def test_failed_transactions_non_negative(self, df_small):
        assert (df_small["failed_transactions"] >= 0).all()

    def test_international_txn_pct_in_0_1(self, df_small):
        assert (df_small["international_txn_pct"] >= 0).all()
        assert (df_small["international_txn_pct"] <= 1).all()

    def test_preferred_channel_id_valid(self, df_small):
        assert df_small["preferred_channel_id"].isin([0, 1, 2, 3]).all()

    def test_app_channel_most_common(self, df_large):
        """Channel 0 (app) has 50 % probability — should be most frequent."""
        most_common = df_large["preferred_channel_id"].value_counts().index[0]
        assert most_common == 0, f"Most common channel was {most_common}, expected 0"

    def test_last_login_days_range(self, df_small):
        assert (df_small["last_login_days_ago"] >= 0).all()
        assert (df_small["last_login_days_ago"] <= 365).all()

    def test_premium_more_transactions_on_average(self, df_large):
        """Premium clients (mu=25) should average more transactions than standard (mu=12)."""
        premium_avg  = df_large[df_large["is_premium_client"] == 1]["monthly_transactions"].mean()
        standard_avg = df_large[df_large["is_premium_client"] == 0]["monthly_transactions"].mean()
        assert premium_avg > standard_avg, (
            f"Premium avg {premium_avg:.1f} not > standard avg {standard_avg:.1f}"
        )


# ---------------------------------------------------------------------------
# 7. Credit columns
# ---------------------------------------------------------------------------

class TestCreditColumns:
    """Verify credit_score, num_loans, default_risk, card limit, payment %."""

    def test_credit_score_fico_range(self, df_small):
        assert df_small["credit_score"].min() >= 300
        assert df_small["credit_score"].max() <= 850

    def test_credit_score_mean_above_midpoint(self, df_large):
        """Beta(5,2) right-skew means most clients should score above 575."""
        assert df_large["credit_score"].mean() > 575

    def test_num_loans_range(self, df_small):
        assert df_small["num_loans"].min() >= 0
        assert df_small["num_loans"].max() <= 8

    def test_loan_default_risk_probability(self, df_small):
        assert (df_small["loan_default_risk"] >= 0.01).all()
        assert (df_small["loan_default_risk"] <= 0.99).all()

    def test_credit_card_limit_range(self, df_small):
        assert df_small["credit_card_limit_usd"].min() >= 500
        assert df_small["credit_card_limit_usd"].max() <= 50_000

    def test_payment_on_time_pct_range(self, df_small):
        assert (df_small["payment_on_time_pct"] >= 0).all()
        assert (df_small["payment_on_time_pct"] <= 1).all()


# ---------------------------------------------------------------------------
# 8. Investment columns
# ---------------------------------------------------------------------------

class TestInvestmentColumns:
    """Verify has_investments, portfolio, monthly_investment, has_pension."""

    def test_has_investments_binary(self, df_small):
        assert df_small["has_investments"].isin([0, 1]).all()

    def test_has_pension_binary(self, df_small):
        assert df_small["has_pension"].isin([0, 1]).all()

    def test_portfolio_zero_for_non_investors(self, df_large):
        """Clients without investments must have a zero portfolio."""
        no_invest = df_large[df_large["has_investments"] == 0]
        assert (no_invest["investment_portfolio_usd"] == 0).all()

    def test_monthly_investment_zero_for_non_investors(self, df_large):
        no_invest = df_large[df_large["has_investments"] == 0]
        assert (no_invest["monthly_investment_usd"] == 0).all()

    def test_portfolio_non_negative_for_investors(self, df_large):
        investors = df_large[df_large["has_investments"] == 1]
        assert (investors["investment_portfolio_usd"] >= 0).all()

    def test_monthly_investment_non_negative_for_investors(self, df_large):
        investors = df_large[df_large["has_investments"] == 1]
        assert (investors["monthly_investment_usd"] >= 0).all()

    def test_pension_rate_near_30_pct(self, df_large):
        """has_pension base probability is 30 %; observed rate should be close."""
        rate = df_large["has_pension"].mean()
        assert 0.15 < rate < 0.50, f"Pension rate {rate:.2%} outside expected range"


# ---------------------------------------------------------------------------
# 9. Logical relationships
# ---------------------------------------------------------------------------

class TestLogicalRelationships:
    """Verify that variables relate to each other as the model intends."""

    def test_high_credit_score_lower_default_risk(self, df_large):
        """Higher credit scores must correlate with lower default risk."""
        high = df_large[df_large["credit_score"] > 700]["loan_default_risk"].mean()
        low  = df_large[df_large["credit_score"] < 500]["loan_default_risk"].mean()
        assert high < low, (
            f"High score default risk ({high:.3f}) >= low score ({low:.3f})"
        )

    def test_high_dti_raises_default_risk(self, df_large):
        """Higher debt-to-income must correlate with higher default risk."""
        high_dti = df_large[df_large["debt_to_income"] > 0.8]["loan_default_risk"].mean()
        low_dti  = df_large[df_large["debt_to_income"] < 0.4]["loan_default_risk"].mean()
        assert high_dti > low_dti, (
            f"High DTI default risk ({high_dti:.3f}) not > low DTI ({low_dti:.3f})"
        )

    def test_premium_clients_higher_net_worth(self, df_large):
        """Premium clients should have a higher average net worth."""
        premium_nw  = df_large[df_large["is_premium_client"] == 1]["net_worth_usd"].mean()
        standard_nw = df_large[df_large["is_premium_client"] == 0]["net_worth_usd"].mean()
        assert premium_nw > standard_nw, (
            f"Premium net worth ({premium_nw:,.0f}) not > standard ({standard_nw:,.0f})"
        )

    def test_salary_positively_correlated_with_balance(self, df_large):
        """Monthly salary and account balance should be positively correlated."""
        corr = df_large[["monthly_salary_usd", "account_balance_usd"]].corr().iloc[0, 1]
        assert corr > 0.3, f"Salary-balance correlation too weak: {corr:.3f}"

    def test_investors_tend_to_have_higher_net_worth(self, df_large):
        """Clients with investments should have higher average net worth."""
        investor_nw    = df_large[df_large["has_investments"] == 1]["net_worth_usd"].mean()
        non_investor_nw = df_large[df_large["has_investments"] == 0]["net_worth_usd"].mean()
        assert investor_nw > non_investor_nw, (
            "Investors don't have higher net worth than non-investors"
        )

    def test_high_default_risk_lower_payment_pct(self, df_large):
        """High-risk clients (risk > 0.5) should pay on time less often."""
        high_risk = df_large[df_large["loan_default_risk"] > 0.5]["payment_on_time_pct"].mean()
        low_risk  = df_large[df_large["loan_default_risk"] < 0.3]["payment_on_time_pct"].mean()
        assert high_risk < low_risk, (
            f"High-risk payment pct ({high_risk:.3f}) not < low-risk ({low_risk:.3f})"
        )


# ---------------------------------------------------------------------------
# 10. Chunking behavior
# ---------------------------------------------------------------------------

class TestChunkingBehavior:
    """Verify that id_offset and variable chunk sizes work correctly."""

    def test_id_offset_zero_starts_at_zero(self):
        df = _generate_chunk(10, id_offset=0)
        assert df["client_id"].iloc[0] == 0

    def test_id_offset_large_value(self):
        df = _generate_chunk(10, id_offset=49_999_990)
        assert df["client_id"].iloc[0] == 49_999_990
        assert df["client_id"].iloc[-1] == 49_999_999

    def test_consecutive_chunks_have_no_id_overlap(self):
        df1 = _generate_chunk(100, id_offset=0)
        df2 = _generate_chunk(100, id_offset=100)
        combined_ids = pd.concat([df1["client_id"], df2["client_id"]])
        assert combined_ids.is_unique, "IDs overlap between consecutive chunks"

    def test_single_row_chunk(self):
        df = _generate_chunk(1)
        assert len(df) == 1
        assert len(df.columns) == 30

    def test_large_chunk_size(self):
        df = _generate_chunk(5_000)
        assert len(df) == 5_000
        assert df["client_id"].is_unique


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])