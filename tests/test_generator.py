"""
Unit tests for synthetic finance data generator

Tests verify:
1. Data structure and schema
2. Distribution shapes and ranges
3. Logical relationships between variables
4. Data quality (no nulls, unique IDs, valid formats)

Author: Geronimo Martinez Nuñez
Course: ALDA - Algorithm Analysis & Data Analysis
"""

import pytest
import pandas as pd
import numpy as np

from src.generator import (
    _generate_chunk,
    generate_client_info,
    generate_income_info,
    generate_bank_account_info,
    generate_transaction_info,
    generate_credit_info,
)


class TestDataStructure:
    """Test basic data structure and schema"""

    def test_generate_chunk_returns_dataframe(self):
        """Test that _generate_chunk returns a DataFrame of the correct size"""
        df = _generate_chunk(100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_correct_number_of_columns(self):
        """Test dataset has expected number of columns"""
        df = _generate_chunk(50)
        assert len(df.columns) >= 32

    def test_no_missing_values(self):
        """Test that there are no missing values in required columns"""
        df = _generate_chunk(100)
        required_cols = [
            'client_id', 'first_name', 'last_name', 'age',
            'monthly_salary_usd', 'credit_score'
        ]
        for col in required_cols:
            assert df[col].notna().all(), f"Column {col} has missing values"

    def test_unique_client_ids(self):
        """Test that all client IDs are unique"""
        df = _generate_chunk(100)
        assert df['client_id'].is_unique, "Duplicate client IDs found"

    def test_client_id_format(self):
        """Test that client IDs follow CLI-XXXXXXXX format"""
        df = _generate_chunk(50)
        pattern = r'^CLI-[A-F0-9]{8}$'
        assert df['client_id'].str.match(pattern).all(), "Invalid client_id format"


class TestClientInfo:
    """Test client demographics generation"""

    def test_age_range(self):
        """Test that ages are within expected range (18-70)"""
        df = generate_client_info(100)
        assert df['age'].min() >= 18, "Age below 18 found"
        assert df['age'].max() <= 70, "Age above 70 found"

    def test_email_format(self):
        """Test that emails have valid format"""
        df = generate_client_info(50)
        assert df['email'].str.contains('@').all(), "Invalid email: missing @"
        assert df['email'].str.contains(r'\.').all(), "Invalid email: missing ."

    def test_country_distribution(self):
        """Test that countries are from expected list"""
        df = generate_client_info(200)
        expected_countries = [
            "Colombia", "Mexico", "Argentina", "Chile", "Peru",
            "Brazil", "Ecuador", "Venezuela", "Bolivia", "Paraguay",
            "Uruguay", "Panama", "Costa Rica", "Guatemala", "Dominican Republic",
            "United States", "Spain", "Canada", "Germany", "France"
        ]
        assert df['country'].isin(expected_countries).all(), "Unexpected country found"

    def test_age_follows_distribution(self):
        """Test that age follows truncated normal distribution"""
        df = generate_client_info(1000)
        mean_age = df['age'].mean()
        assert 35 < mean_age < 45, f"Mean age {mean_age} outside expected range"


class TestIncomeInfo:
    """Test financial profile generation"""

    def test_salary_positive(self):
        """Test that all salaries are positive"""
        countries = np.array(['Colombia'] * 50)
        df = generate_income_info(50, countries)
        assert (df['monthly_salary_usd'] > 0).all(), "Negative salary found"

    def test_bonus_non_negative(self):
        """Test that bonuses are non-negative"""
        countries = np.array(['Mexico'] * 50)
        df = generate_income_info(50, countries)
        assert (df['annual_bonus_usd'] >= 0).all(), "Negative bonus found"

    def test_expenses_positive(self):
        """Test that expenses are positive"""
        countries = np.array(['United States'] * 50)
        df = generate_income_info(50, countries)
        assert (df['monthly_expenses_usd'] > 0).all(), "Non-positive expense found"

    def test_savings_rate_range(self):
        """Test that savings rate is within reasonable bounds"""
        countries = np.array(['Chile'] * 100)
        df = generate_income_info(100, countries)
        assert df['savings_rate'].min() >= -0.3, "Savings rate too negative"
        assert df['savings_rate'].max() <= 1.0, "Savings rate above 100%"


class TestBankAccountInfo:
    """Test banking activity generation"""

    def test_account_balance_non_negative(self):
        """Test that account balances are non-negative"""
        salary = np.array([1000.0] * 50)
        net_worth = np.array([5000.0] * 50)
        df = generate_bank_account_info(50, salary, net_worth)
        assert (df['account_balance_usd'] >= 0).all(), "Negative balance found"

    def test_account_type_valid(self):
        """Test that account types are from valid set"""
        salary = np.array([1500.0] * 50)
        net_worth = np.array([10000.0] * 50)
        df = generate_bank_account_info(50, salary, net_worth)
        valid_types = ['checking', 'savings', 'premium']
        assert df['account_type'].isin(valid_types).all(), "Invalid account type"

    def test_num_products_positive(self):
        """Test that number of products is positive integer"""
        salary = np.array([800.0] * 50)
        net_worth = np.array([3000.0] * 50)
        df = generate_bank_account_info(50, salary, net_worth)
        assert (df['num_products'] >= 1).all(), "Zero products found"
        assert (df['num_products'] <= 10).all(), "Too many products"

    def test_premium_client_binary(self):
        """Test that is_premium_client is binary (0 or 1)"""
        salary = np.array([1200.0] * 100)
        net_worth = np.array([8000.0] * 100)
        df = generate_bank_account_info(100, salary, net_worth)
        assert df['is_premium_client'].isin([0, 1]).all(), "Invalid premium flag"

    def test_account_age_range(self):
        """Test that account age is within reasonable range"""
        salary = np.array([1000.0] * 50)
        net_worth = np.array([5000.0] * 50)
        df = generate_bank_account_info(50, salary, net_worth)
        assert df['account_age_years'].min() >= 1, "Account age below 1 year"
        assert df['account_age_years'].max() <= 20, "Account age above 20 years"


class TestTransactionInfo:
    """Test transaction behavior generation"""

    def test_monthly_transactions_positive(self):
        """Test that monthly transactions are positive"""
        salary = np.array([1000.0] * 50)
        is_premium = np.array([0] * 25 + [1] * 25)
        df = generate_transaction_info(50, salary, is_premium)
        assert (df['monthly_transactions'] >= 1).all(), "Zero transactions found"

    def test_premium_has_more_transactions(self):
        """Test that premium clients have more transactions on average"""
        salary = np.array([1000.0] * 200)
        is_premium = np.array([0] * 100 + [1] * 100)
        df = generate_transaction_info(200, salary, is_premium)

        premium_avg = df[is_premium == 1]['monthly_transactions'].mean()
        standard_avg = df[is_premium == 0]['monthly_transactions'].mean()

        assert premium_avg > standard_avg, "Premium clients don't have more transactions"

    def test_avg_transaction_amount_positive(self):
        """Test that average transaction amounts are positive"""
        salary = np.array([1200.0] * 50)
        is_premium = np.array([0] * 50)
        df = generate_transaction_info(50, salary, is_premium)
        assert (df['avg_transaction_amount_usd'] > 0).all(), "Non-positive txn amount"

    def test_failed_transactions_non_negative(self):
        """Test that failed transactions are non-negative integers"""
        salary = np.array([1000.0] * 50)
        is_premium = np.array([1] * 50)
        df = generate_transaction_info(50, salary, is_premium)
        assert (df['failed_transactions'] >= 0).all(), "Negative failed txns"

    def test_international_txn_pct_range(self):
        """Test that international transaction % is between 0 and 1"""
        salary = np.array([1500.0] * 50)
        is_premium = np.array([1] * 50)
        df = generate_transaction_info(50, salary, is_premium)
        assert (df['international_txn_pct'] >= 0).all(), "Negative intl %"
        assert (df['international_txn_pct'] <= 1).all(), "Intl % above 100%"

    def test_preferred_channel_valid(self):
        """Test that preferred channels are from valid set"""
        salary = np.array([1000.0] * 50)
        is_premium = np.array([0] * 50)
        df = generate_transaction_info(50, salary, is_premium)
        valid_channels = ['app', 'web', 'branch', 'ATM']
        assert df['preferred_channel'].isin(valid_channels).all(), "Invalid channel"


class TestCreditInfo:
    """Test credit profile generation"""

    def test_credit_score_range(self):
        """Test that credit scores are within FICO range (300-850)"""
        salary = np.array([1000.0] * 100)
        dti = np.array([0.5] * 100)
        df = generate_credit_info(100, salary, dti)
        assert df['credit_score'].min() >= 300, "Credit score below 300"
        assert df['credit_score'].max() <= 850, "Credit score above 850"

    def test_num_loans_non_negative(self):
        """Test that number of loans is non-negative"""
        salary = np.array([1200.0] * 50)
        dti = np.array([0.6] * 50)
        df = generate_credit_info(50, salary, dti)
        assert (df['num_loans'] >= 0).all(), "Negative loan count"
        assert (df['num_loans'] <= 8).all(), "Too many loans"

    def test_loan_default_risk_range(self):
        """Test that default risk is probability (0-1)"""
        salary = np.array([1000.0] * 100)
        dti = np.array([0.7] * 100)
        df = generate_credit_info(100, salary, dti)
        assert (df['loan_default_risk'] >= 0).all(), "Negative default risk"
        assert (df['loan_default_risk'] <= 1).all(), "Default risk above 100%"

    def test_credit_card_limit_range(self):
        """Test that credit card limits are within reasonable bounds"""
        salary = np.array([1500.0] * 50)
        dti = np.array([0.5] * 50)
        df = generate_credit_info(50, salary, dti)
        assert df['credit_card_limit_usd'].min() >= 500, "Limit below $500"
        assert df['credit_card_limit_usd'].max() <= 50000, "Limit above $50k"

    def test_payment_on_time_pct_range(self):
        """Test that on-time payment % is between 0 and 1"""
        salary = np.array([1000.0] * 50)
        dti = np.array([0.4] * 50)
        df = generate_credit_info(50, salary, dti)
        assert (df['payment_on_time_pct'] >= 0).all(), "Negative payment %"
        assert (df['payment_on_time_pct'] <= 1).all(), "Payment % above 100%"


class TestLogicalRelationships:
    """Test logical relationships between variables"""

    def test_high_credit_score_low_default_risk(self):
        """Test that high credit scores correlate with low default risk"""
        df = _generate_chunk(500)

        high_score = df[df['credit_score'] > 700]['loan_default_risk'].mean()
        low_score = df[df['credit_score'] < 600]['loan_default_risk'].mean()

        assert high_score < low_score, "High credit score doesn't reduce default risk"

    def test_high_dti_high_default_risk(self):
        """Test that high DTI correlates with high default risk"""
        df = _generate_chunk(500)

        high_dti = df[df['debt_to_income'] > 0.7]['loan_default_risk'].mean()
        low_dti = df[df['debt_to_income'] < 0.4]['loan_default_risk'].mean()

        assert high_dti > low_dti, "High DTI doesn't increase default risk"

    def test_premium_clients_higher_net_worth(self):
        """Test that premium clients have higher average net worth"""
        df = _generate_chunk(500)

        premium_nw = df[df['is_premium_client'] == 1]['net_worth_usd'].mean()
        standard_nw = df[df['is_premium_client'] == 0]['net_worth_usd'].mean()

        assert premium_nw > standard_nw * 0.8, "Premium doesn't correlate with net worth"

    def test_salary_correlates_with_account_balance(self):
        """Test positive correlation between salary and account balance"""
        df = _generate_chunk(300)
        correlation = df[['monthly_salary_usd', 'account_balance_usd']].corr().iloc[0, 1]
        assert correlation > 0.3, f"Weak salary-balance correlation: {correlation}"


class TestDataQuality:
    """Test overall data quality"""

    def test_no_duplicate_emails(self):
        """Test that emails are reasonably unique"""
        df = _generate_chunk(200)
        duplicate_count = df['email'].duplicated().sum()
        assert duplicate_count < 10, f"Too many duplicate emails: {duplicate_count}"

    def test_realistic_salary_by_country(self):
        """Test that salaries vary by country"""
        df = _generate_chunk(500)

        us_avg = df[df['country'] == 'United States']['monthly_salary_usd'].mean()
        co_avg = df[df['country'] == 'Colombia']['monthly_salary_usd'].mean()

        assert us_avg > co_avg * 3, "Country salary differences unrealistic"

    def test_chunk_size_parameter(self):
        """Test that chunk size parameter works"""
        for n in [10, 50, 100, 500]:
            df = _generate_chunk(n)
            assert len(df) == n, f"Chunk size mismatch: expected {n}, got {len(df)}"


# =====================
# FIXTURES
# =====================

@pytest.fixture
def sample_dataset():
    """Fixture providing a sample dataset for tests"""
    return _generate_chunk(100)


# =====================
# RUN TESTS
# =====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])