# ALDA - Synthetic Data Generation (Finance Domain)

**Author:** Geronimo Martinez Nuñez  
**Course:** ALDA - Algorithm Analysis & Data Analysis  
**Date:** March 2026  
**Domain:** Banking & Personal Finance

---

## 📋 Overview

This project generates **realistic synthetic financial data** for banking customers using complex probability distributions. The dataset is designed for testing, ML training, and financial analytics without exposing real customer information.

### Key Features

- ✅ **50,000,000 synthetic clients** with interconnected financial profiles
- ✅ **30 variables** spanning demographics, income, banking, transactions, credit, and investments
- ✅ **Complex distributions** (Lognormal, Beta, Pareto, Truncnorm, Poisson, Exponential)
- ✅ **Realistic correlations** (e.g., high credit score → low default risk)
- ✅ **Country-specific data** (20 countries with realistic salary/wage distributions)
- ✅ **Expert validated** (see `expert_validation.md`)

---

## 🗂️ Dataset Structure

### 1. Identity (3 columns)

| Column | Type | Description |
|--------|------|-------------|
| `client_id` | Integer | Sequential unique identifier starting at 0 |
| `age` | Integer | 18–70 years (truncated normal, μ=40, σ=10) |
| `country_iso` | String | ISO 3166-1 alpha-2 code (e.g. CO, US, MX) |

### 2. Income (7 columns)

| Column | Type | Distribution | Rationale |
|--------|------|--------------|-----------|
| `monthly_salary_usd` | Integer | **Lognormal**(σ=0.7, scale=country_median) | Income is right-skewed |
| `annual_bonus_usd` | Integer | **Beta**(2, 8) × 12 × salary | Most get small bonuses |
| `monthly_expenses_usd` | Integer | min_wage + salary × **Normal**(μ=0.55, σ=0.1) | Expenses scale with income |
| `debt_to_income` | Float | Derived: expenses / salary | Standard financial metric |
| `savings_rate` | Float | **Beta**(2, 5) or negative if DTI > 1 | Few save aggressively |
| `net_worth_usd` | Integer | **Pareto**(b=1.5) × 12 × salary | Wealth inequality (80/20 rule) |
| `industry_id` | Integer | Uniform over 10 categories | Client's employment sector |

### 3. Account (5 columns)

| Column | Type | Distribution | Logic |
|--------|------|--------------|-------|
| `account_balance_usd` | Integer | salary × **Lognormal**(σ=0.8, scale=2.5 months) | Months of salary saved |
| `account_type_id` | Integer | 0=checking, 1=savings, 2=premium | Premium if `is_premium_client=1` |
| `join_days` | Integer | Days since 2000-01-01 | Account opening date (encoded) |
| `num_products` | Integer | **Poisson**(μ=2.2) + 1, max 10 | Credit cards, loans, etc. |
| `is_premium_client` | Integer | **Logistic**(net_worth / 50 000) | Wealthy → premium status |

### 4. Transactions (6 columns)

| Column | Type | Distribution |
|--------|------|--------------|
| `monthly_transactions` | Integer | **Poisson**(μ=25 if premium, μ=12 if standard) |
| `avg_transaction_amount_usd` | Integer | **Exponential**(scale=salary × 0.15) |
| `failed_transactions` | Integer | **Poisson**(μ=1.2) |
| `international_txn_pct` | Float | **Beta**(1.2, 8) × 1.5 if premium |
| `preferred_channel_id` | Integer | 0=app(50%), 1=web(25%), 2=branch(15%), 3=ATM(10%) |
| `last_login_days_ago` | Integer | **Exponential**(scale=4), max 365 |

### 5. Credit (5 columns)

| Column | Type | Distribution | Range |
|--------|------|--------------|-------|
| `credit_score` | Integer | **Beta**(5, 2) → [300, 850] | FICO-like |
| `num_loans` | Integer | **Poisson**(μ=1.5), max 8 | Active loans |
| `loan_default_risk` | Float | **Logistic**(credit_score, DTI) | 0.01–0.99 |
| `credit_card_limit_usd` | Integer | salary × 3 × (score/850) | $500–$50,000 |
| `payment_on_time_pct` | Float | **Beta**(8, 2) if low risk, else **Beta**(2, 5) | 0–1 |

### 6. Investment (4 columns)

| Column | Type | Distribution | Logic |
|--------|------|--------------|-------|
| `has_investments` | Integer | **Logistic**(net_worth / 20 000) | Wealthier → more likely to invest |
| `investment_portfolio_usd` | Integer | **Lognormal**(scale=net_worth × 0.3) | 0 if no investments |
| `monthly_investment_usd` | Integer | **Beta**(1.5, 6) × salary | 0 if no investments |
| `has_pension` | Integer | Bernoulli(~0.30) | ~30 % of clients have a pension |

---

## 🔑 Decoding Dictionaries

```python
country_iso          = {CO, MX, AR, CL, PE, BR, EC, VE, BO, PY,
                        UY, PA, CR, GT, DO, US, ES, CA, DE, FR}

account_type_id      = {0: "checking", 1: "savings", 2: "premium"}

preferred_channel_id = {0: "app", 1: "web", 2: "branch", 3: "ATM"}

industry_id          = {0: "tech", 1: "finance", 2: "health", 3: "retail",
                        4: "manufacturing", 5: "education", 6: "government",
                        7: "agriculture", 8: "construction", 9: "other"}

# Reconstruct join date:
from datetime import date, timedelta
join_date = date(2000, 1, 1) + timedelta(days=join_days)
```

---

## 🔬 Why These Distributions?

| Variable | Distribution | Real-World Justification |
|----------|-------------|--------------------------|
| **Salary** | Lognormal | Most earn median, few earn exponentially more (CEO effect) |
| **Bonus** | Beta(2, 8) | 80 % of people get less than 20 % of the total bonus pool |
| **Net Worth** | Pareto | Wealth follows a power law (Pareto principle: 80/20 rule) |
| **Credit Score** | Beta(5, 2) → [300, 850] | Matches actual FICO distribution (right-skewed toward higher scores) |
| **Transactions** | Poisson | Discrete event counts in a fixed time window |
| **Txn Amount** | Exponential | Many small purchases, few large ones |
| **Savings Rate** | Beta(2, 5) | Most save little (< 10 %), few save aggressively (> 40 %) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repo
git clone https://github.com/MimiRandomS/ALDA_synthetic_data_finance.git
cd ALDA_synthetic_data_finance

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac / Linux

# Install dependencies
pip install -r requirements.txt
```

### Generate Data

```bash
python src/generator.py
```

**Output:** `data/synthetic_finance.csv` (~5.5 GB, 50 000 000 rows × 30 columns)

To generate a smaller test sample, edit the call at the bottom of `generator.py`:

```python
generate_dataset(
    total_n=50_000,      # number of rows
    chunk_size=20_000,   # rows per in-memory chunk
)
```

### Run Tests

```bash
pytest tests/ -v
```

### Validate Data Quality

```bash
python validation_analysis.py
```

**Output:**
- `results/correlation_heatmap.png`
- `results/distribution_analysis.png`
- Statistical test results in terminal

---

## 📊 Sample Output

```csv
client_id,age,country_iso,monthly_salary_usd,annual_bonus_usd,monthly_expenses_usd,debt_to_income,savings_rate,net_worth_usd,industry_id,account_balance_usd,account_type_id,join_days,num_products,is_premium_client,monthly_transactions,avg_transaction_amount_usd,failed_transactions,international_txn_pct,preferred_channel_id,last_login_days_ago,credit_score,num_loans,loan_default_risk,credit_card_limit_usd,payment_on_time_pct,has_investments,investment_portfolio_usd,monthly_investment_usd,has_pension
0,38,CO,812,1243,681,0.839,0.214,9841,3,1950,1,7302,3,0,11,98,1,0.042,0,2,714,1,0.083,2868,0.921,1,3120,67,0
1,45,MX,1104,2876,824,0.747,0.189,24310,1,3201,2,6187,4,1,27,201,0,0.134,0,1,768,0,0.031,3892,0.964,1,8740,143,1
2,29,CO,694,890,589,0.849,0.092,5230,7,871,0,8014,2,0,9,55,2,0.021,1,8,631,2,0.241,1648,0.712,0,0,0,0
```

---

## ✅ Validation

### Expert Review
See `expert_validation.md` for full domain expert analysis.

**Key findings:**
- ✅ Salary ranges match World Bank / BLS / Numbeo data (2025–2026)
- ✅ Credit score distribution aligns with real FICO data (~68 % between 600–750)
- ✅ Debt-to-income → default risk relationship validated against underwriting models
- ✅ Premium vs standard client transaction patterns are realistic

### Statistical Tests
- **Chi-squared test** on country distribution (p > 0.05 ✅)
- **Kolmogorov-Smirnov test** on salary log-normality (p > 0.05 ✅)
- **Correlation analysis** confirms expected relationships:
  - Credit score ↔ Default risk: **r = −0.65** (strong negative ✅)
  - Net worth ↔ Premium status: **r = +0.48** (moderate positive ✅)
  - Salary ↔ Account balance: **r = +0.34** (weak-moderate positive ✅)

---

## 🎯 Use Cases

1. **Machine Learning Training**
   - Fraud detection models
   - Credit scoring algorithms
   - Customer churn prediction
   - Marketing segmentation

2. **System Testing**
   - Database load testing (~5.5 GB realistic CSV)
   - ETL pipeline validation
   - Dashboard / visualization prototypes

3. **Educational Purposes**
   - Data analysis coursework
   - Statistical modeling examples
   - SQL practice datasets

---

## 📚 Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pytest>=7.3.0
```

---

## 📁 Project Structure

```
ALDA_synthetic_data_finance/
├── src/
│   ├── generator.py             # Main data generation script
│   └── __init__.py
├── data/
│   └── synthetic_finance.csv    # Generated dataset (created at runtime)
├── tests/
│   ├── test_generator.py        # Unit tests
│   └── __init__.py
├── results/
│   ├── correlation_heatmap.png
│   └── distribution_analysis.png
├── validation_analysis.py       # Statistical validation script
├── expert_validation.md         # Domain expert review
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🎓 Academic Context

This project fulfills **Task B (10 %)** of the ALDA course:

**Requirements:**
- ✅ Generate ~30 columns with variable row counts (up to 50 M)
- ✅ Include unique keys and realistic distributions
- ✅ Use complex distributions (not just uniform/normal)
- ✅ Expert validation of realism
- ✅ Domain: Finance (banking / personal finance)

**Course:** Algorithm Analysis & Data Analysis  
**Institution:** Escuela Colombiana de Ingeniería Julio Garavito  
**Professor:** Nicolas Quevedo  
**Semester:** 2026-1

---

## 📧 Contact

**Geronimo Martinez Nuñez**  
Systems Engineering Student

- GitHub: [@MimiRandomS](https://github.com/MimiRandomS)
- LinkedIn: [geronimo-martinez-nunez](https://www.linkedin.com/in/geronimo-martinez-nunez/)

---

## 📄 License

This project is for educational purposes as part of the ALDA course curriculum.

---

## 🙏 Acknowledgments

- Real-world salary data: World Bank, BLS, Numbeo, Statista
- Statistical distributions: SciPy documentation