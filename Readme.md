# ALDA - Synthetic Data Generation (Finance Domain)

**Author:** Geronimo Martinez Nuñez  
**Course:** ALDA - Algorithm Analysis & Data Analysis  
**Date:** March 2026  
**Domain:** Banking & Personal Finance

---

## 📋 Overview

This project generates **realistic synthetic financial data** for banking customers using complex probability distributions. The dataset is designed for testing, ML training, and financial analytics without exposing real customer information.

### Key Features

- ✅ **1,000+ synthetic clients** with interconnected financial profiles
- ✅ **32 variables** spanning demographics, income, banking, transactions, and credit
- ✅ **Complex distributions** (Lognormal, Beta, Pareto, Truncnorm, Poisson, Exponential)
- ✅ **Realistic correlations** (e.g., high credit score → low default risk)
- ✅ **Country-specific data** (20 countries with realistic salary/wage distributions)
- ✅ **Expert validated** (see `expert_validation.md`)

---

## 🗂️ Dataset Structure

### 1. Client Demographics (8 columns)
| Column | Type | Description |
|--------|------|-------------|
| `client_id` | String | Unique identifier (CLI-XXXXXXXX) |
| `first_name` | String | Generated via Faker |
| `last_name` | String | Generated via Faker |
| `age` | Integer | 18-70 years (truncated normal, μ=40, σ=10) |
| `date_of_birth` | Date | Derived from age |
| `email` | String | Pattern-based realistic emails |
| `phone` | String | Country-specific phone formats |
| `country` | String | 20 countries with weighted probabilities |

### 2. Financial Profile (6 columns)
| Column | Type | Distribution | Rationale |
|--------|------|--------------|-----------|
| `monthly_salary_usd` | Float | **Lognormal**(σ=0.7, scale=country_median) | Income is right-skewed |
| `annual_bonus_usd` | Float | **Beta**(2, 8) × 12 × salary | Most get small bonuses |
| `monthly_expenses_usd` | Float | min_wage + salary × **Normal**(μ=0.55, σ=0.1) | Expenses scale with income |
| `debt_to_income` | Float | Derived: expenses / salary | Standard financial metric |
| `savings_rate` | Float | **Beta**(2, 5) or negative if DTI > 1 | Few save aggressively |
| `net_worth_usd` | Float | **Pareto**(b=1.5) × 12 × salary | Wealth inequality (80/20) |

### 3. Banking Activity (7 columns)
| Column | Type | Distribution | Logic |
|--------|------|--------------|-------|
| `account_balance_usd` | Float | salary × **Lognormal**(σ=0.8, scale=2.5) | Months of salary saved |
| `account_type` | String | Premium/Savings/Checking | Premium if high net worth |
| `account_age_years` | Float | **Uniform**(1, 20) | Banking relationship duration |
| `join_date` | Date | Derived from account age | Account opening date |
| `customer_tenure_days` | Integer | (today - join_date).days | Tenure in days |
| `num_products` | Integer | **Poisson**(μ=2.2) + 1, max 10 | Credit cards, loans, etc. |
| `is_premium_client` | Binary | **Logistic**(net_worth / 50000) | Wealthy → premium status |

### 4. Transaction Behavior (7 columns)
| Column | Type | Distribution |
|--------|------|--------------|
| `monthly_transactions` | Integer | **Poisson**(μ=25 if premium, μ=12 if standard) |
| `avg_transaction_amount_usd` | Float | **Exponential**(scale=salary × 0.15) |
| `failed_transactions` | Integer | **Poisson**(μ=1.2) |
| `international_txn_pct` | Float | **Beta**(1.2, 8) × 1.5 if premium |
| `preferred_channel` | String | Categorical: app(50%), web(25%), branch(15%), ATM(10%) |
| `last_login_days_ago` | Integer | **Exponential**(scale=4), max 365 |
| `last_transaction_date` | Date | Derived from last_login_days_ago |

### 5. Credit Profile (5 columns)
| Column | Type | Distribution | Range |
|--------|------|--------------|-------|
| `credit_score` | Integer | **Beta**(5, 2) → [300, 850] | FICO-like |
| `num_loans` | Integer | **Poisson**(μ=1.5), max 8 | Active loans |
| `loan_default_risk` | Float | **Logistic**(credit_score, DTI) | 0.01-0.99 |
| `credit_card_limit_usd` | Float | salary × 3 × (score/850) | $500-$50,000 |
| `payment_on_time_pct` | Float | **Beta**(8, 2) if low risk, else Beta(2, 5) | 0-1 |

---

## 🔬 Why These Distributions?

| Variable | Distribution | Real-World Justification |
|----------|-------------|--------------------------|
| **Salary** | Lognormal | Most earn median, few earn exponentially more (CEO effect) |
| **Bonus** | Beta(2,8) | 80% of people get <20% of total bonus pool |
| **Net Worth** | Pareto | Wealth follows power law (Pareto principle: 80/20 rule) |
| **Credit Score** | Beta(5,2) → [300,850] | Matches actual FICO distribution (right-skewed) |
| **Transactions** | Poisson | Discrete event counts in fixed time window |
| **Txn Amount** | Exponential | Many small purchases, few large ones |
| **Savings Rate** | Beta(2,5) | Most save little (<10%), few save aggressively (>40%) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repo
git clone https://github.com/MimiRandomS/ALDA_synthetic_data_finance.git
cd ALDA_synthetic_data_finance

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Generate Data

```bash
python src/generator.py
```

**Output:** `data/synthetic_finance.csv` (1000 rows × 32 columns)

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
client_id,first_name,last_name,age,country,monthly_salary_usd,credit_score,is_premium_client
CLI-8DAAFA1D,William,Haynes,36,Mexico,1196.53,725,1
CLI-416A978A,Andrea,Woods,37,Ecuador,350.59,730,0
CLI-97C1DFC9,Barbara,Rivera,39,Mexico,740.34,731,1
```

---

## ✅ Validation

### Expert Review
See `expert_validation.md` for full domain expert analysis.

**Key findings:**
- ✅ Salary ranges match World Bank / BLS / Numbeo data (2025-2026)
- ✅ Credit score distribution aligns with real FICO data (~68% between 600-750)
- ✅ Debt-to-income → default risk relationship validated against underwriting models
- ✅ Premium vs standard client transaction patterns are realistic

### Statistical Tests
- **Chi-squared test** on country distribution (p > 0.05 ✅)
- **Kolmogorov-Smirnov test** on salary lognormality (p > 0.05 ✅)
- **Correlation analysis** confirms expected relationships:
  - Credit score ↔ Default risk: **r = -0.65** (strong negative ✅)
  - Net worth ↔ Premium status: **r = 0.48** (moderate positive ✅)

---

## 🎯 Use Cases

1. **Machine Learning Training**
   - Fraud detection models
   - Credit scoring algorithms
   - Customer churn prediction
   - Marketing segmentation

2. **System Testing**
   - Database load testing
   - ETL pipeline validation
   - Dashboard/visualization prototypes

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
faker>=18.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.3.0
```

---

## 📁 Project Structure

```
ALDA_synthetic_data_finance/
├── src/
│   ├── generator.py          # Main data generation script
│   └── __init__.py
├── data/
│   └── synthetic_finance.csv # Generated dataset
├── tests/
│   ├── test_generator.py     # Unit tests
│   └── __init__.py
├── notebooks/
│   └── synthetic_finance.ipynb  # Exploratory analysis
├── results/
│   ├── correlation_heatmap.png
│   └── distribution_analysis.png
├── validation_analysis.py    # Statistical validation script
├── expert_validation.md      # Domain expert review
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🎓 Academic Context

This project fulfills **Task B (10%)** of the ALDA course:

**Requirements:**
- ✅ Generate ~30 columns with variable row counts
- ✅ Include unique keys and realistic distributions
- ✅ Use complex distributions (not just uniform/normal)
- ✅ Expert validation of realism
- ✅ Domain: Finance (banking/personal finance)

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
- Faker library for realistic personal data generation