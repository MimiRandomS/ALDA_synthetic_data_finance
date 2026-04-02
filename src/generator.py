"""
synthetic_finance_generator.py
================================
Generates a synthetic CSV dataset in the financial domain with:
  - Exactly 30 columns with realistic and complex distributions
  - 50,000,000 rows (~5.5 GB in CSV)
  - Unique key per row: client_id (sequential integer)

Usage:
    python synthetic_finance_generator.py

Columns (30):
    Identity      (3):  client_id, age, country_iso
    Income        (7):  monthly_salary_usd, annual_bonus_usd, monthly_expenses_usd,
                        debt_to_income, savings_rate, net_worth_usd, industry_id
    Account       (5):  account_balance_usd, account_type_id, join_days,
                        num_products, is_premium_client
    Transactions  (6):  monthly_transactions, avg_transaction_amount_usd,
                        failed_transactions, international_txn_pct,
                        preferred_channel_id, last_login_days_ago
    Credit        (5):  credit_score, num_loans, loan_default_risk,
                        credit_card_limit_usd, payment_on_time_pct
    Investment    (4):  has_investments, investment_portfolio_usd,
                        monthly_investment_usd, has_pension

Decoding dictionaries:
    country_iso          -> {CO, MX, AR, CL, PE, BR, EC, VE, BO, PY,
                             UY, PA, CR, GT, DO, US, ES, CA, DE, FR}
    account_type_id      -> {0:"checking", 1:"savings", 2:"premium"}
    preferred_channel_id -> {0:"app", 1:"web", 2:"branch", 3:"ATM"}
    industry_id          -> {0:"tech", 1:"finance", 2:"health", 3:"retail",
                             4:"manufacturing", 5:"education", 6:"government",
                             7:"agriculture", 8:"construction", 9:"other"}
    join_days            -> days since date(2000-01-01)
                            reconstruct: date(2000,1,1) + timedelta(days=value)
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Full country names used for internal sampling; mapped to ISO codes before output.
COUNTRIES = [
    "Colombia", "Mexico", "Argentina", "Chile", "Peru",
    "Brazil", "Ecuador", "Venezuela", "Bolivia", "Paraguay",
    "Uruguay", "Panama", "Costa Rica", "Guatemala", "Dominican Republic",
    "United States", "Spain", "Canada", "Germany", "France",
]

# Sampling probabilities per country, must sum to 1.0.
# Colombia and Mexico together represent ~45% of the client base,
# reflecting a dataset skewed toward Latin American markets.
PROBABILITIES = [
    0.25, 0.20, 0.10, 0.06, 0.06,
    0.05, 0.04, 0.04, 0.03, 0.02,
    0.02, 0.02, 0.02, 0.02, 0.02,
    0.02, 0.01, 0.01, 0.005, 0.005,
]

# Maps each country's full name to its ISO 3166-1 alpha-2 code.
COUNTRY_ISO = {
    "Colombia": "CO", "Mexico": "MX", "Argentina": "AR",
    "Chile": "CL", "Peru": "PE", "Brazil": "BR", "Ecuador": "EC",
    "Venezuela": "VE", "Bolivia": "BO", "Paraguay": "PY",
    "Uruguay": "UY", "Panama": "PA", "Costa Rica": "CR",
    "Guatemala": "GT", "Dominican Republic": "DO", "United States": "US",
    "Spain": "ES", "Canada": "CA", "Germany": "DE", "France": "FR",
}

# Approximate median monthly salary in USD per country (2023–2024 estimates).
# Used as the scale parameter of the log-normal salary distribution so that
# the median generated salary aligns with local economic conditions.
COUNTRY_MEDIAN_SALARY_USD = {
    "Colombia": 700,  "Mexico": 850,   "Argentina": 500,  "Chile": 1100,
    "Peru": 650,      "Brazil": 750,   "Ecuador": 600,    "Venezuela": 130,
    "Bolivia": 450,   "Paraguay": 520, "Uruguay": 900,    "Panama": 1100,
    "Costa Rica": 950,"Guatemala": 480,"Dominican Republic": 600,
    "United States": 5174, "Spain": 2000, "Canada": 3200,
    "Germany": 3400,  "France": 2800,
}

# Approximate legal monthly minimum wage in USD per country.
# Added as a floor to the monthly expense calculation so that expenses
# are always at least as large as the statutory minimum.
COUNTRY_MIN_WAGE_USD = {
    "Colombia": 384,  "Mexico": 501,   "Argentina": 287,  "Chile": 555,
    "Peru": 297,      "Brazil": 262,   "Ecuador": 482,    "Venezuela": 1,
    "Bolivia": 469,   "Paraguay": 406, "Uruguay": 632,    "Panama": 543,
    "Costa Rica": 726,"Guatemala": 513,"Dominican Republic": 200,
    "United States": 1256, "Spain": 1636, "Canada": 2192,
    "Germany": 2829,  "France": 2160,
}

# Reference epoch for encoding calendar dates as integer day offsets.
# join_days = (account_open_date - EPOCH).days
EPOCH = date(2000, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_epoch_days(dates: list) -> np.ndarray:
    """Convert a list of ``datetime.date`` objects to integer day offsets from EPOCH.

    Parameters
    ----------
    dates : list of datetime.date
        Calendar dates to convert.

    Returns
    -------
    np.ndarray of dtype int32
        Each element is ``(d - EPOCH).days`` for the corresponding date ``d``.
    """
    return np.array([(d - EPOCH).days for d in dates], dtype=np.int32)


# ---------------------------------------------------------------------------
# Chunk generator — produces exactly 30 columns
# ---------------------------------------------------------------------------

def _generate_chunk(n: int, id_offset: int = 0) -> pd.DataFrame:
    """Generate a single chunk of ``n`` synthetic client records.

    All distributions are chosen to match realistic financial patterns:

    * **Salaries** follow a log-normal distribution whose median is calibrated
      per country via ``COUNTRY_MEDIAN_SALARY_USD``.
    * **Bonuses** are a Beta-distributed fraction of annual salary.
    * **Expenses** include a minimum-wage floor plus a normally distributed
      fraction of salary, then clipped to a plausible range.
    * **Net worth** follows a Pareto distribution (wealth concentration), with
      15 % of clients assigned negative net worth (over-indebted).
    * **Premium status** is derived from net worth via a logistic function.
    * **Credit score** follows a right-skewed Beta distribution (most clients
      are in good standing), mapped to the standard 300–850 range.
    * **Loan default risk** is a logistic function of credit score and
      debt-to-income ratio.
    * **Investment fields** are conditional on having a positive net worth;
      portfolio size follows a log-normal anchored to net worth.

    Parameters
    ----------
    n : int
        Number of rows to generate in this chunk.
    id_offset : int, optional
        Starting value for ``client_id`` (default 0).  Used to produce a
        globally unique, sequential ID across multiple chunks.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly 30 columns and ``n`` rows.
    """
    today = date.today()

    # ------------------------------------------------------------------
    # Identity columns
    # ------------------------------------------------------------------

    # Sequential unique identifier for each client.
    client_id = np.arange(id_offset, id_offset + n, dtype=np.int32)

    # Age drawn from a truncated normal (18–70, mean 40, std 10).
    age_dist = stats.truncnorm(a=(18 - 40) / 10, b=(70 - 40) / 10, loc=40, scale=10)
    age = age_dist.rvs(n).astype(np.int8)

    # Country sampled according to PROBABILITIES; immediately encoded as ISO codes.
    countries_arr = np.random.choice(COUNTRIES, size=n, p=PROBABILITIES)
    country_iso   = np.array([COUNTRY_ISO[c] for c in countries_arr])

    # ------------------------------------------------------------------
    # Country-level lookup arrays (one value per row)
    # ------------------------------------------------------------------

    # Per-client median salary and minimum wage based on sampled country.
    medians   = np.array([COUNTRY_MEDIAN_SALARY_USD[c] for c in countries_arr])
    min_wages = np.array([COUNTRY_MIN_WAGE_USD[c]      for c in countries_arr])

    # ------------------------------------------------------------------
    # Income columns
    # ------------------------------------------------------------------

    # Monthly salary: log-normal with country-specific median.
    # shape=0.7 gives moderate right-skew; scale=medians sets the median.
    salary_raw         = stats.lognorm.rvs(s=0.7, scale=medians, size=n)
    monthly_salary_usd = np.round(salary_raw).astype(np.int32)

    # Annual bonus: Beta(2, 8) fraction of gross annual salary.
    # Mean bonus ≈ 20 % of annual salary; most clients receive < 30 %.
    bonus_rate       = stats.beta.rvs(a=2, b=8, size=n)
    annual_bonus_usd = np.round(salary_raw * 12 * bonus_rate).astype(np.int32)

    # Monthly expenses: minimum wage + expense_rate * salary, clipped to [20 %, 95 %] of salary.
    expense_rate         = np.clip(np.random.normal(0.55, 0.1, n), 0.2, 0.95)
    monthly_expenses_usd = np.round(min_wages + salary_raw * expense_rate).astype(np.int32)

    # Debt-to-income: ratio of monthly expenses to monthly salary.
    # Values > 1 indicate clients spending more than they earn.
    debt_to_income = np.round(monthly_expenses_usd / monthly_salary_usd, 3)

    # Savings rate: Beta(2, 5) when client earns more than they spend;
    # slightly negative uniform draw otherwise (dissaving / borrowing).
    raw_savings  = stats.beta.rvs(a=2, b=5, size=n)
    savings_rate = np.round(np.where(
        monthly_expenses_usd < monthly_salary_usd,
        raw_savings,
        np.random.uniform(-0.15, 0.05, n),
    ), 3)

    # Net worth: Pareto(1.5) scaled by annual salary (wealth concentration).
    # 15 % of clients are assigned a negative net worth (30 % of the Pareto draw).
    net_worth_raw = stats.pareto.rvs(b=1.5, size=n) * salary_raw * 12
    in_debt       = np.random.random(n) < 0.15
    net_worth_usd = np.round(
        np.where(in_debt, -np.abs(net_worth_raw) * 0.3, net_worth_raw)
    ).astype(np.int32)

    # Industry sector: uniform draw over 10 categories (see module docstring).
    industry_id = np.random.choice(10, size=n).astype(np.int8)

    # ------------------------------------------------------------------
    # Account columns
    # ------------------------------------------------------------------

    # Premium flag: logistic function of net worth, bounded to [1 %, 95 %].
    # Wealthier clients are much more likely to be premium.
    nw_pos       = np.clip(net_worth_usd.astype(float), 0, None)
    # Clamp the exponent to [-500, 500] to avoid float64 overflow in exp().
    # Values outside this range produce probabilities indistinguishable from 0 or 1.
    premium_prob = np.clip(1 / (1 + np.exp(np.clip(-nw_pos / 50000, -500, 500))), 0.01, 0.95)
    is_premium   = np.random.binomial(1, premium_prob, n).astype(np.int8)

    # Account balance: log-normal number of months of salary saved.
    # Scale 2.5 months, shape 0.8; clipped to at most 24 months.
    months_saved        = np.clip(stats.lognorm.rvs(s=0.8, scale=2.5, size=n), 0.1, 24)
    account_balance_usd = np.round(salary_raw * months_saved).astype(np.int32)

    # Account type: premium clients lean toward type 2 (premium) and 1 (savings);
    # regular clients favor type 0 (checking) and 1 (savings).
    account_type_id = np.where(
        is_premium == 1,
        np.random.choice([2, 1], n, p=[0.7, 0.3]),
        np.random.choice([0, 1], n, p=[0.6, 0.4]),
    ).astype(np.int8)

    # Join date encoded as days since EPOCH.
    # Account age drawn uniformly from 1 to 20 years.
    account_age_days = (np.random.uniform(1, 20, n) * 365.25).astype(int)
    join_dates       = [today - timedelta(days=int(d)) for d in account_age_days]
    join_days        = _to_epoch_days(join_dates)

    # Number of active products: Poisson(2.2) + 1, clipped to [1, 10].
    num_products = np.clip(stats.poisson.rvs(mu=2.2, size=n) + 1, 1, 10).astype(np.int8)

    # ------------------------------------------------------------------
    # Transaction columns
    # ------------------------------------------------------------------

    # Monthly transaction count: Poisson with mean 25 for premium, 12 for regular.
    txn_mu               = np.where(is_premium == 1, 25, 12)
    monthly_transactions = np.clip(
        stats.poisson.rvs(mu=txn_mu, size=n), 1, 100
    ).astype(np.int8)

    # Average transaction amount: exponential scaled to 15 % of salary; clipped at 3× salary.
    avg_txn = np.clip(
        stats.expon.rvs(scale=salary_raw * 0.15, size=n), 1, salary_raw * 3
    )
    avg_transaction_amount_usd = np.round(avg_txn).astype(np.int32)

    # Failed transactions per month: Poisson(1.2).
    failed_transactions = stats.poisson.rvs(mu=1.2, size=n).astype(np.int8)

    # International transaction share: Beta(1.2, 8), boosted 1.5× for premium clients.
    intl_raw = stats.beta.rvs(a=1.2, b=8, size=n)
    international_txn_pct = np.round(
        np.clip(np.where(is_premium == 1, intl_raw * 1.5, intl_raw), 0, 1), 3
    )

    # Preferred channel: app (50 %), web (25 %), branch (15 %), ATM (10 %).
    preferred_channel_id = np.random.choice(
        4, n, p=[0.50, 0.25, 0.15, 0.10]
    ).astype(np.int8)

    # Days since last login: exponential with mean 4 days, clipped at 365.
    last_login_days_ago = np.clip(
        stats.expon.rvs(scale=4, size=n).astype(int), 0, 365
    ).astype(np.int16)

    # ------------------------------------------------------------------
    # Credit columns
    # ------------------------------------------------------------------

    # Credit score: Beta(5, 2) right-skewed to [300, 850] range.
    # Most synthetic clients have good credit (> 700).
    raw_score    = stats.beta.rvs(a=5, b=2, size=n)
    credit_score = np.clip((raw_score * 550 + 300).astype(int), 300, 850).astype(np.int16)

    # Number of active loans: Poisson(1.5), clipped to [0, 8].
    num_loans = np.clip(stats.poisson.rvs(mu=1.5, size=n), 0, 8).astype(np.int8)

    # Loan default risk: logistic function of normalized credit score and DTI.
    # Higher credit scores and lower DTI reduce default probability.
    norm_score        = (credit_score - 300) / 550
    _default_exp      = np.clip(5 * norm_score - 3 * debt_to_income - 1, -500, 500)
    loan_default_risk = np.round(np.clip(
        1 / (1 + np.exp(_default_exp)), 0.01, 0.99
    ), 3)

    # Credit card limit: 3× monthly salary scaled by creditworthiness, in [500, 50 000].
    credit_card_limit_usd = np.round(
        np.clip(salary_raw * 3 * (credit_score / 850), 500, 50000)
    ).astype(np.int32)

    # On-time payment rate: Beta(8, 2) for low-risk clients; Beta(2, 5) for high-risk.
    pct_base = stats.beta.rvs(a=8, b=2, size=n)
    pct_risk = stats.beta.rvs(a=2, b=5, size=n)
    payment_on_time_pct = np.round(
        np.where(loan_default_risk > 0.5, pct_risk, pct_base), 3
    )

    # ------------------------------------------------------------------
    # Investment columns
    # ------------------------------------------------------------------

    # Investment flag: logistic function of net worth, bounded to [5 %, 90 %].
    invest_prob     = np.clip(1 / (1 + np.exp(np.clip(-net_worth_usd.astype(float) / 20000, -500, 500))), 0.05, 0.90)
    has_investments = np.random.binomial(1, invest_prob, n).astype(np.int8)

    # Portfolio value: log-normal anchored to 30 % of net worth; 0 for non-investors.
    nw_inv           = np.clip(net_worth_usd.astype(float), 1000, None)
    portfolio_raw    = np.clip(stats.lognorm.rvs(s=0.8, scale=nw_inv * 0.3, size=n), 0, 5_000_000)
    investment_portfolio_usd = np.round(
        np.where(has_investments == 1, portfolio_raw, 0)
    ).astype(np.int32)

    # Monthly investment amount: Beta(1.5, 6) fraction of salary; 0 for non-investors.
    inv_rate               = stats.beta.rvs(a=1.5, b=6, size=n)
    monthly_investment_usd = np.round(
        np.where(has_investments == 1, salary_raw * inv_rate, 0)
    ).astype(np.int32)

    # Pension flag: roughly 30 % base probability with Gaussian noise, clipped to [5 %, 90 %].
    pension_prob = np.clip(0.3 + np.random.normal(0, 0.1, n), 0.05, 0.90)
    has_pension  = np.random.binomial(1, pension_prob, n).astype(np.int8)

    # ------------------------------------------------------------------
    # Assemble and return the DataFrame (30 columns, n rows)
    # ------------------------------------------------------------------
    return pd.DataFrame({
        "client_id":                  client_id,
        "age":                        age,
        "country_iso":                country_iso,
        "monthly_salary_usd":         monthly_salary_usd,
        "annual_bonus_usd":           annual_bonus_usd,
        "monthly_expenses_usd":       monthly_expenses_usd,
        "debt_to_income":             debt_to_income,
        "savings_rate":               savings_rate,
        "net_worth_usd":              net_worth_usd,
        "industry_id":                industry_id,
        "account_balance_usd":        account_balance_usd,
        "account_type_id":            account_type_id,
        "join_days":                  join_days,
        "num_products":               num_products,
        "is_premium_client":          is_premium,
        "monthly_transactions":       monthly_transactions,
        "avg_transaction_amount_usd": avg_transaction_amount_usd,
        "failed_transactions":        failed_transactions,
        "international_txn_pct":      international_txn_pct,
        "preferred_channel_id":       preferred_channel_id,
        "last_login_days_ago":        last_login_days_ago,
        "credit_score":               credit_score,
        "num_loans":                  num_loans,
        "loan_default_risk":          loan_default_risk,
        "credit_card_limit_usd":      credit_card_limit_usd,
        "payment_on_time_pct":        payment_on_time_pct,
        "has_investments":            has_investments,
        "investment_portfolio_usd":   investment_portfolio_usd,
        "monthly_investment_usd":     monthly_investment_usd,
        "has_pension":                has_pension,
    })


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(
    total_n:    int = 50_000_000,
    chunk_size: int = 200_000,
    output_dir: str | None = None,
) -> None:
    """Generate the full synthetic financial dataset and write it to CSV.

    The file is written incrementally in chunks to keep peak memory usage low.
    Each chunk of ``chunk_size`` rows is generated in memory, appended to the
    output CSV, then discarded before the next chunk is processed.

    The output directory structure is::

        <output_dir>/
        └── data/
            └── synthetic_finance.csv

    Progress is printed to stdout after each chunk, including elapsed time,
    cumulative row count, file size on disk, and estimated time remaining.

    After generation, an integrity check counts the newlines in the output
    file and warns if the count does not match ``total_n``.

    Parameters
    ----------
    total_n : int, optional
        Total number of rows to generate (default 50 000 000).
    chunk_size : int, optional
        Number of rows per in-memory chunk (default 200 000).  Larger chunks
        improve throughput but consume more RAM.
    output_dir : str or None, optional
        Root directory for output.  Defaults to the directory that contains
        this script file.  The ``data/`` sub-directory is created automatically.

    Returns
    -------
    None
        Writes ``synthetic_finance.csv`` to disk; does not return any object.

    Notes
    -----
    Estimated disk usage: ``total_n * 112`` bytes ≈ 5.5 GB for the default
    50 M rows.  Ensure sufficient free space before running.

    The random seed is **not** fixed, so each run produces a different dataset.
    Set ``np.random.seed(...)`` before calling this function for reproducibility.
    """

    # Resolve output directory: default to the script's own directory.
    if output_dir is None:
        BASE_DIR = Path(__file__).parent.parent  # go up from src/ to project root
    else:
        BASE_DIR = Path(output_dir)

    DATA_DIR    = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / "synthetic_finance.csv"

    # Total number of chunks needed (ceiling division).
    total_chunks = (total_n + chunk_size - 1) // chunk_size

    print(f"Generating {total_n:,} rows in {total_chunks} chunks of {chunk_size:,}...")
    print(f"Columns: 30 | Estimated size: ~{total_n * 112 / 1e9:.1f} GB")
    print(f"Output:  {output_path}\n")

    written_rows = 0
    global_start = time.time()

    for chunk_num in range(total_chunks):
        # Last chunk may be smaller than chunk_size.
        current_size = min(chunk_size, total_n - written_rows)
        t0           = time.time()

        # Generate chunk with globally unique client IDs.
        chunk = _generate_chunk(current_size, id_offset=written_rows)

        # Write header only on the first chunk; all subsequent chunks append.
        chunk.to_csv(
            output_path,
            index=False,
            mode="w" if chunk_num == 0 else "a",
            header=chunk_num == 0,
            float_format="%.4g",
        )

        # Progress reporting.
        elapsed       = time.time() - t0
        written_rows += current_size
        size_gb       = output_path.stat().st_size / 1e9
        pct           = written_rows / total_n * 100
        avg_t         = (time.time() - global_start) / (chunk_num + 1)
        eta_min       = (total_chunks - chunk_num - 1) * avg_t / 60

        print(
            f"  Chunk {chunk_num + 1:>4}/{total_chunks} | "
            f"{elapsed:.1f}s | "
            f"rows: {written_rows:>12,} ({pct:5.1f}%) | "
            f"{size_gb:.2f} GB | "
            f"ETA: {eta_min:.1f} min"
        )

    total_elapsed = time.time() - global_start
    final_gb      = output_path.stat().st_size / 1e9

    print(f"\nDone:")
    print(f"  Rows:    {written_rows:,}")
    print(f"  Columns: 30")
    print(f"  Size:    {final_gb:.2f} GB")
    print(f"  Time:    {total_elapsed / 60:.1f} min  ({written_rows / total_elapsed:,.0f} rows/sec)")
    print(f"  File:    {output_path}")

    # Integrity check: count lines (excluding header) and compare to total_n.
    print("\nVerifying integrity...", end=" ", flush=True)
    actual = sum(1 for _ in open(output_path, encoding="utf-8")) - 1
    if actual == total_n:
        print(f"OK -- {actual:,} rows confirmed.")
    else:
        print(f"WARNING: expected {total_n:,}, found {actual:,}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_dataset(
        total_n=50_000,
        chunk_size=20_000,
    )