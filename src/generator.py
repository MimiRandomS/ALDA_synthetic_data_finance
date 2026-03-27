import random
import time
import numpy as np
import pandas as pd
from scipy import stats
from faker import Faker
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

countries = [
    "Colombia", "Mexico", "Argentina", "Chile", "Peru",
    "Brazil", "Ecuador", "Venezuela", "Bolivia", "Paraguay",
    "Uruguay", "Panama", "Costa Rica", "Guatemala", "Dominican Republic",
    "United States", "Spain", "Canada", "Germany", "France"
]

probabilities = [
    0.25, 0.20, 0.10, 0.06, 0.06,
    0.05, 0.04, 0.04, 0.03, 0.02,
    0.02, 0.02, 0.02, 0.02, 0.02,
    0.02, 0.01, 0.01, 0.005, 0.005
]

locale_map = {
    "Colombia": "es_CO", "Mexico": "es_MX", "Argentina": "es_AR",
    "Chile": "es_CL", "Peru": "es_PE", "Brazil": "pt_BR",
    "Ecuador": "es_EC", "Venezuela": "es_VE", "Bolivia": "es_BO",
    "Paraguay": "es_PY", "Uruguay": "es_UY", "Panama": "es_PA",
    "Costa Rica": "es_CR", "Guatemala": "es_GT",
    "Dominican Republic": "es_DO", "United States": "en_US",
    "Spain": "es_ES", "Canada": "en_CA",
    "Germany": "de_DE", "France": "fr_FR",
}

COUNTRY_MEDIAN_SALARY_USD = {
    "Colombia": 700, "Mexico": 850, "Argentina": 500, "Chile": 1_100,
    "Peru": 650, "Brazil": 750, "Ecuador": 600, "Venezuela": 130,
    "Bolivia": 450, "Paraguay": 520, "Uruguay": 900, "Panama": 1_100,
    "Costa Rica": 950, "Guatemala": 480, "Dominican Republic": 600,
    "United States": 5_174, "Spain": 2_000, "Canada": 3_200,
    "Germany": 3_400, "France": 2_800,
}

COUNTRY_MIN_WAGE_USD = {
    "Colombia": 384, "Mexico": 501, "Argentina": 287, "Chile": 555,
    "Peru": 297, "Brazil": 262, "Ecuador": 482, "Venezuela": 1,
    "Bolivia": 469, "Paraguay": 406, "Uruguay": 632, "Panama": 543,
    "Costa Rica": 726, "Guatemala": 513, "Dominican Republic": 200,
    "United States": 1256, "Spain": 1636, "Canada": 2192,
    "Germany": 2829, "France": 2160,
}

domains = ["gmail.com", "hotmail.com", "yahoo.com", "outlook.com", "icloud.com"]

# ---------------------------------------------------------------------------
# Faker cache — one instance per locale, reused across all rows
# ---------------------------------------------------------------------------

_faker_cache: dict[str, Faker] = {}
_faker_default = Faker("en_US")

def _get_faker(locale: str) -> Faker:
    """
    Retrieve a cached Faker instance for the given locale.

    Creates a new Faker instance on first access and caches it for reuse,
    avoiding the overhead of instantiating Faker once per row. Falls back
    to the default en_US instance if the locale is unsupported.

    Parameters
    ----------
    locale : str
        A valid Faker locale string (e.g. 'es_CO', 'pt_BR').

    Returns
    -------
    Faker
        A cached Faker instance for the requested locale.
    """
    if locale not in _faker_cache:
        try:
            _faker_cache[locale] = Faker(locale)
        except Exception:
            _faker_cache[locale] = _faker_default
    return _faker_cache[locale]

# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def generate_client_info(n: int) -> pd.DataFrame:
    """
    Generate a synthetic dataset of client information.

    This function creates a DataFrame containing randomly generated
    customer data such as names, ages, birthdays, emails, phone numbers,
    and country of origin. Ages are sampled from a truncated normal
    distribution to produce realistic values within a defined range.

    Faker instances are grouped by locale and cached to avoid the overhead
    of creating a new instance per row.

    Parameters
    ----------
    n : int
        Number of clients to generate.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing synthetic client records with the
        following fields:
        - client_id
        - first_name
        - last_name
        - age
        - date_of_birth
        - email
        - phone
        - country
    """
    today = date.today()

    age_dist = stats.truncnorm(a=(18 - 40) / 10, b=(70 - 40) / 10, loc=40, scale=10)
    ages = age_dist.rvs(n).astype(int)

    countries_arr = np.random.choice(countries, size=n, p=probabilities)

    # Group by locale to call Faker once per group instead of once per row
    locales = np.array([locale_map.get(c, "en_US") for c in countries_arr])
    first_names = np.empty(n, dtype=object)
    last_names  = np.empty(n, dtype=object)
    phones      = np.empty(n, dtype=object)

    for locale in np.unique(locales):
        mask = locales == locale
        fk = _get_faker(locale)
        count = mask.sum()
        first_names[mask] = [fk.first_name() for _ in range(count)]
        last_names[mask]  = [fk.last_name()  for _ in range(count)]
        phones[mask]      = [fk.phone_number() for _ in range(count)]

    # Vectorized birth dates
    offsets = (ages * 365.25).astype(int) + np.random.randint(0, 366, size=n)
    birth_dates = [today - timedelta(days=int(o)) for o in offsets]

    # Vectorized email generation
    fn_clean = np.char.lower(np.char.replace(first_names.astype(str), " ", ""))
    ln_clean = np.char.lower(np.char.replace(last_names.astype(str),  " ", ""))
    patterns = np.random.randint(0, 4, size=n)
    rand_nums = np.random.randint(1, 100, size=n)
    dom_idx   = np.random.randint(0, len(domains), size=n)

    emails = []
    for i in range(n):
        f, l, p, rn, d = fn_clean[i], ln_clean[i], patterns[i], rand_nums[i], dom_idx[i]
        if p == 0:   local = f"{f}.{l}"
        elif p == 1: local = f"{f}{rn}"
        elif p == 2: local = f"{f[0]}{l}"
        else:        local = f"{l}.{f[0]}"
        emails.append(f"{local}@{domains[d]}")

    # UUIDs generated in batch using the default faker instance
    uuids = [f"CLI-{_faker_default.uuid4()[:8].upper()}" for _ in range(n)]

    return pd.DataFrame({
        "client_id":     uuids,
        "first_name":    first_names,
        "last_name":     last_names,
        "age":           ages,
        "date_of_birth": birth_dates,
        "email":         emails,
        "phone":         phones,
        "country":       countries_arr,
    })


def generate_income_info(n: int, countries_list: np.ndarray) -> pd.DataFrame:
    """
    Generate synthetic financial and economic information for a set of individuals.

    This function simulates income-related variables using probabilistic
    distributions commonly observed in real-world economic data. The generated
    variables include salary, bonuses, expenses, savings behavior, and net worth.
    The simulation incorporates country-specific economic parameters such as
    median salaries and minimum wages.

    Distributions used:
        - Lognormal distribution for salaries and bonus rates (income tends to be right-skewed).
        - Normal distribution for expense ratios (centered around typical spending behavior).
        - Beta distribution for savings rates (bounded percentage values).
        - Pareto distribution for net worth (models wealth inequality).

    Parameters
    ----------
    n : int
        Number of individuals to simulate.

    countries_list : np.ndarray
        Array of country names corresponding to each individual.
        Used to retrieve country-specific median salaries and minimum wages.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - monthly_salary_usd
        - annual_bonus_usd
        - monthly_expenses_usd
        - debt_to_income
        - savings_rate
        - net_worth_usd
    """
    medians   = np.array([COUNTRY_MEDIAN_SALARY_USD[c] for c in countries_list])
    min_wages = np.array([COUNTRY_MIN_WAGE_USD[c]      for c in countries_list])

    salaries      = stats.lognorm.rvs(s=0.7, scale=medians, size=n)
    bonus_rate    = stats.beta.rvs(a=2, b=8, size=n)
    annual_bonus  = salaries * 12 * bonus_rate

    expense_rate  = np.clip(np.random.normal(0.55, 0.1, n), 0.2, 0.95)
    monthly_exp   = min_wages + salaries * expense_rate

    raw_savings   = stats.beta.rvs(a=2, b=5, size=n)
    savings_rate  = np.where(
        monthly_exp < salaries,
        raw_savings,
        np.random.uniform(-0.15, 0.05, n)
    )

    debt_to_income = monthly_exp / salaries

    net_worth = stats.pareto.rvs(b=1.5, size=n) * salaries * 12
    in_debt   = np.random.random(n) < 0.15
    net_worth = np.where(in_debt, -np.abs(net_worth) * 0.3, net_worth)

    return pd.DataFrame({
        "monthly_salary_usd":   np.round(salaries, 2),
        "annual_bonus_usd":     np.round(annual_bonus, 2),
        "monthly_expenses_usd": np.round(monthly_exp, 2),
        "debt_to_income":       debt_to_income,
        "savings_rate":         np.round(savings_rate, 4),
        "net_worth_usd":        np.round(net_worth, 2),
    })


def generate_bank_account_info(
    n: int, salary: np.ndarray, net_worth: np.ndarray
) -> pd.DataFrame:
    """
    Generate synthetic banking account information for a set of individuals.

    This function simulates common banking attributes such as account balances,
    account types, account age, number of financial products, and premium
    customer status. The generated variables are designed to mimic realistic
    financial behaviors observed in retail banking systems.

    Several probabilistic models are used to produce plausible relationships
    between financial variables. In particular, an individual's probability of
    being a premium client is influenced by their net worth, and their account
    balance is derived from their monthly salary and simulated savings behavior.

    Distributions used:
        - Logistic function for premium client probability based on net worth.
        - Bernoulli distribution to determine premium client status.
        - Lognormal distribution to simulate the number of months of salary saved.
        - Uniform distribution to model the age of the bank account.
        - Poisson distribution to generate the number of banking products owned.
        - Categorical distribution to determine account type.

    Parameters
    ----------
    n : int
        Number of individuals (bank clients) to simulate.

    salary : np.ndarray
        Array containing the monthly salary (in USD) for each individual.

    net_worth : np.ndarray
        Array containing the total net worth (in USD) of each individual.
        Influences the probability of being classified as a premium client.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - account_balance_usd
        - account_type
        - account_age_years
        - join_date
        - customer_tenure_days
        - num_products
        - is_premium_client
    """
    today = date.today()

    nw_pos       = np.clip(net_worth, 0, None)
    premium_prob = np.clip(1 / (1 + np.exp(-nw_pos / 50000)), 0.01, 0.95)
    is_premium   = np.random.binomial(1, premium_prob, n)

    months_saved    = np.clip(stats.lognorm.rvs(s=0.8, scale=2.5, size=n), 0.1, 24)
    account_balance = salary * months_saved

    # Vectorized account_type assignment
    account_type = np.where(
        is_premium == 1,
        np.random.choice(["premium", "savings"], n, p=[0.7, 0.3]),
        np.random.choice(["checking", "savings"], n, p=[0.6, 0.4])
    )

    # Vectorized account age
    account_age_years = np.random.uniform(1, 20, n)
    age_days          = (account_age_years * 365.25).astype(int)

    # Vectorized join_date and tenure (no loop)
    join_dates         = [today - timedelta(days=int(d)) for d in age_days]
    customer_tenure    = age_days  # equivalent to (today - join_date).days

    num_products = np.clip(stats.poisson.rvs(mu=2.2, size=n) + 1, 1, 10)

    return pd.DataFrame({
        "account_balance_usd":   np.round(account_balance, 2),
        "account_type":          account_type,
        "account_age_years":     np.round(account_age_years, 1),
        "join_date":             join_dates,
        "customer_tenure_days":  customer_tenure,
        "num_products":          num_products,
        "is_premium_client":     is_premium,
    })


def generate_transaction_info(
    n: int, salary: np.ndarray, is_premium: np.ndarray
) -> pd.DataFrame:
    """
    Generate synthetic monthly banking transaction behavior.

    This function simulates transactional activity for a set of banking
    clients over a monthly period. The generated variables model realistic
    digital banking behavior, including the number of transactions,
    transaction sizes, failed payments, international activity,
    preferred banking channel, and user engagement with online services.

    Premium clients are modeled as more active users of banking services,
    performing more transactions, having higher transaction volumes,
    and engaging more frequently in international transactions.

    Distributions used:
        - Poisson distribution for the number of monthly transactions
          and failed transactions.
        - Exponential distribution for transaction amounts and time since
          last login.
        - Beta distribution for the percentage of international transactions.
        - Categorical distribution for preferred banking channel.

    Parameters
    ----------
    n : int
        Number of clients to simulate.

    salary : np.ndarray
        Array containing the monthly salary (in USD) for each client.
        Influences the expected size of transactions.

    is_premium : np.ndarray
        Binary array indicating whether a client is classified as
        a premium banking customer (1) or a standard customer (0).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - monthly_transactions
        - avg_transaction_amount_usd
        - failed_transactions
        - international_txn_pct
        - preferred_channel
        - last_login_days_ago
        - last_transaction_date
    """
    today = date.today()

    txn_mu             = np.where(is_premium == 1, 25, 12)
    monthly_txn        = np.clip(stats.poisson.rvs(mu=txn_mu, size=n), 1, 100)
    avg_txn_amount     = np.clip(stats.expon.rvs(scale=salary * 0.15, size=n), 1, salary * 3)
    failed_txn         = stats.poisson.rvs(mu=1.2, size=n)

    intl_pct = stats.beta.rvs(a=1.2, b=8, size=n)
    intl_pct = np.clip(
        np.where(is_premium == 1, intl_pct * 1.5, intl_pct), 0, 1
    )

    preferred_channel = np.random.choice(
        ["app", "web", "branch", "ATM"], n, p=[0.50, 0.25, 0.15, 0.10]
    )

    last_login = np.clip(stats.expon.rvs(scale=4, size=n).astype(int), 0, 365)
    last_txn_dates = [
        today - timedelta(days=int(d) + random.randint(0, 10))
        for d in last_login
    ]

    return pd.DataFrame({
        "monthly_transactions":      monthly_txn,
        "avg_transaction_amount_usd": np.round(avg_txn_amount, 2),
        "failed_transactions":        failed_txn,
        "international_txn_pct":      np.round(intl_pct, 4),
        "preferred_channel":          preferred_channel,
        "last_login_days_ago":        last_login,
        "last_transaction_date":      last_txn_dates,
    })


def generate_credit_info(
    n: int, salary: np.ndarray, debt_to_income: np.ndarray
) -> pd.DataFrame:
    """
    Generate synthetic credit profile information for a set of clients.

    This function simulates key credit-related variables commonly used in
    financial risk analysis, credit scoring models, and lending decisions.
    The generated variables approximate realistic relationships between
    income, debt burden, credit quality, and repayment behavior.

    A logistic function is used to approximate the probability of loan
    default based on credit quality and debt burden. Higher credit scores
    decrease default risk, while higher debt-to-income ratios increase it.

    Distributions used:
        - Beta distribution for credit score quality and repayment behavior.
        - Poisson distribution for the number of active loans.
        - Logistic function for loan default probability.

    Parameters
    ----------
    n : int
        Number of clients to simulate.

    salary : np.ndarray
        Monthly salary of each client in USD. Influences the credit card
        limit assigned to the client.

    debt_to_income : np.ndarray
        Debt-to-income ratio for each client. Higher values indicate
        greater financial stress and increase default probability.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - credit_score
        - num_loans
        - loan_default_risk
        - credit_card_limit_usd
        - payment_on_time_pct
    """
    raw_score    = stats.beta.rvs(a=5, b=2, size=n)
    credit_score = np.clip((raw_score * 550 + 300).astype(int), 300, 850)
    num_loans    = np.clip(stats.poisson.rvs(mu=1.5, size=n), 0, 8)

    norm_score       = (credit_score - 300) / 550
    default_risk     = np.clip(
        1 / (1 + np.exp(5 * norm_score - 3 * debt_to_income - 1)), 0.01, 0.99
    )

    score_factor     = credit_score / 850
    cc_limit         = np.clip(salary * 3 * score_factor, 500, 50000)

    payment_pct      = stats.beta.rvs(a=8, b=2, size=n)
    payment_pct      = np.where(
        default_risk > 0.5,
        stats.beta.rvs(a=2, b=5, size=n),
        payment_pct
    )

    return pd.DataFrame({
        "credit_score":           credit_score,
        "num_loans":              num_loans,
        "loan_default_risk":      np.round(default_risk, 4),
        "credit_card_limit_usd":  np.round(cc_limit, 2),
        "payment_on_time_pct":    np.round(payment_pct, 4),
    })


# ---------------------------------------------------------------------------
# Chunk helper
# ---------------------------------------------------------------------------

def _generate_chunk(n: int) -> pd.DataFrame:
    """
    Generate a single chunk of synthetic banking data.

    Internal helper that composes all five generators (client, income,
    bank account, transactions, credit) and concatenates their outputs
    into a single flat DataFrame ready to be written to disk.

    Parameters
    ----------
    n : int
        Number of records to generate in this chunk.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with all columns from the five sub-generators.
    """
    client_df = generate_client_info(n)
    income_df = generate_income_info(n, client_df["country"].values)
    bank_df   = generate_bank_account_info(
        n, income_df["monthly_salary_usd"].values, income_df["net_worth_usd"].values
    )
    txn_df    = generate_transaction_info(
        n, income_df["monthly_salary_usd"].values, bank_df["is_premium_client"].values
    )
    credit_df = generate_credit_info(
        n, income_df["monthly_salary_usd"].values, income_df["debt_to_income"].values
    )
    return pd.concat([client_df, income_df, bank_df, txn_df, credit_df], axis=1)


# ---------------------------------------------------------------------------
# Robust CSV writer with row count verification
# ---------------------------------------------------------------------------

def generate_large_dataset(total_n: int = 50_000_000, chunk_size: int = 100_000):
    """
    Generate a large synthetic dataset in chunks and write it to a CSV file.

    Processes data in fixed-size chunks to avoid memory issues. The first
    chunk is written with the CSV header; subsequent chunks are appended
    without a header. Progress is printed to stdout with per-chunk timing
    and a running ETA. A final row count verification is performed after
    writing to detect any silent write failures.

    Parameters
    ----------
    total_n : int, optional
        Total number of records to generate. Default is 50,000,000.

    chunk_size : int, optional
        Number of records per chunk. Default is 100,000.
    """
    BASE_DIR    = Path(__file__).parent.parent
    DATA_DIR    = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / "synthetic_finance.csv"

    total_chunks = (total_n + chunk_size - 1) // chunk_size
    print(f"🚀 Generating {total_n:,} records in {total_chunks} chunks of {chunk_size:,}...")

    written_rows = 0
    global_start = time.time()

    for chunk_num in range(total_chunks):
        current_size = min(chunk_size, total_n - written_rows)
        print(f"   Chunk {chunk_num + 1}/{total_chunks}...", end=" ", flush=True)
        t0 = time.time()

        chunk = _generate_chunk(current_size)

        # First chunk: write with header; subsequent chunks: append without header
        mode   = "w" if chunk_num == 0 else "a"
        header = chunk_num == 0
        chunk.to_csv(output_path, index=False, mode=mode, header=header)

        elapsed     = time.time() - t0
        written_rows += current_size

        # Compute ETA based on average time per chunk so far
        avg_per_chunk = (time.time() - global_start) / (chunk_num + 1)
        remaining     = (total_chunks - chunk_num - 1) * avg_per_chunk
        eta_min       = remaining / 60

        print(f"✅ ({elapsed:.1f}s) | rows written: {written_rows:,} | ETA: {eta_min:.1f} min")

    total_elapsed = time.time() - global_start
    print(f"\n✅ {total_n:,} records generated in {total_elapsed / 60:.1f} min "
          f"({total_n / total_elapsed:.0f} rows/sec)")
    print(f"📁 Saved to: {output_path}")

    # Final row count integrity check
    print("🔍 Verifying file integrity...", end=" ", flush=True)
    actual = sum(1 for _ in open(output_path, encoding="utf-8")) - 1  # subtract 1 for the header row
    if actual == total_n:
        print(f"✅ {actual:,} rows confirmed.")
    else:
        print(f"⚠️  Expected {total_n:,} rows but found {actual:,}. Check disk space and permissions.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_large_dataset(total_n=50_000, chunk_size=10_000)
    print("👍")