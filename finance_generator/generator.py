import random
import numpy as np
import pandas as pd
from scipy import stats
from faker import Faker
from datetime import date, timedelta, datetime

fake = Faker()

countries = [
    "Colombia", "Mexico", "Argentina", "Chile", "Peru",
    "Brazil", "Ecuador", "Venezuela", "Bolivia", "Paraguay",
    "Uruguay", "Panama", "Costa Rica", "Guatemala", "Dominican Republic",
    "United States", "Spain", "Canada", "Germany", "France"
]

probabilities = [
    0.25,  # Colombia
    0.20,  # Mexico
    0.10,  # Argentina
    0.06,  # Chile
    0.06,  # Peru
    0.05,  # Brazil
    0.04,  # Ecuador
    0.04,  # Venezuela
    0.03,  # Bolivia
    0.02,  # Paraguay
    0.02,  # Uruguay
    0.02,  # Panama
    0.02,  # Costa Rica
    0.02,  # Guatemala
    0.02,  # Dominican Republic
    0.02,  # United States
    0.01,  # Spain
    0.01,  # Canada
    0.005, # Germany
    0.005  # France
]

locale_map = {
    "Colombia": "es_CO",
    "Mexico": "es_MX",
    "Argentina": "es_AR",
    "Chile": "es_CL",
    "Peru": "es_PE",
    "Brazil": "pt_BR",
    "Ecuador": "es_EC",
    "Venezuela": "es_VE",
    "Bolivia": "es_BO",
    "Paraguay": "es_PY",
    "Uruguay": "es_UY",
    "Panama": "es_PA",
    "Costa Rica": "es_CR",
    "Guatemala": "es_GT",
    "Dominican Republic": "es_DO",
    "United States": "en_US",
    "Spain": "es_ES",
    "Canada": "en_CA",
    "Germany": "de_DE",
    "France": "fr_FR",
}

# Median monthly salary in USD (net, after taxes) — 2025/2026
# Sources: BLS, Numbeo, Statista/Bloomberg, SalaryExplorer, ILO
COUNTRY_MEDIAN_SALARY_USD = {
    "Colombia":            700,   # ~COP 2,900,000 net; your $800 was slightly high
    "Mexico":              850,   # ~MXN 14,000 net; your $900 was close
    "Argentina":           500,   # very volatile; tied to official vs. parallel exchange rate
    "Chile":             1_100,   # CLP has depreciated; Numbeo avg ~$690, median ~$1,100 gross
    "Peru":                650,   # ~PEN 2,400/month; your $700 was close
    "Brazil":              750,   # ~BRL 4,500 net; your $1,000 was high
    "Ecuador":             600,   # dollarized economy; confirmed
    "Venezuela":           130,   # real incomes ~$100–200; your $200 was generous
    "Bolivia":             450,   # ~BOB 3,100/month; confirmed
    "Paraguay":            520,   # ~PYG 3,700,000/month; your $550 was close
    "Uruguay":             900,   # UYU median ~29,600 (~$760 net); your $1,600 was too high
    "Panama":            1_100,   # Numbeo avg ~$1,000; your $1,200 was slightly high
    "Costa Rica":          950,   # Numbeo/Statista avg ~$947; your $1,100 was slightly high
    "Guatemala":           480,   # ~GTQ 3,700/month; your $500 was close
    "Dominican Republic":  600,   # highly sector-dependent ($400–$800); your $700 was ok
    "United States":     5_174,   # BLS 2025 confirmed median; your $5,500 was slightly high
    "Spain":             2_000,   # ~€1,800 net median; your $2,200 was slightly high
    "Canada":            3_200,   # ~CAD 4,400 net median; your $4,500 was high
    "Germany":           3_400,   # ~€3,100 net median; your $4,000 was high
    "France":            2_800,   # ~€2,600 net median; your $3,500 was high
}

# Monthly minimum wage in USD (approx. 2025–2026)
# Sources: World Population Review, Nearshore Americas, WowRemoteTeams
COUNTRY_MIN_WAGE_USD = {
    "Colombia":            384,   # COP ~1,423,500 + transportation subsidy
    "Mexico":              501,   # general zone (~$312) / border zone (~$501)
    "Argentina":           287,   # volatile due to inflation and exchange rate
    "Chile":               555,   # CLP ~515,500
    "Peru":                297,   # PEN ~1,130
    "Brazil":              262,   # BRL ~1,518 (+ mandatory 13th month salary)
    "Ecuador":             482,   # dollarized economy; raised to $480 in 2025
    "Venezuela":             1,   # practically nonexistent (~$0.33 real value)
    "Bolivia":             469,   # increased significantly in 2025
    "Paraguay":            406,
    "Uruguay":             632,   # highest minimum wage in South America
    "Panama":              543,   # varies by sector (~$341–$1,015)
    "Costa Rica":          726,   # also varies by sector and skill level
    "Guatemala":           513,   # includes mandatory bonus (bonificación de ley)
    "Dominican Republic":  200,   # highly variable by sector (~$150–$450)
    "United States":      1256,   # federal: $7.25/hr (many states pay higher)
    "Spain":              1636,   # SMI 2025: ~€1,134/month
    "Canada":             2192,   # varies by province (~$16–$17.75 CAD/hr)
    "Germany":            2829,   # €12.82/hr since Jan 2025
    "France":             2160,   # SMIC ~€1,801/month
}


domains = ["gmail.com", "hotmail.com", "yahoo.com", "outlook.com", "icloud.com"]

def phone_for_country(country: str) -> str:
    """
    Generate a realistic phone number based on the country.

    This function selects an appropriate Faker locale according to the
    provided country and generates a phone number formatted according
    to that locale. If the locale is not supported or an error occurs,
    a fallback locale is used.

    Parameters
    ----------
    country : str
        Name of the country used to determine the locale.

    Returns
    -------
    str
        A randomly generated phone number formatted according to the
        selected locale.
    """
    locale = locale_map.get(country, "es_ES")
    try:
        return Faker(locale).phone_number()
    except Exception:
        return Faker("es_ES").phone_number()

def generate_email(first_name: str, last_name: str) -> str:
    """
    Generate a realistic email address using name-based patterns.

    The function creates an email by combining the first and last name
    in different common formats (e.g., firstname.lastname, firstinitiallastname,
    etc.) and attaches a randomly selected email domain.

    Parameters
    ----------
    first_name : str
        First name of the person.
    last_name : str
        Last name of the person.

    Returns
    -------
    str
        A randomly generated email address.
    """
    first = first_name.lower().replace(" ", "")
    last = last_name.lower().replace(" ", "")
    pattern = random.choice([
        f"{first}.{last}",
        f"{first}{random.randint(1, 99)}",
        f"{first[0]}{last}",
        f"{last}.{first[0]}",
    ])
    domain = random.choice(domains)
    return f"{pattern}@{domain}"

def generate_client_info(n: int) -> pd.DataFrame:
    """
    Generate a synthetic dataset of client information.

    This function creates a DataFrame containing randomly generated
    customer data such as names, ages, birthdays, emails, phone numbers,
    and country of origin. Ages are sampled from a truncated normal
    distribution to produce realistic values within a defined range.

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

    age_dist = stats.truncnorm(
        a=(18 - 40) / 10,
        b=(70 - 40) / 10,
        loc=40,
        scale=10
    )

    countries_list = np.random.choice(countries, size=n, p=probabilities)
    first_names = [fake.first_name() for _ in range(n)]
    last_names = [fake.last_name() for _ in range(n)]
    ages = age_dist.rvs(n).astype(int)
    birth_dates = [
        today - timedelta(days=int(age * 365.25) + random.randint(0, 365))
        for age in ages
    ]
    emails = [generate_email(f, l) for f, l in zip(first_names, last_names)]
    phones = [phone_for_country(c) for c in countries_list]

    return pd.DataFrame({
        "client_id": [f"CLI-{fake.uuid4()[:8].upper()}" for _ in range(n)],
        "first_name": first_names,
        "last_name": last_names,
        "age": ages,
        "date_of_birth": birth_dates,
        "email": emails,
        "phone": phones,
        "country": countries_list
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

    Args:
        n (int):
            Number of individuals to simulate.

        countries_list (np.ndarray):
            Array of country names corresponding to each individual.
            The values are used to retrieve country-specific parameters
            such as median salaries and minimum wages.

    Returns:
        pd.DataFrame:
            A pandas DataFrame containing the following synthetic financial variables:

            - monthly_salary_usd (float):
                Simulated monthly salary in USD, generated using a lognormal
                distribution centered around the country's median salary.

            - annual_bonus_usd (float):
                Annual bonus amount in USD, derived from a stochastic bonus
                rate applied to yearly salary.

            - monthly_expenses_usd (float):
                Estimated monthly expenses in USD, calculated as the sum of
                a country's minimum wage baseline and a variable proportion
                of the individual's salary.

            - savings_rate (float):
                Proportion of income saved by the individual. Typically between
                0 and 1, but may become negative if expenses exceed income.

            - net_worth_usd (float):
                Total estimated net worth in USD, generated using a Pareto
                distribution to simulate wealth inequality. A fraction of
                individuals may have negative net worth to represent debt.

    Notes:
        - The function assumes the existence of the following global dictionaries:
            COUNTRY_MEDIAN_SALARY_USD
            COUNTRY_MIN_WAGE_USD

        - Values are rounded for readability but remain suitable for
          statistical analysis or synthetic dataset generation.

        - The generated data is intended for simulations, testing,
          machine learning experiments, or synthetic data pipelines.
    """

    medians = np.array([COUNTRY_MEDIAN_SALARY_USD[c] for c in countries_list])

    salaries = stats.lognorm.rvs(s=0.7, scale=medians, size=n)

    bonus_scale = 0.08 + salaries / medians

    bonus_rate = stats.lognorm.rvs(s=0.1, scale=bonus_scale, size=n)

    annual_bonus_usd = salaries * 12 * bonus_rate

    min_wages = np.array([COUNTRY_MIN_WAGE_USD[c] for c in countries_list])
    expense_rate = np.random.normal(loc=0.55, scale=0.1, size=n)
    expense_rate = np.clip(expense_rate, 0.2, 0.95)
    monthly_expenses_usd = min_wages + salaries * expense_rate

    raw_savings = stats.beta.rvs(a=2, b=5, size=n)
    savings_rate = np.where(
        monthly_expenses_usd < salaries,
        raw_savings,
        np.random.uniform(-0.2, 0, size=n)
    )

    net_worth_usd = stats.pareto.rvs(b=1.5, size=n) * salaries * 12
    is_in_debt = np.random.random(size=n) < 0.15
    net_worth_usd = np.where(is_in_debt, -np.abs(net_worth_usd) * 0.3, net_worth_usd)

    return pd.DataFrame({
        "monthly_salary_usd": np.round(salaries, 2),
        "annual_bonus_usd": np.round(annual_bonus_usd, 2),
        "monthly_expenses_usd": np.round(monthly_expenses_usd, 2),
        "savings_rate": np.round(savings_rate, 4),
        "net_worth_usd": np.round(net_worth_usd, 2),
    })

def generate_bank_account_info(
    n: int,
    salary: np.ndarray,
    net_worth: np.ndarray
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

    Args:
        n (int):
            Number of individuals (bank clients) to simulate.

        salary (np.ndarray):
            Array containing the monthly salary (in USD) for each individual.
            This variable is used to estimate account balances by modeling
            how many months of salary have been accumulated as savings.

        net_worth (np.ndarray):
            Array containing the total net worth (in USD) of each individual.
            This value influences the probability that a client is classified
            as a premium banking customer.

    Returns:
        pd.DataFrame:
            A pandas DataFrame containing the following synthetic banking variables:

            - account_balance_usd (float):
                Estimated account balance in USD. This value is calculated as
                the product of monthly salary and a stochastic number of months
                of accumulated savings generated from a lognormal distribution.
                The distribution is truncated to prevent unrealistic values.

            - account_type (str):
                Type of bank account held by the client. Possible values include:
                "checking", "savings", or "premium". Premium clients are more
                likely to hold premium accounts, while standard clients are
                distributed between checking and savings accounts.

            - account_age_years (float):
                Age of the bank account in years, simulated using a uniform
                distribution between 1 and 20 years. This represents the length
                of the relationship between the client and the bank.

            - num_products (int):
                Number of banking products owned by the client, such as credit
                cards, loans, savings accounts, or investment products. This
                variable is generated using a Poisson distribution to model
                discrete event counts, with a minimum of one product per client.

            - is_premium_client (int):
                Binary indicator (0 or 1) specifying whether the client is
                classified as a premium banking customer. The probability of
                being premium is determined using a logistic function of the
                individual's net worth, reflecting the tendency for wealthier
                clients to receive premium banking services.

    Notes:
        - Negative net worth values are clipped to zero when calculating the
          premium client probability, ensuring that indebted individuals do
          not receive artificially inflated probabilities.

        - Account balances are derived from salary-based savings behavior
          rather than directly from net worth, which helps maintain realistic
          correlations between income and liquid assets.

        - The Poisson distribution used for the number of banking products
          reflects the fact that most customers own a small number of financial
          products, while a minority hold several.

        - The generated dataset is suitable for financial simulations,
          machine learning experiments, banking analytics prototypes,
          and synthetic data pipelines.
    """

    net_worth_positive = np.clip(net_worth, 0, None)

    premium_probability = 1 / (1 + np.exp(-net_worth_positive / 50000))
    premium_probability = np.clip(premium_probability, 0.01, 0.95)

    is_premium_client = np.random.binomial(1, premium_probability, size=n)

    months_saved = stats.lognorm.rvs(s=0.8, scale=2.5, size=n)
    months_saved = np.clip(months_saved, 0.1, 24)

    account_balance = salary * months_saved

    account_type = np.where(
        is_premium_client == 1,
        np.random.choice(["premium", "savings"], size=n, p=[0.7, 0.3]),
        np.random.choice(["checking", "savings"], size=n, p=[0.6, 0.4])
    )

    account_age_years = np.random.uniform(1, 20, size=n)

    num_products = stats.poisson.rvs(mu=2.2, size=n) + 1
    num_products = np.clip(num_products, 1, 10)

    return pd.DataFrame({
        "account_balance_usd": np.round(account_balance, 2),
        "account_type": account_type,
        "account_age_years": np.round(account_age_years, 1),
        "num_products": num_products,
        "is_premium_client": is_premium_client
    })

def generate_transaction_info(
    n: int,
    salary: np.ndarray,
    is_premium: np.ndarray
) -> pd.DataFrame:
    """
    Generate synthetic monthly banking transaction behavior.

    This function simulates transactional activity for a set of banking
    clients over a monthly period. The generated variables model realistic
    digital banking behavior, including the number of transactions,
    transaction sizes, failed payments, international activity,
    preferred banking channel, and user engagement with online services.

    The simulation relies on several probability distributions commonly
    used to represent financial and behavioral data. In particular,
    discrete distributions are used for event counts (such as the number
    of transactions), while continuous distributions are used for
    monetary values, time intervals, and proportions.

    Premium clients are modeled as more active users of banking services,
    performing more transactions, having higher transaction volumes,
    and engaging more frequently in international transactions.

    Distributions used:
        - Poisson distribution for the number of monthly transactions
          and failed transactions (counts of events in a fixed time period).
        - Exponential distribution for transaction amounts and time since
          last login (many small values with fewer large values).
        - Beta distribution for the percentage of international
          transactions (bounded between 0 and 1).
        - Categorical distribution for preferred banking channel.

    Parameters
    ----------
    n : int
        Number of clients to simulate.

    salary : np.ndarray
        Array containing the monthly salary (in USD) for each client.
        Salary influences the expected size of transactions, assuming
        that higher-income individuals tend to perform larger financial
        transactions on average.

    is_premium : np.ndarray
        Binary array indicating whether a client is classified as
        a premium banking customer (1) or a standard customer (0).
        Premium clients are assumed to perform more transactions,
        transact larger amounts, and have a slightly higher share
        of international payments.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing synthetic transactional variables:

        - monthly_transactions (int):
            Total number of transactions performed during the month.
            Generated using a Poisson distribution with different
            expected values depending on premium status.

        - avg_transaction_amount_usd (float):
            Average monetary value of a transaction in USD. This value
            is generated using an exponential distribution scaled by
            the client's salary to reflect realistic spending behavior.

        - failed_transactions (int):
            Number of failed transactions during the month, such as
            declined payments, insufficient funds, or technical errors.
            Simulated using a low-rate Poisson distribution.

        - international_txn_pct (float):
            Proportion of transactions that are international
            (values between 0 and 1). This variable is generated using
            a Beta distribution, producing realistic skew toward
            low percentages. Premium clients tend to have slightly
            higher international activity.

        - preferred_channel (str):
            Client's primary banking interaction channel. Possible values:
            "app", "web", "branch", or "ATM". The probabilities reflect
            modern banking behavior where mobile applications dominate.

        - last_login_days_ago (int):
            Number of days since the client's last login to the banking
            platform. Generated using an exponential distribution,
            reflecting the tendency for most users to log in frequently
            while some remain inactive for longer periods.

    Notes
    -----
    - The Poisson distribution is used for modeling discrete event counts
      such as transactions or failures within a fixed time window.

    - The exponential distribution captures the heavy concentration of
      small financial transactions with occasional larger values.

    - The Beta distribution ensures that percentages remain bounded
      between 0 and 1 while allowing flexible skew toward lower values.

    - The generated dataset can be used for simulations, financial
      analytics prototypes, synthetic banking datasets, behavioral
      modeling, or machine learning experiments such as fraud detection,
      churn prediction, or transaction pattern analysis.
    """

    txn_mu = np.where(is_premium == 1, 25, 12)
    monthly_transactions = stats.poisson.rvs(mu=txn_mu, size=n)
    monthly_transactions = np.clip(monthly_transactions, 1, 100)

    avg_txn_amount = stats.expon.rvs(scale=salary * 0.15, size=n)
    avg_txn_amount = np.clip(avg_txn_amount, 1, salary * 3)

    failed_transactions = stats.poisson.rvs(mu=1.2, size=n)

    international_txn_pct = stats.beta.rvs(a=1.2, b=8, size=n)
    international_txn_pct = np.where(is_premium == 1,
                                      international_txn_pct * 1.5,
                                      international_txn_pct)
    international_txn_pct = np.clip(international_txn_pct, 0, 1)

    preferred_channel = np.random.choice(
        ["app", "web", "branch", "ATM"],
        size=n,
        p=[0.50, 0.25, 0.15, 0.10]
    )

    last_login_days_ago = stats.expon.rvs(scale=4, size=n).astype(int)
    last_login_days_ago = np.clip(last_login_days_ago, 0, 365)

    return pd.DataFrame({
        "monthly_transactions": monthly_transactions,
        "avg_transaction_amount_usd": np.round(avg_txn_amount, 2),
        "failed_transactions": failed_transactions,
        "international_txn_pct": np.round(international_txn_pct, 4),
        "preferred_channel": preferred_channel,
        "last_login_days_ago": last_login_days_ago
    })