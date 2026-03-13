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

    salaries = np.array([
        stats.lognorm.rvs(
            s=0.7,
            scale=COUNTRY_MEDIAN_SALARY_USD[country]
        )
        for country in countries_list
    ])

    bonus_rate = np.array([
        stats.lognorm.rvs(s=0.1, scale=0.08 + salary / COUNTRY_MEDIAN_SALARY_USD[country])
        for country, salary in zip(countries_list, salaries)
    ])

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
