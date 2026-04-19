"""
FEATURE ENGINEERING — EMERALD LANKA FILLING STATION
=====================================================
Converts the date column into rich numeric features for the ML model.

This version includes:
  ✅ All official Sri Lanka public holidays (2025 + 2026), gazette-verified
  ✅ All 12 monthly Poya (Full Moon) days per year
  ✅ Weekend flags
  ✅ Holiday eve & post-holiday features
  ✅ Long-weekend detection
  ✅ Hettipola weekly pola (fair) day
  ✅ Sri Lanka agricultural seasons (affects diesel/farming machinery demand)
  ✅ Paddy harvest & planting seasons (Yala + Maha)
  ✅ School term / school holiday periods
  ✅ Monsoon / dry season flags
  ✅ Month-end salary effect
  ✅ Cyclical date encoding (sin/cos)
  ✅ Lag & rolling average features
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — OFFICIAL SRI LANKA PUBLIC HOLIDAYS
#  Source: Department of Government Printing, Sri Lanka
#  Covers every holiday in your dataset range (Apr 2025 → Jan 2026)
#  plus 2026 for future predictions.
# ══════════════════════════════════════════════════════════════════════════════

# --- 2025 Public Holidays (official gazette) ---
HOLIDAYS_2025 = {
    # --- Poya Days (Full Moon) ---
    "2025-01-13": "Duruthu Poya",
    "2025-02-12": "Navam Poya",
    "2025-03-13": "Madin Poya",
    "2025-04-12": "Bak Poya",
    "2025-05-12": "Vesak Poya",
    "2025-05-13": "Vesak Poya Holiday",   # Day after Vesak (special)
    "2025-06-10": "Poson Poya",
    "2025-07-10": "Esala Poya",
    "2025-08-08": "Nikini Poya",
    "2025-09-07": "Binara Poya",
    "2025-10-06": "Vap Poya",
    "2025-11-05": "Ill Poya",
    "2025-12-04": "Unduvap Poya",
    # --- National / Religious Holidays ---
    "2025-01-01": "New Year's Day",
    "2025-01-14": "Tamil Thai Pongal Day",
    "2025-02-04": "Independence Day",
    "2025-02-26": "Maha Sivarathri Day",
    "2025-03-31": "Id-Ul-Fitr (Ramazan)",
    "2025-04-13": "Sinhala & Tamil New Year Eve",
    "2025-04-14": "Sinhala & Tamil New Year",
    "2025-04-18": "Good Friday",
    "2025-05-01": "Labour Day",
    "2025-06-07": "Id-Ul-Alha (Hajj Festival)",
    "2025-09-05": "Milad-Un-Nabi",
    "2025-10-20": "Deepavali Festival",
    "2025-12-25": "Christmas Day",
}

# --- 2026 Public Holidays (official gazette No. 2438/22) ---
HOLIDAYS_2026 = {
    # --- Poya Days ---
    "2026-01-03": "Duruthu Poya",
    "2026-02-01": "Nawam Poya",
    "2026-03-02": "Medin Poya",
    "2026-04-01": "Bak Poya",
    "2026-05-01": "Vesak Poya",           # Coincides with May Day in 2026
    "2026-05-02": "Day Following Vesak",
    "2026-05-30": "Adhi Poson Poya",
    "2026-06-29": "Poson Poya",
    "2026-07-29": "Esala Poya",
    "2026-08-27": "Nikini Poya",
    "2026-09-26": "Binara Poya",
    "2026-10-25": "Vap Poya",
    "2026-11-24": "Il Poya",
    "2026-12-23": "Unduvap Poya",
    # --- National / Religious Holidays ---
    "2026-01-01": "New Year's Day",
    "2026-01-15": "Tamil Thai Pongal Day",
    "2026-02-04": "Independence Day",
    "2026-02-15": "Maha Sivarathri Day",
    "2026-03-21": "Id-Ul-Fitr (Ramazan)",
    "2026-04-03": "Good Friday",
    "2026-04-13": "Sinhala & Tamil New Year Eve",
    "2026-04-14": "Sinhala & Tamil New Year",
    "2026-05-28": "Id-Ul-Alha (Hajj Festival)",
    "2026-08-26": "Milad-Un-Nabi",
    "2026-11-08": "Deepavali Festival",
    "2026-12-25": "Christmas Day",
}

# Merge all holidays into one dict  {date_string: holiday_name}
ALL_HOLIDAYS = {**HOLIDAYS_2025, **HOLIDAYS_2026}
ALL_HOLIDAY_DATES = pd.to_datetime(list(ALL_HOLIDAYS.keys()))

# ── POYA CATEGORY (separate flag — Poya days close alcohol shops, temples busy)
POYA_DATES = pd.to_datetime([
    d for d, name in ALL_HOLIDAYS.items() if "Poya" in name
])


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — HETTIPOLA LOCAL CONTEXT
#
#  Key local facts (researched):
#  • Hettipola is in Kurunegala District, North Western Province
#  • Main crops: coconut, paddy, rubber → heavy diesel use for tractors/transport
#  • Weekly pola (fair/market) held every WEDNESDAY in Hettipola
#    → market vendors, farmers, trucks all fill up before the pola
#  • Major transport hub between Colombo-Kuliyapitiya-Chilaw
#    → through-traffic affects petrol/diesel demand
# ══════════════════════════════════════════════════════════════════════════════

# Hettipola pola day = Wednesday (dayofweek == 2)
HETTIPOLA_POLA_DAY = 2  # 0=Monday ... 6=Sunday


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SRI LANKA AGRICULTURAL SEASONS
#
#  Kurunegala District farming calendar:
#
#  YALA season  (May → August):
#    - Paddy planting: May–June
#    - Paddy harvest:  August–September
#    → Tractors, lorries, irrigation pumps → HIGH diesel demand
#
#  MAHA season  (October → February):
#    - Paddy planting: October–November
#    - Paddy harvest:  January–February
#    → HIGH diesel demand in planting & harvest months
#
#  Coconut / rubber: harvested year-round, slight peak Oct–Dec
# ══════════════════════════════════════════════════════════════════════════════

def is_yala_planting(month: int) -> int:
    """May–June: Yala paddy planting. Tractors, soil prep → high diesel."""
    return 1 if month in [5, 6] else 0

def is_yala_harvest(month: int) -> int:
    """August–September: Yala paddy harvest. Combine harvesters, lorries."""
    return 1 if month in [8, 9] else 0

def is_maha_planting(month: int) -> int:
    """October–November: Maha paddy planting season."""
    return 1 if month in [10, 11] else 0

def is_maha_harvest(month: int) -> int:
    """January–February: Maha paddy harvest season."""
    return 1 if month in [1, 2] else 0

def is_agricultural_peak(month: int) -> int:
    """Combined peak: any planting or harvest month."""
    return 1 if month in [1, 2, 5, 6, 8, 9, 10, 11] else 0


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SRI LANKA MONSOON / WEATHER SEASONS
#
#  North Western Province (where Hettipola is) rain pattern:
#
#  Southwest Monsoon:  May → September  (heavy rains, people stay home)
#    → May reduce travel, potentially lower fuel sales some days
#  Dry season:         December → April (clear roads, more travel)
#    → Generally higher traffic through Hettipola
#  Inter-monsoon:      October–November (short rains)
#
#  NOTE: This is a proxy feature. If you can get actual rainfall data
#  from a nearby weather station later, add it as a direct feature.
# ══════════════════════════════════════════════════════════════════════════════

def get_monsoon_flag(month: int) -> int:
    """1 if Southwest Monsoon period (May–September), 0 otherwise."""
    return 1 if month in [5, 6, 7, 8, 9] else 0

def get_dry_season_flag(month: int) -> int:
    """1 if main dry season (December–April), 0 otherwise."""
    return 1 if month in [12, 1, 2, 3, 4] else 0


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SCHOOL TERMS (Sri Lanka national school calendar)
#
#  School terms affect traffic patterns:
#  - School days: parents doing school runs (petrol), school buses (diesel)
#  - School holidays: leisure travel increases
#
#  Sri Lanka school terms (approximate — varies by year):
#  Term 1: ~Jan 6 → Apr 5
#  Term 2: ~Apr 20 → Aug 8
#  Term 3: ~Aug 24 → Nov 28
#  School holidays: Apr 5–20, Aug 8–24, Dec (entire month), Jan 1–6
# ══════════════════════════════════════════════════════════════════════════════

def is_school_term(date: pd.Timestamp) -> int:
    """Returns 1 if date falls within a school term, 0 during holidays."""
    month = date.month
    day   = date.day

    # Term 1: January 6 – April 5
    if month == 1 and day >= 6:
        return 1
    if month in [2, 3]:
        return 1
    if month == 4 and day <= 5:
        return 1

    # Term 2: April 20 – August 8
    if month == 4 and day >= 20:
        return 1
    if month in [5, 6, 7]:
        return 1
    if month == 8 and day <= 8:
        return 1

    # Term 3: August 24 – November 28
    if month == 8 and day >= 24:
        return 1
    if month in [9, 10, 11]:
        return 1

    # December and early January = school holidays
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — SALARY / MONTH-END EFFECT
#
#  In Sri Lanka, most workers (government & private) receive salary at
#  end of month (typically around 25th–last day).
#  The week AFTER salary day sees higher fuel purchases.
#  Conversely, the week BEFORE salary (around 18th–24th) can be a slow period.
# ══════════════════════════════════════════════════════════════════════════════

def get_salary_period_flag(day: int) -> int:
    """1 if post-salary period (day 25 – end of month), 0 otherwise."""
    return 1 if day >= 25 else 0

def get_pre_salary_flag(day: int) -> int:
    """1 if pre-salary lean period (day 18–24), 0 otherwise."""
    return 1 if 18 <= day <= 24 else 0


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN FEATURE CREATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with a 'date' column.
    Adds ALL engineered features (except lag/rolling — those are separate).
    Returns the same DataFrame with new columns.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # ── BASIC DATE COMPONENTS ─────────────────────────────────────────────────
    df['day_of_week']  = df['date'].dt.dayofweek      # 0=Monday … 6=Sunday
    df['day_of_month'] = df['date'].dt.day
    df['month']        = df['date'].dt.month
    df['quarter']      = df['date'].dt.quarter
    df['day_of_year']  = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['year']         = df['date'].dt.year

    # ── WEEKEND ───────────────────────────────────────────────────────────────
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)   # Sat=5, Sun=6

    # ── PUBLIC HOLIDAY FLAGS ──────────────────────────────────────────────────
    df['is_public_holiday']  = df['date'].isin(ALL_HOLIDAY_DATES).astype(int)
    df['is_poya_day']        = df['date'].isin(POYA_DATES).astype(int)

    # Holiday type flags (fuel sales behave differently by holiday type)
    sinhala_new_year = pd.to_datetime(["2025-04-13", "2025-04-14",
                                       "2026-04-13", "2026-04-14"])
    vesak_days = pd.to_datetime(["2025-05-12", "2025-05-13",
                                 "2026-05-01", "2026-05-02"])
    df['is_sinhala_new_year'] = df['date'].isin(sinhala_new_year).astype(int)
    df['is_vesak']            = df['date'].isin(vesak_days).astype(int)

    # ── HOLIDAY EVE & POST-HOLIDAY ────────────────────────────────────────────
    # People fill up fuel the day BEFORE a holiday (known consumer behaviour)
    # The day AFTER a holiday also sees a pickup as people return
    df['is_day_before_holiday'] = df['date'].apply(
        lambda d: 1 if (d + pd.Timedelta(days=1)) in ALL_HOLIDAY_DATES else 0
    )
    df['is_day_after_holiday'] = df['date'].apply(
        lambda d: 1 if (d - pd.Timedelta(days=1)) in ALL_HOLIDAY_DATES else 0
    )
    df['is_day_before_poya'] = df['date'].apply(
        lambda d: 1 if (d + pd.Timedelta(days=1)) in POYA_DATES else 0
    )

    # ── LONG WEEKEND DETECTION ────────────────────────────────────────────────
    # A long weekend = public holiday adjacent to a Saturday or Sunday.
    # Creates a strong travel demand spike (people drive to visit family).
    def is_long_weekend(d):
        tomorrow = d + pd.Timedelta(days=1)
        yesterday = d - pd.Timedelta(days=1)
        is_hol = d in ALL_HOLIDAY_DATES
        is_fri = d.dayofweek == 4
        is_mon = d.dayofweek == 0
        next_is_hol = tomorrow in ALL_HOLIDAY_DATES
        prev_is_hol = yesterday in ALL_HOLIDAY_DATES
        if (is_fri and next_is_hol) or (is_mon and prev_is_hol):
            return 1
        if is_hol and (d.dayofweek in [4, 0]):
            return 1
        return 0

    df['is_long_weekend'] = df['date'].apply(is_long_weekend)

    # ── HETTIPOLA LOCAL FEATURES ──────────────────────────────────────────────
    # Weekly pola (market fair) day in Hettipola = Wednesday
    # Vendors, farmers, and supply trucks fill up before / on pola day
    df['is_pola_day']        = (df['day_of_week'] == HETTIPOLA_POLA_DAY).astype(int)
    df['is_day_before_pola'] = (df['day_of_week'] == (HETTIPOLA_POLA_DAY - 1)).astype(int)

    # ── AGRICULTURAL SEASON FEATURES ─────────────────────────────────────────
    df['is_yala_planting']    = df['month'].apply(is_yala_planting)
    df['is_yala_harvest']     = df['month'].apply(is_yala_harvest)
    df['is_maha_planting']    = df['month'].apply(is_maha_planting)
    df['is_maha_harvest']     = df['month'].apply(is_maha_harvest)
    df['is_agri_peak']        = df['month'].apply(is_agricultural_peak)

    # ── MONSOON / SEASON FEATURES ─────────────────────────────────────────────
    df['is_monsoon']          = df['month'].apply(get_monsoon_flag)
    df['is_dry_season']       = df['month'].apply(get_dry_season_flag)

    # ── SCHOOL TERM ───────────────────────────────────────────────────────────
    df['is_school_term']      = df['date'].apply(is_school_term)

    # ── SALARY / MONTH-END EFFECT ─────────────────────────────────────────────
    df['is_post_salary']      = df['day_of_month'].apply(get_salary_period_flag)
    df['is_pre_salary']       = df['day_of_month'].apply(get_pre_salary_flag)

    # ── CYCLICAL ENCODING ─────────────────────────────────────────────────────
    # Converts periodic features into continuous sin/cos pairs so that
    # the model understands "Sunday is adjacent to Monday" (circular week).
    df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week']  / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week']  / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']        / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']        / 12)
    df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year']  / 365)
    df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year']  / 365)

    # ── TREND ─────────────────────────────────────────────────────────────────
    # Linear counter — helps model learn if overall sales are growing over time.
    df['trend'] = range(len(df))

    return df


def create_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
    """
    Lag features: sales value N days ago.

    lag_1 = yesterday's sales  (most powerful feature)
    lag_7 = sales exactly one week ago (same day of week effect)
    lag_14 = two weeks ago
    lag_30 = one month ago (seasonal comparison)
    """
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str, windows: list) -> pd.DataFrame:
    """
    Rolling mean: average of past N days.
    Uses shift(1) so we never leak today's value into today's prediction.

    rolling_3  = 3-day average  (very short-term trend)
    rolling_7  = 7-day average  (weekly trend)
    rolling_14 = 2-week average (bi-weekly trend)
    rolling_30 = monthly average (monthly baseline)
    """
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_{window}'] = (
            df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        )
    return df


def create_rolling_std_features(df: pd.DataFrame, target_col: str, windows: list) -> pd.DataFrame:
    """
    Rolling standard deviation: measures how volatile recent sales have been.
    A high std means sales have been unpredictable — useful for super fuels.
    """
    df = df.copy()
    for window in windows:
        df[f'{target_col}_std_{window}'] = (
            df[target_col].shift(1).rolling(window=window, min_periods=2).std()
        )
    return df


def prepare_features_for_fuel(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Full feature preparation pipeline for one fuel type.
    Call this with a DataFrame containing 'date' and the target column only.

    Returns a fully featured DataFrame, with NaN rows dropped.
    """
    df = create_features(df)
    df = create_lag_features(df, target_col, lags=[1, 2, 3, 7, 14, 30])
    df = create_rolling_features(df, target_col, windows=[3, 7, 14, 30])
    df = create_rolling_std_features(df, target_col, windows=[7, 14])

    # Drop rows where lag/rolling features could not be computed
    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns(target_col: str) -> list:
    """
    Returns the ordered list of feature column names used as model inputs.
    Must stay consistent between training and prediction.
    """
    base_features = [
        # Date components
        'day_of_week', 'day_of_month', 'month', 'quarter',
        'day_of_year', 'week_of_year', 'year',
        # Weekend & holiday
        'is_weekend', 'is_public_holiday', 'is_poya_day',
        'is_sinhala_new_year', 'is_vesak',
        'is_day_before_holiday', 'is_day_after_holiday',
        'is_day_before_poya', 'is_long_weekend',
        # Hettipola local
        'is_pola_day', 'is_day_before_pola',
        # Agricultural seasons
        'is_yala_planting', 'is_yala_harvest',
        'is_maha_planting', 'is_maha_harvest', 'is_agri_peak',
        # Weather / season
        'is_monsoon', 'is_dry_season',
        # School
        'is_school_term',
        # Economic
        'is_post_salary', 'is_pre_salary',
        # Cyclical
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        # Trend
        'trend',
    ]
    lag_features     = [f'{target_col}_lag_{l}'    for l in [1, 2, 3, 7, 14, 30]]
    rolling_features = [f'{target_col}_rolling_{w}' for w in [3, 7, 14, 30]]
    std_features     = [f'{target_col}_std_{w}'     for w in [7, 14]]

    return base_features + lag_features + rolling_features + std_features


# ── FEATURE SUMMARY ──────────────────────────────────────────────────────────

def print_feature_summary():
    """Print a human-readable summary of all features."""
    sample_df = pd.DataFrame({'date': pd.date_range('2025-04-01', periods=60), 'petrol_sales': 2500.0})
    sample_df = prepare_features_for_fuel(sample_df, 'petrol_sales')
    cols = get_feature_columns('petrol_sales')
    print(f"\nTotal features: {len(cols)}")
    print("\nFeature breakdown:")
    groups = {
        'Date components (7)':   [c for c in cols if c in ['day_of_week','day_of_month','month','quarter','day_of_year','week_of_year','year']],
        'Holiday/Poya (9)':      [c for c in cols if 'holiday' in c or 'poya' in c or 'vesak' in c or 'new_year' in c or 'long_weekend' in c],
        'Hettipola local (2)':   [c for c in cols if 'pola' in c],
        'Agricultural season (5)': [c for c in cols if 'yala' in c or 'maha' in c or 'agri' in c],
        'Weather/season (3)':    [c for c in cols if 'monsoon' in c or 'dry_season' in c or 'weekend' in c],
        'School (1)':            [c for c in cols if 'school' in c],
        'Economic (2)':          [c for c in cols if 'salary' in c],
        'Cyclical (6)':          [c for c in cols if '_sin' in c or '_cos' in c],
        'Trend (1)':             [c for c in cols if c == 'trend'],
        'Lag features (6)':      [c for c in cols if '_lag_' in c],
        'Rolling mean (4)':      [c for c in cols if '_rolling_' in c],
        'Rolling std (2)':       [c for c in cols if '_std_' in c],
    }
    for group, features in groups.items():
        print(f"  {group}: {features}")


if __name__ == "__main__":
    print_feature_summary()