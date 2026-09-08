import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, date, timedelta

# ============================================================
# GOLDEN QOD SCANNER
# Production Tradier Market Data
# ============================================================

st.set_page_config(
    page_title="Golden QOD Scanner",
    page_icon="🏆",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://api.tradier.com/v1"

MIN_STOCK_PRICE = 10.00
MIN_RELATIVE_VOLUME = 1.20

MIN_SCORE = 76
MAX_RESULTS = 15

MIN_DTE = 7
MAX_DTE = 40

MIN_DELTA = 0.40
MAX_DELTA = 0.99

MIN_OPEN_INTEREST = 100
MIN_OPTION_VOLUME = 10

MAX_SPREAD_PCT = 0.20

# Chuck Hughes-style 1% time-value rule
HUGHES_TIME_VALUE_PCT = 0.01

# User's requested option stop:
# stop = 70% of entry premium
STOP_PERCENT_OF_PREMIUM = 0.70

# Minimum modeled reward/risk
MIN_RISK_REWARD = 1.25

# Historical lookback
HISTORY_DAYS = 450

# Account sizing
DEFAULT_ACCOUNT_SIZE = 5000
RISK_PER_TRADE = 0.02

# Mag 7 exclusion
MAG_7 = {
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA"
}

# ============================================================
# STOCK / ETF UNIVERSE
# ============================================================

UNIVERSE = [
    # Broad ETFs
    "SPY", "QQQ", "IWM", "DIA",
    "XLF", "XLK", "XLE", "XLV",
    "XLI", "XLP", "XLY", "XLB",
    "XLU", "SMH", "XBI",

    # Technology
    "AMD", "AVGO", "CRM", "ORCL", "ADBE",
    "NOW", "INTC", "MU", "MRVL", "DELL",
    "CSCO", "IBM", "QCOM", "TXN",

    # Financial
    "V", "MA", "JPM", "BAC", "GS",
    "MS", "C", "COF", "AXP", "PYPL",
    "HOOD",

    # Consumer
    "WMT", "COST", "HD", "LOW",
    "NKE", "TGT", "KO", "PEP", "MCD",

    # Industrials
    "CAT", "DE", "GE", "HON", "RTX",
    "BA", "UPS",

    # Energy
    "XOM", "CVX", "SLB",

    # Healthcare
    "UNH", "LLY", "JNJ", "ABBV",
    "PFE", "MRK", "ABT",

    # Communication / internet / growth
    "DIS", "NFLX", "UBER", "SHOP",
    "CRWD", "COIN", "PLTR"
]

UNIVERSE = [
    symbol for symbol in UNIVERSE
    if symbol not in MAG_7
]

# ============================================================
# AUTHENTICATION
# ============================================================

try:
    TOKEN = st.secrets["TRADIER_API_TOKEN"]
except Exception:
    st.error(
        "🔴 Tradier production token not found.\n\n"
        "Open Streamlit → Settings → Secrets and add:\n\n"
        "TRADIER_API_TOKEN = \"YOUR_PRODUCTION_TOKEN\""
    )
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json"
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return default

        return float(value)
    except Exception:
        return default


def flatten_records(value):
    """
    Tradier sometimes returns one object and sometimes a list.
    Normalize both into a list.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        return [value]

    return []


def get_nested(data, *keys):
    value = data

    for key in keys:
        if not isinstance(value, dict):
            return None

        value = value.get(key)

    return value


def request_json(endpoint, params=None, timeout=20):
    """
    Production Tradier request helper.
    """
    url = f"{BASE_URL}{endpoint}"

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params or {},
            timeout=timeout
        )

        if response.status_code == 401:
            raise RuntimeError(
                "Tradier returned 401 Unauthorized. "
                "Verify that Streamlit is using your PRODUCTION Tradier token."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Tradier HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        return response.json()

    except requests.RequestException as exc:
        raise RuntimeError(f"Tradier connection error: {exc}")


# ============================================================
# TRADIER MARKET DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_history(symbol):
    end_date = date.today()
    start_date = end_date - timedelta(days=HISTORY_DAYS)

    data = request_json(
        "/markets/history",
        {
            "symbol": symbol,
            "interval": "daily",
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        }
    )

    history = get_nested(data, "history", "day")

    rows = flatten_records(history)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "date" not in df.columns:
        return pd.DataFrame()

    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "close"]
    ).sort_values("date")

    return df.reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def get_quote(symbol):
    data = request_json(
        "/markets/quotes",
        {
            "symbols": symbol,
            "greeks": "true"
        }
    )

    quote = get_nested(data, "quotes", "quote")

    rows = flatten_records(quote)

    if not rows:
        return {}

    return rows[0]


@st.cache_data(ttl=300, show_spinner=False)
def get_expirations(symbol):
    data = request_json(
        "/markets/options/expirations",
        {
            "symbol": symbol,
            "includeAllRoots": "false",
            "strikes": "false",
            "contractSize": "false",
            "expirationType": "true"
        }
    )

    expirations = get_nested(
        data,
        "expirations",
        "date"
    )

    if expirations is None:
        expirations = get_nested(
            data,
            "expirations"
        )

    if isinstance(expirations, dict):
        expirations = expirations.get("date")

    if expirations is None:
        return []

    if not isinstance(expirations, list):
        expirations = [expirations]

    result = []

    for item in expirations:
        if isinstance(item, dict):
            value = item.get("date")
        else:
            value = item

        if value:
            result.append(str(value))

    return sorted(set(result))


@st.cache_data(ttl=120, show_spinner=False)
def get_option_chain(symbol, expiration):
    data = request_json(
        "/markets/options/chains",
        {
            "symbol": symbol,
            "expiration": expiration,
            "greeks": "true"
        }
    )

    options = get_nested(
        data,
        "options",
        "option"
    )

    rows = flatten_records(options)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi.fillna(50)


def calculate_atr(df, period=14):
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] - previous_close
    ).abs()

    tr3 = (
        df["low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def calculate_macd(series):
    ema12 = calculate_ema(series, 12)
    ema26 = calculate_ema(series, 26)

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = macd - signal

    return macd, signal, histogram


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = calculate_ema(
        df["close"],
        20
    )

    df["EMA50"] = calculate_ema(
        df["close"],
        50
    )

    df["EMA100"] = calculate_ema(
        df["close"],
        100
    )

    df["EMA200"] = calculate_ema(
        df["close"],
        200
    )

    df["RSI"] = calculate_rsi(
        df["close"]
    )

    (
        df["MACD"],
        df["MACD_SIGNAL"],
        df["MACD_HIST"]
    ) = calculate_macd(
        df["close"]
    )

    df["ATR"] = calculate_atr(
        df
    )

    df["Volume20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["RelativeVolume"] = (
        df["volume"]
        / df["Volume20"]
    )

    # Keltner Channel
    df["KC_MID"] = df["EMA20"]

    df["KC_UPPER"] = (
        df["KC_MID"]
        + 2 * df["ATR"]
    )

    df["KC_LOWER"] = (
        df["KC_MID"]
        - 2 * df["ATR"]
    )

    # Momentum
    df["Momentum10"] = (
        df["close"].pct_change(10) * 100
    )

    df["Momentum20"] = (
        df["close"].pct_change(20) * 100
    )

    # Candle body
    df["CandleBody"] = (
        df["close"] - df["open"]
    )

    df["CandleRange"] = (
        df["high"] - df["low"]
    )

    df["BodyPct"] = np.where(
        df["CandleRange"] > 0,
        abs(df["CandleBody"])
        / df["CandleRange"],
        0
    )

    return df


# ============================================================
# 50 / 100 EMA CROSSOVER
# ============================================================

def find_recent_crossover(df, lookback=30):
    if len(df) < 105:
        return None

    recent = df.tail(
        lookback + 1
    ).copy()

    bullish = (
        (recent["EMA50"] > recent["EMA100"])
        &
        (
            recent["EMA50"].shift(1)
            <=
            recent["EMA100"].shift(1)
        )
    )

    bearish = (
        (recent["EMA50"] < recent["EMA100"])
        &
        (
            recent["EMA50"].shift(1)
            >=
            recent["EMA100"].shift(1)
        )
    )

    bullish_dates = recent.index[
        bullish.fillna(False)
    ]

    bearish_dates = recent.index[
        bearish.fillna(False)
    ]

    candidates = []

    for idx in bullish_dates:
        candidates.append(
            ("BULLISH", idx)
        )

    for idx in bearish_dates:
        candidates.append(
            ("BEARISH", idx)
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[1]
    )

    direction, idx = candidates[-1]

    return {
        "direction": direction,
        "index": idx,
        "date": recent.loc[idx, "date"]
    }


# ============================================================
# KELTNER ANALYSIS
# ============================================================

def analyze_keltner(df, direction):
    if len(df) < 30:
        return False, "Insufficient Keltner history"

    recent = df.tail(15)

    latest = df.iloc[-1]

    if direction == "BULLISH":

        pullback = (
            recent["low"]
            <= recent["KC_MID"] * 1.015
        ).any()

        momentum_recovery = (
            latest["close"]
            > latest["KC_MID"]
        )

        if pullback and momentum_recovery:
            return True, "Keltner pullback + recovery"

        if (
            latest["close"]
            > latest["KC_MID"]
            and
            latest["close"]
            < latest["KC_UPPER"]
        ):
            return True, "Keltner bullish momentum"

    else:

        pullback = (
            recent["high"]
            >= recent["KC_MID"] * 0.985
        ).any()

        momentum_recovery = (
            latest["close"]
            < latest["KC_MID"]
        )

        if pullback and momentum_recovery:
            return True, "Keltner pullback + rejection"

        if (
            latest["close"]
            < latest["KC_MID"]
            and
            latest["close"]
            > latest["KC_LOWER"]
        ):
            return True, "Keltner bearish momentum"

    return False, "No Keltner confirmation"


# ============================================================
# PRICE ACTION
# ============================================================

def analyze_price_action(df, direction):
    if len(df) < 5:
        return False, "Insufficient price history"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    bullish_engulfing = (
        last["close"] > last["open"]
        and
        prev["close"] < prev["open"]
        and
        last["close"] > prev["open"]
        and
        last["open"] < prev["close"]
    )

    bearish_engulfing = (
        last["close"] < last["open"]
        and
        prev["close"] > prev["open"]
        and
        last["close"] < prev["open"]
        and
        last["open"] > prev["close"]
    )

    if direction == "BULLISH":

        higher_close = (
            last["close"] > prev["close"]
        )

        strong_body = (
            last["BodyPct"] >= 0.50
            and
            last["CandleBody"] > 0
        )

        if bullish_engulfing:
            return True, "Bullish engulfing"

        if higher_close and strong_body:
            return True, "Bullish price action"

    else:

        lower_close = (
            last["close"] < prev["close"]
        )

        strong_body = (
            last["BodyPct"] >= 0.50
            and
            last["CandleBody"] < 0
        )

        if bearish_engulfing:
            return True, "Bearish engulfing"

        if lower_close and strong_body:
            return True, "Bearish price action"

    return False, "No strong candle confirmation"


# ============================================================
# BUYING / SELLING PRESSURE
# ============================================================

def calculate_pressure(df):
    recent = df.tail(10).copy()

    bullish_volume = (
        recent.loc[
            recent["close"] > recent["open"],
            "volume"
        ].sum()
    )

    bearish_volume = (
        recent.loc[
            recent["close"] < recent["open"],
            "volume"
        ].sum()
    )

    total = (
        bullish_volume
        + bearish_volume
    )

    if total <= 0:
        return 50.0, 50.0

    buying = (
        bullish_volume
        / total
        * 100
    )

    selling = (
        bearish_volume
        / total
        * 100
    )

    return buying, selling


# ============================================================
# RELATIVE STRENGTH
# ============================================================

def relative_strength(
    symbol_df,
    spy_df,
    period=20
):
    if (
        len(symbol_df) < period + 1
        or
        len(spy_df) < period + 1
    ):
        return 0.0

    symbol_return = (
        symbol_df["close"].iloc[-1]
        /
        symbol_df["close"].iloc[-period - 1]
        - 1
    )

    spy_return = (
        spy_df["close"].iloc[-1]
        /
        spy_df["close"].iloc[-period - 1]
        - 1
    )

    return (
        symbol_return
        - spy_return
    ) * 100


# ============================================================
# TECHNICAL SCORING
# ============================================================

def score_symbol(
    symbol,
    df,
    spy_df
):
    if df.empty or len(df) < 210:
        return None

    latest = df.iloc[-1]

    price = safe_float(
        latest["close"]
    )

    if (
        np.isnan(price)
        or price < MIN_STOCK_PRICE
    ):
        return None

    crossover = find_recent_crossover(
        df,
        30
    )

    if crossover is None:
        return None

    direction = crossover["direction"]

    score = 0
    confluences = []

    # --------------------------------------------------------
    # 50 / 100 EMA trend
    # --------------------------------------------------------

    ema50 = latest["EMA50"]
    ema100 = latest["EMA100"]
    ema200 = latest["EMA200"]

    if direction == "BULLISH":

        if ema50 > ema100:
            score += 18
            confluences.append(
                "50 EMA > 100 EMA"
            )

        if price > ema200:
            score += 8
            confluences.append(
                "Price > 200 EMA"
            )

    else:

        if ema50 < ema100:
            score += 18
            confluences.append(
                "50 EMA < 100 EMA"
            )

        if price < ema200:
            score += 8
            confluences.append(
                "Price < 200 EMA"
            )

    # --------------------------------------------------------
    # Relative volume
    # --------------------------------------------------------

    relative_volume = safe_float(
        latest["RelativeVolume"],
        0
    )

    if relative_volume >= MIN_RELATIVE_VOLUME:
        score += 12
        confluences.append(
            f"Relative volume {relative_volume:.2f}x"
        )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    momentum10 = safe_float(
        latest["Momentum10"],
        0
    )

    momentum20 = safe_float(
        latest["Momentum20"],
        0
    )

    if direction == "BULLISH":

        if momentum10 > 0:
            score += 8
            confluences.append(
                "10D momentum bullish"
            )

        if momentum20 > 0:
            score += 7
            confluences.append(
                "20D momentum bullish"
            )

    else:

        if momentum10 < 0:
            score += 8
            confluences.append(
                "10D momentum bearish"
            )

        if momentum20 < 0:
            score += 7
            confluences.append(
                "20D momentum bearish"
            )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = safe_float(
        latest["RSI"],
        50
    )

    if direction == "BULLISH":

        if 50 <= rsi <= 70:
            score += 7
            confluences.append(
                f"RSI {rsi:.1f}"
            )

    else:

        if 30 <= rsi <= 50:
            score += 7
            confluences.append(
                f"RSI {rsi:.1f}"
            )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_hist = safe_float(
        latest["MACD_HIST"],
        0
    )

    if direction == "BULLISH" and macd_hist > 0:
        score += 8
        confluences.append(
            "MACD bullish"
        )

    if direction == "BEARISH" and macd_hist < 0:
        score += 8
        confluences.append(
            "MACD bearish"
        )

    # --------------------------------------------------------
    # Keltner
    # --------------------------------------------------------

    keltner_ok, keltner_setup = analyze_keltner(
        df,
        direction
    )

    if keltner_ok:
        score += 12
        confluences.append(
            keltner_setup
        )

    # --------------------------------------------------------
    # Price action
    # --------------------------------------------------------

    price_action_ok, price_action = analyze_price_action(
        df,
        direction
    )

    if price_action_ok:
        score += 10
        confluences.append(
            price_action
        )

    # --------------------------------------------------------
    # Pressure
    # --------------------------------------------------------

    buying_pressure, selling_pressure = calculate_pressure(
        df
    )

    if direction == "BULLISH":

        if buying_pressure >= 55:
            score += 8
            confluences.append(
                f"Buying pressure {buying_pressure:.0f}%"
            )

    else:

        if selling_pressure >= 55:
            score += 8
            confluences.append(
                f"Selling pressure {selling_pressure:.0f}%"
            )

    # --------------------------------------------------------
    # Relative strength vs SPY
    # --------------------------------------------------------

    rs = relative_strength(
        df,
        spy_df
    )

    if direction == "BULLISH" and rs > 0:
        score += 7
        confluences.append(
            f"RS vs SPY +{rs:.1f}%"
        )

    if direction == "BEARISH" and rs < 0:
        score += 7
        confluences.append(
            f"RS vs SPY {rs:.1f}%"
        )

    # --------------------------------------------------------
    # Setup label
    # --------------------------------------------------------

    if keltner_ok and price_action_ok:
        setup = (
            "EMA Crossover + Keltner + "
            "Price Action"
        )
    elif keltner_ok:
        setup = (
            "EMA Crossover + Keltner Momentum"
        )
    elif price_action_ok:
        setup = (
            "EMA Crossover + Price Action"
        )
    else:
        setup = (
            "EMA Crossover Momentum"
        )

    # --------------------------------------------------------
    # Confluence count
    # --------------------------------------------------------

    confluence_count = len(
        confluences
    )

    return {
        "Ticker": symbol,
        "Price": round(price, 2),
        "Direction": direction,
        "Score": int(score),
        "Setup": setup,
        "Crossover Date": crossover["date"].strftime(
            "%Y-%m-%d"
        ),
        "Crossover Sessions Ago": max(
            0,
            len(df) - 1 - crossover["index"]
        ),
        "Buying Pressure %": round(
            buying_pressure,
            1
        ),
        "Selling Pressure %": round(
            selling_pressure,
            1
        ),
        "Relative Volume": round(
            relative_volume,
            2
        ),
        "RSI": round(
            rsi,
            1
        ),
        "Momentum 10D %": round(
            momentum10,
            2
        ),
        "Momentum 20D %": round(
            momentum20,
            2
        ),
        "RS vs SPY %": round(
            rs,
            2
        ),
        "Keltner": (
            "CONFIRMED"
            if keltner_ok
            else "NO"
        ),
        "Price Action": (
            "CONFIRMED"
            if price_action_ok
            else "NO"
        ),
        "MACD": (
            "BULLISH"
            if macd_hist > 0
            else "BEARISH"
        ),
        "Confluences": confluence_count,
        "Confluence List": " | ".join(
            confluences
        )
    }


# ============================================================
# OPTION HELPERS
# ============================================================

def normalize_option_chain(df):
    if df.empty:
        return df

    df = df.copy()

    numeric_columns = [
        "strike",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "change",
        "percent_change"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Tradier commonly returns option_type as call/put
    if "option_type" in df.columns:
        df["option_type"] = (
            df["option_type"]
            .astype(str)
            .str.lower()
        )

    if "greeks" not in df.columns:
        df["greeks"] = None

    return df


def extract_greek(row, name):
    value = row.get(name)

    if value is not None:
        return safe_float(value)

    greeks = row.get("greeks")

    if isinstance(greeks, dict):
        return safe_float(
            greeks.get(name)
        )

    return np.nan


def prepare_options(df):
    if df.empty:
        return df

    df = normalize_option_chain(df)

    for greek in [
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "mid_iv",
        "smv_vol"
    ]:
        df[greek] = df.apply(
            lambda row: extract_greek(
                row,
                greek
            ),
            axis=1
        )

    if "bid" not in df.columns:
        df["bid"] = np.nan

    if "ask" not in df.columns:
        df["ask"] = np.nan

    if "volume" not in df.columns:
        df["volume"] = 0

    if "open_interest" not in df.columns:
        df["open_interest"] = 0

    return df


def calculate_option_values(
    option,
    stock_price,
    direction
):
    strike = safe_float(
        option.get("strike")
    )

    bid = safe_float(
        option.get("bid")
    )

    ask = safe_float(
        option.get("ask")
    )

    last = safe_float(
        option.get("last")
    )

    if np.isnan(strike):
        return None

    # Conservative entry:
    # buy at ask
    entry = ask

    if np.isnan(entry) or entry <= 0:
        entry = last

    if np.isnan(entry) or entry <= 0:
        return None

    if direction == "BULLISH":
        intrinsic = max(
            stock_price - strike,
            0
        )
    else:
        intrinsic = max(
            strike - stock_price,
            0
        )

    time_value = max(
        entry - intrinsic,
        0
    )

    return {
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "last": last,
        "entry": entry,
        "intrinsic": intrinsic,
        "time_value": time_value
    }


def option_is_liquid(
    option,
    stock_price
):
    bid = safe_float(
        option.get("bid")
    )

    ask = safe_float(
        option.get("ask")
    )

    volume = safe_float(
        option.get("volume"),
        0
    )

    oi = safe_float(
        option.get("open_interest"),
        0
    )

    if (
        np.isnan(bid)
        or np.isnan(ask)
        or bid <= 0
        or ask <= 0
    ):
        return False

    if oi < MIN_OPEN_INTEREST:
        return False

    if volume < MIN_OPTION_VOLUME:
        return False

    mid = (
        bid + ask
    ) / 2

    if mid <= 0:
        return False

    spread_pct = (
        ask - bid
    ) / mid

    if spread_pct > MAX_SPREAD_PCT:
        return False

    return True


# ============================================================
# HUGHES 1% OPTION ENGINE
# ============================================================

def build_single_option_candidates(
    symbol,
    stock_price,
    direction,
    expiration,
    chain,
    technical_score,
    account_size
):
    chain = prepare_options(chain)

    if chain.empty:
        return []

    option_type = (
        "call"
        if direction == "BULLISH"
        else "put"
    )

    if "option_type" not in chain.columns:
        return []

    candidates = []

    filtered = chain[
        chain["option_type"] == option_type
    ].copy()

    for _, row in filtered.iterrows():

        if not option_is_liquid(
            row,
            stock_price
        ):
            continue

        delta = extract_greek(
            row,
            "delta"
        )

        if np.isnan(delta):
            continue

        delta_abs = abs(delta)

        if not (
            MIN_DELTA
            <= delta_abs
            <= MAX_DELTA
        ):
            continue

        values = calculate_option_values(
            row,
            stock_price,
            direction
        )

        if values is None:
            continue

        entry = values["entry"]
        intrinsic = values["intrinsic"]
        time_value = values["time_value"]

        # Hughes 1% rule
        max_time_value = (
            stock_price
            * HUGHES_TIME_VALUE_PCT
        )

        if time_value > max_time_value:
            continue

        # Modeled stop
        stop = (
            entry
            * STOP_PERCENT_OF_PREMIUM
        )

        risk_per_contract = (
            entry - stop
        ) * 100

        if risk_per_contract <= 0:
            continue

        # Expected underlying move
        atr = max(
            stock_price * 0.02,
            0.01
        )

        expected_move = (
            atr * 1.25
        )

        option_move = (
            expected_move
            * delta_abs
        )

        target = max(
            entry + option_move,
            entry * 1.125
        )

        reward = (
            target - entry
        )

        rr = (
            reward
            /
            (entry - stop)
        )

        if rr < MIN_RISK_REWARD:
            continue

        # Position sizing based on intended modeled stop.
        max_risk = (
            account_size
            * RISK_PER_TRADE
        )

        contracts = math.floor(
            max_risk
            /
            risk_per_contract
        )

        contracts = max(
            contracts,
            1
        )

        candidates.append({
            "Ticker": symbol,
            "Direction": direction,
            "Strategy": "Single ITM Option",
            "Expiration": expiration,
            "DTE": (
                pd.to_datetime(expiration).date()
                - date.today()
            ).days,
            "Strike": values["strike"],
            "Option Type": option_type.upper(),
            "Entry": round(entry, 2),
            "Bid": round(
                values["bid"], 2
            ),
            "Ask": round(
                values["ask"], 2
            ),
            "Intrinsic": round(
                intrinsic, 2
            ),
            "Time Value": round(
                time_value, 2
            ),
            "1% Max TV": round(
                max_time_value, 2
            ),
            "Delta": round(
                delta_abs, 3
            ),
            "Stop": round(
                stop, 2
            ),
            "Target": round(
                target, 2
            ),
            "R:R": round(
                rr,
                2
            ),
            "Contracts": contracts,
            "Estimated Risk": round(
                risk_per_contract
                * contracts,
                2
            ),
            "Technical Score": technical_score,
            "Open Interest": int(
                safe_float(
                    row.get(
                        "open_interest"
                    ),
                    0
                )
            ),
            "Volume": int(
                safe_float(
                    row.get(
                        "volume"
                    ),
                    0
                )
            )
        })

    return candidates


# ============================================================
# DEBIT SPREAD ENGINE
# ============================================================

def build_spread_candidates(
    symbol,
    stock_price,
    direction,
    expiration,
    chain,
    technical_score,
    account_size
):
    chain = prepare_options(chain)

    if chain.empty:
        return []

    option_type = (
        "call"
        if direction == "BULLISH"
        else "put"
    )

    if "option_type" not in chain.columns:
        return []

    options = chain[
        chain["option_type"] == option_type
    ].copy()

    if options.empty:
        return []

    options = options[
        options["strike"].notna()
    ].sort_values(
        "strike"
    )

    candidates = []

    # --------------------------------------------------------
    # Build all possible 2-leg debit spreads.
    # --------------------------------------------------------

    rows = list(
        options.iterrows()
    )

    for _, long_leg in rows:

        if not option_is_liquid(
            long_leg,
            stock_price
        ):
            continue

        long_values = calculate_option_values(
            long_leg,
            stock_price,
            direction
        )

        if long_values is None:
            continue

        long_strike = long_values["strike"]
        long_entry = long_values["entry"]
        long_intrinsic = long_values["intrinsic"]
        long_time_value = long_values["time_value"]

        delta = extract_greek(
            long_leg,
            "delta"
        )

        if np.isnan(delta):
            continue

        if not (
            MIN_DELTA
            <= abs(delta)
            <= MAX_DELTA
        ):
            continue

        # Hughes 1% rule applies to the LONG option.
        max_time_value = (
            stock_price
            * HUGHES_TIME_VALUE_PCT
        )

        if long_time_value > max_time_value:
            continue

        for _, short_leg in rows:

            short_strike = safe_float(
                short_leg.get("strike")
            )

            if np.isnan(short_strike):
                continue

            # Bull call spread:
            # long lower strike / short higher strike
            if direction == "BULLISH":

                if short_strike <= long_strike:
                    continue

            # Bear put spread:
            # long higher strike / short lower strike
            else:

                if short_strike >= long_strike:
                    continue

            if not option_is_liquid(
                short_leg,
                stock_price
            ):
                continue

            short_bid = safe_float(
                short_leg.get("bid")
            )

            short_ask = safe_float(
                short_leg.get("ask")
            )

            if (
                np.isnan(short_bid)
                or short_bid <= 0
            ):
                continue

            # Conservative debit:
            # pay long ask, receive short bid.
            debit = (
                long_entry
                - short_bid
            )

            if debit <= 0:
                continue

            width = abs(
                short_strike
                - long_strike
            )

            if width <= 0:
                continue

            max_profit = (
                width
                - debit
            )

            if max_profit <= 0:
                continue

            # ------------------------------------------------
            # User requested 70% of premium stop.
            # Example:
            # $2.50 debit -> $1.75 stop.
            # ------------------------------------------------

            stop = (
                debit
                * STOP_PERCENT_OF_PREMIUM
            )

            modeled_risk = (
                debit - stop
            )

            if modeled_risk <= 0:
                continue

            # Target at 75% of max spread profit,
            # while requiring at least 1.25R.
            target = (
                debit
                + max_profit * 0.75
            )

            minimum_target = (
                debit
                + modeled_risk
                * MIN_RISK_REWARD
            )

            target = max(
                target,
                minimum_target
            )

            if target > width:
                target = width

            reward = (
                target - debit
            )

            rr = (
                reward
                /
                modeled_risk
            )

            if rr < MIN_RISK_REWARD:
                continue

            max_risk = (
                account_size
                * RISK_PER_TRADE
            )

            risk_per_contract = (
                modeled_risk
                * 100
            )

            contracts = math.floor(
                max_risk
                /
                risk_per_contract
            )

            contracts = max(
                contracts,
                1
            )

            candidates.append({
                "Ticker": symbol,
                "Direction": direction,
                "Strategy": "Debit Spread",
                "Expiration": expiration,
                "DTE": (
                    pd.to_datetime(expiration).date()
                    - date.today()
                ).days,
                "Long Strike": long_strike,
                "Short Strike": short_strike,
                "Long Option": option_type.upper(),
                "Debit": round(
                    debit,
                    2
                ),
                "Stop": round(
                    stop,
                    2
                ),
                "Target": round(
                    target,
                    2
                ),
                "Width": round(
                    width,
                    2
                ),
                "Max Profit": round(
                    max_profit,
                    2
                ),
                "R:R": round(
                    rr,
                    2
                ),
                "Long Intrinsic": round(
                    long_intrinsic,
                    2
                ),
                "Long Time Value": round(
                    long_time_value,
                    2
                ),
                "1% Max TV": round(
                    max_time_value,
                    2
                ),
                "Delta": round(
                    abs(delta),
                    3
                ),
                "Contracts": contracts,
                "Estimated Risk": round(
                    risk_per_contract
                    * contracts,
                    2
                ),
                "Technical Score": technical_score,
                "Long OI": int(
                    safe_float(
                        long_leg.get(
                            "open_interest"
                        ),
                        0
                    )
                ),
                "Long Volume": int(
                    safe_float(
                        long_leg.get(
                            "volume"
                        ),
                        0
                    )
                )
            })

    return candidates


# ============================================================
# PHASE 2 OPTION ENGINE
# ============================================================

def build_option_engine(
    symbol,
    stock_price,
    direction,
    technical_score,
    account_size
):
    expirations = get_expirations(
        symbol
    )

    if not expirations:
        return []

    candidates = []

    today = date.today()

    for expiration in expirations:

        try:
            expiration_date = (
                pd.to_datetime(
                    expiration
                ).date()
            )
        except Exception:
            continue

        dte = (
            expiration_date
            - today
        ).days

        if (
            dte < MIN_DTE
            or
            dte > MAX_DTE
        ):
            continue

        try:
            chain = get_option_chain(
                symbol,
                expiration
            )
        except Exception:
            continue

        if chain.empty:
            continue

        singles = build_single_option_candidates(
            symbol,
            stock_price,
            direction,
            expiration,
            chain,
            technical_score,
            account_size
        )

        spreads = build_spread_candidates(
            symbol,
            stock_price,
            direction,
            expiration,
            chain,
            technical_score,
            account_size
        )

        candidates.extend(
            singles
        )

        candidates.extend(
            spreads
        )

    if not candidates:
        return []

    df = pd.DataFrame(
        candidates
    )

    # Best options first:
    # technical score + R:R + lower DTE
    df["Option Rank"] = (
        df["Technical Score"]
        +
        df["R:R"] * 5
        -
        df["DTE"] * 0.10
    )

    df = df.sort_values(
        "Option Rank",
        ascending=False
    )

    return df.to_dict(
        "records"
    )


# ============================================================
# QOD TRADE MESSAGE
# ============================================================

def qod_message(row):
    ticker = row["Ticker"]

    expiration = pd.to_datetime(
        row["Expiration"]
    )

    expiration_text = (
        expiration.strftime("%b %d")
    )

    if row["Strategy"] == "Single ITM Option":

        strike = row["Strike"]

        option_type = (
            "Call"
            if row["Option Type"] == "CALL"
            else "Put"
        )

        return (
            f"BUY-TO-OPEN the ({ticker}) "
            f"{expiration_text} "
            f"{strike:.0f} {option_type} "
            f"at {row['Entry']:.2f} or less.\n\n"
            f"Stop: {row['Stop']:.2f}\n"
            f"Target: {row['Target']:.2f}\n"
            f"Modeled R:R: 1:{row['R:R']:.2f}"
        )

    long_strike = row["Long Strike"]
    short_strike = row["Short Strike"]

    if row["Direction"] == "BULLISH":

        spread_text = (
            f"BUY-TO-OPEN the ({ticker}) "
            f"{expiration_text} "
            f"{long_strike:.0f}/{short_strike:.0f} "
            f"Call Debit Spread"
        )

    else:

        spread_text = (
            f"BUY-TO-OPEN the ({ticker}) "
            f"{expiration_text} "
            f"{long_strike:.0f}/{short_strike:.0f} "
            f"Put Debit Spread"
        )

    return (
        f"{spread_text} "
        f"at {row['Debit']:.2f} debit or less.\n\n"
        f"Stop: {row['Stop']:.2f}\n"
        f"Target: {row['Target']:.2f}\n"
        f"Max Profit: {row['Max Profit']:.2f}\n"
        f"Modeled R:R: 1:{row['R:R']:.2f}"
    )


# ============================================================
# SCANNER
# ============================================================

def run_golden_scan(
    symbols,
    min_score,
    account_size
):
    progress = st.progress(
        0
    )

    status = st.empty()

    try:
        spy_df = get_history(
            "SPY"
        )
    except Exception as exc:
        progress.empty()
        status.empty()

        st.error(
            f"Unable to load SPY data: {exc}"
        )

        return []

    if spy_df.empty:
        progress.empty()
        status.empty()

        st.error(
            "SPY historical data was empty."
        )

        return []

    technical_candidates = []

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        status.write(
            f"Scanning {symbol} "
            f"({i + 1}/{total})..."
        )

        try:
            df = get_history(
                symbol
            )

            if df.empty:
                continue

            df = add_indicators(
                df
            )

            result = score_symbol(
                symbol,
                df,
                spy_df
            )

            if result is None:
                continue

            if result["Score"] < min_score:
                continue

            # Strong liquidity requirement
            if (
                result["Relative Volume"]
                < MIN_RELATIVE_VOLUME
            ):
                continue

            technical_candidates.append(
                result
            )

        except Exception:
            continue

        progress.progress(
            (i + 1) / total
        )

    progress.empty()
    status.empty()

    if not technical_candidates:
        return []

    technical_candidates.sort(
        key=lambda x: (
            x["Score"],
            x["Confluences"],
            x["Relative Volume"]
        ),
        reverse=True
    )

    return technical_candidates[
        :MAX_RESULTS
    ]


# ============================================================
# UI HEADER
# ============================================================

st.title(
    "🏆 GOLDEN QOD SCANNER"
)

st.subheader(
    "1–2 Week Stock & Options Swing Hunter"
)

st.caption(
    "Production Tradier market data • "
    "EMA 50/100 • Keltner • Momentum • "
    "Price Action • Relative Volume • "
    "Pressure • Relative Strength • "
    "Hughes 1% Option Rule"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Scanner Settings"
)

account_size = st.sidebar.number_input(
    "Account Size ($)",
    min_value=500.0,
    max_value=1000000.0,
    value=float(DEFAULT_ACCOUNT_SIZE),
    step=500.0
)

min_score = st.sidebar.slider(
    "Minimum Technical Score",
    min_value=50,
    max_value=95,
    value=MIN_SCORE,
    step=1
)

max_results = st.sidebar.slider(
    "Maximum Technical Candidates",
    min_value=3,
    max_value=25,
    value=MAX_RESULTS,
    step=1
)

st.sidebar.divider()

st.sidebar.write(
    "### Golden Filters"
)

st.sidebar.write(
    f"• Stock price: ${MIN_STOCK_PRICE:.2f}+"
)

st.sidebar.write(
    f"• Relative volume: {MIN_RELATIVE_VOLUME:.2f}x+"
)

st.sidebar.write(
    f"• EMA crossover: last 30 sessions"
)

st.sidebar.write(
    f"• Option DTE: {MIN_DTE}–{MAX_DTE}"
)

st.sidebar.write(
    "• Hughes 1% time-value rule"
)

st.sidebar.write(
    f"• Minimum R:R: 1:{MIN_RISK_REWARD:.2f}"
)

st.sidebar.write(
    "• 70% premium stop"
)

st.sidebar.divider()

st.sidebar.success(
    "🟢 PRODUCTION TRADIER API"
)

# ============================================================
# CONNECTION TEST
# ============================================================

st.markdown(
    "### 🔌 Tradier Connection"
)

col1, col2 = st.columns(
    [1, 4]
)

with col1:

    test_connection = st.button(
        "Test Connection",
        use_container_width=True
    )

with col2:

    st.caption(
        "Uses the production Tradier API. "
        "No orders are placed by this scanner."
    )

if test_connection:

    try:

        quote = get_quote(
            "SPY"
        )

        if quote:

            test_price = safe_float(
                quote.get("last")
            )

            st.success(
                f"🟢 Production Tradier connected. "
                f"SPY last: "
                f"${test_price:.2f}"
                if not np.isnan(test_price)
                else
                "🟢 Production Tradier connected."
            )

        else:

            st.warning(
                "Tradier responded, but no SPY quote was returned."
            )

    except Exception as exc:

        st.error(
            f"🔴 Tradier connection failed: {exc}"
        )

# ============================================================
# RUN SCAN
# ============================================================

st.divider()

run_scan = st.button(
    "🏆 RUN GOLDEN SCAN",
    type="primary",
    use_container_width=True
)

if run_scan:

    with st.spinner(
        "Hunting for the needle in the haystack..."
    ):

        technical_results = run_golden_scan(
            UNIVERSE,
            min_score,
            account_size
        )

    if not technical_results:

        st.warning(
            "No candidates passed the current Golden filters."
        )

        st.info(
            "Try lowering the Minimum Technical Score "
            "slightly if you want a wider search."
        )

    else:

        technical_results = (
            technical_results[:max_results]
        )

        st.session_state[
            "technical_results"
        ] = technical_results

        st.session_state[
            "account_size"
        ] = account_size

# ============================================================
# TECHNICAL RESULTS
# ============================================================

if (
    "technical_results"
    in st.session_state
):

    results = st.session_state[
        "technical_results"
    ]

    account_size = st.session_state.get(
        "account_size",
        account_size
    )

    st.divider()

    st.header(
        "🎯 Phase 1 — Technical Candidates"
    )

    display_columns = [
        "Ticker",
        "Price",
        "Direction",
        "Score",
        "Setup",
        "Crossover Date",
        "Crossover Sessions Ago",
        "Buying Pressure %",
        "Selling Pressure %",
        "Relative Volume",
        "RSI",
        "Momentum 10D %",
        "Momentum 20D %",
        "RS vs SPY %",
        "Keltner",
        "Price Action",
        "MACD",
        "Confluences"
    ]

    technical_df = pd.DataFrame(
        results
    )

    st.dataframe(
        technical_df[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # GOLDEN CANDIDATE
    # ========================================================

    golden = results[0]

    st.divider()

    st.header(
        "🥇 #1 GOLDEN CANDIDATE"
    )

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.metric(
            "Ticker",
            golden["Ticker"]
        )

    with g2:
        st.metric(
            "Direction",
            golden["Direction"]
        )

    with g3:
        st.metric(
            "Technical Score",
            golden["Score"]
        )

    with g4:
        st.metric(
            "Confluences",
            golden["Confluences"]
        )

    st.write(
        f"**Setup:** {golden['Setup']}"
    )

    st.write(
        f"**Crossover:** "
        f"{golden['Crossover Date']} "
        f"({golden['Crossover Sessions Ago']} "
        f"sessions ago)"
    )

    st.write(
        f"**Relative Volume:** "
        f"{golden['Relative Volume']:.2f}x"
    )

    st.write(
        f"**Buying Pressure:** "
        f"{golden['Buying Pressure %']:.1f}%"
    )

    st.write(
        f"**Selling Pressure:** "
        f"{golden['Selling Pressure %']:.1f}%"
    )

    st.write(
        f"**Relative Strength vs SPY:** "
        f"{golden['RS vs SPY %']:+.2f}%"
    )

    st.write(
        f"**Keltner:** {golden['Keltner']}"
    )

    st.write(
        f"**Price Action:** "
        f"{golden['Price Action']}"
    )

    st.write(
        f"**Confluence:** "
        f"{golden['Confluence List']}"
    )

    # ========================================================
    # PHASE 2
    # ========================================================

    st.divider()

    st.header(
        "⚙️ Phase 2 — Hughes Option Engine"
    )

    st.caption(
        "The option engine searches 7–40 DTE, "
        "requires liquidity, uses the long-leg "
        "1% time-value rule, and models the "
        "requested 70% premium stop."
    )

    run_options = st.button(
        f"🚀 RUN PHASE 2 ON {golden['Ticker']}",
        type="secondary",
        use_container_width=True
    )

    if run_options:

        with st.spinner(
            f"Analyzing options for "
            f"{golden['Ticker']}..."
        ):

            option_results = build_option_engine(
                golden["Ticker"],
                golden["Price"],
                golden["Direction"],
                golden["Score"],
                account_size
            )

        if not option_results:

            st.warning(
                "No option structure passed all "
                "Phase 2 filters."
            )

            st.info(
                "That is intentional. "
                "The scanner would rather return "
                "NO TRADE than force a poor option setup."
            )

            st.session_state[
                "option_results"
            ] = []

        else:

            st.session_state[
                "option_results"
            ] = option_results

# ============================================================
# OPTION RESULTS
# ============================================================

if (
    "option_results"
    in st.session_state
):

    option_results = st.session_state[
        "option_results"
    ]

    if option_results:

        st.divider()

        st.header(
            "🏆 Phase 2 — Option Candidates"
        )

        option_df = pd.DataFrame(
            option_results
        )

        # ----------------------------------------------------
        # Best option
        # ----------------------------------------------------

        best_option = option_results[0]

        st.subheader(
            "🥇 Best Option Structure"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric(
                "Strategy",
                best_option["Strategy"]
            )

        with c2:
            st.metric(
                "Expiration",
                best_option["Expiration"]
            )

        with c3:
            st.metric(
                "DTE",
                best_option["DTE"]
            )

        with c4:
            st.metric(
                "R:R",
                f"1:{best_option['R:R']:.2f}"
            )

        with c5:
            st.metric(
                "Contracts",
                best_option["Contracts"]
            )

        # ----------------------------------------------------
        # QOD message
        # ----------------------------------------------------

        st.markdown(
            "### 📲 QOD-Style Trade"
        )

        st.code(
            qod_message(
                best_option
            ),
            language="text"
        )

        # ----------------------------------------------------
        # Option details
        # ----------------------------------------------------

        st.markdown(
            "### Option Details"
        )

        if best_option["Strategy"] == "Single ITM Option":

            details = {
                "Ticker": best_option["Ticker"],
                "Direction": best_option["Direction"],
                "Strategy": best_option["Strategy"],
                "Expiration": best_option["Expiration"],
                "DTE": best_option["DTE"],
                "Strike": best_option["Strike"],
                "Entry": best_option["Entry"],
                "Intrinsic": best_option["Intrinsic"],
                "Time Value": best_option["Time Value"],
                "1% Max Time Value": best_option["1% Max TV"],
                "Delta": best_option["Delta"],
                "Stop": best_option["Stop"],
                "Target": best_option["Target"],
                "R:R": best_option["R:R"],
                "Contracts": best_option["Contracts"],
                "Estimated Risk": best_option["Estimated Risk"]
            }

        else:

            details = {
                "Ticker": best_option["Ticker"],
                "Direction": best_option["Direction"],
                "Strategy": best_option["Strategy"],
                "Expiration": best_option["Expiration"],
                "DTE": best_option["DTE"],
                "Long Strike": best_option["Long Strike"],
                "Short Strike": best_option["Short Strike"],
                "Debit": best_option["Debit"],
                "Width": best_option["Width"],
                "Long Intrinsic": best_option["Long Intrinsic"],
                "Long Time Value": best_option["Long Time Value"],
                "1% Max Time Value": best_option["1% Max TV"],
                "Delta": best_option["Delta"],
                "Stop": best_option["Stop"],
                "Target": best_option["Target"],
                "Max Profit": best_option["Max Profit"],
                "R:R": best_option["R:R"],
                "Contracts": best_option["Contracts"],
                "Estimated Risk": best_option["Estimated Risk"]
            }

        detail_df = pd.DataFrame(
            list(details.items()),
            columns=[
                "Metric",
                "Value"
            ]
        )

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # All option candidates
        # ----------------------------------------------------

        st.markdown(
            "### Phase 2 Candidate Table"
        )

        preferred_columns = [
            "Ticker",
            "Strategy",
            "Expiration",
            "DTE",
            "R:R",
            "Technical Score",
            "Contracts",
            "Estimated Risk"
        ]

        if (
            "Strike"
            in option_df.columns
        ):
            preferred_columns += [
                "Strike",
                "Entry",
                "Stop",
                "Target",
                "Time Value",
                "1% Max TV",
                "Delta"
            ]

        if (
            "Long Strike"
            in option_df.columns
        ):
            preferred_columns += [
                "Long Strike",
                "Short Strike",
                "Debit",
                "Stop",
                "Target",
                "Max Profit",
                "Long Time Value",
                "1% Max TV",
                "Delta"
            ]

        preferred_columns = [
            column
            for column in preferred_columns
            if column in option_df.columns
        ]

        st.dataframe(
            option_df[
                preferred_columns
            ],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        st.info(
            "⚠️ Risk note: the displayed stop is a "
            "MODELED premium stop at 70% of entry "
            "premium. It is not a guarantee of execution "
            "at that price. Options can gap and spreads "
            "can widen."
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏆 Golden QOD Scanner • "
    "Production Tradier Market Data • "
    "Technical analysis only — no orders are submitted."
)

st.caption(
    "The scanner is designed to reject weak setups "
    "rather than manufacture trades."
)
