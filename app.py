import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta


# ============================================================
# GOLDEN QOD SCANNER
# Tradier-only | 7-40 DTE | Hughes 1% Rule
# ============================================================

st.set_page_config(
    page_title="Golden QOD Scanner",
    page_icon="🥇",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

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

HUGHES_TIME_VALUE_PCT = 0.01

STOP_PERCENT_OF_PREMIUM = 0.70

MIN_RISK_REWARD = 1.25

HISTORY_DAYS = 450


# ============================================================
# MAG 7 EXCLUSION
# ============================================================

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
# ESTABLISHED STOCK / ETF UNIVERSE
# ============================================================

SCANNER_UNIVERSE = [
    # ETFs
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLF",
    "XLK",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLY",
    "XLB",
    "XLU",
    "SMH",
    "XBI",

    # Technology
    "AMD",
    "AVGO",
    "CRM",
    "ORCL",
    "ADBE",
    "NOW",
    "INTC",
    "MU",
    "MRVL",
    "DELL",
    "CSCO",
    "IBM",
    "QCOM",
    "TXN",

    # Financial
    "V",
    "MA",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "C",
    "COF",
    "AXP",
    "PYPL",
    "HOOD",

    # Consumer
    "WMT",
    "COST",
    "HD",
    "LOW",
    "NKE",
    "TGT",
    "KO",
    "PEP",
    "MCD",

    # Industrial / Energy
    "CAT",
    "DE",
    "GE",
    "HON",
    "RTX",
    "BA",
    "UPS",
    "XOM",
    "CVX",
    "SLB",

    # Healthcare
    "UNH",
    "LLY",
    "JNJ",
    "ABBV",
    "PFE",
    "MRK",
    "ABT",

    # Communications / Other
    "DIS",
    "NFLX",
    "UBER",
    "SHOP",
    "CRWD",
    "COIN",
    "PLTR"
]


# ============================================================
# SIDEBAR
# ============================================================

st.title("🥇 GOLDEN QOD SCANNER")
st.caption(
    "Needle-in-the-haystack swing scanner • Tradier market data • "
    "7–40 DTE • Hughes-style 1% option rule"
)

st.sidebar.header("⚙️ Scanner Settings")

account_size = st.sidebar.selectbox(
    "Account Size",
    [2000, 2500, 3000, 3500, 4000, 4500, 5000],
    index=3
)

test_symbol = st.sidebar.text_input(
    "Test Symbol",
    "BLCO"
).upper().strip()

min_score_setting = st.sidebar.slider(
    "Minimum Technical Score",
    70,
    90,
    MIN_SCORE
)

run_test = st.sidebar.button(
    "🔬 Analyze Test Symbol"
)

run_scan = st.sidebar.button(
    "🥇 RUN GOLDEN SCAN"
)


# ============================================================
# TRADIER AUTHENTICATION
# ============================================================

try:
    token = st.secrets["TRADIER_API_TOKEN"]

except Exception:
    st.error(
        "🔴 Tradier Production API token not found.\n\n"
        "Go to Streamlit → Settings → Secrets and make sure "
        "TRADIER_API_TOKEN is present."
    )
    st.stop()


HEADERS = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/json"
}

BASE_URL = "https://api.tradier.com/v1"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        return float(value)

    except Exception:
        return default


def get_history(symbol):
    """
    Pull daily historical data from Tradier.
    """

    end_date = date.today()
    start_date = end_date - timedelta(days=HISTORY_DAYS)

    try:

        response = requests.get(
            f"{BASE_URL}/markets/history",
            params={
                "symbol": symbol,
                "interval": "daily",
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            headers=HEADERS,
            timeout=15
        )

        if response.status_code != 200:
            return None

        payload = response.json()

        rows = (
            payload
            .get("history", {})
            .get("day", [])
        )

        if not rows:
            return None

        df = pd.DataFrame(rows)

        required = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for col in required:
            if col not in df.columns:
                return None

        df["date"] = pd.to_datetime(df["date"])

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = (
            df
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(df) < 220:
            return None

        return df

    except Exception:
        return None


# ============================================================
# INDICATOR ENGINE
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --------------------------------------------------------
    # EMAs
    # --------------------------------------------------------

    df["EMA20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA100"] = close.ewm(
        span=100,
        adjust=False
    ).mean()

    df["EMA200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    change = close.diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    # --------------------------------------------------------
    # Relative Volume
    # --------------------------------------------------------

    df["AVG_VOLUME20"] = (
        volume
        .rolling(20)
        .mean()
    )

    df["REL_VOLUME"] = (
        volume /
        df["AVG_VOLUME20"]
    )

    # --------------------------------------------------------
    # KELTNER CHANNEL
    # --------------------------------------------------------

    df["KC_MIDDLE"] = df["EMA20"]

    df["KC_UPPER"] = (
        df["EMA20"] +
        2.0 * df["ATR"]
    )

    df["KC_LOWER"] = (
        df["EMA20"] -
        2.0 * df["ATR"]
    )

    # --------------------------------------------------------
    # Candle / Price Action
    # --------------------------------------------------------

    candle_range = (
        high - low
    ).replace(0, np.nan)

    df["BODY_PERCENT"] = (
        (close - df["open"]).abs()
        / candle_range
    )

    df["CLOSE_LOCATION"] = (
        (close - low)
        / candle_range
    )

    return df


# ============================================================
# CROSSOVER ANALYSIS
# ============================================================

def crossover_analysis(df):

    ema50 = df["EMA50"]
    ema100 = df["EMA100"]

    bullish_cross = (
        (ema50 > ema100)
        &
        (ema50.shift(1) <= ema100.shift(1))
    )

    bearish_cross = (
        (ema50 < ema100)
        &
        (ema50.shift(1) >= ema100.shift(1))
    )

    bullish_indices = np.where(
        bullish_cross.fillna(False)
    )[0]

    bearish_indices = np.where(
        bearish_cross.fillna(False)
    )[0]

    last_bull = (
        bullish_indices[-1]
        if len(bullish_indices)
        else None
    )

    last_bear = (
        bearish_indices[-1]
        if len(bearish_indices)
        else None
    )

    return last_bull, last_bear


# ============================================================
# KELTNER / MOMENTUM ANALYSIS
# ============================================================

def keltner_analysis(df, direction):

    recent = df.tail(30)

    if direction == "Bullish":

        touched_pullback = (
            recent["low"]
            <= recent["KC_MIDDLE"]
        ).any()

        recovered = (
            recent["close"].iloc[-1]
            > recent["KC_MIDDLE"].iloc[-1]
        )

        breakout = (
            recent["close"].iloc[-1]
            > recent["KC_UPPER"].iloc[-1]
        )

        if touched_pullback and recovered:
            return "Bullish Keltner Pullback", 100

        if breakout:
            return "Bullish Keltner Momentum", 95

        return "No Strong Keltner Signal", 50

    else:

        touched_pullback = (
            recent["high"]
            >= recent["KC_MIDDLE"]
        ).any()

        recovered = (
            recent["close"].iloc[-1]
            < recent["KC_MIDDLE"].iloc[-1]
        )

        breakdown = (
            recent["close"].iloc[-1]
            < recent["KC_LOWER"].iloc[-1]
        )

        if touched_pullback and recovered:
            return "Bearish Keltner Pullback", 100

        if breakdown:
            return "Bearish Keltner Momentum", 95

        return "No Strong Keltner Signal", 50


# ============================================================
# PRICE ACTION
# ============================================================

def price_action_analysis(df, direction):

    recent = df.tail(5)

    last = recent.iloc[-1]

    body = abs(
        last["close"] -
        last["open"]
    )

    candle_range = (
        last["high"] -
        last["low"]
    )

    if candle_range <= 0:
        return "Neutral", 50

    body_pct = body / candle_range

    close_location = (
        last["close"] -
        last["low"]
    ) / candle_range

    if direction == "Bullish":

        bullish_candle = (
            last["close"] >
            last["open"]
            and body_pct >= 0.50
            and close_location >= 0.65
        )

        higher_high = (
            last["high"] >
            recent["high"].iloc[-2]
        )

        if bullish_candle and higher_high:
            return "Bullish Expansion", 100

        if bullish_candle:
            return "Bullish Price Action", 90

        return "Neutral Price Action", 55

    else:

        bearish_candle = (
            last["close"] <
            last["open"]
            and body_pct >= 0.50
            and close_location <= 0.35
        )

        lower_low = (
            last["low"] <
            recent["low"].iloc[-2]
        )

        if bearish_candle and lower_low:
            return "Bearish Expansion", 100

        if bearish_candle:
            return "Bearish Price Action", 90

        return "Neutral Price Action", 55


# ============================================================
# BUYING / SELLING PRESSURE
# ============================================================

def pressure_analysis(df):

    recent = df.tail(10)

    ranges = (
        recent["high"] -
        recent["low"]
    ).replace(0, np.nan)

    pressure = (
        (
            recent["close"] -
            recent["low"]
        ) / ranges
    )

    buying_pressure = (
        pressure.mean() * 100
    )

    selling_pressure = (
        100 - buying_pressure
    )

    return (
        round(buying_pressure, 1),
        round(selling_pressure, 1)
    )


# ============================================================
# TECHNICAL CANDIDATE ENGINE
# ============================================================

def analyze_symbol(symbol, spy_df=None):

    if symbol in MAG_7:
        return None

    df = get_history(symbol)

    if df is None:
        return None

    df = calculate_indicators(df)

    latest = df.iloc[-1]

    price = safe_float(
        latest["close"]
    )

    if price < MIN_STOCK_PRICE:
        return None

    ema20 = safe_float(
        latest["EMA20"]
    )

    ema50 = safe_float(
        latest["EMA50"]
    )

    ema100 = safe_float(
        latest["EMA100"]
    )

    ema200 = safe_float(
        latest["EMA200"]
    )

    rsi = safe_float(
        latest["RSI"]
    )

    macd = safe_float(
        latest["MACD"]
    )

    macd_signal = safe_float(
        latest["MACD_SIGNAL"]
    )

    atr = safe_float(
        latest["ATR"]
    )

    relative_volume = safe_float(
        latest["REL_VOLUME"],
        0
    )

    # --------------------------------------------------------
    # Determine Direction
    # --------------------------------------------------------

    bullish_structure = (
        ema50 > ema100
        and price > ema50
    )

    bearish_structure = (
        ema50 < ema100
        and price < ema50
    )

    if not bullish_structure and not bearish_structure:
        return None

    direction = (
        "Bullish"
        if bullish_structure
        else "Bearish"
    )

    # --------------------------------------------------------
    # 50 / 100 EMA crossover
    # --------------------------------------------------------

    last_bull, last_bear = crossover_analysis(df)

    crossover_index = (
        last_bull
        if direction == "Bullish"
        else last_bear
    )

    if crossover_index is None:
        crossover_sessions_ago = 999
        crossover_date = None
    else:
        crossover_sessions_ago = (
            len(df) - 1 - crossover_index
        )

        crossover_date = (
            df.iloc[crossover_index]["date"]
            .date()
        )

    crossover_recent = (
        crossover_sessions_ago <= 30
    )

    # --------------------------------------------------------
    # Trend Score
    # --------------------------------------------------------

    trend_score = 0

    if direction == "Bullish":

        if ema50 > ema100:
            trend_score += 35

        if price > ema50:
            trend_score += 25

        if ema100 > ema200:
            trend_score += 20

        if ema20 > ema50:
            trend_score += 20

    else:

        if ema50 < ema100:
            trend_score += 35

        if price < ema50:
            trend_score += 25

        if ema100 < ema200:
            trend_score += 20

        if ema20 < ema50:
            trend_score += 20

    # --------------------------------------------------------
    # Momentum Score
    # --------------------------------------------------------

    momentum_score = 0

    if direction == "Bullish":

        if 50 <= rsi <= 72:
            momentum_score += 40

        if macd > macd_signal:
            momentum_score += 30

        if price > ema20:
            momentum_score += 30

    else:

        if 28 <= rsi <= 50:
            momentum_score += 40

        if macd < macd_signal:
            momentum_score += 30

        if price < ema20:
            momentum_score += 30

    # --------------------------------------------------------
    # Relative Volume Score
    # --------------------------------------------------------

    if relative_volume >= 1.50:
        volume_score = 100

    elif relative_volume >= 1.20:
        volume_score = 85

    elif relative_volume >= 1.00:
        volume_score = 65

    else:
        volume_score = 35

    # --------------------------------------------------------
    # Keltner
    # --------------------------------------------------------

    keltner_setup, keltner_score = (
        keltner_analysis(
            df,
            direction
        )
    )

    # --------------------------------------------------------
    # Price Action
    # --------------------------------------------------------

    price_action, price_action_score = (
        price_action_analysis(
            df,
            direction
        )
    )

    # --------------------------------------------------------
    # Pressure
    # --------------------------------------------------------

    buying_pressure, selling_pressure = (
        pressure_analysis(df)
    )

    if direction == "Bullish":

        pressure_score = buying_pressure

    else:

        pressure_score = selling_pressure

    # --------------------------------------------------------
    # Relative Strength vs SPY
    # --------------------------------------------------------

    relative_strength_score = 50

    if spy_df is not None:

        try:

            stock_return = (
                df["close"].iloc[-1]
                /
                df["close"].iloc[-21]
                - 1
            )

            spy_return = (
                spy_df["close"].iloc[-1]
                /
                spy_df["close"].iloc[-21]
                - 1
            )

            relative_strength = (
                stock_return -
                spy_return
            )

            if direction == "Bullish":

                if relative_strength > 0.05:
                    relative_strength_score = 100

                elif relative_strength > 0.02:
                    relative_strength_score = 85

                elif relative_strength > 0:
                    relative_strength_score = 70

                else:
                    relative_strength_score = 40

            else:

                if relative_strength < -0.05:
                    relative_strength_score = 100

                elif relative_strength < -0.02:
                    relative_strength_score = 85

                elif relative_strength < 0:
                    relative_strength_score = 70

                else:
                    relative_strength_score = 40

        except Exception:
            relative_strength_score = 50

    # --------------------------------------------------------
    # 30-Day Momentum
    # --------------------------------------------------------

    thirty_day_return = (
        df["close"].iloc[-1]
        /
        df["close"].iloc[-31]
        - 1
    )

    if direction == "Bullish":

        momentum_30_score = (
            100
            if thirty_day_return > 0.10
            else 85
            if thirty_day_return > 0.05
            else 70
            if thirty_day_return > 0
            else 40
        )

    else:

        momentum_30_score = (
            100
            if thirty_day_return < -0.10
            else 85
            if thirty_day_return < -0.05
            else 70
            if thirty_day_return < 0
            else 40
        )

    # --------------------------------------------------------
    # Final Technical Score
    # --------------------------------------------------------

    final_score = (
        trend_score * 0.20
        + momentum_score * 0.15
        + volume_score * 0.10
        + keltner_score * 0.15
        + price_action_score * 0.15
        + pressure_score * 0.10
        + relative_strength_score * 0.10
        + momentum_30_score * 0.05
    )

    final_score = round(
        final_score,
        1
    )

    # --------------------------------------------------------
    # Confluence Count
    # --------------------------------------------------------

    confluences = 0

    if trend_score >= 70:
        confluences += 1

    if momentum_score >= 70:
        confluences += 1

    if volume_score >= 70:
        confluences += 1

    if keltner_score >= 85:
        confluences += 1

    if price_action_score >= 85:
        confluences += 1

    if pressure_score >= 65:
        confluences += 1

    if relative_strength_score >= 70:
        confluences += 1

    if momentum_30_score >= 70:
        confluences += 1

    # Require at least 3 confluences

    if confluences < 3:
        return None

    # Require minimum relative volume

    if relative_volume < MIN_RELATIVE_VOLUME:
        return None

    # --------------------------------------------------------
    # Setup Name
    # --------------------------------------------------------

    if direction == "Bullish":

        if crossover_recent and keltner_score >= 85:
            setup = "50/100 EMA Cross + Keltner Bullish"

        elif keltner_score >= 85:
            setup = "Keltner Bullish Momentum"

        elif crossover_recent:
            setup = "Fresh 50/100 EMA Bullish Cross"

        else:
            setup = "Bullish Trend Continuation"

    else:

        if crossover_recent and keltner_score >= 85:
            setup = "50/100 EMA Cross + Keltner Bearish"

        elif keltner_score >= 85:
            setup = "Keltner Bearish Momentum"

        elif crossover_recent:
            setup = "Fresh 50/100 EMA Bearish Cross"

        else:
            setup = "Bearish Trend Continuation"

    return {
        "symbol": symbol,
        "direction": direction,
        "score": final_score,
        "setup": setup,
        "crossover_date": crossover_date,
        "crossover_sessions_ago": crossover_sessions_ago,
        "buying_pressure": buying_pressure,
        "selling_pressure": selling_pressure,
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "keltner": keltner_setup,
        "keltner_score": keltner_score,
        "price_action": price_action,
        "price_action_score": price_action_score,
        "relative_volume": relative_volume,
        "relative_strength_score": relative_strength_score,
        "momentum_30_score": momentum_30_score,
        "confluences": confluences,
        "price": price,
        "atr": atr,
        "rsi": rsi,
        "df": df
    }


# ============================================================
# OPTION EXPIRATIONS
# ============================================================

def get_expirations(symbol):

    try:

        response = requests.get(
            f"{BASE_URL}/markets/options/expirations",
            params={
                "symbol": symbol,
                "includeAllRoots": "true"
            },
            headers=HEADERS,
            timeout=15
        )

        if response.status_code != 200:
            return []

        payload = response.json()

        dates = (
            payload
            .get("expirations", {})
            .get("date", [])
        )

        if isinstance(dates, str):
            dates = [dates]

        today = date.today()

        valid = []

        for value in dates:

            try:

                expiry = pd.to_datetime(
                    value
                ).date()

                dte = (
                    expiry - today
                ).days

                if (
                    MIN_DTE
                    <= dte
                    <= MAX_DTE
                ):
                    valid.append(
                        (expiry, dte)
                    )

            except Exception:
                continue

        return valid

    except Exception:
        return []


# ============================================================
# OPTION CHAIN
# ============================================================

def get_option_chain(
    symbol,
    expiration
):

    try:

        response = requests.get(
            f"{BASE_URL}/markets/options/chains",
            params={
                "symbol": symbol,
                "expiration": expiration.isoformat(),
                "greeks": "true"
            },
            headers=HEADERS,
            timeout=15
        )

        if response.status_code != 200:
            return []

        payload = response.json()

        options = (
            payload
            .get("options", {})
            .get("option", [])
        )

        if isinstance(options, dict):
            options = [options]

        return options

    except Exception:
        return []


# ============================================================
# OPTION MATH
# ============================================================

def option_math(
    option,
    underlying_price
):

    strike = safe_float(
        option.get("strike")
    )

    bid = safe_float(
        option.get("bid"),
        0
    )

    ask = safe_float(
        option.get("ask"),
        0
    )

    last = safe_float(
        option.get("last"),
        0
    )

    volume = safe_float(
        option.get("volume"),
        0
    )

    open_interest = safe_float(
        option.get("open_interest"),
        0
    )

    greeks = option.get(
        "greeks",
        {}
    ) or {}

    delta = safe_float(
        greeks.get("delta")
    )

    if np.isnan(delta):
        delta = safe_float(
            option.get("delta")
        )

    # --------------------------------------------------------
    # Executable price
    # --------------------------------------------------------

    if ask > 0:
        premium = ask

    elif last > 0:
        premium = last

    elif bid > 0:
        premium = bid

    else:
        premium = 0

    # --------------------------------------------------------
    # Spread
    # --------------------------------------------------------

    if ask > 0 and bid > 0:

        spread_pct = (
            ask - bid
        ) / ask

    else:

        spread_pct = 999

    # --------------------------------------------------------
    # Intrinsic
    # --------------------------------------------------------

    option_type = (
        str(
            option.get("option_type", "")
        )
        .lower()
    )

    if option_type == "call":

        intrinsic = max(
            underlying_price - strike,
            0
        )

    else:

        intrinsic = max(
            strike - underlying_price,
            0
        )

    # --------------------------------------------------------
    # Time Value
    # --------------------------------------------------------

    time_value = max(
        premium - intrinsic,
        0
    )

    time_value_pct = (
        time_value /
        underlying_price
        if underlying_price > 0
        else 999
    )

    return {
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "last": last,
        "premium": premium,
        "volume": volume,
        "open_interest": open_interest,
        "delta": delta,
        "spread_pct": spread_pct,
        "intrinsic": intrinsic,
        "time_value": time_value,
        "time_value_pct": time_value_pct,
        "option_type": option_type
    }


# ============================================================
# SINGLE OPTION CANDIDATES
# ============================================================

def single_option_candidates(
    candidate
):

    symbol = candidate["symbol"]
    direction = candidate["direction"]
    underlying = candidate["price"]
    atr = candidate["atr"]

    expirations = get_expirations(
        symbol
    )

    candidates = []

    for expiry, dte in expirations:

        chain = get_option_chain(
            symbol,
            expiry
        )

        for option in chain:

            option_type = str(
                option.get(
                    "option_type",
                    ""
                )
            ).lower()

            desired_type = (
                "call"
                if direction == "Bullish"
                else "put"
            )

            if option_type != desired_type:
                continue

            data = option_math(
                option,
                underlying
            )

            if data["premium"] <= 0:
                continue

            if (
                data["open_interest"]
                < MIN_OPEN_INTEREST
            ):
                continue

            if (
                data["volume"]
                < MIN_OPTION_VOLUME
            ):
                continue

            if (
                data["spread_pct"]
                > MAX_SPREAD_PCT
            ):
                continue

            if np.isnan(data["delta"]):
                continue

            delta_abs = abs(
                data["delta"]
            )

            if (
                delta_abs < MIN_DELTA
                or
                delta_abs > MAX_DELTA
            ):
                continue

            # Must be ITM

            if direction == "Bullish":

                if data["strike"] >= underlying:
                    continue

            else:

                if data["strike"] <= underlying:
                    continue

            # ------------------------------------------------
            # Hughes 1% Rule
            # ------------------------------------------------

            if (
                data["time_value_pct"]
                > HUGHES_TIME_VALUE_PCT
            ):
                continue

            entry = data["premium"]

            stop = (
                entry *
                STOP_PERCENT_OF_PREMIUM
            )

            risk = (
                entry -
                stop
            )

            # ------------------------------------------------
            # Modeled Stock Move
            # ------------------------------------------------

            expected_move = max(
                atr * 1.50,
                underlying * 0.03
            )

            if direction == "Bullish":

                stock_target = (
                    underlying +
                    expected_move
                )

                option_gain = (
                    delta_abs *
                    expected_move
                )

            else:

                stock_target = (
                    underlying -
                    expected_move
                )

                option_gain = (
                    delta_abs *
                    expected_move
                )

            modeled_target = (
                entry +
                option_gain
            )

            # Minimum desired upside

            modeled_target = max(
                modeled_target,
                entry * 1.125
            )

            reward = (
                modeled_target -
                entry
            )

            if risk <= 0:
                continue

            rr = (
                reward /
                risk
            )

            if rr < MIN_RISK_REWARD:
                continue

            profit_pct = (
                reward /
                entry
                * 100
            )

            candidates.append({

                "structure":
                    "Single ITM Option",

                "symbol":
                    symbol,

                "direction":
                    direction,

                "expiration":
                    expiry,

                "dte":
                    dte,

                "option_type":
                    option_type,

                "long_strike":
                    data["strike"],

                "short_strike":
                    None,

                "long_bid":
                    data["bid"],

                "long_ask":
                    data["ask"],

                "short_bid":
                    None,

                "short_ask":
                    None,

                "entry":
                    entry,

                "stop":
                    stop,

                "target":
                    modeled_target,

                "risk":
                    risk,

                "reward":
                    reward,

                "rr":
                    rr,

                "profit_pct":
                    profit_pct,

                "delta":
                    delta_abs,

                "intrinsic":
                    data["intrinsic"],

                "time_value":
                    data["time_value"],

                "time_value_pct":
                    data["time_value_pct"],

                "max_profit":
                    None,

                "option_volume":
                    data["volume"],

                "open_interest":
                    data["open_interest"],

                "technical_score":
                    candidate["score"],

                "confluences":
                    candidate["confluences"]
            })

    return candidates


# ============================================================
# DEBIT SPREAD CANDIDATES
# ============================================================

def debit_spread_candidates(
    candidate
):

    symbol = candidate["symbol"]
    direction = candidate["direction"]
    underlying = candidate["price"]

    expirations = get_expirations(
        symbol
    )

    candidates = []

    for expiry, dte in expirations:

        chain = get_option_chain(
            symbol,
            expiry
        )

        desired_type = (
            "call"
            if direction == "Bullish"
            else "put"
        )

        options = []

        for option in chain:

            if str(
                option.get(
                    "option_type",
                    ""
                )
            ).lower() != desired_type:

                continue

            data = option_math(
                option,
                underlying
            )

            if data["premium"] <= 0:
                continue

            if (
                data["open_interest"]
                < MIN_OPEN_INTEREST
            ):
                continue

            if (
                data["volume"]
                < MIN_OPTION_VOLUME
            ):
                continue

            if (
                data["spread_pct"]
                > MAX_SPREAD_PCT
            ):
                continue

            if np.isnan(data["delta"]):
                continue

            if abs(data["delta"]) < MIN_DELTA:
                continue

            if abs(data["delta"]) > MAX_DELTA:
                continue

            # Long leg must be ITM

            if direction == "Bullish":

                if data["strike"] >= underlying:
                    continue

            else:

                if data["strike"] <= underlying:
                    continue

            # Hughes 1% rule applies to LONG leg

            if (
                data["time_value_pct"]
                > HUGHES_TIME_VALUE_PCT
            ):
                continue

            data["raw"] = option

            options.append(data)

        if len(options) < 2:
            continue

        # ----------------------------------------------------
        # Build vertical spreads
        # ----------------------------------------------------

        for long_leg in options:

            for short_leg in options:

                if (
                    long_leg["strike"]
                    ==
                    short_leg["strike"]
                ):
                    continue

                if direction == "Bullish":

                    # Bull call spread:
                    # Long lower strike
                    # Short higher strike

                    if (
                        long_leg["strike"]
                        >=
                        short_leg["strike"]
                    ):
                        continue

                    width = (
                        short_leg["strike"]
                        -
                        long_leg["strike"]
                    )

                else:

                    # Bear put spread:
                    # Long higher strike
                    # Short lower strike

                    if (
                        long_leg["strike"]
                        <=
                        short_leg["strike"]
                    ):
                        continue

                    width = (
                        long_leg["strike"]
                        -
                        short_leg["strike"]
                    )

                if width <= 0:
                    continue

                # ------------------------------------------------
                # Conservative executable debit
                # ------------------------------------------------

                long_price = (
                    long_leg["ask"]
                    if long_leg["ask"] > 0
                    else long_leg["premium"]
                )

                short_credit = (
                    short_leg["bid"]
                    if short_leg["bid"] > 0
                    else short_leg["premium"]
                )

                debit = (
                    long_price -
                    short_credit
                )

                if debit <= 0:
                    continue

                if debit >= width:
                    continue

                max_profit = (
                    width -
                    debit
                )

                # ------------------------------------------------
                # 70% premium stop
                # ------------------------------------------------

                stop = (
                    debit *
                    STOP_PERCENT_OF_PREMIUM
                )

                risk_to_stop = (
                    debit -
                    stop
                )

                # ------------------------------------------------
                # Target at 75% of max spread profit
                # ------------------------------------------------

                target = (
                    debit +
                    max_profit * 0.75
                )

                reward = (
                    target -
                    debit
                )

                if risk_to_stop <= 0:
                    continue

                rr = (
                    reward /
                    risk_to_stop
                )

                if rr < MIN_RISK_REWARD:
                    continue

                profit_pct = (
                    reward /
                    debit
                    * 100
                )

                candidates.append({

                    "structure":
                        "Debit Spread",

                    "symbol":
                        symbol,

                    "direction":
                        direction,

                    "expiration":
                        expiry,

                    "dte":
                        dte,

                    "option_type":
                        desired_type,

                    "long_strike":
                        long_leg["strike"],

                    "short_strike":
                        short_leg["strike"],

                    "long_bid":
                        long_leg["bid"],

                    "long_ask":
                        long_leg["ask"],

                    "short_bid":
                        short_leg["bid"],

                    "short_ask":
                        short_leg["ask"],

                    "entry":
                        debit,

                    "stop":
                        stop,

                    "target":
                        target,

                    "risk":
                        risk_to_stop,

                    "reward":
                        reward,

                    "rr":
                        rr,

                    "profit_pct":
                        profit_pct,

                    "delta":
                        abs(
                            long_leg["delta"]
                        ),

                    "intrinsic":
                        long_leg["intrinsic"],

                    "time_value":
                        long_leg["time_value"],

                    "time_value_pct":
                        long_leg[
                            "time_value_pct"
                        ],

                    "max_profit":
                        max_profit,

                    "option_volume":
                        long_leg["volume"],

                    "open_interest":
                        long_leg[
                            "open_interest"
                        ],

                    "technical_score":
                        candidate["score"],

                    "confluences":
                        candidate["confluences"]
                })

    return candidates


# ============================================================
# OPTION ENGINE
# ============================================================

def build_option_engine(candidate):

    singles = single_option_candidates(
        candidate
    )

    spreads = debit_spread_candidates(
        candidate
    )

    all_candidates = (
        singles +
        spreads
    )

    if not all_candidates:
        return None

    # Rank option structure

    all_candidates.sort(
        key=lambda x: (
            x["technical_score"],
            x["rr"],
            x["profit_pct"],
            x["confluences"]
        ),
        reverse=True
    )

    return all_candidates[0]


# ============================================================
# FORMAT OPTION TRADE
# ============================================================

def option_description(trade):

    symbol = trade["symbol"]

    expiry = trade["expiration"]

    direction = trade["direction"]

    if trade["structure"] == "Single ITM Option":

        option_name = (
            "Call"
            if direction == "Bullish"
            else "Put"
        )

        return (
            f"Buy-to-Open {symbol} "
            f"{expiry} "
            f"${trade['long_strike']:.2f} "
            f"{option_name}"
        )

    else:

        if direction == "Bullish":

            return (
                f"Buy-to-Open {symbol} "
                f"${trade['long_strike']:.2f} Call / "
                f"Sell-to-Open ${trade['short_strike']:.2f} Call "
                f"{expiry}"
            )

        else:

            return (
                f"Buy-to-Open {symbol} "
                f"${trade['long_strike']:.2f} Put / "
                f"Sell-to-Open ${trade['short_strike']:.2f} Put "
                f"{expiry}"
            )


# ============================================================
# POSITION SIZING
# ============================================================

def position_size(
    entry,
    stop,
    account
):

    risk_per_contract = (
        entry -
        stop
    ) * 100

    if risk_per_contract <= 0:
        return 0

    # Maximum intended risk = 2% account

    max_account_risk = (
        account * 0.02
    )

    contracts = int(
        max_account_risk
        /
        risk_per_contract
    )

    return max(
        contracts,
        0
    )


# ============================================================
# DISPLAY TRADE
# ============================================================

def display_trade(
    trade,
    candidate
):

    st.subheader(
        "🎯 Golden Option Structure"
    )

    st.success(
        option_description(trade)
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Entry",
        f"${trade['entry']:.2f}"
    )

    col2.metric(
        "70% Stop",
        f"${trade['stop']:.2f}"
    )

    col3.metric(
        "Target",
        f"${trade['target']:.2f}"
    )

    col4.metric(
        "R:R",
        f"1:{trade['rr']:.2f}"
    )

    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "DTE",
        trade["dte"]
    )

    c2.metric(
        "Delta",
        f"{trade['delta']:.2f}"
    )

    c3.metric(
        "Time Value",
        f"{trade['time_value_pct'] * 100:.2f}%"
    )

    c4.metric(
        "Potential Gain",
        f"{trade['profit_pct']:.1f}%"
    )

    contracts = position_size(
        trade["entry"],
        trade["stop"],
        account_size
    )

    c5.metric(
        "Contracts",
        contracts
    )

    # --------------------------------------------------------
    # Hughes rule
    # --------------------------------------------------------

    if (
        trade["time_value_pct"]
        <= HUGHES_TIME_VALUE_PCT
    ):

        st.success(
            f"✅ Hughes 1% Rule PASSED — "
            f"Time value = "
            f"{trade['time_value_pct'] * 100:.2f}%"
        )

    else:

        st.error(
            "❌ Hughes 1% Rule FAILED"
        )

    # --------------------------------------------------------
    # Debit spread details
    # --------------------------------------------------------

    if trade["structure"] == "Debit Spread":

        st.write(
            f"Long leg ask: "
            f"${trade['long_ask']:.2f}"
        )

        st.write(
            f"Short leg bid: "
            f"${trade['short_bid']:.2f}"
        )

        st.write(
            f"Net debit: "
            f"${trade['entry']:.2f}"
        )

        st.write(
            f"Maximum expiration profit: "
            f"${trade['max_profit']:.2f} "
            f"per share / "
            f"${trade['max_profit'] * 100:.0f} "
            f"per contract"
        )

    # --------------------------------------------------------
    # Technical confirmation
    # --------------------------------------------------------

    st.subheader(
        "🧠 Technical Confirmation"
    )

    tech = pd.DataFrame({
        "Factor": [
            "Direction",
            "Technical Score",
            "Confluences",
            "Setup",
            "Keltner",
            "Price Action",
            "Relative Volume",
            "Buying Pressure",
            "Selling Pressure",
            "50/100 Cross"
        ],

        "Result": [
            candidate["direction"],
            f"{candidate['score']}/100",
            candidate["confluences"],
            candidate["setup"],
            candidate["keltner"],
            candidate["price_action"],
            f"{candidate['relative_volume']:.2f}x",
            f"{candidate['buying_pressure']:.1f}%",
            f"{candidate['selling_pressure']:.1f}%",
            (
                str(
                    candidate[
                        "crossover_date"
                    ]
                )
            )
        ]
    })

    st.dataframe(
        tech,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TEST SYMBOL ANALYSIS
# ============================================================

if run_test:

    st.divider()

    st.header(
        f"🔬 TEST ANALYSIS — {test_symbol}"
    )

    with st.spinner(
        f"Analyzing {test_symbol}..."
    ):

        spy_test = get_history("SPY")

        if spy_test is not None:
            spy_test = calculate_indicators(
                spy_test
            )

        candidate = analyze_symbol(
            test_symbol,
            spy_test
        )

    if candidate is None:

        st.warning(
            f"{test_symbol} does not currently meet "
            f"the Golden technical requirements."
        )

    else:

        st.success(
            f"{test_symbol} passed the technical engine."
        )

        cols = st.columns(5)

        cols[0].metric(
            "Direction",
            candidate["direction"]
        )

        cols[1].metric(
            "Score",
            candidate["score"]
        )

        cols[2].metric(
            "Confluences",
            candidate["confluences"]
        )

        cols[3].metric(
            "Relative Volume",
            f"{candidate['relative_volume']:.2f}x"
        )

        cols[4].metric(
            "RSI",
            f"{candidate['rsi']:.1f}"
        )

        with st.spinner(
            "Searching 7–40 DTE option structures..."
        ):

            option_trade = build_option_engine(
                candidate
            )

        if option_trade is None:

            st.warning(
                "Technical setup passed, but no option "
                "structure currently passes the Hughes "
                "1% rule + liquidity + R:R requirements."
            )

        else:

            display_trade(
                option_trade,
                candidate
            )


# ============================================================
# GOLDEN SCAN
# ============================================================

if run_scan:

    st.divider()

    st.header(
        "🥇 GOLDEN SCAN"
    )

    st.write(
        "Searching for the highest-quality technical "
        "setup first, then validating the option structure."
    )

    # --------------------------------------------------------
    # SPY baseline
    # --------------------------------------------------------

    with st.spinner(
        "Loading SPY relative-strength baseline..."
    ):

        spy_df = get_history("SPY")

        if spy_df is not None:
            spy_df = calculate_indicators(
                spy_df
            )

    technical_candidates = []

    progress = st.progress(0)

    status = st.empty()

    universe = [
        x
        for x in SCANNER_UNIVERSE
        if x not in MAG_7
    ]

    total = len(universe)

    for index, symbol in enumerate(
        universe
    ):

        status.write(
            f"Scanning {symbol}..."
        )

        try:

            candidate = analyze_symbol(
                symbol,
                spy_df
            )

            if (
                candidate is not None
                and
                candidate["score"]
                >= min_score_setting
            ):

                technical_candidates.append(
                    candidate
                )

        except Exception:
            pass

        progress.progress(
            (index + 1) / total
        )

    status.empty()
    progress.empty()

    # --------------------------------------------------------
    # Rank technical candidates
    # --------------------------------------------------------

    technical_candidates.sort(
        key=lambda x: (
            x["score"],
            x["confluences"],
            x["relative_volume"]
        ),
        reverse=True
    )

    technical_candidates = (
        technical_candidates[
            :MAX_RESULTS
        ]
    )

    if not technical_candidates:

        st.warning(
            "No stocks currently meet the Golden "
            "technical requirements."
        )

        st.stop()

    st.success(
        f"Found {len(technical_candidates)} "
        "technical candidate(s)."
    )

    # --------------------------------------------------------
    # Technical candidate table
    # --------------------------------------------------------

    technical_rows = []

    for c in technical_candidates:

        technical_rows.append({

            "Ticker":
                c["symbol"],

            "Direction":
                c["direction"],

            "Score":
                c["score"],

            "Setup":
                c["setup"],

            "50/100 Cross":
                c["crossover_date"],

            "Sessions Ago":
                c["crossover_sessions_ago"],

            "Buying Pressure":
                f"{c['buying_pressure']:.1f}%",

            "Selling Pressure":
                f"{c['selling_pressure']:.1f}%",

            "Keltner":
                c["keltner"],

            "Momentum":
                c["momentum_score"],

            "Price Action":
                c["price_action"],

            "Relative Volume":
                f"{c['relative_volume']:.2f}x",

            "Confluences":
                c["confluences"],

            "Price":
                f"${c['price']:.2f}"
        })

    technical_df = pd.DataFrame(
        technical_rows
    )

    st.subheader(
        "📊 Technical Candidates"
    )

    st.dataframe(
        technical_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Phase 2 — Option Engine
    # --------------------------------------------------------

    st.divider()

    st.header(
        "⚙️ PHASE 2 — HUGHES OPTION ENGINE"
    )

    option_candidates = []

    option_progress = st.progress(0)

    option_status = st.empty()

    total_options = len(
        technical_candidates
    )

    for index, candidate in enumerate(
        technical_candidates
    ):

        symbol = candidate["symbol"]

        option_status.write(
            f"Analyzing options for {symbol}..."
        )

        try:

            trade = build_option_engine(
                candidate
            )

            if trade is not None:

                trade["candidate"] = candidate

                option_candidates.append(
                    trade
                )

        except Exception:
            pass

        option_progress.progress(
            (index + 1) /
            total_options
        )

    option_status.empty()
    option_progress.empty()

    if not option_candidates:

        st.warning(
            "Technical candidates were found, "
            "but none currently have an option structure "
            "that passes all Golden requirements."
        )

        st.stop()

    # --------------------------------------------------------
    # FINAL RANKING
    # --------------------------------------------------------

    option_candidates.sort(
        key=lambda x: (
            x["technical_score"],
            x["confluences"],
            x["rr"],
            x["profit_pct"]
        ),
        reverse=True
    )

    golden = option_candidates[0]

    # ========================================================
    # GOLDEN CANDIDATE
    # ========================================================

    st.divider()

    st.header(
        "🏆 #1 GOLDEN CANDIDATE"
    )

    candidate = golden["candidate"]

    st.success(
        f"{candidate['symbol']} — "
        f"{candidate['direction']} — "
        f"{candidate['score']}/100"
    )

    top1, top2, top3, top4, top5 = st.columns(5)

    top1.metric(
        "Ticker",
        candidate["symbol"]
    )

    top2.metric(
        "Technical Score",
        f"{candidate['score']}/100"
    )

    top3.metric(
        "Confluence",
        candidate["confluences"]
    )

    top4.metric(
        "Relative Volume",
        f"{candidate['relative_volume']:.2f}x"
    )

    top5.metric(
        "R:R",
        f"1:{golden['rr']:.2f}"
    )

    # --------------------------------------------------------
    # Trade alert
    # --------------------------------------------------------

    st.subheader(
        "🚨 GOLDEN QOD TRADE ALERT"
    )

    st.code(
        option_description(golden)
    )

    st.write(
        f"Entry: ${golden['entry']:.2f}"
    )

    st.write(
        f"Stop: ${golden['stop']:.2f} "
        f"(70% of premium)"
    )

    st.write(
        f"Target: ${golden['target']:.2f}"
    )

    st.write(
        f"Modeled upside: "
        f"{golden['profit_pct']:.1f}%"
    )

    st.write(
        f"DTE: {golden['dte']}"
    )

    st.write(
        f"Long-leg delta: "
        f"{golden['delta']:.2f}"
    )

    st.write(
        f"Long-leg time value: "
        f"{golden['time_value_pct'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Position sizing
    # --------------------------------------------------------

    contracts = position_size(
        golden["entry"],
        golden["stop"],
        account_size
    )

    intended_risk = (
        golden["risk"]
        * 100
        * contracts
    )

    st.info(
        f"Account: ${account_size:,.0f} | "
        f"Suggested contracts: {contracts} | "
        f"Intended modeled stop risk: "
        f"${intended_risk:,.2f}"
    )

    # --------------------------------------------------------
    # Hughes Rule
    # --------------------------------------------------------

    st.subheader(
        "🎯 Hughes 1% Rule"
    )

    if (
        golden["time_value_pct"]
        <= HUGHES_TIME_VALUE_PCT
    ):

        st.success(
            f"PASS — Time value "
            f"{golden['time_value_pct'] * 100:.2f}% "
            f"of underlying price."
        )

    else:

        st.error(
            "FAIL — Time value exceeds 1%."
        )

    # --------------------------------------------------------
    # Spread details
    # --------------------------------------------------------

    if golden["structure"] == "Debit Spread":

        st.subheader(
            "💰 Debit Spread Construction"
        )

        spread_df = pd.DataFrame({

            "Leg": [
                "Long",
                "Short"
            ],

            "Strike": [
                golden["long_strike"],
                golden["short_strike"]
            ],

            "Bid": [
                golden["long_bid"],
                golden["short_bid"]
            ],

            "Ask": [
                golden["long_ask"],
                golden["short_ask"]
            ]
        })

        st.dataframe(
            spread_df,
            use_container_width=True,
            hide_index=True
        )

        st.write(
            f"Net Debit: "
            f"${golden['entry']:.2f}"
        )

        st.write(
            f"Maximum expiration profit: "
            f"${golden['max_profit']:.2f}/share "
            f"or "
            f"${golden['max_profit'] * 100:.0f}/contract"
        )

    # ========================================================
    # ALL OPTION CANDIDATES
    # ========================================================

    st.divider()

    st.header(
        "📋 Phase 2 Option Candidates"
    )

    option_rows = []

    for trade in option_candidates:

        option_rows.append({

            "Ticker":
                trade["symbol"],

            "Direction":
                trade["direction"],

            "Structure":
                trade["structure"],

            "Expiration":
                trade["expiration"],

            "DTE":
                trade["dte"],

            "Long Strike":
                trade["long_strike"],

            "Short Strike":
                trade["short_strike"],

            "Entry":
                round(
                    trade["entry"],
                    2
                ),

            "Stop":
                round(
                    trade["stop"],
                    2
                ),

            "Target":
                round(
                    trade["target"],
                    2
                ),

            "R:R":
                f"1:{trade['rr']:.2f}",

            "Profit %":
                f"{trade['profit_pct']:.1f}%",

            "Delta":
                round(
                    trade["delta"],
                    2
                ),

            "Time Value %":
                f"{trade['time_value_pct'] * 100:.2f}%",

            "Technical Score":
                trade["technical_score"],

            "Confluences":
                trade["confluences"]
        })

    option_df = pd.DataFrame(
        option_rows
    )

    st.dataframe(
        option_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🥇 Golden QOD Scanner • Tradier market data • "
    "Hughes-style 1% time-value filter • "
    "7–40 DTE • 70% premium stop • "
    "Minimum 1:1.25 modeled R:R • "
    "No trades are placed • For research/educational use."
)
