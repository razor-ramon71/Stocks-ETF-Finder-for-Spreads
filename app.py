import os
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# ============================================================
# GOLDEN SCANNER
# Swing Stock + Options Scanner
# Stocks ONLY — No ETFs
# Tradier Market Data
# ============================================================

st.set_page_config(
    page_title="Golden Scanner",
    page_icon="🏆",
    layout="wide"
)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

MIN_PRICE = 10.00
MAX_PRICE = 500.00

MIN_REL_VOLUME = 1.20

EMA_FAST = 20
EMA_50 = 50
EMA_100 = 100

RSI_LENGTH = 14
ATR_LENGTH = 14

MIN_SCORE = 60

OPTION_DTE_MIN = 14
OPTION_DTE_MAX = 45

ACCOUNT_MIN = 2000
ACCOUNT_MAX = 5000

# Chuck Hughes 1% rule:
# Preferred option time value <= 1% of underlying price
ONE_PERCENT_RULE = 0.01


# ------------------------------------------------------------
# TRADIER CONNECTION
# ------------------------------------------------------------

TRADIER_TOKEN = st.secrets.get(
    "TRADIER_TOKEN",
    os.getenv("TRADIER_TOKEN", "")
)

TRADIER_BASE = "https://api.tradier.com/v1"

HEADERS = {
    "Authorization": f"Bearer {TRADIER_TOKEN}",
    "Accept": "application/json"
}


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

if "option_results" not in st.session_state:
    st.session_state.option_results = {}


# ------------------------------------------------------------
# PAGE HEADER
# ------------------------------------------------------------

st.title("🏆 Golden Scanner")
st.caption(
    "Golden Swing Hunter • Stocks Only • Optionable • 14–45 DTE"
)

if not TRADIER_TOKEN:
    st.error(
        "Tradier token not found. Add TRADIER_TOKEN to Streamlit secrets."
    )
    st.stop()

st.success("Tradier connection configured.")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def get_json(url, params=None):
    try:
        r = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=20
        )

        if r.status_code != 200:
            return None

        return r.json()

    except Exception:
        return None


# ------------------------------------------------------------
# STOCK UNIVERSE
# ------------------------------------------------------------

# Large, liquid individual-stock universe.
# ETFs are deliberately NOT included.

STOCK_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA",
    "AVGO","AMD","NFLX","CRM","ORCL","ADBE","CSCO","QCOM",
    "INTC","MU","AMAT","LRCX","KLAC","TXN","MRVL","ARM",
    "SMCI","PANW","CRWD","PLTR","NOW","SNOW","DDOG","NET",
    "SHOP","UBER","ABNB","BKNG","DASH","PYPL","SQ","COIN",
    "HOOD","SOFI","RIVN","F","GM","NIO","XPEV",
    "JPM","BAC","WFC","C","GS","MS","BLK","SCHW",
    "V","MA","AXP","COF",
    "WMT","COST","TGT","HD","LOW","MCD","SBUX","NKE",
    "KO","PEP","PG","CL","KHC",
    "XOM","CVX","COP","OXY","SLB","HAL","EOG","MPC",
    "CAT","DE","GE","HON","RTX","BA","LMT","UPS","UNP",
    "LLY","JNJ","PFE","MRK","ABBV","AMGN",
    "TMO","DHR","ISRG",
    "DIS","CMCSA","T","VZ",
    "DELL","HPQ","IBM",
    "ETN","EMR","PH","CARR",
    "MAR","HLT","DAL","UAL","LUV",
    "ROKU","RBLX","TTD","PINS","SNAP",
    "CVS","CI","HUM","ELV",
    "CVNA","DKNG","RBLX",
    "TENB","ZS","OKTA","FTNT",
    "MSTR","MARA","RIOT",
    "APP","AXON","HOUS",
]


# ------------------------------------------------------------
# HISTORICAL DATA
# ------------------------------------------------------------

@st.cache_data(ttl=900)
def get_history(symbol, days=260):

    end = datetime.now()
    start = end - timedelta(days=days)

    data = get_json(
        f"{TRADIER_BASE}/markets/history",
        {
            "symbol": symbol,
            "interval": "daily",
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d")
        }
    )

    if not data:
        return None

    try:
        rows = data["history"]["day"]

        if not isinstance(rows, list):
            rows = [rows]

        df = pd.DataFrame(rows)

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        df = df.sort_values("date").reset_index(drop=True)

        return df

    except Exception:
        return None


# ------------------------------------------------------------
# TECHNICAL INDICATORS
# ------------------------------------------------------------

def calculate_indicators(df):

    df = df.copy()

    # EMAs
    df["ema20"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=EMA_50,
        adjust=False
    ).mean()

    df["ema100"] = df["close"].ewm(
        span=EMA_100,
        adjust=False
    ).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_LENGTH,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_LENGTH,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - prev_close)
    tr3 = abs(df["low"] - prev_close)

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(ATR_LENGTH).mean()

    # Average volume
    df["avg_volume"] = df["volume"].rolling(20).mean()

    df["relative_volume"] = (
        df["volume"] /
        df["avg_volume"].replace(0, np.nan)
    )

    # Keltner-style bands
    df["kc_mid"] = df["ema20"]

    df["kc_upper"] = (
        df["ema20"] +
        2 * df["atr"]
    )

    df["kc_lower"] = (
        df["ema20"] -
        2 * df["atr"]
    )

    # Momentum
    df["momentum_10"] = (
        df["close"].pct_change(10) * 100
    )

    # 50/100 EMA crossover
    df["ema_cross_bull"] = (
        (df["ema50"] > df["ema100"]) &
        (df["ema50"].shift(1) <= df["ema100"].shift(1))
    )

    df["ema_cross_bear"] = (
        (df["ema50"] < df["ema100"]) &
        (df["ema50"].shift(1) >= df["ema100"].shift(1))
    )

    # Recent crossover
    df["recent_bull_cross"] = (
        df["ema_cross_bull"]
        .rolling(30)
        .max()
        .fillna(0)
        .astype(bool)
    )

    df["recent_bear_cross"] = (
        df["ema_cross_bear"]
        .rolling(30)
        .max()
        .fillna(0)
        .astype(bool)
    )

    return df


# ------------------------------------------------------------
# CANDLE / PRICE ACTION
# ------------------------------------------------------------

def candle_pattern(df):

    if len(df) < 3:
        return "None"

    c = df.iloc[-1]

    body = abs(c["close"] - c["open"])

    upper_wick = (
        c["high"] -
        max(c["open"], c["close"])
    )

    lower_wick = (
        min(c["open"], c["close"]) -
        c["low"]
    )

    total_range = c["high"] - c["low"]

    if total_range <= 0:
        return "None"

    # Bullish engulfing
    p = df.iloc[-2]

    if (
        p["close"] < p["open"] and
        c["close"] > c["open"] and
        c["close"] > p["open"] and
        c["open"] < p["close"]
    ):
        return "Bullish Engulfing"

    # Bearish engulfing
    if (
        p["close"] > p["open"] and
        c["close"] < c["open"] and
        c["open"] > p["close"] and
        c["close"] < p["open"]
    ):
        return "Bearish Engulfing"

    # Hammer
    if (
        lower_wick > body * 2 and
        upper_wick < body
    ):
        return "Hammer"

    # Shooting star
    if (
        upper_wick > body * 2 and
        lower_wick < body
    ):
        return "Shooting Star"

    # Strong candle
    if body / total_range > 0.65:
        if c["close"] > c["open"]:
            return "Strong Bull Candle"
        else:
            return "Strong Bear Candle"

    return "Neutral"


# ------------------------------------------------------------
# SUPPORT / RESISTANCE
# ------------------------------------------------------------

def support_resistance(df):

    recent = df.tail(20)

    support = recent["low"].min()
    resistance = recent["high"].max()

    return support, resistance


# ------------------------------------------------------------
# STOCK SCORING
# ------------------------------------------------------------

def score_stock(df):

    if df is None or len(df) < 120:
        return None

    d = calculate_indicators(df)

    last = d.iloc[-1]

    price = safe_float(last["close"])

    if np.isnan(price):
        return None

    # Price filter
    if price < MIN_PRICE or price > MAX_PRICE:
        return None

    score_bull = 0
    score_bear = 0

    reasons_bull = []
    reasons_bear = []

    # --------------------------------------------------------
    # EMA STRUCTURE
    # --------------------------------------------------------

    if last["ema20"] > last["ema50"]:
        score_bull += 10
        reasons_bull.append("20 EMA > 50 EMA")

    if last["ema50"] > last["ema100"]:
        score_bull += 15
        reasons_bull.append("50 EMA > 100 EMA")

    if last["ema20"] < last["ema50"]:
        score_bear += 10
        reasons_bear.append("20 EMA < 50 EMA")

    if last["ema50"] < last["ema100"]:
        score_bear += 15
        reasons_bear.append("50 EMA < 100 EMA")

    # --------------------------------------------------------
    # RECENT CROSSOVER
    # --------------------------------------------------------

    if last["recent_bull_cross"]:
        score_bull += 15
        reasons_bull.append("50/100 bullish crossover ≤30 sessions")

    if last["recent_bear_cross"]:
        score_bear += 15
        reasons_bear.append("50/100 bearish crossover ≤30 sessions")

    # --------------------------------------------------------
    # PRICE ABOVE / BELOW EMA
    # --------------------------------------------------------

    if price > last["ema20"]:
        score_bull += 8
        reasons_bull.append("Price above 20 EMA")
    else:
        score_bear += 8
        reasons_bear.append("Price below 20 EMA")

    if price > last["ema50"]:
        score_bull += 5
        reasons_bull.append("Price above 50 EMA")
    else:
        score_bear += 5
        reasons_bear.append("Price below 50 EMA")

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = safe_float(last["rsi"])

    if not np.isnan(rsi):

        if 50 <= rsi <= 70:
            score_bull += 10
            reasons_bull.append(f"RSI bullish ({rsi:.1f})")

        elif 30 <= rsi < 50:
            score_bear += 7
            reasons_bear.append(f"RSI bearish ({rsi:.1f})")

        elif rsi > 70:
            score_bull += 4
            reasons_bull.append(f"RSI strong ({rsi:.1f})")

        elif rsi < 30:
            score_bear += 4
            reasons_bear.append(f"RSI weak ({rsi:.1f})")

    # --------------------------------------------------------
    # RELATIVE VOLUME
    # --------------------------------------------------------

    rv = safe_float(last["relative_volume"])

    if not np.isnan(rv):

        if rv >= MIN_REL_VOLUME:
            score_bull += 8
            score_bear += 8

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = safe_float(last["momentum_10"])

    if not np.isnan(momentum):

        if momentum > 3:
            score_bull += 10
            reasons_bull.append(
                f"10-day momentum +{momentum:.1f}%"
            )

        elif momentum < -3:
            score_bear += 10
            reasons_bear.append(
                f"10-day momentum {momentum:.1f}%"
            )

    # --------------------------------------------------------
    # KELTNER
    # --------------------------------------------------------

    if price > last["kc_mid"]:
        score_bull += 5
        reasons_bull.append("Keltner bullish")

    if price < last["kc_mid"]:
        score_bear += 5
        reasons_bear.append("Keltner bearish")

    # --------------------------------------------------------
    # PRICE ACTION
    # --------------------------------------------------------

    pattern = candle_pattern(d)

    if "Bull" in pattern or pattern == "Hammer":
        score_bull += 7
        reasons_bull.append(pattern)

    if "Bear" in pattern or pattern == "Shooting Star":
        score_bear += 7
        reasons_bear.append(pattern)

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    support, resistance = support_resistance(d)

    distance_to_resistance = (
        (resistance - price) / price * 100
    )

    distance_to_support = (
        (price - support) / price * 100
    )

    if 0 <= distance_to_resistance <= 5:
        score_bull += 5
        reasons_bull.append("Near breakout resistance")

    if 0 <= distance_to_support <= 5:
        score_bear += 5
        reasons_bear.append("Near breakdown support")

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if score_bull >= score_bear:
        direction = "BULLISH"
        score = score_bull
        reasons = reasons_bull
    else:
        direction = "BEARISH"
        score = score_bear
        reasons = reasons_bear

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if score >= 80:
        confidence = "A+"
    elif score >= 70:
        confidence = "A"
    elif score >= 60:
        confidence = "B"
    elif score >= 50:
        confidence = "C"
    else:
        confidence = "PASS"

    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------

    if direction == "BULLISH":

        if last["ema50"] > last["ema100"] and last["recent_bull_cross"]:
            setup = "Golden Trend / Pullback"

        elif price > resistance * 0.995:
            setup = "Breakout Watch"

        elif price > last["ema20"]:
            setup = "Momentum Continuation"

        else:
            setup = "Bullish Reversal"

    else:

        if last["ema50"] < last["ema100"] and last["recent_bear_cross"]:
            setup = "Death Trend / Rally"

        elif price < support * 1.005:
            setup = "Breakdown Watch"

        elif price < last["ema20"]:
            setup = "Momentum Continuation"

        else:
            setup = "Bearish Reversal"

    # --------------------------------------------------------
    # STOP / TARGET
    # --------------------------------------------------------

    atr = safe_float(last["atr"])

    if np.isnan(atr) or atr <= 0:
        atr = price * 0.03

    if direction == "BULLISH":

        stop = price - 1.2 * atr
        target = price + 1.5 * (price - stop)

    else:

        stop = price + 1.2 * atr
        target = price - 1.5 * (stop - price)

    risk_pct = abs(price - stop) / price * 100

    return {
        "price": price,
        "direction": direction,
        "score": score,
        "confidence": confidence,
        "setup": setup,
        "crossover_date": (
            d.loc[d["ema_cross_bull"], "date"].iloc[-1]
            if direction == "BULLISH"
            and d["ema_cross_bull"].any()
            else
            d.loc[d["ema_cross_bear"], "date"].iloc[-1]
            if direction == "BEARISH"
            and d["ema_cross_bear"].any()
            else None
        ),
        "relative_volume": rv,
        "rsi": rsi,
        "momentum": momentum,
        "support": support,
        "resistance": resistance,
        "pattern": pattern,
        "stop": stop,
        "target": target,
        "risk_pct": risk_pct,
        "reasons": reasons
    }


# ------------------------------------------------------------
# OPTION CHAIN
# ------------------------------------------------------------

@st.cache_data(ttl=300)
def get_option_expirations(symbol):

    data = get_json(
        f"{TRADIER_BASE}/markets/options/expirations",
        {
            "symbol": symbol,
            "includeAllRoots": "true",
            "strikes": "false"
        }
    )

    if not data:
        return []

    try:
        dates = data["expirations"]["date"]

        if isinstance(dates, str):
            dates = [dates]

        return dates

    except Exception:
        return []


@st.cache_data(ttl=300)
def get_option_chain(symbol, expiration):

    data = get_json(
        f"{TRADIER_BASE}/markets/options/chains",
        {
            "symbol": symbol,
            "expiration": expiration,
            "greeks": "true"
        }
    )

    if not data:
        return None

    try:

        options = data["options"]["option"]

        if not isinstance(options, list):
            options = [options]

        return pd.DataFrame(options)

    except Exception:
        return None


# ------------------------------------------------------------
# HUGHES 1% RULE
# ------------------------------------------------------------

def option_time_value(
    option_price,
    underlying_price,
    strike,
    option_type
):

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

    return max(
        option_price - intrinsic,
        0
    )


def select_option(
    symbol,
    stock_data
):

    price = stock_data["price"]
    direction = stock_data["direction"]

    expirations = get_option_expirations(symbol)

    if not expirations:
        return None

    today = datetime.now().date()

    candidates = []

    for exp in expirations:

        try:
            exp_date = datetime.strptime(
                exp,
                "%Y-%m-%d"
            ).date()
        except Exception:
            continue

        dte = (exp_date - today).days

        if dte < OPTION_DTE_MIN:
            continue

        if dte > OPTION_DTE_MAX:
            continue

        chain = get_option_chain(
            symbol,
            exp
        )

        if chain is None or chain.empty:
            continue

        option_type = (
            "call"
            if direction == "BULLISH"
            else "put"
        )

        chain = chain[
            chain["option_type"] == option_type
        ].copy()

        if chain.empty:
            continue

        chain["strike"] = pd.to_numeric(
            chain["strike"],
            errors="coerce"
        )

        chain["bid"] = pd.to_numeric(
            chain["bid"],
            errors="coerce"
        )

        chain["ask"] = pd.to_numeric(
            chain["ask"],
            errors="coerce"
        )

        chain = chain.dropna(
            subset=["strike", "bid", "ask"]
        )

        # Avoid absurdly wide spreads
        chain["mid"] = (
            chain["bid"] +
            chain["ask"]
        ) / 2

        chain = chain[
            chain["mid"] > 0
        ]

        if chain.empty:
            continue

        # ----------------------------------------------------
        # Strike selection
        #
        # Slightly OTM / near ATM.
        # ----------------------------------------------------

        if direction == "BULLISH":

            chain["distance"] = abs(
                chain["strike"] - price * 1.03
            )

        else:

            chain["distance"] = abs(
                chain["strike"] - price * 0.97
            )

        chain = chain.sort_values("distance")

        for _, row in chain.head(12).iterrows():

            strike = row["strike"]
            mid = row["mid"]

            tv = option_time_value(
                mid,
                price,
                strike,
                option_type
            )

            max_tv = price * ONE_PERCENT_RULE

            # Hughes rule
            if tv > max_tv:
                continue

            spread = row["ask"] - row["bid"]

            spread_pct = (
                spread / mid * 100
                if mid > 0
                else 999
            )

            if spread_pct > 25:
                continue

            candidates.append({
                "symbol": symbol,
                "expiration": exp,
                "dte": dte,
                "type": option_type,
                "strike": strike,
                "bid": row["bid"],
                "ask": row["ask"],
                "mid": mid,
                "intrinsic": (
                    max(price - strike, 0)
                    if option_type == "call"
                    else
                    max(strike - price, 0)
                ),
                "time_value": tv,
                "max_time_value": max_tv,
                "spread_pct": spread_pct
            })

    if not candidates:
        return None

    # Prefer lowest time value and reasonable DTE
    candidates.sort(
        key=lambda x: (
            x["time_value"] / price,
            abs(x["dte"] - 30),
            x["spread_pct"]
        )
    )

    return candidates[0]


# ------------------------------------------------------------
# OPTION TRADE PLAN
# ------------------------------------------------------------

def build_option_trade(stock, option):

    if option is None:
        return None

    price = stock["price"]
    direction = stock["direction"]

    entry = option["ask"]

    if entry <= 0:
        entry = option["mid"]

    # Option stop around 45% loss
    stop = entry * 0.55

    # Target based on stock setup
    target = entry * 1.50

    max_risk_dollars = entry * 100

    reward_dollars = (
        target - entry
    ) * 100

    rr = (
        reward_dollars /
        max_risk_dollars
        if max_risk_dollars > 0
        else 0
    )

    action = (
        "BUY CALL"
        if direction == "BULLISH"
        else
        "BUY PUT"
    )

    return {
        "action": action,
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk_per_contract": max_risk_dollars,
        "reward_per_contract": reward_dollars,
        "rr": rr
    }


# ------------------------------------------------------------
# STOCK SCANNER
# ------------------------------------------------------------

def run_stock_scan():

    results = []

    progress = st.progress(0)

    total = len(STOCK_UNIVERSE)

    for i, symbol in enumerate(STOCK_UNIVERSE):

        df = get_history(symbol)

        if df is None:
            progress.progress(
                min((i + 1) / total, 1.0)
            )
            continue

        if len(df) < 120:
            continue

        last_price = safe_float(
            df.iloc[-1]["close"]
        )

        # Price filter
        if (
            np.isnan(last_price)
            or last_price < MIN_PRICE
            or last_price > MAX_PRICE
        ):
            continue

        analysis = score_stock(df)

        if analysis is None:
            continue

        if analysis["score"] < MIN_SCORE:
            continue

        # Relative volume requirement
        rv = analysis["relative_volume"]

        if (
            not np.isnan(rv)
            and rv < MIN_REL_VOLUME
        ):
            continue

        results.append({
            "Ticker": symbol,
            "Price": analysis["price"],
            "Direction": analysis["direction"],
            "Score": analysis["score"],
            "Confidence": analysis["confidence"],
            "Setup": analysis["setup"],
            "Crossover": (
                analysis["crossover_date"].strftime("%Y-%m-%d")
                if analysis["crossover_date"] is not None
                else "—"
            ),
            "RV": analysis["relative_volume"],
            "RSI": analysis["rsi"],
            "Momentum %": analysis["momentum"],
            "Price Action": analysis["pattern"],
            "Support": analysis["support"],
            "Resistance": analysis["resistance"],
            "Stop": analysis["stop"],
            "Target": analysis["target"],
            "Risk %": analysis["risk_pct"],
            "Reasons": " • ".join(
                analysis["reasons"]
            )
        })

        progress.progress(
            min((i + 1) / total, 1.0)
        )

    progress.empty()

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        ["Score", "RV"],
        ascending=[False, False]
    )

    return result_df.reset_index(drop=True)


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Universe",
        f"{len(STOCK_UNIVERSE)} stocks"
    )

with col2:
    st.metric(
        "Price Floor",
        f"${MIN_PRICE:.0f}"
    )

with col3:
    st.metric(
        "Min Relative Volume",
        f"{MIN_REL_VOLUME:.1f}x"
    )


st.subheader("Phase 1 — Golden Technical Scan")

st.write(
    "Finds individual stocks with trend, momentum, "
    "EMA structure, recent 50/100 crossover, "
    "relative volume and price-action confluence."
)

run_scan = st.button(
    "🏆 RUN GOLDEN SCAN",
    type="primary",
    use_container_width=True
)


if run_scan:

    with st.spinner(
        "Scanning the Golden universe..."
    ):

        results = run_stock_scan()

        st.session_state.scan_results = results

        # Reset options
        st.session_state.option_results = {}


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

results = st.session_state.scan_results

if results is not None:

    st.divider()

    st.subheader(
        f"Phase 1 Results — {len(results)} Candidates"
    )

    if results.empty:

        st.warning(
            "No stocks currently meet the Golden Scanner filters."
        )

    else:

        display_cols = [
            "Ticker",
            "Price",
            "Direction",
            "Score",
            "Confidence",
            "Setup",
            "Crossover",
            "RV",
            "RSI",
            "Momentum %",
            "Price Action",
            "Stop",
            "Target",
            "Risk %"
        ]

        st.dataframe(
            results[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # SELECT STOCK
        # ----------------------------------------------------

        tickers = results["Ticker"].tolist()

        selected = st.selectbox(
            "Select a Golden candidate for Phase 2:",
            tickers
        )

        selected_row = results[
            results["Ticker"] == selected
        ].iloc[0]

        st.subheader(
            f"🎯 {selected} — Golden Trade Candidate"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Price",
                f"${selected_row['Price']:.2f}"
            )

        with c2:
            st.metric(
                "Direction",
                selected_row["Direction"]
            )

        with c3:
            st.metric(
                "Score",
                f"{selected_row['Score']}"
            )

        with c4:
            st.metric(
                "Setup",
                selected_row["Setup"]
            )

        st.info(
            selected_row["Reasons"]
        )

        st.write(
            f"**Price Action:** "
            f"{selected_row['Price Action']}"
        )

        st.write(
            f"**Support:** ${selected_row['Support']:.2f}  |  "
            f"**Resistance:** ${selected_row['Resistance']:.2f}"
        )

        st.write(
            f"**Technical Stop:** ${selected_row['Stop']:.2f}  |  "
            f"**Technical Target:** ${selected_row['Target']:.2f}"
        )

        # ----------------------------------------------------
        # PHASE 2
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Phase 2 — Hughes Options Engine"
        )

        st.write(
            "Searches 14–45 DTE options and applies the "
            "Chuck Hughes 1% rule to option time value."
        )

        if st.button(
            f"🔎 FIND OPTION TRADE FOR {selected}",
            use_container_width=True
        ):

            stock_dict = {
                "price": selected_row["Price"],
                "direction": selected_row["Direction"],
                "score": selected_row["Score"]
            }

            with st.spinner(
                f"Searching {selected} option chain..."
            ):

                option = select_option(
                    selected,
                    stock_dict
                )

            st.session_state.option_results[
                selected
            ] = option


        option = st.session_state.option_results.get(
            selected
        )


        if option is not None:

            trade = build_option_trade(
                {
                    "price": selected_row["Price"],
                    "direction": selected_row["Direction"]
                },
                option
            )

            st.success(
                "HUGHES 1% RULE PASSED"
            )

            st.write(
                f"### {trade['action']}"
            )

            o1, o2, o3, o4 = st.columns(4)

            with o1:
                st.metric(
                    "Expiration",
                    option["expiration"]
                )

            with o2:
                st.metric(
                    "DTE",
                    option["dte"]
                )

            with o3:
                st.metric(
                    "Strike",
                    f"${option['strike']:.2f}"
                )

            with o4:
                st.metric(
                    "Entry",
                    f"${trade['entry']:.2f}"
                )

            st.write(
                f"**Bid:** ${option['bid']:.2f}  |  "
                f"**Ask:** ${option['ask']:.2f}  |  "
                f"**Mid:** ${option['mid']:.2f}"
            )

            st.write(
                f"**Intrinsic Value:** "
                f"${option['intrinsic']:.2f}"
            )

            st.write(
                f"**Time Value:** "
                f"${option['time_value']:.2f}"
            )

            st.write(
                f"**Maximum 1% Time Value:** "
                f"${option['max_time_value']:.2f}"
            )

            st.write(
                f"**Bid/Ask Spread:** "
                f"{option['spread_pct']:.1f}%"
            )

            st.divider()

            t1, t2, t3, t4 = st.columns(4)

            with t1:
                st.metric(
                    "Entry",
                    f"${trade['entry']:.2f}"
                )

            with t2:
                st.metric(
                    "Stop",
                    f"${trade['stop']:.2f}"
                )

            with t3:
                st.metric(
                    "Target",
                    f"${trade['target']:.2f}"
                )

            with t4:
                st.metric(
                    "R:R",
                    f"{trade['rr']:.2f}"
                )

            st.success(
                f"Potential trade: "
                f"Buy-to-Open the {selected} "
                f"{option['expiration']} "
                f"{option['strike']:.0f} "
                f"{option['type'].capitalize()} "
                f"at ${trade['entry']:.2f} or less."
            )

            st.warning(
                "This is a scanner-generated candidate, "
                "not a guarantee of profit. Verify the live "
                "option chain, liquidity and price before trading."
            )

        elif option is None and selected in st.session_state.option_results:

            st.warning(
                "No option contract passed the Hughes 1% "
                "time-value and liquidity filters for this stock."
            )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "Golden Scanner • Technical Phase 1 + Hughes Options Phase 2"
)

st.caption(
    "Stocks only • $10+ • Optionable universe • "
    "14–45 DTE • 1% time-value filter"
)
