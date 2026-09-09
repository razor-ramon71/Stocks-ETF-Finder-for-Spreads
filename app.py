import os
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st

from datetime import datetime, timedelta 


# ============================================================
# GOLDEN SCANNER
# ============================================================
# Golden Swing Hunter
#
# PHASE 1:
#   Large dynamic U.S. stock universe
#   Stocks ONLY
#   $10+
#   Technical Golden setup
#   Bullish setups only
#
# PHASE 2:
#   ONLY stocks selected by the user
#   CALLS ONLY
#   14-45 DTE
#   Chuck Hughes 1% time-value rule
# ============================================================


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Golden Scanner",
    page_icon="🏆",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_PRICE = 10.00
MAX_PRICE = 250.00

MIN_RELATIVE_VOLUME = 1.20

EMA_FAST = 20
EMA_50 = 50
EMA_100 = 100

RSI_LENGTH = 14
ATR_LENGTH = 14

CROSSOVER_LOOKBACK = 30

MIN_SCORE = 60

OPTION_DTE_MIN = 14
OPTION_DTE_MAX = 45

HUGHES_TIME_VALUE_LIMIT = 0.01

MAX_OPTION_SPREAD_PERCENT = 25

HISTORY_DAYS = 260


# ============================================================
# TRADIER
# ============================================================

TRADIER_TOKEN = st.secrets.get(
    "TRADIER_TOKEN",
    os.getenv("TRADIER_TOKEN", "")
)

TRADIER_BASE = "https://api.tradier.com/v1"

HEADERS = {
    "Authorization": f"Bearer {TRADIER_TOKEN}",
    "Accept": "application/json"
}


# ============================================================
# SESSION STATE
# ============================================================

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = []

if "option_results" not in st.session_state:
    st.session_state.option_results = None


# ============================================================
# HEADER
# ============================================================

st.title("🏆 Golden Scanner")

st.caption(
    "Golden Swing Hunter • Stocks Only • Bullish Call Setups"
)


if not TRADIER_TOKEN:
    st.error(
        "TRADIER_TOKEN was not found. "
        "Add it to Streamlit Secrets."
    )
    st.stop()


st.success("Tradier connection configured.")


# ============================================================
# HTTP HELPER
# ============================================================

def get_json(url, params=None, timeout=20):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=timeout
        )

        if response.status_code != 200:
            return None

        return response.json()

    except Exception:
        return None


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(value, default=np.nan):

    try:

        if value is None:
            return default

        return float(value)

    except Exception:

        return default


# ============================================================
# GOLDEN UNIVERSE — SECTOR BALANCED
# ============================================================
#
# Approximately 400-500 liquid, option-friendly U.S. stocks.
# All 11 major market sectors represented.
#
# ETFs excluded.
# Mag 7 excluded.
# Price filter remains $10-$250.
#
# Phase 1 scans this universe.
# Phase 2 ONLY analyzes stocks selected by the user.
# ============================================================

GOLDEN_SECTOR_UNIVERSE = {

    "Technology": [
        "ACN","ADBE","ADI","ADSK","AMAT","AMD","ANET","APH",
        "AVGO","CDNS","CRM","CSCO","CTSH","FTNT","IBM","INTC",
        "INTU","KLAC","LRCX","MCHP","MU","NOW","NTAP","NXPI",
        "ON","ORCL","PANW","QCOM","SNPS","STX","TEL","TER",
        "TXN","VRSN","WDC","WDAY","ZS","DELL","HPQ","HPE"
    ],

    "Health Care": [
        "ABT","ALGN","AMGN","BAX","BDX","BIIB","BMY","BSX",
        "CAH","CI","CNC","COO","CVS","DHR","DXCM","ELV",
        "EW","GILD","HCA","HOLX","HSIC","HUM","IDXX","ILMN",
        "INCY","ISRG","JNJ","LH","LLY","MRK","MTD","PFE",
        "REGN","RMD","SYK","TMO","UHS","UNH","VRTX","ZBH"
    ],

    "Financials": [
        "AFL","AIG","AIZ","AJG","ALL","AMP","AON","APO",
        "AXP","BAC","BEN","BK","BKNG","BLK","BX","C","CB",
        "CBOE","CME","COF","DFS","FITB","GS","HBAN","ICE",
        "JPM","KEY","KKR","MA","MET","MKTX","MS","MSCI",
        "MTB","NDAQ","PNC","PRU","SCHW","STT","TFC","USB"
    ],

    "Industrials": [
        "AOS","CAT","CHRW","CMI","CARR","CSX","CTAS","DAL",
        "DE","DOV","EFX","EMR","ETN","EXPD","FAST","FDX",
        "GD","GE","GWW","HON","HUBB","IEX","ITW","JCI",
        "LHX","LMT","MAS","MMM","NOC","NSC","ODFL","PCAR",
        "PH","PNR","ROK","RTX","SWK","TT","UAL","UPS"
    ],

    "Consumer Discretionary": [
        "AAP","AMZN","APTV","BBY","BKNG","BWA","CCL","CMG",
        "CROX","CVNA","DHI","DKS","DRI","EBAY","EXPE","F",
        "GM","GPC","HD","LOW","LULU","MAR","MCD","MGM",
        "NKE","ORLY","PHM","POOL","RCL","ROST","SBUX","TJX",
        "TGT","TSLA","ULTA","WHR","WYNN","YUM","CVCO","LEN"
    ],

    "Communication Services": [
        "CHTR","CMCSA","DIS","EA","FOXA","FOX","GOOG","GOOGL",
        "IPG","LYV","META","MTCH","NFLX","NWS","NWSA","OMC",
        "PARA","PINS","ROKU","T","TMUS","TTWO","TWLO","VZ",
        "WBD","WPP","ZI","ZG","Z","SPOT"
    ],

    "Energy": [
        "APA","BKR","COP","CTRA","CVX","DVN","EOG","EQT",
        "FANG","HAL","HES","KMI","MPC","MRO","MUR","OKE",
        "OXY","PSX","PXD","SLB","TRGP","VLO","WMB","XOM",
        "CHRD","CNX","CIVI","EOG","OVV","TPL"
    ],

    "Consumer Staples": [
        "ADM","BF.B","CAG","CHD","CL","CLX","COST","CPB",
        "EL","GIS","HSY","HRL","K","KDP","KHC","KO",
        "KR","KVUE","LW","MDLZ","MKC","MO","PEP","PG",
        "PM","SJM","STZ","SYY","TAP","TSN"
    ],

    "Utilities": [
        "AES","AEE","AEP","AWK","CEG","CMS","CNP","D",
        "DTE","DUK","ED","EIX","ES","ETR","EVRG","EXC",
        "FE","LNT","NEE","NI","NRG","PCG","PEG","PNW",
        "SO","SRE","VST","WEC","XEL","ATO"
    ],

    "Materials": [
        "AA","ALB","APD","AVY","BALL","CF","CLF","DD",
        "ECL","EMN","FCX","FMC","IFF","IP","LIN","LYB",
        "MLM","MOS","NEM","NUE","PKG","PPG","SEE","SHW",
        "STLD","VMC","WRK","CTVA","CE","RPM"
    ],

    "Real Estate": [
        "AMT","ARE","AVB","BXP","CBRE","CCI","CPT","CSGP",
        "DLR","DOC","EQC","EQR","ESS","EXR","FRT","HST",
        "IRM","KIM","MAA","O","PLD","PSA","REG","RHP",
        "RITM","SBRA","SUI","VICI","VNO","WELL"
    ]
}


@st.cache_data(ttl=86400)
def get_dynamic_stock_universe():

    symbols = []

    for sector_symbols in GOLDEN_SECTOR_UNIVERSE.values():
        symbols.extend(sector_symbols)

    # Remove duplicates while preserving order
    symbols = list(dict.fromkeys(symbols))

    # Exclude Mag 7
    mag7 = {
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "TSLA"
    }

    symbols = [
        symbol
        for symbol in symbols
        if symbol not in mag7
    ]

    return symbols


@st.cache_data(ttl=900)
def filter_universe_by_price(symbols):

    filtered = []

    batch_size = 100

    for i in range(0, len(symbols), batch_size):

        batch = symbols[
            i:i + batch_size
        ]

        try:

            data = get_json(
                f"{TRADIER_BASE}/markets/quotes",
                params={
                    "symbols": ",".join(batch)
                },
                timeout=15
            )

            if not data:
                continue

            quotes = (
                data
                .get("quotes", {})
                .get("quote", [])
            )

            if isinstance(quotes, dict):
                quotes = [quotes]

            for quote in quotes:

                symbol = quote.get("symbol")
                price = quote.get("last")

                if (
                    symbol is None
                    or price is None
                ):
                    continue

                try:
                    price = float(price)
                except (
                    TypeError,
                    ValueError
                ):
                    continue

                if MIN_PRICE <= price <= MAX_PRICE:
                    filtered.append(symbol)

        except Exception:
            continue

    return filtered
# ============================================================
# HISTORICAL DATA
# ============================================================

@st.cache_data(ttl=900)
def get_history(symbol):

    end = datetime.now()
    start = end - timedelta(
        days=HISTORY_DAYS
    )

    data = get_json(
        f"{TRADIER_BASE}/markets/history",
        {
            "symbol": symbol,
            "interval": "daily",
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d")
        },
        timeout=15
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

        df["date"] = pd.to_datetime(
            df["date"]
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna(
            subset=["close"]
        )

        df = df.sort_values(
            "date"
        )

        df = df.reset_index(
            drop=True
        )

        return df

    except Exception:
        return None


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMAs
    # --------------------------------------------------------

    df["ema20"] = (
        df["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(
            span=EMA_50,
            adjust=False
        )
        .mean()
    )

    df["ema100"] = (
        df["close"]
        .ewm(
            span=EMA_100,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    average_gain = (
        gain
        .ewm(
            alpha=1 / RSI_LENGTH,
            adjust=False
        )
        .mean()
    )

    average_loss = (
        loss
        .ewm(
            alpha=1 / RSI_LENGTH,
            adjust=False
        )
        .mean()
    )

    rs = (
        average_gain /
        average_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = (
        df["close"].shift(1)
    )

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = abs(
        df["high"] -
        previous_close
    )

    tr3 = abs(
        df["low"] -
        previous_close
    )

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        true_range
        .rolling(
            ATR_LENGTH
        )
        .mean()
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["average_volume"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["relative_volume"] = (
        df["volume"] /
        df["average_volume"]
        .replace(0, np.nan)
    )

    # --------------------------------------------------------
    # KELTNER
    # --------------------------------------------------------

    df["keltner_mid"] = (
        df["ema20"]
    )

    df["keltner_upper"] = (
        df["ema20"] +
        2 * df["atr"]
    )

    df["keltner_lower"] = (
        df["ema20"] -
        2 * df["atr"]
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    df["momentum_10"] = (
        df["close"]
        .pct_change(10)
        * 100
    )

    # --------------------------------------------------------
    # EMA CROSSOVER
    # --------------------------------------------------------

    df["bull_cross"] = (
        (df["ema50"] > df["ema100"]) &
        (
            df["ema50"].shift(1) <=
            df["ema100"].shift(1)
        )
    )

    # --------------------------------------------------------
    # Sessions since most recent bullish crossover
    # --------------------------------------------------------

    cross_positions = np.where(
        df["bull_cross"].values
    )[0]

    if len(cross_positions) > 0:

        last_cross_position = (
            cross_positions[-1]
        )

        sessions_since_cross = (
            len(df) -
            1 -
            last_cross_position
        )

        crossover_date = (
            df.iloc[
                last_cross_position
            ]["date"]
        )

    else:

        sessions_since_cross = None
        crossover_date = None

    df.attrs[
        "sessions_since_cross"
    ] = sessions_since_cross

    df.attrs[
        "crossover_date"
    ] = crossover_date

    return df


# ============================================================
# CANDLE PATTERN
# ============================================================

def get_candle_pattern(df):

    if len(df) < 3:
        return "None"

    current = df.iloc[-1]
    previous = df.iloc[-2]

    body = abs(
        current["close"] -
        current["open"]
    )

    total_range = (
        current["high"] -
        current["low"]
    )

    if total_range <= 0:
        return "None"

    upper_wick = (
        current["high"] -
        max(
            current["open"],
            current["close"]
        )
    )

    lower_wick = (
        min(
            current["open"],
            current["close"]
        ) -
        current["low"]
    )

    # --------------------------------------------------------
    # Bullish engulfing
    # --------------------------------------------------------

    if (
        previous["close"] <
        previous["open"]
        and
        current["close"] >
        current["open"]
        and
        current["close"] >
        previous["open"]
        and
        current["open"] <
        previous["close"]
    ):
        return "Bullish Engulfing"

    # --------------------------------------------------------
    # Hammer
    # --------------------------------------------------------

    if (
        lower_wick >= body * 2
        and
        upper_wick <= body
    ):
        return "Hammer"

    # --------------------------------------------------------
    # Strong bullish candle
    # --------------------------------------------------------

    if (
        current["close"] >
        current["open"]
        and
        body / total_range >= 0.65
    ):
        return "Strong Bull Candle"

    # --------------------------------------------------------
    # Bullish inside recovery
    # --------------------------------------------------------

    if (
        current["close"] >
        current["open"]
        and
        current["close"] >
        previous["close"]
    ):
        return "Bullish Recovery"

    return "Neutral"


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def calculate_support_resistance(df):

    recent = df.tail(20)

    support = safe_float(
        recent["low"].min()
    )

    resistance = safe_float(
        recent["high"].max()
    )

    return support, resistance


# ============================================================
# GOLDEN TECHNICAL ANALYSIS
# ============================================================

def analyze_stock(df):

    if df is None:
        return None

    if len(df) < 120:
        return None

    df = calculate_indicators(df)

    last = df.iloc[-1]

    price = safe_float(
        last["close"]
    )

    if np.isnan(price):
        return None

    # --------------------------------------------------------
    # PRICE FILTER
    # --------------------------------------------------------

    if price < MIN_PRICE:
        return None

    if price > MAX_PRICE:
        return None

    # --------------------------------------------------------
    # REQUIRED RELATIVE VOLUME
    # --------------------------------------------------------

    relative_volume = safe_float(
        last["relative_volume"]
    )

    if (
        np.isnan(relative_volume)
        or
        relative_volume <
        MIN_RELATIVE_VOLUME
    ):
        return None

    # --------------------------------------------------------
    # BULLISH SCORE
    # --------------------------------------------------------

    score = 0

    reasons = []

    # --------------------------------------------------------
    # 20 EMA > 50 EMA
    # --------------------------------------------------------

    if (
        last["ema20"] >
        last["ema50"]
    ):

        score += 10

        reasons.append(
            "20 EMA > 50 EMA"
        )

    # --------------------------------------------------------
    # 50 EMA > 100 EMA
    # --------------------------------------------------------

    if (
        last["ema50"] >
        last["ema100"]
    ):

        score += 15

        reasons.append(
            "50 EMA > 100 EMA"
        )

    else:

        # We only want bullish setups
        return None

    # --------------------------------------------------------
    # RECENT 50/100 CROSS
    # --------------------------------------------------------

    sessions_since_cross = (
        df.attrs[
            "sessions_since_cross"
        ]
    )

    crossover_date = (
        df.attrs[
            "crossover_date"
        ]
    )

    if (
        sessions_since_cross is not None
        and
        sessions_since_cross <=
        CROSSOVER_LOOKBACK
    ):

        score += 15

        reasons.append(
            "50/100 bullish crossover "
            f"within {CROSSOVER_LOOKBACK} sessions"
        )

    # --------------------------------------------------------
    # PRICE ABOVE 20 EMA
    # --------------------------------------------------------

    if (
        price >
        last["ema20"]
    ):

        score += 8

        reasons.append(
            "Price above 20 EMA"
        )

    # --------------------------------------------------------
    # PRICE ABOVE 50 EMA
    # --------------------------------------------------------

    if (
        price >
        last["ema50"]
    ):

        score += 5

        reasons.append(
            "Price above 50 EMA"
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = safe_float(
        last["rsi"]
    )

    if not np.isnan(rsi):

        if 50 <= rsi <= 70:

            score += 10

            reasons.append(
                f"RSI bullish ({rsi:.1f})"
            )

        elif 45 <= rsi < 50:

            score += 4

            reasons.append(
                f"RSI improving ({rsi:.1f})"
            )

        elif rsi > 70:

            score += 4

            reasons.append(
                f"RSI strong ({rsi:.1f})"
            )

    # --------------------------------------------------------
    # RELATIVE VOLUME
    # --------------------------------------------------------

    if relative_volume >= 1.5:

        score += 10

        reasons.append(
            f"Strong relative volume "
            f"({relative_volume:.2f}x)"
        )

    elif relative_volume >= 1.2:

        score += 6

        reasons.append(
            f"Relative volume "
            f"({relative_volume:.2f}x)"
        )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = safe_float(
        last["momentum_10"]
    )

    if not np.isnan(momentum):

        if momentum >= 5:

            score += 10

            reasons.append(
                f"10-day momentum +{momentum:.1f}%"
            )

        elif momentum >= 2:

            score += 6

            reasons.append(
                f"10-day momentum +{momentum:.1f}%"
            )

        elif momentum > 0:

            score += 3

            reasons.append(
                f"Positive momentum +{momentum:.1f}%"
            )

    # --------------------------------------------------------
    # KELTNER
    # --------------------------------------------------------

    if (
        price >
        last["keltner_mid"]
    ):

        score += 5

        reasons.append(
            "Keltner bullish"
        )

    # --------------------------------------------------------
    # PRICE ACTION
    # --------------------------------------------------------

    pattern = get_candle_pattern(
        df
    )

    if pattern in [
        "Bullish Engulfing",
        "Hammer",
        "Strong Bull Candle"
    ]:

        score += 7

        reasons.append(
            pattern
        )

    elif pattern == "Bullish Recovery":

        score += 4

        reasons.append(
            pattern
        )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    support, resistance = (
        calculate_support_resistance(
            df
        )
    )

    if (
        not np.isnan(resistance)
        and
        resistance > price
    ):

        distance_to_resistance = (
            (
                resistance -
                price
            )
            /
            price
            *
            100
        )

    else:

        distance_to_resistance = 0

    # Near breakout resistance
    if (
        0 <=
        distance_to_resistance <=
        5
    ):

        score += 5

        reasons.append(
            "Near breakout resistance"
        )

    # --------------------------------------------------------
    # CONFLUENCE COUNT
    # --------------------------------------------------------

    confluence_count = 0

    confluence_keywords = [
        "EMA",
        "crossover",
        "RSI",
        "volume",
        "momentum",
        "Keltner",
        "Bullish",
        "Hammer",
        "resistance"
    ]

    for reason in reasons:

        if any(
            keyword.lower()
            in reason.lower()
            for keyword in confluence_keywords
        ):

            confluence_count += 1

    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------

    if (
        sessions_since_cross is not None
        and
        sessions_since_cross <=
        CROSSOVER_LOOKBACK
        and
        price >
        last["ema20"]
    ):

        setup = (
            "Bull Trend Pullback"
        )

    elif (
        price >=
        resistance * 0.995
    ):

        setup = (
            "Breakout Setup"
        )

    elif (
        momentum > 3
        and
        price >
        last["ema20"]
    ):

        setup = (
            "Momentum Continuation"
        )

    elif (
        pattern in [
            "Bullish Engulfing",
            "Hammer"
        ]
    ):

        setup = (
            "Bullish Reversal"
        )

    else:

        setup = (
            "Bullish Trend"
        )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if score >= 85:
        confidence = "A+"

    elif score >= 75:
        confidence = "A"

    elif score >= 65:
        confidence = "B+"

    elif score >= 60:
        confidence = "B"

    else:
        confidence = "PASS"

    # --------------------------------------------------------
    # TECHNICAL STOP / TARGET
    # --------------------------------------------------------

    atr = safe_float(
        last["atr"]
    )

    if (
        np.isnan(atr)
        or
        atr <= 0
    ):

        atr = price * 0.03

    stop = (
        price -
        1.20 * atr
    )

    risk = (
        price -
        stop
    )

    target = (
        price +
        1.50 * risk
    )

    risk_percent = (
        risk /
        price *
        100
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {

        "Price": price,

        "Direction": "BULLISH",

        "Score": int(score),

        "Confidence": confidence,

        "Setup": setup,

        "Crossover Date": (
            crossover_date
            if crossover_date is not None
            else None
        ),

        "Sessions Since Cross": (
            sessions_since_cross
            if sessions_since_cross is not None
            else "—"
        ),

        "Relative Volume": relative_volume,

        "RSI": rsi,

        "Momentum %": momentum,

        "Price Action": pattern,

        "Confluence": confluence_count,

        "Support": support,

        "Resistance": resistance,

        "Stop": stop,

        "Target": target,

        "Risk %": risk_percent,

        "Reasons": " • ".join(
            reasons
        )
    }


# ============================================================
# OPTION EXPIRATIONS
# ============================================================

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

        dates = (
            data[
                "expirations"
            ]["date"]
        )

        if isinstance(
            dates,
            str
        ):

            dates = [dates]

        return dates

    except Exception:

        return []


# ============================================================
# OPTION CHAIN
# ============================================================

@st.cache_data(ttl=300)
def get_option_chain(
    symbol,
    expiration
):

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

        options = (
            data[
                "options"
            ]["option"]
        )

        if not isinstance(
            options,
            list
        ):

            options = [options]

        df = pd.DataFrame(
            options
        )

        return df

    except Exception:

        return None


# ============================================================
# CALL INTRINSIC VALUE
# ============================================================

def call_intrinsic_value(
    underlying_price,
    strike
):

    return max(
        underlying_price -
        strike,
        0
    )


# ============================================================
# HUGHES 1% RULE
# ============================================================

def passes_hughes_rule(
    underlying_price,
    option_mid,
    strike
):

    intrinsic = (
        call_intrinsic_value(
            underlying_price,
            strike
        )
    )

    time_value = max(
        option_mid -
        intrinsic,
        0
    )

    maximum_time_value = (
        underlying_price *
        HUGHES_TIME_VALUE_LIMIT
    )

    return (
        time_value <=
        maximum_time_value
    )


# ============================================================
# SELECT CALL
# ============================================================

def find_best_call(
    symbol,
    stock_data
):

    underlying_price = (
        stock_data["Price"]
    )

    expirations = (
        get_option_expirations(
            symbol
        )
    )

    if not expirations:
        return None

    today = datetime.now().date()

    candidates = []

    for expiration in expirations:

        try:

            expiration_date = (
                datetime.strptime(
                    expiration,
                    "%Y-%m-%d"
                ).date()
            )

        except Exception:

            continue

        dte = (
            expiration_date -
            today
        ).days

        if (
            dte <
            OPTION_DTE_MIN
        ):

            continue

        if (
            dte >
            OPTION_DTE_MAX
        ):

            continue

        chain = (
            get_option_chain(
                symbol,
                expiration
            )
        )

        if (
            chain is None
            or
            chain.empty
        ):

            continue

        # ----------------------------------------------------
        # CALLS ONLY
        # ----------------------------------------------------

        chain = chain[
            chain["option_type"]
            .astype(str)
            .str.lower()
            ==
            "call"
        ].copy()

        if chain.empty:
            continue

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        for column in [
            "strike",
            "bid",
            "ask"
        ]:

            chain[column] = (
                pd.to_numeric(
                    chain[column],
                    errors="coerce"
                )
            )

        chain = chain.dropna(
            subset=[
                "strike",
                "bid",
                "ask"
            ]
        )

        if chain.empty:
            continue

        # ----------------------------------------------------
        # Valid markets only
        # ----------------------------------------------------

        chain = chain[
            (chain["bid"] > 0) &
            (chain["ask"] > 0) &
            (chain["ask"] >= chain["bid"])
        ]

        if chain.empty:
            continue

        chain["mid"] = (
            chain["bid"] +
            chain["ask"]
        ) / 2

        # ----------------------------------------------------
        # We focus on ATM to modestly OTM calls.
        #
        # We do NOT want the algorithm choosing ridiculous
        # deep-ITM calls like the HD example.
        # ----------------------------------------------------

        lower_strike = (
            underlying_price *
            0.98
        )

        upper_strike = (
            underlying_price *
            1.08
        )

        chain = chain[
            (chain["strike"] >= lower_strike) &
            (chain["strike"] <= upper_strike)
        ]

        if chain.empty:
            continue

        # ----------------------------------------------------
        # Evaluate contracts
        # ----------------------------------------------------

        for _, option in chain.iterrows():

            strike = safe_float(
                option["strike"]
            )

            bid = safe_float(
                option["bid"]
            )

            ask = safe_float(
                option["ask"]
            )

            mid = safe_float(
                option["mid"]
            )

            if (
                np.isnan(strike)
                or
                np.isnan(mid)
                or
                mid <= 0
            ):

                continue

            spread = (
                ask -
                bid
            )

            spread_percent = (
                spread /
                mid *
                100
            )

            if (
                spread_percent >
                MAX_OPTION_SPREAD_PERCENT
            ):

                continue

            intrinsic = (
                call_intrinsic_value(
                    underlying_price,
                    strike
                )
            )

            time_value = max(
                mid -
                intrinsic,
                0
            )

            maximum_time_value = (
                underlying_price *
                HUGHES_TIME_VALUE_LIMIT
            )

            # ------------------------------------------------
            # Hughes rule
            # ------------------------------------------------

            if (
                time_value >
                maximum_time_value
            ):

                continue

            # ------------------------------------------------
            # Prefer ATM / slightly OTM
            # ------------------------------------------------

            strike_distance = (
                abs(
                    strike -
                    underlying_price
                )
                /
                underlying_price
            )

            candidates.append({

                "Expiration": expiration,

                "DTE": dte,

                "Type": "CALL",

                "Strike": strike,

                "Bid": bid,

                "Ask": ask,

                "Mid": mid,

                "Intrinsic": intrinsic,

                "Time Value": time_value,

                "Max Time Value": (
                    maximum_time_value
                ),

                "Spread %": spread_percent,

                "Strike Distance": (
                    strike_distance
                )
            })

    if not candidates:
        return None

    # --------------------------------------------------------
    # Rank contracts
    #
    # First priority:
    #   Hughes time value
    #
    # Second:
    #   near ATM
    #
    # Third:
    #   around 30 DTE
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            x["Time Value"] /
            max(
                x["Max Time Value"],
                0.0001
            ),

            x["Strike Distance"],

            abs(
                x["DTE"] -
                30
            ),

            x["Spread %"]
        )
    )

    return candidates[0]


# ============================================================
# OPTION TRADE PLAN
# ============================================================

def build_call_trade(
    stock,
    option
):

    if option is None:
        return None

    entry = (
        option["Ask"]
    )

    if (
        np.isnan(entry)
        or
        entry <= 0
    ):

        entry = (
            option["Mid"]
        )

    # --------------------------------------------------------
    # Defined option stop
    # --------------------------------------------------------

    stop = (
        entry *
        0.55
    )

    # --------------------------------------------------------
    # 1.5R option target
    # --------------------------------------------------------

    risk = (
        entry -
        stop
    )

    target = (
        entry +
        1.50 * risk
    )

    risk_dollars = (
        risk * 100
    )

    reward_dollars = (
        target -
        entry
    ) * 100

    rr = (
        reward_dollars /
        risk_dollars
        if risk_dollars > 0
        else 0
    )

    return {

        "Entry": entry,

        "Stop": stop,

        "Target": target,

        "Risk / Contract": (
            risk_dollars
        ),

        "Reward / Contract": (
            reward_dollars
        ),

        "R:R": rr
    }


# ============================================================
# PHASE 1 SCANNER
# ============================================================

def run_golden_scan():

    universe = get_dynamic_stock_universe()

    universe = filter_universe_by_price(universe)

    if not universe:
        return pd.DataFrame(), 0

    results = []

    progress = st.progress(
        0
    )

    status = st.empty()

    total = len(
        universe
    )

    for index, symbol in enumerate(
        universe
    ):

        status.write(
            f"Scanning {index + 1:,} "
            f"of {total:,}: {symbol}"
        )

        df = get_history(
            symbol
        )

        if df is None:

            progress.progress(
                min(
                    (index + 1) /
                    total,
                    1.0
                )
            )

            continue

        analysis = (
            analyze_stock(
                df
            )
        )

        if analysis is None:

            progress.progress(
                min(
                    (index + 1) /
                    total,
                    1.0
                )
            )

            continue

        # ----------------------------------------------------
        # Minimum Golden score
        # ----------------------------------------------------

        if (
            analysis["Score"] <
            MIN_SCORE
        ):

            progress.progress(
                min(
                    (index + 1) /
                    total,
                    1.0
                )
            )

            continue

        results.append({

            "Ticker": symbol,

            "Price": analysis["Price"],

            "Direction": (
                analysis["Direction"]
            ),

            "Score": (
                analysis["Score"]
            ),

            "Confidence": (
                analysis["Confidence"]
            ),

            "Setup": (
                analysis["Setup"]
            ),

            "Crossover Date": (
                analysis["Crossover Date"]
            ),

            "Sessions Since Cross": (
                analysis[
                    "Sessions Since Cross"
                ]
            ),

            "Relative Volume": (
                analysis[
                    "Relative Volume"
                ]
            ),

            "RSI": (
                analysis["RSI"]
            ),

            "Momentum %": (
                analysis["Momentum %"]
            ),

            "Price Action": (
                analysis["Price Action"]
            ),

            "Confluence": (
                analysis["Confluence"]
            ),

            "Support": (
                analysis["Support"]
            ),

            "Resistance": (
                analysis["Resistance"]
            ),

            "Stop": (
                analysis["Stop"]
            ),

            "Target": (
                analysis["Target"]
            ),

            "Risk %": (
                analysis["Risk %"]
            ),

            "Reasons": (
                analysis["Reasons"]
            )
        })

        progress.progress(
            min(
                (index + 1) /
                total,
                1.0
            )
        )

    status.empty()
    progress.empty()

    if not results:

        return (
            pd.DataFrame(),
            len(universe)
        )

    result_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "Score",
                "Relative Volume",
                "Confluence"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        result_df,
        len(universe)
    )


# ============================================================
# UI — UNIVERSE
# ============================================================

st.divider()

universe = get_dynamic_stock_universe()

universe = filter_universe_by_price(universe)
u1, u2, u3, u4 = st.columns(4)

with u1:

    st.metric(
        "Dynamic Universe",
        f"{len(universe):,}"
    )

with u2:

    st.metric(
        "Price",
        "$10+"
    )

with u3:

    st.metric(
        "Min Rel. Volume",
        "1.20x"
    )

with u4:

    st.metric(
        "Option Strategy",
        "CALLS ONLY"
    )


st.info(
    "The universe is dynamically built from U.S. "
    "exchange-listed securities with ETFs removed. "
    "Phase 1 finds bullish technical candidates first; "
    "Phase 2 verifies the option chain only for stocks "
    "you select."
)


# ============================================================
# PHASE 1
# ============================================================

st.header(
    "Phase 1 — Golden Technical Scan"
)

st.write(
    "Find the strongest bullish swing candidates "
    "using EMA structure, recent 50/100 crossover, "
    "relative volume, momentum, Keltner position, "
    "price action and support/resistance."
)


if st.button(
    "🏆 RUN GOLDEN SCAN",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Scanning the expanded stock universe..."
    ):

        results, universe_count = (
            run_golden_scan()
        )

        st.session_state.scan_results = (
            results
        )

        st.session_state.selected_tickers = []

        st.session_state.option_results = None


# ============================================================
# DISPLAY PHASE 1
# ============================================================

results = (
    st.session_state.scan_results
)


if results is not None:

    st.divider()

    if results.empty:

        st.warning(
            "No bullish Golden candidates passed "
            "the current filters."
        )

    else:

        st.subheader(
            f"🏆 {len(results)} Golden Candidates"
        )

        st.success(
            "Select the stocks you want to investigate "
            "with the CALL options engine."
        )

        # ----------------------------------------------------
        # Selection UI
        # ----------------------------------------------------

        st.write(
            "### ☑️ Select candidates for Phase 2"
        )

        selected = []

        for index, row in results.iterrows():

            ticker = row["Ticker"]

            label = (
                f"{ticker}  |  "
                f"Score {row['Score']}  |  "
                f"{row['Setup']}  |  "
                f"${row['Price']:.2f}"
            )

            checked = st.checkbox(
                label,
                key=f"select_{ticker}"
            )

            if checked:

                selected.append(
                    ticker
                )

        st.session_state.selected_tickers = (
            selected
        )

        # ----------------------------------------------------
        # Selected count
        # ----------------------------------------------------

        st.info(
            f"{len(selected)} stock(s) selected "
            "for Phase 2."
        )

        # ----------------------------------------------------
        # Main technical table
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Golden Technical Candidates"
        )

        display_columns = [

            "Ticker",

            "Price",

            "Direction",

            "Score",

            "Confidence",

            "Setup",

            "Crossover Date",

            "Sessions Since Cross",

            "Relative Volume",

            "RSI",

            "Momentum %",

            "Price Action",

            "Confluence",

            "Support",

            "Resistance",

            "Stop",

            "Target"
        ]

        display_df = (
            results[
                display_columns
            ].copy()
        )

        # Formatting
        for column in [
            "Price",
            "Support",
            "Resistance",
            "Stop",
            "Target"
        ]:

            display_df[column] = (
                display_df[column]
                .apply(
                    lambda x:
                    f"${x:.2f}"
                    if pd.notna(x)
                    else "—"
                )
            )

        display_df["Relative Volume"] = (
            display_df[
                "Relative Volume"
            ].apply(
                lambda x:
                f"{x:.2f}x"
                if pd.notna(x)
                else "—"
            )
        )

        display_df["RSI"] = (
            display_df[
                "RSI"
            ].apply(
                lambda x:
                f"{x:.1f}"
                if pd.notna(x)
                else "—"
            )
        )

        display_df["Momentum %"] = (
            display_df[
                "Momentum %"
            ].apply(
                lambda x:
                f"{x:.1f}%"
                if pd.notna(x)
                else "—"
            )
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Detailed selected candidates
        # ----------------------------------------------------

        if selected:

            st.divider()

            st.subheader(
                "Selected Golden Setups"
            )

            for ticker in selected:

                row = results[
                    results["Ticker"] ==
                    ticker
                ].iloc[0]

                with st.expander(
                    f"🎯 {ticker} — "
                    f"{row['Setup']} — "
                    f"Score {row['Score']}",
                    expanded=True
                ):

                    c1, c2, c3, c4 = (
                        st.columns(4)
                    )

                    with c1:

                        st.metric(
                            "Price",
                            f"${row['Price']:.2f}"
                        )

                    with c2:

                        st.metric(
                            "Score",
                            row["Score"]
                        )

                    with c3:

                        st.metric(
                            "Relative Volume",
                            f"{row['Relative Volume']:.2f}x"
                        )

                    with c4:

                        st.metric(
                            "RSI",
                            f"{row['RSI']:.1f}"
                            if pd.notna(
                                row["RSI"]
                            )
                            else "—"
                        )

                    st.write(
                        f"**Setup:** {row['Setup']}"
                    )

                    st.write(
                        f"**Price Action:** "
                        f"{row['Price Action']}"
                    )

                    st.write(
                        f"**Crossover:** "
                        f"{row['Crossover Date']} "
                        f"({row['Sessions Since Cross']} "
                        f"sessions ago)"
                    )

                    st.write(
                        f"**Support:** "
                        f"${row['Support']:.2f}   |   "
                        f"**Resistance:** "
                        f"${row['Resistance']:.2f}"
                    )

                    st.write(
                        f"**Technical Stop:** "
                        f"${row['Stop']:.2f}   |   "
                        f"**Technical Target:** "
                        f"${row['Target']:.2f}"
                    )

                    st.write(
                        f"**Confluence:** "
                        f"{row['Confluence']} elements"
                    )

                    st.caption(
                        row["Reasons"]
                    )


        # ====================================================
        # PHASE 2 BUTTON
        # ====================================================

        st.divider()

        st.header(
            "Phase 2 — Hughes CALL Engine"
        )

        st.write(
            "The options engine will run ONLY on the "
            "stocks you checked above."
        )

        if st.button(
            "🚀 RUN CALL ENGINE ON SELECTED STOCKS",
            type="primary",
            use_container_width=True
        ):

            if not selected:

                st.warning(
                    "Please check at least one stock "
                    "before running Phase 2."
                )

            else:

                option_results = []

                progress = st.progress(
                    0
                )

                status = st.empty()

                total_selected = (
                    len(selected)
                )

                for index, ticker in enumerate(
                    selected
                ):

                    status.write(
                        f"Analyzing CALLs: "
                        f"{ticker} "
                        f"({index + 1} of "
                        f"{total_selected})"
                    )

                    row = results[
                        results["Ticker"] ==
                        ticker
                    ].iloc[0]

                    stock_data = {
                        "Price": row["Price"],
                        "Score": row["Score"],
                        "Setup": row["Setup"]
                    }

                    option = (
                        find_best_call(
                            ticker,
                            stock_data
                        )
                    )

                    if option is None:

                        option_results.append({

                            "Ticker": ticker,

                            "Status": (
                                "NO CALL PASSED"
                            ),

                            "Expiration": "—",

                            "DTE": "—",

                            "Strike": "—",

                            "Bid": "—",

                            "Ask": "—",

                            "Time Value": "—",

                            "1% Max TV": "—",

                            "Hughes Rule": (
                                "NOT PASSED"
                            )
                        })

                    else:

                        trade = (
                            build_call_trade(
                                row,
                                option
                            )
                        )

                        option_results.append({

                            "Ticker": ticker,

                            "Status": (
                                "CALL PASSED"
                            ),

                            "Expiration": (
                                option[
                                    "Expiration"
                                ]
                            ),

                            "DTE": (
                                option["DTE"]
                            ),

                            "Strike": (
                                option["Strike"]
                            ),

                            "Bid": (
                                option["Bid"]
                            ),

                            "Ask": (
                                option["Ask"]
                            ),

                            "Time Value": (
                                option[
                                    "Time Value"
                                ]
                            ),

                            "1% Max TV": (
                                option[
                                    "Max Time Value"
                                ]
                            ),

                            "Hughes Rule": (
                                "PASSED"
                            ),

                            "Intrinsic": (
                                option[
                                    "Intrinsic"
                                ]
                            ),

                            "Spread %": (
                                option[
                                    "Spread %"
                                ]
                            ),

                            "Entry": (
                                trade["Entry"]
                            ),

                            "Stop": (
                                trade["Stop"]
                            ),

                            "Target": (
                                trade["Target"]
                            ),

                            "R:R": (
                                trade["R:R"]
                            )
                        })

                    progress.progress(
                        (
                            index + 1
                        ) /
                        total_selected
                    )

                status.empty()
                progress.empty()

                st.session_state.option_results = (
                    pd.DataFrame(
                        option_results
                    )
                )


# ============================================================
# PHASE 2 RESULTS
# ============================================================

option_results = (
    st.session_state.option_results
)


if option_results is not None:

    st.divider()

    st.header(
        "🎯 Phase 2 — CALL Results"
    )

    passed = option_results[
        option_results[
            "Status"
        ] ==
        "CALL PASSED"
    ]

    failed = option_results[
        option_results[
            "Status"
        ] ==
        "NO CALL PASSED"
    ]

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            "CALLs Passed",
            len(passed)
        )

    with p2:

        st.metric(
            "No Contract Passed",
            len(failed)
        )

    # --------------------------------------------------------
    # Passed contracts
    # --------------------------------------------------------

    if not passed.empty:

        st.subheader(
            "🔥 CALL Contracts That Passed"
        )

        for _, row in passed.iterrows():

            ticker = row["Ticker"]

            with st.expander(
                f"🏆 {ticker} — "
                f"{row['Expiration']} "
                f"${row['Strike']:.0f} CALL",
                expanded=True
            ):

                c1, c2, c3, c4 = (
                    st.columns(4)
                )

                with c1:

                    st.metric(
                        "Expiration",
                        row["Expiration"]
                    )

                with c2:

                    st.metric(
                        "DTE",
                        row["DTE"]
                    )

                with c3:

                    st.metric(
                        "Strike",
                        f"${row['Strike']:.2f}"
                    )

                with c4:

                    st.metric(
                        "Entry",
                        f"${row['Entry']:.2f}"
                    )

                st.success(
                    "HUGHES 1% RULE PASSED"
                )

                st.write(
                    f"**Underlying:** {ticker}"
                )

                st.write(
                    f"**CALL:** "
                    f"{row['Expiration']} "
                    f"${row['Strike']:.0f}"
                )

                st.write(
                    f"**Bid:** ${row['Bid']:.2f}  |  "
                    f"**Ask:** ${row['Ask']:.2f}"
                )

                st.write(
                    f"**Intrinsic Value:** "
                    f"${row['Intrinsic']:.2f}"
                )

                st.write(
                    f"**Time Value:** "
                    f"${row['Time Value']:.2f}"
                )

                st.write(
                    f"**Maximum 1% Time Value:** "
                    f"${row['1% Max TV']:.2f}"
                )

                st.write(
                    f"**Bid/Ask Spread:** "
                    f"{row['Spread %']:.1f}%"
                )

                st.divider()

                t1, t2, t3, t4 = (
                    st.columns(4)
                )

                with t1:

                    st.metric(
                        "Buy To Open",
                        f"${row['Entry']:.2f}"
                    )

                with t2:

                    st.metric(
                        "Option Stop",
                        f"${row['Stop']:.2f}"
                    )

                with t3:

                    st.metric(
                        "Option Target",
                        f"${row['Target']:.2f}"
                    )

                with t4:

                    st.metric(
                        "R:R",
                        f"{row['R:R']:.2f}"
                    )

                st.success(
                    f"BUY TO OPEN {ticker} "
                    f"{row['Expiration']} "
                    f"${row['Strike']:.0f} CALL "
                    f"at ${row['Entry']:.2f} or less."
                )

                st.caption(
                    "The contract is near-ATM to modestly "
                    "OTM and passed the 1% time-value rule "
                    "and liquidity filter."
                )


    # --------------------------------------------------------
    # Failed contracts
    # --------------------------------------------------------

    if not failed.empty:

        st.divider()

        st.subheader(
            "Contracts That Did Not Pass"
        )

        st.dataframe(
            failed[
                [
                    "Ticker",
                    "Status"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # Complete option table
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Phase 2 Summary"
    )

    summary_columns = [
        "Ticker",
        "Status",
        "Expiration",
        "DTE",
        "Strike",
        "Bid",
        "Ask",
        "Time Value",
        "1% Max TV",
        "Hughes Rule"
    ]

    summary = option_results[
        summary_columns
    ].copy()

    for column in [
        "Strike",
        "Bid",
        "Ask",
        "Time Value",
        "1% Max TV"
    ]:

        summary[column] = (
            summary[column]
            .apply(
                lambda x:
                f"${x:.2f}"
                if isinstance(
                    x,
                    (int, float, np.integer, np.floating)
                )
                and
                pd.notna(x)
                else str(x)
            )
        )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Golden Scanner • Stocks Only • Bullish CALL setups"
)

st.caption(
    "Technical Phase 1 → User Selection → "
    "CALL Phase 2 → Hughes 1% Time-Value Rule"
)

st.caption(
    "Scanner output is a research tool, not a guarantee "
    "of future performance. Verify live quotes, liquidity, "
    "expiration and contract details before trading."
)
