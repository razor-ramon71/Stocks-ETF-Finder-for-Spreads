import os
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
# ============================================================
# STOCKS + ETF FINDER FOR SPREADS
# Version 1
#
# Philosophy:
# FOLLOW THE TREND — DON'T PREDICT THE TREND
#
# Core components:
#   1. 50 / 100 EMA trend
#   2. Buying vs selling pressure
#   3. Volume confirmation
#   4. Keltner Channel location
#   5. Price-action confirmation
#
# NO AUTOMATIC TRADING
# ============================================================

st.set_page_config(
    page_title="Stocks + ETF Finder for Spreads",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================

TRADIER_BASE_URL = "https://api.tradier.com/v1"

DEFAULT_MIN_PRICE = 10.0
DEFAULT_MIN_AVG_VOLUME = 500_000

# Keltner settings
KC_EMA_LENGTH = 20
KC_ATR_LENGTH = 20
KC_MULTIPLIER = 2.0

# How many trading days of data we request
HISTORY_DAYS = 260

# ============================================================
# 500-SYMBOL DIVERSIFIED UNIVERSE
#
# This is intentionally NOT a Mag-7-heavy universe.
# It contains stocks and ETFs across multiple sectors.
#
# The scanner can be expanded/rebalanced later.
# ============================================================

UNIVERSE = [

    # ---------------- TECHNOLOGY ----------------
    "ACN","ADBE","ADI","ADSK","AKAM","AMAT","AMD","ANET","APH","APP",
    "AVGO","CDNS","CIEN","CLS","CRM","CSCO","CTSH","DDOG","DELL","DOCU",
    "FIS","FISV","FTNT","GEN","GLW","HPQ","IBM","INTC","INTU","KEYS",
    "KLAC","LRCX","MCHP","MPWR","MRVL","MU","NOW","NTAP","NVDA","ON",
    "ORCL","PANW","PLTR","QCOM","ROP","SMCI","SNPS","STX","TEL","TER",
    "TXN","VEEV","WDC","WDAY","ZBRA",

    # ---------------- FINANCIALS ----------------
    "AFL","AIG","AIZ","AJG","ALL","AMP","AXP","BAC","BEN","BK",
    "BKNG","BLK","BMO","BNS","C","CB","CBOE","CFG","CINF","CMA",
    "COF","DFS","EG","ERIE","FITB","GL","GS","HBAN","HIG","HII",
    "HOOD","ICE","JPM","KEY","KKR","L","LPLA","MA","MET","MKTX",
    "MMC","MS","MSCI","MTB","NDAQ","NTRS","ORI","PNC","PFG","PGR",
    "PRU","RJF","SCHW","SPGI","STT","SYF","TFC","TROW","TRV","USB",
    "V","WFC","WRB",

    # ---------------- HEALTHCARE ----------------
    "ABBV","ABT","ALGN","AMGN","BAX","BDX","BIIB","BMY","BSX","CAH",
    "CNC","COO","COR","CRL","CVS","DHR","DVA","DXCM","ELV","EW",
    "GEHC","GILD","HCA","HOLX","HSIC","HUM","IDXX","ILMN","INCY","IQV",
    "ISRG","JNJ","LH","LLY","MCK","MDT","MOH","MRK","MTD","PFE",
    "PODD","REGN","RMD","RMD","RVTY","SYK","TECH","TMO","UHS","UNH",
    "VRTX","WAT","WST","XRAY","ZBH","ZTS",

    # ---------------- INDUSTRIALS ----------------
    "AAL","ALK","AME","AOS","CAT","CHRW","CMI","CSX","CTAS","DAL",
    "DE","DOV","EMR","ETN","EXPD","FAST","FDX","GD","GE","GNRC",
    "GWW","HON","HUBB","IEX","IR","ITW","J","JCI","LDOS","LHX",
    "LMT","LUV","MAS","MMM","NDSN","NOC","NSC","ODFL","OTIS","PCAR",
    "PH","PNR","PWR","ROK","RSG","RTX","SNA","SWK","TDG","TT",
    "TXT","UAL","UNP","UPS","URI","VRSK","WAB","WM","XYL",

    # ---------------- ENERGY ----------------
    "APA","BKR","COP","CTRA","CVX","DVN","EOG","EQT","FANG","HAL",
    "HES","KMI","MPC","MRO","MUR","OKE","OXY","PSX","PXD","SLB",
    "TRGP","VLO","WMB","XOM",

    # ---------------- MATERIALS ----------------
    "AA","ALB","APD","AVY","BALL","CF","CLF","CTVA","DD","DOW",
    "ECL","EMN","FCX","FMC","IFF","IP","LIN","LYB","MOS","NEM",
    "NUE","PKG","PPG","SHW","STLD","SW","VMC","WRK",

    # ---------------- CONSUMER DISCRETIONARY ----------------
    "ABNB","AZO","BBY","BBWI","BWA","CHWY","CMG","COST","CPRT","CVNA",
    "DHI","DKS","DPZ","DRI","EBAY","ETSY","EXPE","F","GM","GPC",
    "GRMN","HAS","HD","KMX","LEN","LOW","LULU","LVS","MCD","MGM",
    "NCLH","NKE","ORLY","PHM","POOL","RCL","ROST","SBUX","TJX","TGT",
    "TPR","TSCO","TSLA","ULTA","WHR","WYNN","YUM",

    # ---------------- CONSUMER STAPLES ----------------
    "ADM","BG","CAG","CHD","CL","CLX","COKE","CPB","DG","DLTR",
    "EL","GIS","HSY","HRL","KDP","KHC","KMB","KO","KR","MDLZ",
    "MKC","MNST","MO","PEP","PG","PM","SJM","STZ","SYY","TAP",
    "TSN","WBA","WMT",

    # ---------------- COMMUNICATION SERVICES ----------------
    "CHTR","CMCSA","DIS","EA","FOXA","FOX","GOOG","GOOGL","LYV","MTCH",
    "NFLX","NWS","NWSA","OMC","PARA","T","TMUS","TTWO","VZ","WBD",

    # ---------------- UTILITIES ----------------
    "AEE","AEP","AES","ATO","AWK","CEG","CMS","CNP","D","DTE",
    "DUK","ED","EIX","ES","ETR","EVRG","EXC","FE","LNT","NEE",
    "NI","NRG","PCG","PEG","PNW","SO","SRE","WEC","XEL",

    # ---------------- REAL ESTATE ----------------
    "AMH","AVB","BXP","CBRE","CCI","CPT","DLR","EQIX","EQR","ESS",
    "EXR","HST","INVH","IRM","KIM","MAA","O","PLD","PSA","REG",
    "SBAC","SPG","UDR","VICI","VTR","WELL","WY",

    # ========================================================
    # ETFs — BROAD MARKET / SECTOR / THEMATIC / COMMODITIES
    # ========================================================

    "SPY","QQQ","IWM","DIA","MDY","VOO","VTI","SCHX","SPLG","RSP",

    "XLK","XLF","XLV","XLI","XLE","XLP","XLY","XLC","XLU","XLB","XLRE",

    "SMH","SOXX","IGV","HACK","CIBR","ARKK","ARKW","ARKQ","BOTZ","ROBO",

    "XBI","IBB","IHI","XPH","XHE","VHT","FHLC",

    "XOP","OIH","XES","AMLP","UNG","USO","DBO","DBC","PDBC",

    "GLD","IAU","SLV","SIL","COPX","DBA",

    "TLT","IEF","SHY","LQD","HYG","JNK","TIP","AGG",

    "EEM","EFA","VEA","VWO","FXI","EWZ","EWJ","EWG","EWU","INDA",
    "KWEB","MCHI","IEMG","ACWI",

    "VNQ","IYR","XLRE",

    "ARKF","FINX","KRE","KBE","XRT","IBUY",

    "ITA","PPA","XAR","JETS",

    "ICLN","TAN","FAN","PBW",

    "XME","PICK","URA",

    "SOXL","SOXS","TQQQ","SQQQ"
]

# Remove accidental duplicate symbols
UNIVERSE = sorted(list(set(UNIVERSE)))


# ============================================================
# TRADIER
# ============================================================

def get_token():
    """
    Reads Tradier token from Streamlit secrets first,
    then environment variable.
    """

    try:
        token = st.secrets["TRADIER_TOKEN"]
        if token:
            return token
    except Exception:
        pass

    return os.getenv("TRADIER_TOKEN")


def tradier_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }


# ============================================================
# HISTORICAL DATA
# ============================================================

@st.cache_data(ttl=900)
def get_history(symbol, token):

    url = f"{TRADIER_BASE_URL}/markets/history"

    params = {
        "symbol": symbol,
        "interval": "daily",
        "start": (
            pd.Timestamp.today() -
            pd.Timedelta(days=HISTORY_DAYS + 30)
        ).strftime("%Y-%m-%d"),
        "end": pd.Timestamp.today().strftime("%Y-%m-%d")
    }

    try:

        response = requests.get(
            url,
            headers=tradier_headers(token),
            params=params,
            timeout=15
        )

        if response.status_code != 200:
            return None

        data = response.json()

        history = data.get("history")

        if not history:
            return None

        rows = history.get("day", [])

        if not rows:
            return None

        df = pd.DataFrame(rows)

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for column in numeric_columns:

            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        df = df.dropna(
            subset=["close", "high", "low", "volume"]
        )

        df = df.sort_values("date").reset_index(drop=True)

        return df

    except Exception:
        return None


# ============================================================
# TECHNICAL CALCULATIONS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["ema100"] = (
        df["close"]
        .ewm(span=100, adjust=False)
        .mean()
    )

    # --------------------------------------------------------
    # TRUE RANGE / ATR
    # --------------------------------------------------------

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] - previous_close
    ).abs()

    tr3 = (
        df["low"] - previous_close
    ).abs()

    df["true_range"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        df["true_range"]
        .ewm(
            span=KC_ATR_LENGTH,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # KELTNER CHANNEL
    # --------------------------------------------------------

    df["kc_middle"] = (
        df["close"]
        .ewm(
            span=KC_EMA_LENGTH,
            adjust=False
        )
        .mean()
    )

    df["kc_upper"] = (
        df["kc_middle"] +
        KC_MULTIPLIER * df["atr"]
    )

    df["kc_lower"] = (
        df["kc_middle"] -
        KC_MULTIPLIER * df["atr"]
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    df["avg_volume20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["relative_volume"] = (
        df["volume"] /
        df["avg_volume20"]
    )

    # --------------------------------------------------------
    # CANDLE PRESSURE
    #
    # Close location within candle:
    #
    # +1 = close at high
    # -1 = close at low
    #  0 = middle
    # --------------------------------------------------------

    candle_range = (
        df["high"] -
        df["low"]
    )

    candle_range = candle_range.replace(
        0,
        np.nan
    )

    df["close_location"] = (
        (
            df["close"] -
            df["low"]
        ) /
        candle_range
    )

    df["candle_pressure"] = (
        df["close_location"] * 2
    ) - 1

    # --------------------------------------------------------
    # DIRECTIONAL VOLUME
    # --------------------------------------------------------

    price_change = df["close"].diff()

    df["up_volume"] = np.where(
        price_change > 0,
        df["volume"],
        0
    )

    df["down_volume"] = np.where(
        price_change < 0,
        df["volume"],
        0
    )

    df["up_volume20"] = (
        pd.Series(df["up_volume"])
        .rolling(20)
        .sum()
    )

    df["down_volume20"] = (
        pd.Series(df["down_volume"])
        .rolling(20)
        .sum()
    )

    total_directional_volume = (
        df["up_volume20"] +
        df["down_volume20"]
    )

    df["volume_pressure"] = np.where(
        total_directional_volume > 0,
        (
            df["up_volume20"] -
            df["down_volume20"]
        ) /
        total_directional_volume,
        0
    )

    # --------------------------------------------------------
    # ACCUMULATION / DISTRIBUTION
    # --------------------------------------------------------

    money_flow_multiplier = (
        (
            (df["close"] - df["low"]) -
            (df["high"] - df["close"])
        )
        /
        candle_range
    )

    money_flow_multiplier = (
        money_flow_multiplier
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    money_flow_volume = (
        money_flow_multiplier *
        df["volume"]
    )

    df["ad_line"] = (
        money_flow_volume
        .cumsum()
    )

    df["ad_slope"] = (
        df["ad_line"] -
        df["ad_line"].shift(20)
    )

    # --------------------------------------------------------
    # PRICE MOMENTUM
    # --------------------------------------------------------

    df["return5"] = (
        df["close"]
        .pct_change(5)
    )

    df["return20"] = (
        df["close"]
        .pct_change(20)
    )

    # --------------------------------------------------------
    # EMA SLOPE
    # --------------------------------------------------------

    df["ema50_slope"] = (
        df["ema50"] -
        df["ema50"].shift(10)
    )

    df["ema100_slope"] = (
        df["ema100"] -
        df["ema100"].shift(10)
    )

    return df


# ============================================================
# PRESSURE SCORE
# ============================================================

def pressure_score(row):

    # Candle strength
    candle_component = (
        (row["candle_pressure"] + 1) / 2
    ) * 100

    # Directional volume
    volume_component = (
        (row["volume_pressure"] + 1) / 2
    ) * 100

    # Relative volume confirmation
    relative_volume = row["relative_volume"]

    if pd.isna(relative_volume):
        volume_strength = 50

    elif relative_volume >= 2:
        volume_strength = 100

    elif relative_volume >= 1.5:
        volume_strength = 85

    elif relative_volume >= 1:
        volume_strength = 65

    else:
        volume_strength = 45

    # A/D direction
    ad_component = 50

    if row["ad_slope"] > 0:
        ad_component = 75

    elif row["ad_slope"] < 0:
        ad_component = 25

    buying_pressure = (
        candle_component * 0.30 +
        volume_component * 0.35 +
        volume_strength * 0.15 +
        ad_component * 0.20
    )

    buying_pressure = max(
        0,
        min(100, buying_pressure)
    )

    selling_pressure = 100 - buying_pressure

    return buying_pressure, selling_pressure


# ============================================================
# KELTNER POSITION
# ============================================================

def keltner_position(row):

    price = row["close"]
    middle = row["kc_middle"]
    upper = row["kc_upper"]
    lower = row["kc_lower"]

    channel_width = upper - lower

    if channel_width <= 0:
        return "Unknown", 50

    position = (
        price - lower
    ) / channel_width

    if position >= 0.85:
        label = "Upper / Extended"
        quality = 25

    elif position >= 0.65:
        label = "Upper-Mid"
        quality = 55

    elif position >= 0.35:
        label = "Middle / Preferred"
        quality = 100

    elif position >= 0.15:
        label = "Lower-Mid"
        quality = 75

    else:
        label = "Lower / Oversold"
        quality = 45

    return label, quality


# ============================================================
# PRICE ACTION
# ============================================================

def price_action_score(df):

    if len(df) < 5:
        return 50, "Insufficient data"

    last = df.iloc[-1]
    previous = df.iloc[-2]

    score = 50
    signals = []

    # Bullish close
    if last["close"] > last["open"]:
        score += 10
        signals.append("Bullish close")

    # Strong close inside candle
    if last["close_location"] >= 0.70:
        score += 10
        signals.append("Strong close")

    # Higher low
    if last["low"] > previous["low"]:
        score += 10
        signals.append("Higher low")

    # Price above previous close
    if last["close"] > previous["close"]:
        score += 5

    # Recent momentum
    if last["return5"] > 0:
        score += 5

    score = max(0, min(100, score))

    if score >= 75:
        label = "Bullish confirmation"

    elif score >= 60:
        label = "Positive"

    elif score <= 40:
        label = "Weak"

    else:
        label = "Neutral"

    return score, label


# ============================================================
# SECURITY ANALYSIS
# ============================================================

def analyze_symbol(symbol, token):

    df = get_history(symbol, token)

    if df is None:
        return None

    if len(df) < 120:
        return None

    df = calculate_indicators(df)

    row = df.iloc[-1]

    if pd.isna(row["ema100"]):
        return None

    price = float(row["close"])

    avg_volume = float(
        row["avg_volume20"]
    )

    # Basic liquidity/price filter
    if price < DEFAULT_MIN_PRICE:
        return None

    if avg_volume < DEFAULT_MIN_AVG_VOLUME:
        return None

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if (
        row["ema50"] > row["ema100"]
        and
        row["ema50_slope"] > 0
    ):
        direction = "BULLISH"

    elif (
        row["ema50"] < row["ema100"]
        and
        row["ema50_slope"] < 0
    ):
        direction = "BEARISH"

    else:
        direction = "NEUTRAL"

    # --------------------------------------------------------
    # TREND SCORE
    # --------------------------------------------------------

    trend_score = 50

    if direction == "BULLISH":

        trend_score = 70

        if price > row["ema50"]:
            trend_score += 10

        if row["ema50_slope"] > 0:
            trend_score += 10

        if row["ema100_slope"] > 0:
            trend_score += 10

    elif direction == "BEARISH":

        trend_score = 70

        if price < row["ema50"]:
            trend_score += 10

        if row["ema50_slope"] < 0:
            trend_score += 10

        if row["ema100_slope"] < 0:
            trend_score += 10

    # --------------------------------------------------------
    # PRESSURE
    # --------------------------------------------------------

    buying_pressure, selling_pressure = (
        pressure_score(row)
    )

    net_pressure = (
        buying_pressure -
        selling_pressure
    )

    # --------------------------------------------------------
    # KELTNER
    # --------------------------------------------------------

    kc_label, kc_quality = (
        keltner_position(row)
    )

    # --------------------------------------------------------
    # PRICE ACTION
    # --------------------------------------------------------

    pa_score, pa_label = (
        price_action_score(df)
    )

    # --------------------------------------------------------
    # PRESSURE DIRECTION
    # --------------------------------------------------------

    if len(df) >= 6:

        recent_pressure = []

        for i in range(-5, 0):

            r = df.iloc[i]

            bp, sp = pressure_score(r)

            recent_pressure.append(
                bp - sp
            )

        older_pressure = np.mean(
            recent_pressure[:2]
        )

        recent_pressure_avg = np.mean(
            recent_pressure[-2:]
        )

        if recent_pressure_avg > older_pressure + 5:
            pressure_direction = "Increasing ↑"

        elif recent_pressure_avg < older_pressure - 5:
            pressure_direction = "Decreasing ↓"

        else:
            pressure_direction = "Stable →"

    else:
        pressure_direction = "Unknown"

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    # Trend = 35%
    # Pressure = 35%
    # Keltner = 20%
    # Price action = 10%

    directional_pressure = (
        buying_pressure
        if direction == "BULLISH"
        else selling_pressure
    )

    final_score = (
        trend_score * 0.35 +
        directional_pressure * 0.35 +
        kc_quality * 0.20 +
        pa_score * 0.10
    )

    final_score = round(
        max(0, min(100, final_score)),
        1
    )

    # --------------------------------------------------------
    # SETUP QUALITY
    # --------------------------------------------------------

    if final_score >= 85:
        setup = "PRIME"

    elif final_score >= 75:
        setup = "STRONG"

    elif final_score >= 65:
        setup = "GOOD"

    elif final_score >= 55:
        setup = "WATCH"

    else:
        setup = "WEAK"

    # --------------------------------------------------------
    # ONLY FOLLOW ESTABLISHED TRENDS
    # --------------------------------------------------------

    follow_trend = (
        direction != "NEUTRAL"
    )

    return {
        "Ticker": symbol,
        "Price": round(price, 2),

        "Direction": direction,

        "Score": final_score,
        "Setup": setup,

        "Buying Pressure": round(
            buying_pressure, 1
        ),

        "Selling Pressure": round(
            selling_pressure, 1
        ),

        "Net Pressure": round(
            net_pressure, 1
        ),

        "Pressure Trend": pressure_direction,

        "50 EMA": round(
            float(row["ema50"]), 2
        ),

        "100 EMA": round(
            float(row["ema100"]), 2
        ),

        "Keltner": kc_label,

        "KC Middle": round(
            float(row["kc_middle"]), 2
        ),

        "KC Upper": round(
            float(row["kc_upper"]), 2
        ),

        "KC Lower": round(
            float(row["kc_lower"]), 2
        ),

        "Price Action": pa_label,

        "Relative Volume": round(
            float(row["relative_volume"]), 2
        ),

        "Follow Trend": follow_trend
    }


# ============================================================
# SCANNER
# ============================================================

def run_scan(token, symbols):

    results = []

    progress = st.progress(0)

    status = st.empty()

    total = len(symbols)

    for count, symbol in enumerate(symbols, start=1):

        status.text(
            f"Scanning {symbol} "
            f"({count}/{total})"
        )

        try:

            result = analyze_symbol(
                symbol,
                token
            )

            if result is not None:
                results.append(result)

        except Exception:
            pass

        progress.progress(
            count / total
        )

    status.text(
        f"Scan complete — "
        f"{len(results)} qualifying securities."
    )

    return pd.DataFrame(results)


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("📈 Stocks + ETF Finder for Spreads")

st.markdown(
    """
### Follow the trend — don't predict the trend.

This scanner searches a diversified universe of stocks and ETFs
for established trends supported by measurable buying/selling
pressure and favorable Keltner Channel positioning.

**Version 1 does not place trades.**
"""
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("Scanner Settings")

min_price = st.sidebar.number_input(
    "Minimum stock/ETF price",
    min_value=1.0,
    max_value=1000.0,
    value=10.0,
    step=1.0
)

min_volume = st.sidebar.number_input(
    "Minimum average volume",
    min_value=0,
    max_value=100_000_000,
    value=500_000,
    step=100_000
)

direction_filter = st.sidebar.selectbox(
    "Direction",
    [
        "Both",
        "Bullish Only",
        "Bearish Only"
    ]
)

minimum_score = st.sidebar.slider(
    "Minimum score",
    0,
    100,
    65
)

max_results = st.sidebar.slider(
    "Maximum results",
    5,
    50,
    20
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Universe: **{len(UNIVERSE)} symbols**"
)

# ------------------------------------------------------------
# TOKEN
# ------------------------------------------------------------

token = get_token()

if not token:

    st.error(
        """
        Tradier API token not found.

        Add your Tradier token to Streamlit secrets as:

        TRADIER_TOKEN = "YOUR_TOKEN_HERE"
        """
    )

    st.stop()


# ------------------------------------------------------------
# RUN SCAN
# ------------------------------------------------------------

if st.button(
    "🚀 RUN GOLDEN SPREAD SCAN",
    type="primary",
    use_container_width=True
):

    # Apply UI price/volume filters
    global DEFAULT_MIN_PRICE
    global DEFAULT_MIN_AVG_VOLUME

    DEFAULT_MIN_PRICE = min_price
    DEFAULT_MIN_AVG_VOLUME = min_volume

    results = run_scan(
        token,
        UNIVERSE
    )

    if results.empty:

        st.warning(
            "No qualifying securities found. "
            "Try lowering the minimum score or liquidity filters."
        )

        st.stop()

    # Direction filter
    if direction_filter == "Bullish Only":

        results = results[
            results["Direction"] == "BULLISH"
        ]

    elif direction_filter == "Bearish Only":

        results = results[
            results["Direction"] == "BEARISH"
        ]

    # Score filter
    results = results[
        results["Score"] >= minimum_score
    ]

    # Follow trend only
    results = results[
        results["Follow Trend"] == True
    ]

    # Sort
    results = results.sort_values(
        "Score",
        ascending=False
    )

    results = results.head(
        max_results
    )

    if results.empty:

        st.warning(
            "No securities passed the selected filters."
        )

        st.stop()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    bullish_count = (
        results["Direction"]
        .eq("BULLISH")
        .sum()
    )

    bearish_count = (
        results["Direction"]
        .eq("BEARISH")
        .sum()
    )

    prime_count = (
        results["Setup"]
        .eq("PRIME")
        .sum()
    )

    col1.metric(
        "Candidates",
        len(results)
    )

    col2.metric(
        "Bullish",
        bullish_count
    )

    col3.metric(
        "Bearish",
        bearish_count
    )

    col4.metric(
        "Prime",
        prime_count
    )

    st.markdown("---")

    # --------------------------------------------------------
    # TOP CANDIDATES
    # --------------------------------------------------------

    st.subheader(
        "🏆 Top Candidates"
    )

    display_columns = [
        "Ticker",
        "Price",
        "Direction",
        "Score",
        "Setup",
        "Buying Pressure",
        "Selling Pressure",
        "Net Pressure",
        "Pressure Trend",
        "Keltner",
        "Price Action",
        "Relative Volume"
    ]

    st.dataframe(
        results[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DETAILED CARDS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔎 Candidate Details"
    )

    for _, row in results.iterrows():

        with st.expander(
            f"{row['Ticker']}  |  "
            f"{row['Direction']}  |  "
            f"Score {row['Score']}"
        ):

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Price",
                f"${row['Price']}"
            )

            c2.metric(
                "50 EMA",
                f"${row['50 EMA']}"
            )

            c3.metric(
                "100 EMA",
                f"${row['100 EMA']}"
            )

            c4.metric(
                "QOD Score",
                row["Score"]
            )

            st.markdown("### Market Pressure")

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Buying Pressure",
                row["Buying Pressure"]
            )

            p2.metric(
                "Selling Pressure",
                row["Selling Pressure"]
            )

            p3.metric(
                "Net Pressure",
                row["Net Pressure"]
            )

            st.write(
                f"**Pressure Direction:** "
                f"{row['Pressure Trend']}"
            )

            st.markdown("### Keltner Channel")

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "Position",
                row["Keltner"]
            )

            k2.metric(
                "Middle",
                f"${row['KC Middle']}"
            )

            k3.metric(
                "Upper",
                f"${row['KC Upper']}"
            )

            k4.metric(
                "Lower",
                f"${row['KC Lower']}"
            )

            st.markdown("### Price Action")

            st.write(
                row["Price Action"]
            )

            st.markdown(
                f"""
**Trend:** `{row['Direction']}`

**50 EMA:** ${row['50 EMA']}

**100 EMA:** ${row['100 EMA']}

**Relative Volume:** {row['Relative Volume']}x

**Setup:** `{row['Setup']}`

**Interpretation:** The scanner is looking for an established
trend with pressure supporting that direction while avoiding
entries that are excessively extended within the Keltner Channel.
"""
            )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    st.markdown("---")

    csv = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Results CSV",
        csv,
        "golden_spread_candidates.csv",
        "text/csv",
        use_container_width=True
    )

else:

    st.info(
        """
        Press **RUN GOLDEN SPREAD SCAN** to scan the universe.

        The scanner will search for:

        • 50 EMA / 100 EMA established trends  
        • Increasing directional pressure  
        • Buying vs. selling volume  
        • Favorable Keltner positioning  
        • Price-action confirmation  
        • Adequate liquidity  

        The eventual goal is to identify candidates suitable
        for 1–2 month defined-risk option spreads.
        """
    )

st.markdown("---")

st.caption(
    "Stocks + ETF Finder for Spreads — Version 1 | "
    "Research tool only. No automatic trading."
)
