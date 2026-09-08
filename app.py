#!/usr/bin/env python3
GOLDEN QOD SCANNER — TRADIER EDITION
------------------------------------
Finds short-term, liquid stock/option setups and prints ONLY the
QOD-style trade instruction the user asked for.

Data source: Tradier API only. No Yahoo Finance.

The scanner is QOD-inspired: it uses observable technical/market
confluence to rank opportunities. It does not reproduce any
proprietary Price Headley formula.

Install:
    pip install requests pandas numpy

Set your Tradier token:
    Windows PowerShell:
        $env:TRADIER_TOKEN="YOUR_TOKEN"
    macOS/Linux:
        export TRADIER_TOKEN="YOUR_TOKEN"

Run:
    python golden_qod_tradier.py

Optional:
    python golden_qod_tradier.py --symbols WMT,V,HD,COST,MSFT,AAPL
    python golden_qod_tradier.py --max-results 15
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

@dataclass
class Config:
    min_stock_price: float = 10.0
    min_avg_volume: int = 500_000
    min_market_cap: float = 2_000_000_000
    min_score: float = 76.0
    max_results: int = 15

    # QOD-style option window: short term, but not so close that
    # ordinary noise/theta dominates the setup.
    min_dte: int = 7
    max_dte: int = 40

    # Prefer liquid near-the-money options.
    min_delta: float = 0.40
    max_delta: float = 0.99
    min_open_interest: int = 100
    min_option_volume: int = 10
    max_spread_pct: float = 0.20

    # Risk model for the generated stop/target.
    stop_atr_multiple: float = 1.15
    target_r_multiple: float = 1.5

    # Historical lookback.
    history_days: int = 420


CFG = Config()

# Broad, established-company universe. This is deliberately NOT just
# the Magnificent 7. Users can replace it with their own list via
# --symbols.
UNIVERSE = """
AAPL MSFT AMZN GOOGL GOOG META NVDA AVGO ORCL CRM ADBE AMD INTC QCOM TXN
AMAT MU LRCX KLAC ADI CSCO IBM NOW INTU PANW CRWD PLTR ACN
WMT COST TGT HD LOW TJX ROST NKE MCD SBUX CMG YUM
KO PEP MDLZ KHC GIS HSY CL PG CLX KMB EL UL
JNJ ABBV MRK PFE BMY LLY AMGN GILD REGN ISRG MDT SYK ABT BSX
UNH CVS HUM CI ELV HCA
JPM BAC WFC C C GS MS BLK SCHW AXP USB PNC COF
V MA AIG ALL TRV MET PRU
CAT DE HON GE RTX LMT NOC GD ETN PH DOV EMR ITW MMM CMI
UPS FDX UNP CSX NSC DAL UAL LUV
XOM CVX COP SLB EOG OXY MPC PSX VLO HAL
NEE DUK SO D AEP EXC XEL
DIS NFLX CMCSA TMO DHR AMT PLD CCI EQIX
VZ T TMO
SPGI MCO ICE CME
BKNG MAR HLT ABNB
ORLY AZO OREO
DELL HPQ
CVNA F GM F
"""


# ---------------------------------------------------------------------
# TRADIER CLIENT
# ---------------------------------------------------------------------

class TradierError(RuntimeError):
    pass


class TradierClient:
    def __init__(self, token: str, base_url: Optional[str] = None):
        self.token = token.strip()
        self.base_url = (
            base_url or os.getenv("TRADIER_BASE_URL")
            or "https://api.tradier.com/v1"
        ).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        })

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = self.session.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            raise TradierError(f"Tradier connection error: {e}") from e

        if r.status_code != 200:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:500]
            raise TradierError(f"Tradier HTTP {r.status_code}: {detail}")

        try:
            return r.json()
        except Exception as e:
            raise TradierError("Tradier returned invalid JSON") from e

    def quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        if not symbols:
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        # Keep requests comfortably sized.
        for i in range(0, len(symbols), 100):
            chunk = symbols[i:i + 100]
            data = self.get("/markets/quotes", {
                "symbols": ",".join(chunk),
                "greeks": "false",
            })
            q = data.get("quotes", {}).get("quote", [])
            if isinstance(q, dict):
                q = [q]
            for item in q or []:
                sym = str(item.get("symbol", "")).upper()
                if sym:
                    out[sym] = item
        return out

    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        data = self.get("/markets/history", {
            "symbol": symbol,
            "interval": "daily",
            "start": start.isoformat(),
            "end": end.isoformat(),
        })
        days = data.get("history", {}).get("day", [])
        if isinstance(days, dict):
            days = [days]
        if not days:
            return pd.DataFrame()

        df = pd.DataFrame(days)
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return df

    def expirations(self, symbol: str) -> List[str]:
        data = self.get("/markets/options/expirations", {
            "symbol": symbol,
            "includeAllRoots": "true",
            "strikes": "false",
            "contractSize": "false",
            "expirationType": "true",
        })
        raw = data.get("expirations", {}).get("date", [])
        if isinstance(raw, dict):
            raw = [raw.get("date")]
        return [str(x) for x in raw if x]

    def option_chain(self, symbol: str, expiration: str) -> List[Dict[str, Any]]:
        data = self.get("/markets/options/chains", {
            "symbol": symbol,
            "expiration": expiration,
            "greeks": "true",
        })
        raw = data.get("options", {}).get("option", [])
        if isinstance(raw, dict):
            raw = [raw]
        return raw or []


# ---------------------------------------------------------------------
# TECHNICAL ENGINE
# ---------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def macd(s: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def stochastic(df: pd.DataFrame, n: int = 14) -> pd.Series:
    lo = df["low"].rolling(n).min()
    hi = df["high"].rolling(n).max()
    return 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["rsi"] = rsi(d["close"], 14)
    d["atr"] = atr(d, 14)
    d["atr_pct"] = d["atr"] / d["close"]
    d["macd"], d["macd_signal"], d["macd_hist"] = macd(d["close"])
    d["stoch"] = stochastic(d, 14)
    d["avg_vol20"] = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / d["avg_vol20"].replace(0, np.nan)

    # Daily VWAP-like relationship using cumulative typical-price volume.
    tp = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap"] = (tp * d["volume"]).cumsum() / d["volume"].cumsum()

    d["prior20_high"] = d["high"].rolling(20).max().shift(1)
    d["prior20_low"] = d["low"].rolling(20).min().shift(1)
    d["prior60_high"] = d["high"].rolling(60).max().shift(1)
    d["prior60_low"] = d["low"].rolling(60).min().shift(1)

    return d


def score_stock(
    symbol: str,
    d: pd.DataFrame,
    spy: Optional[pd.DataFrame],
    quote: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if len(d) < 210:
        return None

    x = d.iloc[-1]
    prev = d.iloc[-2]
    close = float(x["close"])

    if not np.isfinite(close) or close < CFG.min_stock_price:
        return None

    avg_vol = float(x["avg_vol20"]) if np.isfinite(x["avg_vol20"]) else 0
    if avg_vol < CFG.min_avg_volume:
        return None

    # Tradier quote marketcap is not guaranteed on every feed.
    marketcap = quote.get("marketcap")
    try:
        marketcap = float(marketcap)
    except Exception:
        marketcap = 0.0
    if marketcap and marketcap < CFG.min_market_cap:
        return None

    e20, e50, e200 = map(float, [x["ema20"], x["ema50"], x["ema200"]])
    r = float(x["rsi"])
    atrv = float(x["atr"])
    vr = float(x["vol_ratio"]) if np.isfinite(x["vol_ratio"]) else 1.0
    macdh = float(x["macd_hist"])
    macdh_prev = float(prev["macd_hist"])
    st = float(x["stoch"]) if np.isfinite(x["stoch"]) else 50.0
    vwapv = float(x["vwap"])

    # --- Bullish score ------------------------------------------------
    bull = 0.0
    bear = 0.0
    bull_reasons: List[str] = []
    bear_reasons: List[str] = []

    # Trend: 20/50/200 structure.
    if close > e20 > e50 > e200:
        bull += 20
        bull_reasons.append("trend aligned")
    elif close > e20 > e50:
        bull += 15
        bull_reasons.append("20/50 EMA bullish")
    elif close > e50:
        bull += 9
    elif close < e20 < e50 < e200:
        bear += 20
        bear_reasons.append("trend aligned")
    elif close < e20 < e50:
        bear += 15
        bear_reasons.append("20/50 EMA bearish")
    elif close < e50:
        bear += 9

    # Momentum.
    if 52 <= r <= 72 and macdh > 0:
        bull += 14
        bull_reasons.append("momentum strong")
    elif r > 72 and macdh > 0:
        bull += 9
        bull_reasons.append("momentum extended")
    elif r < 48 and macdh < 0:
        bear += 14
        bear_reasons.append("momentum strong")
    elif r < 28 and macdh < 0:
        bear += 9
        bear_reasons.append("momentum extended")

    # MACD acceleration.
    if macdh > 0 and macdh >= macdh_prev:
        bull += 5
        bull_reasons.append("MACD improving")
    if macdh < 0 and macdh <= macdh_prev:
        bear += 5
        bear_reasons.append("MACD weakening")

    # Stochastic confirmation without requiring extreme readings.
    if st > 50 and st < 90:
        bull += 5
        bull_reasons.append("stochastic confirms")
    elif st < 50 and st > 10:
        bear += 5
        bear_reasons.append("stochastic confirms")

    # VWAP relationship.
    if close > vwapv:
        bull += 6
        bull_reasons.append("above VWAP")
    elif close < vwapv:
        bear += 6
        bear_reasons.append("below VWAP")

    # Volume / accumulation.
    if vr >= 1.25 and close > float(prev["close"]):
        bull += 10
        bull_reasons.append("volume expansion")
    elif vr >= 1.25 and close < float(prev["close"]):
        bear += 10
        bear_reasons.append("volume expansion")

    # Support / resistance reaction and breakout.
    p20h = float(x["prior20_high"]) if np.isfinite(x["prior20_high"]) else np.nan
    p20l = float(x["prior20_low"]) if np.isfinite(x["prior20_low"]) else np.nan
    p60h = float(x["prior60_high"]) if np.isfinite(x["prior60_high"]) else np.nan
    p60l = float(x["prior60_low"]) if np.isfinite(x["prior60_low"]) else np.nan

    if np.isfinite(p20h) and close > p20h:
        bull += 10
        bull_reasons.append("20-day breakout")
    elif np.isfinite(p20l) and close < p20l:
        bear += 10
        bear_reasons.append("20-day breakdown")

    # Pullback / reaction near 20 EMA.
    near20 = abs(close - e20) / max(close, 0.01)
    if near20 <= 0.025 and close > e50:
        bull += 7
        bull_reasons.append("20 EMA reaction")
    if near20 <= 0.025 and close < e50:
        bear += 7
        bear_reasons.append("20 EMA rejection")

    # Higher-timeframe confirmation: 20/50 relationship and 60-day
    # direction.
    close20ago = float(d["close"].iloc[-21])
    close60ago = float(d["close"].iloc[-61])
    if close > close20ago > close60ago and e20 > e50:
        bull += 8
        bull_reasons.append("higher timeframe confirms")
    if close < close20ago < close60ago and e20 < e50:
        bear += 8
        bear_reasons.append("higher timeframe confirms")

    # Relative strength versus SPY.
    rs = 0.0
    if spy is not None and len(spy) >= 61:
        spy_close = spy["close"].astype(float)
        stock_ret20 = close / float(d["close"].iloc[-21]) - 1
        spy_ret20 = float(spy_close.iloc[-1]) / float(spy_close.iloc[-21]) - 1
        rs = stock_ret20 - spy_ret20
        if rs > 0.03:
            bull += 10
            bull_reasons.append("relative strength")
        elif rs < -0.03:
            bear += 10
            bear_reasons.append("relative weakness")

    # Volatility sanity: avoid absurdly thin/violent names.
    if 0.015 <= float(x["atr_pct"]) <= 0.10:
        if bull > bear:
            bull += 5
        elif bear > bull:
            bear += 5

    direction = "CALL" if bull >= bear else "PUT"
    score = max(bull, bear)

    if score < CFG.min_score:
        return None

    reasons = bull_reasons if direction == "CALL" else bear_reasons
    setup = classify_setup(direction, x)

    return {
        "symbol": symbol,
        "direction": direction,
        "score": round(score, 1),
        "bull": round(bull, 1),
        "bear": round(bear, 1),
        "price": close,
        "atr": atrv,
        "rsi": r,
        "volume_ratio": vr,
        "rs": rs,
        "setup": setup,
        "reasons": reasons[:6],
        "marketcap": marketcap,
        "avg_volume": avg_vol,
    }


def classify_setup(direction: str, x: pd.Series) -> str:
    close = float(x["close"])
    e20 = float(x["ema20"])
    e50 = float(x["ema50"])
    if direction == "CALL":
        if np.isfinite(x["prior20_high"]) and close > float(x["prior20_high"]):
            return "BREAKOUT"
        if abs(close - e20) / close <= 0.025 and close > e50:
            return "20 EMA REACTION"
        return "MOMENTUM"
    else:
        if np.isfinite(x["prior20_low"]) and close < float(x["prior20_low"]):
            return "BREAKDOWN"
        if abs(close - e20) / close <= 0.025 and close < e50:
            return "20 EMA REJECTION"
        return "MOMENTUM"


# ---------------------------------------------------------------------
# OPTIONS ENGINE
# ---------------------------------------------------------------------

def num(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def get_greek(opt: Dict[str, Any], name: str) -> float:
    g = opt.get("greeks")
    if isinstance(g, dict):
        return num(g.get(name), 0.0)
    return num(opt.get(name), 0.0)


def option_contract_score(opt: Dict[str, Any], direction: str, stock_score: float) -> float:
    bid = num(opt.get("bid"))
    ask = num(opt.get("ask"))
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else num(opt.get("last"))
    if mid <= 0:
        return -1e9

    spread = (ask - bid) / mid if ask > 0 and bid >= 0 else 1.0
    oi = num(opt.get("open_interest"))
    vol = num(opt.get("volume"))
    delta = get_greek(opt, "delta")
    abs_delta = abs(delta)

    if abs_delta < CFG.min_delta or abs_delta > CFG.max_delta:
        return -1e9
    if oi < CFG.min_open_interest:
        return -1e9
    if vol < CFG.min_option_volume:
        return -1e9
    if spread > CFG.max_spread_pct:
        return -1e9

    # Near-the-money/liquid is preferred.
    score = stock_score
    score += min(10, math.log10(max(oi, 1)) * 2)
    score += min(8, math.log10(max(vol, 1)) * 2)
    score += max(0, 8 - spread * 40)
    score += max(0, 8 - abs(abs_delta - 0.50) * 25)

    # Calls/puts need the correct Greek sign.
    if direction == "CALL" and delta <= 0:
        return -1e9
    if direction == "PUT" and delta >= 0:
        return -1e9

    return score


def choose_expiration(expirations: List[str]) -> Optional[Tuple[str, int]]:
    today = date.today()
    candidates = []
    for e in expirations:
        try:
            d = datetime.strptime(e[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        dte = (d - today).days
        if CFG.min_dte <= dte <= CFG.max_dte:
            candidates.append((e[:10], dte))

    if not candidates:
        return None

    # Prefer roughly 7 DTE, while staying within the 2-week limit.
    return min(candidates, key=lambda x: abs(x[1] - 7))


def select_option(
    client: TradierClient,
    stock: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    symbol = stock["symbol"]
    direction = stock["direction"]
    expirations = client.expirations(symbol)
    chosen = choose_expiration(expirations)
    if not chosen:
        return None

    expiration, dte = chosen
    chain = client.option_chain(symbol, expiration)
    if not chain:
        return None

    best = None
    best_score = -1e12

    for opt in chain:
        typ = str(opt.get("option_type", opt.get("type", ""))).lower()
        if direction == "CALL" and typ not in ("call", "c"):
            continue
        if direction == "PUT" and typ not in ("put", "p"):
            continue

        bid = num(opt.get("bid"))
        ask = num(opt.get("ask"))
        last = num(opt.get("last"))
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        if mid <= 0:
            continue

        s = option_contract_score(opt, direction, stock["score"])
        if s <= best_score:
            continue

        strike = num(opt.get("strike"))
        if strike <= 0:
            continue

        best = dict(opt)
        best["_mid"] = mid
        best["_expiration"] = expiration
        best["_dte"] = dte
        best_score = s

    if best is None:
        return None

    # Generate the stock-based stop/target first.
    price = float(stock["price"])
    atrv = float(stock["atr"])
    if direction == "CALL":
        # Structural-ish stop: ATR under current price, but don't put
        # it absurdly far away.
        stop_stock = price - CFG.stop_atr_multiple * atrv
        target_stock = price + CFG.target_r_multiple * (price - stop_stock)
    else:
        stop_stock = price + CFG.stop_atr_multiple * atrv
        target_stock = price - CFG.target_r_multiple * (stop_stock - price)

    entry = float(best["_mid"])
    delta = abs(get_greek(best, "delta"))

    # First-order option-premium projection. This is intentionally an
    # estimate, not a promise: IV/theta/gamma can materially change it.
    move_to_target = abs(target_stock - price)
    move_to_stop = abs(price - stop_stock)
    option_target = max(entry + delta * move_to_target, entry * 1.25)
    option_stop = max(0.05, entry - delta * move_to_stop)

    # Keep output sane and quote-like.
    option_target = round(option_target, 2)
    option_stop = round(option_stop, 2)
    entry = round(entry, 2)

    return {
        **stock,
        "expiration": expiration,
        "dte": dte,
        "strike": num(best.get("strike")),
        "option_type": "CALL" if direction == "CALL" else "PUT",
        "option_symbol": best.get("symbol", ""),
        "bid": bid,
        "ask": ask,
        "mid": entry,
        "delta": get_greek(best, "delta"),
        "open_interest": int(num(best.get("open_interest"))),
        "option_volume": int(num(best.get("volume"))),
        "entry": entry,
        "stock_stop": round(stop_stock, 2),
        "stock_target": round(target_stock, 2),
        "option_stop": option_stop,
        "option_target": option_target,
    }


# ---------------------------------------------------------------------
# OUTPUT + STREAMLIT APP
# ---------------------------------------------------------------------

def month_week_label(expiration: str) -> str:
    d = datetime.strptime(expiration[:10], "%Y-%m-%d").date()
    week = ((d.day - 1) // 7) + 1
    return f"{d.strftime('%B')} Week {week}"

def fmt_money(x: float) -> str:
    return f"{x:.2f}"

def grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 85: return "A"
    if score >= 80: return "B+"
    if score >= 75: return "B"
    return "C+"

def qod_confirmation(direction: str, d: pd.DataFrame, rs: float):
    x=d.iloc[-1]; p=d.iloc[-2]
    close=float(x['close']); e20=float(x['ema20']); e50=float(x['ema50']); e200=float(x['ema200'])
    r=float(x['rsi']); mh=float(x['macd_hist']); mph=float(p['macd_hist']); st=float(x['stoch']) if np.isfinite(x['stoch']) else 50.0
    vr=float(x['vol_ratio']) if np.isfinite(x['vol_ratio']) else 1.0; vw=float(x['vwap'])
    score=0; reasons=[]
    if direction=='CALL':
        if close>e20 and e20>e50: score+=10; reasons.append('trend: price above rising 20/50 EMA')
        if e50>e200: score+=10; reasons.append('trend: 50 EMA above 200 EMA')
        if r>=52: score+=8; reasons.append('momentum: bullish RSI')
        if mh>0: score+=7; reasons.append('momentum: MACD positive')
        if mh>=mph: score+=10; reasons.append('momentum: MACD improving')
        near20=abs(close-e20)/max(close,.01)
        if near20<=.025 and close>e50: score+=15; reasons.append('timing: 20 EMA reaction')
        elif np.isfinite(x['prior20_high']) and close>float(x['prior20_high']): score+=15; reasons.append('timing: 20-day breakout')
        if np.isfinite(x['swing_low_10']) and close>float(x['swing_low_10']): score+=10; reasons.append('structure: holding support')
        if close>vw: score+=10; reasons.append('VWAP: above VWAP')
        if vr>=1.15: score+=10; reasons.append('volume: participation expanding')
        if st>50: score+=5; reasons.append('stochastic: bullish')
        if rs>0: score+=5; reasons.append('relative strength vs SPY')
    else:
        if close<e20 and e20<e50: score+=10; reasons.append('trend: price below falling 20/50 EMA')
        if e50<e200: score+=10; reasons.append('trend: 50 EMA below 200 EMA')
        if r<=48: score+=8; reasons.append('momentum: bearish RSI')
        if mh<0: score+=7; reasons.append('momentum: MACD negative')
        if mh<=mph: score+=10; reasons.append('momentum: MACD weakening')
        near20=abs(close-e20)/max(close,.01)
        if near20<=.025 and close<e50: score+=15; reasons.append('timing: 20 EMA rejection')
        elif np.isfinite(x['prior20_low']) and close<float(x['prior20_low']): score+=15; reasons.append('timing: 20-day breakdown')
        if np.isfinite(x['swing_high_10']) and close<float(x['swing_high_10']): score+=10; reasons.append('structure: respecting resistance')
        if close<vw: score+=10; reasons.append('VWAP: below VWAP')
        if vr>=1.15: score+=10; reasons.append('volume: participation expanding')
        if st<50: score+=5; reasons.append('stochastic: bearish')
        if rs<0: score+=5; reasons.append('relative weakness vs SPY')
    return min(score,100), reasons

def score_stock_qod(symbol,d,spy,quote):
    # Run the proven Golden engine first, then add the separate QOD gate.
    original_min=CFG.min_score
    CFG.min_score=0
    try: base=score_stock(symbol,d,spy,quote)
    finally: CFG.min_score=original_min
    if not base: return None
    qod,reasons=qod_confirmation(base['direction'],d,base['rs'])
    combined=base['score']*.60+qod*.40
    return {**base,'golden_score':base['score'],'qod_score':round(qod,1),'combined_score':round(combined,1),'qod_reasons':reasons}

def format_trade(t):
    contract=(f"({t['symbol']}) {month_week_label(t['expiration'])} "
              f"({datetime.strptime(t['expiration'][:10], '%Y-%m-%d').strftime('%m/%d')}) "
              f"{t['strike']:g} {t['option_type'].title()}")
    return ("NEW TRADE:\n\nBuy-to-Open the\n"
            f"{contract} at {fmt_money(t['entry'])} or less.\n\n"
            f"Apply a stop of {fmt_money(t['option_stop'])}\n"
            f"Target to {fmt_money(t['option_target'])} or more in Full position.")

def position_size(entry,stop,account,risk_pct):
    risk_dollars=account*risk_pct/100
    per_contract=max(abs(entry-stop)*100,.01)
    contracts=max(1,int(risk_dollars//per_contract))
    return contracts,contracts*per_contract

# Streamlit import is intentionally isolated here so the original Tradier
# scanner remains recognizable and all market data still comes from Tradier.
import streamlit as st

st.set_page_config(page_title='Golden QOD Scanner — Tradier',page_icon='🟡',layout='wide')
st.title('🟡 GOLDEN QOD SCANNER')
st.caption('Tradier only • Golden Score → QOD Confirmation → Options Engine → Trade Plan')

def get_token():
    token=''
    try: token=st.secrets.get('TRADIER_TOKEN','')
    except Exception: pass
    return token or os.getenv('TRADIER_TOKEN','')

def cached_history(token,symbol,start,end):
    client=TradierClient(token)
    return client.history(symbol,start,end)

def cached_quotes(token,symbols):
    return TradierClient(token).quotes(symbols)

def cached_chain(token,symbol,expiration):
    return TradierClient(token).option_chain(symbol,expiration)

def select_option_cached(token,stock):
    client=TradierClient(token); symbol=stock['symbol']; direction=stock['direction']; price=float(stock['price']); atrv=float(stock['atr'])
    today=date.today(); exps=[]
    for e in client.expirations(symbol):
        try: d=datetime.strptime(e[:10],'%Y-%m-%d').date()
        except Exception: continue
        dte=(d-today).days
        if CFG.min_dte<=dte<=CFG.max_dte: exps.append((e[:10],dte))
    best=None; best_score=-1e18
    for expiration,dte in sorted(exps,key=lambda x:x[1]):
        chain=cached_chain(token,symbol,expiration)
        opts=[]
        for opt in chain:
            typ=str(opt.get('option_type',opt.get('type',''))).lower()
            if direction=='CALL' and typ not in ('call','c'): continue
            if direction=='PUT' and typ not in ('put','p'): continue
            bid=num(opt.get('bid')); ask=num(opt.get('ask')); last=num(opt.get('last'))
            mid=(bid+ask)/2 if bid>0 and ask>0 else last
            strike=num(opt.get('strike')); delta=get_greek(opt,'delta'); oi=int(num(opt.get('open_interest'))); vol=int(num(opt.get('volume')))
            if mid<=0 or strike<=0 or bid<=0 or ask<=0 or oi<CFG.min_open_interest or vol<CFG.min_option_volume: continue
            spread=(ask-bid)/mid if mid else 1
            if spread>CFG.max_spread_pct or abs(delta)<CFG.min_delta or abs(delta)>CFG.max_delta: continue
            if direction=='CALL' and delta<=0: continue
            if direction=='PUT' and delta>=0: continue
            intrinsic=max(price-strike,0) if direction=='CALL' else max(strike-price,0)
            tv=max(0,mid-intrinsic); tvpct=tv/price if price>0 else 1
            if intrinsic<=0 or tvpct>0.01: continue
            opts.append({'opt':opt,'mid':mid,'strike':strike,'delta':delta,'oi':oi,'vol':vol,'spread':spread,'intrinsic':intrinsic,'tv':tv,'tvpct':tvpct})
        # Single option: 70% premium stop and at least 1.25R modeled target.
        stock_stop=price-CFG.stop_atr_multiple*atrv if direction=='CALL' else price+CFG.stop_atr_multiple*atrv
        stock_target=price+CFG.target_r_multiple*abs(price-stock_stop) if direction=='CALL' else price-CFG.target_r_multiple*abs(stock_stop-price)
        for o in opts:
            entry=o['mid']; stop=round(max(.05,entry*.70),2); target=max(entry+abs(o['delta'])*abs(stock_target-price),entry*1.125); rr=(target-entry)/(entry-stop) if entry>stop else 0
            if rr<1.25: continue
            score=stock['combined_score']+max(0,12-o['tvpct']*1000)+min(10,rr*4)+max(0,8-o['spread']*40)
            if score>best_score:
                best_score=score; best={**stock,'structure':'SINGLE OPTION','expiration':expiration,'dte':dte,'strike':o['strike'],'option_type':direction,'option_symbol':o['opt'].get('symbol',''),'bid':o['opt'].get('bid',0),'ask':o['opt'].get('ask',0),'entry':round(entry,2),'delta':o['delta'],'open_interest':o['oi'],'option_volume':o['vol'],'spread_pct':o['spread'],'intrinsic':o['intrinsic'],'time_value':o['tv'],'time_value_pct':o['tvpct'],'option_stop':stop,'option_target':round(target,2),'rr':rr}
        # Debit spreads: long ITM + short farther OTM, 1% rule on long leg, max R:R >= 1.25.
        for long in opts:
            for short in opts:
                if long['strike']==short['strike']: continue
                if direction=='CALL' and short['strike']<=long['strike']: continue
                if direction=='PUT' and short['strike']>=long['strike']: continue
                width=abs(short['strike']-long['strike']); debit=long['mid']-short['mid']
                if width<=0 or debit<=0 or debit>=width: continue
                max_profit=width-debit; rr=max_profit/debit
                if rr<1.25: continue
                score=stock['combined_score']+max(0,12-long['tvpct']*1000)+min(10,rr*4)+max(0,8-long['spread']*40)
                if score>best_score:
                    best_score=score; best={**stock,'structure':'DEBIT SPREAD','expiration':expiration,'dte':dte,'long_strike':long['strike'],'short_strike':short['strike'],'strike':long['strike'],'short_entry':round(short['mid'],2),'entry':round(debit,2),'debit':round(debit,2),'width':round(width,2),'max_loss':round(debit,2),'max_profit':round(max_profit,2),'max_return_pct':max_profit/debit,'rr':rr,'option_type':direction,'option_symbol':long['opt'].get('symbol',''),'delta':long['delta'],'open_interest':long['oi'],'option_volume':long['vol'],'spread_pct':long['spread'],'intrinsic':long['intrinsic'],'time_value':long['tv'],'time_value_pct':long['tvpct'],'option_stop':round(debit*.70,2),'option_target':round(debit+max_profit*.75,2)}
    return best


def run_scan(token,symbols,min_golden,min_qod,max_results):
    today=date.today(); start=today-timedelta(days=CFG.history_days)
    quotes=cached_quotes(token,symbols)
    spy=cached_history(token,'SPY',start,today)
    spy=add_indicators(spy) if not spy.empty else None
    stocks=[]; trades=[]
    bar=st.progress(0); status=st.empty()
    for i,symbol in enumerate(symbols):
        status.write(f'Scanning {symbol} ({i+1}/{len(symbols)})')
        q=quotes.get(symbol)
        try:
            if not q: continue
            # Fast quote filter before spending a history call.
            if num(q.get('last',q.get('close'))) < CFG.min_stock_price: continue
            d=cached_history(token,symbol,start,today)
            if d.empty: continue
            d=add_indicators(d)
            c=score_stock_qod(symbol,d,spy,q)
            if not c or c['golden_score']<min_golden or c['qod_score']<min_qod: continue
            stocks.append(c)
        except Exception:
            pass
        bar.progress((i+1)/len(symbols))
    # Only the strongest confirmed stocks go to the option-chain engine.
    stocks.sort(key=lambda x:(x['combined_score'],x['qod_score'],x['golden_score']),reverse=True)
    for c in stocks[:max_results*3]:
        try:
            t=select_option_cached(token,c)
            if t: trades.append(t)
        except Exception:
            pass
    trades.sort(key=lambda x:(x['combined_score'],x['qod_score'],x['golden_score'],x['option_volume'],x['open_interest']),reverse=True)
    status.empty(); bar.empty()
    return stocks[:max_results*2],trades[:max_results]

with st.sidebar:
    st.header('Scanner Controls')
    token=get_token()
    if token: st.success('Tradier token detected')
    else: st.error('TRADIER_TOKEN not found')
    min_golden=st.slider('Minimum Golden Score',55,95,70)
    min_qod=st.slider('Minimum QOD Confirmation',55,95,70)
    max_results=st.slider('Maximum trade alerts',5,25,15)
    account=st.number_input('Account size ($)',1000.0,10000000.0,10000.0,1000.0)
    risk_pct=st.number_input('Risk per trade (%)',0.25,5.0,1.0,0.25)
    mode=st.radio('Universe',['Golden Universe','Custom symbols'])
    if mode=='Custom symbols':
        raw=st.text_input('Symbols','WMT,V,HD,COST,MSFT,AAPL,NVDA,AMZN')
        symbols=sorted(set(x.strip().upper() for x in raw.split(',') if x.strip()))
    else: symbols=sorted(set(UNIVERSE.split()))
    st.caption(f'{len(symbols)} symbols')
    run=st.button('🚀 RUN GOLDEN SCANNER',type='primary',use_container_width=True)

if not token:
    st.warning('Add TRADIER_TOKEN to Streamlit Secrets or your environment. Do not put the token in app.py.')
    st.stop()

if run:
    with st.spinner('Building Golden + QOD opportunities from Tradier...'):
        stocks,trades=run_scan(token,symbols,min_golden,min_qod,max_results)
    st.session_state['stocks']=stocks
    st.session_state['trades']=trades
    st.session_state['scan_time']=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

stocks=st.session_state.get('stocks',[])
trades=st.session_state.get('trades',[])

if not stocks and not trades:
    st.info('Set your Tradier token, choose the universe, then press RUN GOLDEN SCANNER.')
    st.stop()

st.success(f"Scan complete • {len(stocks)} confirmed stock candidates • {len(trades)} option trade plans • {st.session_state.get('scan_time','')}")

if trades:
    st.header('🚨 GOLDEN QOD TRADE ALERTS')
    for i,t in enumerate(trades,1):
        with st.container(border=True):
            st.subheader(f"#{i}  {t['symbol']}  •  {grade(t['combined_score'])}  •  {t['direction']}  •  {t.get('structure','SINGLE OPTION')}")
            st.metric('Combined Score',f"{t['combined_score']:.1f}")
            c1,c2,c3,c4=st.columns(4)
            c1.metric('Golden',f"{t['golden_score']:.1f}")
            c2.metric('QOD',f"{t['qod_score']:.1f}")
            c3.metric('Stock',f"${t['price']:.2f}")
            c4.metric('DTE',t['dte'])
            st.success('QOD CONFIRMED')
            st.code(format_trade(t),language='text')
            c1,c2,c3,c4=st.columns(4)
            c1.write(f"**Strike:** {t['strike']:g}" if t.get('structure')!='DEBIT SPREAD' else f"**Spread:** {t['long_strike']:g}/{t['short_strike']:g}")
            c2.write(f"**Delta:** {t['delta']:.3f}")
            c3.write(f"**Entry:** ${t['entry']:.2f} debit" if t.get('structure')=='DEBIT SPREAD' else f"**Entry:** ${t['entry']:.2f}")
            c4.write(f"**Spread:** {t['spread_pct']*100:.1f}%")
            c1.write(f"**OI:** {t['open_interest']:,}")
            c2.write(f"**Option volume:** {t['option_volume']:,}")
            c3.write(f"**Option stop:** ${t['option_stop']:.2f}")
            c4.write(f"**Max/Target:** ${t.get('max_profit',t.get('option_target',0)):.2f}" if t.get('structure')=='DEBIT SPREAD' else f"**Target:** ${t['option_target']:.2f}")
            contracts,risk=position_size(t['entry'],t['option_stop'],account,risk_pct)
            st.write(f"**Modeled position:** {contracts} contract(s) • approx. premium risk ${risk:,.2f} at the modeled stop. Actual risk can differ with fills, gaps and slippage.")
            st.write('**Confluence:** ' + ' • '.join(t['qod_reasons'][:8]))
else:
    st.warning('No option contracts passed the liquidity/delta/spread filters. Try a slightly lower QOD threshold or a custom symbol list.')

if stocks:
    st.header('🏆 Confirmed Golden + QOD Candidates')
    table=[]
    for s in stocks:
        table.append({'Ticker':s['symbol'],'Combined':s['combined_score'],'Golden':s['golden_score'],'QOD':s['qod_score'],
                      'Grade':grade(s['combined_score']),'Direction':s['direction'],'Setup':s['setup'],'Price':round(s['price'],2),
                      'RSI':round(s['rsi'],1),'Stoch':round(float(s.get('stoch',np.nan)),1) if np.isfinite(s.get('stoch',np.nan)) else np.nan,
                      'Vol x':round(s['volume_ratio'],2),'ATR':round(s['atr'],2)})
    df=pd.DataFrame(table)
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button('⬇️ Download confirmed candidates CSV',df.to_csv(index=False).encode(),file_name='golden_qod_candidates.csv',mime='text/csv')

if trades:
    rows=[]
    for i,t in enumerate(trades,1):
        rows.append({'Rank':i,'Ticker':t['symbol'],'Combined':t['combined_score'],'Golden':t['golden_score'],'QOD':t['qod_score'],
                     'Grade':grade(t['combined_score']),'Direction':t['direction'],'Setup':t['setup'],'Expiration':t['expiration'],
                     'DTE':t['dte'],'Structure':t.get('structure','SINGLE OPTION'),'Strike':t['strike'],'Entry':t['entry'],'Stop':t['option_stop'],'Target':t['option_target'],'Time Value %':round(t.get('time_value_pct',0)*100,2),'R:R':round(t.get('rr',0),2),
                     'Delta':round(t['delta'],3),'OI':t['open_interest'],'Option Volume':t['option_volume'],'Spread %':round(t['spread_pct']*100,2)})
    trade_df=pd.DataFrame(rows)
    st.download_button('⬇️ Download trade plans CSV',trade_df.to_csv(index=False).encode(),file_name='golden_qod_trade_plans.csv',mime='text/csv')

st.caption('Model note: this is an independent QOD-inspired scanner using observable technical/market confluence. It is not a reproduction of any proprietary trading formula or a guarantee of results.')
