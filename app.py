"""
Market Scanner web app (works on phone and PC).
Signals only, no trades are placed. Not financial advice.
"""

import io

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Scanner", page_icon="📈", layout="centered")

FOREX = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X", "EURGBP=X",
     "EURJPY=X", "GBPJPY=X","GC=F", "SI=F", "CL=F",
]
BATCH = 50
@st.cache_data(ttl=86400)
def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
    table = pd.read_html(io.StringIO(html))[0]
    return [t.replace(".", "-") for t in table["Symbol"].tolist()]


def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def analyze(symbol, df, allow_shorts, account, risk_pct):
    df = df.dropna()
    if len(df) < 60:
        return None
    close = df["Close"]
    e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    r = rsi(close).iloc[-1]
    a = atr(df).iloc[-1]
    h20 = df["High"].shift(1).rolling(20).max().iloc[-1]
    l20 = df["Low"].shift(1).rolling(20).min().iloc[-1]
    price = close.iloc[-1]
    if np.isnan(a) or a == 0:
        return None

    side = None
    if e20 > e50 and price > e50 and price > h20 and 50 <= r <= 75:
        side = "LONG"
    elif allow_shorts and e20 < e50 and price < e50 and price < l20 and 25 <= r <= 50:
        side = "SHORT"
    if not side:
        return None

    s = 1 if side == "LONG" else -1
    stop = price - s * 1.5 * a
    target = price + s * 3.0 * a
    risk_per_unit = abs(price - stop)
    size = (account * risk_pct / 100) / risk_per_unit if risk_per_unit else 0
    dec = 5 if ("=X" in symbol and "JPY" not in symbol) else 3
    return {
        "Symbol": symbol.replace("=X", ""),
        "Type": "Forex" if "=X" in symbol else "Stock",
        GOLD = ["GC=F"]  # gold futures, price per ounce in USD
        "Side": side,
        "Entry": round(price, dec),
        "Stop-Loss": round(stop, dec),
        "Take-Profit": round(target, dec),
        "RSI": round(r, 1),
        "Size": round(size, 2),
        "Strength": round(abs(e20 - e50) / a, 2),
    }


@st.cache_data(ttl=900, show_spinner=False)
def download(chunk):
    return yf.download(list(chunk), period="1y", interval="1d", group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)


def color_side(v):
    if v == "LONG":
        return "background-color: #d6f5e9; color: #0a5c40"
    if v == "SHORT":
        return "background-color: #fbe0e0; color: #8a1f1f"
    return ""


st.title("📈 Market Scanner")
st.caption("S&P 500 stocks + major forex pairs. Entry, stop-loss and take-profit levels.")

with st.expander("Settings", expanded=False):
    c1, c2 = st.columns(2)
    use_stocks = c1.checkbox("Stocks", True)
    use_forex = c2.checkbox("Forex", True)
    use_shorts = st.checkbox("Also scan stock shorts", False)
    account = st.number_input("Account size ($)", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

if st.button("🔍 SCAN NOW", type="primary", use_container_width=True):
    if not (use_stocks or use_forex):
        st.error("Turn on Stocks and/or Forex in Settings.")
        st.stop()

    symbols = []
    if use_forex:
        symbols += FOREX
    if use_stocks:
        try:
            symbols += get_sp500()
        except Exception:
            st.error("Could not load the S&P 500 list. Check the connection and try again.")
            st.stop()

    bar = st.progress(0.0, text="Starting...")
    rows = []
    total = len(symbols)
    for i in range(0, total, BATCH):
        chunk = symbols[i:i + BATCH]
        try:
            data = download(tuple(chunk))
        except Exception:
            continue
        for sym in chunk:
            try:
                df = data[sym] if len(chunk) > 1 else data
                shorts = ("=X" in sym) or use_shorts
                res = analyze(sym, df, shorts, account, risk_pct)
                if res:
                    rows.append(res)
            except Exception:
                pass
        done = min(i + BATCH, total)
        bar.progress(done / total, text=f"Scanned {done} of {total}... {len(rows)} setups")
    bar.empty()

    if not rows:
        st.info("No setups found right now.")
    else:
        df = pd.DataFrame(rows).sort_values("Strength", ascending=False).reset_index(drop=True)
        st.success(f"{len(df)} setups found")
        st.dataframe(df.style.map(color_side, subset=["Side"]),
                     use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False),
                           file_name="signals.csv", mime="text/csv",
                           use_container_width=True)

st.caption("Size = shares/units that risk your Risk % at the stop-loss. "
           "Signals only, not financial advice. Paper-trade first.")
