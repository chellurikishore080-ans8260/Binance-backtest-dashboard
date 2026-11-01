import streamlit as st
import pandas as pd
import numpy as np
import ccxt

st.set_page_config(layout="wide")

exchange = ccxt.binance({
    "enableRateLimit": True,
    "timeout": 20000
})

# -----------------------------
# Historical data fetch
# -----------------------------
def fetch_ohlcv_range(symbol, timeframe, days=365):
    now_ms = exchange.milliseconds()
    since_ms = now_ms - (days * 24 * 60 * 60 * 1000)

    all_rows = []
    while since_ms < now_ms:
        try:
            new_rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
            if not new_rows:
                break
            all_rows.extend(new_rows)
            since_ms = new_rows[-1][0] + 1
        except Exception:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)
    return df

# -----------------------------
# Strategy functions
# -----------------------------
def trend_direction(series):
    if len(series) < 2:
        return "Neutral"
    if series.iloc[-1] > series.iloc[-2]:
        return "Bullish"
    elif series.iloc[-1] < series.iloc[-2]:
        return "Bearish"
    return "Neutral"

def liquidity_sweep(df, lookback=20):
    if len(df) < lookback + 2:
        return False, "None"

    recent_high = df["high"].tail(lookback).max()
    recent_low = df["low"].tail(lookback).min()
    last_high = df["high"].iloc[-1]
    last_low = df["low"].iloc[-1]
    last_close = df["close"].iloc[-1]

    bullish = (last_low < recent_low) and (last_close > recent_low)
    bearish = (last_high > recent_high) and (last_close < recent_high)

    if bullish:
        return True, "Bullish"
    if bearish:
        return True, "Bearish"
    return False, "None"

def bos_check(df, lookback=10):
    if len(df) < lookback + 1:
        return False, "None"

    recent_high = df["high"].tail(lookback).max()
    recent_low = df["low"].tail(lookback).min()
    last_close = df["close"].iloc[-1]

    if last_close > recent_high:
        return True, "Bullish"
    if last_close < recent_low:
        return True, "Bearish"
    return False, "None"

def volume_confirm(df, mult=1.2):
    if len(df) < 20:
        return False
    avg_vol = df["volume"].rolling(20).mean().iloc[-1]
    last_vol = df["volume"].iloc[-1]
    return last_vol > avg_vol * mult

def get_signal(df_1h, df_4h, df_1d):
    if len(df_1h) < 50 or len(df_4h) < 20 or len(df_1d) < 10:
        return "NoTrade"

    trend_1d = trend_direction(df_1d["close"])
    trend_4h = trend_direction(df_4h["close"])

    sweep_ok, sweep_dir = liquidity_sweep(df_1h)
    bos_ok, bos_dir = bos_check(df_1h)
    vol_ok = volume_confirm(df_1h)

    if trend_1d == "Bullish" and trend_4h == "Bullish" and sweep_ok and sweep_dir == "Bullish" and bos_ok and bos_dir == "Bullish" and vol_ok:
        return "Long"

    if trend_1d == "Bearish" and trend_4h == "Bearish" and sweep_ok and sweep_dir == "Bearish" and bos_ok and bos_dir == "Bearish" and vol_ok:
        return "Short"

    return "NoTrade"

def backtest(symbol="BTC/USDT", days=365):
    df_1h = fetch_ohlcv_range(symbol, "1h", days)
    df_4h = fetch_ohlcv_range(symbol, "4h", days)
    df_1d = fetch_ohlcv_range(symbol, "1d", days)

    if df_1h.empty or df_4h.empty or df_1d.empty:
        return pd.DataFrame()

    position = 0
    entry_price = 0
    trades = []

    max_len = max(len(df_1h), len(df_4h), len(df_1d))
    for i in range(max_len):
        if i < 50:
            continue

        h = df_1h.iloc[:i+1].copy()
        h4 = df_4h.iloc[:i+1].copy()
        d1 = df_1d.iloc[:i+1].copy()

        if len(h4) < 20 or len(d1) < 10:
            continue

        signal = get_signal(h, h4, d1)
        current_price = h["close"].iloc[-1]

        if signal == "Long" and position == 0:
            position = 1
            entry_price = current_price

        elif signal == "Short" and position == 0:
            position = -1
            entry_price = current_price

        elif signal == "NoTrade" and position != 0:
            if position == 1:
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100

            trades.append({
                "entry": entry_price,
                "exit": current_price,
                "pnl_pct": pnl_pct,
                "side": "Long" if position == 1 else "Short"
            })
            position = 0

    if position != 0:
        last_price = df_1h["close"].iloc[-1]
        if position == 1:
            pnl_pct = (last_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - last_price) / entry_price * 100

        trades.append({
            "entry": entry_price,
            "exit": last_price,
            "pnl_pct": pnl_pct,
            "side": "Long" if position == 1 else "Short"
        })

    return pd.DataFrame(trades)

def generate_report(trades_df):
    if trades_df is None or trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl_pct": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0
        }

    df = trades_df.copy()
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0)

    total_trades = len(df)
    wins = (df["pnl_pct"] > 0).sum()
    losses = (df["pnl_pct"] < 0).sum()
    win_rate = (wins / total_trades) * 100 if total_trades else 0.0
    total_pnl_pct = df["pnl_pct"].sum()
    avg_trade_pct = df["pnl_pct"].mean()

    gross_profit = df[df["pnl_pct"] > 0]["pnl_pct"].sum()
    gross_loss = abs(df[df["pnl_pct"] < 0]["pnl_pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = 1000.0
    peak = equity
    max_drawdown = 0.0
    for value in df["pnl_pct"]:
        equity *= (1 + value / 100)
        if equity > peak:
            peak = equity
        dd = ((peak - equity) / peak) * 100
        if dd > max_drawdown:
            max_drawdown = dd

    report = {
        "total_trades": int(total_trades),
        "win_rate": round(win_rate, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_trade_pct": round(avg_trade_pct, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "best_trade_pct": round(df["pnl_pct"].max(), 2),
        "worst_trade_pct": round(df["pnl_pct"].min(), 2),
        "wins": int(wins),
        "losses": int(losses)
    }
    return report

# -----------------------------
# UI
# -----------------------------
st.title("Binance Smart Money Backtest Dashboard")

symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
days = st.slider("Backtest Days", 30, 365, 365)

if st.button("Run Backtest"):
    with st.spinner("Running strategy..."):
        trades_df = backtest(symbol=symbol, days=days)
        report = generate_report(trades_df)

    st.success("Backtest completed")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win Rate %", f"{report['win_rate']}%")
    col2.metric("Total PnL %", f"{report['total_pnl_pct']}%")
    col3.metric("Profit Factor", f"{report['profit_factor']}")
    col4.metric("Max Drawdown %", f"{report['max_drawdown_pct']}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Trades", report["total_trades"])
    col6.metric("Avg Trade %", f"{report['avg_trade_pct']}%")
    col7.metric("Best Trade %", f"{report['best_trade_pct']}%")
    col8.metric("Worst Trade %", f"{report['worst_trade_pct']}%")

    if trades_df.empty:
        st.warning("No trades found for selected period.")
    else:
        st.subheader("Trade Log")
        st.dataframe(trades_df, use_container_width=True)

    st.subheader("Summary")
    st.json(report)
