import io
import zipfile
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Double SuperTrend Binance Backtest", layout="wide")

VISION_BASE = "https://data.binance.vision/data/spot"
BINANCE_API = "https://api.binance.com"
EXCLUDED_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "BUSD", "DAI", "USDE", "EUR", "USDT", "UST", "USTC", "PAXG"}
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
INTERVALS = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


@st.cache_data(ttl=900, show_spinner=False)
def get_top_usdt_symbols(limit: int = 50) -> pd.DataFrame:
    """Current active spot USDT pairs ranked by 24-hour quote volume."""
    session = requests.Session()
    info_response = session.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=25)
    info_response.raise_for_status()
    ticker_response = session.get(f"{BINANCE_API}/api/v3/ticker/24hr", timeout=35)
    ticker_response.raise_for_status()
    eligible = {
        item["symbol"] for item in info_response.json()["symbols"]
        if item.get("status") == "TRADING" and item.get("quoteAsset") == "USDT"
        and item.get("isSpotTradingAllowed", True)
        and not item.get("symbol", "").endswith(LEVERAGED_SUFFIXES)
        and item.get("baseAsset") not in EXCLUDED_BASES
    }
    rows = [{"symbol": row["symbol"], "quote_volume_24h": float(row["quoteVolume"])}
            for row in ticker_response.json() if row.get("symbol") in eligible]
    if not rows:
        return pd.DataFrame(columns=["symbol", "quote_volume_24h"])
    return pd.DataFrame(rows).sort_values("quote_volume_24h", ascending=False).head(limit).reset_index(drop=True)


def _read_vision_zip(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=40)
    if response.status_code == 404:
        return pd.DataFrame()
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next((name for name in archive.namelist() if name.endswith(".csv")), None)
        if csv_name is None:
            return pd.DataFrame()
        frame = pd.read_csv(archive.open(csv_name), header=None)
    if frame.empty or frame.shape[1] < 6:
        return pd.DataFrame()
    frame = frame.iloc[:, :len(KLINE_COLUMNS)].copy()
    frame.columns = KLINE_COLUMNS[:len(frame.columns)]
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["open_time"], errors="coerce"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[["timestamp", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_vision_month(symbol: str, interval: str, month: str) -> pd.DataFrame:
    return _read_vision_zip(f"{VISION_BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip")


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_vision_day(symbol: str, interval: str, day: str) -> pd.DataFrame:
    return _read_vision_zip(f"{VISION_BASE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day}.zip")


def month_range(start: date, end: date):
    current = pd.Timestamp(start).replace(day=1)
    last = pd.Timestamp(end).replace(day=1)
    while current <= last:
        yield current.strftime("%Y-%m")
        current += pd.offsets.MonthBegin(1)


def fetch_vision_range(symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
    # Download a 90-day warm-up window before the requested test period so ATR/SuperTrend
    # state is initialized before the first eligible entry signal.
    warmup_start = start - timedelta(days=90)
    parts = [fetch_vision_month(symbol, interval, month) for month in month_range(warmup_start, end)]
    # Binance monthly archives may not yet exist for the current month. Add available daily
    # files for the current month; duplicate timestamps are removed below.
    today_utc = datetime.now(timezone.utc).date()
    if end.year == today_utc.year and end.month == today_utc.month:
        daily_start = max(date(end.year, end.month, 1), warmup_start)
        daily_end = min(end, today_utc)
        day = daily_start
        while day <= daily_end:
            parts.append(fetch_vision_day(symbol, interval, day.isoformat()))
            day += timedelta(days=1)
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = pd.concat(parts, ignore_index=True).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    end_exclusive = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return frame[frame["timestamp"] < end_exclusive].reset_index(drop=True)


def pine_supertrend(frame: pd.DataFrame, factor: float, period: int):
    """SuperTrend with Pine's direction convention (-1 up, +1 down)."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    previous_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    midpoint = (high + low) / 2
    basic_upper, basic_lower = midpoint + factor * atr, midpoint - factor * atr
    final_upper, final_lower = basic_upper.copy(), basic_lower.copy()
    line = pd.Series(np.nan, index=frame.index, dtype=float)
    direction = pd.Series(np.nan, index=frame.index, dtype=float)
    for i in range(len(frame)):
        if pd.isna(atr.iloc[i]):
            continue
        if i == 0 or pd.isna(direction.iloc[i - 1]):
            direction.iloc[i] = 1
            line.iloc[i] = final_upper.iloc[i]
            continue
        prev_upper, prev_lower = final_upper.iloc[i - 1], final_lower.iloc[i - 1]
        final_upper.iloc[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < prev_upper or close.iloc[i - 1] > prev_upper else prev_upper
        final_lower.iloc[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > prev_lower or close.iloc[i - 1] < prev_lower else prev_lower
        if direction.iloc[i - 1] == 1:
            direction.iloc[i] = -1 if close.iloc[i] > final_upper.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] < final_lower.iloc[i] else -1
        line.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == -1 else final_upper.iloc[i]
    return line, direction


def build_signals(frame: pd.DataFrame, fast_atr: int, fast_factor: float, slow_atr: int,
                  slow_factor: float, entry_rule: str, exit_rule: str) -> pd.DataFrame:
    data = frame.copy().reset_index(drop=True)
    data["fast_st"], data["fast_dir"] = pine_supertrend(data, fast_factor, fast_atr)
    data["slow_st"], data["slow_dir"] = pine_supertrend(data, slow_factor, slow_atr)
    fast_up, fast_down = data["fast_dir"].eq(-1), data["fast_dir"].eq(1)
    slow_up, slow_down = data["slow_dir"].eq(-1), data["slow_dir"].eq(1)
    both_up, both_down = fast_up & slow_up, fast_down & slow_down
    if entry_rule == "Both agree":
        long_entry = both_up & ~both_up.shift(1, fill_value=False)
        short_entry = both_down & ~both_down.shift(1, fill_value=False)
    else:
        long_entry = data["fast_st"].gt(data["slow_st"]) & data["fast_st"].shift(1).le(data["slow_st"].shift(1)) & fast_up
        short_entry = data["fast_st"].lt(data["slow_st"]) & data["fast_st"].shift(1).ge(data["slow_st"].shift(1)) & fast_down
    data["long_signal"] = long_entry.fillna(False)
    data["short_signal"] = short_entry.fillna(False)
    data["exit_long"] = (fast_down if exit_rule == "Fast flips" else slow_down).fillna(False)
    data["exit_short"] = (fast_up if exit_rule == "Fast flips" else slow_up).fillna(False)
    return data


def backtest_symbol(symbol: str, frame: pd.DataFrame, start: date, end: date, notional: float,
                    fee_rate: float, fast_atr: int, fast_factor: float, slow_atr: int,
                    slow_factor: float, entry_rule: str, exit_rule: str,
                    allow_shorts: bool) -> list[dict]:
    data = build_signals(frame, fast_atr, fast_factor, slow_atr, slow_factor, entry_rule, exit_rule)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_exclusive = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    position: Optional[dict] = None
    pending: Optional[str] = None
    trades = []

    for i in range(len(data)):
        row = data.iloc[i]
        # Execute orders queued by the prior candle's confirmed close at this candle's open.
        if pending is not None:
            if position is not None:
                exit_price = float(row["open"])
                quantity = notional / position["entry_price"]
                gross = (exit_price - position["entry_price"]) * quantity * position["side"]
                fees = fee_rate * (notional + quantity * exit_price)
                trades.append({"symbol": symbol, "side": "Long" if position["side"] == 1 else "Short",
                    "entry_time": position["entry_time"], "exit_time": row["timestamp"],
                    "entry_price": position["entry_price"], "exit_price": exit_price,
                    "notional_usd": notional, "gross_pnl_usd": gross, "fees_usd": fees,
                    "net_pnl_usd": gross - fees, "return_pct": (gross - fees) / notional * 100})
                position = None
            if pending in ("long", "short") and (pending == "long" or allow_shorts):
                position = {"side": 1 if pending == "long" else -1,
                            "entry_price": float(row["open"]), "entry_time": row["timestamp"]}
            pending = None

        # No orders are opened after the selected test window; last eligible open is end-day.
        if row["timestamp"] < start_ts or row["timestamp"] >= end_exclusive:
            continue
        long_sig, short_sig = bool(row["long_signal"]), bool(row["short_signal"]) and allow_shorts
        if position is None:
            if long_sig:
                pending = "long"
            elif short_sig:
                pending = "short"
        elif position["side"] == 1 and (short_sig or bool(row["exit_long"])):
            pending = "short" if short_sig else "exit"
        elif position["side"] == -1 and (long_sig or bool(row["exit_short"])):
            pending = "long" if long_sig else "exit"

    # Close/mark any remaining position at the last close within the selected period.
    eligible = data[data["timestamp"] < end_exclusive]
    if position is not None and not eligible.empty:
        last = eligible.iloc[-1]
        quantity = notional / position["entry_price"]
        exit_price = float(last["close"])
        gross = (exit_price - position["entry_price"]) * quantity * position["side"]
        fees = fee_rate * (notional + quantity * exit_price)
        trades.append({"symbol": symbol, "side": "Long" if position["side"] == 1 else "Short",
            "entry_time": position["entry_time"], "exit_time": last["timestamp"],
            "entry_price": position["entry_price"], "exit_price": exit_price,
            "notional_usd": notional, "gross_pnl_usd": gross, "fees_usd": fees,
            "net_pnl_usd": gross - fees, "return_pct": (gross - fees) / notional * 100,
            "note": "Open position marked to final close"})
    return trades


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"Trades": 0, "Win rate": 0.0, "Net P&L ($)": 0.0, "Profit factor": 0.0, "Max drawdown ($)": 0.0}
    ordered = trades.sort_values(["exit_time", "symbol"]).copy()
    pnl = ordered["net_pnl_usd"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    equity = pnl.cumsum()
    drawdown = equity.cummax().combine(equity, max) - equity
    return {"Trades": int(len(ordered)), "Win rate": float((pnl > 0).mean() * 100),
        "Net P&L ($)": float(pnl.sum()),
        "Profit factor": float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf"),
        "Max drawdown ($)": float(drawdown.max()) if not drawdown.empty else 0.0,
        "Average trade ($)": float(pnl.mean()), "Gross profit ($)": float(wins.sum()),
        "Gross loss ($)": float(abs(losses.sum()))}


st.title("Tanmay Kurtkoti Double SuperTrend — Binance Spot Backtest")
st.caption("Binance Vision spot klines · no API keys · no live orders · fixed USD notional per trade")

with st.sidebar:
    st.header("Backtest settings")
    interval = st.selectbox("Candle timeframe", INTERVALS, index=3)
    start_date = st.date_input("Start date (UTC)", value=date(2024, 1, 1))
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    end_date = st.date_input("End date (UTC)", value=yesterday, max_value=yesterday)
    top_n = st.slider("Top USDT spot coins", min_value=1, max_value=50, value=50)
    notional = st.number_input("USD per trade", min_value=10.0, value=1000.0, step=100.0)
    fee_pct = st.number_input("Fee per side (%)", min_value=0.0, value=0.1, step=0.01, format="%.3f")
    st.subheader("SuperTrend parameters")
    fast_atr = st.number_input("Fast ATR length", min_value=1, value=10, step=1)
    fast_factor = st.number_input("Fast factor", min_value=0.1, value=1.5, step=0.1)
    slow_atr = st.number_input("Slow ATR length", min_value=1, value=10, step=1)
    slow_factor = st.number_input("Slow factor", min_value=0.1, value=3.0, step=0.1)
    entry_rule = st.selectbox("Entry rule", ["Both agree", "Fast line crosses slow line"])
    exit_rule = st.selectbox("Exit rule", ["Fast flips", "Slow flips"])
    allow_shorts = st.checkbox("Include shorts (not ordinary spot trading)", value=False)

st.info("The universe uses today's 24-hour USDT quote-volume ranking. Using today's Top 50 for past periods causes survivorship bias; this app does not reconstruct historical membership.")
if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

if st.button("Run Top-Coin Backtest", type="primary"):
    try:
        with st.spinner("Loading current top USDT spot pairs..."):
            universe = get_top_usdt_symbols(top_n)
        if universe.empty:
            st.error("Binance returned no eligible USDT spot pairs. Try again later.")
            st.stop()
        st.write("Coins selected by current 24h quote volume:", ", ".join(universe["symbol"].tolist()))
        all_trades, failed = [], []
        progress, status = st.progress(0), st.empty()
        for idx, symbol in enumerate(universe["symbol"], start=1):
            status.text(f"{idx}/{len(universe)} — loading {symbol} from Binance Vision")
            try:
                candles = fetch_vision_range(symbol, interval, start_date, end_date)
                if candles.empty:
                    failed.append(f"{symbol}: archive data unavailable for selected period")
                else:
                    all_trades.extend(backtest_symbol(symbol, candles, start_date, end_date, notional,
                        fee_pct / 100, int(fast_atr), float(fast_factor), int(slow_atr),
                        float(slow_factor), entry_rule, exit_rule, allow_shorts))
            except Exception as exc:
                failed.append(f"{symbol}: {exc}")
            progress.progress(idx / len(universe))
        status.empty()
        progress.empty()
        st.session_state["double_st_trades"] = pd.DataFrame(all_trades)
        st.session_state["double_st_failed"] = failed
    except requests.RequestException as exc:
        st.error(f"Could not reach Binance public market-data services: {exc}")
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")

if "double_st_trades" in st.session_state:
    trades_df = st.session_state["double_st_trades"]
    failed = st.session_state.get("double_st_failed", [])
    if failed:
        with st.expander(f"Data warnings ({len(failed)})"):
            st.write("\n".join(failed))
    if trades_df.empty:
        st.warning("No trades found for the selected settings and data range.")
    else:
        report = summarize(trades_df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades", f"{report['Trades']:,}")
        c2.metric("Win rate", f"{report['Win rate']:.2f}%")
        c3.metric("Net P&L", f"${report['Net P&L ($)']:,.2f}")
        c4.metric("Profit factor", "∞" if np.isinf(report["Profit factor"]) else f"{report['Profit factor']:.2f}")
        c5.metric("Max drawdown", f"${report['Max drawdown ($)']:,.2f}")
        st.subheader("Equity (realized P&L; $1,000 fixed notional per entry by default)")
        equity = trades_df.sort_values(["exit_time", "symbol"]).set_index("exit_time")["net_pnl_usd"].cumsum().rename("Cumulative net P&L ($)")
        st.line_chart(equity)
        st.subheader("Per-coin performance")
        per_coin = trades_df.groupby("symbol").agg(trades=("net_pnl_usd", "size"),
            net_pnl_usd=("net_pnl_usd", "sum"),
            win_rate_pct=("net_pnl_usd", lambda values: (values > 0).mean() * 100)).sort_values("net_pnl_usd", ascending=False)
        st.dataframe(per_coin, use_container_width=True)
        st.subheader("Trade log")
        st.dataframe(trades_df.sort_values("entry_time", ascending=False), use_container_width=True)
        st.download_button("Download trades CSV", trades_df.to_csv(index=False), "double_supertrend_trades.csv", "text/csv")
        st.caption("Fees apply on entry and exit. Slippage, taxes, minimum order sizes, and historical constituent changes are not modeled. Multiple symbols may each hold a $1,000 position concurrently. Results are hypothetical, not investment advice.")
