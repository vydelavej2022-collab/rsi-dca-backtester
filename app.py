import io
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class Params:
    rsi_length: int = 14
    overbought: float = 70.0
    oversold: float = 30.0
    max_adds: int = 2
    step: float = 50.0
    max_sl_move: float = 0.0


def read_mt_csv(file_bytes: bytes) -> pd.DataFrame:
    raw = file_bytes.decode("utf-8", errors="ignore")
    sep = "\t" if "\t" in raw.splitlines()[0] else ","
    df = pd.read_csv(io.StringIO(raw), sep=sep, engine="python")
    if df.shape[1] == 1:
        df = pd.read_csv(io.StringIO(raw), sep=",", engine="python")

    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        raise ValueError("CSV must contain either (date+time) or datetime/time column.")

    def pick(name: str) -> str:
        if name in df.columns:
            return name
        for c in df.columns:
            if name in c:
                return c
        raise ValueError(f"Missing column: {name} (need open/high/low/close).")

    o = pick("open")
    h = pick("high")
    l = pick("low")
    c = pick("close")

    out = df[["datetime", o, h, l, c]].copy()
    out.columns = ["datetime", "open", "high", "low", "close"]

    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
    return out


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rma_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    rma_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = rma_gain / rma_loss
    return 100 - (100 / (1 + rs))


def backtest(df: pd.DataFrame, p: Params) -> Tuple[pd.Series, dict]:
    df = df.copy()
    df["rsi"] = rsi_wilder(df["close"], p.rsi_length)

    allowed_trades = 1 + int(p.max_adds)
    reached_overbought = False
    reached_oversold = False
    last_signal = 0

    side = 0
    fills: List[float] = []
    last_add_price_long = np.nan
    last_add_price_short = np.nan
    last_add_bar = None

    realized = 0.0
    equity = []
    times = []
    trade_pnls = []

    def pos_avg() -> float:
        return float(np.mean(fills)) if fills else np.nan

    def pos_units() -> int:
        return len(fills)

    def floating(mark: float) -> float:
        if side == 0 or not fills:
            return 0.0
        return side * pos_units() * (mark - pos_avg())

    for i in range(1, len(df) - 1):
        rsi_now = df.loc[i, "rsi"]
        rsi_prev = df.loc[i - 1, "rsi"]
        close_i = float(df.loc[i, "close"])
        next_open = float(df.loc[i + 1, "open"])
        t = df.loc[i, "datetime"]

        if side == 0:
            last_add_price_long = np.nan
            last_add_price_short = np.nan
            last_add_bar = None

        if not np.isnan(rsi_now):
            if rsi_now >= p.overbought:
                reached_overbought = True
            if rsi_now <= p.oversold:
                reached_oversold = True

        rsi_below_overbought = (rsi_prev > p.overbought) and (rsi_now <= p.overbought)
        rsi_above_oversold = (rsi_prev < p.oversold) and (rsi_now >= p.oversold)

        buy_signal = reached_oversold and rsi_above_oversold and (last_signal != 1)
        sell_signal = reached_overbought and rsi_below_overbought and (last_signal != -1)

        if side != 0 and p.max_sl_move > 0 and fills:
            avg = pos_avg()
            sl_hit = (side == 1 and close_i <= avg - p.max_sl_move) or (side == -1 and close_i >= avg + p.max_sl_move)
            if sl_hit:
                pnl = side * pos_units() * (next_open - avg)
                realized += pnl
                trade_pnls.append(pnl)
                side = 0
                fills = []
                reached_overbought = False
                reached_oversold = False
                equity.append(realized)
                times.append(t)
                continue

        if buy_signal:
            if side == -1 and fills:
                avg = pos_avg()
                pnl = side * pos_units() * (next_open - avg)
                realized += pnl
                trade_pnls.append(pnl)
                side = 0
                fills = []
                last_add_price_short = np.nan
                last_add_bar = None
            if side == 0:
                side = 1
                fills = [next_open]
                last_signal = 1
                reached_oversold = False
                reached_overbought = False
                last_add_price_long = close_i

        if sell_signal:
            if side == 1 and fills:
                avg = pos_avg()
                pnl = side * pos_units() * (next_open - avg)
                realized += pnl
                trade_pnls.append(pnl)
                side = 0
                fills = []
                last_add_price_long = np.nan
                last_add_bar = None
            if side == 0:
                side = -1
                fills = [next_open]
                last_signal = -1
                reached_oversold = False
                reached_overbought = False
                last_add_price_short = close_i

        if side != 0 and p.step > 0 and fills and (pos_units() < allowed_trades):
            if last_add_bar is None or last_add_bar != i:
                if side == 1 and not np.isnan(last_add_price_long) and close_i <= last_add_price_long - p.step:
                    fills.append(next_open)
                    last_add_price_long = close_i
                    last_add_bar = i
                elif side == -1 and not np.isnan(last_add_price_short) and close_i >= last_add_price_short + p.step:
                    fills.append(next_open)
                    last_add_price_short = close_i
                    last_add_bar = i

        equity.append(realized + floating(close_i))
        times.append(t)

    eq = pd.Series(equity, index=pd.to_datetime(times), name="equity")
    dd = eq.cummax() - eq

    trade_pnls_arr = np.array(trade_pnls, dtype=float)
    n_trades = int((trade_pnls_arr != 0).sum())
    wins = int((trade_pnls_arr > 0).sum())
    losses = int((trade_pnls_arr < 0).sum())
    gross_profit = float(trade_pnls_arr[trade_pnls_arr > 0].sum()) if n_trades else 0.0
    gross_loss = float(-trade_pnls_arr[trade_pnls_arr < 0].sum()) if n_trades else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    stats = {
        "bars": len(df),
        "final_pnl_price_units": float(eq.iloc[-1]) if len(eq) else 0.0,
        "max_drawdown_price_units": float(dd.max()) if len(dd) else 0.0,
        "trades_closed_cycles": n_trades,
        "winrate_%": (wins / n_trades * 100.0) if n_trades else 0.0,
        "profit_factor": pf,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }
    return eq, stats


# -------------------- UI --------------------

st.set_page_config(page_title="RSI + DCA Backtester", layout="wide")
st.title("RSI + DCA + MaxSL Backtester")
st.caption("Nahraj CSV, uprav parametry a dostaneš výsledky + equity křivku (fill na next bar open).")

uploaded = st.file_uploader("Nahraj CSV (MT4/MT5 export nebo standardní OHLC CSV)", type=["csv", "txt"])

with st.sidebar:
    st.header("Parametry")
    rsi_length = st.number_input("RSI length", 2, 200, 14, 1)
    overbought = st.number_input("Overbought", 1.0, 99.0, 70.0, 1.0)
    oversold = st.number_input("Oversold", 1.0, 99.0, 30.0, 1.0)
    max_adds = st.selectbox("maxAdds", [0, 1, 2], index=2)
    step = st.number_input("DCA step (price units)", 0.0, 1e9, 50.0, 1.0)
    max_sl = st.number_input("Max SL move (price units, 0=OFF)", 0.0, 1e9, 0.0, 1.0)
    run = st.button("Spočítat", type="primary")

if uploaded is None:
    st.info("Nahraj CSV a klikni **Spočítat**.")
    st.stop()

try:
    df = read_mt_csv(uploaded.getvalue())
except Exception as e:
    st.error(f"Chyba při načítání CSV: {e}")
    st.stop()

st.subheader("Preview dat")
st.write(df.head(10))
st.write(f"Rows: {len(df):,} | Range: {df['datetime'].min()} → {df['datetime'].max()}")

if run:
    p = Params(
        rsi_length=int(rsi_length),
        overbought=float(overbought),
        oversold=float(oversold),
        max_adds=int(max_adds),
        step=float(step),
        max_sl_move=float(max_sl),
    )

    eq, stats = backtest(df, p)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Výsledky")
        st.json({"stats": stats, "params": p.__dict__})

        out_df = eq.reset_index()
        out_df.columns = ["datetime", "equity"]
        st.download_button(
            "Stáhnout equity_curve.csv",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="equity_curve.csv",
            mime="text/csv",
        )

    with c2:
        st.subheader("Equity křivka")
        st.line_chart(eq)
