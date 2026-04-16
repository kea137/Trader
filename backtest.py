#!/usr/bin/env python3
"""
PROBOT Backtester — full-cost simulation on historical OHLCV data.

Accounts for:
  • Taker fees (market orders)
  • Maker fees (limit orders, if applicable)
  • Estimated slippage (configurable)
  • ATR-based stops, trailing stops, confluence gate, volume penalty
  • Signal-flip, sell-pressure, momentum exit logic (same as live bot)

Usage:
    python backtest.py                        # BTC/USDT 4h, conservative defaults
    python backtest.py --symbol ETH/USDT --timeframe 1h --days 180
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Reuse the strategy module directly
from trader_app.strategy import (
    add_moving_averages,
    latest_signal,
    compute_atr,
    compute_atr_stops,
    compute_confluence_score,
    compute_rsi,
    compute_adx,
    compute_bollinger_bands,
    compute_macd,
    compute_trailing_stop,
    compute_volatility_position_size,
    compute_price_position,
    has_volume_confirmation,
)


# ─── Fee / slippage model ────────────────────────────────────────────────────

@dataclass
class FeeModel:
    """Exchange fee schedule.  Bybit unified: taker=0.055%, maker=0.02%"""
    taker_pct: float = 0.00055       # 0.055 %
    maker_pct: float = 0.0002        # 0.02 %
    slippage_pct: float = 0.0001     # 0.01 % estimated market impact

    def market_buy_price(self, mid: float) -> float:
        """Price paid when buying at market (mid + slippage)."""
        return mid * (1 + self.slippage_pct)

    def market_sell_price(self, mid: float) -> float:
        """Price received when selling at market (mid − slippage)."""
        return mid * (1 - self.slippage_pct)

    def entry_cost(self, price: float, amount: float) -> float:
        """Total cost of a market entry including taker fee."""
        return price * amount * (1 + self.taker_pct)

    def exit_proceeds(self, price: float, amount: float) -> float:
        """Net proceeds of a market exit after taker fee."""
        return price * amount * (1 - self.taker_pct)


# ─── Trade record ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time: Any
    exit_time: Any
    side: str
    entry_price: float
    exit_price: float
    amount: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float
    exit_reason: str
    hold_bars: int


# ─── Simulated snapshot (lightweight stand-in for MarketSnapshot) ─────────────

@dataclass
class SimSnapshot:
    signal: str
    latest_close: float
    long_ma: float
    price_position: float | None
    momentum: float | None
    long_ma_slope: float | None
    macd_histogram: float | None
    confluence_score: int
    volume_confirmed: bool
    order_book_bias: str = "NEUTRAL"
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None
    ml_bias: str | None = None
    spread: float | None = None
    volatility: float | None = None


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    short_window: int = 20
    long_window: int = 50
    order_amount: float = 0.001
    initial_equity: float = 10_000.0   # quote currency
    # Strategy
    confluence_threshold: int = 3
    min_adx: float = 25.0
    rsi_filter: bool = True
    volume_confirmation: bool = True
    use_atr_stops: bool = True
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0
    use_trailing_stop: bool = True
    trail_atr_multiplier: float = 2.0
    use_atr_sizing: bool = True
    atr_risk_pct: float = 0.01
    allow_short: bool = False
    stop_loss: float = 0.01
    take_profit: float = 0.02
    min_hold_bars: int = 3          # minimum bars before momentum/signal-flip exits
    # Fees
    fee_model: FeeModel = field(default_factory=FeeModel)
    # Data
    days: int = 365
    exchange_id: str = "bybit"


# ─── Backtest engine ─────────────────────────────────────────────────────────

def fetch_historical(cfg: BacktestConfig) -> pd.DataFrame:
    """Download OHLCV candles from the exchange via CCXT."""
    import ccxt
    exchange_class = getattr(ccxt, cfg.exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    tf_ms = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }
    candle_ms = tf_ms.get(cfg.timeframe, 14_400_000)
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - cfg.days * 86_400_000

    all_bars: list[list] = []
    cursor = since_ms
    limit = 1000

    print(f"  Fetching {cfg.symbol} {cfg.timeframe} candles ({cfg.days} days)…")
    while cursor < now_ms:
        bars = exchange.fetch_ohlcv(cfg.symbol, cfg.timeframe, since=cursor, limit=limit)
        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + candle_ms
        time.sleep(exchange.rateLimit / 1000)

    if not all_bars:
        raise RuntimeError("No OHLCV data returned from exchange")

    df = pd.DataFrame(all_bars, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    print(f"  Fetched {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    return df


def _build_snapshot(frame: pd.DataFrame, idx: int, cfg: BacktestConfig) -> SimSnapshot | None:
    """Build a SimSnapshot from historical candles up to `idx`."""
    window_needed = cfg.long_window + 50
    if idx < window_needed:
        return None

    sub = frame.iloc[idx - window_needed:idx + 1].copy().reset_index(drop=True)
    analyzed = add_moving_averages(sub, cfg.short_window, cfg.long_window)

    try:
        signal = latest_signal(analyzed)
    except ValueError:
        return None

    close = float(analyzed.iloc[-1]["close"])
    long_ma = float(analyzed.iloc[-1]["ma_long"])

    # Price position
    _, _, price_pos = compute_price_position(sub, lookback=20)

    # Momentum
    prev_close = float(sub.iloc[-2]["close"]) if len(sub) >= 2 else close
    momentum = close - prev_close

    # Long MA slope
    long_ma_slope = None
    if len(analyzed) >= 3:
        long_ma_slope = float(analyzed["ma_long"].iloc[-1] - analyzed["ma_long"].iloc[-3])

    # MACD histogram
    _, _, hist = compute_macd(sub["close"])
    macd_hist = float(hist.iloc[-1])

    # Confluence
    confluence = compute_confluence_score(sub, signal)

    # Volume
    vol_confirmed = has_volume_confirmation(sub)

    return SimSnapshot(
        signal=signal,
        latest_close=close,
        long_ma=long_ma,
        price_position=price_pos,
        momentum=momentum,
        long_ma_slope=long_ma_slope,
        macd_histogram=macd_hist,
        confluence_score=confluence,
        volume_confirmed=vol_confirmed,
    )


def _should_enter(snap: SimSnapshot, cfg: BacktestConfig) -> tuple[bool, str]:
    """Replicates bot.should_enter_position logic in backtest context."""
    raw_score = snap.confluence_score

    # Volume penalty
    if cfg.volume_confirmation and not snap.volume_confirmed:
        raw_score = max(0, raw_score - 1)

    # Confluence gate
    if cfg.confluence_threshold > 0 and raw_score < cfg.confluence_threshold:
        return False, f"low_confluence_{raw_score}"

    if snap.signal == "BUY":
        if snap.long_ma_slope is not None and snap.long_ma_slope < 0:
            return False, "trend_down"
        if snap.price_position is not None and snap.price_position >= 0.92:
            return False, "price_too_high"
        if snap.macd_histogram is not None and snap.macd_histogram < 0:
            return False, "macd_bearish"
        return True, "entry"

    if snap.signal == "SELL":
        if not cfg.allow_short:
            return False, "shorts_disabled"
        if snap.long_ma_slope is not None and snap.long_ma_slope > 0:
            return False, "trend_up"
        if snap.price_position is not None and snap.price_position <= 0.08:
            return False, "price_too_low"
        if snap.macd_histogram is not None and snap.macd_histogram > 0:
            return False, "macd_bullish"
        return True, "entry"

    return False, "no_signal"


def run_backtest(cfg: BacktestConfig, df: pd.DataFrame) -> tuple[list[Trade], pd.DataFrame]:
    """Run the full backtest simulation, returning trades and equity curve."""
    fees = cfg.fee_model
    trades: list[Trade] = []
    equity = cfg.initial_equity
    equity_curve: list[dict] = []

    # Position state
    in_position = False
    entry_signal: str = ""
    entry_price: float = 0.0
    entry_amount: float = 0.0
    entry_cost_total: float = 0.0  # cost including fees
    entry_bar: int = 0
    # ATR-based stops
    sl_price: float = 0.0
    tp_price: float = 0.0
    trailing_extreme: float = 0.0

    start_idx = cfg.long_window + 50
    total_bars = len(df)

    print(f"  Simulating {total_bars - start_idx} bars…")

    for idx in range(start_idx, total_bars):
        snap = _build_snapshot(df, idx, cfg)
        if snap is None:
            continue

        bar_time = df.iloc[idx]["time"]
        close = snap.latest_close

        # ── Manage open position ──────────────────────────────────────────
        if in_position:
            exit_now = False
            reason = "hold"

            if entry_signal == "BUY":
                # Fixed SL/TP check (always active as floor)
                if close <= entry_price * (1 - cfg.stop_loss):
                    exit_now, reason = True, "stop_loss_fixed"
                elif close >= entry_price * (1 + cfg.take_profit):
                    exit_now, reason = True, "take_profit_fixed"

                # ATR-based stops (override fixed if active)
                if cfg.use_atr_stops:
                    if close <= sl_price:
                        exit_now, reason = True, "stop_loss_atr"
                    elif close >= tp_price:
                        exit_now, reason = True, "take_profit_atr"

                # Trailing stop
                if cfg.use_trailing_stop and cfg.use_atr_stops:
                    if close > trailing_extreme:
                        trailing_extreme = close
                    sub = df.iloc[max(0, idx - 14):idx + 1]
                    atr = float(compute_atr(sub).iloc[-1]) if len(sub) >= 14 else 0.0
                    if atr > 0:
                        ts = compute_trailing_stop(trailing_extreme, atr, cfg.trail_atr_multiplier)
                        if close <= ts:
                            exit_now, reason = True, "trailing_stop"

                # Signal-flip exit
                if not exit_now and snap.signal == "SELL":
                    if idx - entry_bar >= cfg.min_hold_bars:
                        exit_now, reason = True, "signal_flip"

                # Momentum exit
                if not exit_now and snap.momentum is not None and snap.momentum < 0:
                    if idx - entry_bar >= cfg.min_hold_bars:
                        exit_now, reason = True, "negative_momentum"

            elif entry_signal == "SELL":
                # Short: Fixed SL/TP
                if close >= entry_price * (1 + cfg.stop_loss):
                    exit_now, reason = True, "stop_loss_fixed"
                elif close <= entry_price * (1 - cfg.take_profit):
                    exit_now, reason = True, "take_profit_fixed"

                if cfg.use_atr_stops:
                    if close >= sl_price:
                        exit_now, reason = True, "stop_loss_atr"
                    elif close <= tp_price:
                        exit_now, reason = True, "take_profit_atr"

                if cfg.use_trailing_stop and cfg.use_atr_stops:
                    if close < trailing_extreme:
                        trailing_extreme = close
                    sub = df.iloc[max(0, idx - 14):idx + 1]
                    atr = float(compute_atr(sub).iloc[-1]) if len(sub) >= 14 else 0.0
                    if atr > 0:
                        ts = compute_trailing_stop(trailing_extreme, atr, cfg.trail_atr_multiplier, is_short=True)
                        if close >= ts:
                            exit_now, reason = True, "trailing_stop"

                if not exit_now and snap.signal == "BUY":
                    if idx - entry_bar >= cfg.min_hold_bars:
                        exit_now, reason = True, "signal_flip"

            if exit_now:
                # Execute exit with slippage + fees
                if entry_signal == "BUY":
                    exit_price = fees.market_sell_price(close)
                    gross_pnl = (exit_price - entry_price) * entry_amount
                else:
                    exit_price = fees.market_buy_price(close)
                    gross_pnl = (entry_price - exit_price) * entry_amount

                exit_proceeds = fees.exit_proceeds(exit_price, entry_amount)
                entry_fee = entry_cost_total - (entry_price * entry_amount)
                exit_fee = (exit_price * entry_amount) * fees.taker_pct
                total_fees = entry_fee + exit_fee

                slippage_entry = abs(entry_price - close) * entry_amount * fees.slippage_pct if entry_price else 0
                slippage_exit = close * entry_amount * fees.slippage_pct
                total_slippage = slippage_entry + slippage_exit

                net_pnl = gross_pnl - total_fees - total_slippage

                equity += net_pnl
                trades.append(Trade(
                    entry_time=df.iloc[entry_bar]["time"],
                    exit_time=bar_time,
                    side=entry_signal,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    amount=entry_amount,
                    gross_pnl=gross_pnl,
                    fees=total_fees,
                    slippage_cost=total_slippage,
                    net_pnl=net_pnl,
                    exit_reason=reason,
                    hold_bars=idx - entry_bar,
                ))
                in_position = False

        # ── Try to enter a new position ───────────────────────────────────
        if not in_position:
            enter, _ = _should_enter(snap, cfg)
            if enter:
                # Compute ATR for stops and sizing
                sub = df.iloc[max(0, idx - 14):idx + 1]
                atr = float(compute_atr(sub).iloc[-1]) if len(sub) >= 14 else 0.0

                # Position sizing
                if cfg.use_atr_sizing and atr > 0:
                    amount = compute_volatility_position_size(
                        equity, close, atr, cfg.atr_risk_pct,
                    )
                else:
                    amount = cfg.order_amount

                # Can we afford it?
                est_cost = close * amount * (1 + fees.taker_pct + fees.slippage_pct)
                if est_cost > equity * 0.95:  # never use >95% of equity
                    amount = (equity * 0.95) / (close * (1 + fees.taker_pct + fees.slippage_pct))

                if amount <= 0:
                    continue

                is_short = snap.signal == "SELL"
                if is_short:
                    entry_price = fees.market_sell_price(close)
                else:
                    entry_price = fees.market_buy_price(close)

                entry_cost_total = fees.entry_cost(entry_price, amount)
                entry_amount = amount
                entry_signal = snap.signal
                entry_bar = idx
                in_position = True

                # Set stops
                if cfg.use_atr_stops and atr > 0:
                    sl_price, tp_price = compute_atr_stops(
                        entry_price, atr, cfg.atr_sl_multiplier, cfg.atr_tp_multiplier, is_short,
                    )
                else:
                    if is_short:
                        sl_price = entry_price * (1 + cfg.stop_loss)
                        tp_price = entry_price * (1 - cfg.take_profit)
                    else:
                        sl_price = entry_price * (1 - cfg.stop_loss)
                        tp_price = entry_price * (1 + cfg.take_profit)

                trailing_extreme = entry_price

        equity_curve.append({"time": bar_time, "equity": equity, "close": close})

    # Force-close any open position at end
    if in_position:
        close = float(df.iloc[-1]["close"])
        if entry_signal == "BUY":
            exit_price = fees.market_sell_price(close)
            gross_pnl = (exit_price - entry_price) * entry_amount
        else:
            exit_price = fees.market_buy_price(close)
            gross_pnl = (entry_price - exit_price) * entry_amount

        exit_fee = (exit_price * entry_amount) * fees.taker_pct
        entry_fee = entry_cost_total - (entry_price * entry_amount)
        total_fees = entry_fee + exit_fee
        total_slippage = close * entry_amount * fees.slippage_pct * 2
        net_pnl = gross_pnl - total_fees - total_slippage
        equity += net_pnl
        trades.append(Trade(
            entry_time=df.iloc[entry_bar]["time"],
            exit_time=df.iloc[-1]["time"],
            side=entry_signal,
            entry_price=entry_price,
            exit_price=exit_price,
            amount=entry_amount,
            gross_pnl=gross_pnl,
            fees=total_fees,
            slippage_cost=total_slippage,
            net_pnl=net_pnl,
            exit_reason="force_close",
            hold_bars=len(df) - 1 - entry_bar,
        ))

    eq_df = pd.DataFrame(equity_curve)
    return trades, eq_df


# ─── Analysis & reporting ────────────────────────────────────────────────────

def analyze(trades: list[Trade], eq_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """Compute performance metrics."""
    if not trades:
        return {"error": "No trades executed"}

    net_pnls = [t.net_pnl for t in trades]
    gross_pnls = [t.gross_pnl for t in trades]
    all_fees = [t.fees for t in trades]
    all_slip = [t.slippage_cost for t in trades]

    winners = [p for p in net_pnls if p > 0]
    losers = [p for p in net_pnls if p <= 0]

    total_net = sum(net_pnls)
    total_gross = sum(gross_pnls)
    total_fees = sum(all_fees)
    total_slippage = sum(all_slip)

    win_rate = len(winners) / len(trades) * 100 if trades else 0
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    profit_factor = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float("inf")
    expectancy = np.mean(net_pnls) if net_pnls else 0

    # Max drawdown from equity curve
    peak = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - peak) / peak
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0

    # Sharpe (annualized, assuming ~6 trades per day for 4h candles)
    if len(net_pnls) > 1:
        returns = np.array(net_pnls) / cfg.initial_equity
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(365)
    else:
        sharpe = 0

    # Hold-time analysis
    hold_bars = [t.hold_bars for t in trades]
    avg_hold = np.mean(hold_bars) if hold_bars else 0

    # Exit reason breakdown
    exit_reasons: dict[str, int] = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "total_trades": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate_pct": round(win_rate, 2),
        "total_net_pnl": round(total_net, 4),
        "total_gross_pnl": round(total_gross, 4),
        "total_fees_paid": round(total_fees, 4),
        "total_slippage_cost": round(total_slippage, 4),
        "fees_pct_of_gross": round(total_fees / abs(total_gross) * 100, 2) if total_gross else 0,
        "avg_win": round(float(avg_win), 4),
        "avg_loss": round(float(avg_loss), 4),
        "profit_factor": round(profit_factor, 3),
        "expectancy_per_trade": round(float(expectancy), 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "avg_hold_bars": round(float(avg_hold), 1),
        "roi_pct": round(total_net / cfg.initial_equity * 100, 2),
        "exit_reasons": exit_reasons,
    }


def print_report(metrics: dict, cfg: BacktestConfig, trades: list[Trade]) -> None:
    """Pretty-print backtest results."""
    B = "\033[1m"
    R = "\033[0m"
    G = "\033[32m"
    RD = "\033[31m"
    C = "\033[36m"
    Y = "\033[33m"

    print(f"\n{B}{'═' * 70}{R}")
    print(f"{B}  PROBOT BACKTEST REPORT{R}")
    print(f"{B}{'═' * 70}{R}")
    print(f"  Symbol:      {C}{cfg.symbol}{R}  |  Timeframe: {C}{cfg.timeframe}{R}  |  Days: {C}{cfg.days}{R}")
    print(f"  Windows:     short={cfg.short_window}  long={cfg.long_window}")
    print(f"  Confluence:  threshold={cfg.confluence_threshold}  ADX≥{cfg.min_adx}  RSI={cfg.rsi_filter}")
    print(f"  ATR stops:   SL={cfg.atr_sl_multiplier}x  TP={cfg.atr_tp_multiplier}x  trail={cfg.trail_atr_multiplier}x")
    print(f"  Fees:        taker={cfg.fee_model.taker_pct*100:.3f}%  slippage={cfg.fee_model.slippage_pct*100:.3f}%")
    print(f"  Initial:     ${cfg.initial_equity:,.2f}")
    print(f"{B}{'─' * 70}{R}")

    pnl = metrics["total_net_pnl"]
    pnl_c = G if pnl >= 0 else RD
    roi = metrics["roi_pct"]
    roi_c = G if roi >= 0 else RD

    print(f"\n  {B}PERFORMANCE{R}")
    print(f"  Total trades:     {metrics['total_trades']}")
    print(f"  Winners / Losers: {G}{metrics['winners']}{R} / {RD}{metrics['losers']}{R}")
    print(f"  Win rate:         {metrics['win_rate_pct']:.1f}%")
    print(f"  Profit factor:    {metrics['profit_factor']:.3f}")
    print(f"  Expectancy:       {pnl_c}${metrics['expectancy_per_trade']:.4f}{R} per trade")
    print(f"  Avg hold:         {metrics['avg_hold_bars']:.1f} bars")

    print(f"\n  {B}P&L BREAKDOWN{R}")
    print(f"  Gross P&L:        {pnl_c}${metrics['total_gross_pnl']:+,.4f}{R}")
    print(f"  Total fees:       {RD}-${metrics['total_fees_paid']:,.4f}{R}  ({metrics['fees_pct_of_gross']:.1f}% of gross)")
    print(f"  Total slippage:   {RD}-${metrics['total_slippage_cost']:,.4f}{R}")
    print(f"  {B}Net P&L:          {pnl_c}${pnl:+,.4f}{R}")
    print(f"  {B}ROI:              {roi_c}{roi:+.2f}%{R}")

    print(f"\n  {B}RISK{R}")
    print(f"  Max drawdown:     {RD}{metrics['max_drawdown_pct']:.2f}%{R}")
    print(f"  Sharpe ratio:     {metrics['sharpe_ratio']:.3f}")
    print(f"  Avg win:          {G}${metrics['avg_win']:+,.4f}{R}")
    print(f"  Avg loss:         {RD}${metrics['avg_loss']:+,.4f}{R}")

    print(f"\n  {B}EXIT REASONS{R}")
    for reason, count in sorted(metrics["exit_reasons"].items(), key=lambda x: -x[1]):
        pct = count / metrics["total_trades"] * 100
        print(f"    {reason:<25} {count:>4}  ({pct:5.1f}%)")

    # Show worst and best trades
    if trades:
        best = max(trades, key=lambda t: t.net_pnl)
        worst = min(trades, key=lambda t: t.net_pnl)
        print(f"\n  {B}NOTABLE TRADES{R}")
        print(f"  Best:   {G}${best.net_pnl:+,.4f}{R}  ({best.side} @ {best.entry_price:.2f} → {best.exit_price:.2f}, {best.hold_bars} bars, {best.exit_reason})")
        print(f"  Worst:  {RD}${worst.net_pnl:+,.4f}{R}  ({worst.side} @ {worst.entry_price:.2f} → {worst.exit_price:.2f}, {worst.hold_bars} bars, {worst.exit_reason})")

    print(f"\n{B}{'═' * 70}{R}\n")


# ─── Parameter sweep ─────────────────────────────────────────────────────────

def parameter_sweep(df: pd.DataFrame) -> None:
    """Run a grid search over key parameters and rank by Sharpe."""
    print("\n  Running parameter sweep…\n")

    configs = []
    for ct in [2, 3, 4]:
        for atr_sl in [1.5, 2.0, 3.0]:
            for atr_tp in [2.0, 3.0, 4.0]:
                for trail in [1.5, 2.0, 3.0]:
                    for min_hold in [1, 3, 6]:
                        c = BacktestConfig(
                            confluence_threshold=ct,
                            atr_sl_multiplier=atr_sl,
                            atr_tp_multiplier=atr_tp,
                            trail_atr_multiplier=trail,
                            min_hold_bars=min_hold,
                        )
                        configs.append(c)

    results: list[tuple[dict, BacktestConfig]] = []
    total = len(configs)
    for i, c in enumerate(configs):
        trades, eq_df = run_backtest(c, df)
        if not trades:
            continue
        m = analyze(trades, eq_df, c)
        results.append((m, c))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{total} configs tested…")

    # Rank by Sharpe, then by ROI
    results.sort(key=lambda x: (x[0].get("sharpe_ratio", 0), x[0].get("roi_pct", 0)), reverse=True)

    B = "\033[1m"
    R = "\033[0m"
    G = "\033[32m"
    C = "\033[36m"

    print(f"\n{B}  TOP 10 CONFIGURATIONS (by Sharpe){R}")
    print(f"  {'#':<4} {'CT':>3} {'SL':>5} {'TP':>5} {'Trail':>6} {'Hold':>5} {'Trades':>7} {'Win%':>6} {'ROI%':>8} {'MaxDD%':>7} {'Sharpe':>7} {'Fees$':>10}")
    print(f"  {'─'*4} {'─'*3} {'─'*5} {'─'*5} {'─'*6} {'─'*5} {'─'*7} {'─'*6} {'─'*8} {'─'*7} {'─'*7} {'─'*10}")

    for rank, (m, c) in enumerate(results[:10], 1):
        roi_s = f"{m['roi_pct']:+.2f}"
        print(
            f"  {rank:<4} {c.confluence_threshold:>3} "
            f"{c.atr_sl_multiplier:>5.1f} {c.atr_tp_multiplier:>5.1f} {c.trail_atr_multiplier:>6.1f} "
            f"{c.min_hold_bars:>5} "
            f"{m['total_trades']:>7} {m['win_rate_pct']:>5.1f}% {roi_s:>8} "
            f"{m['max_drawdown_pct']:>6.2f}% {m['sharpe_ratio']:>7.3f} "
            f"{m['total_fees_paid']:>10.4f}"
        )

    if results:
        best_m, best_c = results[0]
        print(f"\n  {B}RECOMMENDED CONFIG:{R}")
        print(f"    confluence_threshold = {G}{best_c.confluence_threshold}{R}")
        print(f"    atr_sl_multiplier    = {G}{best_c.atr_sl_multiplier}{R}")
        print(f"    atr_tp_multiplier    = {G}{best_c.atr_tp_multiplier}{R}")
        print(f"    trail_atr_multiplier = {G}{best_c.trail_atr_multiplier}{R}")
        print(f"    min_hold_bars        = {G}{best_c.min_hold_bars}{R}")
        print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="PROBOT backtester")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--short-window", type=int, default=20)
    parser.add_argument("--long-window", type=int, default=50)
    parser.add_argument("--initial-equity", type=float, default=10_000)
    parser.add_argument("--confluence", type=int, default=3)
    parser.add_argument("--min-adx", type=float, default=25.0)
    parser.add_argument("--atr-sl", type=float, default=2.0)
    parser.add_argument("--atr-tp", type=float, default=3.0)
    parser.add_argument("--trail", type=float, default=2.0)
    parser.add_argument("--min-hold", type=int, default=3, help="Minimum bars to hold before soft exits")
    parser.add_argument("--taker-fee", type=float, default=0.00055)
    parser.add_argument("--slippage", type=float, default=0.0001)
    parser.add_argument("--sweep", action="store_true", help="Run parameter optimization sweep")
    parser.add_argument("--exchange", default="bybit")
    args = parser.parse_args()

    cfg = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        short_window=args.short_window,
        long_window=args.long_window,
        initial_equity=args.initial_equity,
        confluence_threshold=args.confluence,
        min_adx=args.min_adx,
        atr_sl_multiplier=args.atr_sl,
        atr_tp_multiplier=args.atr_tp,
        trail_atr_multiplier=args.trail,
        min_hold_bars=args.min_hold,
        fee_model=FeeModel(taker_pct=args.taker_fee, slippage_pct=args.slippage),
        exchange_id=args.exchange,
    )

    df = fetch_historical(cfg)

    # Run default config
    trades, eq_df = run_backtest(cfg, df)
    metrics = analyze(trades, eq_df, cfg)
    print_report(metrics, cfg, trades)

    # Optional sweep
    if args.sweep:
        parameter_sweep(df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
