from __future__ import annotations

import argparse
import os

from trader_app.bot import run_bot
from trader_app.config import DEFAULT_SETTINGS, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a moving-average trading bot."
    )

    parser.add_argument("--exchange", default=DEFAULT_SETTINGS.exchange_id)
    parser.add_argument("--symbol", default=DEFAULT_SETTINGS.symbol)
    parser.add_argument("--timeframe", default=DEFAULT_SETTINGS.timeframe)
    parser.add_argument("--short-window", type=int, default=DEFAULT_SETTINGS.short_window)
    parser.add_argument("--long-window", type=int, default=DEFAULT_SETTINGS.long_window)

    parser.add_argument(
        "--order-amount",
        type=float,
        default=DEFAULT_SETTINGS.order_amount,
        help="Base-asset size to use for market buy/sell orders.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_SETTINGS.poll_seconds,
        help="Seconds to wait between signal checks. Use 0 to run once.",
    )
    parser.add_argument(
        "--order-book-depth",
        type=int,
        default=DEFAULT_SETTINGS.order_book_depth,
        help="Number of bid/ask levels to inspect when supervising an open trade.",
    )
    parser.add_argument(
        "--sell-pressure-ratio",
        type=float,
        default=DEFAULT_SETTINGS.sell_pressure_ratio,
        help="Sell if ask volume is at least this multiple of bid volume at the top of the order book.",
    )
    parser.add_argument(
        "--state-file",
        default=DEFAULT_SETTINGS.state_file,
        help="Path to the JSON file used to persist the bot's trade state across restarts.",
    )
    parser.add_argument(
        "--record-file",
        default=DEFAULT_SETTINGS.record_file,
        help="Append market snapshots and equity records to a CSV file during each cycle.",
    )
    parser.add_argument(
        "--max-hold",
        default=DEFAULT_SETTINGS.max_hold,
        help="Maximum time to hold a position before forcing a sell, for example 30m or 1h.",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=DEFAULT_SETTINGS.stop_loss,
        help="Stop-loss fraction for open positions, for example 0.01 for 1%%.",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=DEFAULT_SETTINGS.take_profit,
        help="Take-profit fraction for open positions, for example 0.02 for 2%%.",
    )

    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow short entries when the signal and order-book bias indicate a bearish trend.",
    )
    parser.add_argument(
        "--use-xgboost",
        action="store_true",
        help="Enable XGBoost-based model confirmation for entry signals.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Place live market orders. Without this flag the bot runs in dry-run mode.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use the exchange sandbox or testnet environment when supported by CCXT.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the exchange demo-trading environment. For Bybit this targets api-demo.",
    )

    parser.add_argument(
        "--use-atr-stops",
        action="store_true",
        help="Replace fixed SL/TP with ATR-based stop-loss and take-profit levels.",
    )
    parser.add_argument(
        "--atr-sl-multiplier",
        type=float,
        default=DEFAULT_SETTINGS.atr_sl_multiplier,
        help="ATR multiples for the stop-loss when --use-atr-stops is active (default: 2.0).",
    )
    parser.add_argument(
        "--atr-tp-multiplier",
        type=float,
        default=DEFAULT_SETTINGS.atr_tp_multiplier,
        help="ATR multiples for the take-profit when --use-atr-stops is active (default: 3.0).",
    )
    parser.add_argument(
        "--use-trailing-stop",
        action="store_true",
        help="Enable a trailing stop that ratchets behind price using ATR distance.",
    )
    parser.add_argument(
        "--trail-atr-multiplier",
        type=float,
        default=DEFAULT_SETTINGS.trail_atr_multiplier,
        help="ATR multiples for the trailing stop distance (default: 2.0).",
    )
    parser.add_argument(
        "--use-atr-sizing",
        action="store_true",
        help="Size each position so that one ATR of adverse move risks --atr-risk-pct of equity.",
    )
    parser.add_argument(
        "--atr-risk-pct",
        type=float,
        default=DEFAULT_SETTINGS.atr_risk_pct,
        help="Fraction of equity to risk per trade when --use-atr-sizing is active (default: 0.01).",
    )
    parser.add_argument(
        "--min-adx",
        type=float,
        default=DEFAULT_SETTINGS.min_adx,
        help="Minimum ADX trend strength required to enter a trade (0 = disabled, 25 recommended).",
    )
    parser.add_argument(
        "--rsi-filter",
        action="store_true",
        help="Block BUY entries when RSI > 70 and SELL entries when RSI < 30.",
    )
    parser.add_argument(
        "--loss-cooldown",
        type=int,
        default=DEFAULT_SETTINGS.loss_cooldown,
        help="Seconds to pause new entries after a losing trade (0 = disabled).",
    )

    return parser


def parse_settings() -> Settings:
    args = build_parser().parse_args()
    return Settings(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        short_window=args.short_window,
        long_window=args.long_window,
        api_key=os.getenv("TRADER_API_KEY"),
        api_secret=os.getenv("TRADER_API_SECRET"),
        api_password=os.getenv("TRADER_API_PASSWORD"),
        order_amount=args.order_amount,
        execute_orders=args.execute,
        sandbox=args.sandbox,
        demo=args.demo,
        allow_short=args.allow_short,
        use_xgboost=args.use_xgboost,
        poll_seconds=args.poll_seconds,
        order_book_depth=args.order_book_depth,
        sell_pressure_ratio=args.sell_pressure_ratio,
        state_file=args.state_file,
        record_file=args.record_file,
        max_hold=args.max_hold,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        use_atr_stops=args.use_atr_stops,
        atr_sl_multiplier=args.atr_sl_multiplier,
        atr_tp_multiplier=args.atr_tp_multiplier,
        use_trailing_stop=args.use_trailing_stop,
        trail_atr_multiplier=args.trail_atr_multiplier,
        use_atr_sizing=args.use_atr_sizing,
        atr_risk_pct=args.atr_risk_pct,
        min_adx=args.min_adx,
        rsi_filter=args.rsi_filter,
        loss_cooldown=args.loss_cooldown,
    )


def main() -> int:
    settings = parse_settings()
    return run_bot(settings)
