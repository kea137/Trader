from __future__ import annotations

import argparse
import os

from trader_app.config import DEFAULT_SETTINGS, Settings
from trader_app.bot import run_bot


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
        max_hold=args.max_hold,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )


def main() -> int:
    settings = parse_settings()
    return run_bot(settings)
