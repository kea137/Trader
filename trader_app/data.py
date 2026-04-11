from __future__ import annotations

from typing import Optional

import ccxt
import pandas as pd


OHLCV_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


def create_exchange(
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_password: Optional[str] = None,
    sandbox: bool = False,
    demo: bool = False,
):
    try:
        exchange_class = getattr(ccxt, exchange_id)
    except AttributeError as exc:
        raise ValueError(f"Unsupported exchange: {exchange_id}") from exc

    options = {"enableRateLimit": True}

    if api_key:
        options["apiKey"] = api_key
    if api_secret:
        options["secret"] = api_secret
    if api_password:
        options["password"] = api_password

    exchange = exchange_class(options)

    if sandbox and demo:
        raise ValueError("Choose only one environment mode: sandbox or demo.")

    if sandbox:
        try:
            exchange.set_sandbox_mode(True)
        except NotImplementedError as exc:
            raise ValueError(
                f"Sandbox mode is not supported for exchange: {exchange_id}"
            ) from exc
    elif demo:
        if exchange_id != "bybit":
            raise ValueError(f"Demo mode is not supported for exchange: {exchange_id}")
        try:
            exchange.enable_demo_trading(True)
        except AttributeError as exc:
            raise ValueError(
                f"Demo mode is not supported by the installed CCXT adapter for exchange: {exchange_id}"
            ) from exc

    return exchange


def fetch_ohlcv_frame(exchange, symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except TypeError:
        try:
            if limit is None:
                bars = exchange.fetch_ohlcv(symbol, timeframe)
            else:
                bars = exchange.fetch_ohlcv(symbol, timeframe, limit)
        except TypeError:
            bars = exchange.fetch_ohlcv(symbol, timeframe)

    if not bars:
        raise ValueError(
            f"No OHLCV data returned for {symbol} on {exchange.id} ({timeframe})."
        )

    frame = pd.DataFrame(bars, columns=OHLCV_COLUMNS)
    frame["time"] = pd.to_datetime(frame["time"], unit="ms", utc=True)

    return frame


def fetch_order_book(exchange, symbol: str, depth: int) -> dict:
    if depth <= 0:
        raise ValueError("order_book_depth must be a positive integer.")
    return exchange.fetch_order_book(symbol, limit=depth)
