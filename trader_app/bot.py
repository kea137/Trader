from __future__ import annotations

import json
import select
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ccxt

from trader_app.config import Settings
from trader_app.data import create_exchange, fetch_ohlcv_frame, fetch_order_book
from trader_app.strategy import add_moving_averages, latest_signal


@dataclass
class BotState:
    has_position: bool = False
    last_entry_signal: str | None = None
    entry_timestamp: float | None = None
    entry_price: float | None = None
    entry_amount: float | None = None
    entry_cost: float | None = None


@dataclass(frozen=True)
class OrderExecution:
    success: bool
    message: str
    filled_amount: float | None = None
    average_price: float | None = None
    cost: float | None = None


@dataclass(frozen=True)
class MarketSnapshot:
    signal: str
    bid_volume: float
    ask_volume: float
    order_book_bias: str
    latest_close: float
    best_bid: float | None
    best_ask: float | None
    long_ma: float
    ml_bias: str | None = None


@dataclass(frozen=True)
class CycleOutcome:
    message: str
    terminate: bool = False


def read_user_command() -> str | None:
    if not sys.stdin or sys.stdin.closed or not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    command = sys.stdin.readline()
    if not command:
        return None
    return command.strip().lower()


def load_state(state_file: str) -> BotState:
    path = Path(state_file)
    if not path.exists():
        return BotState()

    data = json.loads(path.read_text())
    return BotState(
        has_position=bool(data.get("has_position", False)),
        last_entry_signal=data.get("last_entry_signal"),
        entry_timestamp=data.get("entry_timestamp"),
        entry_price=data.get("entry_price"),
        entry_amount=data.get("entry_amount"),
        entry_cost=data.get("entry_cost"),
    )


def save_state(state_file: str, state: BotState) -> None:
    path = Path(state_file)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n")


def update_state(
    state: BotState,
    *,
    has_position: bool,
    last_entry_signal: str | None,
    entry_timestamp: float | None,
    entry_price: float | None,
    entry_amount: float | None,
    entry_cost: float | None,
) -> bool:
    changed = (
        state.has_position != has_position
        or state.last_entry_signal != last_entry_signal
        or state.entry_timestamp != entry_timestamp
        or state.entry_price != entry_price
        or state.entry_amount != entry_amount
        or state.entry_cost != entry_cost
    )
    state.has_position = has_position
    state.last_entry_signal = last_entry_signal
    state.entry_timestamp = entry_timestamp
    state.entry_price = entry_price
    state.entry_amount = entry_amount
    state.entry_cost = entry_cost
    return changed


def parse_duration(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip().lower()
    if not stripped:
        return None
    units = {"s": 1, "m": 60, "h": 3600}
    unit = stripped[-1]
    if unit not in units:
        raise ValueError("max_hold must end with s, m, or h, for example 30m or 1h.")
    amount = stripped[:-1]
    if not amount.isdigit():
        raise ValueError("max_hold must use an integer amount, for example 30m or 1h.")
    seconds = int(amount) * units[unit]
    if seconds <= 0:
        raise ValueError("max_hold must be greater than zero.")
    return seconds


def execute_trade(
    exchange: Any,
    symbol: str,
    signal: str,
    amount: float,
    live: bool,
    fallback_price: float | None = None,
) -> OrderExecution:
    side = signal.lower()

    if side not in {"buy", "sell"}:
        raise ValueError(f"Unsupported signal: {signal}")
    if amount <= 0:
        raise ValueError("order_amount must be a positive number.")

    if not live:
        cost = fallback_price * amount if fallback_price is not None else None
        return OrderExecution(
            success=True,
            message=f"DRY_RUN {signal} {amount} {symbol}",
            filled_amount=amount,
            average_price=fallback_price,
            cost=cost,
        )

    try:
        order = exchange.create_order(symbol=symbol, type="market", side=side, amount=amount)
    except ccxt.AuthenticationError:
        raise
    except ccxt.ExchangeError as exc:
        return OrderExecution(
            success=False,
            message=f"FAILED {signal} {amount} {symbol} {exc}",
        )

    order_id = order.get("id", "unknown")
    status = order.get("status", "open")
    filled_amount = order.get("filled")
    average_price = order.get("average") or order.get("price") or fallback_price
    cost = order.get("cost")
    if cost is None and average_price is not None and filled_amount is not None:
        cost = average_price * filled_amount
    if filled_amount is None:
        filled_amount = amount
    return OrderExecution(
        success=True,
        message=f"EXECUTED {signal} {amount} {symbol} order_id={order_id} status={status}",
        filled_amount=filled_amount,
        average_price=average_price,
        cost=cost,
    )


def execute_signal(exchange: Any, symbol: str, signal: str, amount: float, live: bool) -> str:
    return execute_trade(exchange, symbol, signal, amount, live).message


def format_realized_profit(state: BotState, exit_execution: OrderExecution) -> str:
    entry_cost = state.entry_cost
    exit_cost = exit_execution.cost
    if entry_cost is None and state.entry_price is not None and state.entry_amount is not None:
        entry_cost = state.entry_price * state.entry_amount
    if exit_cost is None and exit_execution.average_price is not None and exit_execution.filled_amount is not None:
        exit_cost = exit_execution.average_price * exit_execution.filled_amount
    if entry_cost is None or exit_cost is None:
        return "profit=unavailable"

    if state.last_entry_signal == "SELL":
        pnl = entry_cost - exit_cost
    else:
        pnl = exit_cost - entry_cost
    pnl_pct = 0.0 if entry_cost == 0 else (pnl / entry_cost) * 100
    return f"profit={pnl:.6f} quote_currency profit_pct={pnl_pct:.2f}%"


def summarize_order_book(order_book: dict, sell_pressure_ratio: float) -> tuple[float, float, str]:
    bid_volume = sum(level[1] for level in order_book.get("bids", []))
    ask_volume = sum(level[1] for level in order_book.get("asks", []))

    if bid_volume <= 0 and ask_volume <= 0:
        return bid_volume, ask_volume, "NEUTRAL"
    if ask_volume >= bid_volume * sell_pressure_ratio:
        return bid_volume, ask_volume, "SELL"
    if bid_volume >= ask_volume * sell_pressure_ratio:
        return bid_volume, ask_volume, "BUY"
    return bid_volume, ask_volume, "NEUTRAL"


def should_enter_position(snapshot: MarketSnapshot, allow_short: bool, use_xgboost: bool) -> tuple[bool, str]:
    if snapshot.signal == "BUY":
        if snapshot.order_book_bias != "BUY":
            return False, "order_book_conflict"
        if snapshot.latest_close <= snapshot.long_ma:
            return False, "price_below_long_ma"
        if use_xgboost and snapshot.ml_bias is not None and snapshot.ml_bias != "BUY":
            return False, "ml_conflict"
        return True, "signal_and_order_book"

    if snapshot.signal == "SELL":
        if not allow_short:
            return False, "shorts_disabled"
        if snapshot.order_book_bias != "SELL":
            return False, "order_book_conflict"
        if snapshot.latest_close >= snapshot.long_ma:
            return False, "price_above_long_ma"
        if use_xgboost and snapshot.ml_bias is not None and snapshot.ml_bias != "SELL":
            return False, "ml_conflict"
        return True, "signal_and_order_book"

    return False, "no_entry_signal"


def inspect_market(settings: Settings, exchange: Any) -> MarketSnapshot:
    frame = fetch_ohlcv_frame(
        exchange=exchange,
        symbol=settings.symbol,
        timeframe=settings.timeframe,
    )
    analyzed = add_moving_averages(
        frame=frame,
        short_window=settings.short_window,
        long_window=settings.long_window,
    )
    signal = latest_signal(analyzed)
    latest_close = float(analyzed.iloc[-1]["close"])
    long_ma = float(analyzed.iloc[-1]["ma_long"])
    ml_bias = None
    if settings.use_xgboost:
        from trader_app.strategy import compute_ml_bias

        try:
            ml_bias = compute_ml_bias(analyzed, settings.short_window, settings.long_window)
        except Exception:
            ml_bias = None
    order_book = fetch_order_book(
        exchange=exchange,
        symbol=settings.symbol,
        depth=settings.order_book_depth,
    )
    bid_volume, ask_volume, order_book_bias = summarize_order_book(
        order_book,
        settings.sell_pressure_ratio,
    )
    return MarketSnapshot(
        signal=signal,
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        order_book_bias=order_book_bias,
        latest_close=latest_close,
        best_bid=order_book.get("bids", [[None]])[0][0] if order_book.get("bids") else None,
        best_ask=order_book.get("asks", [[None]])[0][0] if order_book.get("asks") else None,
        long_ma=long_ma,
        ml_bias=ml_bias,
    )


def should_exit_position(
    snapshot: MarketSnapshot,
    state: BotState,
    max_hold_seconds: int | None,
    now: float,
) -> tuple[bool, str]:
    if (
        max_hold_seconds is not None
        and state.entry_timestamp is not None
        and now - state.entry_timestamp >= max_hold_seconds
    ):
        return True, "max_hold"
    entry_signal = state.last_entry_signal or "BUY"
    if entry_signal == "BUY":
        if snapshot.signal == "SELL":
            return True, "signal_flip"
        if snapshot.order_book_bias == "SELL":
            return True, "sell_pressure"
        return False, "hold"
    if entry_signal == "SELL":
        if snapshot.signal == "BUY":
            return True, "signal_flip"
        if snapshot.order_book_bias == "BUY":
            return True, "buy_pressure"
        return False, "hold"
    return False, "hold"


def liquidate_position(settings: Settings, exchange: Any, state: BotState, reason: str) -> CycleOutcome:
    snapshot = inspect_market(settings, exchange)
    close_amount = state.entry_amount or settings.order_amount
    close_signal = "SELL" if (state.last_entry_signal or "BUY") == "BUY" else "BUY"
    exit_execution = execute_trade(
        exchange=exchange,
        symbol=settings.symbol,
        signal=close_signal,
        amount=close_amount,
        live=settings.execute_orders,
        fallback_price=(snapshot.best_bid if close_signal == "SELL" else snapshot.best_ask) or snapshot.latest_close,
    )
    profit_message = format_realized_profit(state, exit_execution)
    if exit_execution.success:
        update_state(
            state,
            has_position=False,
            last_entry_signal=None,
            entry_timestamp=None,
            entry_price=None,
            entry_amount=None,
            entry_cost=None,
        )
    return CycleOutcome(
        f"MANUAL | signal={snapshot.signal} order_book={snapshot.order_book_bias} "
        f"bids={snapshot.bid_volume:.6f} asks={snapshot.ask_volume:.6f} "
        f"decision={close_signal} reason={reason} | {exit_execution.message} | {profit_message}",
        terminate=exit_execution.success,
    )


def handle_user_command(settings: Settings, exchange: Any, state: BotState, command: str) -> CycleOutcome:
    if command in {"help", "?"}:
        return CycleOutcome(
            "COMMANDS | help | status | cashout | stop",
            terminate=False,
        )
    if command == "status":
        return CycleOutcome(describe_state_file(settings, state), terminate=False)
    if command == "cashout":
        if not state.has_position:
            return CycleOutcome("MANUAL | no open position | decision=STOP", terminate=True)
        return liquidate_position(settings, exchange, state, "manual_cashout")
    if command == "stop":
        return CycleOutcome("MANUAL | stop requested | decision=STOP", terminate=True)
    return CycleOutcome(
        f"COMMANDS | unknown command={command} | valid=help,status,cashout,stop",
        terminate=False,
    )


def run_cycle(settings: Settings, exchange: Any, state: BotState) -> CycleOutcome:
    now = time.time()
    max_hold_seconds = parse_duration(settings.max_hold)
    snapshot = inspect_market(settings, exchange)

    if state.has_position:
        should_exit, reason = should_exit_position(snapshot, state, max_hold_seconds, now)
        if should_exit:
            close_amount = state.entry_amount or settings.order_amount
            entry_signal = state.last_entry_signal or "BUY"
            exit_signal = "SELL" if entry_signal == "BUY" else "BUY"
            exit_execution = execute_trade(
                exchange=exchange,
                symbol=settings.symbol,
                signal=exit_signal,
                amount=close_amount,
                live=settings.execute_orders,
                fallback_price=(snapshot.best_bid if exit_signal == "SELL" else snapshot.best_ask) or snapshot.latest_close,
            )
            profit_message = format_realized_profit(state, exit_execution)
            if exit_execution.success:
                update_state(
                    state,
                    has_position=False,
                    last_entry_signal=None,
                    entry_timestamp=None,
                    entry_price=None,
                    entry_amount=None,
                    entry_cost=None,
                )
            return CycleOutcome(
                f"HOLDING | signal={snapshot.signal} order_book={snapshot.order_book_bias} "
                f"bids={snapshot.bid_volume:.6f} asks={snapshot.ask_volume:.6f} "
                f"decision={exit_signal} reason={reason} | {exit_execution.message} | {profit_message}"
                ,
                terminate=False,
            )
        return CycleOutcome(
            f"HOLDING | signal={snapshot.signal} order_book={snapshot.order_book_bias} "
            f"bids={snapshot.bid_volume:.6f} asks={snapshot.ask_volume:.6f} decision=HOLD"
        )

    if snapshot.signal in {"BUY", "SELL"}:
        enter, reason = should_enter_position(snapshot, settings.allow_short, settings.use_xgboost)
        if not enter:
            return CycleOutcome(
                f"FLAT | signal={snapshot.signal} order_book={snapshot.order_book_bias} "
                f"bids={snapshot.bid_volume:.6f} asks={snapshot.ask_volume:.6f} "
                f"decision=WAIT reason={reason}"
            )
        entry_execution = execute_trade(
            exchange=exchange,
            symbol=settings.symbol,
            signal=snapshot.signal,
            amount=settings.order_amount,
            live=settings.execute_orders,
            fallback_price=(snapshot.best_ask if snapshot.signal == "BUY" else snapshot.best_bid) or snapshot.latest_close,
        )
        if entry_execution.success:
            update_state(
                state,
                has_position=True,
                last_entry_signal=snapshot.signal,
                entry_timestamp=now,
                entry_price=entry_execution.average_price,
                entry_amount=entry_execution.filled_amount,
                entry_cost=entry_execution.cost,
            )
        return CycleOutcome(
            f"FLAT | signal={snapshot.signal} order_book={snapshot.order_book_bias} "
            f"bids={snapshot.bid_volume:.6f} asks={snapshot.ask_volume:.6f} "
            f"decision={snapshot.signal} | {entry_execution.message}"
        )

    return CycleOutcome(
        f"FLAT | signal={snapshot.signal} order_book={snapshot.order_book_bias} "
        f"bids={snapshot.bid_volume:.6f} asks={snapshot.ask_volume:.6f} decision=WAIT"
    )


def describe_mode(settings: Settings, exchange: Any | None = None) -> str:
    if settings.demo:
        environment = "demo"
    elif settings.sandbox:
        environment = "sandbox"
    else:
        environment = "mainnet"
    execution = "live" if settings.execute_orders else "dry-run"
    api_base = None

    if exchange is not None:
        api_urls = getattr(exchange, "urls", {}).get("api")
        if isinstance(api_urls, dict):
            api_base = api_urls.get("public") or api_urls.get("private") or api_urls.get("spot")
        elif isinstance(api_urls, str):
            api_base = api_urls
        if api_base and hasattr(exchange, "implode_hostname"):
            api_base = exchange.implode_hostname(api_base)

    suffix = f" api={api_base}" if api_base else ""
    return (
        f"Starting bot on exchange={settings.exchange_id} "
        f"environment={environment} execution={execution} symbol={settings.symbol}{suffix}"
    )


def describe_state_file(settings: Settings, state: BotState) -> str:
    return (
        f"State file={Path(settings.state_file).resolve()} "
        f"has_position={state.has_position} "
        f"last_entry_signal={state.last_entry_signal} "
        f"entry_timestamp={state.entry_timestamp} "
        f"entry_price={state.entry_price} "
        f"entry_amount={state.entry_amount}"
    )


def format_auth_error(settings: Settings, exc: Exception) -> str:
    if settings.demo:
        environment = "demo"
    elif settings.sandbox:
        environment = "sandbox/testnet"
    else:
        environment = "mainnet/live"
    return (
        f"Authentication failed for exchange={settings.exchange_id} in {environment} mode: {exc}. "
        "Check that your API key and secret match the selected environment, have trading permissions, "
        "and are not blocked by IP restrictions."
    )


def fetch_exchange_preflight(exchange: Any) -> str | None:
    if getattr(exchange, "id", None) != "bybit":
        return None
    if not hasattr(exchange, "privateGetV5UserQueryApi"):
        return None

    response = exchange.privateGetV5UserQueryApi({})
    result = response.get("result", {})
    permissions = result.get("permissions", {})
    spot_permissions = permissions.get("Spot", [])
    ips = result.get("ips", [])
    read_only = result.get("readOnly")

    return (
        "Bybit API key info: "
        f"read_only={read_only} "
        f"spot_permissions={spot_permissions} "
        f"ips={ips}"
    )


def run_bot(settings: Settings) -> int:
    try:
        state = load_state(settings.state_file)
        exchange = create_exchange(
            exchange_id=settings.exchange_id,
            api_key=settings.api_key,
            api_secret=settings.api_secret,
            api_password=settings.api_password,
            sandbox=settings.sandbox,
            demo=settings.demo,
        )

        print(describe_mode(settings, exchange=exchange), flush=True)
        print(describe_state_file(settings, state), flush=True)
        print("Interactive commands: help, status, cashout, stop", flush=True)
        preflight = fetch_exchange_preflight(exchange)
        if preflight:
            print(preflight, flush=True)

        while True:
            command = read_user_command()
            if command:
                before_command = BotState(
                    has_position=state.has_position,
                    last_entry_signal=state.last_entry_signal,
                    entry_timestamp=state.entry_timestamp,
                    entry_price=state.entry_price,
                    entry_amount=state.entry_amount,
                    entry_cost=state.entry_cost,
                )
                command_outcome = handle_user_command(settings, exchange, state, command)
                if before_command != state:
                    save_state(settings.state_file, state)
                print(command_outcome.message, flush=True)
                if command_outcome.terminate:
                    break

            before = BotState(
                has_position=state.has_position,
                last_entry_signal=state.last_entry_signal,
                entry_timestamp=state.entry_timestamp,
                entry_price=state.entry_price,
                entry_amount=state.entry_amount,
                entry_cost=state.entry_cost,
            )
            outcome = run_cycle(settings=settings, exchange=exchange, state=state)
            if before != state:
                save_state(settings.state_file, state)
            print(outcome.message, flush=True)
            if outcome.terminate:
                break
            if settings.poll_seconds <= 0:
                break
            time.sleep(settings.poll_seconds)
    except ccxt.AuthenticationError as exc:
        print(format_auth_error(settings, exc), flush=True)
        return 1
    except KeyboardInterrupt:
        print("Bot stopped by user.", flush=True)

    return 0
