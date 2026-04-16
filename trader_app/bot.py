from __future__ import annotations

import csv
import json
import queue as _queue_module
import select
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ccxt

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_WHITE = "\033[37m"


def _color_text(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + ANSI_RESET


def _dashboard_label(text: str) -> str:
    return _color_text(text, ANSI_BOLD, ANSI_CYAN)


def _dashboard_value(text: str, color_code: str | None = None) -> str:
    if color_code:
        return _color_text(text, ANSI_BOLD, color_code)
    return _color_text(text, ANSI_BOLD, ANSI_WHITE)


ANSI_CLEAR_SCREEN = "\033[2J\033[H"

_TRABOT_ART = [
    "████████╗██████╗  █████╗ ██████╗  ██████╗ ████████╗",
    "╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝",
    "   ██║   ██████╔╝███████║██████╔╝██║   ██║   ██║   ",
    "   ██║   ██╔══██╗██╔══██║██╔══██╗██║   ██║   ██║   ",
    "   ██║   ██║  ██║██║  ██║██████╔╝╚██████╔╝   ██║   ",
    "   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝   ╚═╝   ",
]

# Thread-safe queue for commands typed by the user.
_command_queue: _queue_module.Queue[str] = _queue_module.Queue()


def _print_splash() -> None:
    """Print the TRABOT splash screen on startup (TTY only)."""
    if not sys.stdout.isatty():
        return
    sys.stdout.write(ANSI_CLEAR_SCREEN)
    sys.stdout.flush()
    print()
    for line in _TRABOT_ART:
        print(_color_text("  " + line, ANSI_BOLD, ANSI_MAGENTA), flush=True)
    print()
    print(_color_text("  Automated Trading Bot  ·  type  help  to begin", ANSI_BOLD, ANSI_CYAN), flush=True)
    print()


def _run_input_thread() -> None:
    """Background thread: block on stdin and push each line to the command queue."""
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip().lower()
            if cmd:
                _command_queue.put(cmd)
    except Exception:
        pass


def _start_input_thread() -> None:
    """Start the background input reader thread if stdin is a TTY."""
    if not sys.stdin or sys.stdin.closed or not sys.stdin.isatty():
        return
    t = threading.Thread(target=_run_input_thread, daemon=True)
    t.start()


def _read_trade_history(record_file: str, n: int = 12) -> list[dict]:
    """Return the last *n* rows from the record CSV, newest first."""
    path = Path(record_file)
    if not path.exists():
        return []
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        return list(reversed(rows[-n:]))
    except Exception:
        return []


def _format_history_lines(rows: list[dict], inner_width: int = 62) -> list[str]:
    """Format trade history rows into display lines that fit inside the dashboard box."""
    if not rows:
        return ["  no records found"]
    col_ts = 16
    col_dec = 10
    col_sig = 6
    col_price = 10
    col_equity = 10
    header = (
        f"  {'Timestamp':<{col_ts}}  {'Decision':<{col_dec}}  "
        f"{'Signal':<{col_sig}}  {'Price':>{col_price}}  {'Equity':>{col_equity}}"
    )
    sep = "  " + "─" * (inner_width - 4)
    lines: list[str] = [header, sep]
    for row in rows:
        ts = str(row.get("timestamp", ""))[:col_ts]
        dec = str(row.get("decision", ""))[:col_dec]
        sig = str(row.get("signal", ""))[:col_sig]
        try:
            price = f"{float(row.get('latest_close', 0)):.2f}"[:col_price]
        except (ValueError, TypeError):
            price = str(row.get("latest_close", ""))[:col_price]
        try:
            equity = f"{float(row.get('last_total_equity', 0)):.2f}"[:col_equity]
        except (ValueError, TypeError):
            equity = str(row.get("last_total_equity", ""))[:col_equity]
        line = (
            f"  {ts:<{col_ts}}  {dec:<{col_dec}}  "
            f"{sig:<{col_sig}}  {price:>{col_price}}  {equity:>{col_equity}}"
        )
        lines.append(line[:inner_width])
    return lines


from trader_app.config import Settings
from trader_app.data import (
    create_exchange,
    fetch_ohlcv_frame,
    fetch_order_book,
    retry_network_call,
    NETWORK_RETRY_EXCEPTIONS,
)
from trader_app.strategy import add_moving_averages, latest_signal


@dataclass
class BotState:
    has_position: bool = False
    last_entry_signal: str | None = None
    entry_timestamp: float | None = None
    entry_price: float | None = None
    entry_amount: float | None = None
    entry_cost: float | None = None
    last_total_equity: float | None = None
    last_candle_time: float | None = None
    peak_equity: float | None = None


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
    recent_low: float | None = None
    recent_high: float | None = None
    price_position: float | None = None
    momentum: float | None = None
    volatility: float | None = None
    order_book_imbalance: float | None = None
    spread: float | None = None
    long_ma_slope: float | None = None
    latest_bar_time: float | None = None
    macd_histogram: float | None = None
    confluence_score: int | None = None
    volume_confirmed: bool | None = None


@dataclass(frozen=True)
class CycleOutcome:
    message: str
    terminate: bool = False
    snapshot: "MarketSnapshot" | None = None


def read_user_command() -> str | None:
    """Poll the command queue for a pending user command (non-blocking)."""
    try:
        return _command_queue.get_nowait()
    except _queue_module.Empty:
        return None


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
        last_total_equity=data.get("last_total_equity"),
        last_candle_time=data.get("last_candle_time"),
        peak_equity=data.get("peak_equity"),
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
    last_total_equity: float | None,
    last_candle_time: float | None = None,
) -> bool:
    changed = (
        state.has_position != has_position
        or state.last_entry_signal != last_entry_signal
        or state.entry_timestamp != entry_timestamp
        or state.entry_price != entry_price
        or state.entry_amount != entry_amount
        or state.entry_cost != entry_cost
        or state.last_total_equity != last_total_equity
        or state.last_candle_time != last_candle_time
    )
    state.has_position = has_position
    state.last_entry_signal = last_entry_signal
    state.entry_timestamp = entry_timestamp
    state.entry_price = entry_price
    state.entry_amount = entry_amount
    state.entry_cost = entry_cost
    state.last_total_equity = last_total_equity
    state.last_candle_time = last_candle_time
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


def _write_trade_record(path: Path, record: dict[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = [
        "timestamp",
        "signal",
        "order_book_bias",
        "latest_close",
        "best_bid",
        "best_ask",
        "long_ma",
        "price_position",
        "momentum",
        "volatility",
        "spread",
        "long_ma_slope",
        "order_book_imbalance",
        "has_position",
        "entry_signal",
        "entry_price",
        "entry_amount",
        "last_total_equity",
        "decision",
        "outcome",
    ]
    with path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def _format_record_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def record_trade_snapshot(
    settings: Settings,
    state: BotState,
    snapshot: MarketSnapshot,
    decision: str | None = None,
    outcome: str = "",
) -> None:
    if not settings.record_file:
        return
    path = Path(settings.record_file)
    if path.parent and path.parent != Path('.'):
        path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "signal": snapshot.signal,
        "order_book_bias": snapshot.order_book_bias,
        "latest_close": snapshot.latest_close,
        "best_bid": _format_record_value(snapshot.best_bid),
        "best_ask": _format_record_value(snapshot.best_ask),
        "long_ma": snapshot.long_ma,
        "price_position": _format_record_value(snapshot.price_position),
        "momentum": _format_record_value(snapshot.momentum),
        "volatility": _format_record_value(snapshot.volatility),
        "spread": _format_record_value(snapshot.spread),
        "long_ma_slope": _format_record_value(snapshot.long_ma_slope),
        "order_book_imbalance": _format_record_value(snapshot.order_book_imbalance),
        "has_position": state.has_position,
        "entry_signal": _format_record_value(state.last_entry_signal),
        "entry_price": _format_record_value(state.entry_price),
        "entry_amount": _format_record_value(state.entry_amount),
        "last_total_equity": _format_record_value(state.last_total_equity),
        "decision": decision or "",
        "outcome": outcome,
    }
    _write_trade_record(path, record)


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
        order = retry_network_call(lambda: exchange.create_order(symbol=symbol, type="market", side=side, amount=amount))
    except ccxt.AuthenticationError:
        raise
    except ccxt.NetworkError as exc:
        return OrderExecution(
            success=False,
            message=f"NETWORK_ERROR {signal} {amount} {symbol} {exc}",
        )
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


def _format_equity_delta_text(prior_equity: float | None, current_equity: float | None) -> str:
    if prior_equity is None or current_equity is None:
        return ""
    delta = current_equity - prior_equity
    return f" equity_delta={delta:+.2f}"


def _is_network_outage(exc: BaseException) -> bool:
    return isinstance(exc, NETWORK_RETRY_EXCEPTIONS)


def _safe_balance_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, dict):
        if "total" in value:
            try:
                return float(value["total"])
            except (TypeError, ValueError):
                pass
        free = float(value.get("free", 0.0)) if value.get("free") is not None else 0.0
        used = float(value.get("used", 0.0)) if value.get("used") is not None else 0.0
        return free + used
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_equity_from_info(info: Any) -> float | None:
    if info is None:
        return None

    exact_fields = [
        "totalEquity",
        "accountEquity",
        "equity",
        "total_equity",
        "equityTotal",
        "account_equity",
        "totalWalletBalance",
        "equityBalance",
        "totalBalance",
        "wallet_balance",
    ]

    def _search_equity(value: Any) -> float | None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in exact_fields:
                    try:
                        return float(child)
                    except (TypeError, ValueError):
                        continue
            for key, child in value.items():
                lowered = key.lower()
                if "equity" in lowered or "total" in lowered and "balance" in lowered:
                    try:
                        return float(child)
                    except (TypeError, ValueError):
                        pass
            for child in value.values():
                result = _search_equity(child)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for item in value:
                result = _search_equity(item)
                if result is not None:
                    return result
        return None

    return _search_equity(info)


def _balance_currency_amount(balance: dict, currency: str) -> float:
    amount = _safe_balance_value(balance.get(currency))
    if amount:
        return amount
    for section in ("total", "free", "used"):
        section_value = balance.get(section)
        if isinstance(section_value, dict):
            amount = _safe_balance_value(section_value.get(currency))
            if amount:
                return amount
    return 0.0


def fetch_total_equity(settings: Settings, exchange: Any, last_price: float | None = None) -> float | None:
    if not hasattr(exchange, "fetch_balance"):
        return None
    try:
        balance = retry_network_call(lambda: exchange.fetch_balance())
    except Exception:
        return None

    info_equity = _extract_equity_from_info(balance.get("info"))
    if info_equity is not None:
        return info_equity

    symbol_parts = settings.symbol.replace(" ", "").split("/")
    if len(symbol_parts) != 2:
        return None

    base, quote = symbol_parts
    base_amount = _balance_currency_amount(balance, base)
    quote_amount = _balance_currency_amount(balance, quote)

    if quote_amount == 0.0 and base_amount == 0.0:
        return None

    if last_price is None and base_amount != 0.0:
        return None

    price = last_price if last_price is not None else 0.0
    return quote_amount + base_amount * price


def reward_equity_delta(settings: Settings, state: BotState, snapshot: MarketSnapshot, prior_equity: float | None, current_equity: float | None) -> None:
    if not settings.use_xgboost:
        return
    if prior_equity is None or current_equity is None:
        return

    equity_delta = current_equity - prior_equity
    if equity_delta == 0.0:
        return

    ml_bias = _effective_ml_bias(snapshot)
    if ml_bias not in {"BUY", "SELL"}:
        return

    from trader_app.strategy import reward_ml_model

    reward_ml_model(ml_bias, equity_delta)


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


def _effective_ml_bias(snapshot: MarketSnapshot) -> str | None:
    ml_bias = getattr(snapshot, "ml_bias", None)
    if ml_bias in {"BUY", "SELL"}:
        return ml_bias
    return None


def _can_override_order_book_conflict(snapshot: MarketSnapshot, desired_signal: str) -> bool:
    if desired_signal == "BUY":
        return (
            (snapshot.momentum is not None and snapshot.momentum > 0 and snapshot.price_position is not None and snapshot.price_position <= 0.25)
            or (snapshot.price_position is not None and snapshot.price_position <= 0.10)
        )
    if desired_signal == "SELL":
        return (
            (snapshot.momentum is not None and snapshot.momentum < 0 and snapshot.price_position is not None and snapshot.price_position >= 0.75)
            or (snapshot.price_position is not None and snapshot.price_position >= 0.90)
        )
    return False


def _should_ignore_ml_conflict(snapshot: MarketSnapshot) -> bool:
    return (
        snapshot.price_position is not None
        and (snapshot.price_position <= 0.10 or snapshot.price_position >= 0.90)
    ) or snapshot.order_book_bias == snapshot.signal


def format_decision_summary(snapshot: MarketSnapshot, use_xgboost: bool) -> str:
    ml_tag = ""
    if use_xgboost:
        ml_bias_value = snapshot.ml_bias if snapshot.ml_bias is not None else "UNAVAILABLE"
        ml_tag = f" ml_bias={ml_bias_value}"
    price_position_tag = ""
    if snapshot.price_position is not None:
        price_position_tag = f" price_position={snapshot.price_position:.2f}"
    return (
        f"signal={snapshot.signal} order_book={snapshot.order_book_bias}{ml_tag}{price_position_tag} "
        f"bids={snapshot.bid_volume:.6f} asks={snapshot.ask_volume:.6f}"
    )


def should_enter_position(snapshot: MarketSnapshot, allow_short: bool, use_xgboost: bool, settings: Settings | None = None) -> tuple[bool, str]:
    ml_bias = _effective_ml_bias(snapshot)
    momentum = getattr(snapshot, "momentum", None)

    # Compute effective confluence score, applying a -1 penalty when
    # volume_confirmation is enabled and current volume is below average.
    # This makes low-volume entries harder (not impossible) — other strong
    # indicators can still compensate.
    raw_score = snapshot.confluence_score if snapshot.confluence_score is not None else 0
    if (
        settings is not None
        and settings.volume_confirmation
        and snapshot.volume_confirmed is not None
        and not snapshot.volume_confirmed
    ):
        raw_score = max(0, raw_score - 1)

    # Confluence gate
    if settings is not None and settings.confluence_threshold > 0:
        if raw_score < settings.confluence_threshold:
            return False, f"low_confluence_{raw_score}"

    if snapshot.signal == "BUY":
        if snapshot.order_book_bias == "SELL" and not _can_override_order_book_conflict(snapshot, "BUY"):
            return False, "order_book_conflict"
        if snapshot.long_ma_slope is not None and snapshot.long_ma_slope < 0:
            return False, "trend_down"
        if snapshot.price_position is not None and snapshot.price_position >= 0.92:
            return False, "price_too_high"
        if use_xgboost and ml_bias is not None and ml_bias != "BUY" and not _should_ignore_ml_conflict(snapshot):
            return False, "ml_conflict"
        # MACD confirmation — reject BUY when histogram is strongly bearish
        if snapshot.macd_histogram is not None and snapshot.macd_histogram < 0 and snapshot.order_book_bias != "BUY":
            return False, "macd_bearish"
        return True, "signal_and_order_book"

    if snapshot.signal == "SELL":
        if not allow_short:
            return False, "shorts_disabled"
        if snapshot.order_book_bias == "BUY" and not _can_override_order_book_conflict(snapshot, "SELL"):
            return False, "order_book_conflict"
        if snapshot.long_ma_slope is not None and snapshot.long_ma_slope > 0:
            return False, "trend_up"
        if snapshot.price_position is not None and snapshot.price_position <= 0.08:
            return False, "price_too_low"
        if use_xgboost and ml_bias is not None and ml_bias != "SELL" and not _should_ignore_ml_conflict(snapshot):
            return False, "ml_conflict"
        # MACD confirmation — reject SELL when histogram is strongly bullish
        if snapshot.macd_histogram is not None and snapshot.macd_histogram > 0 and snapshot.order_book_bias != "SELL":
            return False, "macd_bullish"
        return True, "signal_and_order_book"

    return False, "no_entry_signal"


def inspect_market(settings: Settings, exchange: Any) -> MarketSnapshot:
    frame = fetch_ohlcv_frame(
        exchange=exchange,
        symbol=settings.symbol,
        timeframe=settings.timeframe,
        limit=settings.long_window + 50,
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
    from trader_app.strategy import (
        compute_price_position,
        compute_latest_macd,
        compute_confluence_score,
        has_volume_confirmation,
    )

    recent_low, recent_high, price_position = compute_price_position(frame)
    previous_close = float(frame.iloc[-2]["close"]) if len(frame) >= 2 else latest_close
    momentum = latest_close - previous_close
    volatility = float(frame["close"].rolling(10).std().iloc[-1]) if len(frame) >= 10 else 0.0
    order_book = fetch_order_book(
        exchange=exchange,
        symbol=settings.symbol,
        depth=settings.order_book_depth,
    )
    bid_volume, ask_volume, order_book_bias = summarize_order_book(
        order_book,
        settings.sell_pressure_ratio,
    )
    imbalance = 0.0
    if bid_volume + ask_volume > 0:
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    spread = 0.0
    best_bid = order_book.get("bids", [[None]])[0][0] if order_book.get("bids") else None
    best_ask = order_book.get("asks", [[None]])[0][0] if order_book.get("asks") else None
    if best_bid is not None and best_ask is not None and best_ask > best_bid:
        spread = (best_ask - best_bid) / float(best_bid)

    long_ma_slope = None
    if len(analyzed) >= 3:
        long_ma_slope = float(analyzed["ma_long"].iloc[-1] - analyzed["ma_long"].iloc[-3])

    latest_bar_time = float(frame["time"].iloc[-1].timestamp())

    # MACD histogram
    _, _, macd_hist_val = compute_latest_macd(frame)

    # Confluence score
    confluence = compute_confluence_score(frame, signal)

    # Volume confirmation
    vol_confirmed = has_volume_confirmation(frame)

    if settings.use_xgboost:
        from trader_app.strategy import compute_ml_bias

        try:
            ml_bias = compute_ml_bias(
                analyzed,
                settings.short_window,
                settings.long_window,
                imbalance,
                spread,
            )
        except TypeError:
            try:
                ml_bias = compute_ml_bias(analyzed, settings.short_window, settings.long_window)
            except Exception:
                ml_bias = "UNAVAILABLE"
        except Exception:
            ml_bias = "UNAVAILABLE"
    return MarketSnapshot(
        signal=signal,
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        order_book_bias=order_book_bias,
        latest_close=latest_close,
        best_bid=best_bid,
        best_ask=best_ask,
        long_ma=long_ma,
        ml_bias=ml_bias,
        recent_low=recent_low,
        recent_high=recent_high,
        price_position=price_position,
        momentum=momentum,
        volatility=volatility,
        order_book_imbalance=imbalance,
        spread=spread,
        long_ma_slope=long_ma_slope,
        latest_bar_time=latest_bar_time,
        macd_histogram=macd_hist_val,
        confluence_score=confluence,
        volume_confirmed=vol_confirmed,
    )


def should_exit_position(
    snapshot: MarketSnapshot,
    state: BotState,
    max_hold_seconds: int | None,
    now: float,
    settings: Settings,
) -> tuple[bool, str]:
    if (
        max_hold_seconds is not None
        and state.entry_timestamp is not None
        and now - state.entry_timestamp >= max_hold_seconds
    ):
        return True, "max_hold"
    entry_signal = state.last_entry_signal or "BUY"
    if state.entry_price is not None:
        current_price = snapshot.latest_close
        if entry_signal == "BUY":
            stop_threshold = state.entry_price * (1 - settings.stop_loss)
            take_threshold = state.entry_price * (1 + settings.take_profit)
            if current_price <= stop_threshold + 1e-9:
                return True, "stop_loss"
            if current_price >= take_threshold - 1e-9:
                return True, "take_profit"
        else:
            stop_threshold = state.entry_price * (1 + settings.stop_loss)
            take_threshold = state.entry_price * (1 - settings.take_profit)
            if current_price >= stop_threshold - 1e-9:
                return True, "stop_loss"
            if current_price <= take_threshold + 1e-9:
                return True, "take_profit"
    ml_bias = _effective_ml_bias(snapshot)
    price_position = getattr(snapshot, "price_position", None)
    if settings.use_xgboost and ml_bias is not None:
        if entry_signal == "BUY" and ml_bias == "SELL":
            return True, "ml_signal"
        if entry_signal == "SELL" and ml_bias == "BUY":
            return True, "ml_signal"
    if entry_signal == "BUY":
        if settings.use_xgboost and price_position is not None and price_position >= 0.88:
            return True, "price_peak"
        if snapshot.signal == "SELL":
            return True, "signal_flip"
        if snapshot.order_book_bias == "SELL":
            return True, "sell_pressure"
        if snapshot.momentum is not None and snapshot.momentum < 0 and snapshot.order_book_bias == "SELL":
            return True, "negative_momentum"
        return False, "hold"
    if entry_signal == "SELL":
        if settings.use_xgboost and price_position is not None and price_position <= 0.12:
            return True, "price_trough"
        if snapshot.signal == "BUY":
            return True, "signal_flip"
        if snapshot.order_book_bias == "BUY":
            return True, "buy_pressure"
        if snapshot.momentum is not None and snapshot.momentum > 0:
            return True, "positive_momentum"
        return False, "hold"
    return False, "hold"


def _compute_realized_profit_amount(state: BotState, exit_execution: OrderExecution) -> float | None:
    entry_cost = state.entry_cost
    if entry_cost is None and state.entry_price is not None and state.entry_amount is not None:
        entry_cost = state.entry_price * state.entry_amount
    exit_cost = exit_execution.cost
    if exit_cost is None and exit_execution.average_price is not None and exit_execution.filled_amount is not None:
        exit_cost = exit_execution.average_price * exit_execution.filled_amount
    if entry_cost is None or exit_cost is None:
        return None
    return entry_cost - exit_cost if state.last_entry_signal == "SELL" else exit_cost - entry_cost


def liquidate_position(settings: Settings, exchange: Any, state: BotState, reason: str) -> CycleOutcome:
    try:
        snapshot = inspect_market(settings, exchange)
    except Exception as exc:
        if _is_network_outage(exc):
            return CycleOutcome(
                f"OUTAGE | decision=WAIT reason=network_unavailable error={type(exc).__name__}",
                terminate=False,
            )
        raise
    summary = format_decision_summary(snapshot, settings.use_xgboost)
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
    exit_equity = fetch_total_equity(settings, exchange, snapshot.latest_close)
    equity_delta_text = _format_equity_delta_text(state.last_total_equity, exit_equity)
    if exit_execution.success and exit_equity is not None:
        state.last_total_equity = exit_equity
    if exit_execution.success and settings.use_xgboost:
        from trader_app.strategy import reward_ml_model

        profit_amount = _compute_realized_profit_amount(state, exit_execution)
        ml_bias = _effective_ml_bias(snapshot)
        if profit_amount is not None and profit_amount > 0 and ml_bias == state.last_entry_signal:
            reward_ml_model(ml_bias, profit_amount)
        reward_equity_delta(settings, state, snapshot, state.last_total_equity, exit_equity)
    if exit_execution.success:
        update_state(
            state,
            has_position=False,
            last_entry_signal=None,
            entry_timestamp=None,
            entry_price=None,
            entry_amount=None,
            entry_cost=None,
            last_total_equity=state.last_total_equity,
        )
    return CycleOutcome(
        f"MANUAL | {summary} decision={close_signal} reason={reason} | {exit_execution.message} | {profit_message}{equity_delta_text}",
        terminate=exit_execution.success,
    )


def handle_user_command(settings: Settings, exchange: Any, state: BotState, command: str) -> CycleOutcome:
    if command in {"help", "?"}:
        return CycleOutcome(
            "COMMANDS | help | status | history | cashout | stop",
            terminate=False,
        )
    if command == "status":
        return CycleOutcome(describe_state_file(settings, state), terminate=False)
    if command in {"history", "hist"}:
        if not settings.record_file:
            return CycleOutcome("history | no record file configured (use --record-file)", terminate=False)
        rows = _read_trade_history(settings.record_file)
        lines = _format_history_lines(rows)
        return CycleOutcome("\n".join(lines), terminate=False)
    if command == "cashout":
        if not state.has_position:
            return CycleOutcome("MANUAL | no open position | decision=STOP", terminate=True)
        return liquidate_position(settings, exchange, state, "manual_cashout")
    if command == "stop":
        return CycleOutcome("MANUAL | stop requested | decision=STOP", terminate=True)
    return CycleOutcome(
        f"COMMANDS | unknown command={command} | valid=help,status,history,cashout,stop",
        terminate=False,
    )


def run_cycle(settings: Settings, exchange: Any, state: BotState) -> CycleOutcome:
    now = time.time()
    max_hold_seconds = parse_duration(settings.max_hold)
    try:
        snapshot = inspect_market(settings, exchange)
    except Exception as exc:
        if _is_network_outage(exc):
            return CycleOutcome(
                f"OUTAGE | decision=WAIT reason=network_unavailable error={type(exc).__name__}",
                terminate=False,
            )
        raise

    prior_equity = state.last_total_equity
    current_equity = fetch_total_equity(settings, exchange, snapshot.latest_close)
    if current_equity is not None:
        state.last_total_equity = current_equity
        # Track historical peak equity for accurate drawdown calculation
        if state.peak_equity is None or current_equity > state.peak_equity:
            state.peak_equity = current_equity

    # Drawdown protection — stop trading if peak-equity drawdown exceeds limit
    if settings.max_drawdown > 0:
        if current_equity is not None and state.peak_equity is not None and state.peak_equity > 0:
            dd = (state.peak_equity - current_equity) / state.peak_equity
            if dd >= settings.max_drawdown:
                summary = format_decision_summary(snapshot, settings.use_xgboost)
                position_tag = "HOLDING" if state.has_position else "FLAT"
                return CycleOutcome(
                    f"{position_tag} | {summary} decision=STOP reason=max_drawdown_exceeded drawdown={dd:.2%}",
                    terminate=True,
                    snapshot=snapshot,
                )

    if state.has_position:
        should_exit, reason = should_exit_position(
            snapshot,
            state,
            max_hold_seconds,
            now,
            settings,
        )
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
            exit_equity = fetch_total_equity(settings, exchange, snapshot.latest_close)
            equity_delta_text = _format_equity_delta_text(prior_equity, exit_equity)
            if exit_execution.success and exit_equity is not None:
                state.last_total_equity = exit_equity
            if exit_execution.success and settings.use_xgboost:
                from trader_app.strategy import reward_ml_model

                profit_amount = _compute_realized_profit_amount(state, exit_execution)
                if profit_amount is not None and profit_amount > 0:
                    ml_bias = _effective_ml_bias(snapshot)
                    if ml_bias == state.last_entry_signal:
                        reward_ml_model(ml_bias, profit_amount)
                reward_equity_delta(settings, state, snapshot, prior_equity, exit_equity)
            summary = format_decision_summary(snapshot, settings.use_xgboost)
            if exit_execution.success:
                update_state(
                    state,
                    has_position=False,
                    last_entry_signal=None,
                    entry_timestamp=None,
                    entry_price=None,
                    entry_amount=None,
                    entry_cost=None,
                    last_total_equity=state.last_total_equity,
                    last_candle_time=snapshot.latest_bar_time,
                )
            return CycleOutcome(
                f"HOLDING | {summary} decision={exit_signal} reason={reason} | {exit_execution.message} | {profit_message}{equity_delta_text}",
                terminate=False,
                snapshot=snapshot,
            )
        summary = format_decision_summary(snapshot, settings.use_xgboost)
        reward_equity_delta(settings, state, snapshot, prior_equity, current_equity)
        if current_equity is not None:
            state.last_total_equity = current_equity
        return CycleOutcome(
            f"HOLDING | {summary} decision=HOLD",
            snapshot=snapshot,
        )

    prior_equity = state.last_total_equity
    if prior_equity is None:
        prior_equity = fetch_total_equity(settings, exchange, snapshot.latest_close)
        if prior_equity is not None:
            state.last_total_equity = prior_equity

    if snapshot.signal in {"BUY", "SELL"}:
        summary = format_decision_summary(snapshot, settings.use_xgboost)
        enter, reason = should_enter_position(snapshot, settings.allow_short, settings.use_xgboost, settings)
        if not enter:
            current_equity = fetch_total_equity(settings, exchange, snapshot.latest_close)
            reward_equity_delta(settings, state, snapshot, prior_equity, current_equity)
            if current_equity is not None:
                state.last_total_equity = current_equity
            return CycleOutcome(
                f"FLAT | {summary} decision=WAIT reason={reason}"
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
                last_total_equity=state.last_total_equity,
                last_candle_time=snapshot.latest_bar_time,
            )
        current_equity = fetch_total_equity(settings, exchange, snapshot.latest_close)
        reward_equity_delta(settings, state, snapshot, prior_equity, current_equity)
        if current_equity is not None:
            state.last_total_equity = current_equity
        return CycleOutcome(
            f"FLAT | {summary} decision={snapshot.signal} | {entry_execution.message}",
            snapshot=snapshot,
        )

    summary = format_decision_summary(snapshot, settings.use_xgboost)
    current_equity = fetch_total_equity(settings, exchange, snapshot.latest_close)
    reward_equity_delta(settings, state, snapshot, prior_equity, current_equity)
    if current_equity is not None:
        state.last_total_equity = current_equity
    return CycleOutcome(
        f"FLAT | {summary} decision=WAIT",
        snapshot=snapshot,
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


def render_dashboard(
    settings: Settings,
    state: BotState,
    snapshot: MarketSnapshot,
    outcome_message: str,
    last_command_output: str = "",
) -> None:
    W = 62  # inner box width

    # Clear screen and redraw from top on TTY; plain append otherwise.
    if sys.stdout.isatty():
        sys.stdout.write(ANSI_CLEAR_SCREEN)
        sys.stdout.flush()

    position_text = "OPEN" if state.has_position else "FLAT"
    position_color = ANSI_GREEN if state.has_position else ANSI_YELLOW
    entry_price = f"{state.entry_price:.4f}" if state.entry_price is not None else "n/a"
    entry_amount = f"{state.entry_amount:.4f}" if state.entry_amount is not None else "n/a"
    ml_bias = snapshot.ml_bias if snapshot.ml_bias is not None else "UNAVAILABLE"
    price_position = f"{snapshot.price_position:.2f}" if snapshot.price_position is not None else "n/a"
    momentum = f"{snapshot.momentum:.4f}" if snapshot.momentum is not None else "n/a"
    volatility = f"{snapshot.volatility:.4f}" if snapshot.volatility is not None else "n/a"
    equity = f"{state.last_total_equity:.2f}" if state.last_total_equity is not None else "n/a"
    signal_color = ANSI_GREEN if snapshot.signal == "BUY" else ANSI_RED
    order_book_color = (
        ANSI_GREEN if snapshot.order_book_bias == "BUY"
        else ANSI_RED if snapshot.order_book_bias == "SELL"
        else ANSI_YELLOW
    )
    macd_text = f"{snapshot.macd_histogram:.4f}" if snapshot.macd_histogram is not None else "n/a"
    confluence_text = str(snapshot.confluence_score) if snapshot.confluence_score is not None else "n/a"
    vol_text = "YES" if snapshot.volume_confirmed else "NO" if snapshot.volume_confirmed is not None else "n/a"
    vol_color = ANSI_GREEN if snapshot.volume_confirmed else ANSI_RED
    timestamp = time.strftime("%H:%M:%S")

    # ── header ──────────────────────────────────────────────────────────────
    title = (
        f" TRABOT  ·  {settings.symbol}  ·  {settings.exchange_id}"
        f"  ·  {'live' if settings.execute_orders else 'dry-run'}  ·  {timestamp}"
    )
    title_padded = title[:W].ljust(W)
    print(_color_text("╔" + "═" * W + "╗", ANSI_BOLD, ANSI_CYAN), flush=True)
    print(
        _color_text("║", ANSI_CYAN)
        + _color_text(title_padded, ANSI_BOLD, ANSI_MAGENTA)
        + _color_text("║", ANSI_CYAN),
        flush=True,
    )
    print(_color_text("╠" + "═" * W + "╣", ANSI_CYAN), flush=True)

    # ── data rows ────────────────────────────────────────────────────────────
    print(
        f"{_dashboard_label('Position')}: {_dashboard_value(position_text, position_color)}  "
        f"{_dashboard_label('Entry')}: {_dashboard_value(entry_price, ANSI_WHITE)}  "
        f"{_dashboard_label('Size')}: {_dashboard_value(entry_amount, ANSI_WHITE)}",
        flush=True,
    )
    print(
        f"{_dashboard_label('Signal')}: {_dashboard_value(snapshot.signal, signal_color)}    "
        f"{_dashboard_label('Order')}: {_dashboard_value(snapshot.order_book_bias, order_book_color)}  "
        f"{_dashboard_label('ML')}: {_dashboard_value(ml_bias, ANSI_CYAN)}",
        flush=True,
    )
    print(
        f"{_dashboard_label('Price')}: {_dashboard_value(f'{snapshot.latest_close:.2f}', ANSI_WHITE)}  "
        f"{_dashboard_label('MA')}: {_dashboard_value(f'{snapshot.long_ma:.2f}', ANSI_WHITE)}  "
        f"{_dashboard_label('Range')}: {_dashboard_value(price_position, ANSI_WHITE)}",
        flush=True,
    )
    print(
        f"{_dashboard_label('Equity')}: {_dashboard_value(equity, ANSI_GREEN)}  "
        f"{_dashboard_label('SL/TP')}: {_dashboard_value(f'{settings.stop_loss*100:.1f}%/{settings.take_profit*100:.1f}%', ANSI_WHITE)}  "
        f"{_dashboard_label('Mode')}: {_dashboard_value(settings.exchange_id, ANSI_WHITE)}",
        flush=True,
    )
    print(
        f"{_dashboard_label('MACD')}: {_dashboard_value(macd_text, ANSI_YELLOW)}  "
        f"{_dashboard_label('Confluence')}: {_dashboard_value(confluence_text, ANSI_CYAN)}  "
        f"{_dashboard_label('VolOK')}: {_dashboard_value(vol_text, vol_color)}",
        flush=True,
    )
    print(
        f"{_dashboard_label('Bids')}: {_dashboard_value(f'{snapshot.bid_volume:.2f}', ANSI_WHITE)}  "
        f"{_dashboard_label('Asks')}: {_dashboard_value(f'{snapshot.ask_volume:.2f}', ANSI_WHITE)}  "
        f"{_dashboard_label('Momentum')}: {_dashboard_value(momentum, ANSI_YELLOW)}  "
        f"{_dashboard_label('Vol')}: {_dashboard_value(volatility, ANSI_YELLOW)}",
        flush=True,
    )

    # ── last decision ────────────────────────────────────────────────────────
    print(_color_text("╠" + "═" * W + "╣", ANSI_CYAN), flush=True)
    decision_line = outcome_message[:W - 2]
    print(
        _color_text("║ ", ANSI_CYAN)
        + _color_text(decision_line.ljust(W - 2), ANSI_WHITE)
        + _color_text(" ║", ANSI_CYAN),
        flush=True,
    )

    # ── command output (shown until the next command replaces it) ────────────
    if last_command_output:
        print(_color_text("╠" + "═" * W + "╣", ANSI_CYAN), flush=True)
        for raw_line in last_command_output.splitlines()[:10]:
            display = raw_line[:W - 2].ljust(W - 2)
            print(
                _color_text("║ ", ANSI_CYAN)
                + _color_text(display, ANSI_YELLOW)
                + _color_text(" ║", ANSI_CYAN),
                flush=True,
            )

    print(_color_text("╚" + "═" * W + "╝", ANSI_BOLD, ANSI_CYAN), flush=True)

    # ── Codex-style prompt ───────────────────────────────────────────────────
    if sys.stdout.isatty():
        prompt_bar = "─ TRABOT " + "─" * (W - 8)
        print(_color_text("╭" + prompt_bar + "╮", ANSI_BOLD, ANSI_CYAN), flush=True)
        sys.stdout.write(
            _color_text("│  ❯ ", ANSI_BOLD, ANSI_GREEN)
            if sys.stdout.isatty()
            else "│  ❯ "
        )
        sys.stdout.flush()


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

        _print_splash()
        print(_color_text(describe_mode(settings, exchange=exchange), ANSI_BOLD, ANSI_MAGENTA), flush=True)
        print(_color_text(describe_state_file(settings, state), ANSI_BOLD, ANSI_WHITE), flush=True)
        preflight = fetch_exchange_preflight(exchange)
        if preflight:
            print(_color_text(preflight, ANSI_CYAN), flush=True)

        _start_input_thread()

        last_command_output: str = ""

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
                    last_total_equity=state.last_total_equity,
                    peak_equity=state.peak_equity,
                )
                command_outcome = handle_user_command(settings, exchange, state, command)
                if before_command != state:
                    save_state(settings.state_file, state)
                last_command_output = command_outcome.message
                if command_outcome.terminate:
                    print(last_command_output, flush=True)
                    break

            before = BotState(
                has_position=state.has_position,
                last_entry_signal=state.last_entry_signal,
                entry_timestamp=state.entry_timestamp,
                entry_price=state.entry_price,
                entry_amount=state.entry_amount,
                entry_cost=state.entry_cost,
                last_total_equity=state.last_total_equity,
                peak_equity=state.peak_equity,
            )
            outcome = run_cycle(settings=settings, exchange=exchange, state=state)
            if before != state:
                save_state(settings.state_file, state)
            if outcome.snapshot is not None:
                decision = None
                if "decision=" in outcome.message:
                    decision = outcome.message.split("decision=")[1].split()[0].strip()
                record_trade_snapshot(settings, state, outcome.snapshot, decision=decision, outcome=outcome.message)
                render_dashboard(settings, state, outcome.snapshot, outcome.message, last_command_output)
            else:
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
        print("\nBot stopped by user.", flush=True)

    return 0
