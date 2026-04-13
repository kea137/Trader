import json
import ccxt
import pandas as pd

from trader_app.bot import (
    BotState,
    MarketSnapshot,
    OrderExecution,
    describe_mode,
    describe_state_file,
    execute_trade,
    execute_signal,
    fetch_exchange_preflight,
    format_decision_summary,
    format_realized_profit,
    format_auth_error,
    handle_user_command,
    fetch_total_equity,
    inspect_market,
    liquidate_position,
    load_state,
    parse_duration,
    run_cycle,
    run_bot,
    save_state,
    should_enter_position,
    should_exit_position,
    summarize_order_book,
    update_state,
)
from trader_app.config import Settings
from trader_app.data import create_exchange
from trader_app.strategy import add_moving_averages, latest_signal


def test_latest_signal_returns_buy_when_short_average_is_above_long_average():
    frame = pd.DataFrame({"close": list(range(1, 301))})

    analyzed = add_moving_averages(frame, short_window=50, long_window=200)

    assert latest_signal(analyzed) == "BUY"


def test_latest_signal_returns_sell_when_short_average_is_below_long_average():
    frame = pd.DataFrame({"close": list(range(300, 0, -1))})

    analyzed = add_moving_averages(frame, short_window=50, long_window=200)

    assert latest_signal(analyzed) == "SELL"


def test_execute_signal_returns_dry_run_message_without_live_execution():
    message = execute_signal(object(), "BTC/USDT", "BUY", 0.01, live=False)

    assert message == "DRY_RUN BUY 0.01 BTC/USDT"


def test_execute_trade_captures_fill_details():
    class FakeExchange:
        def create_order(self, symbol, type, side, amount):
            return {
                "id": "buy1",
                "status": "closed",
                "filled": 0.5,
                "average": 100.0,
                "cost": 50.0,
            }

    execution = execute_trade(FakeExchange(), "BTC/USDT", "BUY", 0.5, live=True)

    assert execution.success is True
    assert execution.filled_amount == 0.5
    assert execution.average_price == 100.0
    assert execution.cost == 50.0


def test_run_cycle_submits_market_sell_when_signal_is_sell():
    class FakeExchange:
        id = "fake"

        def __init__(self):
            self.orders = []

        def fetch_ohlcv(self, symbol, timeframe):
            return [[index * 60_000, 0, 0, 0, price, 0] for index, price in enumerate(range(300, 0, -1), start=1)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100, 5]], "asks": [[101, 12]]}

        def create_order(self, symbol, type, side, amount):
            self.orders.append(
                {"symbol": symbol, "type": type, "side": side, "amount": amount}
            )
            return {"id": "abc123", "status": "closed", "filled": 0.25, "average": 120.0, "cost": 30.0}

    exchange = FakeExchange()
    settings = Settings(symbol="BTC/USDT", execute_orders=True, order_amount=0.25)
    state = BotState(has_position=True, entry_price=100.0, entry_amount=0.25, entry_cost=25.0)

    outcome = run_cycle(settings, exchange, state)

    assert "decision=SELL reason=stop_loss" in outcome.message
    assert "EXECUTED SELL 0.25 BTC/USDT order_id=abc123 status=closed" in outcome.message
    assert "profit=5.000000 quote_currency" in outcome.message
    assert outcome.terminate is False
    assert exchange.orders == [
        {
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "sell",
            "amount": 0.25,
        }
    ]
    assert state.has_position is False


def test_run_cycle_reports_equity_delta_on_sell():
    class FakeExchange:
        id = "fake"

        def __init__(self):
            self.orders = []

        def fetch_ohlcv(self, symbol, timeframe):
            return [[index * 60_000, 0, 0, 0, price, 0] for index, price in enumerate(range(300, 0, -1), start=1)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100, 5]], "asks": [[101, 12]]}

        def create_order(self, symbol, type, side, amount):
            self.orders.append({"symbol": symbol, "type": type, "side": side, "amount": amount})
            return {"id": "sell123", "status": "closed", "filled": amount, "average": 100.0, "cost": 25.0}

        def fetch_balance(self):
            return {"BTC": {"free": 0.0, "used": 0.0}, "USDT": {"free": 1100.0, "used": 0.0}}

    state = BotState(
        has_position=True,
        last_entry_signal="BUY",
        entry_timestamp=0.0,
        entry_price=100.0,
        entry_amount=0.25,
        entry_cost=25.0,
        last_total_equity=1000.0,
    )
    exchange = FakeExchange()
    settings = Settings(symbol="BTC/USDT", execute_orders=True, order_amount=0.25)

    outcome = run_cycle(settings, exchange, state)

    assert "equity_delta=+100.00" in outcome.message
    assert state.last_total_equity == 1100.0


def test_run_cycle_rewards_model_when_equity_grows_while_holding(monkeypatch):
    class FakeExchange:
        id = "fake"

        def fetch_balance(self):
            return {
                "info": {"result": {"accountEquity": "1100.0"}},
                "BTC": {"free": 0.0, "used": 0.0},
                "USDT": {"free": 1100.0, "used": 0.0},
            }

    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10.0,
        ask_volume=3.0,
        order_book_bias="BUY",
        latest_close=109.0,
        best_bid=109.0,
        best_ask=111.0,
        long_ma=100.0,
        ml_bias="BUY",
        price_position=0.5,
        momentum=1.0,
        volatility=0.1,
    )

    called = {"signal": None, "amount": 0.0}

    def fake_reward_ml_model(predicted_signal, profit):
        called["signal"] = predicted_signal
        called["amount"] = profit

    monkeypatch.setattr("trader_app.strategy.reward_ml_model", fake_reward_ml_model)
    monkeypatch.setattr("trader_app.bot.inspect_market", lambda settings, exchange: snapshot)

    state = BotState(
        has_position=True,
        last_entry_signal="BUY",
        entry_timestamp=0.0,
        entry_price=100.0,
        entry_amount=0.25,
        entry_cost=25.0,
        last_total_equity=1000.0,
    )
    settings = Settings(symbol="BTC/USDT", execute_orders=False, use_xgboost=True, order_amount=0.25, take_profit=0.20)

    outcome = run_cycle(settings, FakeExchange(), state)

    assert called["signal"] == "BUY"
    assert called["amount"] == 100.0
    assert "decision=HOLD" in outcome.message


def test_save_and_load_state_round_trip(tmp_path):
    state_file = tmp_path / "bot_state.json"
    state = BotState(
        has_position=True,
        last_entry_signal="BUY",
        entry_timestamp=123.0,
        entry_price=100.0,
        entry_amount=0.5,
        entry_cost=50.0,
        last_total_equity=1200.0,
    )

    save_state(str(state_file), state)
    loaded = load_state(str(state_file))

    assert loaded == state
    assert json.loads(state_file.read_text()) == {
        "entry_amount": 0.5,
        "entry_cost": 50.0,
        "entry_price": 100.0,
        "entry_timestamp": 123.0,
        "has_position": True,
        "last_entry_signal": "BUY",
        "last_total_equity": 1200.0,
    }


def test_fetch_total_equity_uses_balance_and_price():
    class FakeExchange:
        def fetch_balance(self):
            return {
                "BTC": {"free": 0.1, "used": 0.0},
                "USDT": {"free": 1000.0, "used": 0.0},
            }

    equity = fetch_total_equity(Settings(symbol="BTC/USDT"), FakeExchange(), last_price=20000.0)

    assert equity == 3000.0


def test_fetch_total_equity_uses_info_equity_when_available():
    class FakeExchange:
        def fetch_balance(self):
            return {
                "info": {
                    "result": {
                        "accountEquity": "1500.0"
                    }
                },
                "BTC": {"free": 0.0, "used": 0.0},
                "USDT": {"free": 0.0, "used": 0.0},
            }

    equity = fetch_total_equity(Settings(symbol="BTC/USDT"), FakeExchange(), last_price=20000.0)

    assert equity == 1500.0


def test_fetch_total_equity_uses_nested_info_equity_list():
    class FakeExchange:
        def fetch_balance(self):
            return {
                "info": {
                    "result": [
                        {"equity": "1750.0"}
                    ]
                },
                "BTC": {"free": 0.0, "used": 0.0},
                "USDT": {"free": 0.0, "used": 0.0},
            }

    equity = fetch_total_equity(Settings(symbol="BTC/USDT"), FakeExchange(), last_price=20000.0)

    assert equity == 1750.0


def test_execute_trade_returns_network_error_after_retries():
    class FakeExchange:
        def __init__(self):
            self.attempts = 0

        def create_order(self, symbol, type, side, amount):
            self.attempts += 1
            raise ccxt.NetworkError("connection lost")

    exchange = FakeExchange()
    execution = execute_trade(exchange, "BTC/USDT", "BUY", 0.01, live=True)

    assert execution.success is False
    assert "NETWORK_ERROR" in execution.message
    assert exchange.attempts == 3


def test_run_cycle_handles_network_outage_from_market_data():
    class FakeExchange:
        id = "fake"

        def fetch_ohlcv(self, symbol, timeframe, limit=None):
            raise ccxt.NetworkError("connection lost")

        def fetch_order_book(self, symbol, limit):
            return {"bids": [], "asks": []}

    settings = Settings(symbol="BTC/USDT", execute_orders=False)
    outcome = run_cycle(settings, FakeExchange(), BotState())

    assert "OUTAGE | decision=WAIT reason=network_unavailable" in outcome.message
    assert outcome.terminate is False
    assert outcome.snapshot is None


def test_load_state_returns_default_when_file_missing(tmp_path):
    state = load_state(str(tmp_path / "missing.json"))

    assert state == BotState()


def test_update_state_reports_change_only_when_values_differ():
    state = BotState()

    changed = update_state(
        state,
        has_position=True,
        last_entry_signal="BUY",
        entry_timestamp=100.0,
        entry_price=100.0,
        entry_amount=0.5,
        entry_cost=50.0,
        last_total_equity=None,
    )

    assert changed is True
    assert state == BotState(
        has_position=True,
        last_entry_signal="BUY",
        entry_timestamp=100.0,
        entry_price=100.0,
        entry_amount=0.5,
        entry_cost=50.0,
        last_total_equity=None,
    )
    assert update_state(
        state,
        has_position=True,
        last_entry_signal="BUY",
        entry_timestamp=100.0,
        entry_price=100.0,
        entry_amount=0.5,
        entry_cost=50.0,
        last_total_equity=None,
    ) is False


def test_format_realized_profit_uses_entry_and_exit_costs():
    state = BotState(has_position=True, entry_cost=50.0, entry_amount=0.5, entry_price=100.0)
    profit = format_realized_profit(
        state,
        execute_trade(
            type(
                "FakeExchange",
                (),
                {
                    "create_order": lambda self, symbol, type, side, amount: {
                        "id": "sell1",
                        "status": "closed",
                        "filled": 0.5,
                        "average": 120.0,
                        "cost": 60.0,
                    }
                },
            )(),
            "BTC/USDT",
            "SELL",
            0.5,
            live=True,
        ),
    )

    assert profit == "profit=10.000000 quote_currency profit_pct=20.00%"


def test_parse_duration_supports_minutes_and_hours():
    assert parse_duration("30m") == 1800
    assert parse_duration("1h") == 3600


def test_parse_duration_rejects_invalid_values():
    try:
        parse_duration("hour")
    except ValueError as exc:
        assert "max_hold" in str(exc)
    else:
        raise AssertionError("Expected parse_duration to reject invalid values")


def test_summarize_order_book_detects_sell_pressure():
    bid_volume, ask_volume, bias = summarize_order_book(
        {"bids": [[100, 2], [99, 1]], "asks": [[101, 5], [102, 1]]},
        sell_pressure_ratio=1.2,
    )

    assert bid_volume == 3
    assert ask_volume == 6
    assert bias == "SELL"


def test_inspect_market_combines_signal_and_order_book():
    class FakeExchange:
        id = "fake"

        def fetch_ohlcv(self, symbol, timeframe):
            return [[index * 60_000, 0, 0, 0, price, 0] for index, price in enumerate(range(1, 301), start=1)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100, 10]], "asks": [[101, 3]]}

    snapshot = inspect_market(Settings(), FakeExchange())

    assert snapshot.signal == "BUY"
    assert snapshot.order_book_bias == "BUY"
    assert snapshot.bid_volume == 10
    assert snapshot.ask_volume == 3
    assert snapshot.latest_close == 300.0
    assert snapshot.best_bid == 100
    assert snapshot.best_ask == 101
    assert snapshot.long_ma > 0


def test_inspect_market_uses_ml_bias_when_enabled(monkeypatch):
    class FakeExchange:
        id = "fake"

        def fetch_ohlcv(self, symbol, timeframe, limit=None):
            return [[index * 60_000, 0, 0, 0, price, 0] for index, price in enumerate(range(1, 101), start=1)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100, 10]], "asks": [[101, 3]]}

    called = {"invoked": False}

    def fake_compute_ml_bias(frame, short_window, long_window):
        called["invoked"] = True
        return "SELL"

    monkeypatch.setattr("trader_app.strategy.compute_ml_bias", fake_compute_ml_bias)

    settings = Settings(use_xgboost=True, short_window=5, long_window=20)
    snapshot = inspect_market(settings, FakeExchange())

    assert called["invoked"] is True
    assert snapshot.ml_bias == "SELL"


def test_inspect_market_reports_unavailable_ml_bias_when_model_fails(monkeypatch):
    class FakeExchange:
        id = "fake"

        def fetch_ohlcv(self, symbol, timeframe, limit=None):
            return [[index * 60_000, 0, 0, 0, price, 0] for index, price in enumerate(range(1, 101), start=1)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100, 10]], "asks": [[101, 3]]}

    def fake_compute_ml_bias(frame, short_window, long_window):
        raise ValueError("missing training data")

    monkeypatch.setattr("trader_app.strategy.compute_ml_bias", fake_compute_ml_bias)

    settings = Settings(use_xgboost=True, short_window=5, long_window=20)
    snapshot = inspect_market(settings, FakeExchange())

    assert snapshot.ml_bias == "UNAVAILABLE"


def test_should_enter_position_ignores_unavailable_ml_bias():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=5,
        order_book_bias="BUY",
        latest_close=300.0,
        best_bid=100.0,
        best_ask=101.0,
        long_ma=250.0,
        ml_bias="UNAVAILABLE",
    )

    enter, reason = should_enter_position(snapshot, allow_short=False, use_xgboost=True)

    assert enter is True
    assert reason == "signal_and_order_book"


def test_should_enter_position_rejects_buy_when_price_is_too_high():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=5,
        order_book_bias="BUY",
        latest_close=110.0,
        best_bid=109.0,
        best_ask=111.0,
        long_ma=100.0,
        price_position=0.95,
        ml_bias="SELL",
    )

    enter, reason = should_enter_position(snapshot, allow_short=False, use_xgboost=True)

    assert enter is False
    assert reason == "price_too_high"


def test_compute_ml_bias_fallbacks_without_xgboost(monkeypatch):
    from trader_app.strategy import compute_ml_bias

    monkeypatch.setattr("trader_app.strategy.xgb", None)

    frame = pd.DataFrame({"close": list(range(1, 101))})
    analyzed = add_moving_averages(frame, short_window=5, long_window=20)

    bias = compute_ml_bias(analyzed, 5, 20)

    assert bias in {"BUY", "SELL"}


def test_compute_ml_bias_fallbacks_for_default_windows_and_xgboost_unavailable(monkeypatch):
    from trader_app.strategy import compute_ml_bias

    monkeypatch.setattr("trader_app.strategy.xgb", None)

    frame = pd.DataFrame({"close": list(range(1, 251))})
    analyzed = add_moving_averages(frame, short_window=50, long_window=200)

    bias = compute_ml_bias(analyzed, 50, 200)

    assert bias in {"BUY", "SELL"}


def test_reward_ml_model_updates_preference():
    from trader_app.strategy import reward_ml_model, ml_bias_preference

    ml_bias_preference["BUY"] = 0.0
    ml_bias_preference["SELL"] = 0.0

    reward_ml_model("BUY", 10.0)
    assert ml_bias_preference["BUY"] == 0.1

    reward_ml_model("SELL", -5.0)
    assert ml_bias_preference["SELL"] == -0.05


def test_compute_ml_bias_uses_preference_adjustment_in_fallback(monkeypatch):
    from trader_app.strategy import compute_ml_bias, ml_bias_preference

    monkeypatch.setattr("trader_app.strategy.xgb", None)
    ml_bias_preference["BUY"] = 1.0
    ml_bias_preference["SELL"] = -1.0

    frame = pd.DataFrame({"close": list(range(1, 101))})
    analyzed = add_moving_averages(frame, short_window=5, long_window=20)

    bias = compute_ml_bias(analyzed, 5, 20)
    assert bias in {"BUY", "SELL"}


def test_compute_price_position_ranges():
    from trader_app.strategy import compute_price_position

    frame = pd.DataFrame({"close": [100, 95, 105, 90, 110]})
    low, high, position = compute_price_position(frame, lookback=5)

    assert low == 90.0
    assert high == 110.0
    assert position == 1.0


def test_should_enter_position_waits_when_price_is_too_high():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=5,
        order_book_bias="BUY",
        latest_close=110.0,
        best_bid=109.0,
        best_ask=111.0,
        long_ma=100.0,
        price_position=0.98,
    )

    enter, reason = should_enter_position(snapshot, allow_short=False, use_xgboost=True)

    assert enter is False
    assert reason == "price_too_high"


def test_should_enter_position_allows_buy_when_order_book_is_opposed_but_momentum_positive():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=20,
        order_book_bias="SELL",
        latest_close=110.0,
        best_bid=109.0,
        best_ask=111.0,
        long_ma=100.0,
        price_position=0.10,
        momentum=2.0,
    )

    enter, reason = should_enter_position(snapshot, allow_short=False, use_xgboost=False)

    assert enter is True
    assert reason == "signal_and_order_book"


def test_should_enter_position_allows_neutral_order_book_with_positive_signal():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=5,
        order_book_bias="NEUTRAL",
        latest_close=110.0,
        best_bid=109.0,
        best_ask=111.0,
        long_ma=100.0,
        price_position=0.60,
        momentum=1.0,
    )

    enter, reason = should_enter_position(snapshot, allow_short=False, use_xgboost=False)

    assert enter is True
    assert reason == "signal_and_order_book"


def test_should_exit_position_on_price_peak():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=5,
        order_book_bias="BUY",
        latest_close=110.0,
        best_bid=109.0,
        best_ask=111.0,
        long_ma=100.0,
        price_position=0.92,
    )

    should_exit, reason = should_exit_position(
        snapshot,
        BotState(has_position=True, last_entry_signal="BUY", entry_timestamp=0.0),
        None,
        10.0,
        Settings(use_xgboost=True),
    )

    assert should_exit is True
    assert reason == "price_peak"


def test_should_enter_position_requires_order_book_confirmation():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=20,
        order_book_bias="SELL",
        latest_close=300.0,
        best_bid=100.0,
        best_ask=101.0,
        long_ma=250.0,
    )

    enter, reason = should_enter_position(snapshot, allow_short=False, use_xgboost=False)

    assert enter is False
    assert reason == "order_book_conflict"


def test_should_enter_short_position_requires_bearish_confirmation():
    snapshot = MarketSnapshot(
        signal="SELL",
        bid_volume=10,
        ask_volume=20,
        order_book_bias="SELL",
        latest_close=200.0,
        best_bid=100.0,
        best_ask=101.0,
        long_ma=250.0,
    )

    enter, reason = should_enter_position(snapshot, allow_short=True, use_xgboost=False)

    assert enter is True
    assert reason == "signal_and_order_book"


def test_run_cycle_waits_when_buy_signal_lacks_order_book_confirmation():
    class FakeExchange:
        id = "fake"

        def fetch_ohlcv(self, symbol, timeframe):
            return [[index * 60_000, 0, 0, 0, price, 0] for index, price in enumerate(range(1, 301), start=1)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100, 1]], "asks": [[101, 4]]}

    outcome = run_cycle(Settings(execute_orders=True, order_amount=0.5), FakeExchange(), BotState())

    assert "decision=WAIT reason=order_book_conflict" in outcome.message
    assert outcome.terminate is False


def test_run_cycle_shorts_when_flat_and_signal_is_sell():
    class FakeExchange:
        id = "fake"

        def __init__(self):
            self.orders = []

        def fetch_ohlcv(self, symbol, timeframe):
            prices = list(range(300, 50, -1)) + [60, 65, 70, 75, 80]
            return [[index * 60_000, 0, 0, 0, price, 0] for index, price in enumerate(prices, start=1)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100, 1]], "asks": [[101, 4]]}

        def create_order(self, symbol, type, side, amount):
            self.orders.append({"symbol": symbol, "type": type, "side": side, "amount": amount})
            return {"id": "short123", "status": "closed", "filled": amount, "average": 100.0, "cost": 100.0}

    state = BotState()
    exchange = FakeExchange()

    outcome = run_cycle(Settings(execute_orders=True, order_amount=0.5, allow_short=True), exchange, state)

    assert "decision=SELL" in outcome.message
    assert "EXECUTED SELL 0.5 BTC/USDT order_id=short123 status=closed" in outcome.message
    assert state.has_position is True
    assert state.last_entry_signal == "SELL"
    assert exchange.orders[0]["side"] == "sell"


def test_should_exit_position_sells_on_bearish_order_book():
    class Snapshot:
        signal = "BUY"
        order_book_bias = "SELL"

    should_exit, reason = should_exit_position(
        Snapshot(),
        BotState(has_position=True, entry_timestamp=0.0),
        None,
        10.0,
        Settings(use_xgboost=False),
    )

    assert should_exit is True
    assert reason == "sell_pressure"


def test_should_exit_position_sells_when_ml_bias_reverses():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=5,
        order_book_bias="BUY",
        latest_close=300.0,
        best_bid=100.0,
        best_ask=101.0,
        long_ma=250.0,
        ml_bias="SELL",
    )

    should_exit, reason = should_exit_position(
        snapshot,
        BotState(has_position=True, last_entry_signal="BUY", entry_timestamp=0.0),
        None,
        10.0,
        Settings(use_xgboost=True),
    )

    assert should_exit is True
    assert reason == "ml_signal"


def test_should_exit_position_respects_long_stop_loss():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=5,
        order_book_bias="BUY",
        latest_close=95.0,
        best_bid=94.0,
        best_ask=96.0,
        long_ma=100.0,
    )

    should_exit, reason = should_exit_position(
        snapshot,
        BotState(has_position=True, last_entry_signal="BUY", entry_price=100.0, entry_timestamp=0.0),
        None,
        10.0,
        Settings(use_xgboost=False, stop_loss=0.05, take_profit=0.10),
    )

    assert should_exit is True
    assert reason == "stop_loss"


def test_should_exit_position_reaches_long_take_profit():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=5,
        order_book_bias="BUY",
        latest_close=110.0,
        best_bid=109.0,
        best_ask=111.0,
        long_ma=100.0,
    )

    should_exit, reason = should_exit_position(
        snapshot,
        BotState(has_position=True, last_entry_signal="BUY", entry_price=100.0, entry_timestamp=0.0),
        None,
        10.0,
        Settings(use_xgboost=False, stop_loss=0.05, take_profit=0.10),
    )

    assert should_exit is True
    assert reason == "take_profit"


def test_should_exit_position_respects_short_stop_loss():
    snapshot = MarketSnapshot(
        signal="SELL",
        bid_volume=5,
        ask_volume=10,
        order_book_bias="SELL",
        latest_close=105.0,
        best_bid=104.0,
        best_ask=106.0,
        long_ma=100.0,
    )

    should_exit, reason = should_exit_position(
        snapshot,
        BotState(has_position=True, last_entry_signal="SELL", entry_price=100.0, entry_timestamp=0.0),
        None,
        10.0,
        Settings(use_xgboost=False, stop_loss=0.05, take_profit=0.10),
    )

    assert should_exit is True
    assert reason == "stop_loss"


def test_should_exit_position_reaches_short_take_profit():
    snapshot = MarketSnapshot(
        signal="SELL",
        bid_volume=5,
        ask_volume=10,
        order_book_bias="SELL",
        latest_close=90.0,
        best_bid=89.0,
        best_ask=91.0,
        long_ma=100.0,
    )

    should_exit, reason = should_exit_position(
        snapshot,
        BotState(has_position=True, last_entry_signal="SELL", entry_price=100.0, entry_timestamp=0.0),
        None,
        10.0,
        Settings(use_xgboost=False, stop_loss=0.05, take_profit=0.10),
    )

    assert should_exit is True
    assert reason == "take_profit"


def test_should_enter_position_rejects_long_when_long_ma_slopes_down():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10,
        ask_volume=5,
        order_book_bias="BUY",
        latest_close=110.0,
        best_bid=109.0,
        best_ask=111.0,
        long_ma=100.0,
        price_position=0.50,
        momentum=1.0,
        long_ma_slope=-0.5,
    )

    enter, reason = should_enter_position(snapshot, allow_short=False, use_xgboost=False)

    assert enter is False
    assert reason == "trend_down"


def test_should_exit_position_sells_when_max_hold_expires():
    class Snapshot:
        signal = "BUY"
        order_book_bias = "BUY"

    should_exit, reason = should_exit_position(
        Snapshot(),
        BotState(has_position=True, entry_timestamp=0.0),
        1800,
        1800.0,
        Settings(use_xgboost=False),
    )

    assert should_exit is True
    assert reason == "max_hold"


def test_run_cycle_buys_when_flat_and_signal_is_buy():
    class FakeExchange:
        id = "fake"

        def __init__(self):
            self.orders = []

        def fetch_ohlcv(self, symbol, timeframe):
            prices = list(range(40, 240)) + [190] * 100
            return [[index * 60_000, 0, 0, 0, price, 0] for index, price in enumerate(prices, start=1)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100, 9]], "asks": [[101, 4]]}

        def create_order(self, symbol, type, side, amount):
            self.orders.append(
                {"symbol": symbol, "type": type, "side": side, "amount": amount}
            )
            return {"id": "buy123", "status": "closed", "filled": 0.5, "average": 100.0, "cost": 50.0}

    state = BotState()
    exchange = FakeExchange()

    outcome = run_cycle(Settings(execute_orders=True, order_amount=0.5), exchange, state)

    assert "decision=BUY" in outcome.message
    assert "EXECUTED BUY 0.5 BTC/USDT order_id=buy123 status=closed" in outcome.message
    assert outcome.terminate is False
    assert state.has_position is True
    assert state.entry_timestamp is not None
    assert state.entry_price == 100.0
    assert state.entry_amount == 0.5
    assert state.entry_cost == 50.0
    assert exchange.orders[0]["side"] == "buy"


def test_run_cycle_waits_when_flat_and_signal_is_sell():
    class FakeExchange:
        id = "fake"

        def fetch_ohlcv(self, symbol, timeframe):
            return [[index * 60_000, 0, 0, 0, price, 0] for index, price in enumerate(range(300, 0, -1), start=1)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100, 9]], "asks": [[101, 4]]}

    outcome = run_cycle(Settings(), FakeExchange(), BotState())

    assert "decision=WAIT" in outcome.message


def test_run_cycle_uses_fallback_price_for_profit_and_terminates_on_max_hold():
    class FakeExchange:
        id = "fake"

        def __init__(self):
            self.orders = []

        def fetch_ohlcv(self, symbol, timeframe):
            return [[index * 60_000, 0, 0, 0, 110, 0] for index in range(1, 301)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[111, 5]], "asks": [[112, 3]]}

        def create_order(self, symbol, type, side, amount):
            self.orders.append(
                {"symbol": symbol, "type": type, "side": side, "amount": amount}
            )
            return {"id": "sellmax", "status": "closed", "filled": amount}

    state = BotState(
        has_position=True,
        entry_timestamp=0.0,
        entry_price=100.0,
        entry_amount=0.4,
        entry_cost=40.0,
    )
    outcome = run_cycle(
        Settings(execute_orders=True, order_amount=0.1, max_hold="5m"),
        FakeExchange(),
        state,
    )

    assert "reason=max_hold" in outcome.message
    assert "profit=4.400000 quote_currency profit_pct=11.00%" in outcome.message
    assert outcome.terminate is False


def test_liquidate_position_sells_full_held_amount_and_terminates():
    class FakeExchange:
        id = "fake"

        def __init__(self):
            self.orders = []

        def fetch_ohlcv(self, symbol, timeframe):
            return [[index * 60_000, 0, 0, 0, 110, 0] for index in range(1, 301)]

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[111, 5]], "asks": [[112, 3]]}

        def create_order(self, symbol, type, side, amount):
            self.orders.append({"symbol": symbol, "type": type, "side": side, "amount": amount})
            return {"id": "manual1", "status": "closed", "filled": amount}

    state = BotState(
        has_position=True,
        entry_price=100.0,
        entry_amount=0.4,
        entry_cost=40.0,
        entry_timestamp=0.0,
    )
    exchange = FakeExchange()

    outcome = liquidate_position(Settings(execute_orders=True, order_amount=0.1), exchange, state, "manual_cashout")

    assert outcome.terminate is True
    assert "reason=manual_cashout" in outcome.message
    assert exchange.orders == [{"symbol": "BTC/USDT", "type": "market", "side": "sell", "amount": 0.4}]
    assert state.has_position is False


def test_handle_user_command_cashout_without_position_stops_cleanly():
    outcome = handle_user_command(Settings(), object(), BotState(), "cashout")

    assert outcome.message == "MANUAL | no open position | decision=STOP"
    assert outcome.terminate is True


def test_handle_user_command_status_reports_state():
    state = BotState(has_position=True, last_entry_signal="BUY", entry_timestamp=1.0)

    outcome = handle_user_command(Settings(), object(), state, "status")

    assert "has_position=True" in outcome.message


def test_format_decision_summary_includes_ml_bias_when_enabled():
    snapshot = MarketSnapshot(
        signal="BUY",
        bid_volume=10.0,
        ask_volume=5.0,
        order_book_bias="BUY",
        latest_close=300.0,
        best_bid=100.0,
        best_ask=101.0,
        long_ma=250.0,
        ml_bias="BUY",
    )

    summary = format_decision_summary(snapshot, use_xgboost=True)

    assert "ml_bias=BUY" in summary


def test_format_decision_summary_omits_ml_bias_when_disabled():
    snapshot = MarketSnapshot(
        signal="SELL",
        bid_volume=5.0,
        ask_volume=10.0,
        order_book_bias="SELL",
        latest_close=200.0,
        best_bid=99.0,
        best_ask=100.0,
        long_ma=250.0,
        ml_bias="SELL",
    )

    summary = format_decision_summary(snapshot, use_xgboost=False)

    assert "ml_bias" not in summary


def test_execute_signal_returns_failed_message_when_order_creation_fails():
    class FakeExchange:
        def create_order(self, symbol, type, side, amount):
            raise ccxt.ExchangeError(
                'bybit {"retCode":170381,"retMsg":"The quantity of a single market order must be less than the maximum allowed per order: 120BTC.","result":{},"retExtInfo":{},"time":1775911454193}'
            )

    message = execute_signal(FakeExchange(), "BTC/USDT", "BUY", 10000.0, live=True)

    assert "FAILED BUY 10000.0 BTC/USDT" in message
    assert "The quantity of a single market order must be less than the maximum allowed per order" in message


def test_create_exchange_enables_sandbox_mode(monkeypatch):
    captured = {}

    class FakeExchange:
        id = "fake"

        def __init__(self, options):
            captured["options"] = options
            captured["sandbox_enabled"] = False

        def set_sandbox_mode(self, enabled):
            captured["sandbox_enabled"] = enabled

    import trader_app.data as data_module

    monkeypatch.setattr(data_module.ccxt, "fakeexchange", FakeExchange, raising=False)

    exchange = create_exchange(
        "fakeexchange",
        api_key="key",
        api_secret="secret",
        api_password="password",
        sandbox=True,
    )

    assert exchange.id == "fake"
    assert captured["options"] == {
        "enableRateLimit": True,
        "apiKey": "key",
        "secret": "secret",
        "password": "password",
    }
    assert captured["sandbox_enabled"] is True


def test_create_exchange_enables_bybit_demo_mode(monkeypatch):
    captured = {}

    class FakeExchange:
        id = "bybit"

        def __init__(self, options):
            captured["options"] = options
            captured["demo_enabled"] = False

        def enable_demo_trading(self, enabled):
            captured["demo_enabled"] = enabled

    import trader_app.data as data_module

    monkeypatch.setattr(data_module.ccxt, "bybit", FakeExchange, raising=False)

    exchange = create_exchange("bybit", api_key="key", api_secret="secret", demo=True)

    assert exchange.id == "bybit"
    assert captured["options"] == {
        "enableRateLimit": True,
        "apiKey": "key",
        "secret": "secret",
    }
    assert captured["demo_enabled"] is True


def test_describe_mode_reports_sandbox_and_live_execution():
    settings = Settings(exchange_id="bybit", sandbox=True, execute_orders=True)

    message = describe_mode(settings)

    assert message == (
        "Starting bot on exchange=bybit environment=sandbox "
        "execution=live symbol=BTC/USDT"
    )


def test_describe_mode_reports_demo_environment():
    settings = Settings(exchange_id="bybit", demo=True, execute_orders=True)

    message = describe_mode(settings)

    assert message == (
        "Starting bot on exchange=bybit environment=demo "
        "execution=live symbol=BTC/USDT"
    )


def test_describe_mode_includes_api_base_url_when_exchange_is_provided():
    class FakeExchange:
        urls = {"api": {"public": "https://api-demo.{hostname}"}}

        def implode_hostname(self, url):
            return url.replace("{hostname}", "bybit.com")

    settings = Settings(exchange_id="bybit", demo=True, execute_orders=True)

    message = describe_mode(settings, exchange=FakeExchange())

    assert message == (
        "Starting bot on exchange=bybit environment=demo "
        "execution=live symbol=BTC/USDT api=https://api-demo.bybit.com"
    )


def test_describe_state_file_reports_resolved_path(tmp_path):
    settings = Settings(state_file=str(tmp_path / "bot_state.json"))
    state = BotState(
        has_position=True,
        last_entry_signal="BUY",
        entry_timestamp=123.0,
        entry_price=100.0,
        entry_amount=0.5,
    )

    message = describe_state_file(settings, state)

    assert f"State file={(tmp_path / 'bot_state.json').resolve()}" in message
    assert "has_position=True" in message
    assert "last_entry_signal=BUY" in message
    assert "entry_timestamp=123.0" in message
    assert "entry_price=100.0" in message
    assert "entry_amount=0.5" in message


def test_format_auth_error_mentions_environment_mismatch():
    settings = Settings(exchange_id="bybit", sandbox=True)

    message = format_auth_error(
        settings,
        ccxt.AuthenticationError('bybit {"retCode":10003,"retMsg":"API key is invalid."}'),
    )

    assert "sandbox/testnet mode" in message
    assert "match the selected environment" in message


def test_format_realized_profit_supports_short_positions():
    state = BotState(last_entry_signal="SELL", entry_amount=1.0, entry_cost=100.0)
    exit_execution = OrderExecution(success=True, message="", filled_amount=1.0, average_price=90.0, cost=90.0)

    assert format_realized_profit(state, exit_execution) == "profit=10.000000 quote_currency profit_pct=10.00%"


def test_format_auth_error_mentions_demo_environment():
    settings = Settings(exchange_id="bybit", demo=True)

    message = format_auth_error(
        settings,
        ccxt.AuthenticationError('bybit {"retCode":10003,"retMsg":"API key is invalid."}'),
    )

    assert "demo mode" in message
    assert "selected environment" in message


def test_fetch_exchange_preflight_reports_bybit_permissions():
    class FakeExchange:
        id = "bybit"

        def privateGetV5UserQueryApi(self, params):
            return {
                "result": {
                    "readOnly": 1,
                    "permissions": {"Spot": ["SpotTrade"]},
                    "ips": ["*"],
                }
            }

    message = fetch_exchange_preflight(FakeExchange())

    assert message == (
        "Bybit API key info: read_only=1 spot_permissions=['SpotTrade'] ips=['*']"
    )


def test_run_bot_returns_one_on_authentication_error(monkeypatch, capsys):
    class FakeExchange:
        id = "bybit"
        urls = {"api": {"public": "https://api-testnet.{hostname}"}}

        def implode_hostname(self, url):
            return url.replace("{hostname}", "bybit.com")

        def privateGetV5UserQueryApi(self, params):
            return {
                "result": {
                    "readOnly": 0,
                    "permissions": {"Spot": ["SpotTrade"]},
                    "ips": ["*"],
                }
            }

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100, 1]], "asks": [[101, 2]]}

    def fake_create_exchange(**kwargs):
        return FakeExchange()

    def fake_run_cycle(settings, exchange, state):
        raise ccxt.AuthenticationError("API key is invalid.")

    import trader_app.bot as bot_module

    monkeypatch.setattr(bot_module, "create_exchange", fake_create_exchange)
    monkeypatch.setattr(bot_module, "run_cycle", fake_run_cycle)

    exit_code = run_bot(Settings(exchange_id="bybit", sandbox=True, execute_orders=True))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "environment=sandbox execution=live symbol=BTC/USDT api=https://api-testnet.bybit.com" in captured.out
    assert "Authentication failed for exchange=bybit in sandbox/testnet mode" in captured.out
