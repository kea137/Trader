# Trader

A small Python trading bot that generates a moving-average signal from exchange OHLCV data and manages trades using order-book bias, stop-loss, take-profit, and optional shorting.

## Overview

This bot is designed for demo and paper trading. It:

- fetches OHLCV data from a CCXT-supported exchange
- computes short and long moving averages
- generates a `BUY` signal when the short moving average is above the long moving average
- generates a `SELL` signal when the short moving average is below the long moving average
- optionally enters short trades with `--allow-short`
- supervises open positions until an exit condition is met

The bot uses market orders and keeps a persistent state file so it can continue supervising a trade across restarts.

## Features

- dry-run mode by default
- live execution with `--execute`
- Bybit demo mode with `--demo`
- optional short selling with `--allow-short`
- stop-loss and take-profit thresholds
- ATR-based dynamic stop-loss and take-profit with `--use-atr-stops`
- trailing stop that locks in profits with `--use-trailing-stop`
- ATR-based volatility position sizing with `--use-atr-sizing`
- maximum hold timer
- order-book confirmation for entry and exit decisions
- optional XGBoost-based bias filtering with `--use-xgboost`
- MACD momentum confirmation for entries
- multi-indicator confluence scoring with `--confluence-threshold`
- volume confirmation filter with `--volume-confirmation`
- RSI overbought/oversold filter with `--rsi-filter`
- ADX trend-strength gate with `--min-adx`
- max drawdown protection with `--max-drawdown`
- max daily loss limit with `--max-daily-loss`
- loss cooldown between trades with `--loss-cooldown`
- interactive terminal commands while the bot runs
- CSV trade logging with `--record-file`

## Requirements

- Python 3.10+
- `ccxt`
- `pandas`
- `pytest` for tests
- Optional: `xgboost` if you want ML-based signal confirmation

Install dependencies:

```bash
pip install ccxt pandas pytest
```

If you want XGBoost support:

```bash
pip install xgboost
```

## Running the bot

### Interactive wizard (recommended)

Running with no arguments launches the step-by-step setup wizard:

```bash
python3 trader.py
```

The wizard walks through exchange, market, strategy profile, execution mode,
API credentials, and file paths — then shows a summary before starting.
All CLI flags below can also be passed directly to skip the wizard.

### Continuous polling

```bash
python3 trader.py --poll-seconds 60
```

### Live trading

```bash
export TRADER_API_KEY=your_key
export TRADER_API_SECRET=your_secret
python3 trader.py --execute --order-amount 0.001 --poll-seconds 60
```

### Bybit demo trading

```bash
export TRADER_API_KEY=your_demo_key
export TRADER_API_SECRET=your_demo_secret
python3 trader.py --exchange bybit --demo --execute --order-amount 0.001
```

### Use a custom state file or record file

```bash
python3 trader.py --state-file state/bybit-btcusdt-demo.json --record-file logs/bybit-btcusdt.csv --poll-seconds 60
```

## Recommended conservative command

```bash
python3 trader.py \
  --exchange bybit \
  --demo \
  --symbol BTC/USDT \
  --timeframe 4h \
  --short-window 20 \
  --long-window 50 \
  --poll-seconds 60 \
  --use-atr-stops \
  --atr-sl-multiplier 2.0 \
  --atr-tp-multiplier 3.0 \
  --use-trailing-stop \
  --confluence-threshold 3 \
  --volume-confirmation \
  --rsi-filter \
  --min-adx 25 \
  --max-drawdown 0.05 \
  --loss-cooldown 300 \
  --use-atr-sizing \
  --atr-risk-pct 0.01 \
  --record-file logs/conservative.csv
```

For the full trading guide, strategy explanation, and multiple optimized
command profiles, see **[TRADING.md](TRADING.md)**.

## Command-line options

- `--exchange`: CCXT exchange id (default: `binance`)
- `--symbol`: market symbol (default: `BTC/USDT`)
- `--timeframe`: candle timeframe (default: `1h`)
- `--short-window`: short moving-average period (default: `50`)
- `--long-window`: long moving-average period (default: `200`)
- `--order-amount`: base asset amount for market orders
- `--poll-seconds`: seconds between market checks
- `--order-book-depth`: depth of bids/asks fetched for supervision
- `--sell-pressure-ratio`: ask/bid volume ratio threshold for exit bias
- `--state-file`: path to the JSON state file
- `--record-file`: path to a CSV file that logs market snapshots and equity records each cycle
- `--max-hold`: max time to hold a position, e.g. `30m`, `1h`
- `--stop-loss`: stop-loss fraction, e.g. `0.01` for 1%
- `--take-profit`: take-profit fraction, e.g. `0.02` for 2%
- `--allow-short`: permit short entries when signal and order book agree
- `--use-xgboost`: enable optional XGBoost bias filtering
- `--execute`: place real orders instead of dry-run
- `--sandbox`: use exchange sandbox/testnet when supported
- `--demo`: use Bybit demo trading
- `--use-atr-stops`: replace fixed SL/TP with ATR-based stops
- `--atr-sl-multiplier`: ATR multiples for stop-loss (default: `2.0`)
- `--atr-tp-multiplier`: ATR multiples for take-profit (default: `3.0`)
- `--use-trailing-stop`: enable trailing stop using ATR distance
- `--trail-atr-multiplier`: trailing stop ATR distance (default: `2.0`)
- `--use-atr-sizing`: size positions using volatility-based sizing
- `--atr-risk-pct`: fraction of equity to risk per trade (default: `0.01`)
- `--min-adx`: minimum ADX trend strength to enter (default: `0`, recommended: `25`)
- `--rsi-filter`: block entries at RSI extremes (>70 for BUY, <30 for SELL)
- `--loss-cooldown`: seconds to pause after a losing trade (default: `0`)
- `--confluence-threshold`: minimum confirming indicators needed to enter (default: `0`, recommended: `2–3`)
- `--volume-confirmation`: apply a -1 confluence penalty when volume is below the 20-period average (entry harder, not impossible)
- `--max-daily-loss`: max fraction of equity lost per day before stopping (default: `0`)
- `--max-drawdown`: max fraction drawdown from peak equity before stopping (default: `0`)

## How the bot decides

### Entry logic

- `BUY` when the short MA is above the long MA
- `SELL` when the short MA is below the long MA
- `BUY` entries require buy-side order-book pressure unless overridden by momentum/price position
- `SELL` entries require sell-side order-book pressure when shorts are enabled
- long entries are blocked if the long MA slope is negative
- short entries are blocked if the long MA slope is positive
- MACD histogram must agree with direction (when order book is neutral)
- confluence threshold must be met (when configured)
- low volume applies a -1 penalty to the confluence score (when `--volume-confirmation` is set), making entry harder rather than impossible
- RSI must not be at extremes (when `--rsi-filter` is set)
- ADX must exceed the minimum (when `--min-adx` is set)

### Exit logic

For an open long position, the bot exits when:

- the signal flips to `SELL`
- the close price hits the stop-loss threshold
- the close price hits the take-profit threshold
- order-book sell pressure exceeds the configured ratio
- momentum turns negative

For an open short position, the bot exits when:

- the signal flips to `BUY`
- the close price hits the stop-loss threshold
- the close price hits the take-profit threshold
- order-book buy pressure exceeds the configured ratio
- momentum turns positive

## Interactive commands

While the bot is running, type:

- `help` — show available commands
- `status` — print current position state
- `history` — display the last 12 trades from the record file
- `cashout` — exit any open position and terminate
- `stop` — stop the bot without opening a new trade

## Risk guidance

This bot is intended for demo and paper trading. It is not a production trading system.

- always validate on sandbox/demo before real funds
- do not use large order sizes until the strategy is proven
- avoid enabling both `--allow-short` and `--use-xgboost` until you understand the signal behavior
- start with clean state files and small position sizes
- run for many trades before considering live risk

## Testing

Run the test suite with:

```bash
PYTHONPATH=. pytest -q tests/test_strategy.py
```

## Project structure

- `trader.py`: entry point
- `trader_app/config.py`: default settings
- `trader_app/data.py`: CCXT exchange and market data helpers
- `trader_app/strategy.py`: signal generation, indicators (RSI, MACD, ADX, ATR, Bollinger Bands, VWAP), confluence scoring, ML bias
- `trader_app/bot.py`: bot loop, position supervision, execution logic, drawdown protection
- `trader_app/cli.py`: command-line argument handling
- `tests/test_strategy.py`: strategy and bot regression tests
- `TRADING.md`: comprehensive trading guide with optimized command profiles

## Important note

The bot always uses market orders and does not manage exchange fees, slippage, or partial fills beyond the CCXT order response. Real trading carries risk. Use this code for experimentation and learn from demo trading before considering live deployment.
