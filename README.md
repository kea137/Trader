# Trader

Small Python trading bot that derives a moving-average signal from exchange
OHLCV data, enters a long trade on a buy signal, and supervises that position
until conditions justify an exit.

## Structure

- `trader.py`: runnable entry point
- `trader_app/config.py`: default runtime settings
- `trader_app/data.py`: exchange client creation and market-data loading
- `trader_app/strategy.py`: moving-average calculations and signal generation
- `trader_app/bot.py`: signal execution, order-book supervision, and bot loop
- `trader_app/cli.py`: command-line argument handling and application wiring
- `tests/test_strategy.py`: basic strategy behavior checks

## Usage

Install dependencies:

```bash
pip install ccxt pandas pytest
```

Run the bot in dry-run mode:

```bash
source .venv/bin/activate
python3 trader.py
```

Run continuously every 60 seconds:

```bash
python3 trader.py --poll-seconds 60
```

While the bot is running interactively, you can type:

```text
help
status
cashout
stop
```

Enable live trading:

```bash
export TRADER_API_KEY=your_key
export TRADER_API_SECRET=your_secret
python3 trader.py --execute --order-amount 0.001 --poll-seconds 60
```

Use Bybit demo or another CCXT-supported sandbox:

```bash
export TRADER_API_KEY=your_demo_key
export TRADER_API_SECRET=your_demo_secret
python3 trader.py --exchange bybit --demo --execute --order-amount 0.001
```

Use a custom state file if you want to separate sessions or symbols:

```bash
python3 trader.py --state-file state/bybit-btcusdt.json --poll-seconds 60
```

Force a cash-out after a maximum hold time:

```bash
python3 trader.py --poll-seconds 60 --max-hold 30m
python3 trader.py --poll-seconds 60 --max-hold 1h
```

Override strategy settings from the command line:

```bash
python3 trader.py --exchange binance --symbol ETH/USDT --timeframe 4h --short-window 20 --long-window 100 --order-amount 0.01
```

## Notes

- Without `--execute`, the bot stays in dry-run mode and only prints the order it would place.
- `--sandbox` tells CCXT to switch to the exchange's testnet environment when supported.
- `--demo` enables Bybit demo trading through `api-demo.bybit.com`.
- Live execution uses market orders for the configured `--order-amount`.
- The bot now requires both a `BUY` moving-average signal and buy-side order-book pressure before opening a new long position, making entries more conservative.
- The bot can also open short positions when run with `--allow-short`. In that mode it shorts on a `SELL` signal only when order-book pressure also supports selling.
- A held long position is sold when either the moving-average signal flips to `SELL` or the top-of-book asks outweigh bids by the configured sell-pressure threshold.
- A held short position is covered when the signal flips to `BUY` or when buy-side order-book pressure dominates.
- The bot persists its position state in `bot_state.json` by default, so a restart keeps supervising the previous trade state.
- Use `--state-file` to isolate state per exchange, symbol, or environment.
- Use `--max-hold 5m`, `--max-hold 30m`, or `--max-hold 3h` to force a sell once a position has been held that long.
- Timed cash-outs sell the full persisted position size, not just the configured default order amount.
- Automatic exits do not stop the bot; it keeps running and can look for the next trade.
- In an interactive terminal session, `cashout` liquidates the open position immediately, prints profit, and exits. `stop` exits without placing a new order.
- When a trade closes, the bot prints realized profit based on entry and exit pricing. If the exchange omits fill price or cost fields, the bot falls back to current market prices so profit still shows more reliably.
- The signal is based on the latest short/long moving-average relationship, not on crossover detection.
- Exchange APIs, order sizing rules, fees, and market-buy behavior vary. Validate on a paper or sandbox account before using real funds.
