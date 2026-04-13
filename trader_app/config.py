from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    exchange_id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    short_window: int = 50
    long_window: int = 200
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_password: Optional[str] = None
    order_amount: float = 0.001
    execute_orders: bool = False
    sandbox: bool = False
    demo: bool = False
    allow_short: bool = False
    use_xgboost: bool = False
    poll_seconds: int = 0
    order_book_depth: int = 5
    sell_pressure_ratio: float = 1.2
    state_file: str = "bot_state.json"
    max_hold: str | None = None
    stop_loss: float = 0.01
    take_profit: float = 0.02


DEFAULT_SETTINGS = Settings()
