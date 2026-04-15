from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

ml_bias_preference = {"BUY": 0.0, "SELL": 0.0}

FEATURE_COLUMNS = [
    "ma_short",
    "ma_long",
    "ma_ratio",
    "momentum",
    "volatility",
    "price_change",
    "recent_low",
    "recent_high",
    "position_in_range",
    "volume_change",
    "volume_ratio",
    # New technical features
    "rsi",
    "atr",
    "bb_position",
    "ema_cross",
    "adx",
    "vwap_distance",
]

# —————————————————————————
# Technical indicator helpers
# —————————————————————————


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100). <30 = oversold, >70 = overbought."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — volatility in price units."""
    high = frame["high"] if "high" in frame.columns else frame["close"]
    low = frame["low"] if "low" in frame.columns else frame["close"]
    close = frame["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period).mean()
    return atr.fillna(0.0)


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower) Bollinger Bands."""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_vwap(frame: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP using typical price."""
    typical = (
        frame.get("high", frame["close"]) + frame.get("low", frame["close"]) + frame["close"]
    ) / 3
    vol = frame["volume"] if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
    cum_tp_vol = (typical * vol).cumsum()
    cum_vol = vol.cumsum().replace(0, np.nan)
    return (cum_tp_vol / cum_vol).fillna(typical)


def compute_adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX trend-strength indicator (0-100). >25 = trending, >40 = strong trend."""
    high = frame["high"] if "high" in frame.columns else frame["close"]
    low = frame["low"] if "low" in frame.columns else frame["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )
    atr = compute_atr(frame, period)
    safe_atr = atr.replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(span=period, min_periods=period).mean() / safe_atr
    minus_di = 100 * minus_dm.ewm(span=period, min_periods=period).mean() / safe_atr
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(span=period, min_periods=period).mean()
    return adx.fillna(0.0)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span).mean()

# —————————————————————————
# ATR-based risk helpers
# —————————————————————————


def compute_atr_stops(
    entry_price: float,
    atr: float,
    sl_multiplier: float = 2.0,
    tp_multiplier: float = 3.0,
    is_short: bool = False,
) -> tuple[float, float]:
    """Returns (stop_loss_price, take_profit_price) using ATR multiples.

    Risk:reward defaults to 1:1.5 (2 ATR stop, 3 ATR target).
    """
    if is_short:
        return entry_price + sl_multiplier * atr, entry_price - tp_multiplier * atr
    return entry_price - sl_multiplier * atr, entry_price + tp_multiplier * atr


def compute_trailing_stop(
    highest_extreme: float,
    atr: float,
    trail_multiplier: float = 2.0,
    is_short: bool = False,
) -> float:
    """Compute the trailing stop price.

    highest_extreme: highest close seen for a long, lowest close seen for a short.
    """
    if is_short:
        return highest_extreme + trail_multiplier * atr
    return highest_extreme - trail_multiplier * atr


def compute_volatility_position_size(
    equity: float,
    price: float,
    atr: float,
    risk_pct: float = 0.01,
    min_amount: float = 0.001,
    max_amount: float = 10.0,
) -> float:
    """Kelly-lite position sizing: risk risk_pct of equity per ATR of adverse move."""
    if atr <= 0 or price <= 0 or equity <= 0:
        return min_amount
    risk_quote = equity * risk_pct
    amount_base = risk_quote / atr
    return float(np.clip(amount_base, min_amount, max_amount))

# —————————————————————————
# Core MA functions (unchanged API)
# —————————————————————————


def _preference_adjustment() -> float:
    return (ml_bias_preference["BUY"] - ml_bias_preference["SELL"]) * 0.01


def add_moving_averages(
    frame: pd.DataFrame, short_window: int, long_window: int
) -> pd.DataFrame:
    if short_window <= 0 or long_window <= 0:
        raise ValueError("Moving-average windows must be positive integers.")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window.")
    analyzed = frame.copy()
    analyzed["ma_short"] = analyzed["close"].rolling(short_window).mean()
    analyzed["ma_long"] = analyzed["close"].rolling(long_window).mean()
    return analyzed


def latest_signal(frame: pd.DataFrame) -> str:
    latest = frame.iloc[-1]
    if pd.isna(latest["ma_short"]) or pd.isna(latest["ma_long"]):
        raise ValueError("Not enough data to compute the configured moving averages.")
    return "BUY" if latest["ma_short"] > latest["ma_long"] else "SELL"

# —————————————————————————
# Feature engineering
# —————————————————————————


def build_ml_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    for col in ("volume", "high", "low"):
        if col not in features.columns:
            features[col] = features["close"]

    features["ma_ratio"] = (
        (features["ma_short"] - features["ma_long"]) / features["ma_long"].replace(0, 1.0)
    )
    rolling_low = features["close"].rolling(20, min_periods=1).min()
    rolling_high = features["close"].rolling(20, min_periods=1).max()
    features["recent_low"] = rolling_low
    features["recent_high"] = rolling_high
    features["position_in_range"] = (
        (features["close"] - rolling_low) / (rolling_high - rolling_low).replace(0, 1.0)
    )
    features["momentum"] = features["close"].diff().fillna(0.0)
    features["volatility"] = features["close"].rolling(10).std().fillna(0.0)
    features["price_change"] = features["close"].pct_change().fillna(0.0)
    features["volume_change"] = features["volume"].diff().fillna(0.0)
    rolling_volume = features["volume"].rolling(20, min_periods=1).mean().replace(0, 1.0)
    features["volume_ratio"] = features["volume"] / rolling_volume

    features["rsi"] = compute_rsi(features["close"], period=14)
    features["atr"] = compute_atr(features, period=14)

    bb_upper, _, bb_lower = compute_bollinger_bands(features["close"], period=20)
    bb_range = (bb_upper - bb_lower).replace(0, 1.0)
    features["bb_position"] = (features["close"] - bb_lower) / bb_range

    ema_fast = compute_ema(features["close"], span=9)
    ema_slow = compute_ema(features["close"], span=21)
    features["ema_cross"] = (ema_fast - ema_slow) / features["close"].replace(0, 1.0)

    features["adx"] = compute_adx(features, period=14)

    vwap = compute_vwap(features)
    features["vwap_distance"] = (features["close"] - vwap) / vwap.replace(0, 1.0)
    return features

# —————————————————————————
# ML bias computation
# —————————————————————————


def compute_fallback_ml_bias(
    frame: pd.DataFrame,
    short_window: int,
    long_window: int,
    order_book_imbalance: float = 0.0,
    spread: float = 0.0,
) -> str:
    if len(frame) < long_window + 10:
        raise ValueError("Not enough historical data to build a fallback ML model.")

    features = build_ml_features(frame).dropna(subset=["ma_short", "ma_long"])
    if len(features) < 6:
        raise ValueError("Not enough cleaned data to build fallback ML features.")

    feature_columns = [c for c in FEATURE_COLUMNS if c in features.columns]
    X = features[feature_columns].astype(float).iloc[:-1]
    y = (features["close"].shift(-1) > features["close"]).astype(int).iloc[:-1]

    if len(X) < 5 or len(y) < 5:
        raise ValueError("Not enough aligned training rows for fallback ML bias.")

    n = len(X)
    sample_weights = np.linspace(0.5, 1.0, n)
    X_arr = X.to_numpy()
    y_arr = y.to_numpy().astype(float)

    # Replace non-finite values to prevent SVD failures
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)

    X_w = X_arr * sample_weights[:, None]
    y_w = y_arr * sample_weights
    try:
        weights = np.linalg.pinv(X_w) @ y_w
    except np.linalg.LinAlgError:
        try:
            weights = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Degenerate matrix — fall back to simple mean bias
            score = float(y_arr.mean())
            score += _preference_adjustment()
            score += order_book_imbalance * 0.2
            score -= spread * 0.5
            return "BUY" if score >= 0.5 else "SELL"

    latest_features = features.iloc[[-1]][feature_columns].astype(float).to_numpy()[0]
    score = float(np.dot(latest_features, weights))
    score += _preference_adjustment()
    score += order_book_imbalance * 0.2
    score -= spread * 0.5
    return "BUY" if score >= 0.5 else "SELL"


def reward_ml_model(predicted_signal: str, profit: float) -> None:
    if predicted_signal not in {"BUY", "SELL"}:
        return
    adjustment = float(profit) * 0.01
    if profit > 0:
        ml_bias_preference[predicted_signal] = min(
            10.0, ml_bias_preference[predicted_signal] + adjustment
        )
    else:
        ml_bias_preference[predicted_signal] = max(
            -10.0, ml_bias_preference[predicted_signal] - abs(adjustment)
        )


def compute_price_position(
    frame: pd.DataFrame, lookback: int = 20
) -> tuple[float, float, float]:
    if len(frame) < 1:
        raise ValueError("Frame must contain at least one row to compute price position.")
    window = frame["close"].iloc[-lookback:]
    recent_low = float(window.min())
    recent_high = float(window.max())
    latest = float(window.iloc[-1])
    if recent_high > recent_low:
        position = (latest - recent_low) / (recent_high - recent_low)
    else:
        position = 0.5
    return recent_low, recent_high, position


def compute_ml_bias(
    frame: pd.DataFrame,
    short_window: int,
    long_window: int,
    order_book_imbalance: float = 0.0,
    spread: float = 0.0,
) -> str:
    if len(frame) < long_window + 10:
        raise ValueError("Not enough historical data to train the ML bias model.")

    features = build_ml_features(frame).dropna(subset=["ma_short", "ma_long"])
    if len(features) < 6:
        raise ValueError("Not enough cleaned data to build ML features.")

    if xgb is not None:
        try:
            feature_columns = [c for c in FEATURE_COLUMNS if c in features.columns]
            X = features[feature_columns].astype(float).iloc[:-1]
            y = (features["close"].shift(-1) > features["close"]).astype(int).iloc[:-1]

            if len(X) < 5 or len(y) < 5:
                raise ValueError("Not enough training rows for the XGBoost model.")

            n = len(X)
            sample_weights = np.linspace(0.3, 1.0, n)

            model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
            )
            model.fit(X, y, sample_weight=sample_weights)

            latest_features = features.iloc[[-1]][feature_columns].astype(float)
            probabilities = model.predict_proba(latest_features)[0]
            score = float(probabilities[1])
            score += _preference_adjustment()
            score += order_book_imbalance * 0.2
            score -= spread * 0.5
            return "BUY" if score >= 0.5 else "SELL"
        except Exception:
            return compute_fallback_ml_bias(
                frame, short_window, long_window, order_book_imbalance, spread
            )

    return compute_fallback_ml_bias(
        frame, short_window, long_window, order_book_imbalance, spread
    )

# —————————————————————————
# Convenience helpers used by bot.py
# —————————————————————————


def compute_trend_strength(frame: pd.DataFrame) -> float:
    """Latest ADX value. >25 = trending market, >40 = strong trend."""
    adx_series = compute_adx(frame, period=14)
    return float(adx_series.iloc[-1]) if len(adx_series) > 0 else 0.0


def compute_rsi_signal(frame: pd.DataFrame) -> tuple[float, str]:
    """Returns (rsi_value, 'BUY'|'SELL'|'NEUTRAL') using classic RSI thresholds."""
    rsi_series = compute_rsi(frame["close"], period=14)
    rsi_value = float(rsi_series.iloc[-1])
    if rsi_value < 30:
        return rsi_value, "BUY"
    if rsi_value > 70:
        return rsi_value, "SELL"
    return rsi_value, "NEUTRAL"


def compute_latest_atr(frame: pd.DataFrame, period: int = 14) -> float:
    """Returns the ATR for the most recent bar."""
    atr_series = compute_atr(frame, period=period)
    return float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0
