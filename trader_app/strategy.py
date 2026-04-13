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
]


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


def build_ml_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    if "volume" not in features:
        features["volume"] = 0.0

    features["ma_ratio"] = (
        (features["ma_short"] - features["ma_long"]) /
        features["ma_long"].replace({0: 1.0})
    )
    rolling_low = features["close"].rolling(20, min_periods=1).min()
    rolling_high = features["close"].rolling(20, min_periods=1).max()
    features["recent_low"] = rolling_low
    features["recent_high"] = rolling_high
    features["position_in_range"] = (
        (features["close"] - rolling_low) /
        (rolling_high - rolling_low).replace({0: 1.0})
    )
    features["momentum"] = features["close"].diff().fillna(0.0)
    features["volatility"] = features["close"].rolling(10).std().fillna(0.0)
    features["price_change"] = features["close"].pct_change().fillna(0.0)
    features["volume_change"] = features["volume"].diff().fillna(0.0)
    rolling_volume = features["volume"].rolling(20, min_periods=1).mean().replace({0: 1.0})
    features["volume_ratio"] = features["volume"] / rolling_volume
    return features


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

    feature_columns = FEATURE_COLUMNS
    X = features[feature_columns].astype(float).iloc[:-1]
    y = (features["close"].shift(-1) > features["close"]).astype(int).iloc[:-1]

    if len(X) < 5 or len(y) < 5:
        raise ValueError("Not enough aligned training rows for fallback ML bias.")

    weights = np.linalg.pinv(X.to_numpy()) @ y.to_numpy()
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
        ml_bias_preference[predicted_signal] = min(10.0, ml_bias_preference[predicted_signal] + adjustment)
    else:
        ml_bias_preference[predicted_signal] = max(-10.0, ml_bias_preference[predicted_signal] - abs(adjustment))


def compute_price_position(frame: pd.DataFrame, lookback: int = 20) -> tuple[float, float, float]:
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
            feature_columns = FEATURE_COLUMNS
            X = features[feature_columns].astype(float).iloc[:-1]
            y = (features["close"].shift(-1) > features["close"]).astype(int).iloc[:-1]

            if len(X) < 5 or len(y) < 5:
                raise ValueError("Not enough training rows for the XGBoost model.")

            model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            model.fit(X, y)

            latest_features = features.iloc[[-1]][feature_columns].astype(float)
            probabilities = model.predict_proba(latest_features)[0]
            score = float(probabilities[1])
            score += _preference_adjustment()
            score += order_book_imbalance * 0.2
            score -= spread * 0.5
            return "BUY" if score >= 0.5 else "SELL"
        except Exception:
            return compute_fallback_ml_bias(
                frame,
                short_window,
                long_window,
                order_book_imbalance,
                spread,
            )

    return compute_fallback_ml_bias(
        frame,
        short_window,
        long_window,
        order_book_imbalance,
        spread,
    )
