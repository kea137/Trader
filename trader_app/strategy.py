from __future__ import annotations

import pandas as pd

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


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
    features["ma_ratio"] = (
        (features["ma_short"] - features["ma_long"]) /
        features["ma_long"].replace({0: 1.0})
    )
    features["momentum"] = features["close"].diff().fillna(0.0)
    features["volatility"] = features["close"].rolling(5).std().fillna(0.0)
    features["price_change"] = features["close"].pct_change().fillna(0.0)
    return features


def compute_ml_bias(frame: pd.DataFrame, short_window: int, long_window: int) -> str:
    if xgb is None:
        raise ImportError("XGBoost must be installed to use the ML bias model.")

    if len(frame) < long_window + 10:
        raise ValueError("Not enough historical data to train the XGBoost model.")

    features = build_ml_features(frame).dropna(subset=["ma_short", "ma_long"])
    if len(features) < long_window + 5:
        raise ValueError("Not enough cleaned data to build ML features.")

    training_data = features.iloc[long_window:-1]
    if len(training_data) < 5:
        raise ValueError("Not enough training rows for the XGBoost model.")

    feature_columns = ["ma_short", "ma_long", "ma_ratio", "momentum", "volatility", "price_change"]
    X = training_data[feature_columns].astype(float)
    y = (training_data["close"].shift(-1) > training_data["close"]).astype(int).iloc[:-1]
    X = X.iloc[:-1]

    if len(X) < 5:
        raise ValueError("Not enough training rows after label alignment.")

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        n_estimators=25,
        max_depth=3,
    )
    model.fit(X, y)

    latest_features = features.iloc[[-1]][feature_columns].astype(float)
    prediction = model.predict(latest_features)[0]
    return "BUY" if int(prediction) == 1 else "SELL"
