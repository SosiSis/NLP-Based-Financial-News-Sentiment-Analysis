"""
Train an LSTM model and save it for serving.

Usage:
    python scripts/train_model.py --data Data/final_sentiment_dataset.csv
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ================= TECHNICAL INDICATORS =================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()

    def _ind(group):
        group["SMA_5"] = group["Close"].rolling(5, min_periods=1).mean()
        group["EMA_12"] = group["Close"].ewm(span=12, adjust=False).mean()
        group["EMA_26"] = group["Close"].ewm(span=26, adjust=False).mean()
        group["MACD"] = group["EMA_12"] - group["EMA_26"]
        group["RSI"] = 100 - (
            100 / (1 + (
                group["Close"].diff().clip(lower=0).rolling(14, min_periods=1).mean() /
                (-group["Close"].diff().clip(upper=0)).rolling(14, min_periods=1).mean().replace(0, np.nan)
            ))
        ).fillna(0)
        group["Price_change"] = group["Close"].pct_change().fillna(0)
        group["Price_change_5d"] = group["Close"].pct_change(5).fillna(0)
        return group

    return df.groupby("Ticker", group_keys=False).apply(_ind).reset_index(drop=True)


# ================= TARGET =================
def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()
    df["next_close"] = df.groupby("Ticker")["Close"].shift(-1)
    df = df.dropna(subset=["next_close"])
    df["target_up"] = (df["next_close"] > df["Close"]).astype(int)
    return df


# ================= SEQUENCE BUILDER =================
def make_sequences(df, features, target, window):
    X, y = [], []

    for _, group in df.groupby("Ticker"):
        group = group.sort_values("Date")
        values = group[features].values
        labels = group[target].values

        for i in range(len(group) - window):
            X.append(values[i:i + window])
            y.append(labels[i + window])

    return np.array(X), np.array(y)


# ================= MAIN =================
def main(args):
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["Date"])

    if "Ticker" not in df.columns:
        df["Ticker"] = "__ALL__"

    print("Adding indicators...")
    df = add_technical_indicators(df)
    df = build_target(df)

    # ---------- SENTIMENT ----------
    print("Loading sentiment models (FinBERT + VADER)...")
    try: 
        from backend.app.services import predict_service 
        predict_service.load_models() 
        
    except Exception as e: 
        print(f"Warning: failed to load predict_service: {e}")
        predict_service = None

    sentiment_cols = [
        "finbert_positive",
        "finbert_negative",
        "finbert_neutral",
        "vader_compound"
    ]

    if not all(c in df.columns for c in sentiment_cols):
        scores = []
        for t in tqdm(df["Headlines_clean"].fillna(""), desc="Sentiment"):
            fin = predict_service._finbert_scores(t)
            v = predict_service._vader_compound(t)
            fin["vader_compound"] = v
            scores.append(fin)
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)

    # ---------- FEATURES ----------
    features = [
        "finbert_positive", "finbert_negative", "finbert_neutral", "vader_compound",
        "SMA_5", "EMA_12", "MACD", "RSI",
        "Price_change", "Price_change_5d", "Volume"
    ]

    df[features] = df[features].fillna(0)

    # ---------- SCALE ----------
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # ---------- SEQUENCES ----------
    window_size = args.window
    X, y = make_sequences(df, features, "target_up", window_size)

    print("Sequence shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # ---------- MODEL ----------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, len(features))),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print(model.summary())

    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("Training LSTM...")
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[es],
        verbose=1
    )

    # ---------- EVALUATION ----------
    probs = model.predict(X_test).ravel()
    preds = (probs >= 0.5).astype(int)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))

    # ---------- SAVE ----------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir / "lstm_model.keras")
    joblib.dump(scaler, out_dir / "scaler.joblib")

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(features, f)

    with open(out_dir / "window_size.txt", "w") as f:
        f.write(str(window_size))

    print(f"✅ Model saved to {out_dir}")


# ================= CLI =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Data/final_sentiment_dataset.csv")
    parser.add_argument("--out-dir", default="ml/serving")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    main(args)
