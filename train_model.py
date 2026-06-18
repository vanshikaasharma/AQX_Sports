# trains a model to predict mvp vote share from player stats

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/mvp_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mvp_model.joblib")

FEATURES = [
    "Age", "G", "GS", "MP", "PTS", "TRB", "AST", "STL", "BLK", "TOV",
    "FG%", "3P%", "FT%", "eFG%", "PER", "TS%", "USG%",
    "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP", "WinPct",
]


# turn stat columns into numbers the model can use
def prep_features(df):
    out = df.copy()
    for col in FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=FEATURES + ["Share"])
    return out


# train on older seasons, test on the most recent ones
def train_model(df, test_years=3):
    df = prep_features(df)
    max_year = df["Season"].max()
    train = df[df["Season"] <= max_year - test_years]
    test = df[df["Season"] > max_year - test_years]

    x_train = train[FEATURES]
    y_train = train["Share"]
    x_test = test[FEATURES]
    y_test = test["Share"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "train_rows": len(train),
        "test_rows": len(test),
    }
    return model, metrics


# save model and print how it did
def main():
    if not os.path.exists(DATA_PATH):
        print("run data_loader.py first")
        return

    df = pd.read_csv(DATA_PATH)
    model, metrics = train_model(df)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)

    print("saved model to", MODEL_PATH)
    print("mae:", round(metrics["mae"], 4))
    print("r2:", round(metrics["r2"], 4))
    print("trained on", metrics["train_rows"], "rows")
    print("tested on", metrics["test_rows"], "rows")


if __name__ == "__main__":
    main()
