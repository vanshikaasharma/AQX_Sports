# trains and compares models to predict mvp vote share

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/mvp_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mvp_model.joblib")
COMPARISON_PATH = os.path.join(MODEL_DIR, "model_comparison.csv")
CV_PATH = os.path.join(MODEL_DIR, "cv_results.csv")

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


# models we want to compare
def get_models():
    return {
        "linear regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42,
        ),
        "gradient boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
        ),
    }


# split by year so we train on the past and test on recent seasons
def temporal_split(df, test_years=3):
    max_year = df["Season"].max()
    train = df[df["Season"] <= max_year - test_years]
    test = df[df["Season"] > max_year - test_years]
    return train, test


# score one model on a train/test split
def score_model(model, train, test):
    x_train = train[FEATURES]
    y_train = train["Share"]
    x_test = test[FEATURES]
    y_test = test["Share"]

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }


# compare all models on the recent-season holdout set
def compare_holdout(df, test_years=3):
    train, test = temporal_split(df, test_years)
    rows = []
    for name, model in get_models().items():
        scores = score_model(model, train, test)
        rows.append({
            "model": name,
            "mae": scores["mae"],
            "r2": scores["r2"],
            "eval": "holdout",
        })
    return pd.DataFrame(rows).sort_values("mae")


# cross validate by season - train on past years, test on the next year
def compare_cv(df, min_train_seasons=10):
    seasons = sorted(df["Season"].unique())
    rows = []

    for name, model in get_models().items():
        fold_mae = []
        fold_r2 = []
        for i in range(min_train_seasons, len(seasons)):
            train_seasons = seasons[:i]
            test_season = seasons[i]
            train = df[df["Season"].isin(train_seasons)]
            test = df[df["Season"] == test_season]
            if len(test) == 0:
                continue
            scores = score_model(model, train, test)
            fold_mae.append(scores["mae"])
            fold_r2.append(scores["r2"])

        rows.append({
            "model": name,
            "mae": np.mean(fold_mae),
            "r2": np.mean(fold_r2),
            "folds": len(fold_mae),
            "eval": "season cv",
        })

    return pd.DataFrame(rows).sort_values("mae")


# pick the model with the lowest cv mae and retrain it
def train_best(df, test_years=3):
    df = prep_features(df)
    holdout = compare_holdout(df, test_years)
    cv = compare_cv(df)

    best_name = cv.iloc[0]["model"]
    best_model = get_models()[best_name]
    train, test = temporal_split(df, test_years)
    final_scores = score_model(best_model, train, test)

    return best_model, best_name, holdout, cv, final_scores, len(train), len(test)


# save the winning model and comparison tables
def main():
    if not os.path.exists(DATA_PATH):
        print("run data_loader.py first")
        return

    df = pd.read_csv(DATA_PATH)
    model, name, holdout, cv, scores, train_n, test_n = train_best(df)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": FEATURES,
        "model_name": name,
    }, MODEL_PATH)
    holdout.to_csv(COMPARISON_PATH, index=False)
    cv.to_csv(CV_PATH, index=False)

    print("picked model:", name)
    print("holdout mae:", round(scores["mae"], 4))
    print("holdout r2:", round(scores["r2"], 4))
    print("saved to", MODEL_PATH)
    print()
    print("holdout comparison:")
    print(holdout.round(4).to_string(index=False))
    print()
    print("season cv comparison:")
    print(cv.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
