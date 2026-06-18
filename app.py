# streamlit dashboard for nba mvp predictions

import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = "data/mvp_dataset.csv"
MODEL_PATH = "models/mvp_model.joblib"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["features"]


# run the model on one season and add predicted vote share
def add_predictions(df, model, features):
    out = df.copy()
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=features)
    out["PredShare"] = model.predict(out[features])
    out["PredShare"] = out["PredShare"].clip(lower=0)
    return out


# bar chart + table of top mvp candidates
def show_mvp_race(season_df, season):
    st.subheader("MVP race - {}".format(season))
    top = season_df.sort_values("PredShare", ascending=False).head(10)

    fig = px.bar(
        top.sort_values("PredShare"),
        x="PredShare",
        y="Player",
        orientation="h",
        labels={"PredShare": "predicted vote share", "Player": ""},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    show_cols = ["Player", "Team", "PTS", "TRB", "AST", "PER", "WS", "WinPct",
                 "Share", "PredShare"]
    show_cols = [c for c in show_cols if c in top.columns]
    st.dataframe(top[show_cols].round(3), use_container_width=True, hide_index=True)


# side by side stat comparison for selected players
def show_comparison(season_df):
    st.subheader("compare players")
    players = sorted(season_df["Player"].unique())
    picked = st.multiselect("pick players", players, default=players[:2])

    if len(picked) < 2:
        st.info("pick at least 2 players")
        return

    compare = season_df[season_df["Player"].isin(picked)]
    stats = ["PTS", "TRB", "AST", "PER", "WS", "VORP", "WinPct", "PredShare"]
    stats = [c for c in stats if c in compare.columns]

    melted = compare.melt(
        id_vars="Player", value_vars=stats,
        var_name="stat", value_name="value",
    )
    fig = px.bar(
        melted, x="stat", y="value", color="Player", barmode="group",
        labels={"value": "", "stat": ""},
    )
    st.plotly_chart(fig, use_container_width=True)


# which stats the model weights most
def show_importance(model, features):
    st.subheader("what matters most")
    imp = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(12)

    fig = px.bar(
        imp.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "", "feature": ""},
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="NBA MVP Predictor", layout="wide")
    st.title("NBA MVP Predictor")
    st.caption("predicts mvp vote share from real nba season stats")

    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        st.error("run data_loader.py and train_model.py first")
        return

    df = load_data()
    model, features = load_model()

    seasons = sorted(df["Season"].unique(), reverse=True)
    season = st.sidebar.selectbox("season", seasons)

    season_df = add_predictions(df[df["Season"] == season], model, features)

    tab1, tab2, tab3 = st.tabs(["mvp race", "compare", "importance"])
    with tab1:
        show_mvp_race(season_df, season)
    with tab2:
        show_comparison(season_df)
    with tab3:
        show_importance(model, features)


if __name__ == "__main__":
    main()
