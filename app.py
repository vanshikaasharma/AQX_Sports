# streamlit dashboard for nba mvp predictions

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

DATA_PATH = "data/mvp_dataset.csv"
PHOTO_PATH = "data/player_photos.json"
MODEL_PATH = "models/mvp_model.joblib"
COMPARISON_PATH = "models/model_comparison.csv"
CV_PATH = "models/cv_results.csv"

TEAL = "#4C9CB0"
CREAM = "#FFEBAF"
GOLD = "#E8C56A"
CHARCOAL = "#2D3436"

TEAM_LOGOS = {
    "ATL": "1610612737", "BOS": "1610612738", "BKN": "1610612751", "NJN": "1610612751",
    "CHA": "1610612766", "CHH": "1610612766", "CHI": "1610612741", "CLE": "1610612739",
    "DAL": "1610612742", "DEN": "1610612743", "DET": "1610612765", "GSW": "1610612744",
    "HOU": "1610612745", "IND": "1610612754", "LAC": "1610612746", "LAL": "1610612747",
    "MEM": "1610612763", "MIA": "1610612748", "MIL": "1610612749", "MIN": "1610612750",
    "NOP": "1610612740", "NOH": "1610612740", "NYK": "1610612752", "OKC": "1610612760",
    "ORL": "1610612753", "PHI": "1610612755", "PHO": "1610612756", "PHX": "1610612756",
    "POR": "1610612757", "SAC": "1610612758", "SAS": "1610612759", "TOR": "1610612761",
    "UTA": "1610612762", "WAS": "1610612764",
}

FEATURE_LABELS = {
    "PER": "PER", "WS": "Win Shares", "PTS": "PPG", "BPM": "BPM",
    "TS%": "TS%", "AST": "Assists", "TRB": "Rebounds",
}

PLOT_CFG = {"displayModeBar": False, "staticPlot": False}

CARD_CSS = """
body { margin: 0; padding: 0; font-family: Inter, sans-serif; background: transparent; }
.mvp-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    width: 100%;
}
.player-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 0.9rem;
    border: 1.5px solid rgba(76, 156, 176, 0.35);
    box-shadow: 0 6px 22px rgba(76, 156, 176, 0.18);
    display: grid;
    grid-template-columns: 70px 1fr 68px;
    align-items: center;
    gap: 0.6rem;
    box-sizing: border-box;
    min-height: 148px;
}
.player-card.leader {
    border: 2px solid #4C9CB0;
    box-shadow: 0 8px 28px rgba(76, 156, 176, 0.32);
}
.photo-box { position: relative; width: 64px; height: 64px; }
.player-photo {
    width: 64px; height: 64px; border-radius: 14px;
    object-fit: cover; background: #e8f4f7; border: 2px solid #b8dde6; display: block;
}
.avatar {
    width: 64px; height: 64px; border-radius: 14px;
    background: linear-gradient(135deg, #4C9CB0, #6bb8c9);
    color: white; font-weight: 700; font-size: 1rem;
    display: none; align-items: center; justify-content: center;
}
.team-badge {
    position: absolute; top: -5px; left: -5px;
    width: 20px; height: 20px; z-index: 2;
}
.card-center { min-width: 0; display: flex; flex-direction: column; gap: 0.45rem; }
.full-name {
    font-size: 0.88rem; font-weight: 700; color: #2D3436;
    line-height: 1.25; word-wrap: break-word;
}
.stats-row { display: flex; justify-content: space-between; gap: 0.25rem; }
.stat-box { text-align: center; flex: 1; }
.stat-num { font-size: 0.82rem; font-weight: 700; color: #2D3436; line-height: 1.1; }
.stat-lbl {
    font-size: 0.55rem; color: #8a9399; font-weight: 600;
    margin-top: 0.1rem; text-transform: uppercase;
}
.card-right { display: flex; align-items: center; justify-content: center; }
.ring-wrap { position: relative; width: 64px; height: 64px; }
.ring-pct {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-weight: 700; font-size: 0.82rem; color: #4C9CB0;
}
"""

INSIGHT_CSS = """
body { margin: 0; font-family: Inter, sans-serif; background: transparent; }
.insight-card {
    background: rgba(255,255,255,0.7); backdrop-filter: blur(8px);
    border-radius: 18px; padding: 0.75rem 0.5rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.85);
    box-shadow: 0 4px 16px rgba(76,156,176,0.1);
}
.insight-photo {
    width: 44px; height: 44px; border-radius: 12px; object-fit: cover;
    border: 2px solid #4C9CB0; margin-bottom: 0.3rem;
}
.insight-fallback {
    width: 44px; height: 44px; border-radius: 12px; margin: 0 auto 0.3rem;
    background: #4C9CB0; color: white; font-weight: 700; font-size: 0.8rem;
    display: none; align-items: center; justify-content: center;
}
.insight-label { font-size: 0.68rem; color: #5c6670; }
.insight-value { font-size: 0.78rem; font-weight: 700; color: #2D3436; margin-top: 0.15rem; }
"""

COMPARE_CSS = """
body { margin: 0; font-family: Inter, sans-serif; background: transparent; }
.compare-row {
    display: flex; align-items: center; justify-content: center;
    gap: 1.5rem; padding: 0.5rem 0 0.25rem;
}
.compare-col { text-align: center; flex: 1; }
.compare-photo-wrap {
    display: flex; align-items: center; justify-content: center; gap: 0.4rem;
}
.compare-photo {
    width: 64px; height: 64px; border-radius: 14px; object-fit: cover;
    border: 2px solid #4C9CB0;
}
.compare-fallback {
    width: 64px; height: 64px; border-radius: 14px;
    background: #4C9CB0; color: white; font-weight: 700;
    display: none; align-items: center; justify-content: center;
}
.team-logo-sm { width: 28px; height: 28px; }
.compare-name {
    font-weight: 700; color: #2D3436; font-size: 0.88rem; margin-top: 0.35rem;
}
.vs-circle {
    width: 42px; height: 42px; border-radius: 50%; background: #4C9CB0;
    color: white; font-weight: 700; font-size: 0.8rem;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(76,156,176,0.35); flex-shrink: 0;
}
"""


# custom css to match the design mockup
def load_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #2D3436;
        }

        .stApp {
            background: #FFEBAF;
        }

        section[data-testid="stSidebar"] {
            display: none !important;
        }

        [data-testid="collapsedControl"] {
            display: none !important;
        }

        .block-container {
            padding-top: 1rem;
            padding-left: 3rem;
            padding-right: 3rem;
            max-width: 100% !important;
        }

        .stApp .block-container {
            max-width: 100% !important;
        }

        [data-testid="stAppViewContainer"] {
            max-width: 100%;
        }

        iframe {
            border: none !important;
            width: 100% !important;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 0.75rem;
        }

        div[data-testid="stSelectbox"] > div > div {
            background: white;
            border-radius: 14px;
            border: 1px solid rgba(76,156,176,0.2);
        }

        .season-wrap label {
            font-weight: 600 !important;
            color: #4C9CB0 !important;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .dash-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1.5rem;
        }

        .dash-title {
            font-size: 2rem;
            font-weight: 700;
            color: #2D3436;
            margin: 0;
        }

        .dash-sub {
            color: #5c6670;
            font-size: 1rem;
            margin-top: 0.35rem;
        }

        .season-pill {
            display: none;
        }

        .section-label {
            font-size: 1.05rem;
            font-weight: 700;
            color: #2D3436;
            margin: 0 0 0.65rem 0;
        }

        .main [data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            border-radius: 20px !important;
            border: 1px solid rgba(76,156,176,0.2) !important;
            padding: 1rem 1.15rem !important;
            box-shadow: 0 6px 24px rgba(76,156,176,0.12) !important;
        }

        .sidebar-brand {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            padding-top: 0.5rem;
        }

        div[data-testid="stPlotlyChart"] {
            background: transparent;
            box-shadow: none;
            border: none;
            padding: 0;
        }

        div[data-testid="stDataFrame"] {
            background: #ffffff;
            border-radius: 12px;
            padding: 0.25rem;
        }

        div[data-testid="stDataFrame"] div[data-testid="stTable"] {
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["features"], saved.get("model_name", "random forest")


@st.cache_data
def load_comparisons():
    holdout = pd.read_csv(COMPARISON_PATH)
    cv = pd.read_csv(CV_PATH)
    return holdout, cv


@st.cache_data
def load_player_photos():
    if os.path.exists(PHOTO_PATH):
        with open(PHOTO_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


# run the model on one season and add predicted vote share
def add_predictions(df, model, features):
    out = df.copy()
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=features)
    out["PredShare"] = model.predict(out[features]).clip(min=0)
    return out


# get initials for the avatar circle
def player_initials(name):
    parts = name.replace(".", "").split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper()


# team logo url from nba cdn
def team_logo_url(team):
    team_id = TEAM_LOGOS.get(str(team), "")
    if not team_id:
        return ""
    return "https://cdn.nba.com/logos/nba/{}/global/L/logo.svg".format(team_id)


# render html in an iframe so streamlit doesnt escape it
def render_html(body, css, height):
    doc = (
        "<!DOCTYPE html><html><head><style>{css}</style></head>"
        "<body>{body}</body></html>"
    ).format(css=css, body=body)
    components.html(doc, height=height)


# svg ring showing mvp probability
def progress_ring(pct):
    offset = 190 * (1 - pct / 100)
    return """
    <div class="ring-wrap">
        <svg width="64" height="64" viewBox="0 0 64 64">
            <circle cx="32" cy="32" r="27" fill="none" stroke="#e8f4f7" stroke-width="6"/>
            <circle cx="32" cy="32" r="27" fill="none" stroke="#4C9CB0" stroke-width="6"
                stroke-linecap="round" stroke-dasharray="190"
                stroke-dashoffset="{offset}"
                transform="rotate(-90 32 32)"/>
        </svg>
        <div class="ring-pct">{pct:.0f}%</div>
    </div>
    """.format(offset=offset, pct=pct)


# format ts% whether stored as 0.65 or 65
def fmt_ts(val):
    val = float(val)
    if val <= 1:
        return val * 100
    return val


# build portrait html with fallback to initials
def portrait_html(name, photo_map, photo_class, fallback_class, fb_id):
    url = photo_map.get(name, "")
    init = player_initials(name)
    if url:
        return (
            '<img class="{pc}" src="{url}" '
            'onerror="this.style.display=\'none\';'
            'document.getElementById(\'{fid}\').style.display=\'flex\'"/>'
            '<div id="{fid}" class="{fc}">{init}</div>'
        ).format(pc=photo_class, url=url, fid=fb_id, fc=fallback_class, init=init)
    return '<div class="{fc}" style="display:flex">{init}</div>'.format(
        fc=fallback_class, init=init,
    )


# html for one top-3 player card — pict | name+stats | donut
def player_card(row, prob_pct, photo_map, is_leader=False):
    logo = team_logo_url(row["Team"])
    badge = (
        '<img class="team-badge" src="{}" alt="team"/>'.format(logo)
        if logo else ""
    )
    pid = "p-" + str(abs(hash(row["Player"])))[:8]
    portrait = portrait_html(
        row["Player"], photo_map, "player-photo", "avatar", pid,
    )
    cls = "player-card leader" if is_leader else "player-card"

    return """
    <div class="{cls}">
        <div class="photo-box">
            {badge}
            {portrait}
        </div>
        <div class="card-center">
            <div class="full-name">{name}</div>
            <div class="stats-row">
                <div class="stat-box">
                    <div class="stat-num">{pts:.1f}</div>
                    <div class="stat-lbl">PPG</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num">{reb:.1f}</div>
                    <div class="stat-lbl">RPG</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num">{ast:.1f}</div>
                    <div class="stat-lbl">APG</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num">{ts:.0f}%</div>
                    <div class="stat-lbl">TS%</div>
                </div>
            </div>
        </div>
        <div class="card-right">
            {ring}
        </div>
    </div>
    """.format(
        cls=cls,
        badge=badge,
        portrait=portrait,
        name=row["Player"],
        ring=progress_ring(prob_pct),
        pts=row["PTS"],
        reb=row["TRB"],
        ast=row["AST"],
        ts=fmt_ts(row["TS%"]),
    )


# top 3 mvp cards in one full-width row (avoids cramped streamlit columns)
def show_mvp_race(season_df, season, photo_map):
    st.markdown('<div class="section-label">MVP Race</div>', unsafe_allow_html=True)
    top3 = season_df.nlargest(3, "PredShare").copy()
    total = top3["PredShare"].sum()
    top3["ProbPct"] = (top3["PredShare"] / total * 100) if total > 0 else 0

    cards = []
    for i, (_, row) in enumerate(top3.iterrows()):
        cards.append(player_card(row, row["ProbPct"], photo_map, is_leader=(i == 0)))

    render_html('<div class="mvp-row">' + "".join(cards) + "</div>", CARD_CSS, height=168)


# normalize a stat to 0-100 for radar chart
def norm_stat(value, col, season_df):
    vals = pd.to_numeric(season_df[col], errors="coerce")
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return 50
    return (float(value) - lo) / (hi - lo) * 100


# radar chart for one player
def make_radar(player_row, season_df, color):
    labels = ["Points", "Rebounds", "Assists", "Efficiency", "Usage"]
    cols = ["PTS", "TRB", "AST", "TS%", "USG%"]
    values = [norm_stat(player_row[c], c, season_df) for c in cols]
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(76,156,176,0.25)",
        line=dict(color=color, width=2),
        name=player_row["Player"],
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 100]),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# compare row with square photos, team logos and vs badge
def compare_players_row(p1, p2, row1, row2, photo_map):
    logo1 = team_logo_url(row1["Team"])
    logo2 = team_logo_url(row2["Team"])
    l1 = '<img class="team-logo-sm" src="{}"/>'.format(logo1) if logo1 else ""
    l2 = '<img class="team-logo-sm" src="{}"/>'.format(logo2) if logo2 else ""
    ph1 = portrait_html(p1, photo_map, "compare-photo", "compare-fallback", "h1")
    ph2 = portrait_html(p2, photo_map, "compare-photo", "compare-fallback", "h2")
    return """
    <div class="compare-row">
        <div class="compare-col">
            <div class="compare-photo-wrap">{ph1}{l1}</div>
            <div class="compare-name">{n1}</div>
        </div>
        <div class="vs-circle">VS</div>
        <div class="compare-col">
            <div class="compare-photo-wrap">{ph2}{l2}</div>
            <div class="compare-name">{n2}</div>
        </div>
    </div>
    """.format(ph1=ph1, ph2=ph2, l1=l1, l2=l2, n1=p1, n2=p2)


# side by side player comparison inside one white card
def show_comparison(season_df, photo_map):
    ranked = season_df.sort_values("PredShare", ascending=False)["Player"].tolist()

    st.markdown(
        '<div class="section-label">Compare Players</div>',
        unsafe_allow_html=True,
    )

    pick1, pick2 = st.columns(2)
    with pick1:
        p1 = st.selectbox("player 1", ranked, index=0, key="p1", label_visibility="collapsed")
    with pick2:
        idx2 = 1 if len(ranked) > 1 else 0
        p2 = st.selectbox("player 2", ranked, index=idx2, key="p2", label_visibility="collapsed")

    row1 = season_df[season_df["Player"] == p1].iloc[0]
    row2 = season_df[season_df["Player"] == p2].iloc[0]

    render_html(
        compare_players_row(p1, p2, row1, row2, photo_map),
        COMPARE_CSS, height=130,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(make_radar(row1, season_df, TEAL), use_container_width=True, config=PLOT_CFG)
    with col_r:
        st.plotly_chart(make_radar(row2, season_df, GOLD), use_container_width=True, config=PLOT_CFG)

    compare = season_df[season_df["Player"].isin([p1, p2])]
    table = compare[["Player", "PTS", "TRB", "AST", "TS%", "USG%", "PER", "WS"]].copy()
    table.columns = ["Player", "PPG", "RPG", "APG", "TS%", "Usage", "PER", "Win Shares"]
    for c in ["PPG", "RPG", "APG", "TS%", "Usage", "PER", "Win Shares"]:
        table[c] = pd.to_numeric(table[c], errors="coerce").round(2)
    st.dataframe(table, use_container_width=True, hide_index=True)


# get feature importance values from the model
def get_importance(model, features):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        return np.abs(model.coef_)
    return None


# horizontal bar chart with teal to gold gradient
def show_importance(model, features, model_name):
    st.markdown(
        '<div class="section-label">Feature Importance</div>',
        unsafe_allow_html=True,
    )
    st.caption("model: {}".format(model_name))

    values = get_importance(model, features)
    if values is None:
        st.info("no feature importance for this model")
        return

    imp = pd.DataFrame({"feature": features, "value": values})
    imp["label"] = imp["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    imp = imp.sort_values("value", ascending=True).tail(7)

    total = imp["value"].sum()
    imp["pct"] = (imp["value"] / total * 100).round(0)

    colors = []
    for i in range(len(imp)):
        t = i / max(len(imp) - 1, 1)
        r = int(76 + t * (255 - 76))
        g = int(156 + t * (235 - 156))
        b = int(176 + t * (175 - 176))
        colors.append("rgb({},{},{})".format(r, g, b))

    fig = go.Figure(go.Bar(
        x=imp["pct"],
        y=imp["label"],
        orientation="h",
        text=[str(int(p)) + "%" for p in imp["pct"]],
        textposition="outside",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=50, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=False, title=""),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)


# four insight cards in a 2x2 grid (right column, below importance)
def show_insights(season_df, photo_map):
    st.markdown('<div class="section-label">Insights</div>', unsafe_allow_html=True)

    cards = [
        ("Highest Scorer", season_df.nlargest(1, "PTS").iloc[0]["Player"]),
        ("Best Efficiency", season_df.nlargest(1, "TS%").iloc[0]["Player"]),
        ("Most Assists", season_df.nlargest(1, "AST").iloc[0]["Player"]),
        ("Highest Win Shares", season_df.nlargest(1, "WS").iloc[0]["Player"]),
    ]

    row1 = st.columns(2)
    row2 = st.columns(2)
    slots = [row1[0], row1[1], row2[0], row2[1]]

    for col, (label, player) in zip(slots, cards):
        iid = "i-" + str(abs(hash(label + player)))[:8]
        photo = portrait_html(
            player, photo_map, "insight-photo", "insight-fallback", iid,
        )
        body = (
            '<div class="insight-card">{photo}'
            '<div class="insight-label">{label}</div>'
            '<div class="insight-value">{player}</div></div>'
        ).format(photo=photo, label=label, player=player)
        with col:
            render_html(body, INSIGHT_CSS, height=115)


# model comparison page
def show_model_comparison(holdout, cv, model_name):
    st.markdown('<div class="dash-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dash-sub">picked <b>{}</b> using season cross validation</div>'
        .format(model_name),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Holdout (last 3 seasons)**")
        st.dataframe(holdout.round(4), use_container_width=True, hide_index=True)
        fig = go.Figure(go.Bar(
            x=holdout["model"], y=holdout["mae"],
            marker_color=TEAL, text=holdout["mae"].round(4),
            textposition="outside",
        ))
        fig.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="MAE", xaxis_tickangle=-20,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    with col2:
        st.markdown("**Season CV**")
        st.dataframe(cv.round(4), use_container_width=True, hide_index=True)
        fig = go.Figure(go.Bar(
            x=cv["model"], y=cv["mae"],
            marker_color=TEAL, text=cv["mae"].round(4),
            textposition="outside",
        ))
        fig.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="MAE", xaxis_tickangle=-20,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)


def main():
    st.set_page_config(
        page_title="NBA MVP Predictor",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    load_css()

    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        st.error("run data_loader.py and train_model.py first")
        return

    df = load_data()
    model, features, model_name = load_model()
    seasons = sorted(df["Season"].unique(), reverse=True)

    title_col, season_col = st.columns([4, 1])
    with title_col:
        st.markdown(
            """
            <div class="dash-title">🏀 NBA MVP Predictor</div>
            <div class="dash-sub">Predicting the MVP race through advanced analytics.</div>
            """,
            unsafe_allow_html=True,
        )
    with season_col:
        st.markdown('<div class="season-wrap">', unsafe_allow_html=True)
        season = st.selectbox("Season", seasons, index=0)
        st.markdown("</div>", unsafe_allow_html=True)

    season_df = add_predictions(df[df["Season"] == season], model, features)
    photo_map = load_player_photos()

    show_mvp_race(season_df, season, photo_map)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.55, 1], gap="medium")

    with left_col:
        with st.container(border=True):
            show_comparison(season_df, photo_map)

    with right_col:
        with st.container(border=True):
            show_importance(model, features, model_name)
        with st.container(border=True):
            show_insights(season_df, photo_map)


if __name__ == "__main__":
    main()
