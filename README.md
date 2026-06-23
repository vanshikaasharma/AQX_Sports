# NBA MVP Predictor

Predicts NBA MVP vote share from real player and team stats, shown in an interactive dashboard.

## Problem

MVP voting is subjective. This project uses historical stats and past voting results to estimate how much MVP support a player would get in a given season.

## Features

- **MVP Race** — top 3 predicted candidates with stats and probability
- **Compare Players** — side-by-side radar charts and stat table
- **Feature Importance** — which stats the model weights most
- **Insights** — season leaders in scoring, efficiency, assists, and win shares

## Data

Stats are pulled from [Basketball Reference](https://www.basketball-reference.com) (2001–2024):

- Per-game and advanced player stats
- Team win percentage
- Historical MVP vote share (training target)

## Model

We compared 4 models (linear regression, ridge, random forest, gradient boosting) using season-by-season cross-validation — train on past years, test on the next.

**Random forest** performed best and was used in the dashboard.

| Metric | Value |
|--------|-------|
| Holdout R² | ~0.80 |
| Holdout MAE | ~0.0035 |

Holdout test uses the most recent 3 seasons (2022–2024).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

First time only — download data, train model, and fetch player photos:

```bash
python data_loader.py
python train_model.py
python build_photos.py
```

Start the dashboard:

```bash
streamlit run app.py
```

## Project structure

```
data_loader.py   # scrape and cache nba data
train_model.py   # train and compare models
build_photos.py  # player headshot urls
app.py           # streamlit dashboard
```

## Impact

- Helps fans and analysts track the MVP race with data, not just narratives
- Shows which stats historically matter most for MVP voting
- Reusable pipeline for other awards or seasons
