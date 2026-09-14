# Slingshot Studios Player Lifecycle & Monetization Analytics

A portfolio project using Python, SQL, and Streamlit to analyze a simplified mobile-game live-service dataset.

## Product questions

1. How well do newly acquired players return after onboarding?
2. Which early behaviors separate low-habit, engaged, and monetizing players?
3. Can first-14-day behavior identify players at elevated 30-day churn risk?
4. Where should a game/product team focus retention, monetization, and operational investigation?

## Dataset

The supplied SQLite database contains:

- 112,792 player accounts
- 1.7M daily player-session records
- 9,909 purchase events
- 2016 calendar-year activity

Source tables:

- `account`
- `account_date_session`
- `iap_purchase`

## Key analytical outputs

| Metric | Result |
|---|---:|
| D1 retention | 39.2% |
| D7 retention | 19.5% |
| D30 retention | 10.1% |
| Observed IAP revenue | $42,494.66 |
| Observed payers | 1,548 |
| Churn model | Logistic Regression |
| Test ROC AUC | 0.924 |
| Test PR AUC | 0.968 |

These figures are descriptive of the supplied sample and should not be presented as EA business metrics.

## Streamlit dashboard

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard has three views:

### 1. Executive Player Health
Players, D1/D7/D30 retention, revenue, payer rate, ARPPU, DAU trends, revenue trends, and cohort retention.

### 2. Player Segmentation & Monetization
Behavioral segment distribution, early engagement profiles, ARPU, payer conversion, and country × platform comparisons.

### 3. Churn Risk & Monitoring
At-risk players, churn-risk distribution, churn probability vs early engagement, daily revenue monitoring, and cohort retention heatmap.

The sidebar supports filtering by platform, country, segment, and churn-risk band.

## Project structure

```text
slingshot-player-lifecycle-analytics/
├── README.md
├── requirements.txt
├── app.py
├── notebooks/
│   └── Slingshot_Studios_Player_Lifecycle_Analytics.ipynb
├── data/
│   ├── README.md
│   ├── dashboard/
│   │   ├── player_360.csv
│   │   ├── daily_kpis.csv
│   │   ├── cohort_retention.csv
│   │   ├── segment_summary.csv
│   │   ├── segment_monetization.csv
│   │   ├── country_platform_kpis.csv
│   │   ├── engagement_behavior.csv
│   │   ├── monetization_summary.csv
│   │   └── activity_spending_correlation.csv
│   └── (source SQLite data remains under csv/data/)
└── csv/
    └── data/
        └── sample(1).sqlite
```

## Modeling note

The churn model predicts whether a player returns during days 14–43 using early player behavior. It is a prioritization model, not a causal model. The dataset does not include treatment/control assignment, so the project does not claim that a feature or intervention caused retention changes.

## Resume bullet

**Player Lifecycle & Monetization Analytics — Python, SQL, Streamlit:** analyzed 112K+ mobile-game players to measure D1/D7/D30 retention, segment early player behavior, model 30-day churn risk (ROC AUC 0.924), detect revenue anomalies, and build an interactive Streamlit dashboard translating player behavior into product actions.
