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

A fourth view, **Chat with your player data**, exposes the text-to-SQL agent described below.

## Chat with your player data

Ask questions about the raw SQLite data in plain English. A LangChain agent backed by Google Gemini (`ChatGoogleGenerativeAI`) writes a SQLite query grounded in the live database schema, the query is validated and run read-only, and the response contains the **generated SQL, the result rows and a short natural-language answer**. The SQL is always returned so every answer can be traced to the query that produced it. If a question can't be answered from the available tables, the agent says so instead of guessing.

Example questions:

- Which 5 countries generated the most IAP revenue?
- How many accounts were created on iOS vs Android?
- What is the average session duration in minutes by platform?
- How many distinct players were active each month in 2016?
- What share of payers made more than one purchase?

### Run the API

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=...            # required for /chat
uvicorn api:app --reload
```

```bash
curl localhost:8000/health
# {"status":"ok"}

curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"question": "Which 5 countries generated the most IAP revenue?"}'
# {"sql": "SELECT ... LIMIT 5", "rows": [{"country_code": "US", "revenue_usd": 13148.18}, ...],
#  "answer": "The US leads with $13,148.18, followed by ..."}
```

Errors are clean JSON, `{"error": "<code>", "detail": "<message>"}`: `400` for an invalid question or a rejected/failed query, `503` when the database or LLM is unavailable, `500` for anything unexpected (no internals are returned). Interactive docs are at `/docs`.

Configuration (environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | (none) | Gemini API key (never logged or returned) |
| `CHAT_MODEL` | `gemini-flash-latest` | Gemini model id |
| `PLAYER_DB_PATH` | first `*.sqlite` in `csv/data/` | Path to the SQLite database |
| `CHAT_MAX_ROWS` | `1000` | Maximum rows returned per query |
| `CHAT_QUERY_TIMEOUT_SEC` | `10` | Per-query execution time budget |

### Safety design

The endpoint is **strictly read-only and injection-guarded**, with independent layers so that no single failure allows a write:

1. **Schema-aware prompting.** The prompt contains the real table and column names read from the database. The model is told to emit one SELECT only, to use only listed columns, and to treat the user's question as data, not instructions.
2. **Parser-based validation** (`chat/safety.py`). The SQL is parsed into a syntax tree with `sqlglot`, not matched against strings. It is rejected unless it is **exactly one** `SELECT` (or `UNION`/`WITH ... SELECT`). Any `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `REPLACE`, `ATTACH`/`DETACH`, `PRAGMA`, `VACUUM`, transaction, or `SELECT ... INTO` is rejected anywhere in the tree, including inside CTEs. Semicolon-chained statements are rejected. Dangerous functions (`load_extension`, `readfile`, `writefile`, ...), table-valued functions, schema-qualified names and tables outside `account`, `account_date_session` and `iap_purchase` are rejected too.
3. **Row limit.** The outer query's `LIMIT` is added, or clamped, to at most 1000 rows. That includes `LIMIT -1` and non-literal limits. The SQL that runs is regenerated from the validated tree with comments stripped, so what executes is exactly what was checked.
4. **Read-only connection** (`chat/db.py`). SQLite is opened via a `file:...?mode=ro` URI with `PRAGMA query_only = ON`. A SQLite **authorizer** permits only SELECT, reads of the three player tables and safe functions; the engine itself denies everything else. A progress handler aborts queries that run past the time budget, and results are fetched with a hard row cap.
5. **No leakage.** API errors never include the API key, file paths or stack traces; details are logged server-side only.

The test suite includes adversarial cases (DML/DDL, chaining, `PRAGMA`, `ATTACH`, writes hidden in CTEs, filesystem functions, unbounded limits). It checks the parser and the database layer independently, then confirms the data is unchanged afterwards.

### Tests

```bash
pytest
```

The tests are fully offline. They use an in-memory SQLite fixture that mirrors the real schema and a fake LLM that returns scripted SQL, so no API key or network is needed.

## Project structure

```text
slingshot-player-lifecycle-analytics/
├── README.md
├── requirements.txt
├── app.py
├── api.py                 # FastAPI: /health, /chat
├── chat/                  # text-to-SQL agent package
│   ├── agent.py           # LangChain + Gemini prompt/answer flow
│   ├── safety.py          # parser-based SELECT-only validation + row limit
│   ├── db.py              # read-only, authorizer-guarded SQLite access
│   ├── config.py
│   └── errors.py
├── tests/                 # offline pytest suite (fake LLM, in-memory DB)
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
