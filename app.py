from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Slingshot Studios | Player Lifecycle",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "dashboard"


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    files = {
        "players": "player_360.csv",
        "daily": "daily_kpis.csv",
        "cohort": "cohort_retention.csv",
        "segments": "segment_summary.csv",
        "segment_monetization": "segment_monetization.csv",
        "country_platform": "country_platform_kpis.csv",
        "engagement": "engagement_behavior.csv",
        "monetization": "monetization_summary.csv",
        "correlation": "activity_spending_correlation.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, name in files.items():
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing dashboard dataset: {path}")
        out[key] = pd.read_csv(path)

    out["players"]["created_date"] = pd.to_datetime(out["players"]["created_date"])
    out["players"]["signup_month"] = pd.to_datetime(
        out["players"]["signup_month"].astype(str), format="%Y-%m"
    )
    out["daily"]["date"] = pd.to_datetime(out["daily"]["date"])
    out["cohort"]["signup_month"] = pd.to_datetime(
        out["cohort"]["signup_month"].astype(str), format="%Y-%m"
    )
    return out


@st.cache_data
def get_kpi_value(monetization: pd.DataFrame, metric: str) -> float:
    row = monetization.loc[monetization["metric"].eq(metric), "value"]
    return float(row.iloc[0]) if not row.empty else 0.0


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def apply_player_filters(players: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    platforms = sorted(players["created_platform"].dropna().unique())
    countries = sorted(players["country_code"].dropna().unique())
    segments = sorted(players["segment"].dropna().unique())
    risks = ["Low", "Medium", "High", "Very High"]

    platform = st.sidebar.multiselect("Platform", platforms, default=platforms)
    country = st.sidebar.multiselect("Country", countries, default=countries)
    segment = st.sidebar.multiselect("Player segment", segments, default=segments)
    risk = st.sidebar.multiselect("Churn risk", risks, default=risks)

    filtered = players[
        players["created_platform"].isin(platform)
        & players["country_code"].isin(country)
        & players["segment"].isin(segment)
        & players["churn_risk_band"].isin(risk)
    ].copy()
    return filtered


def metric_strip(players: pd.DataFrame, cohort: pd.DataFrame) -> None:
    eligible = cohort.loc[cohort["day_number"].isin([1, 7, 30])]
    retention = eligible.groupby("day_number")["retention_rate"].mean()

    total_players = len(players)
    revenue = players["lifetime_revenue"].sum()
    payers = int(players["lifetime_payer"].sum())
    payer_rate = payers / total_players if total_players else 0
    arppu = revenue / payers if payers else 0

    c = st.columns(7)
    c[0].metric("Players", f"{total_players:,}")
    c[1].metric("D1 Retention", pct(retention.get(1, 0)))
    c[2].metric("D7 Retention", pct(retention.get(7, 0)))
    c[3].metric("D30 Retention", pct(retention.get(30, 0)))
    c[4].metric("Revenue", money(revenue))
    c[5].metric("Payer Rate", pct(payer_rate))
    c[6].metric("ARPPU", money(arppu))


def executive_page(data: dict[str, pd.DataFrame], players: pd.DataFrame) -> None:
    st.header("Executive Player Health")
    st.caption("Early behavior → player health → retention and monetization")

    metric_strip(players, data["cohort"])
    st.divider()

    left, right = st.columns(2)

    with left:
        daily = data["daily"].copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["dau"],
                name="DAU",
                mode="lines",
            )
        )
        fig.update_layout(
            title="Daily active users",
            xaxis_title="Date",
            yaxis_title="Players",
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        daily = data["daily"].copy()
        fig = px.line(
            daily,
            x="date",
            y="revenue_usd",
            title="Daily revenue",
            labels={"date": "Date", "revenue_usd": "Revenue (USD)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    cohort = (
        data["cohort"]
        .query("day_number in [1, 7, 30]")
        .pivot(index="signup_month", columns="day_number", values="retention_rate")
        .reset_index()
    )
    cohort = cohort.rename(columns={1: "D1", 7: "D7", 30: "D30"})
    cohort_long = cohort.melt("signup_month", var_name="retention_day", value_name="retention")
    fig = px.line(
        cohort_long,
        x="signup_month",
        y="retention",
        color="retention_day",
        markers=True,
        title="Retention by signup cohort",
        labels={"signup_month": "Signup month", "retention": "Retention"},
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def segmentation_page(data: dict[str, pd.DataFrame], players: pd.DataFrame) -> None:
    st.header("Player Segmentation & Monetization")
    st.caption("Behavioral cohorts are prioritization groups, not causal treatment groups.")

    seg_counts = (
        players["segment"]
        .value_counts()
        .rename_axis("segment")
        .reset_index(name="players")
    )
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            seg_counts,
            x="players",
            y="segment",
            orientation="h",
            title="Player distribution by segment",
        )
        st.plotly_chart(fig, use_container_width=True)

    seg = data["segment_summary"].copy()
    seg_long = seg.melt(
        id_vars=["segment"],
        value_vars=["median_active_days_14", "median_sessions_14", "median_hours_14"],
        var_name="metric",
        value_name="value",
    )
    fig = px.bar(
        seg_long,
        x="segment",
        y="value",
        color="metric",
        barmode="group",
        title="Early engagement profile by segment",
    )
    with right:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monetization by segment")
    sm = data["segment_monetization"].copy()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            sm,
            x="segment",
            y="arpu",
            title="ARPU by segment",
            labels={"arpu": "ARPU (USD)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            sm,
            x="segment",
            y="payer_conversion",
            title="Payer conversion by segment",
            labels={"payer_conversion": "Payer conversion"},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Country × platform opportunity")
    cp = data["country_platform"].copy()
    cp["label"] = cp["revenue_usd"].map(money)
    pivot = cp.pivot_table(
        index="country_code",
        columns="created_platform",
        values="payer_rate",
        aggfunc="mean",
    )
    st.dataframe(
        pivot.style.format("{:.1%}"),
        use_container_width=True,
        height=420,
    )


def churn_page(data: dict[str, pd.DataFrame], players: pd.DataFrame) -> None:
    st.header("Churn Risk & Operational Monitoring")
    st.caption("Churn probability is a prioritization signal, not proof of causation.")

    at_risk = int(players["churn_risk_band"].isin(["High", "Very High"]).sum())
    risk_pct = at_risk / len(players) if len(players) else 0
    d30 = (
        data["cohort"].loc[data["cohort"]["day_number"].eq(30), "retention_rate"].mean()
    )
    revenue = players["lifetime_revenue"].sum()

    c = st.columns(4)
    c[0].metric("At-Risk Players", f"{at_risk:,}")
    c[1].metric("High + Very High Risk", pct(risk_pct))
    c[2].metric("D30 Retention", pct(d30))
    c[3].metric("Lifetime Revenue", money(revenue))

    left, right = st.columns(2)
    with left:
        risk = (
            players["churn_risk_band"]
            .value_counts()
            .reindex(["Low", "Medium", "High", "Very High"], fill_value=0)
            .rename_axis("risk_band")
            .reset_index(name="players")
        )
        fig = px.bar(
            risk,
            x="risk_band",
            y="players",
            title="Churn risk distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        sample = players.sample(min(len(players), 12000), random_state=42)
        fig = px.scatter(
            sample,
            x="active_days_14",
            y="churn_probability_30d",
            color="segment",
            size="revenue_14",
            hover_data=["country_code", "created_platform"],
            title="Churn probability vs early engagement",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    daily = data["daily"].sort_values("date").copy()
    daily["rolling_7d"] = daily["revenue_usd"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["revenue_usd"], name="Revenue", mode="lines"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["rolling_7d"],
            name="7-day average",
            mode="lines",
        )
    )
    fig.update_layout(title="Daily revenue anomaly monitor")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cohort retention heatmap")
    heat = data["cohort"].pivot(
        index="signup_month", columns="day_number", values="retention_rate"
    )
    fig = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "Day", "y": "Signup month", "color": "Retention"},
    )
    fig.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Slingshot Studios | Player Lifecycle, Retention & Monetization")
    st.caption("Streamlit dashboard built from the supplied 2016 mobile-game analytical dataset.")

    try:
        data = load_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    players = apply_player_filters(data["players"])

    page = st.sidebar.radio(
        "Dashboard",
        ["Executive Player Health", "Player Segmentation & Monetization", "Churn Risk & Monitoring"],
    )

    if players.empty:
        st.warning("No players match the current filters.")
        return

    if page == "Executive Player Health":
        executive_page(data, players)
    elif page == "Player Segmentation & Monetization":
        segmentation_page(data, players)
    else:
        churn_page(data, players)

    st.caption(
        "Note: metrics are descriptive of the supplied sample and are not business metrics for EA or Slingshot Studios."
    )


if __name__ == "__main__":
    main()
