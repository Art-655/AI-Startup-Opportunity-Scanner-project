"""Streamlit dashboard for the AI Startup Opportunity Scanner."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.genai_engine import GenAIAdvisor
from src.opportunity_engine import OpportunityScorer


st.set_page_config(
    page_title="AI Startup Opportunity Scanner",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_datasets() -> Dict[str, pd.DataFrame]:
    """Load all processed datasets used by the dashboard."""
    processed = PROJECT_ROOT / "data" / "processed"
    return {
        "ai_funding": pd.read_csv(processed / "ai_funding.csv"),
        "indian_funding": pd.read_csv(processed / "indian_funding.csv"),
        "producthunt": pd.read_csv(processed / "producthunt.csv"),
        "startup_success": pd.read_csv(processed / "startup_success.csv"),
    }


@st.cache_data(show_spinner=False)
def build_opportunity_table(
    ai_funding: pd.DataFrame,
    indian_funding: pd.DataFrame,
    producthunt: pd.DataFrame,
    startup_success: pd.DataFrame,
) -> pd.DataFrame:
    """Derive category-level scoring features from observed startup data."""
    ai = ai_funding.copy()
    indian = indian_funding.copy()
    outcomes = startup_success.copy()
    ai["sector"] = ai["sector"].fillna("Unknown")
    indian["sector"] = indian["Industry"].fillna("Unknown")
    ai["amount_usd"] = ai["amount_usd_millions"].fillna(0) * 1_000_000
    indian["amount_usd"] = indian["Funding_Amount_USD"].fillna(0)
    funding = pd.concat(
        [
            ai[["sector", "amount_usd", "year"]],
            indian[["sector", "amount_usd", "Funding_Year"]].rename(
                columns={"Funding_Year": "year"}
            ),
        ],
        ignore_index=True,
    )
    funding_summary = funding.groupby("sector", as_index=False).agg(
        funding_growth=("amount_usd", "sum"), competition_level=("sector", "size")
    )
    trend_summary = funding.groupby("sector", as_index=False).agg(
        trend_momentum=("year", lambda values: float((values >= values.max() - 2).sum()))
    )
    funding_summary = funding_summary.merge(trend_summary, on="sector", how="left")

    outcome_summary = outcomes.groupby("sector", as_index=False).agg(
        market_demand=("market_size_billion", "median"),
        failure_risk=("outcome", lambda values: (values == "Failure").mean()),
    )
    opportunity = funding_summary.merge(outcome_summary, on="sector", how="left")
    opportunity["market_demand"] = opportunity["market_demand"].fillna(
        outcomes["market_size_billion"].median()
    )
    opportunity["failure_risk"] = opportunity["failure_risk"].fillna(
        (outcomes["outcome"] == "Failure").mean()
    )
    return OpportunityScorer().calculate_scores(opportunity)


@st.cache_data(show_spinner=False)
def build_failure_postmortems(startup_success: pd.DataFrame) -> List[str]:
    """Convert observed failed startup records into factual retrieval documents."""
    failures = startup_success[startup_success["outcome"] == "Failure"].copy()
    failures = failures.drop_duplicates().head(50)
    return [
        (
            f"Historical startup record: outcome Failure; sector={row.sector}; "
            f"funding_rounds={row.funding_rounds}; founder_experience_years="
            f"{row.founder_experience_years}; team_size={row.team_size}; "
            f"market_size_billion={row.market_size_billion}; "
            f"product_traction_users={row.product_traction_users}; "
            f"burn_rate_million={row.burn_rate_million}; "
            f"revenue_million={row.revenue_million}; investor_type={row.investor_type}; "
            f"founder_background={row.founder_background}."
        )
        for row in failures.itertuples(index=False)
    ]


@st.cache_resource(show_spinner=False)
def get_advisor(postmortems: tuple[str, ...]) -> GenAIAdvisor:
    """Initialize and populate the cached RAG advisor."""
    advisor = GenAIAdvisor(persist_directory=PROJECT_ROOT / "data" / "chroma")
    advisor.ingest_failure_postmortems(list(postmortems))
    return advisor


def inject_styles() -> None:
    """Apply the dashboard visual system and hide Streamlit chrome."""
    st.markdown(
        """
        <style>
        #MainMenu, footer {visibility: hidden;}
        header {background: transparent;}
        [data-testid="stSidebar"] {border-right: 1px solid #dbe4e8;}
        [data-testid="stMetric"] {background: #f7faf9; border: 1px solid #dce8e5;
            padding: 1rem; border-radius: 10px;}
        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {color: #102a2b !important;}
        .app-kicker {color: #187a70; font-size: .78rem; font-weight: 700;
            letter-spacing: .12em; text-transform: uppercase;}
        .app-title {color: #102a2b; font-size: 2.6rem; line-height: 1.05;
            font-weight: 750; margin: .2rem 0 .8rem;}
        .app-subtitle {color: #5c6d70; font-size: 1.02rem; margin-bottom: 1.7rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(kicker: str, title: str, subtitle: str) -> None:
    """Render a consistent page header."""
    st.markdown(f'<div class="app-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_home(data: Dict[str, pd.DataFrame], opportunity: pd.DataFrame) -> None:
    """Render the dashboard overview."""
    render_header(
        "Signal intelligence",
        "AI Startup Opportunity Scanner",
        "A live view across funding, traction, outcomes, and emerging startup categories.",
    )
    top_sector = opportunity.iloc[0]["sector"] if not opportunity.empty else "N/A"
    top_score = opportunity["opportunity_score"].max() if not opportunity.empty else 0
    metrics = [
        ("Total Startups", f"{len(data['startup_success']):,}"),
        ("Avg Funding", f"${data['indian_funding']['Funding_Amount_USD'].mean() / 1e6:.1f}M"),
        ("Top Trending Sector", str(top_sector)),
        ("Highest Opportunity Score", f"{top_score:.2f}/10"),
    ]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)
    st.divider()
    st.subheader("Funding geography")
    funding = data["ai_funding"].dropna(subset=["hq_city", "hq_country"]).copy()
    funding["funding_usd"] = funding["amount_usd_millions"].fillna(0) * 1_000_000
    figure = px.scatter_geo(
        funding,
        locations="hq_country",
        locationmode="country names",
        size="funding_usd",
        color="sector",
        hover_name="company",
        hover_data={"funding_usd": ":,.0f", "hq_city": True, "sector": True},
        projection="natural earth",
        title="AI funding concentration by headquarters",
    )
    figure.update_layout(height=520, margin=dict(l=0, r=0, t=45, b=0))
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})


def render_funding_analytics(data: Dict[str, pd.DataFrame]) -> None:
    """Render funding trends and round analytics."""
    render_header("Capital flows", "Funding Analytics", "Track where capital is moving and which rounds attract it.")
    ai = data["ai_funding"]
    yearly = ai.groupby("year", as_index=False)["amount_usd_millions"].sum()
    rounds = ai.groupby("round_type", as_index=False)["amount_usd_millions"].sum().sort_values(
        "amount_usd_millions", ascending=False
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.area(yearly, x="year", y="amount_usd_millions", title="AI funding by year"),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with right:
        st.plotly_chart(
            px.bar(rounds, x="amount_usd_millions", y="round_type", orientation="h", title="Capital by round type"),
            use_container_width=True,
            config={"displaylogo": False},
        )
    st.dataframe(ai[["company", "deal_date", "round_type", "amount_usd_millions", "sector", "hq_country"]], use_container_width=True, hide_index=True)


def render_opportunity_scanner(opportunity: pd.DataFrame) -> None:
    """Render the opportunity score treemap and filter."""
    render_header("Market whitespace", "Opportunity Scanner", "Compare categories using demand, capital, competition, momentum, and risk.")
    minimum_score = st.slider("Minimum opportunity score", 0.0, 10.0, 0.0, 0.1)
    filtered = opportunity[opportunity["opportunity_score"] >= minimum_score]
    if filtered.empty:
        st.info("No sectors meet this score threshold.")
        return
    figure = px.treemap(
        filtered,
        path=["sector"],
        values="opportunity_score",
        color="opportunity_score",
        color_continuous_scale=["#dcefeb", "#187a70", "#102a2b"],
        hover_data={"opportunity_score": ":.2f", "failure_risk": ":.2%"},
        title="Opportunity score by sector",
    )
    figure.update_layout(height=560, margin=dict(l=0, r=0, t=45, b=0))
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def render_ai_advisor(data: Dict[str, pd.DataFrame], opportunity: pd.DataFrame) -> None:
    """Render the business-plan generation form."""
    render_header("Strategy copilot", "AI Advisor", "Turn a scored market signal into a failure-aware business plan.")
    industries = opportunity["sector"].dropna().astype(str).tolist()
    with st.form("business_plan_form"):
        industry = st.selectbox("Industry", industries)
        score = float(opportunity.loc[opportunity["sector"] == industry, "opportunity_score"].iloc[0])
        trends = st.text_area("Current trends", value=f"Capital and product activity in {industry}")
        submitted = st.form_submit_button("Generate business plan", type="primary")
    if submitted:
        try:
            postmortems = tuple(build_failure_postmortems(data["startup_success"]))
            with st.spinner("Retrieving historical failures and drafting the plan..."):
                advisor = get_advisor(postmortems)
                plan = advisor.generate_business_plan(industry, score, trends)
            st.markdown(plan)
        except Exception as error:
            st.error(f"AI Advisor is unavailable: {error}")


def main() -> None:
    """Run the Streamlit dashboard."""
    inject_styles()
    data = load_datasets()
    opportunity = build_opportunity_table(
        data["ai_funding"],
        data["indian_funding"],
        data["producthunt"],
        data["startup_success"],
    )
    with st.sidebar:
        st.markdown("## ◈ Scanner")
        st.caption("AI startup intelligence")
        page = st.radio(
            "Navigate",
            ["Home", "Funding Analytics", "Opportunity Scanner", "AI Advisor"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Data refreshed from processed project datasets")
    if page == "Home":
        render_home(data, opportunity)
    elif page == "Funding Analytics":
        render_funding_analytics(data)
    elif page == "Opportunity Scanner":
        render_opportunity_scanner(opportunity)
    else:
        render_ai_advisor(data, opportunity)


if __name__ == "__main__":
    main()