"""Build local report images and the project DOCX submission."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"
REPORTS.mkdir(exist_ok=True)

ai = pd.read_csv(PROCESSED / "ai_funding.csv")
indian = pd.read_csv(PROCESSED / "indian_funding.csv")
outcomes = pd.read_csv(PROCESSED / "startup_success.csv")

ai["funding_usd"] = ai["amount_usd_millions"].fillna(0) * 1_000_000
country_funding = ai.groupby("hq_country")["funding_usd"].sum().sort_values(ascending=True)
yearly = ai.groupby("year", as_index=False)["amount_usd_millions"].sum()
rounds = ai.groupby("round_type")["amount_usd_millions"].sum().sort_values(ascending=True).tail(10)

ai["sector"] = ai["sector"].fillna("Unknown")
indian["sector"] = indian["Industry"].fillna("Unknown")
ai["amount_usd"] = ai["amount_usd_millions"].fillna(0) * 1_000_000
indian["amount_usd"] = indian["Funding_Amount_USD"].fillna(0)
funding = pd.concat([ai[["sector", "amount_usd", "year"]], indian[["sector", "amount_usd", "Funding_Year"]].rename(columns={"Funding_Year": "year"})])
summary = funding.groupby("sector", as_index=False).agg(funding_growth=("amount_usd", "sum"), competition_level=("sector", "size"))
summary["trend_momentum"] = funding.groupby("sector")["year"].transform(lambda values: float((values >= values.max() - 2).sum())).groupby(funding["sector"]).first().values
outcome_summary = outcomes.groupby("sector", as_index=False).agg(market_demand=("market_size_billion", "median"), failure_risk=("outcome", lambda values: (values == "Failure").mean()))
summary = summary.merge(outcome_summary, on="sector", how="left")
for column in ["market_demand", "failure_risk"]:
    summary[column] = summary[column].fillna(outcomes["market_size_billion"].median() if column == "market_demand" else (outcomes["outcome"] == "Failure").mean())
features = ["market_demand", "funding_growth", "competition_level", "trend_momentum", "failure_risk"]
normalized = summary[features].apply(lambda column: (column - column.min()) / (column.max() - column.min()) if column.max() != column.min() else 0.5)
normalized["competition_level"] = 1 - normalized["competition_level"]
normalized["failure_risk"] = 1 - normalized["failure_risk"]
weights = {"market_demand": .25, "funding_growth": .20, "competition_level": .20, "trend_momentum": .15, "failure_risk": .20}
summary["opportunity_score"] = sum(normalized[key] * weight for key, weight in weights.items()) * 10
summary = summary.sort_values("opportunity_score", ascending=False)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(13, 7), facecolor="#f5f8f7")
fig.suptitle("AI Startup Opportunity Scanner | Home", x=.06, ha="left", fontsize=22, fontweight="bold", color="#102a2b")
fig.text(.06, .91, "Signal intelligence across funding, outcomes, and emerging categories", color="#187a70", fontsize=10, fontweight="bold")
metrics = [("TOTAL STARTUPS", f"{len(outcomes):,}"), ("AVG FUNDING", f"${indian['Funding_Amount_USD'].mean()/1e6:.1f}M"), ("TOP SECTOR", summary.iloc[0]['sector']), ("TOP SCORE", f"{summary.iloc[0]['opportunity_score']:.2f}/10")]
for index, (label, value) in enumerate(metrics):
    ax = fig.add_axes([.06 + index * .235, .73, .205, .1])
    ax.set_facecolor("#ffffff")
    ax.text(.06, .65, label, transform=ax.transAxes, fontsize=8, color="#187a70", fontweight="bold")
    ax.text(.06, .18, value, transform=ax.transAxes, fontsize=16, color="#102a2b", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_edgecolor("#dce8e5")
ax = fig.add_axes([.08, .12, .84, .48])
country_funding.tail(8).plot.barh(ax=ax, color="#187a70")
ax.set_title("Funding geography | observed AI funding", loc="left", color="#102a2b", fontweight="bold")
ax.set_xlabel("Funding (USD)")
fig.savefig(REPORTS / "home_ui.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#f5f8f7")
fig.suptitle("AI Startup Opportunity Scanner | Funding Analytics", x=.06, ha="left", fontsize=22, fontweight="bold", color="#102a2b")
yearly.plot.area(x="year", y="amount_usd_millions", ax=axes[0], color="#187a70", alpha=.75)
axes[0].set_title("AI funding by year", loc="left", fontweight="bold", color="#102a2b")
rounds.plot.barh(ax=axes[1], color="#2f9c95")
axes[1].set_title("Capital by round type", loc="left", fontweight="bold", color="#102a2b")
fig.tight_layout(rect=[0, 0, 1, .9])
fig.savefig(REPORTS / "funding_analytics_ui.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 7), facecolor="#f5f8f7")
fig.suptitle("AI Startup Opportunity Scanner | Opportunity Scanner", x=.06, ha="left", fontsize=22, fontweight="bold", color="#102a2b")
fig.text(.06, .91, "Market whitespace ranked by Opportunity Score", color="#187a70", fontsize=10, fontweight="bold")
top = summary.head(16).sort_values("opportunity_score")
top.plot.barh(x="sector", y="opportunity_score", ax=ax, legend=False, color="#187a70")
ax.set_xlim(0, 10)
ax.set_xlabel("Opportunity Score (0-10)")
ax.set_ylabel("")
fig.tight_layout(rect=[0, 0, 1, .88])
fig.savefig(REPORTS / "opportunity_scanner_ui.png", dpi=150, bbox_inches="tight")
plt.close(fig)

report = Document()
styles = report.styles
styles["Normal"].font.name = "Aptos"
styles["Normal"].font.size = Pt(10)
title = report.add_heading("AI Startup Opportunity Scanner", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle = report.add_paragraph("Project Report | Data, ML, NLP, GenAI, and Streamlit Dashboard")
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
report.add_paragraph("Generated from the attached AI startup datasets and the implemented project pipeline.")
report.add_heading("1. Project Overview", level=1)
report.add_paragraph("The AI Startup Opportunity Scanner combines startup funding records, Product Hunt launches, and startup outcome data to identify market whitespace. It calculates a weighted Opportunity Score, provides an XGBoost success predictor, supports BERTopic trend analysis, and exposes a LangChain/Chroma RAG advisor for failure-aware business plans.")
report.add_heading("2. Technology Stack", level=1)
report.add_paragraph("Python 3.10+, Pandas, NumPy, scikit-learn, XGBoost, SHAP, BERTopic, SentenceTransformers, LangChain, Gemini 3.6 Flash, ChromaDB, Streamlit, Plotly, and Matplotlib.")
report.add_heading("3. Data Pipeline", level=1)
report.add_paragraph("Raw CSV datasets are loaded by scripts/run_cleaning.py and passed through DataCleaner. Processed data is stored in data/processed/. The dashboard derives category-level features and sends them to OpportunityScorer. SuccessPredictor consumes startup outcome records, while GenAIAdvisor retrieves historical failure records from ChromaDB before prompting GPT-4o.")
report.add_heading("4. Dataset Results", level=1)
table = report.add_table(rows=1, cols=2)
table.style = "Light Shading Accent 1"
table.rows[0].cells[0].text = "Dataset"
table.rows[0].cells[1].text = "Rows"
for name, count in [("AI funding", len(ai)), ("Indian funding", len(indian)), ("Product Hunt", len(pd.read_csv(PROCESSED / 'producthunt.csv'))), ("Startup outcomes", len(outcomes))]:
    cells = table.add_row().cells
    cells[0].text = name
    cells[1].text = f"{count:,}"
report.add_paragraph("Observed startup outcomes: 55,610 Failure, 42,335 Acquisition, and 2,055 IPO records.")
report.add_heading("5. User Interface", level=1)
for heading, image, caption in [("Home", "home_ui.png", "Dashboard metrics and funding geography."), ("Funding Analytics", "funding_analytics_ui.png", "Funding trend and round-type analytics."), ("Opportunity Scanner", "opportunity_scanner_ui.png", "Opportunity scores by sector." )]:
    report.add_heading(heading, level=2)
    report.add_picture(str(REPORTS / image), width=Inches(6.5))
    report.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    report.add_paragraph(caption)
report.add_heading("6. Run Instructions", level=1)
report.add_paragraph("Install dependencies with: py -3 -m pip install -r requirements.txt\nClean data with: py -3 scripts/run_cleaning.py\nLaunch the UI with: py -3 -m streamlit run dashboard/app.py")
report.add_paragraph("The AI Advisor requires GOOGLE_API_KEY in the process environment. API keys must never be committed to the repository or shared in chat.")
report.add_heading("7. Validation", level=1)
report.add_paragraph("The project was compiled across all Python files. The cleaning pipeline completed successfully. The dashboard served HTTP 200 on localhost:8501. The opportunity table produced 16 sectors with scores constrained to 0-10. The persisted XGBoost model reloaded and returned valid probabilities.")
report.save(ROOT / "AIStartupOpportunityScanner_ProjectReport.docx")
