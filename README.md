# AI Startup Opportunity Scanner

## Project Overview

AI Startup Opportunity Scanner is an end-to-end platform for discovering startup market whitespace. It combines funding activity, product launches, startup outcomes, machine-learning predictions, trend analysis, and Gemini-powered business planning in one Streamlit dashboard.

## Business Problem

Startup ecosystems generate large volumes of funding, product, and outcome data, but founders and investors often lack a consistent way to compare market demand, capital movement, competition, trend momentum, and failure risk. The project converts those signals into an explainable Opportunity Score and a practical, failure-aware business plan.

## Objectives

- Clean and validate startup funding, launch, and outcome datasets.
- Identify sectors with strong demand and emerging momentum.
- Rank categories with a weighted Opportunity Score from 0 to 10.
- Predict startup success probability with XGBoost and class-imbalance handling.
- Explain model behavior with SHAP.
- Extract product trends with BERTopic and SentenceTransformers.
- Generate business plans grounded in historical startup outcomes using Gemini Flash and ChromaDB.

## Project Architecture

Raw Dataset 1 (AI Funding)
            │
Raw Dataset 2 (Indian Startup Funding)
            │
Raw Dataset 3 (Product Hunt Products)
            │
Raw Dataset 4 (Startup Outcomes)
            ▼
      Data Cleaning Pipeline
            ▼
      Quality Reports and CSV Outputs
            ▼
      Feature Engineering
            ▼
      Opportunity Scoring
       ┌────┼──────────────┐
       ▼    ▼              ▼
   XGBoost  BERTopic     ChromaDB
   + SHAP   Trends       + Gemini Flash
       └────┼──────────────┘
            ▼
       Streamlit Dashboard
            ▼
       Business Insights

## Tech Stack

- Python 3.10+
- Pandas and NumPy
- scikit-learn and XGBoost
- SHAP
- BERTopic and SentenceTransformers
- LangChain and Gemini 3.6 Flash
- ChromaDB
- Streamlit and Plotly
- Jupyter Notebook
- Git and GitHub

## Repository Structure

```text
dashboard/app.py                         Streamlit dashboard
scripts/run_cleaning.py                  Data cleaning entry point
scripts/build_submission_report.py       DOCX report generator
src/data_processing/                     DataCleaner
src/opportunity_engine/                  OpportunityScorer
src/ml_models/                           XGBoost success predictor
src/nlp_engine/                          BERTopic trend analyzer
src/genai_engine/                        Gemini RAG advisor
data/                                    Runtime data directory
models/                                  Runtime model directory
reports/                                 Generated report directory
AIStartupOpportunityScanner_Project.ipynb Project notebook
requirements.txt                         Dependency pins
```

## Dataset

The project uses the attached AI startup datasets:

- `ai_startup_funding_database_2014_2025.csv`
- `Indian_Startup_Funding_Dataset.csv`
- `producthunt_products.csv`
- `startup_success_dataset.csv`

The cleaning pipeline writes processed outputs to [`data/processed/`](data/processed/) during execution. Generated data and model artifacts are ignored by Git to keep the repository lightweight.

## Methodology

1. `DataCleaner` removes duplicate records, drops rows without critical identifiers, converts funding fields to numeric values, and imputes missing funding by industry median.
2. Dashboard feature engineering combines funding totals, funding recency, category counts, market size, and observed failure rates.
3. `OpportunityScorer` applies Min-Max normalization and weighted scoring:
   - Market Demand: 25%
   - Funding Growth: 20%
   - Competition Level, inverted: 20%
   - Trend Momentum: 15%
   - Failure Risk, inverted: 20%
4. `SuccessPredictor` uses XGBoost with `scale_pos_weight`, one-hot encoding, joblib persistence, and SHAP explanations.
5. `TrendAnalyzer` uses `all-MiniLM-L6-v2` embeddings with BERTopic and calculates timestamp-based momentum.
6. `GenAIAdvisor` embeds historical failure records in ChromaDB, retrieves the three most relevant records, and asks Gemini Flash to produce a mitigation-focused Markdown plan.

## Dashboard

The Streamlit dashboard is organized into four pages:

- **Home:** Startup count, average funding, top sector, highest Opportunity Score, and funding geography.
- **Funding Analytics:** Funding by year, funding by round type, and source records.
- **Opportunity Scanner:** Sector treemap and minimum-score filter.
- **AI Advisor:** Industry selection, trend context, historical failure retrieval, and Gemini-generated business plan.

Start the dashboard with:

```powershell
py -3 -m streamlit run dashboard/app.py
```

Open `http://localhost:8501`.

## Key Insights

- The cleaned startup outcome dataset contains 100,000 records.
- Observed outcomes include 55,610 failures, 42,335 acquisitions, and 2,055 IPOs.
- The dashboard currently derives 16 scored sectors.
- The observed Opportunity Score range is approximately 1.63 to 6.68.
- Foundation Model / LLM is the highest-ranked sector in the current processed data.
- Funding concentration and failure-rate signals vary substantially by sector, supporting comparative opportunity analysis rather than a single market-wide conclusion.

## Recommendations

- Validate the highest-scoring sectors with customer interviews and primary market research.
- Treat Opportunity Score as a screening signal, not an investment decision by itself.
- Use failure records to define measurable operating, financial, and go-to-market controls.
- Re-run the cleaning pipeline when new funding or launch data is added.
- Monitor Gemini API usage and embedding quotas when ingesting new postmortems.
- Keep API keys in environment variables and never commit them to the repository.

## Future Improvements

- Add automated scheduled ingestion from live funding and product APIs.
- Add sector normalization across funding and Product Hunt taxonomies.
- Add model evaluation metrics, train/test splits, and cross-validation reports.
- Add richer timestamped trend analysis to the dashboard.
- Add user authentication and persistent workspace-level saved scans.
- Add automated tests and CI checks for data contracts and dashboard smoke tests.
- Add support for Gemini's Interactions API as the integration matures.

## Author

AI Startup Opportunity Scanner Project Team
