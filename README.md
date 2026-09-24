# AI Startup Opportunity Scanner

AI Startup Opportunity Scanner analyzes startup funding, product launches, and startup outcomes to rank market opportunities and support failure-aware business planning.

## Dataset

The project uses the attached AI startup datasets:

- `ai_startup_funding_database_2014_2025.csv`
- `Indian_Startup_Funding_Dataset.csv`
- `producthunt_products.csv`
- `startup_success_dataset.csv`

Processed copies are available in [`data/processed/`](data/processed/).

## Technologies

Python 3.10+, Pandas, NumPy, scikit-learn, XGBoost, SHAP, BERTopic, SentenceTransformers, LangChain, Gemini 3.6 Flash, ChromaDB, Streamlit, and Plotly.

## Setup

```powershell
py -3 -m pip install -r requirements.txt
```

The AI Advisor requires a Google Gemini API key in the process environment. Set it locally and never commit it:

```powershell
$env:GOOGLE_API_KEY = "<your-key>"
```

## Run

Clean the attached source data:

```powershell
py -3 scripts/run_cleaning.py
```

Start the dashboard:

```powershell
py -3 -m streamlit run dashboard/app.py
```

Open `http://localhost:8501`.

The dashboard provides Home, Funding Analytics, Opportunity Scanner, and AI Advisor pages. The Opportunity Scanner uses a weighted 0-to-10 score based on demand, funding, competition, trend momentum, and failure risk. The success model persists to `models/success_predictor.joblib`.

## Project Layout

```text
dashboard/app.py                 Streamlit dashboard
scripts/run_cleaning.py          Data cleaning entry point
src/data_processing/             DataCleaner
src/opportunity_engine/          OpportunityScorer
src/ml_models/                   XGBoost success model
src/nlp_engine/                  BERTopic trend analyzer
src/genai_engine/                LangChain RAG advisor
data/processed/                  Cleaned datasets and quality reports
```