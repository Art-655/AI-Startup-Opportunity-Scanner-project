<a name="top"></a>
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=AI%20Startup%20Opportunity%20Scanner&fontSize=30&fontColor=ffffff&animation=fadeIn&fontAlignY=35" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=500&size=20&pause=1000&color=F2C811&center=true&vCenter=true&width=700&lines=AI-Powered+Startup+Intelligence;XGBoost+%2B+SHAP+%2B+BERTopic+%2B+Gemini+RAG;Opportunity+Scoring+Across+16+Sectors;Data-Driven+Startup+Decisions" />
</p>

> **AI Startup Opportunity Scanner** is an end-to-end platform for discovering startup market whitespace. It combines funding activity, product launches, startup outcomes, machine-learning predictions, trend analysis, and Gemini-powered business planning in one Streamlit dashboard.

<!-- TECH BADGES (static) -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-SHAP-success" />
  <img src="https://img.shields.io/badge/Gemini%20Flash-RAG%20Advisor-8E75B2" />
  <img src="https://img.shields.io/badge/BERTopic-Trend%20Analysis-blue" />
</p>

<!-- LIVE REPO STATS — replace YOUR-USERNAME (and the repo name below if it differs) once pushed -->
<p align="center">
  <img src="https://img.shields.io/github/last-commit/YOUR-USERNAME/ai-startup-opportunity-scanner?style=flat-square" />
  <img src="https://img.shields.io/github/repo-size/YOUR-USERNAME/ai-startup-opportunity-scanner?style=flat-square" />
  <img src="https://img.shields.io/github/license/YOUR-USERNAME/ai-startup-opportunity-scanner?style=flat-square" />
  <img src="https://img.shields.io/github/stars/YOUR-USERNAME/ai-startup-opportunity-scanner?style=flat-square&color=yellow" />
</p>

---

## 📑 Table of Contents

- [📊 Project Overview](#project-overview)
- [❓ Business Problem](#business-problem)
- [🎯 Objectives](#objectives)
- [🏗️ Project Architecture](#project-architecture)
- [🛠️ Tech Stack](#tech-stack)
- [📁 Repository Structure](#repository-structure)
- [🗂️ Dataset](#dataset)
- [🧠 Methodology](#methodology)
- [📈 Dashboard](#dashboard)
- [🔍 Key Insights](#key-insights)
- [💼 Recommendations](#recommendations)
- [🔮 Future Improvements](#future-improvements)
- [👤 About the Developer](#about-the-developer)

---

<a name="project-overview"></a>
## 📊 Project Overview

AI Startup Opportunity Scanner is an end-to-end platform for discovering startup market whitespace. It combines funding activity, product launches, startup outcomes, machine-learning predictions, trend analysis, and Gemini-powered business planning in one Streamlit dashboard.

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="business-problem"></a>
## ❓ Business Problem

Startup ecosystems generate large volumes of funding, product, and outcome data, but founders and investors often lack a consistent way to compare market demand, capital movement, competition, trend momentum, and failure risk. The project converts those signals into an explainable Opportunity Score and a practical, failure-aware business plan.

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="objectives"></a>
## 🎯 Objectives

* Clean and validate startup funding, launch, and outcome datasets.
* Identify sectors with strong demand and emerging momentum.
* Rank categories with a weighted Opportunity Score from 0 to 10.
* Predict startup success probability with XGBoost and class-imbalance handling.
* Explain model behavior with SHAP.
* Extract product trends with BERTopic and SentenceTransformers.
* Generate business plans grounded in historical startup outcomes using Gemini Flash and ChromaDB.

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="project-architecture"></a>
## 🏗️ Project Architecture

```text
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
```

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="tech-stack"></a>
## 🛠️ Tech Stack

* Python 3.10+
* Pandas and NumPy
* scikit-learn and XGBoost
* SHAP
* BERTopic and SentenceTransformers
* LangChain and Gemini 3.6 Flash
* ChromaDB
* Streamlit and Plotly
* Jupyter Notebook
* Git and GitHub

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="repository-structure"></a>
## 📁 Repository Structure

<details>
<summary>Click to expand folder structure</summary>

| Path | Description |
| --- | --- |
| `dashboard/app.py` | Streamlit dashboard |
| `scripts/run_cleaning.py` | Data cleaning entry point |
| `scripts/build_submission_report.py` | DOCX report generator |
| `src/data_processing/` | `DataCleaner` |
| `src/opportunity_engine/` | `OpportunityScorer` |
| `src/ml_models/` | XGBoost success predictor |
| `src/nlp_engine/` | BERTopic trend analyzer |
| `src/genai_engine/` | Gemini RAG advisor |
| `data/` | Runtime data directory |
| `models/` | Runtime model directory |
| `reports/` | Generated report directory |
| `AIStartupOpportunityScanner_Project.ipynb` | Project notebook |
| `requirements.txt` | Dependency pins |

</details>

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="dataset"></a>
## 🗂️ Dataset

The project uses the attached AI startup datasets:

* `ai_startup_funding_database_2014_2025.csv`
* `Indian_Startup_Funding_Dataset.csv`
* `producthunt_products.csv`
* `startup_success_dataset.csv`

The cleaning pipeline writes processed outputs to `data/processed/` during execution. Generated data and model artifacts are ignored by Git to keep the repository lightweight.

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="methodology"></a>
## 🧠 Methodology

1. `DataCleaner` removes duplicate records, drops rows without critical identifiers, converts funding fields to numeric values, and imputes missing funding by industry median.
2. Dashboard feature engineering combines funding totals, funding recency, category counts, market size, and observed failure rates.
3. `OpportunityScorer` applies Min-Max normalization and weighted scoring:

   | Factor | Weight |
   | --- | ---: |
   | Market Demand | 25% |
   | Funding Growth | 20% |
   | Competition Level (inverted) | 20% |
   | Trend Momentum | 15% |
   | Failure Risk (inverted) | 20% |

4. `SuccessPredictor` uses XGBoost with `scale_pos_weight`, one-hot encoding, joblib persistence, and SHAP explanations.
5. `TrendAnalyzer` uses `all-MiniLM-L6-v2` embeddings with BERTopic and calculates timestamp-based momentum.
6. `GenAIAdvisor` embeds historical failure records in ChromaDB, retrieves the three most relevant records, and asks Gemini Flash to produce a mitigation-focused Markdown plan.

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="dashboard"></a>
## 📈 Dashboard

The Streamlit dashboard is organized into four pages.

<details>
<summary><strong>🏠 Home</strong> — click to expand</summary>

Startup count, average funding, top sector, highest Opportunity Score, and funding geography.

</details>

<details>
<summary><strong>💰 Funding Analytics</strong> — click to expand</summary>

Funding by year, funding by round type, and source records.

</details>

<details>
<summary><strong>🔎 Opportunity Scanner</strong> — click to expand</summary>

Sector treemap and minimum-score filter.

</details>

<details>
<summary><strong>🤖 AI Advisor</strong> — click to expand</summary>

Industry selection, trend context, historical failure retrieval, and Gemini-generated business plan.

</details>

### ▶️ Run the dashboard

```powershell
py -3 -m streamlit run dashboard/app.py
```

Then open `http://localhost:8501`.

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="key-insights"></a>
## 🔍 Key Insights

* The cleaned startup outcome dataset contains **100,000 records**.
* Observed outcomes include **55,610 failures**, **42,335 acquisitions**, and **2,055 IPOs**.
* The dashboard currently derives **16 scored sectors**.
* The observed Opportunity Score range is approximately **1.63 to 6.68**.
* **Foundation Model / LLM** is the highest-ranked sector in the current processed data.
* Funding concentration and failure-rate signals vary substantially by sector, supporting comparative opportunity analysis rather than a single market-wide conclusion.

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="recommendations"></a>
## 💼 Recommendations

* Validate the highest-scoring sectors with customer interviews and primary market research.
* Treat Opportunity Score as a screening signal, not an investment decision by itself.
* Use failure records to define measurable operating, financial, and go-to-market controls.
* Re-run the cleaning pipeline when new funding or launch data is added.
* Monitor Gemini API usage and embedding quotas when ingesting new postmortems.
* Keep API keys in environment variables and never commit them to the repository.

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="future-improvements"></a>
## 🔮 Future Improvements

* Add automated scheduled ingestion from live funding and product APIs.
* Add sector normalization across funding and Product Hunt taxonomies.
* Add model evaluation metrics, train/test splits, and cross-validation reports.
* Add richer timestamped trend analysis to the dashboard.
* Add user authentication and persistent workspace-level saved scans.
* Add automated tests and CI checks for data contracts and dashboard smoke tests.
* Add support for Gemini's Interactions API as the integration matures.

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

<a name="about-the-developer"></a>
## 👤 About the Developer

**Arunraj Tawalke**

Aspiring Data Analyst passionate about transforming raw data into actionable business insights using **Power BI**, **SQL**, **Excel**, and **Python**.

* **GitHub:** [@Art-655](https://github.com/Art-655)
* **LinkedIn:** [arunraj-tawalke-5a079828a](https://www.linkedin.com/in/arunraj-tawalke-5a079828a/)
* **Email:** [rajtawalke2004@gmail.com](mailto:rajtawalke2004@gmail.com)

<div align="right"><a href="#top">⬆️ Back to top</a></div>

---

## ⭐ If you found this project useful, consider giving it a star on GitHub! ⭐
