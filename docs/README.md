# 🏛️ Ames Real Estate Hedonic Valuation Engine & Econometric Diagnostic Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-OLS_%26_HC3-blue?style=for-the-badge)](https://www.statsmodels.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP_Database-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)](https://duckdb.org)
[![Apache Parquet](https://img.shields.io/badge/Apache_Parquet-Columnar_Storage-5085A5?style=for-the-badge)](https://parquet.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

An end-to-end, econometrically grounded real estate pricing pipeline and interactive valuation system. Built on Dean De Cock's Ames Housing Dataset, this engine applies data hygiene, spatial window analytics, Variance Inflation Factor (VIF) collinearity pruning, causal hypothesis testing, and MacKinnon-White (HC3) robust standard errors, achieving an **Adjusted $R^2$ of 0.9064** across 2,412 arm's-length residential transactions.

---

<details open>
<summary><b>🏛️ 1. Architectural Overview & Data Flow</b></summary>
<br>

The platform is designed around four decoupled phases:

```text
[ Raw Ames Housing Data (2,930 rows) ]
                 │
                 ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 1: Ingestion, Hygiene & Database Persistence     │
 │ • Domain bounds: De Cock's GrLivArea < 4,000 sq ft      │
 │ • Transaction filter: SaleCondition == 'Normal'         │
 │ • Structural & spatial imputation                       │
 │ • DuckDB OLAP persistence: data/housing.duckdb          │
 └─────────────────────────┬───────────────────────────────┘
                           │ (2,412 Clean Baseline Records)
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 2: Feature Engineering & Collinearity Audit       │
 │ • ln(SalePrice) normalization (Skewness: 1.594 ➔ 0.133) │
 │ • Composite metrics: total_living_sf, total_bathrooms   │
 │ • Spatial window queries & neighborhood median indexing │
 │ • Multicollinearity audit via VIF                       │
 │ • Columnar caching: housing_engineered.parquet          │
 └─────────────────────────┬───────────────────────────────┘
                           │ (87 Engineered Features)
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 3: Econometric Modeling & Causal Inference        │
 │ • Hypothesis 1: Renovation premium verification         │
 │ • Hypothesis 2: Marginal bathroom returns               │
 │ • Hedonic OLS regression with neighborhood dummies      │
 │ • MacKinnon-White (HC3) heteroscedasticity correction   │
 │ • Normality auditing (Jarque-Bera, Q-Q diagnostics)    │
 │ • Model artifact: models/hedonic_ols_model.pkl          │
 └─────────────────────────┬───────────────────────────────┘
                           │ (4.15 MB Serialized Engine)
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 4: Production Dashboard Deployment                │
 │ • Real-time point estimates & 95% prediction intervals  │
 │ • Dynamic KDE neighborhood benchmark distributions      │
 │ • Marginal dollar attribution per feature adjustment    │
 └─────────────────────────────────────────────────────────┘
```
</details>

---

<details>
<summary><b>🔬 2. Key Econometric Findings & Causal Impacts</b></summary>
<br>

All statistical estimates are calculated using **MacKinnon-White (HC3) robust standard errors** to protect against heteroscedastic pricing variance across property price tiers:

| Economic Research Question            | Empirical Coefficient ($\beta$) | $t$-statistic | $p$-value              | Economic Return / Valuation Impact                                                                              |
| :------------------------------------ | :------------------------------ | :------------ | :--------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **Renovation Premium ($H_1$)**        | $+0.0412$                       | $+6.700$      | $2.08 \times 10^{-11}$ | **$+4.20\%$ valuation premium** over identical-age unremodeled homes ($\approx \$6,720$ equity on median home). |
| **Marginal Bathroom Utility ($H_2$)** | $+0.0507$                       | $+8.393$      | $4.76 \times 10^{-17}$ | **$+5.21\%$ marginal value** per full bathroom added ($\approx \$8,336$ equity on median home).                 |
| **Living Space Expansion**            | $+0.0002$                       | $+26.302$     | $< 10^{-15}$           | **$+2.02\%$ value increase per $100\text{ sq ft}$** of finished living space.                                   |
| **Material/Finish Quality**           | $+0.0814$                       | $+22.535$     | $< 10^{-15}$           | **$+8.48\%$ value premium** per step increase on the 1–10 Overall Quality index.                                |
| **Garage Vehicle Capacity**           | $+0.0414$                       | $+8.199$      | $< 10^{-15}$           | **$+4.23\%$ value increase** per additional garage bay.                                                         |
| **Property Aging Depreciation**       | $-0.0016$                       | $-6.995$      | $< 10^{-11}$           | **$-0.16\%$ annual structural depreciation** ($-1.6\%$ loss per decade).                                        |
| **Unrenovated Age Lag**               | $-0.0024$                       | $-13.231$     | $< 10^{-15}$           | **$-0.24\%$ annual discount** for deferred property modernization.                                              |
</details>

---

<details>
<summary><b>📊 3. Core Statistical Modeling & Residual Diagnostics</b></summary>
<br>

### Econometric Specification
The hedonic engine estimates sale price as a function of physical structural utility and spatial submarket indicators:

$$\ln(\text{SalePrice}) = \beta_0 + \beta_1(\text{total\_living\_sf}) + \beta_2(\text{overall\_qual}) + \beta_3(\text{total\_bathrooms}) + \beta_4(\text{property\_age}) + \beta_5(\text{years\_since\_remodel}) + \beta_6(\text{garage\_cars}) + \beta_7(\text{fireplaces}) + \beta_8(\ln(\text{lot\_area})) + \sum_{k=1}^{27} \gamma_k (\text{Neighborhood}_k) + \varepsilon$$

### Goodness-of-Fit Summary
- **Sample Observations ($N$):** $2,412$ arm's-length sales
- **R-squared ($R^2$):** $0.9078$
- **Adjusted R-squared ($R^2_{\text{adj}}$):** **$0.9064$**
- **F-statistic:** $620.40$ ($p < 10^{-300}$)
- **Akaike Information Criterion (AIC):** $-3,634.79$
- **Typical Prediction Error ($\sigma_\varepsilon$):** $\approx \pm 11.22\%$

### Residual Diagnostics & HC3 Justification
- **Residual Mean:** $0.000000$ (Confirms unbiased OLS optimization satisfying $\mathbf{X}^T \mathbf{e} = \mathbf{0}$)
- **Jarque-Bera Test:** $1460.80$ ($p < 10^{-300}$), Kurtosis $= 6.457$, Skewness $= -0.804$
- **Inference Implication:** While central quantiles ($-1.5 \le Z \le +2.5$) track Gaussian expectations, unrecorded municipal distress leads to heavy left-tail dispersion. Standard OLS deflates standard errors under non-normal error spreads; implementing **HC3 covariance** guarantees non-inflated $t$-statistics and calibrated hypothesis tests.
</details>

---

<details>
<summary><b>📁 4. Project Directory Structure</b></summary>
<br>

```text
house-price-valuation/
├── app/
│   └── app.py                             # Interactive Streamlit valuation application
├── data/
│   ├── raw/                               # Raw Ames transaction dataset
│   ├── processed/
│   │   └── housing_engineered.parquet     # Curated 87-column Parquet artifact (2,412 rows)
│   └── housing.duckdb                     # Persistent DuckDB embedded database
├── docs/
│   ├── 01_ingestion_and_cleaning_log.md   # Data filtering, boundary conditions & imputation
│   ├── 02_eda_and_feature_engineering.md  # Target normalization, VIF audits & spatial metrics
│   ├── 03_econometric_modeling.md         # OLS regressions, HC3 corrections & hypothesis tests
│   └── 04_deployment_and_dashboard_audit.md# App architecture, smearing & caching log
├── models/
│   └── hedonic_ols_model.pkl              # Serialized statsmodels regression artifact (4.15 MB)
├── notebooks/
│   ├── 01_ingestion_and_sanitation.ipynb  # Pipeline Stage 1 execution
│   ├── 02_exploratory_data_analysis.ipynb # Pipeline Stage 2 execution
│   └── 03_econometrics_modeling.ipynb     # Pipeline Stage 3 execution
├── sql/                                   # Standalone SQL queries and window logic
├── src/
│   ├── config.py                          # Dynamic project root & folder path anchors
│   └── __init__.py
├── pyproject.toml                         # Project metadata and pinned dependencies
└── README.md                              # System documentation
```
</details>

---

<details>
<summary><b>⚙️ 5. Installation, Setup & Reproduction</b></summary>
<br>

### Prerequisites
- Python 3.11+
- Windows PowerShell / Linux Bash / macOS Terminal
- Git

### 1. Clone & Environment Setup
```powershell
git clone https://github.com/<your-username>/house-price-valuation.git
cd house-price-valuation

python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows (or 'source venv/bin/activate' on Linux/macOS)
pip install --upgrade pip
pip install -e .
```

### 2. Sequential Pipeline Execution
1. **Phase 1: Ingestion & Hygiene**  
   Run `notebooks/01_ingestion_and_sanitation.ipynb` to filter outliers and build DuckDB.
2. **Phase 2: Exploratory Data Analysis & Feature Engineering**  
   Run `notebooks/02_exploratory_data_analysis.ipynb` to engineer features and run VIF checks.
3. **Phase 3: Econometric Estimation & Artifact Persistence**  
   Run `notebooks/03_econometrics_modeling.ipynb` to fit HC3 models and export `models/hedonic_ols_model.pkl`.
</details>

---

<details>
<summary><b>💻 6. Interactive Valuation Dashboard (Streamlit)</b></summary>
<br>

Launch the local Streamlit dashboard:

```powershell
streamlit run app/app.py
```

The app will launch at `http://localhost:8501`, featuring:
- **Real-Time Appraisals:** Dynamic point valuations with analytical 95% prediction intervals.
- **Submarket Benchmarking:** Live KDE distribution overlay comparing the configured home against historical sales in the selected Ames neighborhood.
- **Marginal Dollar Attribution:** Precise breakdown of dollar equity gains per square foot, bathroom, or renovation adjustment.
- **Model Transparency:** Audit panel displaying the underlying formula, $R^2$, and degrees of freedom.
</details>

---

<details>
<summary><b>📚 7. Engineering Documentation & Audit Logs</b></summary>
<br>

Each milestone in the pipeline is thoroughly documented with derivations:
- [Phase 1: Ingestion & Cleaning Log](docs/01_ingestion_and_cleaning_log.md)
- [Phase 2: EDA & Feature Engineering Log](docs/02_eda_and_feature_engineering.md)
- [Phase 3: Econometric Modeling & Hypothesis Testing Log](docs/03_econometric_modeling.md)
- [Phase 4: Production Dashboard Audit](docs/04_deployment_and_dashboard_audit.md)
</details>

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.