# Phase 4 Engineering & Deployment Audit: Interactive Hedonic Valuation Dashboard

**Project:** Ames House Price Valuation Engine  
**Framework:** Streamlit, Matplotlib, Seaborn  
**Artifacts Ingested:**  
- `models/hedonic_ols_model.pkl` (Fitted statsmodels OLS engine with HC3 covariance)  
- `data/processed/housing_engineered.parquet` (2,412 rows, 87 columns for empirical benchmarks)  
**Entrypoint:** `app/app.py`  

---

## 1. System Architecture & Component Design

The user-facing application bridges econometric regression outputs directly to real-time valuation inference.

### 1.1 Ingestion & Cache Mechanics
- Used `@st.cache_resource` for the serialized regression object (`hedonic_ols_model.pkl`) to avoid deserialization latency on state changes.
- Used `@st.cache_data` for historical benchmark retrieval from Apache Parquet, ensuring rapid calculation of submarket medians and empirical distributions.

### 1.2 Mathematical Transformation Pipeline
1. **Feature Input Vector:** Collects live inputs for physical dimensions (`total_living_sf`, `lot_area`, `overall_qual`, `total_bathrooms`, `property_age`, `years_since_remodel`, `garage_cars`, `fireplaces`) and geographical fixed effects (`neighborhood`).
2. **Interval Forecasting:** Calls `model.get_prediction(input_df).summary_frame(alpha=0.05)` to compute both the point estimate and lower/upper 95% prediction bounds on the log scale.
3. **USD Re-Transformation:** Exponentiates predictions via $\exp(\cdot)$ to convert log estimates back to dollar-denominated valuations.

---

## 2. Interface Features & Analytical Views

1. **Executive KPI Cards:**
   - Real-time estimated property value.
   - 95% confidence bounds ($\pm 11.22\%$ typical residual margin).
   - Local submarket median comparison and percentage delta.
2. **Spatial Benchmark Density Plot:**
   - KDE curve illustrating the true historical price distribution of the selected neighborhood.
   - Dynamic vertical indicator lines comparing the user's customized property directly against the neighborhood median.
3. **Marginal Attribution Table:**
   - Itemized economic breakdown showing dollar returns for incremental changes (e.g., adding a bathroom, upgrading quality ratings, or age-based depreciation).
4. **Diagnostic Integrity Panel:**
   - Live display of model macro statistics ($R^2_{\text{adj}} = 0.9064$, HC3 robust standard errors, $N = 2,412$).