# Phase 2 Engineering & Diagnostic Audit: Exploratory Data Analysis & Econometric Pricing Dynamics

**Project:** Ames House Price Valuation Engine  
**Dataset:** Dean De Cock's Ames Housing Dataset (Cleaned Baseline: 2,412 rows, 82 columns)  
**Output Artifact:** `data/processed/housing_engineered.parquet` (2,412 rows, 87 columns)  
**Persistent Database:** `data/housing.duckdb` -> Table `housing_engineered`  

---

## 1. Executive Summary & Core Analytical Objectives

In Phase 1, raw property records were ingested into DuckDB, filtered using domain boundaries (Dean De Cock's $4,000\text{ sq ft}$ living area ceiling and the arm's-length transaction filter $\text{SaleCondition} = \text{'Normal'}$), and sanitized of missing values via structural imputation and spatial neighborhood medians.

The objective of Phase 2 was to conduct exploratory data analysis (EDA) and econometric feature engineering prior to model fitting. Specifically, Phase 2 addressed four core analytical questions:
1. **Target Distribution Mechanics:** Is the raw sale price normally distributed, and does applying a natural log transformation satisfy classical linear regression assumptions?
2. **Composite Metric Formulation:** Can fragmented architectural features (e.g., above-ground living space vs. basement footprint; multiple bathroom categories) be combined into higher-signal composite predictors?
3. **Spatial Valuation Benchmarks:** How does property value vary across geographical submarkets, and what is the exact percentage premium or discount of each neighborhood relative to the citywide median?
4. **Collinearity Auditing:** Which explanatory variables carry redundant variance, and how does the Variance Inflation Factor (VIF) guide feature pruning?

---

## 2. Target Variable Diagnosis & Normalization

### 2.1 The Problem with Raw Sale Price
Ordinary Least Squares (OLS) regression minimizes the sum of squared residuals:
$$\min_{\beta} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

When the dependent variable $y$ exhibits heavy right-skewness, large values exert disproportionate leverage on squared errors. A \$50,000 error on a \$600,000 property produces a squared penalty 25 times larger than a \$10,000 error on a \$150,000 home ($2.5 \times 10^9$ vs. $1.0 \times 10^8$), forcing the model to overfit luxury outliers at the expense of typical market transactions.

### 2.2 Empirical Statistical Moments
Evaluation of `sale_price` across the 2,412 clean properties yielded the following distribution metrics:

| Metric       | Raw `sale_price` | Log-Transformed `np.log(sale_price)` | Statistical Standard for Normality        |
| :----------- | :--------------- | :----------------------------------- | :---------------------------------------- |
| **Skewness** | **+1.594**       | **+0.133**                           | $[-0.5, +0.5]$ (Symmetrical)              |
| **Kurtosis** | **+4.390**       | **+0.602**                           | $\approx 0.0$ (Mesokurtic / Normal tails) |

### 2.3 Mathematical Transformation
To normalize the distribution and stabilize error variance (homoscedasticity), the natural log transformation was applied:
$$\text{log\_sale\_price} = \ln(\text{SalePrice})$$

```python
# Implementation snippet from notebooks/02_exploratory_data_analysis.ipynb
price = df["sale_price"]
log_price = np.log(price)

df["log_sale_price"] = log_price
```

### 2.4 Economic Interpretation of Logged Dependent Variables
In a linear model where the target is logged ($\ln(Y) = \beta_0 + \beta_1 X_1$):
$$\frac{d \ln(Y)}{dX_1} = \frac{1}{Y}\frac{dY}{dX_1} = \beta_1 \implies \% \Delta Y \approx (\beta_1 \times 100)\%$$

This shifts the interpretation of regression coefficients from flat dollar amounts to **percentage contributions**, reflecting the real-world principle that amenities scale proportionally with property tier.

---

## 3. Econometric Feature Engineering

Raw data contains fragmented measurements that dilute predictive signal when evaluated separately. Four composite indicators were engineered to mirror appraisal workflows.

### 3.1 Total Usable Living Area (`total_living_sf`)
Real estate appraisers evaluate the combined usable volume of a home rather than treating levels as isolated spaces.
$$\text{total\_living\_sf} = \text{gr\_liv\_area} + \text{total\_bsmt\_sf}$$

```python
df["total_living_sf"] = df["gr_liv_area"] + df["total_bsmt_sf"]
```
- **Validation:** Pearson correlation with $\ln(\text{SalePrice})$ increased from $0.73$ (for `gr_liv_area` alone) and $0.64$ (for `total_bsmt_sf` alone) to **$0.82$** for the combined composite metric.

### 3.2 Total Bathroom Count (`total_bathrooms`)
The dataset originally stored full and half bathrooms across two separate vertical levels (above-grade and basement). In residential appraisal standards, a half-bath (powder room containing sink and toilet without bath/shower) is weighted at $0.5$ of a full bath.
$$\text{total\_bathrooms} = \text{full\_bath} + 0.5(\text{half\_bath}) + \text{bsmt\_full\_bath} + 0.5(\text{bsmt\_half\_bath})$$

```python
df["total_bathrooms"] = (
    df["full_bath"] 
    + (0.5 * df["half_bath"]) 
    + df["bsmt_full_bath"] 
    + (0.5 * df["bsmt_half_bath"])
)
```
- **Validation:** Produces a standardized, continuous metric with a $0.68$ correlation to log sale price.

### 3.3 Property Age & Remodel Depreciation Anchors
Calendar years (`year_built`, `yr_sold`, `year_remod_add`) are unstandardized for regression lines. They were converted into elapsed time durations relative to the transaction date.
$$\text{property\_age} = \text{yr\_sold} - \text{year\_built}$$
$$\text{years\_since\_remodel} = \text{yr\_sold} - \text{year\_remod_add}$$

```python
df["property_age"] = (df["yr_sold"] - df["year_built"]).clip(lower=0)
df["years_since_remodel"] = (df["yr_sold"] - df["year_remod_add"]).clip(lower=0)
```
- **Defensive Sanitation (`.clip(lower=0)`):** New construction properties pre-sold during municipal paperwork processing can occasionally show build completion dates stamped after the deed transfer date (e.g., sold late 2007, marked built in 2008), which produces a negative age ($-1$). Clipping prevents negative durations from entering the regression matrix.

---

## 4. Spatial Benchmarking via DuckDB SQL

Location accounts for major valuation variation independent of physical building attributes. DuckDB was queried directly using Common Table Expressions (CTEs) and SQL window operations to compute neighborhood baseline statistics.

```sql
WITH neighborhood_summary AS (
    SELECT 
        neighborhood,
        COUNT(*) AS total_sales,
        ROUND(AVG(sale_price), 0)::INT AS mean_price,
        ROUND(MEDIAN(sale_price), 0)::INT AS median_price,
        ROUND(AVG(sale_price / total_living_sf), 2) AS avg_price_per_sqft,
        ROUND(STDDEV(sale_price), 0)::INT AS price_volatility
    FROM df_engineered
    GROUP BY neighborhood
),
citywide_metrics AS (
    SELECT 
        MEDIAN(sale_price) AS city_median_price,
        AVG(sale_price / total_living_sf) AS city_avg_ppsf
    FROM df_engineered
)
SELECT 
    ns.neighborhood,
    ns.total_sales,
    ns.median_price,
    ns.avg_price_per_sqft,
    ns.price_volatility,
    ROUND(((ns.median_price - cm.city_median_price) / cm.city_median_price) * 100, 1) AS pct_premium_vs_city
FROM neighborhood_summary ns
CROSS JOIN citywide_metrics cm
WHERE ns.total_sales >= 20
ORDER BY ns.median_price DESC;
```

### 4.1 Geographical Market Tiers (Minimum 20 Arms-Length Transactions)

#### Top 5 Premium Submarkets:
1. **Northridge (`NoRidge`):** Median \$301,750 | \$83.09/sq ft | **+90.1%** premium over city median.
2. **Northridge Heights (`NridgHt`):** Median \$284,000 | \$85.77/sq ft | **+78.9%** premium over city median.
3. **Veenker (`Veenker`):** Median \$257,500 | \$79.57/sq ft | **+62.2%** premium over city median.
4. **Stone Brook (`StoneBr`):** Median \$251,000 | \$83.14/sq ft | **+58.1%** premium over city median.
5. **Somerset (`Somerst`):** Median \$222,250 | \$82.25/sq ft | **+40.0%** premium over city median.

#### Bottom 5 Affordable Submarkets:
1. **Meadow Village (`MeadowV`):** Median \$89,375 | \$57.93/sq ft | **-43.7%** discount relative to city median.
2. **Briardale (`BrDale`):** Median \$107,000 | \$64.82/sq ft | **-32.6%** discount relative to city median.
3. **Iowa DOT / Rail Corridor (`IDOTRR`):** Median \$116,250 | \$57.90/sq ft | **-26.8%** discount relative to city median.
4. **Old Town (`OldTown`):** Median \$122,125 | \$58.66/sq ft | **-23.1%** discount relative to city median.
5. **Edwards (`Edwards`):** Median \$125,000 | \$63.30/sq ft | **-21.3%** discount relative to city median.

---

## 5. Collinearity Diagnosis & Variance Inflation Factor (VIF)

### 5.1 Correlation Matrix Analysis
Pairwise Pearson correlations between proposed regression candidates revealed high inter-feature dependencies:
- `garage_cars` and `garage_area` showed an $r = 0.89$ correlation, indicating direct collinearity.
- `first_flr_sf` and `total_bsmt_sf` showed an $r = 0.77$ correlation, reflecting standard single-story architectural layouts.
- `gr_liv_area` and `total_living_sf` showed an $r = 0.86$ correlation due to the additive relationship of the composite metric.

### 5.2 Multicollinearity Auditing via VIF
The Variance Inflation Factor measures how much the variance of an estimated regression coefficient increases due to collinearity with other features:
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
where $R_j^2$ is the coefficient of determination from regressing feature $X_j$ against all remaining explanatory variables.

```python
# Implementation snippet from notebooks/02_exploratory_data_analysis.ipynb
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = df[vif_candidates].copy()
X_vif["Intercept"] = 1.0

vif_records = [
    {"feature": col, "VIF": round(variance_inflation_factor(X_vif.values, i), 2)}
    for i, col in enumerate(X_vif.columns) if col != "Intercept"
]
```

### 5.3 VIF Audit Results

| Feature Name          | VIF Score | Decision     | Methodological Justification                                                                                                                                                                                       |
| :-------------------- | :-------- | :----------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`garage_cars`**     | **5.30**  | **Retained** | Exceeds the standard rule-of-thumb threshold of 5.0 when paired with `garage_area`. Retained because car capacity correlates more strongly with price ($0.67$ vs. $0.64$) and matches how buyers evaluate utility. |
| **`garage_area`**     | **4.87**  | **Pruned**   | Dropped to eliminate collinearity with `garage_cars`.                                                                                                                                                              |
| `total_living_sf`     | 2.50      | Retained     | Low collinearity; primary proxy for home volume.                                                                                                                                                                   |
| `overall_qual`        | 2.49      | Retained     | Low collinearity; primary proxy for build quality.                                                                                                                                                                 |
| `total_bathrooms`     | 2.03      | Retained     | Clean independent signal.                                                                                                                                                                                          |
| `property_age`        | 1.98      | Retained     | Captures baseline physical depreciation.                                                                                                                                                                           |
| `years_since_remodel` | 1.65      | Retained     | Captures recent structural improvements.                                                                                                                                                                           |
| `fireplaces`          | 1.37      | Retained     | Independent amenity metric.                                                                                                                                                                                        |
| `lot_area`            | 1.14      | Retained     | Independent land parcel size metric.                                                                                                                                                                               |

---

## 6. Persisted Artifacts & Pipeline State

Phase 2 completed the data pipeline preparation, expanding the dataset from 82 to 87 columns without dropping any of the 2,412 baseline records:

```text
data/
├── housing.duckdb
│   ├── raw_housing          (2,930 rows, 82 cols)
│   ├── housing_clean        (2,412 rows, 82 cols)
│   └── housing_engineered   (2,412 rows, 87 cols)  <-- Generated in Phase 2
└── processed/
    ├── housing_clean.parquet      (2,412 rows, 82 cols)
    └── housing_engineered.parquet (2,412 rows, 87 cols)  <-- Generated in Phase 2
```

### Git Commit Record
```bash
git add notebooks/02_exploratory_data_analysis.ipynb data/processed/housing_engineered.parquet docs/
git commit -m "feat: complete phase 2 eda, feature engineering, vif audit, and persistence"
```

The data is fully structured and validated for **Phase 3: Econometric Hypothesis Testing & Ordinary Least Squares (OLS) Modeling**. 