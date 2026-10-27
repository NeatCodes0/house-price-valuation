# Phase 3 Engineering & Diagnostic Audit: Econometric Modeling & Hedonic Pricing Engine

**Project:** Ames House Price Valuation Engine  
**Dataset:** Engineered Curated Baseline (2,412 rows, 87 columns)  
**Input Artifact:** `data/processed/housing_engineered.parquet`  
**Output Artifact:** `models/hedonic_ols_model.pkl` (4.15 MB serialized model)  
**Estimation Framework:** Ordinary Least Squares (OLS) via `statsmodels` with MacKinnon-White (HC3) Robust Standard Errors  

---

## 1. Executive Summary & Core Objectives

Phase 3 transitions the project from descriptive exploratory analysis to causal econometric inference and hedonic pricing estimation. 

The primary objectives achieved in this phase were:
1. **Hypothesis 1 (Remodel Premium):** Statistically isolate the percentage market return of home renovation while controlling for physical footprint, build quality, and property vintage.
2. **Hypothesis 2 (Marginal Bathroom Utility):** Estimate the marginal value contributed by each incremental full bathroom added to a property.
3. **The Hedonic Pricing Engine:** Fit a semi-log multi-variable regression incorporating physical attributes and spatial neighborhood fixed effects.
4. **Gauss-Markov & Robust Inference:** Apply MacKinnon-White (HC3) heteroscedasticity-consistent standard errors and audit error distributions via Q-Q plots and Jarque-Bera diagnostics.
5. **Model Serialization:** Package the fitted model for integration into a real-time web application.

---

## 2. Hypothesis Testing & Causal Inference

### 2.1 Hypothesis 1: The Valuation Impact of Remodeling

* **Economic Motivation:** Homeowners require proof that remodeling expenditure yields measurable price premiums over unremodeled homes of identical age.
* **Specification:** 
  $$\ln(\text{SalePrice}) = \beta_0 + \beta_1 (\text{is\_remodeled}) + \beta_2 (\text{total\_living\_sf}) + \beta_3 (\text{overall\_qual}) + \beta_4 (\text{property\_age}) + \varepsilon$$
* **Indicator Construction:** `is_remodeled = (year_remod_add > year_built).astype(int)`

#### Statistical Findings:
* **Coefficient ($\beta_{\text{remodel}}$):** $+0.0412$
* **HC3 Robust Standard Error:** $0.0061$
* **$t$-statistic:** $6.700$
* **$p$-value:** $2.08 \times 10^{-11}$ ($p < 0.05$)
* **Exact Percentage Impact:** $+4.20\%$ (computed via Halvorsen-Palmquist: $(e^{0.0412} - 1) \times 100$)
* **95% Confidence Interval:** $[+2.96\%, +5.47\%]$
* **Verdict:** Reject the null hypothesis ($H_0: \beta_{\text{remodel}} = 0$). Remodeling produces a statistically significant $+4.20\%$ premium (equivalent to roughly $\$6,720$ on a median $\$160,000$ home).

---

### 2.2 Hypothesis 2: Marginal Valuation Return per Bathroom

* **Economic Motivation:** Determine whether additional bathrooms add distinct value when living space is held constant.
* **Specification:**
  $$\ln(\text{SalePrice}) = \beta_0 + \beta_1 (\text{total\_bathrooms}) + \beta_2 (\text{total\_living\_sf}) + \beta_3 (\text{overall\_qual}) + \beta_4 (\text{property\_age}) + \beta_5 (\text{garage\_cars}) + \varepsilon$$

#### Statistical Findings:
* **Coefficient ($\beta_{\text{bath}}$):** $+0.0507$
* **HC3 Robust Standard Error:** $0.0060$
* **$t$-statistic:** $8.393$
* **$p$-value:** $4.76 \times 10^{-17}$ ($p < 0.05$)
* **Exact Percentage Impact:** $+5.21\%$ per full bathroom added
* **95% Confidence Interval:** $[+3.97\%, +6.46\%]$
* **Verdict:** Reject the null hypothesis ($H_0: \beta_{\text{bath}} = 0$). Adding a full bathroom contributes a statistically significant $+5.21\%$ marginal value increase (approximately $\$8,336$ on a median home).

---

## 3. The Hedonic Pricing Model Specification

### 3.1 Model Formula
To estimate market valuation across all dimensions without multicollinearity, the low-VIF structural variables and categorical neighborhood indicators were combined:

$$\ln(\text{SalePrice}) = \beta_0 + \beta_1(\text{total\_living\_sf}) + \beta_2(\text{overall\_qual}) + \beta_3(\text{total\_bathrooms}) + \beta_4(\text{property\_age}) + \beta_5(\text{years\_since\_remodel}) + \beta_6(\text{garage\_cars}) + \beta_7(\text{fireplaces}) + \beta_8(\ln(\text{lot\_area})) + \sum_{k=1}^{27} \gamma_k (\text{Neighborhood}_k) + \varepsilon$$

### 3.2 Macro Goodness-of-Fit Summary

| Metric                     | Empirical Value           | Diagnostic Assessment                             |
| :------------------------- | :------------------------ | :------------------------------------------------ |
| **Sample Size ($N$)**      | 2,412 records             | Full clean baseline preserved                     |
| **$R^2$**                  | **0.9078**                | Explains 90.78% of log price variation            |
| **Adjusted $R^2$**         | **0.9064**                | Negligible penalty for parameters; no model bloat |
| **$F$-statistic**          | **620.40** ($p = 0.0000$) | Rejects joint null hypothesis                     |
| **Log-Likelihood**         | $1,853.39$                | Optimal log-likelihood fit                        |
| **Akaike Info Crit (AIC)** | $-3,634.79$               | Highly negative penalized fit                     |

### 3.3 Structural Physical Coefficients

| Feature                   | Coefficient ($\beta$) | HC3 Std Error | $t$-statistic | $p$-value    | Exact Economic Valuation Impact                                  |
| :------------------------ | :-------------------- | :------------ | :------------ | :----------- | :--------------------------------------------------------------- |
| **`Intercept`**           | $10.0453$             | $0.0669$      | $150.11$      | $< 10^{-15}$ | Baseline valuation anchor                                        |
| **`total_living_sf`**     | $+0.0002$             | $0.0000$      | $+26.30$      | $< 10^{-15}$ | $+0.02\%$ per sq ft ($\approx +2.0\%$ per 100 sq ft)             |
| **`overall_qual`**        | $+0.0814$             | $0.0036$      | $+22.54$      | $< 10^{-15}$ | $+8.48\%$ per 1-point step on 1–10 quality scale                 |
| **`total_bathrooms`**     | $+0.0434$             | $0.0050$      | $+8.63$       | $< 10^{-15}$ | $+4.43\%$ per full bathroom added                                |
| **`property_age`**        | $-0.0016$             | $0.0002$      | $-7.00$       | $< 10^{-11}$ | $-0.16\%$ annual depreciation ($-1.6\%$ per decade)              |
| **`years_since_remodel`** | $-0.0024$             | $0.0002$      | $-13.23$      | $< 10^{-15}$ | $-0.24\%$ annual discount without renovation                     |
| **`garage_cars`**         | $+0.0414$             | $0.0050$      | $+8.20$       | $< 10^{-15}$ | $+4.23\%$ per car bay capacity                                   |
| **`fireplaces`**          | $+0.0459$             | $0.0047$      | $+9.88$       | $< 10^{-15}$ | $+4.70\%$ per fireplace                                          |
| **`np.log(lot_area)`**    | $+0.1084$             | $0.0076$      | $+14.27$      | $< 10^{-15}$ | $+0.11\%$ elasticity ($10\%$ larger lot $\approx +1.08\%$ price) |

*All structural coefficients are statistically significant at $p < 0.001$.*

---

## 4. Residual Diagnostics & Assumption Auditing

### 4.1 Statistical Distribution of Errors
* **Mean Residual:** $0.000000$ (Confirms unconstrained OLS optimization satisfied first-order conditions).
* **Residual Standard Deviation ($\sigma_{\varepsilon}$):** $0.1122$ (Typical model prediction error is approximately $\pm 11.22\%$).
* **Residual Skewness:** $-0.804$ (Mild left tail skew).
* **Residual Kurtosis:** $6.457$ (Leptokurtic / heavy tails).
* **Jarque-Bera Test:** $\text{JB} = 1460.80$, $p = 0.0000$ (Formal normality rejected).

### 4.2 Visual Verification
1. **Residuals vs. Fitted Plot:** Residuals form an even, horizontal scatter around the zero line across property tiers, confirming the natural log transformation stabilized variance.
2. **Normal Q-Q Plot:** Residuals track the theoretical normal line through the central distribution (quantiles $-1.5$ to $+2.5$), with deviations concentrated primarily in the far left tail.
3. **Econometric Treatment:** The presence of heavy left tails and non-zero kurtosis validates the use of **MacKinnon-White HC3 robust standard errors**, preventing false positives in hypothesis testing.

---

## 5. Serialized Pipeline Artifacts

```text
models/
└── hedonic_ols_model.pkl   (4.15 MB - Serialized statsmodels regression engine)

