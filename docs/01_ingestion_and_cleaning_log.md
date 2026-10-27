---
title: "01 Ingestion and Data Sanitation Log"
project: "House Price Diagnostic & Valuation Engine"
stage: "01_ingestion_and_cleaning"
status: "Completed"
tags:
  - runbook
  - data-engineering
  - duckdb
  - ames-housing
  - data-audit
---

# 📋 Ingestion & Schema Cleaning Audit Log

## 1. Environment & Ingestion Recovery
- **Encountered Issues & Mitigations:**
  - *Jupyter Kernel Desync:* VS Code initially bound `.ipynb` files to the MSSQL notebook provider. Switched and registered virtual environment kernel `Python (House Price Engine)` via `ipykernel install`.
  - *Broken Upstream Mirrors:* 404 response on third-party GitHub raw mirrors resolved by fetching official Dean De Cock JSE dataset via Python standard library `urllib` directly.
  - *SQL Reserved Identifier Conflicts:* DuckDB raised parser exceptions on unquoted `Order` and hyphenated/spaced columns (`Gr Liv Area`, `Sale Condition`). Resolved by introducing strict ANSI double-quoting standards across all queries.

## 2. Statistical Baseline & Filtering Rules
- **Outlier Elimination:**
  - Applied Dean De Cock threshold: Removed 5 commercial/agricultural scale records with `gr_liv_area > 4000 sq ft`.
  - Audited Edwards neighborhood cluster: 3 properties with $> 4600\text{ sq ft}$ sold at deep discounts ($<\$185,000$) under `Partial` conditions.
- **Transaction Arm's-Length Filter:**
  - Isolated `sale_condition = 'Normal'` (2,413 raw records, 82.35% of volume).
  - Combined filter materialized `housing_curated` at exactly **2,412 records**.

## 3. Domain-Aware Missing Value Remediation
- **Structural Null Distinction:**
  - Categorical absence (`pool_qc`, `misc_feature`, `alley`, `garage_*`, `bsmt_*`): Imputed with explicit categorical token `"None"`.
  - Numerical absence (`garage_cars`, `garage_area`, `total_bsmt_sf`, `mas_vnr_area`): Imputed with numerical `0`.
- **Spatial Proxies:**
  - Imputed `lot_frontage` using neighborhood-level medians (`groupby("neighborhood")`), falling back to global median.
- **Verification:** Target table `housing_clean` validated with **0 null values** across all 82 original attributes.

## 4. Pipeline Artifacts
- **Database Table:** `housing.duckdb -> housing_clean` (2,412 rows, 82 columns)
- **Downstream Parquet:** `data/processed/housing_clean.parquet`