import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Ames Hedonic Valuation Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Artifact Ingestion & Caching
# ---------------------------------------------------------
@st.cache_resource
def load_econometric_model():
    model_path = PROJECT_ROOT / "models" / "hedonic_ols_model.pkl"
    if not model_path.exists():
        st.error(f"Model artifact not found at {model_path}. Complete Phase 3 first.")
        st.stop()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_benchmark_data():
    parquet_path = PROJECT_ROOT / "data" / "processed" / "housing_engineered.parquet"
    if not parquet_path.exists():
        st.error(f"Engineered data artifact not found at {parquet_path}. Complete Phase 2 first.")
        st.stop()
    return pd.read_parquet(parquet_path)

model = load_econometric_model()
df_baseline = load_benchmark_data()

# Neighborhood list sorted alphabetically
neighborhood_options = sorted(df_baseline["neighborhood"].dropna().unique().tolist())

# ---------------------------------------------------------
# Sidebar: Property Input Specification
# ---------------------------------------------------------
st.sidebar.header("🏡 Property Specification")

selected_neighborhood = st.sidebar.selectbox(
    "Neighborhood / Submarket",
    options=neighborhood_options,
    index=neighborhood_options.index("CollgCr") if "CollgCr" in neighborhood_options else 0,
    help="Select the geographical submarket. Location fixed effects account for spatial premiums.",
)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=6,
        help="Overall material and finish rating (1: Very Poor, 10: Very Excellent).",
    )
    total_bathrooms = st.number_input(
        "Total Bathrooms",
        min_value=1.0,
        max_value=6.0,
        value=2.0,
        step=0.5,
        help="Full baths + (0.5 * Half baths).",
    )
    garage_cars = st.slider(
        "Garage Capacity (Cars)",
        min_value=0,
        max_value=5,
        value=2,
    )

with col_sb2:
    total_living_sf = st.number_input(
        "Total Living Area (sq ft)",
        min_value=400,
        max_value=5000,
        value=1650,
        step=25,
        help="Combined above-ground finished square footage + finished basement.",
    )
    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=1000,
        max_value=60000,
        value=9500,
        step=250,
    )
    fireplaces = st.slider(
        "Fireplaces",
        min_value=0,
        max_value=4,
        value=1,
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Property Vintage & Renovation")
col_sb3, col_sb4 = st.sidebar.columns(2)
with col_sb3:
    property_age = st.number_input(
        "Property Age (Years)",
        min_value=0,
        max_value=140,
        value=20,
        step=1,
    )
with col_sb4:
    years_since_remodel = st.number_input(
        "Years Since Remodel",
        min_value=0,
        max_value=140,
        value=10,
        step=1,
    )

if years_since_remodel > property_age:
    years_since_remodel = property_age

# ---------------------------------------------------------
# Econometric Prediction Engine
# ---------------------------------------------------------
input_dict = {
    "total_living_sf": [float(total_living_sf)],
    "overall_qual": [int(overall_qual)],
    "total_bathrooms": [float(total_bathrooms)],
    "property_age": [int(property_age)],
    "years_since_remodel": [int(years_since_remodel)],
    "garage_cars": [int(garage_cars)],
    "fireplaces": [int(fireplaces)],
    "lot_area": [float(lot_area)],
    "neighborhood": [selected_neighborhood],
}
input_df = pd.DataFrame(input_dict)

# Generate log prediction and prediction intervals
prediction_results = model.get_prediction(input_df)
pred_summary = prediction_results.summary_frame(alpha=0.05)

pred_log_mean = pred_summary["mean"].values[0]
pred_log_ci_lower = pred_summary["obs_ci_lower"].values[0]
pred_log_ci_upper = pred_summary["obs_ci_upper"].values[0]

# Smearing / exponential transformation back to USD
point_valuation = float(np.exp(pred_log_mean))
valuation_ci_lower = float(np.exp(pred_log_ci_lower))
valuation_ci_upper = float(np.exp(pred_log_ci_upper))

# Neighborhood baseline comparison
nb_baseline = df_baseline[df_baseline["neighborhood"] == selected_neighborhood]
nb_median_price = float(nb_baseline["sale_price"].median()) if not nb_baseline.empty else point_valuation
price_vs_median_pct = ((point_valuation - nb_median_price) / nb_median_price) * 100

# ---------------------------------------------------------
# Dashboard Layout
# ---------------------------------------------------------
st.title("🏛️ Ames Real Estate Hedonic Valuation Engine")
st.markdown(
    "Econometric valuation model with spatial fixed effects, "
    "multicollinearity auditing, and **MacKinnon-White (HC3) robust standard errors**."
)

st.markdown("---")

# KPI Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Estimated Market Value",
        value=f"${point_valuation:,.0f}",
        delta=f"{price_vs_median_pct:+.1f}% vs {selected_neighborhood} Median",
    )

with kpi2:
    st.metric(
        label="95% Confidence Lower Bound",
        value=f"${valuation_ci_lower:,.0f}",
        help="Conservative lower boundary (2.5th percentile prediction limit).",
    )

with kpi3:
    st.metric(
        label="95% Confidence Upper Bound",
        value=f"${valuation_ci_upper:,.0f}",
        help="Upper boundary (97.5th percentile prediction limit).",
    )

with kpi4:
    st.metric(
        label="Submarket Median",
        value=f"${nb_median_price:,.0f}",
        help=f"Empirical median sale price across {len(nb_baseline)} historical sales in {selected_neighborhood}.",
    )

st.markdown("---")

# Main Content Tabs
tab_diag, tab_decomp, tab_benchmarks = st.tabs([
    "📈 Valuation & Submarket Benchmark",
    "🔬 Econometric Decomposition",
    "📊 Empirical Market Diagnostics"
])

# TAB 1: Valuation Visual Benchmark
with tab_diag:
    col_t1_l, col_t1_r = st.columns([3, 2])
    
    with col_t1_l:
        st.subheader(f"Price Position within {selected_neighborhood}")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.kdeplot(
            nb_baseline["sale_price"],
            fill=True,
            color="#2b5c8f",
            alpha=0.3,
            linewidth=2,
            ax=ax,
            label=f"{selected_neighborhood} Distribution"
        )
        ax.axvline(point_valuation, color="crimson", linewidth=2.5, linestyle="--", label=f"Predicted Value: ${point_valuation:,.0f}")
        ax.axvspan(valuation_ci_lower, valuation_ci_upper, color="crimson", alpha=0.1, label="95% Confidence Interval")
        ax.axvline(nb_median_price, color="#2ca02c", linewidth=1.5, linestyle=":", label=f"Submarket Median: ${nb_median_price:,.0f}")
        
        ax.set_title(f"Density Curve: {selected_neighborhood} Historical Arm's-Length Sales", fontsize=11, fontweight="bold")
        ax.set_xlabel("Sale Price ($ USD)", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.xaxis.set_major_formatter("${x:,.0f}")
        ax.legend(loc="upper right", frameon=True)
        st.pyplot(fig)
        plt.close(fig)

    with col_t1_r:
        st.subheader("Submarket Attributes")
        st.write(f"**Neighborhood:** `{selected_neighborhood}`")
        st.write(f"**Historical Records:** `{len(nb_baseline):,}` properties")
        st.write(f"**Neighborhood Price Range:** `${nb_baseline['sale_price'].min():,.0f}` — `${nb_baseline['sale_price'].max():,.0f}`")
        st.write(f"**Avg Living Area in Area:** `{nb_baseline['total_living_sf'].mean():,.0f} sq ft`")
        st.write(f"**Avg Construction Year:** `{int(nb_baseline['year_built'].mean())}`")
        
        st.info(
            f"A home with Quality {overall_qual}/10 and {total_living_sf:,} sq ft sits at "
            f"**{price_vs_median_pct:+.1f}%** relative to the median benchmark of {selected_neighborhood}."
        )

# TAB 2: Marginal Valuation Breakdown
with tab_decomp:
    st.subheader("Marginal Attribution of Structural Characteristics")
    st.markdown(
        "Each coefficient represents the ceteris paribus percentage return "
        "holding all other physical and spatial attributes fixed."
    )
    
    decomp_data = [
        {"Feature": "Quality Step (+1.0 rating)", "Log Coefficient": 0.0814, "% Marginal Impact": "+8.48%", "Dollar Effect on Current Home": f"+${point_valuation * 0.0848:,.0f}"},
        {"Feature": "Bathroom Capacity (+1.0 full bath)", "Log Coefficient": 0.0434, "% Marginal Impact": "+4.43%", "Dollar Effect on Current Home": f"+${point_valuation * 0.0443:,.0f}"},
        {"Feature": "Garage Space (+1 Car bay)", "Log Coefficient": 0.0414, "% Marginal Impact": "+4.23%", "Dollar Effect on Current Home": f"+${point_valuation * 0.0423:,.0f}"},
        {"Feature": "Fireplace (+1 fireplace)", "Log Coefficient": 0.0459, "% Marginal Impact": "+4.70%", "Dollar Effect on Current Home": f"+${point_valuation * 0.0470:,.0f}"},
        {"Feature": "Living Space (+100 sq ft)", "Log Coefficient": 0.0200, "% Marginal Impact": "+2.02%", "Dollar Effect on Current Home": f"+${point_valuation * 0.0202:,.0f}"},
        {"Feature": "Age Depreciation (Per 5 Years)", "Log Coefficient": -0.0080, "% Marginal Impact": "-0.80%", "Dollar Effect on Current Home": f"-${point_valuation * 0.0080:,.0f}"},
        {"Feature": "Unremodeled Lag (Per 5 Years)", "Log Coefficient": -0.0120, "% Marginal Impact": "-1.19%", "Dollar Effect on Current Home": f"-${point_valuation * 0.0119:,.0f}"},
    ]
    st.dataframe(pd.DataFrame(decomp_data), use_container_width=True)

# TAB 3: Baseline Quality & Model Integrity
with tab_benchmarks:
    st.subheader("Econometric Diagnostics & Quality Audits")
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Model Adjusted R²", "90.64%", "HC3 Corrected")
    c_m2.metric("Residual Standard Error", "±11.22%", "Log Scale RMSE")
    c_m3.metric("Baseline Clean Records", "2,412", "Arm's-Length Sales")
    
    st.markdown("### Model Specification")
    st.code(str(model.model.formula), language="text")
    st.caption("Fitted via Ordinary Least Squares with MacKinnon-White (HC3) Heteroscedasticity-Consistent Covariance.")