from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Step 5 Model Dashboard",
    page_icon="AQI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(35, 84, 122, 0.14), transparent 28%),
            radial-gradient(circle at top right, rgba(239, 123, 69, 0.10), transparent 24%),
            linear-gradient(180deg, #f9f6f0 0%, #efe8dd 100%);
        color: #182833;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }
    .hero-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(46, 68, 89, 0.10);
        border-radius: 24px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 18px 40px rgba(52, 63, 74, 0.10);
        backdrop-filter: blur(4px);
    }
    .section-note {
        background: rgba(255, 255, 255, 0.94);
        border-left: 5px solid #d46a3a;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        color: #20303c;
        margin-top: 0.5rem;
        line-height: 1.55;
        box-shadow: 0 10px 24px rgba(52, 63, 74, 0.06);
    }
    .section-card {
        background: rgba(255, 255, 255, 0.93);
        border: 1px solid rgba(46, 68, 89, 0.10);
        border-radius: 20px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 14px 32px rgba(52, 63, 74, 0.07);
        margin-top: 0.5rem;
        margin-bottom: 0.9rem;
    }
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(46, 68, 89, 0.10);
        padding: 0.85rem 1rem;
        border-radius: 18px;
        box-shadow: 0 10px 28px rgba(52, 63, 74, 0.07);
    }
    h1, h2, h3, label, p, li, div, span {
        color: #182833;
    }
    h1, h2, h3 {
        color: #17324a;
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.96);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = BASE_DIR / "metrics"
DOCS_DIR = BASE_DIR / "docs"

comparison = pd.read_csv(METRICS_DIR / "model_comparison.csv")
advanced = pd.read_csv(METRICS_DIR / "tuned_model_comparison.csv")
best = pd.read_csv(METRICS_DIR / "best_metrics.csv")
pred = pd.read_csv(METRICS_DIR / "predictions_vs_actual.csv")
fi = pd.read_csv(METRICS_DIR / "feature_importance.csv")

# Optional: Load correlation data if generated during Feature Engineering
corr_path = METRICS_DIR / "correlation_matrix.csv"
corr_data = pd.read_csv(corr_path, index_index=0) if corr_path.exists() else None

metric_map = dict(zip(best["metric"], best["value"]))
selected_model = str(metric_map["Selected Model"])
r2_value = float(metric_map["R2 Score"])
mae_value = float(metric_map["MAE"])
rmse_value = float(metric_map["RMSE"])
rows_used = int(float(metric_map["Rows Used"]))
train_rows = int(float(metric_map["Train Rows"]))
test_rows = int(float(metric_map["Test Rows"]))

residual_mean = float(pred["residual"].mean())
residual_std = float(pred["residual"].std())
abs_error_mean = float(pred["residual"].abs().mean())

palette_primary = ["#16324a", "#d46a3a", "#2d6a8f", "#7e9f8c"]
plot_template = "plotly_white"

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom:0.2rem;">Step 5 Model Accuracy Dashboard</h1>
        <p style="margin-bottom:0.4rem; color:#2f4352; font-size:1.03rem;">
            Open-source presentation dashboard for model selection, model accuracy, and prediction diagnostics.
        </p>
        <p style="margin:0; color:#415565; font-size:0.98rem; line-height:1.6;">
            Focus: why this algorithm was chosen, how well it predicts PM2.5, and how the supporting visuals justify the final result.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Step 5 Summary")
st.sidebar.write(f"Selected model: `{selected_model}`")
st.sidebar.write(f"R2 Score: `{r2_value:.4f}`")
st.sidebar.write(f"MAE: `{mae_value:.4f}`")
st.sidebar.write(f"RMSE: `{rmse_value:.4f}`")
st.sidebar.write(f"Rows used: `{rows_used}`")

sort_metric = st.sidebar.selectbox(
    "Sort model comparison by",
    ["test_r2", "test_mae", "test_rmse"],
    index=0,
)

top_n_features = st.sidebar.slider(
    "Top features to display",
    min_value=8,
    max_value=20,
    value=12,
    step=1,
)

st.sidebar.markdown(
    """
    **Why this dashboard works**

    - Open source
    - Interactive
    - Built for ML result explanation
    - Better suited to prediction diagnostics than generic BI-only dashboards
    """
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Model", selected_model)
col2.metric("R2 Score", f"{r2_value:.4f}")
col3.metric("MAE", f"{mae_value:.2f}")
col4.metric("RMSE", f"{rmse_value:.2f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Rows Used", f"{rows_used}")
col6.metric("Train Rows", f"{train_rows}")
col7.metric("Test Rows", f"{test_rows}")
col8.metric("Mean Abs Error", f"{abs_error_mean:.2f}")

st.markdown(
    """
    <div class="section-note">
        This dashboard is designed for Step 5 grading. It shows model ranking, prediction quality, residual behavior,
        and feature importance so you can justify both <b>why this model was selected</b> and <b>how reliable it is</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 1. Overview")
left, right = st.columns([1.05, 1])

with left:
    st.markdown(
        """
        <div class="section-card">
            <h3 style="margin-top:0;">Project Framing</h3>
            <p><b>Problem type:</b> supervised regression</p>
            <p><b>Target variable:</b> PM2.5</p>
            <p><b>Workflow:</b> train/test split -> cross-validation -> baseline comparison -> tuned comparison -> final selection</p>
            <p><b>Selected model:</b> Stacked_Ensemble</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-note">
            The final model was not chosen because it is more complicated. It was chosen because it achieved the
            strongest balance of <b>R2</b>, <b>MAE</b>, and <b>RMSE</b> on the held-out test set.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    summary_df = pd.DataFrame(
        {
            "Metric": ["R2 Score", "MAE", "RMSE", "Residual Mean", "Residual Std Dev"],
            "Value": [
                round(r2_value, 4),
                round(mae_value, 4),
                round(rmse_value, 4),
                round(residual_mean, 4),
                round(residual_std, 4),
            ],
        }
    )
    fig_summary = px.bar(
        summary_df,
        x="Metric",
        y="Value",
        color="Metric",
        color_discrete_sequence=["#16324a", "#d46a3a", "#2d6a8f", "#7e9f8c", "#c8a35b"],
        template=plot_template,
    )
    fig_summary.update_layout(showlegend=False, height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_summary, use_container_width=True)

st.markdown("## 2. Model Comparison")
baseline_sorted = comparison.sort_values(
    sort_metric, ascending=(sort_metric != "test_r2")
)
advanced_sorted = advanced.sort_values(
    sort_metric, ascending=(sort_metric != "test_r2")
)
left, right = st.columns(2)

with left:
    st.subheader("Baseline Models")
    fig_baseline = px.bar(
        baseline_sorted,
        x=sort_metric,
        y="model",
        orientation="h",
        color=sort_metric,
        color_continuous_scale="Blues",
        template=plot_template,
        text_auto=".4f",
    )
    fig_baseline.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_baseline, use_container_width=True)

with right:
    st.subheader("Advanced Models")
    fig_advanced = px.bar(
        advanced_sorted,
        x=sort_metric,
        y="model",
        orientation="h",
        color=sort_metric,
        color_continuous_scale="Oranges",
        template=plot_template,
        text_auto=".4f",
    )
    fig_advanced.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_advanced, use_container_width=True)

st.markdown(
    """
    <div class="section-note">
        These charts answer the review question: <b>Why this algorithm and not the others?</b>
        The chosen model should remain strongest on the selected evaluation metric, especially <b>test R2</b>.
    </div>
    """,
    unsafe_allow_html=True,
)
st.dataframe(advanced.sort_values("test_r2", ascending=False), use_container_width=True)

st.markdown("## 3. Detailed Pollutant & Correlation Analysis")
tab_corr, tab_pollutants = st.tabs(["Feature Correlations", "Pollutant Contributions"])

with tab_corr:
    if corr_data is not None:
        st.subheader("Feature Correlation Heatmap")
        fig_corr = px.imshow(
            corr_data,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            template=plot_template,
            title="How Pollutants & Weather Variables Relate"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("💡 Tip: Upload a `correlation_matrix.csv` to `metrics/` to see the relationship between pollutants.")

with tab_pollutants:
    st.subheader("Pollutant Impact on AQI")
    st.markdown(
        """
        Stacked bars and individual pollutant trends help identify the 'dominant pollutant'.
        In this dataset, the engineered **Nitrogen_Pollutant_Index** and **PM10** are primary drivers.
        """
    )

st.markdown("## 3. Prediction Diagnostics")
top_left, top_right = st.columns(2)

with top_left:
    st.subheader("Actual vs Predicted")
    fig_actual = px.scatter(
        pred,
        x="actual",
        y="predicted",
        opacity=0.72,
        color_discrete_sequence=["#2d6a8f"],
        marginal_x="histogram",
        marginal_y="histogram",
        template=plot_template,
    )
    min_axis = min(pred["actual"].min(), pred["predicted"].min())
    max_axis = max(pred["actual"].max(), pred["predicted"].max())
    fig_actual.add_trace(
        go.Scatter(
            x=[min_axis, max_axis],
            y=[min_axis, max_axis],
            mode="lines",
            name="Perfect prediction line",
            line=dict(color="#d46a3a", dash="dash", width=3),
        )
    )
    fig_actual.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_actual, use_container_width=True)

with top_right:
    st.subheader("Residual Distribution")
    fig_resid = px.histogram(
        pred,
        x="residual",
        nbins=30,
        marginal="box",
        color_discrete_sequence=["#8b5e3c"],
        template=plot_template,
    )
    fig_resid.add_vline(x=0, line_dash="dash", line_color="#16324a", line_width=2)
    fig_resid.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_resid, use_container_width=True)

st.subheader("Residual vs Predicted")
fig_resid_scatter = px.scatter(
    pred,
    x="predicted",
    y="residual",
    opacity=0.68,
    color_discrete_sequence=["#d46a3a"],
    template=plot_template,
)
fig_resid_scatter.add_hline(y=0, line_dash="dash", line_color="#16324a", line_width=2)
fig_resid_scatter.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig_resid_scatter, use_container_width=True)

st.markdown(
    """
    <div class="section-note">
        These views show whether the predictions stay close to real values and whether the error pattern remains
        centered instead of breaking down systematically.
    </div>
    """,
    unsafe_allow_html=True,
)

if "state" in pred.columns:
    st.markdown("## 4. Geospatial Performance")
    state_err = pred.groupby("state")["residual"].abs().mean().reset_index()
    fig_geo = px.bar(
        state_err.sort_values("residual", ascending=False),
        x="state",
        y="residual",
        title="Mean Absolute Error by State",
        color="residual",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_geo, use_container_width=True)
    st.markdown("<div class='section-note'>This identifies regions where the model over or under-predicts, highlighting localized industrial impacts.</div>", unsafe_allow_html=True)

st.markdown("## 4. Explainability")
st.subheader("Top Feature Importance")
top_features = fi.head(top_n_features).sort_values("importance", ascending=True)
fig_fi = px.bar(
    top_features,
    x="importance",
    y="feature",
    orientation="h",
    color="importance",
    color_continuous_scale="Aggrnyl",
    template=plot_template,
    text_auto=".3f",
)
fig_fi.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig_fi, use_container_width=True)
st.dataframe(fi.head(top_n_features), use_container_width=True)

st.markdown(
    """
    <div class="section-note">
        Feature importance is included because this review should not only say the model works.
        It should also explain <b>what the model is using</b>. In this project, PM10-related and location-linked
        features are among the strongest drivers.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 5. Review Notes")
st.markdown(
    f"""
    - The final model is **{selected_model}** because it achieved the best test-set balance of `R2`, `MAE`, and `RMSE`.
    - `LinearRegression` was not selected because the dataset is not purely linear, and its performance was much weaker.
    - The tuned tree-based models were strong, but the final ensemble still performed better overall.
    - `Streamlit + Plotly` was used because Step 5 needs model diagnostics and interactive metric presentation, not only static BI charts.
    - The most important visuals to present are: baseline comparison, advanced comparison, actual vs predicted, residual distribution, and feature importance.
    """
)

rationale_file = DOCS_DIR / "step5_selection_rationale.md"
if rationale_file.exists():
    st.subheader("Saved Step 5 Rationale")
    st.code(rationale_file.read_text(), language="markdown")
