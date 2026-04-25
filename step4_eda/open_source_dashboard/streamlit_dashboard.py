from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AQI EDA Dashboard", layout="wide")
st.title("AQI Analytics Dashboard (Open-Source: Streamlit + Plotly)")
st.caption("Separate visualization deliverable for grading")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "step3_data_preprocessing" / "processed" / "featureengineering.xlsx"

if not DATA_PATH.exists():
    st.error(f"Data not found: {DATA_PATH}")
    st.stop()

df = pd.read_excel(DATA_PATH)

states = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else []
selected_states = st.sidebar.multiselect("Filter states", states, default=states[: min(8, len(states))])
if selected_states and "state" in df.columns:
    dff = df[df["state"].isin(selected_states)].copy()
else:
    dff = df.copy()

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(dff):,}")
col2.metric("Mean PM2.5", f"{dff['Pm2.5'].mean():.2f}")
col3.metric("Median PM2.5", f"{dff['Pm2.5'].median():.2f}")

st.subheader("PM2.5 Distribution")
fig1 = px.histogram(dff, x="Pm2.5", nbins=40, marginal="box", color_discrete_sequence=["#1f77b4"])
st.plotly_chart(fig1, use_container_width=True)

if all(c in dff.columns for c in ["Pm10", "Pm2.5"]):
    st.subheader("PM10 vs PM2.5")
    fig2 = px.scatter(dff, x="Pm10", y="Pm2.5", color="state" if "state" in dff.columns else None, opacity=0.7)
    st.plotly_chart(fig2, use_container_width=True)

if "AQI_Category" in dff.columns:
    st.subheader("AQI Category Share")
    cat = dff["AQI_Category"].value_counts().reset_index()
    cat.columns = ["AQI_Category", "count"]
    fig3 = px.pie(cat, names="AQI_Category", values="count", hole=0.45)
    st.plotly_chart(fig3, use_container_width=True)

if all(c in dff.columns for c in ["state", "Pm2.5"]):
    st.subheader("Top States by PM2.5")
    top_states = dff.groupby("state", as_index=False)["Pm2.5"].mean().sort_values("Pm2.5", ascending=False).head(15)
    fig4 = px.bar(top_states, x="Pm2.5", y="state", orientation="h", color="Pm2.5", color_continuous_scale="oranges")
    st.plotly_chart(fig4, use_container_width=True)
