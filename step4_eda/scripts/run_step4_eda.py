from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="talk")

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent
INPUT_XLSX = PROJECT_ROOT / "step3_data_preprocessing" / "processed" / "featureengineering.xlsx"
STATS_DIR = BASE_DIR / "csv"
PLOTS_DIR = BASE_DIR / "plots"

for p in [STATS_DIR, PLOTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

for sub in [
    "01_distribution",
    "02_correlation",
    "03_feature_relationships",
    "04_outliers",
    "05_category_and_geospatial",
]:
    (PLOTS_DIR / sub).mkdir(parents=True, exist_ok=True)

if not INPUT_XLSX.exists():
    raise FileNotFoundError(f"Missing input data: {INPUT_XLSX}")

df = pd.read_excel(INPUT_XLSX)
target = "Pm2.5"

if target not in df.columns:
    raise ValueError("Target column 'Pm2.5' not found in dataset.")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
summary = df[num_cols].describe().T
summary["median"] = df[num_cols].median()
summary["missing_count"] = df[num_cols].isna().sum()
summary = summary[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "median", "missing_count"]]
summary.to_csv(STATS_DIR / "summary_statistics.csv")

corr = df[num_cols].corr(numeric_only=True)
corr.to_csv(STATS_DIR / "correlation_matrix.csv")

target_corr = corr[target].sort_values(ascending=False).rename("correlation").reset_index()
target_corr.to_csv(STATS_DIR / "correlation_with_target.csv", index=False)

# 1) Distribution
plt.figure(figsize=(11, 7))
sns.histplot(df[target], kde=True, bins=30, color="#1f77b4")
plt.title("PM2.5 Distribution")
plt.xlabel("PM2.5")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_distribution" / "pm25_distribution.png", dpi=220)
plt.close()

if "AQI_Category" in df.columns:
    plt.figure(figsize=(10, 6))
    order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    ax = sns.countplot(data=df, x="AQI_Category", order=[x for x in order if x in df["AQI_Category"].unique()], palette="viridis")
    ax.set_title("AQI Category Distribution")
    ax.set_xlabel("AQI Category")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_distribution" / "aqi_category_distribution.png", dpi=220)
    plt.close()

# 2) Correlation
heat_cols = [c for c in [target, "Pm10", "No2", "Nh3", "So2", "Nitrogen_Pollutant_Index", "Pm25_to_Pm10_Ratio", "AQI_Severity_Score", "latitude"] if c in df.columns]
if len(heat_cols) >= 3:
    plt.figure(figsize=(12, 9))
    sns.heatmap(df[heat_cols].corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Focused Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_correlation" / "focused_correlation_heatmap.png", dpi=220)
    plt.close()

# 3) Feature relationships
if "Nitrogen_Pollutant_Index" in df.columns:
    plt.figure(figsize=(10, 7))
    sns.regplot(data=df, x="Nitrogen_Pollutant_Index", y=target, scatter_kws={"alpha": 0.45}, line_kws={"color": "red"})
    plt.title("PM2.5 vs Nitrogen Pollutant Index")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_feature_relationships" / "pm25_vs_nitrogen_index.png", dpi=220)
    plt.close()

if "Pm10" in df.columns:
    plt.figure(figsize=(10, 7))
    sns.regplot(data=df, x="Pm10", y=target, scatter_kws={"alpha": 0.45}, line_kws={"color": "darkgreen"})
    plt.title("PM2.5 vs PM10")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_feature_relationships" / "pm25_vs_pm10.png", dpi=220)
    plt.close()

# 4) Outliers
outlier_cols = [c for c in ["Pm2.5", "Pm10", "No2", "Nh3", "So2"] if c in df.columns]
if outlier_cols:
    melted = df[outlier_cols].melt(var_name="feature", value_name="value")
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=melted, x="feature", y="value", palette="Set2")
    plt.title("Outlier Overview for Core Pollutants")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_outliers" / "core_pollutant_outliers.png", dpi=220)
    plt.close()

# 5) Category + geospatial
if "state" in df.columns and "Pm10" in df.columns:
    state_avg = df.groupby("state", as_index=False)[["Pm2.5", "Pm10"]].mean().sort_values("Pm10", ascending=False).head(12)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=state_avg, y="state", x="Pm10", color="lightgray", label="PM10")
    sns.barplot(data=state_avg, y="state", x="Pm2.5", color="firebrick", label="PM2.5")
    plt.title("Top 12 States: PM10 vs PM2.5")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_category_and_geospatial" / "state_pm_profile_top12.png", dpi=220)
    plt.close()

if "latitude" in df.columns:
    hue_col = "AQI_Category" if "AQI_Category" in df.columns else None
    plt.figure(figsize=(11, 7))
    sns.scatterplot(data=df, x="latitude", y=target, hue=hue_col, alpha=0.6)
    sns.regplot(data=df, x="latitude", y=target, scatter=False, color="black", line_kws={"linestyle": "--"})
    plt.title("Geospatial Trend: PM2.5 vs Latitude")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_category_and_geospatial" / "pm25_vs_latitude.png", dpi=220)
    plt.close()

print("Step 4 EDA artifacts generated successfully.")
print(f"Stats: {STATS_DIR}")
print(f"Plots: {PLOTS_DIR}")
