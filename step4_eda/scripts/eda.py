import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. SETUP & FOLDER CREATION
# ==========================================
filename = 'featureengineering.xlsx'
viz_folder = 'eda_visualizations'
insights_folder = 'eda_insights'

for folder in [viz_folder, insights_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

print(f"Loading {filename}...")
try:
    df = pd.read_excel(filename)
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

sns.set_theme(style="whitegrid", context="talk")
target_col = 'Pm2.5'

expected_features = [
    'Co', 'Nh3', 'No2', 'Pm10', 'So2', 
    'Nitrogen_Pollutant_Index', 'Pm25_to_Pm10_Ratio', 
    'Pm2.5_Rolling_Mean_3', 'AQI_Severity_Score', 'latitude'
]
feature_cols = [c for c in expected_features if c in df.columns]

print("⏳ Generating plots and sorting into folders...")

# ==========================================
# PART A: DATA VISUALIZATIONS (Standard EDA)
# ==========================================
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Target Distribution
plt.figure()
sns.histplot(df[target_col].dropna(), kde=True, color='crimson')
plt.title('Distribution of PM2.5')
plt.tight_layout()
plt.savefig(f'{viz_folder}/1_PM25_Distribution.png')
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df[[target_col] + feature_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Complete Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'{viz_folder}/2_Correlation_Heatmap.png')
plt.close()

# 3. Feature Importance
plt.figure()
target_corr = df[feature_cols].corrwith(df[target_col]).sort_values(ascending=False)
sns.barplot(x=target_corr.values, y=target_corr.index, palette='viridis')
plt.title('Feature Correlation with PM2.5')
plt.tight_layout()
plt.savefig(f'{viz_folder}/3_Feature_Importance.png')
plt.close()

# 4. Outlier Boxplots
plt.figure(figsize=(14, 8))
sns.boxplot(data=df[feature_cols], palette='Set2')
plt.xticks(rotation=45, ha='right')
plt.title('Outlier Check Across All Features')
plt.tight_layout()
plt.savefig(f'{viz_folder}/4_Outlier_Boxplots.png')
plt.close()

# 5. Base Pollutant Density
base_pollutants = [c for c in ['Co', 'Nh3', 'No2', 'Pm10', 'So2'] if c in df.columns]
if base_pollutants:
    plt.figure()
    sns.violinplot(x='variable', y='value', data=df[base_pollutants].melt(), palette='muted')
    plt.title('Distribution Density of Base Pollutants')
    plt.tight_layout()
    plt.savefig(f'{viz_folder}/5_Violin_Plots.png')
    plt.close()

# ==========================================
# PART B: INSIGHTS & TRENDS (Deep Dives)
# ==========================================
plt.rcParams['figure.figsize'] = (14, 9)

# 1. AQI Category Donut Chart
if 'AQI_Category' in df.columns:
    plt.figure(figsize=(10, 10))
    cat_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    counts = df['AQI_Category'].value_counts().reindex(cat_order).dropna()
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#34495e']
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    plt.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))
    plt.title('Distribution of AQI Severity Categories', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{insights_folder}/1_AQI_Donut_Chart.png')
    plt.close()

# 2. Geospatial North-South Trend
if 'latitude' in df.columns and 'AQI_Category' in df.columns:
    plt.figure()
    sns.scatterplot(data=df, x='latitude', y='Pm2.5', hue='AQI_Category', palette='rocket_r', alpha=0.8, edgecolor='k')
    sns.regplot(data=df, x='latitude', y='Pm2.5', scatter=False, color='red', line_kws={"linewidth":2, "linestyle":"--"})
    plt.title('Geospatial Pattern: PM2.5 Rises at Higher Latitudes (North)', fontsize=16, fontweight='bold')
    plt.xlabel('Latitude (Moving North ->)')
    plt.tight_layout()
    plt.savefig(f'{insights_folder}/2_Geospatial_Trend.png')
    plt.close()

# 3. Nitrogen Interaction Trend
if 'Nitrogen_Pollutant_Index' in df.columns:
    plt.figure()
    sns.histplot(data=df, x='Nitrogen_Pollutant_Index', y='Pm2.5', bins=30, pmax=.8, cmap="mako", cbar=True)
    sns.regplot(data=df, x='Nitrogen_Pollutant_Index', y='Pm2.5', scatter=False, color='darkorange', line_kws={"linewidth": 3})
    plt.title('Interaction: Nitrogen Pollutants (NO2+NH3) Driving PM2.5', fontsize=16, fontweight='bold')
    plt.xlabel('Nitrogen Pollutant Index')
    plt.tight_layout()
    plt.savefig(f'{insights_folder}/3_Nitrogen_Driver.png')
    plt.close()

# 4. State PM Profile (Structural Pattern)
if 'state' in df.columns and 'Pm10' in df.columns:
    plt.figure(figsize=(14, 10))
    state_avg = df.groupby('state')[['Pm2.5', 'Pm10']].mean().sort_values('Pm10', ascending=True)
    plt.barh(state_avg.index, state_avg['Pm10'], color='lightgrey', label='Total PM10 (Coarse + Fine)')
    plt.barh(state_avg.index, state_avg['Pm2.5'], color='firebrick', label='Toxic PM2.5 (Fine Only)', alpha=0.9)
    plt.title('Structural Pattern: Composition of Particulate Matter by State', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{insights_folder}/4_State_PM_Profile.png')
    plt.close()

# 5. Local Temporal Trend (Rolling Average)
if 'Pm2.5_Rolling_Mean_3' in df.columns:
    plt.figure()
    trend_df = df[['Pm2.5', 'Pm2.5_Rolling_Mean_3']].dropna().sort_values('Pm2.5_Rolling_Mean_3').reset_index(drop=True)
    trend_df = trend_df.iloc[::max(1, len(trend_df)//150)]
    plt.plot(trend_df.index, trend_df['Pm2.5'], 'o', color='silver', alpha=0.6, label='Raw Observations')
    plt.plot(trend_df.index, trend_df['Pm2.5_Rolling_Mean_3'], '-', color='navy', linewidth=3, label='3-Period Local Trend')
    plt.title('Temporal Trend: Raw Observations vs Smoothed Trend', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{insights_folder}/5_Smoothed_Trend.png')
    plt.close()

print("✅ Success! All standard plots are in 'eda_visualizations' and all analytical plots are in 'eda_insights'.")