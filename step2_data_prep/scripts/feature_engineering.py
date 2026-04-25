import pandas as pd
import numpy as np

# 1. SETUP & LOAD DATA
input_filename = '../outputs/india_aqi_training_ready1.csv'
output_filename = '../../step3_data_preprocessing/processed/featureengineering.xlsx'
model_output_filename = '../../step3_data_preprocessing/processed/featureengineering_model_ready.xlsx'

print("Loading data...")
df = pd.read_csv(input_filename)

# 2. DROP NOISY / UNNECESSARY FEATURES
print("Removing features with ~0 correlation to target...")
# longitude and Ozone have near-zero correlation with Pm2.5. 
# We also drop the old basic encodings to do better ones.
cols_to_drop = ['Ozone', 'longitude', 'state_encoded', 'city_encoded']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])


# 3. DATA-DRIVEN COMBINATIONS
print("Combining correlated features...")
# No2 and Nh3 share a strong correlation (0.44) with each other and both drive Pm2.5 up.
# We create a 'Nitrogen_Pollutant_Index'.
if 'No2' in df.columns and 'Nh3' in df.columns:
    df['Nitrogen_Pollutant_Index'] = df['No2'] + df['Nh3']

# 4. MEANINGFUL INTERACTION FEATURES
print("Creating interaction features...")
# Because Pm10 and Pm2.5 are highly correlated (0.78), the ratio between them is highly predictive.
if 'Pm2.5' in df.columns and 'Pm10' in df.columns:
    df['Pm25_to_Pm10_Ratio'] = df['Pm2.5'] / (df['Pm10'] + 1e-5)

# 5. MOVING AVERAGES (Trend Features)
print("Calculating moving averages per station...")
# Grouping by station to simulate recent local history (assumes row order represents sequence)
pollutants_for_rolling = ['Pm2.5', 'Pm10', 'No2', 'Co']
for col in pollutants_for_rolling:
    if col in df.columns:
        df[f'{col}_Rolling_Mean_3'] = df.groupby('station')[col].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# 6. BETTER CATEGORICAL FEATURES (Target Encoding & Frequency)
print("Processing categorical features...")
# Since 'latitude' correlated strongly with pollution, we encode 'state' by its median Pm2.5 
# (Target encoding is much better for EDA & ML than random numeric assignment)
if 'Pm2.5' in df.columns and 'state' in df.columns:
    state_target_mean = df.groupby('state')['Pm2.5'].median()
    df['state_pollution_encoded'] = df['state'].map(state_target_mean)

# For 'city', we use frequency encoding to represent city data density
city_freq = df['city'].value_counts() / len(df)
df['city_freq_encoded'] = df['city'].map(city_freq)


# 7. AQI CALCULATION FEATURE
print("Calculating standard AQI categories...")
# Standard Indian PM2.5 AQI buckets
def calculate_pm25_aqi_category(pm25):
    if pd.isna(pm25): return 'Unknown'
    elif pm25 <= 30: return 'Good'
    elif pm25 <= 60: return 'Satisfactory'
    elif pm25 <= 90: return 'Moderate'
    elif pm25 <= 120: return 'Poor'
    elif pm25 <= 250: return 'Very Poor'
    else: return 'Severe'

if 'Pm2.5' in df.columns:
    df['AQI_Category'] = df['Pm2.5'].apply(calculate_pm25_aqi_category)
    aqi_mapping = {'Good': 1, 'Satisfactory': 2, 'Moderate': 3, 'Poor': 4, 'Very Poor': 5, 'Severe': 6, 'Unknown': -1}
    df['AQI_Severity_Score'] = df['AQI_Category'].map(aqi_mapping)


# 8. FINAL CLEANUP FOR EDA
print("Performing final feature selection...")
# Drop categorical strings that are too granular to plot well in correlation heatmaps during EDA
cols_to_drop_final = ['station']
df = df.drop(columns=[col for col in cols_to_drop_final if col in df.columns])

# Fill any NaNs created by rolling formulas with the column medians
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 9. SAVE OUTPUT
df.to_excel(output_filename, index=False)
print(f"Saving engineered data to {output_filename}...")
df.to_excel(output_filename, index=False)

# Save a model-ready version with target-derived columns removed
target_leakage_cols = ['AQI_Category', 'AQI_Severity_Score', 'Pm25_to_Pm10_Ratio', 'Pm2.5_Rolling_Mean_3']
model_df = df.drop(columns=[col for col in target_leakage_cols if col in df.columns])
print(f"Saving model-ready engineered data to {model_output_filename}...")
model_df.to_excel(model_output_filename, index=False)

print("✅ Feature engineering based on correlation analysis is complete!")
