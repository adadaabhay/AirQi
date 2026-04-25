import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# -------------------------------------------------
# 1️⃣ LOAD DATA
# -------------------------------------------------
df = pd.read_csv('india_aqi_21k.csv')

print("Columns in dataset:")
print(df.columns.tolist())

# -------------------------------------------------
# 2️⃣ BASIC CLEANING
# -------------------------------------------------
df = df.drop_duplicates().reset_index(drop=True)

text_cols = ["country", "state", "city", "station", "pollutant_id"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# -------------------------------------------------
# 3️⃣ FIND DATE COLUMN AUTOMATICALLY
# -------------------------------------------------
possible_date_cols = [
    'date', 'date_time', 'datetime', 'from_date',
    'to_date', 'sampling_date', 'last_update'
]

date_col = None
for col in possible_date_cols:
    if col in df.columns:
        date_col = col
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    print(f"Using date column: {date_col}")
else:
    print("⚠️ No date column found. Pivot will be station-level only.")

# -------------------------------------------------
# 4️⃣ PIVOT (DATE-AWARE IF POSSIBLE)
# -------------------------------------------------
pivot_index = ['state', 'city', 'station', 'latitude', 'longitude']
if date_col:
    pivot_index = [date_col] + pivot_index

df_pivot = df.pivot_table(
    index=pivot_index,
    columns='pollutant_id',
    values='avg_value'
).reset_index()

# -------------------------------------------------
# 5️⃣ DEFINE TARGET & FEATURES
# -------------------------------------------------
target_col = 'Pm2.5'

feature_cols = [
    c for c in df_pivot.columns
    if c not in pivot_index + [target_col]
]

# -------------------------------------------------
# 6️⃣ HANDLE MISSING VALUES
# -------------------------------------------------
imputer = SimpleImputer(strategy='median')
all_cols = feature_cols + [target_col]
df_pivot[all_cols] = imputer.fit_transform(df_pivot[all_cols])

# -------------------------------------------------
# 7️⃣ OUTLIER HANDLING (WINSORIZATION)
# -------------------------------------------------
for col in all_cols:
    lower = df_pivot[col].quantile(0.05)
    upper = df_pivot[col].quantile(0.95)
    df_pivot[col] = np.clip(df_pivot[col], lower, upper)

# -------------------------------------------------
# 8️⃣ FEATURE SCALING
# -------------------------------------------------
df_pivot[feature_cols] = np.log1p(df_pivot[feature_cols])
scaler = MinMaxScaler()
df_pivot[feature_cols] = scaler.fit_transform(df_pivot[feature_cols])

# -------------------------------------------------
# 9️⃣ ENCODING
# -------------------------------------------------
le_state = LabelEncoder()
le_city = LabelEncoder()

df_pivot['state_encoded'] = le_state.fit_transform(df_pivot['state'])
df_pivot['city_encoded'] = le_city.fit_transform(df_pivot['city'])

# -------------------------------------------------
# 🔟 FINAL CHECK & SAVE
# -------------------------------------------------
print("\nRemaining Missing Values:")
print(df_pivot.isna().sum())

print("\nTotal Records:", len(df_pivot))

df_pivot.to_csv('india_aqi_training_ready.csv', index=False)

print("\n✅ DATA PREPARATION COMPLETED SUCCESSFULLY")
