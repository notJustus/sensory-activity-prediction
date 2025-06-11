import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# === Load the Aggregated Dataset ===
input_path = "aggregated_data.csv"
output_path = "aggregated_data_cleaned.csv"
df = pd.read_csv(input_path)

# === Define Feature Columns ===
non_feature_cols = {'elapsed_sec', 'attribute', 'time', 'activity type'}
feature_cols = [col for col in df.columns if col not in non_feature_cols and pd.api.types.is_numeric_dtype(df[col])]

# === Apply LOF for Outlier Detection ===
for col in feature_cols:
    col_data = df[col].dropna().values.reshape(-1, 1)

    if len(col_data) < 10:
        continue  # Not enough data

    try:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
        y_pred = lof.fit_predict(col_data)

        # Mark LOF outliers (label -1) as NaN
        mask = y_pred == -1
        df.loc[df[col].dropna().index[mask], col] = np.nan
    except ValueError:
        continue  # Skip columns with issues

# === Print missing values before imputation ===
print(f"🟡 Missing values after outlier removal (before imputation): {df.isnull().sum().sum()}")

# === Impute Missing Values with Linear Interpolation ===
df.interpolate(method='linear', limit_direction='both', inplace=True)

# === Save Cleaned Data ===
df.to_csv(output_path, index=False)
print(f"✅ Cleaned data saved to: {output_path}")
