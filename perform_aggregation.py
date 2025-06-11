import pandas as pd
import numpy as np
import os
import re

def aggregate_sensor_data(input_csv='combined_activity_data.csv', 
                          output_csv='aggregated_features_0.01s_window.csv', 
                          window_size_seconds=0.01):
    """
    Aggregates sensor data using tumbling windows.
    Averages most sensors, counts occurrences for p roximity states 0 and 5.
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
        return

    print(f"Loaded data from '{input_csv}'. Shape: {df.shape}")

    # Ensure 'time' column is numeric
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True) # Remove rows where time is not valid

    # Sort data: crucial for windowing
    df.sort_values(by=['activity_type', 'time'], inplace=True)
    df.reset_index(drop=True, inplace=True) # Reset index after sorting

    # --- Identify columns ---
    proximity_value_col = None
    for col in df.columns:
        if col.lower().startswith('proximity_') and df[col].dtype in [np.float64, np.int64]:
            proximity_value_col = col
            break # Assume first one found is the primary proximity value

    if not proximity_value_col:
        print("Warning: No proximity value column found (e.g., 'Proximity_Distance (cm)'). Proximity counts will be zero.")
    else:
        print(f"Using '{proximity_value_col}' for proximity state counting.")

    # Identify other numeric columns to average (excluding 'time' and proximity if found)
    columns_to_average = []
    excluded_cols_for_avg = ['time', 'activity_type']
    if proximity_value_col:
        excluded_cols_for_avg.append(proximity_value_col)

    for col in df.columns:
        if col not in excluded_cols_for_avg and df[col].dtype in [np.float64, np.int64]:
            columns_to_average.append(col)

    print(f"Columns to average: {columns_to_average if columns_to_average else 'None'}")

    all_aggregated_windows = []
    activities = df['activity_type'].unique()

    for activity in activities:
        print(f"Processing activity: {activity}...")
        activity_df = df[df['activity_type'] == activity]
        
        if activity_df.empty:
            continue

        min_time_activity = activity_df['time'].min()
        max_time_activity = activity_df['time'].max()

        current_window_start = min_time_activity
        window_number = 0

        while current_window_start < max_time_activity:
            current_window_end = current_window_start + window_size_seconds
            
            # Select data within the current window [start, end)
            window_data = activity_df[
                (activity_df['time'] >= current_window_start) & 
                (activity_df['time'] < current_window_end)
            ]

            # Initialize aggregated features for this window
            aggregated_features = {'window_start_time': current_window_start, 'activity_type': activity}

            # Proximity aggregation
            count_0 = 0
            count_5 = 0
            if not window_data.empty and proximity_value_col:
                proximity_values_in_window = window_data[proximity_value_col].dropna()
                count_0 = proximity_values_in_window.apply(lambda x: np.isclose(x, 0.0)).sum()
                count_5 = proximity_values_in_window.apply(lambda x: np.isclose(x, 5.0)).sum()
            
            aggregated_features['Proximity_0_count'] = count_0
            aggregated_features['Proximity_5_count'] = count_5

            # Averaging other numeric columns
            if not window_data.empty:
                for col_to_avg in columns_to_average:
                    aggregated_features[col_to_avg] = window_data[col_to_avg].mean() # .mean() handles NaNs correctly
            else: # If window_data is empty, all averages will be NaN
                for col_to_avg in columns_to_average:
                    aggregated_features[col_to_avg] = np.nan
            
            all_aggregated_windows.append(aggregated_features)
            
            current_window_start = current_window_end # Move to the next tumbling window
            window_number += 1
            if window_number % 10000 == 0: # Progress update for long activities
                 print(f"  ..processed {window_number} windows for {activity}")


    if not all_aggregated_windows:
        print("No data was aggregated. Output file will not be created.")
        return

    # Create final aggregated DataFrame
    aggregated_df = pd.DataFrame(all_aggregated_windows)
    
    # Reorder columns to have time and activity first, then proximity counts, then others
    cols_order = ['window_start_time', 'activity_type', 'Proximity_0_count', 'Proximity_5_count']
    remaining_cols = [col for col in aggregated_df.columns if col not in cols_order]
    final_cols_order = cols_order + sorted(remaining_cols) # Sort remaining for consistency
    aggregated_df = aggregated_df[final_cols_order]

    print(f"\nAggregation complete. Shape of aggregated DataFrame: {aggregated_df.shape}")
    print("Head of aggregated DataFrame:")
    print(aggregated_df.head())

    # Save to CSV
    try:
        aggregated_df.to_csv(output_csv, index=False)
        print(f"\nAggregated data saved to '{output_csv}'")
    except Exception as e:
        print(f"\nError saving aggregated data to CSV: {e}")

if __name__ == '__main__':
    # You can change the input/output filenames and window size here if needed
    aggregate_sensor_data(input_csv='trimmed_data.csv', 
                          output_csv='aggregated_data.csv', 
                          window_size_seconds=0.01)
    print("\nScript finished.")
