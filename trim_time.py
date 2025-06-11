import pandas as pd
import os

def clean_negative_time_values(input_csv='combined_activity_data.csv', 
                               output_csv='combined_activity_data_non_negative_time.csv'):
    """
    Loads a CSV, removes rows with negative time values, and saves to a new CSV.
    """
    try:
        df = pd.read_csv(input_csv)
        print(f"Successfully loaded '{input_csv}'. Original shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found. Cannot proceed.")
        return
    except Exception as e:
        print(f"Error loading '{input_csv}': {e}")
        return

    # Ensure 'time' column exists
    time_col_name = None
    # Try to find the time column, being flexible with naming like "Time (s)" or "time"
    if 'time' in df.columns:
        time_col_name = 'time'
    elif 'Time (s)' in df.columns: # From previous discussions
        time_col_name = 'Time (s)'
    else: # Fallback: try to find any column named 'time' case-insensitively
        for col in df.columns:
            if str(col).lower() == 'time':
                time_col_name = col
                break
        if not time_col_name: # Try 'time (s)' case-insensitively if still not found
             for col in df.columns:
                if str(col).lower() == 'time (s)':
                    time_col_name = col
                    break


    if not time_col_name:
        print("Error: A 'time' or 'Time (s)' column was not found in the CSV. Cannot proceed.")
        return
    
    print(f"Using time column: '{time_col_name}'")

    # Convert time column to numeric, coercing errors (though it should be numeric already)
    df[time_col_name] = pd.to_numeric(df[time_col_name], errors='coerce')

    # Store original row count
    original_rows = len(df)

    # Filter out rows where the time column is negative
    # Also handles NaT/NaN in time column if 'coerce' created them, by not including them in > 0
    df_cleaned = df[df[time_col_name] >= 0]

    # Calculate how many rows were removed
    rows_removed = original_rows - len(df_cleaned)

    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with negative (or non-numeric) time values.")
    else:
        print("No negative time values found or removed.")

    print(f"Shape of DataFrame after removing negative times: {df_cleaned.shape}")

    # Save the cleaned DataFrame to a new CSV
    try:
        df_cleaned.to_csv(output_csv, index=False)
        print(f"Cleaned data saved to '{output_csv}'")
    except Exception as e:
        print(f"Error saving cleaned data to '{output_csv}': {e}")

if __name__ == '__main__':
    # Define your input and desired output filenames
    input_file = 'combined_activity_data.csv'
    output_file = 'trimmed_data.csv' # Changed output name slightly for clarity
    
    clean_negative_time_values(input_csv=input_file, output_csv=output_file)
    print("\nScript finished.")