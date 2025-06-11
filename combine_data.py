import pandas as pd
import os


def combine_activity_data():
    """
    Combines data from various sensor CSV files across multiple activity subfolders.
    """
    base_data_path = 'data'
    activity_folders = ['high knees', 'normal run', 'side skips', 'sprints', 'standing', 'walking']
    sensor_files = [
        'Accelerometer.csv',
        'Barometer.csv',
        'Gyroscope.csv',
        'Linear Accelerometer.csv',
        'Location.csv',
        'Magnetometer.csv',
        'Proximity.csv'
    ]

    all_activity_dfs = []

    for activity in activity_folders:
        print(f"Processing activity: {activity}...")
        activity_path = os.path.join(base_data_path, activity)
        
        if not os.path.isdir(activity_path):
            print(f"Warning: Directory not found for activity {activity}. Skipping.")
            continue

        # To store dataframes from each sensor for the current activity
        sensor_dataframes_for_activity = []

        for sensor_file_name in sensor_files:
            file_path = os.path.join(activity_path, sensor_file_name)
            
            if not os.path.exists(file_path):
                print(f"Warning: File {sensor_file_name} not found in {activity_path}. Skipping.")
                continue

            try:
                df = pd.read_csv(file_path)
                
                # Identify the 'time' column (handle variations like 'Time', 'time')
                time_col = None
                for col in df.columns:
                    if col.lower() == 'time (s)':
                        time_col = col
                        break
                
                if not time_col:
                    print(f"Warning: 'time' column not found in {file_path}. Skipping this file.")
                    continue

                # Rename value columns to make them unique, prefixing with sensor name
                # e.g., 'value_x' in Accelerometer.csv becomes 'Accelerometer_value_x'
                # Keep the original 'time' column name for merging
                sensor_name_prefix = sensor_file_name.replace('.csv', '')
                
                columns_to_rename = {}
                for col in df.columns:
                    if col != time_col: # Don't rename the time column itself yet
                        columns_to_rename[col] = f"{sensor_name_prefix}_{col}"
                df = df.rename(columns=columns_to_rename)
                
                # Ensure the time column is consistently named 'time' for merging
                if time_col != 'time':
                    df = df.rename(columns={time_col: 'time'})

                sensor_dataframes_for_activity.append(df)

            except Exception as e:
                print(f"Error reading or processing {file_path}: {e}")
                continue
        
        if not sensor_dataframes_for_activity:
            print(f"No sensor data processed for activity {activity}. Skipping merging.")
            continue

        # Merge all sensor dataframes for the current activity based on 'time'
        # Start with the first dataframe, then merge others into it
        merged_activity_df = sensor_dataframes_for_activity[0]
        for i in range(1, len(sensor_dataframes_for_activity)):
            try:
                # Using outer merge to keep all time points from all files
                # Suffixes are added to handle potential duplicate non-time columns if any were missed in renaming (should not happen with prefixing)
                merged_activity_df = pd.merge(merged_activity_df, sensor_dataframes_for_activity[i], on='time', how='outer', suffixes=('_left', '_right'))
            except Exception as e:
                print(f"Error merging dataframes for activity {activity} (file index {i}): {e}")
                # Fallback or skip this merge if critical
                # For simplicity, we'll just print error and it might lead to incomplete data for this activity
                continue


        # Add the activity type column
        merged_activity_df['activity_type'] = activity
        all_activity_dfs.append(merged_activity_df)
        print(f"Finished processing activity: {activity}. Shape: {merged_activity_df.shape}")


    if not all_activity_dfs:
        print("No data was processed. Exiting.")
        return None

    # Concatenate all activity dataframes
    final_df = pd.concat(all_activity_dfs, ignore_index=True)
    
    # Optional: Sort by time if needed, though merging might mix orders if time isn't globally unique or perfectly aligned
    # final_df = final_df.sort_values(by='time').reset_index(drop=True)

    print("\\nFinal combined DataFrame:")
    print(f"Shape: {final_df.shape}")
    print(final_df.head())
    
    # You can save the final_df to a CSV like this:
    final_df.to_csv('combined_activity_data.csv', index=False)
    print("\\nSaved combined data to 'combined_activity_data.csv'")
    
    return final_df

if __name__ == '__main__':
    combined_data = combine_activity_data()