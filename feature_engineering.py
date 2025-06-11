import pandas as pd
import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import pywt
import warnings
warnings.filterwarnings('ignore')

class ActivityFeatureEngineer:
    """
    Comprehensive feature engineering for activity recognition using sensor data.
    Implements approaches from the book chapter plus domain-specific methods.
    """
    
    def __init__(self):
        self.motion_sensors = [
            'Accelerometer_X (m/s^2)', 'Accelerometer_Y (m/s^2)', 'Accelerometer_Z (m/s^2)',
            'Gyroscope_X (rad/s)', 'Gyroscope_Y (rad/s)', 'Gyroscope_Z (rad/s)',
            'Linear Accelerometer_X (m/s^2)', 'Linear Accelerometer_Y (m/s^2)', 'Linear Accelerometer_Z (m/s^2)',
            'Magnetometer_X (µT)', 'Magnetometer_Y (µT)', 'Magnetometer_Z (µT)'
        ]
        
        self.location_sensors = [
            'Location_Direction (°)', 'Location_Height (m)', 'Location_Horizontal Accuracy (m)',
            'Location_Latitude (°)', 'Location_Longitude (°)', 'Location_Velocity (m/s)', 
            'Location_Vertical Accuracy (°)'
        ]
        
        self.environmental_sensors = [
            'Barometer_X (hPa)', 'Proximity_0_count', 'Proximity_5_count'
        ]
        
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load and prepare the dataset."""
        print("Loading dataset...")
        self.data = pd.read_csv(filepath)
        print(f"Dataset loaded: {self.data.shape}")
        return self.data
    
    def compute_statistical_features(self, data, sensors, prefix=""):
        """
        Compute statistical aggregation features (from book chapter).
        """
        features = {}
        
        for sensor in sensors:
            if sensor in data.columns:
                values = data[sensor].values
                
                # Basic statistics
                features[f'{prefix}{sensor}_mean'] = np.mean(values)
                features[f'{prefix}{sensor}_std'] = np.std(values)
                features[f'{prefix}{sensor}_min'] = np.min(values)
                features[f'{prefix}{sensor}_max'] = np.max(values)
                features[f'{prefix}{sensor}_median'] = np.median(values)
                features[f'{prefix}{sensor}_range'] = np.max(values) - np.min(values)
                
                # Higher-order statistics
                features[f'{prefix}{sensor}_skew'] = stats.skew(values)
                features[f'{prefix}{sensor}_kurtosis'] = stats.kurtosis(values)
                features[f'{prefix}{sensor}_var'] = np.var(values)
                
                # Percentiles
                features[f'{prefix}{sensor}_q25'] = np.percentile(values, 25)
                features[f'{prefix}{sensor}_q75'] = np.percentile(values, 75)
                features[f'{prefix}{sensor}_iqr'] = np.percentile(values, 75) - np.percentile(values, 25)
                
                # Energy and power
                features[f'{prefix}{sensor}_energy'] = np.sum(values**2)
                features[f'{prefix}{sensor}_power'] = np.mean(values**2)
                features[f'{prefix}{sensor}_rms'] = np.sqrt(np.mean(values**2))
                
        return features
    
    def compute_fft_features(self, data, sensors, prefix=""):
        """
        Compute frequency domain features using FFT (from book chapter).
        """
        features = {}
        
        for sensor in sensors:
            if sensor in data.columns:
                values = data[sensor].values
                n = len(values)
                
                if n < 8:  # Need minimum samples for FFT
                    continue
                    
                # Compute FFT
                fft_vals = fft(values)
                fft_freqs = fftfreq(n, d=1.0)  # Assuming 1 Hz sampling
                
                # Get magnitude spectrum (first half due to symmetry)
                magnitude = np.abs(fft_vals[:n//2])
                freqs = fft_freqs[:n//2]
                
                if len(magnitude) > 0:
                    # Peak frequency (from book chapter)
                    peak_freq_idx = np.argmax(magnitude)
                    features[f'{prefix}{sensor}_peak_freq'] = freqs[peak_freq_idx] if peak_freq_idx < len(freqs) else 0
                    features[f'{prefix}{sensor}_peak_magnitude'] = magnitude[peak_freq_idx]
                    
                    # Frequency weighted average (from book chapter)
                    if np.sum(magnitude) > 0:
                        features[f'{prefix}{sensor}_freq_weighted_avg'] = np.sum(freqs * magnitude) / np.sum(magnitude)
                    else:
                        features[f'{prefix}{sensor}_freq_weighted_avg'] = 0
                    
                    # Power spectral entropy (from book chapter)
                    psd = magnitude**2
                    if np.sum(psd) > 0:
                        psd_norm = psd / np.sum(psd)
                        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros for log
                        features[f'{prefix}{sensor}_spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm))
                    else:
                        features[f'{prefix}{sensor}_spectral_entropy'] = 0
                    
                    # Additional spectral features
                    features[f'{prefix}{sensor}_spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
                    features[f'{prefix}{sensor}_spectral_rolloff'] = self._compute_spectral_rolloff(freqs, magnitude)
                    features[f'{prefix}{sensor}_spectral_bandwidth'] = self._compute_spectral_bandwidth(freqs, magnitude)
                    
                    # Energy in frequency bands
                    features[f'{prefix}{sensor}_low_freq_energy'] = np.sum(magnitude[freqs <= 1]**2)
                    features[f'{prefix}{sensor}_mid_freq_energy'] = np.sum(magnitude[(freqs > 1) & (freqs <= 5)]**2)
                    features[f'{prefix}{sensor}_high_freq_energy'] = np.sum(magnitude[freqs > 5]**2)
        
        return features
    
    def _compute_spectral_rolloff(self, freqs, magnitude, rolloff_percent=0.85):
        """Compute spectral rolloff point."""
        total_energy = np.sum(magnitude**2)
        if total_energy == 0:
            return 0
        
        cumulative_energy = np.cumsum(magnitude**2)
        rolloff_idx = np.where(cumulative_energy >= rolloff_percent * total_energy)[0]
        return freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    def _compute_spectral_bandwidth(self, freqs, magnitude):
        """Compute spectral bandwidth around centroid."""
        if np.sum(magnitude) == 0:
            return 0
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        return np.sqrt(np.sum(((freqs - centroid)**2) * magnitude) / np.sum(magnitude))
    
    def compute_motion_magnitude_features(self, data, prefix=""):
        """
        Compute motion magnitude and orientation features.
        """
        features = {}
        
        # Accelerometer magnitude features
        if all(col in data.columns for col in ['Accelerometer_X (m/s^2)', 'Accelerometer_Y (m/s^2)', 'Accelerometer_Z (m/s^2)']):
            acc_x = data['Accelerometer_X (m/s^2)'].values
            acc_y = data['Accelerometer_Y (m/s^2)'].values
            acc_z = data['Accelerometer_Z (m/s^2)'].values
            
            # Total acceleration magnitude
            acc_total = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            features.update(self._compute_magnitude_stats(acc_total, f'{prefix}acc_total'))
            
            # Horizontal acceleration
            acc_horizontal = np.sqrt(acc_x**2 + acc_y**2)
            features.update(self._compute_magnitude_stats(acc_horizontal, f'{prefix}acc_horizontal'))
            
            # Vertical component
            features.update(self._compute_magnitude_stats(acc_z, f'{prefix}acc_vertical'))
            
            # Tilt and orientation
            features[f'{prefix}acc_tilt_mean'] = np.mean(np.arctan2(acc_horizontal, np.abs(acc_z)))
            features[f'{prefix}acc_tilt_std'] = np.std(np.arctan2(acc_horizontal, np.abs(acc_z)))
            
        # Gyroscope magnitude features
        if all(col in data.columns for col in ['Gyroscope_X (rad/s)', 'Gyroscope_Y (rad/s)', 'Gyroscope_Z (rad/s)']):
            gyro_x = data['Gyroscope_X (rad/s)'].values
            gyro_y = data['Gyroscope_Y (rad/s)'].values
            gyro_z = data['Gyroscope_Z (rad/s)'].values
            
            gyro_total = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            features.update(self._compute_magnitude_stats(gyro_total, f'{prefix}gyro_total'))
            
        # Linear accelerometer features
        if all(col in data.columns for col in ['Linear Accelerometer_X (m/s^2)', 'Linear Accelerometer_Y (m/s^2)', 'Linear Accelerometer_Z (m/s^2)']):
            lin_acc_x = data['Linear Accelerometer_X (m/s^2)'].values
            lin_acc_y = data['Linear Accelerometer_Y (m/s^2)'].values
            lin_acc_z = data['Linear Accelerometer_Z (m/s^2)'].values
            
            lin_acc_total = np.sqrt(lin_acc_x**2 + lin_acc_y**2 + lin_acc_z**2)
            features.update(self._compute_magnitude_stats(lin_acc_total, f'{prefix}lin_acc_total'))
            
        return features
    
    def _compute_magnitude_stats(self, magnitude, prefix):
        """Compute statistics for magnitude signals."""
        return {
            f'{prefix}_mean': np.mean(magnitude),
            f'{prefix}_std': np.std(magnitude),
            f'{prefix}_max': np.max(magnitude),
            f'{prefix}_min': np.min(magnitude),
            f'{prefix}_range': np.max(magnitude) - np.min(magnitude),
            f'{prefix}_energy': np.sum(magnitude**2)
        }
    
    def compute_gait_features(self, data, prefix=""):
        """
        Compute gait and movement pattern features.
        """
        features = {}
        
        # Use total acceleration for gait analysis
        if all(col in data.columns for col in ['Accelerometer_X (m/s^2)', 'Accelerometer_Y (m/s^2)', 'Accelerometer_Z (m/s^2)']):
            acc_x = data['Accelerometer_X (m/s^2)'].values
            acc_y = data['Accelerometer_Y (m/s^2)'].values
            acc_z = data['Accelerometer_Z (m/s^2)'].values
            acc_total = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            
            # Peak detection for step counting
            peaks, _ = signal.find_peaks(acc_total, height=np.mean(acc_total), distance=5)
            features[f'{prefix}step_count'] = len(peaks)
            features[f'{prefix}step_frequency'] = len(peaks) / len(acc_total) if len(acc_total) > 0 else 0
            
            # Peak intervals (cadence)
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks)
                features[f'{prefix}cadence_mean'] = np.mean(peak_intervals)
                features[f'{prefix}cadence_std'] = np.std(peak_intervals)
                features[f'{prefix}cadence_regularity'] = 1 / (1 + np.std(peak_intervals)) if np.std(peak_intervals) > 0 else 1
            else:
                features[f'{prefix}cadence_mean'] = 0
                features[f'{prefix}cadence_std'] = 0
                features[f'{prefix}cadence_regularity'] = 0
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(acc_total - np.mean(acc_total))) != 0)
            features[f'{prefix}zero_crossing_rate'] = zero_crossings / len(acc_total) if len(acc_total) > 0 else 0
            
            # Jerk (rate of change of acceleration)
            if len(acc_total) > 1:
                jerk = np.diff(acc_total)
                features[f'{prefix}jerk_mean'] = np.mean(np.abs(jerk))
                features[f'{prefix}jerk_std'] = np.std(jerk)
                features[f'{prefix}jerk_max'] = np.max(np.abs(jerk))
            else:
                features[f'{prefix}jerk_mean'] = 0
                features[f'{prefix}jerk_std'] = 0
                features[f'{prefix}jerk_max'] = 0
                
        return features
    
    def compute_cross_sensor_features(self, data, prefix=""):
        """
        Compute cross-sensor correlation and fusion features.
        """
        features = {}
        
        # Accelerometer-Gyroscope correlation
        acc_sensors = ['Accelerometer_X (m/s^2)', 'Accelerometer_Y (m/s^2)', 'Accelerometer_Z (m/s^2)']
        gyro_sensors = ['Gyroscope_X (rad/s)', 'Gyroscope_Y (rad/s)', 'Gyroscope_Z (rad/s)']
        
        for i, (acc, gyro) in enumerate(zip(acc_sensors, gyro_sensors)):
            if acc in data.columns and gyro in data.columns:
                corr = np.corrcoef(data[acc].values, data[gyro].values)[0, 1]
                features[f'{prefix}acc_gyro_corr_{i}'] = corr if not np.isnan(corr) else 0
        
        # Linear vs Total accelerometer relationship
        lin_acc_sensors = ['Linear Accelerometer_X (m/s^2)', 'Linear Accelerometer_Y (m/s^2)', 'Linear Accelerometer_Z (m/s^2)']
        for i, (acc, lin_acc) in enumerate(zip(acc_sensors, lin_acc_sensors)):
            if acc in data.columns and lin_acc in data.columns:
                corr = np.corrcoef(data[acc].values, data[lin_acc].values)[0, 1]
                features[f'{prefix}acc_lin_acc_corr_{i}'] = corr if not np.isnan(corr) else 0
        
        # Motion coherence across sensors
        motion_sensors = acc_sensors + gyro_sensors
        motion_data = []
        for sensor in motion_sensors:
            if sensor in data.columns:
                motion_data.append(data[sensor].values)
        
        if len(motion_data) >= 2:
            motion_matrix = np.array(motion_data)
            corr_matrix = np.corrcoef(motion_matrix)
            # Average correlation as motion coherence
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            features[f'{prefix}motion_coherence'] = np.mean(upper_triangle[~np.isnan(upper_triangle)])
        
        return features
    
    def compute_location_features(self, data, prefix=""):
        """
        Compute location and movement features.
        """
        features = {}
        
        # Velocity features
        if 'Location_Velocity (m/s)' in data.columns:
            velocity = data['Location_Velocity (m/s)'].values
            features.update(self._compute_magnitude_stats(velocity, f'{prefix}velocity'))
            
            # Speed consistency
            features[f'{prefix}speed_consistency'] = 1 / (1 + np.std(velocity)) if np.std(velocity) > 0 else 1
        
        # Direction changes
        if 'Location_Direction (°)' in data.columns:
            direction = data['Location_Direction (°)'].values
            if len(direction) > 1:
                direction_changes = np.abs(np.diff(direction))
                # Handle circular nature of angles
                direction_changes = np.minimum(direction_changes, 360 - direction_changes)
                features[f'{prefix}direction_change_mean'] = np.mean(direction_changes)
                features[f'{prefix}direction_change_std'] = np.std(direction_changes)
                features[f'{prefix}direction_stability'] = 1 / (1 + np.std(direction_changes))
            else:
                features[f'{prefix}direction_change_mean'] = 0
                features[f'{prefix}direction_change_std'] = 0
                features[f'{prefix}direction_stability'] = 1
        
        # Height/elevation features
        if 'Location_Height (m)' in data.columns:
            height = data['Location_Height (m)'].values
            features.update(self._compute_magnitude_stats(height, f'{prefix}height'))
            
            if len(height) > 1:
                elevation_change = np.abs(np.diff(height))
                features[f'{prefix}elevation_change_mean'] = np.mean(elevation_change)
                features[f'{prefix}elevation_change_max'] = np.max(elevation_change)
        
        # GPS accuracy features
        if 'Location_Horizontal Accuracy (m)' in data.columns:
            h_accuracy = data['Location_Horizontal Accuracy (m)'].values
            features[f'{prefix}gps_h_accuracy_mean'] = np.mean(h_accuracy)
            features[f'{prefix}gps_h_accuracy_std'] = np.std(h_accuracy)
            features[f'{prefix}gps_quality'] = 1 / (1 + np.mean(h_accuracy))  # Inverse of error
        
        return features
    
    def compute_wavelet_features(self, data, sensors, prefix=""):
        """
        Compute wavelet transform features for multi-resolution analysis.
        """
        features = {}
        
        for sensor in sensors:
            if sensor in data.columns:
                values = data[sensor].values
                
                if len(values) >= 4:  # Minimum length for wavelet
                    try:
                        # Discrete Wavelet Transform
                        coeffs = pywt.wavedec(values, 'db4', level=3)
                        
                        # Energy in each level
                        for i, coeff in enumerate(coeffs):
                            energy = np.sum(coeff**2)
                            features[f'{prefix}{sensor}_wavelet_energy_level_{i}'] = energy
                        
                        # Wavelet entropy
                        total_energy = sum(np.sum(c**2) for c in coeffs)
                        if total_energy > 0:
                            energies = [np.sum(c**2) / total_energy for c in coeffs]
                            energies = [e for e in energies if e > 0]
                            wavelet_entropy = -sum(e * np.log(e) for e in energies)
                            features[f'{prefix}{sensor}_wavelet_entropy'] = wavelet_entropy
                        
                    except Exception:
                        # Skip if wavelet transform fails
                        pass
        
        return features
    
    def compute_autocorrelation_features(self, data, sensors, prefix=""):
        """
        Compute autocorrelation features for periodicity detection.
        """
        features = {}
        
        for sensor in sensors:
            if sensor in data.columns:
                values = data[sensor].values
                
                if len(values) > 10:
                    # Compute autocorrelation
                    autocorr = np.correlate(values, values, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
                    autocorr = autocorr / autocorr[0]  # Normalize
                    
                    # Find first significant peak (excluding lag 0)
                    if len(autocorr) > 1:
                        peaks, _ = signal.find_peaks(autocorr[1:], height=0.1)
                        if len(peaks) > 0:
                            features[f'{prefix}{sensor}_autocorr_peak_lag'] = peaks[0] + 1
                            features[f'{prefix}{sensor}_autocorr_peak_value'] = autocorr[peaks[0] + 1]
                        else:
                            features[f'{prefix}{sensor}_autocorr_peak_lag'] = 0
                            features[f'{prefix}{sensor}_autocorr_peak_value'] = 0
                        
                        # Autocorrelation at specific lags
                        for lag in [1, 2, 5, 10]:
                            if lag < len(autocorr):
                                features[f'{prefix}{sensor}_autocorr_lag_{lag}'] = autocorr[lag]
        
        return features
    
    def extract_temporal_patterns(self, activity_data, window_size=50, min_support=0.05):
        """
        Extract temporal patterns from categorical data (from book chapter).
        """
        features = {}
        
        # Get unique activities
        unique_activities = activity_data.unique()
        
        # 1-patterns (individual activities)
        for activity in unique_activities:
            pattern_name = f'pattern_1_{activity.replace(" ", "_")}'
            # Count occurrences in windows
            pattern_count = 0
            total_windows = len(activity_data) - window_size + 1
            
            for i in range(total_windows):
                window = activity_data.iloc[i:i+window_size]
                if activity in window.values:
                    pattern_count += 1
            
            support = pattern_count / total_windows if total_windows > 0 else 0
            if support >= min_support:
                features[pattern_name] = pattern_count
        
        # 2-patterns (activity transitions)
        for act1 in unique_activities:
            for act2 in unique_activities:
                if act1 != act2:
                    pattern_name = f'pattern_2_{act1.replace(" ", "_")}_before_{act2.replace(" ", "_")}'
                    pattern_count = 0
                    total_windows = len(activity_data) - window_size + 1
                    
                    for i in range(total_windows):
                        window = activity_data.iloc[i:i+window_size].values
                        # Check for act1 followed by act2
                        for j in range(len(window) - 1):
                            if window[j] == act1 and window[j+1] == act2:
                                pattern_count += 1
                                break
                    
                    support = pattern_count / total_windows if total_windows > 0 else 0
                    if support >= min_support:
                        features[pattern_name] = pattern_count
        
        return features
    
    def engineer_features(self, data):
        """
        Main feature engineering pipeline.
        """
        print("Starting feature engineering...")
        all_features = {}
        
        # 1. Time Domain Features (from book chapter)
        print("Computing time domain features...")
        all_features.update(self.compute_statistical_features(data, self.motion_sensors, "motion_"))
        all_features.update(self.compute_statistical_features(data, self.location_sensors, "location_"))
        all_features.update(self.compute_statistical_features(data, self.environmental_sensors, "env_"))
        
        # 2. Frequency Domain Features (from book chapter)
        print("Computing frequency domain features...")
        all_features.update(self.compute_fft_features(data, self.motion_sensors, "motion_"))
        
        # 3. Domain-Specific Motion Features
        print("Computing motion magnitude features...")
        all_features.update(self.compute_motion_magnitude_features(data, ""))
        
        # 4. Gait and Movement Features
        print("Computing gait features...")
        all_features.update(self.compute_gait_features(data, ""))
        
        # 5. Cross-Sensor Features
        print("Computing cross-sensor features...")
        all_features.update(self.compute_cross_sensor_features(data, ""))
        
        # 6. Location Features
        print("Computing location features...")
        all_features.update(self.compute_location_features(data, ""))
        
        # 7. Wavelet Features
        print("Computing wavelet features...")
        all_features.update(self.compute_wavelet_features(data, self.motion_sensors[:6], ""))  # Limit to prevent too many features
        
        # 8. Autocorrelation Features
        print("Computing autocorrelation features...")
        all_features.update(self.compute_autocorrelation_features(data, self.motion_sensors[:6], ""))
        
        # 9. Temporal Patterns (from book chapter)
        print("Computing temporal patterns...")
        if 'activity_type' in data.columns:
            temporal_features = self.extract_temporal_patterns(data['activity_type'])
            all_features.update(temporal_features)
        
        print(f"Total features engineered: {len(all_features)}")
        return all_features
    
    def process_dataset(self, filepath):
        """
        Complete processing pipeline for the dataset.
        """
        # Load data
        data = self.load_data(filepath)
        
        # Group data by windows (assuming each row is already a window)
        feature_list = []
        labels = []
        
        print("Processing windows...")
        for idx in range(len(data)):
            if idx % 1000 == 0:
                print(f"Processing window {idx}/{len(data)}")
            
            # Get single window (row)
            window_data = data.iloc[idx:idx+1]
            
            # Extract features for this window
            features = self.engineer_features(window_data)
            
            # Store features and label
            feature_list.append(features)
            labels.append(data.iloc[idx]['activity_type'])
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(feature_list)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Store feature names
        self.feature_names = feature_df.columns.tolist()
        
        print(f"Final feature matrix shape: {feature_df.shape}")
        print(f"Unique activities: {set(labels)}")
        
        return feature_df, np.array(labels)
    
    def select_features(self, X, y, k=300):
        """
        Feature selection using multiple methods.
        """
        print(f"Selecting top {k} features from {X.shape[1]} features...")
        
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        X_filtered = X.drop(columns=constant_features)
        print(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features
        corr_matrix = X_filtered.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        X_filtered = X_filtered.drop(columns=high_corr_features)
        print(f"Removed {len(high_corr_features)} highly correlated features")
        
        # Encode labels for feature selection
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Select top k features using F-test
        if X_filtered.shape[1] > k:
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X_filtered, y_encoded)
            selected_features = X_filtered.columns[selector.get_support()].tolist()
            
            print(f"Selected {len(selected_features)} features using F-test")
            return pd.DataFrame(X_selected, columns=selected_features), selected_features
        else:
            print(f"Keeping all {X_filtered.shape[1]} features (less than k={k})")
            return X_filtered, X_filtered.columns.tolist()
    
    def save_features(self, X, y, selected_features, filepath="engineered_features.csv"):
        """
        Save engineered features to CSV.
        """
        # Combine features and labels
        result_df = X.copy()
        result_df['activity_type'] = y
        
        # Save to CSV
        result_df.to_csv(filepath, index=False)
        print(f"Features saved to {filepath}")
        
        # Save feature names
        with open("selected_features.txt", "w") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        print("Selected feature names saved to selected_features.txt")
        
        return result_df

def main():
    """
    Main execution function.
    """
    # Initialize feature engineer
    engineer = ActivityFeatureEngineer()
    
    # Process the dataset
    X, y = engineer.process_dataset("aggregated_data_cleaned.csv")
    
    # Feature selection
    X_selected, selected_features = engineer.select_features(X, y, k=300)
    
    # Save results
    result_df = engineer.save_features(X_selected, y, selected_features)
    
    # Print summary
    print("\n=== Feature Engineering Summary ===")
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Data points: {X_selected.shape[0]}")
    print(f"Activities: {len(set(y))}")
    print("\nActivity distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for activity, count in zip(unique, counts):
        print(f"  {activity}: {count}")
    
    print("\nFeature engineering complete!")
    return result_df, X_selected, y, selected_features

if __name__ == "__main__":
    result_df, X_selected, y, selected_features = main()