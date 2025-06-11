import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class ActivityClassificationFramework:
    """
    Comprehensive framework for smartphone sensor-based activity classification
    following the research methodology from the ML4QS paper.
    """

    def __init__(self, data_path=None, df=None):
        """
        Initialize the framework with either a file path or DataFrame

        Args:
            data_path (str): Path to the dataset CSV file
            df (DataFrame): Preprocessed DataFrame with sensor data
        """
        if data_path:
            df = pd.read_csv(data_path)
            df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg').replace('/', '_').replace('µT', 'uT') for col in df.columns]
            if 'window_start_time' in df.columns:
                df = df.rename(columns={'window_start_time': 'time'})
            self.df = pd.read_csv(data_path)
        elif df is not None:
            df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg').replace('/', '_').replace('µT', 'uT') for col in df.columns]
            if 'window_start_time' in df.columns:
                df = df.rename(columns={'window_start_time': 'time'})
            self.df = df.copy()
        else:
            raise ValueError("Either data_path or df must be provided")

        self.features = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}

    def engineer_features(self, window_size=10, overlap=0.5):
        """
        Engineer comprehensive features from sensor data

        Args:
            window_size (int): Size of sliding window for feature extraction
            overlap (float): Overlap ratio between windows (0-1)

        Returns:
            DataFrame: Engineered features dataset
        """
        print("🔧 Starting feature engineering...")

        # Identify sensor columns (exclude time and activity_type)
        sensor_cols = [col for col in self.df.columns if col not in ['time', 'activity_type']]

        # Separate high-frequency and low-frequency sensors based on your research
        high_freq_sensors = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z', 'Linear_Accelerometer_X', 'Linear_Accelerometer_Y', 'Linear_Accelerometer_Z', 'Magnetometer_X', 'Magnetometer_Y', 'Magnetometer_Z']

        low_freq_sensors = ['Barometer_X', 'Location_Direction', 'Location_Height', 'Location_Horizontal_Accuracy', 'Location_Latitude', 'Location_Longitude', 'Location_Velocity', 'Location_Vertical_Accuracy', 'Proximity_0_count', 'Proximity_5_count']

        # Filter available sensors
        available_high_freq = [col for col in high_freq_sensors if col in sensor_cols]
        available_low_freq = [col for col in low_freq_sensors if col in sensor_cols]

        engineered_features = []

        # Create sliding windows
        step_size = int(window_size * (1 - overlap))

        for start_idx in range(0, len(self.df) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = self.df.iloc[start_idx:end_idx]

            # Get the most common activity in the window (for labeling)
            activity = window_data['activity_type'].mode().iloc[0]

            feature_dict = {'activity_type': activity}

            # 1. TIME-DOMAIN FEATURES
            for sensor in available_high_freq + available_low_freq:
                if sensor in window_data.columns:
                    values = window_data[sensor].dropna()

                    if len(values) > 0:
                        # Statistical features
                        feature_dict[f'{sensor}_mean'] = values.mean()
                        feature_dict[f'{sensor}_std'] = values.std()
                        feature_dict[f'{sensor}_min'] = values.min()
                        feature_dict[f'{sensor}_max'] = values.max()
                        feature_dict[f'{sensor}_range'] = values.max() - values.min()
                        feature_dict[f'{sensor}_median'] = values.median()
                        feature_dict[f'{sensor}_q25'] = values.quantile(0.25)
                        feature_dict[f'{sensor}_q75'] = values.quantile(0.75)
                        feature_dict[f'{sensor}_iqr'] = values.quantile(0.75) - values.quantile(0.25)
                        feature_dict[f'{sensor}_skew'] = values.skew()
                        feature_dict[f'{sensor}_kurtosis'] = values.kurtosis()

                        # Signal characteristics
                        feature_dict[f'{sensor}_rms'] = np.sqrt(np.mean(values**2))
                        feature_dict[f'{sensor}_energy'] = np.sum(values**2)
                        feature_dict[f'{sensor}_zero_crossings'] = np.sum(np.diff(np.signbit(values)))

                        # Peak analysis
                        peaks, _ = find_peaks(values)
                        feature_dict[f'{sensor}_peak_count'] = len(peaks)
                        feature_dict[f'{sensor}_peak_prominence'] = np.mean([p for p in peaks]) if len(peaks) > 0 else 0

            # 2. MAGNITUDE FEATURES (for 3D sensors)
            for sensor_base in ['Accelerometer', 'Gyroscope', 'Linear Accelerometer', 'Magnetometer']:
                x_col = f'{sensor_base} X'
                y_col = f'{sensor_base} Y'
                z_col = f'{sensor_base} Z'

                if all(col in window_data.columns for col in [x_col, y_col, z_col]):
                    x_vals = window_data[x_col].dropna()
                    y_vals = window_data[y_col].dropna()
                    z_vals = window_data[z_col].dropna()

                    if len(x_vals) > 0 and len(y_vals) > 0 and len(z_vals) > 0:
                        # Ensure same length
                        min_len = min(len(x_vals), len(y_vals), len(z_vals))
                        x_vals = x_vals.iloc[:min_len]
                        y_vals = y_vals.iloc[:min_len]
                        z_vals = z_vals.iloc[:min_len]

                        # Magnitude
                        magnitude = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)
                        feature_dict[f'{sensor_base}_magnitude_mean'] = magnitude.mean()
                        feature_dict[f'{sensor_base}_magnitude_std'] = magnitude.std()
                        feature_dict[f'{sensor_base}_magnitude_max'] = magnitude.max()

                        # Correlations between axes
                        feature_dict[f'{sensor_base}_xy_corr'] = np.corrcoef(x_vals, y_vals)[0,1] if len(x_vals) > 1 else 0
                        feature_dict[f'{sensor_base}_xz_corr'] = np.corrcoef(x_vals, z_vals)[0,1] if len(x_vals) > 1 else 0
                        feature_dict[f'{sensor_base}_yz_corr'] = np.corrcoef(y_vals, z_vals)[0,1] if len(y_vals) > 1 else 0

            # 3. FREQUENCY-DOMAIN FEATURES (for high-frequency sensors)
            for sensor in available_high_freq:
                if sensor in window_data.columns:
                    values = window_data[sensor].dropna()

                    if len(values) > 2:
                        # FFT features
                        fft_values = np.abs(np.fft.fft(values))
                        fft_freqs = np.fft.fftfreq(len(values), d=0.01)  # 0.01s granularity

                        # Keep only positive frequencies
                        pos_freqs = fft_freqs[:len(fft_freqs)//2]
                        pos_fft = fft_values[:len(fft_values)//2]

                        if len(pos_fft) > 0:
                            feature_dict[f'{sensor}_dominant_freq'] = pos_freqs[np.argmax(pos_fft)]
                            feature_dict[f'{sensor}_spectral_centroid'] = np.sum(pos_freqs * pos_fft) / np.sum(pos_fft)
                            feature_dict[f'{sensor}_spectral_rolloff'] = pos_freqs[np.where(np.cumsum(pos_fft) >= 0.85 * np.sum(pos_fft))[0][0]] if len(pos_fft) > 1 else 0
                            feature_dict[f'{sensor}_spectral_bandwidth'] = np.sqrt(np.sum(((pos_freqs - feature_dict[f'{sensor}_spectral_centroid'])**2) * pos_fft) / np.sum(pos_fft))

            # 4. ACTIVITY-SPECIFIC FEATURES
            # Proximity pattern analysis
            if 'Proximity 0 count' in window_data.columns and 'Proximity 5 count' in window_data.columns:
                prox_0 = window_data['Proximity 0 count'].sum()
                prox_5 = window_data['Proximity 5 count'].sum()
                total_prox = prox_0 + prox_5

                feature_dict['proximity_ratio'] = prox_0 / total_prox if total_prox > 0 else 0
                feature_dict['proximity_activity'] = total_prox

            # Movement intensity (from accelerometer magnitude)
            if all(col in window_data.columns for col in ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']):
                acc_mag = np.sqrt(window_data['Accelerometer X']**2 +
                                window_data['Accelerometer Y']**2 +
                                window_data['Accelerometer Z']**2)
                feature_dict['movement_intensity'] = acc_mag.var()
                feature_dict['movement_smoothness'] = 1 / (1 + acc_mag.std()) if acc_mag.std() > 0 else 1

            engineered_features.append(feature_dict)

        # Convert to DataFrame
        features_df = pd.DataFrame(engineered_features)

        # Handle NaN values
        features_df = features_df.fillna(0)

        print(f"✅ Feature engineering complete!")
        print(f"   - Original shape: {self.df.shape}")
        print(f"   - Engineered shape: {features_df.shape}")
        print(f"   - Features created: {features_df.shape[1] - 1}")  # -1 for activity_type

        self.features_df = features_df
        return features_df

    def analyze_features(self, top_k=20):
        """
        Analyze the usefulness of engineered features

        Args:
            top_k (int): Number of top features to display
        """
        print("📊 Analyzing feature importance...")

        if self.features_df is None:
            raise ValueError("Features not engineered yet. Run engineer_features() first.")

        # Prepare data
        X = self.features_df.drop('activity_type', axis=1)
        y = self.features_df['activity_type']

        # 1. Statistical significance (ANOVA F-test)
        f_scores, p_values = f_classif(X, y)
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'f_score': f_scores,
            'p_value': p_values
        }).sort_values('f_score', ascending=False)

        print(f"\n🔍 Top {top_k} most statistically significant features:")
        print(feature_importance_df.head(top_k).to_string(index=False))

        # 2. Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🌲 Top {top_k} Random Forest feature importances:")
        print(rf_importance.head(top_k).to_string(index=False))

        # 3. Correlation analysis
        corr_matrix = X.corr().abs()
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.9:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        if high_corr_pairs:
            print(f"\n🔗 Highly correlated feature pairs (>0.9):")
            for feat1, feat2, corr in high_corr_pairs[:10]:
                print(f"   {feat1} ↔ {feat2}: {corr:.3f}")

        # 4. Feature distribution by activity
        plt.figure(figsize=(15, 10))

        # Plot 1: Feature importance comparison
        plt.subplot(2, 2, 1)
        top_features = feature_importance_df.head(10)
        plt.barh(range(len(top_features)), top_features['f_score'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('F-Score')
        plt.title('Top 10 Features by Statistical Significance')
        plt.gca().invert_yaxis()

        # Plot 2: Random Forest importance
        plt.subplot(2, 2, 2)
        top_rf_features = rf_importance.head(10)
        plt.barh(range(len(top_rf_features)), top_rf_features['importance'])
        plt.yticks(range(len(top_rf_features)), top_rf_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 10 Features by Random Forest Importance')
        plt.gca().invert_yaxis()

        # Plot 3: Activity distribution
        plt.subplot(2, 2, 3)
        activity_counts = self.features_df['activity_type'].value_counts()
        plt.bar(activity_counts.index, activity_counts.values)
        plt.xlabel('Activity Type')
        plt.ylabel('Count')
        plt.title('Activity Distribution in Engineered Dataset')
        plt.xticks(rotation=45)

        # Plot 4: Feature correlation heatmap (top features)
        plt.subplot(2, 2, 4)
        top_feature_names = feature_importance_df.head(15)['feature'].tolist()
        top_corr = X[top_feature_names].corr()
        sns.heatmap(top_corr, annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Top 15 Features')

        plt.tight_layout()
        plt.show()

        # Store analysis results
        self.feature_analysis = {
            'statistical_importance': feature_importance_df,
            'rf_importance': rf_importance,
            'high_correlations': high_corr_pairs
        }

        return self.feature_analysis

    def setup_train_test(self, test_size=0.333, random_state=42, stratify=True):
        """
        Set up train/test split with proper stratification

        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
            stratify (bool): Whether to stratify split by activity type
        """
        print("🔄 Setting up train/test split...")

        if self.features_df is None:
            raise ValueError("Features not engineered yet. Run engineer_features() first.")

        # Prepare features and target
        X = self.features_df.drop('activity_type', axis=1)
        y = self.features_df['activity_type']

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Train/test split
        stratify_param = y_encoded if stratify else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"✅ Train/test split complete!")
        print(f"   - Training set: {self.X_train.shape[0]} instances")
        print(f"   - Test set: {self.X_test.shape[0]} instances")
        print(f"   - Features: {self.X_train.shape[1]}")
        print(f"   - Classes: {len(np.unique(self.y_train))}")

        # Print class distribution
        train_dist = pd.Series(self.y_train).value_counts().sort_index()
        test_dist = pd.Series(self.y_test).value_counts().sort_index()

        print("\n📊 Class distribution:")
        for i, (train_count, test_count) in enumerate(zip(train_dist, test_dist)):
            class_name = self.label_encoder.inverse_transform([i])[0]
            print(f"   {class_name}: Train={train_count}, Test={test_count}")

    def train_models(self, models_to_use=None, optimize_hyperparameters=True):
        """
        Train multiple machine learning models with hyperparameter optimization

        Args:
            models_to_use (list): List of model names to use
            optimize_hyperparameters (bool): Whether to perform grid search
        """
        print("🚀 Training machine learning models...")

        if self.X_train is None:
            raise ValueError("Train/test split not set up. Run setup_train_test() first.")

        # Define models and their hyperparameters
        model_configs = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                } if optimize_hyperparameters else {}
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                } if optimize_hyperparameters else {}
            },
            'SVM': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto']
                } if optimize_hyperparameters else {}
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                } if optimize_hyperparameters else {}
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                } if optimize_hyperparameters else {}
            }
        }

        if models_to_use:
            print(f"Training models: {models_to_use}")
            model_configs = {k: v for k, v in model_configs.items() if k in models_to_use}

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, config in model_configs.items():
            print(f"\n🔧 Training {model_name}...")

            if optimize_hyperparameters and config['params']:
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(self.X_train_scaled, self.y_train)

                self.models[model_name] = grid_search.best_estimator_

                print(f"   Best parameters: {grid_search.best_params_}")
                print(f"   Best CV score: {grid_search.best_score_:.4f}")

            else:
                # Train with default parameters
                config['model'].fit(self.X_train_scaled, self.y_train)
                self.models[model_name] = config['model']

        print("✅ Model training complete!")

    def evaluate_models(self):
        """
        Evaluate all trained models and provide comprehensive results
        """
        print("📈 Evaluating models...")

        if not self.models:
            raise ValueError("No models trained yet. Run train_models() first.")

        results = {}

        for model_name, model in self.models.items():
            print(f"\n🔍 Evaluating {model_name}...")

            # Predictions
            y_pred = model.predict(self.X_test_scaled)

            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)

            # Classification report
            class_names = self.label_encoder.classes_
            report = classification_report(
                self.y_test,
                y_pred,
                target_names=class_names,
                output_dict=True
            )

            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)

            results[model_name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'predictions': y_pred
            }

            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Macro avg F1: {report['macro avg']['f1-score']:.4f}")
            print(f"   Weighted avg F1: {report['weighted avg']['f1-score']:.4f}")

        self.results = results

        # Create visualization
        self.plot_results()

        return results

    def plot_results(self):
        """
        Create comprehensive visualization of model results
        """
        if not self.results:
            return

        n_models = len(self.results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))

        if n_models == 1:
            axes = axes.reshape(-1, 1)

        class_names = self.label_encoder.classes_

        # Model comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        f1_scores = [self.results[name]['classification_report']['weighted avg']['f1-score'] for name in model_names]

        for i, (model_name, result) in enumerate(self.results.items()):
            # Confusion matrix
            sns.heatmap(
                result['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[0, i]
            )
            axes[0, i].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.4f}')
            axes[0, i].set_xlabel('Predicted')
            axes[0, i].set_ylabel('Actual')

        # Performance comparison
        axes[1, 0].bar(model_names, accuracies, alpha=0.7, label='Accuracy')
        axes[1, 0].bar(model_names, f1_scores, alpha=0.7, label='F1-Score')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Model Performance Comparison')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Feature importance for best model
        if len(self.results) > 1:
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            best_model = self.models[best_model_name]

            if hasattr(best_model, 'feature_importances_'):
                feature_names = self.X_train.columns
                importances = best_model.feature_importances_

                # Get top 15 features
                top_indices = np.argsort(importances)[-15:]
                top_features = [feature_names[i] for i in top_indices]
                top_importances = importances[top_indices]

                axes[1, 1].barh(range(len(top_features)), top_importances)
                axes[1, 1].set_yticks(range(len(top_features)))
                axes[1, 1].set_yticklabels(top_features)
                axes[1, 1].set_xlabel('Importance')
                axes[1, 1].set_title(f'Top Features - {best_model_name}')

        # Hide unused subplots
        for i in range(2, n_models):
            if i < len(axes[1]):
                axes[1, i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def get_best_model(self):
        """
        Return the best performing model based on accuracy
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate_models() first.")

        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_model_name]['accuracy']

        print(f"🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

        return {
            'name': best_model_name,
            'model': self.models[best_model_name],
            'accuracy': best_accuracy,
            'results': self.results[best_model_name]
        }

    def run_complete_pipeline(self, window_size=10, overlap=0.5, test_size=0.333,
                            models_to_use=None, optimize_hyperparameters=True):
        """
        Run the complete machine learning pipeline

        Args:
            window_size (int): Window size for feature engineering
            overlap (float): Window overlap ratio
            test_size (float): Test set proportion
            models_to_use (list): Models to train
            optimize_hyperparameters (bool): Whether to optimize hyperparameters

        Returns:
            dict: Complete results including best model
        """
        print("🚀 Running complete ML pipeline...")
        print("=" * 50)

        # Step 1: Feature Engineering
        self.engineer_features(window_size=window_size, overlap=overlap)

        # Step 2: Feature Analysis
        self.analyze_features()

        # Step 3: Train/Test Setup
        self.setup_train_test(test_size=test_size)

        # Step 4: Model Training
        self.train_models(models_to_use=models_to_use,
                         optimize_hyperparameters=optimize_hyperparameters)

        # Step 5: Model Evaluation
        self.evaluate_models()

        # Step 6: Get Best Model
        best_model_info = self.get_best_model()

        print("\n" + "=" * 50)
        print("🎉 Pipeline complete!")

        return {
            'feature_analysis': self.feature_analysis,
            'results': self.results,
            'best_model': best_model_info
        }

# Example usage:
"""
# Initialize framework
framework = ActivityClassificationFramework(data_path='your_preprocessed_data.csv')

# Run complete pipeline
results = framework.run_complete_pipeline(
    models_to_use=[
        'Random Forest',
        'Gradient Boosting',
        'Logistic Regression',
        'K-Nearest Neighbors'
    ],
    window_size=10,           # 10 data points per window (0.1 seconds at 100Hz)
    overlap=0.5,              # 50% overlap between windows
    test_size=0.333,            # 20% for testing
    models_to_use=['Random Forest', 'Gradient Boosting'],
    optimize_hyperparameters=True
)

# Access results
print(f"Best model accuracy: {results['best_model']['accuracy']:.4f}")
"""


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("aggregated_data_cleaned.csv")
    df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg').replace('/', '_').replace('µT', 'uT') for col in df.columns]
    df = df.rename(columns={'window_start_time': 'time'})

    # Run pipeline
    framework = ActivityClassificationFramework(df=df)
    results = framework.run_complete_pipeline(
    models_to_use=[
        'Random Forest',
        'Gradient Boosting',
        'Logistic Regression',
        'K-Nearest Neighbors'
    ],
        window_size=10,
        overlap=0.5,
        test_size=0.333,
        optimize_hyperparameters=True
    )

    print(f"✅ Best model: {results['best_model']['name']} with accuracy {results['best_model']['accuracy']:.4f}")
