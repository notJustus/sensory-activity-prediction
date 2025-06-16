import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class TemporalActivityDataset(Dataset):
    """
    Custom dataset for temporal activity classification
    Creates sequences from sensor data with proper windowing
    """
    
    def __init__(self, data, target, sequence_length=50, overlap=0.5):
        """
        Args:
            data (DataFrame): Sensor data
            target (Series): Activity labels
            sequence_length (int): Length of each sequence
            overlap (float): Overlap between sequences (0-1)
        """
        self.data = data
        self.target = target
        self.sequence_length = sequence_length
        self.overlap = overlap
        
        # Create sequences
        self.sequences, self.labels = self._create_sequences()
        
    def _create_sequences(self):
        """Create overlapping sequences from the data"""
        sequences = []
        labels = []
        
        step_size = int(self.sequence_length * (1 - self.overlap))
        
        for start_idx in range(0, len(self.data) - self.sequence_length + 1, step_size):
            end_idx = start_idx + self.sequence_length
            
            # Get sequence data
            seq_data = self.data.iloc[start_idx:end_idx].values
            
            # Get most common label in sequence for labeling
            seq_labels = self.target.iloc[start_idx:end_idx]
            most_common_label = seq_labels.mode().iloc[0] if not seq_labels.mode().empty else seq_labels.iloc[0]
            
            sequences.append(seq_data)
            labels.append(most_common_label)
            
        return np.array(sequences), np.array(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for activity recognition (Tuned for less overfitting)
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=7, 
                 dropout=0.5, bidirectional=True): # ✨ TUNED: Reduced hidden_size, increased dropout
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            output = hidden[-1,:,:]
        
        output = self.dropout(output)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output

class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(TCNBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=(kernel_size-1)*dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=(kernel_size-1)*dilation, dilation=dilation)
        
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, nn.ReLU(), self.dropout,
                                self.conv2, nn.ReLU(), self.dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        # Remove extra padding
        out = out[:, :, :x.size(2)]
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)

class TCNClassifier(nn.Module):
    """
    Temporal Convolutional Network for activity classification (Tuned for less overfitting)
    """
    
    def __init__(self, input_size, num_channels=[32, 64, 128], kernel_size=3, 
                 dropout=0.2, num_classes=7): # ✨ TUNED: Reduced channels, increased dropout
        super(TCNClassifier, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, 
                                 dilation_size, dropout))
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = self.global_pool(x)
        x = x.squeeze(2)
        return self.classifier(x)

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for activity recognition
    
    Rationale: Transformers excel at modeling long-range dependencies through
    self-attention mechanisms. They can capture complex temporal patterns
    and relationships between different time steps simultaneously.
    """
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=6, 
                 num_classes=7, dropout=0.1, max_seq_len=1000):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = self._generate_positional_encoding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def _generate_positional_encoding(self, max_len, d_model):
        """Generate positional encodings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(1):
            pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_enc
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        
        # Classification
        return self.classifier(x)

class DeepLearningActivityFramework:
    """
    Deep Learning framework for temporal activity classification
    """
    
    def __init__(self, data_path=None, df=None):
        """Initialize the framework"""
        if data_path:
            self.df = pd.read_csv(data_path)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Either data_path or df must be provided")
        
        # Clean column names
        self.df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg').replace('/', '_').replace('µT', 'uT') for col in self.df.columns]
        if 'window_start_time' in self.df.columns:
            self.df = self.df.rename(columns={'window_start_time': 'time'})
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
    def prepare_temporal_data(self, sequence_length=50, overlap=0.5, test_size=0.2):
        """
        Prepare data for temporal deep learning models
        
        Args:
            sequence_length (int): Length of each sequence
            overlap (float): Overlap between sequences
            test_size (float): Proportion for test set
        """
        print("🔄 Preparing temporal data...")
        
        # Identify sensor columns
        sensor_cols = [col for col in self.df.columns if col not in ['time', 'activity_type']]
        
        # Prepare features and target
        X = self.df[sensor_cols].fillna(0)
        y = self.df['activity_type']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=sensor_cols, index=X.index)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_encoded_series = pd.Series(y_encoded, index=y.index)
        
        # Split data first, then create sequences
        X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
            X_scaled_df, y_encoded_series, test_size=test_size, 
            stratify=y_encoded, random_state=42
        )
        
        # Create temporal datasets
        self.train_dataset = TemporalActivityDataset(
            X_train_df, y_train_series, sequence_length, overlap
        )
        self.test_dataset = TemporalActivityDataset(
            X_test_df, y_test_series, sequence_length, overlap
        )
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        self.input_size = X.shape[1]
        self.num_classes = len(np.unique(y_encoded))
        
        print(f"✅ Temporal data prepared!")
        print(f"   - Input size: {self.input_size}")
        print(f"   - Number of classes: {self.num_classes}")
        print(f"   - Training sequences: {len(self.train_dataset)}")
        print(f"   - Test sequences: {len(self.test_dataset)}")
        print(f"   - Sequence length: {sequence_length}")
        
    def train_model(self, model_type='LSTM', epochs=100, lr=0.001, patience=10):
        """
        Train a deep learning model
        
        Args:
            model_type (str): Type of model ('LSTM', 'TCN', 'Transformer')
            epochs (int): Maximum number of epochs
            lr (float): Learning rate
            patience (int): Early stopping patience
        """
        print(f"🚀 Training {model_type} model...")
        
        # Initialize model based on type
        if model_type == 'LSTM':
            model = LSTMClassifier(
                input_size=self.input_size,
                num_classes=self.num_classes
            )
        elif model_type == 'TCN':
            model = TCNClassifier(
                input_size=self.input_size,
                num_classes=self.num_classes
            )
        # NOTE: Transformer has been removed as per user request
        else:
            # Silently skip if it's a transformer
            if model_type == 'Transformer':
                return
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # ✨ TUNED: Added weight_decay for L2 regularization
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # (The rest of the training loop remains the same)
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.squeeze().to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in self.test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.squeeze().to(self.device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(train_loss / len(self.train_loader))
            val_losses.append(val_loss / len(self.test_loader))
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss / len(self.test_loader))
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'best_{model_type.lower()}_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'   Epoch [{epoch+1}/{epochs}]: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{model_type.lower()}_model.pth'))
        
        # Store model and training history
        self.models[model_type] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
        
        print(f"✅ {model_type} training complete! Best validation accuracy: {best_val_acc:.2f}%")
        
    def evaluate_model(self, model_type):
        """Evaluate a trained model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")
        
        model = self.models[model_type]['model']
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.squeeze().to(self.device)
                
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # Classification report
        class_names = self.label_encoder.classes_
        # ✨ FIX: Add the 'labels' parameter to handle missing classes in the test set
        report = classification_report(
            all_targets, all_predictions,
            target_names=class_names,
            labels=np.arange(len(class_names)), # Ensures all classes are included
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions, labels=np.arange(len(class_names)))
        
        self.results[model_type] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        print(f"📊 {model_type} Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        return self.results[model_type]
    
    def train_all_models(self, epochs=100, lr=0.001, patience=10):
        """Train all deep learning models"""
        models_to_train = ['LSTM', 'TCN']
        
        for model_type in models_to_train:
            print(f"\n{'='*50}")
            # Now, pass the 'patience' argument down to the train_model method
            self.train_model(model_type, epochs=epochs, lr=lr, patience=patience)
            self.evaluate_model(model_type)
        
    def plot_results(self):
        """Plot comprehensive results"""
        if not self.results:
            print("No results to plot. Train models first.")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, n_models + 1, figsize=(5*(n_models+1), 10))
        
        if n_models == 1:
            axes = axes.reshape(2, -1)
        
        class_names = self.label_encoder.classes_
        
        # Plot confusion matrices
        for i, (model_name, result) in enumerate(self.results.items()):
            sns.heatmap(
                result['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[0, i]
            )
            axes[0, i].set_title(f'{model_name}\nAcc: {result["accuracy"]:.4f}')
            axes[0, i].set_xlabel('Predicted')
            axes[0, i].set_ylabel('Actual')
        
        # Plot training histories
        for i, (model_name, model_info) in enumerate(self.models.items()):
            if i < len(axes[1]):
                ax = axes[1, i]
                epochs = range(1, len(model_info['train_accuracies']) + 1)
                ax.plot(epochs, model_info['train_accuracies'], 'b-', label='Train Acc')
                ax.plot(epochs, model_info['val_accuracies'], 'r-', label='Val Acc')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy (%)')
                ax.set_title(f'{model_name} Training History')
                ax.legend()
                ax.grid(True)
        
        # Model comparison
        if n_models > 1:
            model_names = list(self.results.keys())
            accuracies = [self.results[name]['accuracy'] for name in model_names]
            f1_scores = [self.results[name]['f1_score'] for name in model_names]
            
            ax = axes[0, -1]
            x = np.arange(len(model_names))
            width = 0.35
            
            ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
            ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_traditional_ml(self, traditional_results):
        """
        Compare deep learning results with traditional ML results
        
        Args:
            traditional_results (dict): Results from traditional ML models
        """
        print("\n" + "="*60)
        print("🔍 COMPARISON: Deep Learning vs Traditional ML")
        print("="*60)
        
        # Traditional ML results summary
        if traditional_results:
            best_traditional = max(traditional_results.items(), 
                                 key=lambda x: x[1]['accuracy'])
            print(f"\n📊 Best Traditional ML Model: {best_traditional[0]}")
            print(f"   Accuracy: {best_traditional[1]['accuracy']:.4f}")
            # ✨ FIX: Access the f1_score directly from the dictionary
            print(f"   F1-Score: {best_traditional[1]['f1_score']:.4f}")
        
        # Deep learning results summary
        if self.results:
            best_dl = max(self.results.items(), key=lambda x: x[1]['accuracy'])
            print(f"\n🧠 Best Deep Learning Model: {best_dl[0]}")
            print(f"   Accuracy: {best_dl[1]['accuracy']:.4f}")
            print(f"   F1-Score: {best_dl[1]['f1_score']:.4f}")
            
            # Improvement analysis
            if traditional_results:
                acc_improvement = best_dl[1]['accuracy'] - best_traditional[1]['accuracy']
                print(f"\n📈 Performance Improvement:")
                print(f"   Accuracy improvement: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
                
                if acc_improvement > 0.05:
                    print("   🎉 Significant improvement with deep learning!")
                elif acc_improvement > 0.01:
                    print("   ✅ Moderate improvement with deep learning")
                else:
                    print("   ⚖️ Similar performance - consider computational trade-offs")
        
        # Create comparison visualization
        self._plot_comparison(traditional_results)
    
    def _plot_comparison(self, traditional_results):
        """Plot comparison between traditional ML and DL models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        all_models = {}
        
        # Add traditional ML results
        if traditional_results:
            for model_name, result in traditional_results.items():
                all_models[f"ML: {model_name}"] = result['accuracy']
        
        # Add deep learning results
        if self.results:
            for model_name, result in self.results.items():
                all_models[f"DL: {model_name}"] = result['accuracy']
        
        if all_models:
            models = list(all_models.keys())
            accuracies = list(all_models.values())
            
            colors = ['skyblue' if 'ML:' in model else 'salmon' for model in models]
            
            bars = ax1.bar(range(len(models)), accuracies, color=colors, alpha=0.7)
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy Comparison: Traditional ML vs Deep Learning')
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{acc:.3f}', ha='center', va='bottom')
        
        # Model complexity comparison (approximate)
        complexity_info = {
            'Traditional ML': ['Low Complexity', 'Fast Training', 'Interpretable'],
            'Deep Learning': ['High Complexity', 'Longer Training', 'Better Temporal Modeling']
        }
        
        ax2.axis('off')
        ax2.set_title('Model Characteristics Comparison', fontsize=14, fontweight='bold')
        
        y_pos = 0.8
        for category, characteristics in complexity_info.items():
            ax2.text(0.1, y_pos, category, fontsize=12, fontweight='bold')
            y_pos -= 0.1
            for char in characteristics:
                ax2.text(0.15, y_pos, f"• {char}", fontsize=10)
                y_pos -= 0.08
            y_pos -= 0.05
        
        plt.tight_layout()
        plt.show()
    
    def get_model_insights(self):
        """Provide insights about the trained models"""
        print("\n" + "="*60)
        print("🧠 MODEL INSIGHTS AND RATIONALE")
        print("="*60)
        
        insights = {
            'LSTM': {
                'rationale': """
                LSTMs excel at capturing long-term temporal dependencies in sequential data.
                For activity recognition, they can model the temporal evolution of sensor
                readings and identify patterns that span multiple time steps. The bidirectional
                architecture allows the model to consider both past and future context.
                """,
                'strengths': [
                    "Excellent at modeling sequential patterns",
                    "Can handle variable-length sequences",
                    "Good at capturing long-term dependencies",
                    "Bidirectional processing provides rich context"
                ],
                'hyperparameters': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'bidirectional': True,
                    'dropout': 0.3
                }
            },
            'TCN': {
                'rationale': """
                Temporal Convolutional Networks use dilated convolutions to capture
                long-range dependencies efficiently. They offer parallelizable training
                and can model different temporal scales simultaneously through the
                dilation mechanism, making them well-suited for sensor data analysis.
                """,
                'strengths': [
                    "Efficient parallel training",
                    "Captures multi-scale temporal patterns",
                    "Lower memory requirements than RNNs",
                    "Stable gradients"
                ],
                'hyperparameters': {
                    'num_channels': [64, 128, 256],
                    'kernel_size': 3,
                    'dropout': 0.1,
                    'dilation_levels': [1, 2, 4]
                }
            },
            'Transformer': {
                'rationale': """
                Transformers use self-attention mechanisms to capture relationships between
                all time steps simultaneously. This allows them to model complex temporal
                patterns and long-range dependencies without the sequential processing
                limitations of RNNs. They excel at understanding global context in sequences.
                """,
                'strengths': [
                    "Global attention to all time steps",
                    "Captures complex temporal relationships",
                    "Parallelizable training and inference",
                    "Excellent at modeling long sequences",
                    "Position encoding preserves temporal order"
                ],
                'hyperparameters': {
                    'd_model': 128,
                    'nhead': 8,
                    'num_layers': 4,
                    'dropout': 0.1,
                    'max_seq_len': 1000
                }
            }
        }
        
        # Display insights for each trained model
        for model_type in ['LSTM', 'TCN', 'Transformer']:
            if model_type in self.models:
                print(f"\n📊 {model_type} MODEL ANALYSIS")
                print("-" * 40)
                
                insight = insights[model_type]
                
                # Display rationale
                print("🎯 Rationale:")
                print(insight['rationale'].strip())
                
                # Display strengths
                print("\n💪 Key Strengths:")
                for strength in insight['strengths']:
                    print(f"   • {strength}")
                
                # Display hyperparameters
                print("\n⚙️ Key Hyperparameters:")
                for param, value in insight['hyperparameters'].items():
                    print(f"   • {param}: {value}")
                
                # Display performance if available
                if model_type in self.results:
                    result = self.results[model_type]
                    print(f"\n📈 Performance:")
                    print(f"   • Accuracy: {result['accuracy']:.4f}")
                    print(f"   • F1-Score: {result['f1_score']:.4f}")
                    
                    # Best validation accuracy from training
                    if 'best_val_acc' in self.models[model_type]:
                        best_val = self.models[model_type]['best_val_acc']
                        print(f"   • Best Validation Accuracy: {best_val:.2f}%")
        
        # General recommendations
        print(f"\n🎯 GENERAL RECOMMENDATIONS")
        print("-" * 40)
        
        if self.results:
            # Find best performing model
            best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
            best_name, best_result = best_model
            
            print(f"🏆 Best Performing Model: {best_name}")
            print(f"   • Accuracy: {best_result['accuracy']:.4f}")
            print(f"   • F1-Score: {best_result['f1_score']:.4f}")
            
            # Model-specific recommendations
            if best_name == 'LSTM':
                print("\n💡 LSTM performed best - Consider:")
                print("   • The sequential nature of your data benefits from RNN architectures")
                print("   • Try increasing hidden_size or num_layers for more complex patterns")
                print("   • Experiment with different dropout rates for regularization")
                
            elif best_name == 'TCN':
                print("\n💡 TCN performed best - Consider:")
                print("   • Your data has multi-scale temporal patterns")
                print("   • Try deeper networks with more dilation levels")
                print("   • Experiment with different kernel sizes for various receptive fields")
                
            elif best_name == 'Transformer':
                print("\n💡 Transformer performed best - Consider:")
                print("   • Your data benefits from global attention mechanisms")
                print("   • Try increasing d_model or num_layers for more capacity")
                print("   • Experiment with different numbers of attention heads")
        
        # Data-specific insights
        print(f"\n📊 DATA-SPECIFIC INSIGHTS")
        print("-" * 40)
        print(f"   • Input Features: {self.input_size}")
        print(f"   • Number of Activity Classes: {self.num_classes}")
        print(f"   • Activity Types: {', '.join(self.label_encoder.classes_)}")
        
        if hasattr(self, 'train_dataset'):
            print(f"   • Training Sequences: {len(self.train_dataset)}")
            print(f"   • Test Sequences: {len(self.test_dataset)}")
            print(f"   • Sequence Length: {self.train_dataset.sequence_length}")
            print(f"   • Sequence Overlap: {self.train_dataset.overlap}")
        
        # Computational considerations
        print(f"\n⚡ COMPUTATIONAL CONSIDERATIONS")
        print("-" * 40)
        print("   • LSTM: Moderate training time, sequential processing")
        print("   • TCN: Fast training, parallel processing, efficient inference")
        print("   • Transformer: Slower training, high memory usage, excellent performance")
        print(f"   • Device Used: {self.device}")
        
        # Future improvements
        print(f"\n🚀 POTENTIAL IMPROVEMENTS")
        print("-" * 40)
        print("   • Ensemble methods combining multiple architectures")
        print("   • Hyperparameter tuning using grid/random search")
        print("   • Data augmentation techniques for temporal data")
        print("   • Feature engineering from raw sensor data")
        print("   • Cross-validation for more robust evaluation")
        print("   • Transfer learning from pre-trained models")
        
        print("\n" + "="*60)

def main():
    """Main function to run the deep learning framework with real data."""
    
    # 1. Load and clean the data from your CSV file
    # ✨ FIX: Updated filename to use the combined dataset
    print("🔄 Loading and cleaning data from aggregated_data_cleaned_combined.csv...")
    df = pd.read_csv("aggregated_data_cleaned_combined.csv")
    df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg').replace('/', '_').replace('µT', 'uT') for col in df.columns]
    df = df.rename(columns={'window_start_time': 'time'})
    print("✅ Data loaded successfully.")

    # 2. Initialize the framework with the loaded data
    framework = DeepLearningActivityFramework(df=df)

    # 3. Prepare the data for temporal modeling
    # This creates overlapping sequences for training and testing.
    framework.prepare_temporal_data(sequence_length=100, overlap=0.5, test_size=0.2)

    # 4. Train all deep learning models
    # These parameters are suitable for a real training session.
    framework.train_all_models(epochs=50, lr=0.001, patience=10)

    # 5. Plot the results from the deep learning models
    framework.plot_results()

    # 6. (Optional) Define and compare with traditional ML model results
    # You can replace these with your actual results if you have them.
    traditional_results = {
        'RandomForest': {'accuracy': 0.85, 'f1_score': 0.84},
        'SVM': {'accuracy': 0.81, 'f1_score': 0.80}
    }
    framework.compare_with_traditional_ml(traditional_results)

    # 7. Provide final insights and rationale for the trained models
    framework.get_model_insights()
    
    print("\n🎉 Workflow complete!")

if __name__ == "__main__":
    main()