import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import optuna
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

# Fine Tuning Configuration
FINE_TUNE_CONFIG = {
    'n_trials': 10,  # Optuna試驗次數
    'epochs_per_trial': 10,  # 每次試驗的訓練輪數
    'batch_size': 32,
    'learning_rate_range': (1e-5, 1e-2),
    'dropout_range': (0.1, 0.5),
    'hidden_size_range': (32, 256),
    'num_layers_range': (2, 5)
}

# Model definitions (same as in aclr.py)
class FineTunableANNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=None, dropout_rate=0.2, learning_rate=0.001):
        super(FineTunableANNModel, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)

class FineTunableCNNModel(nn.Module):
    def __init__(self, input_length, num_filters=None, kernel_sizes=None, dropout_rate=0.2, learning_rate=0.001):
        super(FineTunableCNNModel, self).__init__()
        if num_filters is None:
            num_filters = [64, 32]
        if kernel_sizes is None:
            kernel_sizes = [3, 3]
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        in_channels = 1
        for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            self.conv_layers.append(nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2))
            self.pool_layers.append(nn.MaxPool1d(2))
            self.batch_norms.append(nn.BatchNorm1d(filters))
            in_channels = filters
        
        # 計算展平後的大小
        self.flatten_size = self._get_flatten_size(input_length, num_filters, kernel_sizes)
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.learning_rate = learning_rate
        
    def _get_flatten_size(self, input_length, num_filters, kernel_sizes):
        x = torch.randn(1, 1, input_length)
        for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            x = self.pool_layers[i](x)
        return x.view(1, -1).size(1)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            x = self.pool_layers[i](x)
            x = self.batch_norms[i](x)
        
        x = x.view(x.size(0), -1)
        return self.fc(x)

class FineTunableLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout_rate=0.2, learning_rate=0.001):
        super(FineTunableLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.learning_rate = learning_rate
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.sigmoid(self.fc(x))

# 原始模型定義（保持向後兼容）
class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

class CNNModel(nn.Module):
    def __init__(self, input_length):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten_size = self._get_flatten_size(input_length)
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def _get_flatten_size(self, input_length):
        x = torch.randn(1, 1, input_length)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        flatten_size = x.view(1, -1).size(1)
        print(f"Calculated flatten_size: {flatten_size} for input_length: {input_length}")
        return flatten_size
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class RNNModel(nn.Module):
    def __init__(self, input_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, 64, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        return self.sigmoid(self.fc(x))

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.sigmoid(self.fc(x))

# 新增 FineTunableRNNModel
class FineTunableRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout_rate=0.2, learning_rate=0.001):
        super(FineTunableRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu', dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.learning_rate = learning_rate
    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.sigmoid(self.fc(x))

# Fine Tuning Functions
class BotnetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model_with_config(model, train_loader, val_loader, device, epochs, config):
    """使用給定配置訓練模型"""
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    # 添加進度條
    pbar = tqdm(range(epochs), desc=f"訓練 {model.__class__.__name__}", leave=False)
    
    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # 更新進度條描述
        pbar.set_description(f"訓練 {model.__class__.__name__} (Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f})")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            pbar.set_description(f"訓練 {model.__class__.__name__} (Early Stopping at Epoch {epoch+1})")
            break
    
    # Calculate final metrics
    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets)
    val_preds_binary = (val_preds > 0.5).astype(int)
    
    accuracy = accuracy_score(val_targets, val_preds_binary)
    f1 = f1_score(val_targets, val_preds_binary)
    
    return {
        'val_loss': best_val_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': val_preds
    }

def objective_ann(trial, X_train, y_train, X_val, y_val, device, input_size):
    """Optuna目標函數 - ANN模型"""
    n_layers = trial.suggest_int('n_layers', 2, 4)
    hidden_sizes = []
    for i in range(n_layers):
        hidden_sizes.append(trial.suggest_int(f'hidden_size_{i}', 32, 256))
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    model = FineTunableANNModel(input_size, hidden_sizes, dropout_rate, learning_rate).to(device)
    try:
        model.load_state_dict(torch.load('final_ann.pth', map_location=device))
        print("僅載入 final_ann.pth 權重（不重新訓練）")
    except Exception as e:
        raise FileNotFoundError("[錯誤] 找不到 final_ann.pth，請先執行微調產生final_ann.pth後再進行Optuna微調！")
    train_dataset = BotnetDataset(X_train, np.array(y_train))
    val_dataset = BotnetDataset(X_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    config = {'learning_rate': learning_rate}
    results = train_model_with_config(model, train_loader, val_loader, device, FINE_TUNE_CONFIG['epochs_per_trial'], config)
    return results['f1_score']

def objective_cnn(trial, X_train, y_train, X_val, y_val, device, input_size):
    """Optuna目標函數 - CNN模型"""
    n_conv_layers = trial.suggest_int('n_conv_layers', 2, 4)
    num_filters = []
    kernel_sizes = []
    for i in range(n_conv_layers):
        num_filters.append(trial.suggest_int(f'num_filters_{i}', 32, 128))
        kernel_sizes.append(trial.suggest_int(f'kernel_size_{i}', 3, 7))
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    model = FineTunableCNNModel(input_size, num_filters, kernel_sizes, dropout_rate, learning_rate).to(device)
    try:
        model.load_state_dict(torch.load('final_cnn.pth', map_location=device))
        print("僅載入 final_cnn.pth 權重（不重新訓練）")
    except Exception as e:
        raise FileNotFoundError("[錯誤] 找不到 final_cnn.pth，請先執行微調產生final_cnn.pth後再進行Optuna微調！")
    train_dataset = BotnetDataset(X_train, np.array(y_train))
    val_dataset = BotnetDataset(X_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    config = {'learning_rate': learning_rate}
    results = train_model_with_config(model, train_loader, val_loader, device, FINE_TUNE_CONFIG['epochs_per_trial'], config)
    return results['f1_score']

def objective_lstm(trial, X_train, y_train, X_val, y_val, device, input_size):
    """Optuna目標函數 - LSTM模型"""
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    model = FineTunableLSTMModel(input_size, hidden_size, num_layers, dropout_rate, learning_rate).to(device)
    try:
        model.load_state_dict(torch.load('final_lstm.pth', map_location=device))
        print("僅載入 final_lstm.pth 權重（不重新訓練）")
    except Exception as e:
        raise FileNotFoundError("[錯誤] 找不到 final_lstm.pth，請先執行微調產生final_lstm.pth後再進行Optuna微調！")
    train_dataset = BotnetDataset(X_train, np.array(y_train))
    val_dataset = BotnetDataset(X_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    config = {'learning_rate': learning_rate}
    results = train_model_with_config(model, train_loader, val_loader, device, FINE_TUNE_CONFIG['epochs_per_trial'], config)
    return results['f1_score']

def objective_rnn(trial, X_train, y_train, X_val, y_val, device, input_size):
    """Optuna目標函數 - RNN模型"""
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    model = FineTunableRNNModel(input_size, hidden_size, num_layers, dropout_rate, learning_rate).to(device)
    try:
        model.load_state_dict(torch.load('final_rnn.pth', map_location=device))
        print("僅載入 final_rnn.pth 權重（不重新訓練）")
    except Exception as e:
        raise FileNotFoundError("[錯誤] 找不到 final_rnn.pth，請先執行微調產生final_rnn.pth後再進行Optuna微調！")
    train_dataset = BotnetDataset(X_train, np.array(y_train))
    val_dataset = BotnetDataset(X_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    config = {'learning_rate': learning_rate}
    results = train_model_with_config(model, train_loader, val_loader, device, FINE_TUNE_CONFIG['epochs_per_trial'], config)
    return results['f1_score']

def fine_tune_models(X_train, y_train, X_val, y_val, device, input_size):
    """對所有模型進行fine tuning"""
    print("開始Fine Tuning...")
    
    # 創建Optuna研究對象
    study_ann = optuna.create_study(direction='maximize')
    study_cnn = optuna.create_study(direction='maximize')
    study_lstm = optuna.create_study(direction='maximize')
    study_rnn = optuna.create_study(direction='maximize')
    
    # 定義目標函數
    def ann_objective(trial):
        return objective_ann(trial, X_train, y_train, X_val, y_val, device, input_size)
    
    def cnn_objective(trial):
        return objective_cnn(trial, X_train, y_train, X_val, y_val, device, input_size)
    
    def lstm_objective(trial):
        return objective_lstm(trial, X_train, y_train, X_val, y_val, device, input_size)
    
    def rnn_objective(trial):
        return objective_rnn(trial, X_train, y_train, X_val, y_val, device, input_size)
    
    # 執行優化
    print("Fine tuning ANN...")
    study_ann.optimize(ann_objective, n_trials=FINE_TUNE_CONFIG['n_trials'])
    
    print("Fine tuning CNN...")
    study_cnn.optimize(cnn_objective, n_trials=FINE_TUNE_CONFIG['n_trials'])
    
    print("Fine tuning LSTM...")
    study_lstm.optimize(lstm_objective, n_trials=FINE_TUNE_CONFIG['n_trials'])
    
    print("Fine tuning RNN...")
    study_rnn.optimize(rnn_objective, n_trials=FINE_TUNE_CONFIG['n_trials'])
    
    # 獲取最佳參數
    best_params = {
        'ann': study_ann.best_params,
        'cnn': study_cnn.best_params,
        'lstm': study_lstm.best_params,
        'rnn': study_rnn.best_params
    }
    
    best_scores = {
        'ann': study_ann.best_value,
        'cnn': study_cnn.best_value,
        'lstm': study_lstm.best_value,
        'rnn': study_rnn.best_value
    }
    
    print("\n最佳參數:")
    for model_name, params in best_params.items():
        print(f"{model_name.upper()}: {params}")
        print(f"最佳F1分數: {best_scores[model_name]:.4f}")
    
    # 儲存最佳參數（檔名統一final_開頭）
    pickle.dump(best_params['ann'], open('final_ann_params.pkl', 'wb'))
    pickle.dump(best_params['cnn'], open('final_cnn_params.pkl', 'wb'))
    pickle.dump(best_params['lstm'], open('final_lstm_params.pkl', 'wb'))
    pickle.dump(best_params['rnn'], open('final_rnn_params.pkl', 'wb'))
    
    return best_params, best_scores

# Prediction function
def predict(model, X, device='cpu', batch_size=256):
    model.eval()
    predictions = []
    total_samples = len(X)
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            current_batch_size = min(batch_size, total_samples - i)
            batch_X = X[i:i + current_batch_size]
            X_tensor = torch.FloatTensor(batch_X).to(device)
            batch_pred = model(X_tensor)
            predictions.append(batch_pred.cpu().numpy())
            del X_tensor
            if device == 'cuda':
                torch.cuda.empty_cache()
    return np.concatenate(predictions).flatten()

# === 新增：aclr.py 資料前處理 ===
def preprocess_data_aclr_style(df, scalers=None, label_encoders=None, selected_features=None, final_scaler=None, fit=True):
    """與aclr.py完全相同的前處理邏輯"""
    df = df.drop('id', axis=1, errors='ignore')  # 安全刪除 id 欄位
    
    # Remove unnecessary columns
    columns_to_drop = ['attack_cat']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    # Define feature columns (與aclr.py相同)
    categorical_cols = ['proto', 'service', 'state']
    numerical_cols = df.columns.drop(categorical_cols + ['label']).tolist()
    
    # Label Encoding for categorical variables (與aclr.py相同)
    if fit or label_encoders is None:
        label_encoders = {}
        X_categorical = pd.DataFrame()
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            values = df[col].fillna('').astype(str).tolist()
            if '' not in values:
                values.append('')
            label_encoders[col].fit(values)
            X_categorical[col] = label_encoders[col].transform(df[col].fillna(''))
    else:
        X_categorical = pd.DataFrame()
        for col in categorical_cols:
            valid_values = set(label_encoders[col].classes_)
            df[col] = df[col].fillna('').apply(lambda x: x if x in valid_values else '')
            X_categorical[col] = label_encoders[col].transform(df[col].fillna(''))
    
    # 數值型特徵的處理 (與aclr.py相同)
    X_numeric = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    
    # 處理異常值 (與aclr.py相同)
    def handle_outliers(df, columns, method='iqr'):
        df_clean = df.copy()
        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            elif method == 'zscore':
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                df_clean[col] = df_clean[col].clip(mean - 3*std, mean + 3*std)
        return df_clean
    
    X_numeric = handle_outliers(X_numeric, numerical_cols, method='iqr')
    
    # 特徵縮放 (與aclr.py相同)
    if fit or scalers is None:
        scalers = {}
        X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
        for col in numerical_cols:
            # 對偏態分佈的特徵進行對數轉換
            if X_numeric[col].skew() > 1:
                X_numeric[col] = np.log1p(X_numeric[col] - X_numeric[col].min() + 1e-6)
            
            # 使用 RobustScaler 處理異常值
            scalers[col] = RobustScaler()
            X_numeric_scaled[col] = scalers[col].fit_transform(X_numeric[[col]]).ravel()
    else:
        X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
        for col in numerical_cols:
            # 對偏態分佈的特徵進行對數轉換
            if X_numeric[col].skew() > 1:
                X_numeric[col] = np.log1p(X_numeric[col] - X_numeric[col].min() + 1e-6)
            
            X_numeric_scaled[col] = scalers[col].transform(X_numeric[[col]]).ravel()
    
    # 特徵選擇 (與aclr.py相同)
    def select_features(X, y, threshold=0.01):
        correlations = []
        for col in X.columns:
            correlation = np.abs(np.corrcoef(X[col], y)[0, 1])
            correlations.append((col, correlation))
        selected_features = [col for col, corr in correlations if corr > threshold]
        return selected_features
    
    # 特徵交互 (與aclr.py相同)
    def create_interaction_features(X, selected_features):
        # X 必須是 DataFrame，selected_features 是欄位名list
        interaction_features = {}
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                feat1, feat2 = selected_features[i], selected_features[j]
                # 只做同一樣本的交互，不做全對全
                interaction_features[f'{feat1}_{feat2}_inter'] = X[feat1] * X[feat2]
        # 合併所有交互特徵
        if interaction_features:
            X_inter = pd.concat([X] + [pd.Series(v, name=k) for k, v in interaction_features.items()], axis=1)
        else:
            X_inter = X
        return X_inter
    
    # 應用特徵選擇
    if fit or selected_features is None:
        y_label = df['label'].values if 'label' in df.columns else np.zeros(len(df))
        selected_features = select_features(X_numeric_scaled, y_label)
    
    X_numeric_scaled = X_numeric_scaled[selected_features]
    
    # 創建特徵交互
    X_numeric_scaled = create_interaction_features(X_numeric_scaled, selected_features)
    
    # 合併特徵
    X = np.concatenate([X_numeric_scaled, X_categorical.values], axis=1).astype(np.float32)
    
    # 最終標準化 (與aclr.py相同)
    if fit or final_scaler is None:
        final_scaler = StandardScaler()
        X = final_scaler.fit_transform(X)
    else:
        X = final_scaler.transform(X)
    
    # CNN reshape
    X_cnn = X.reshape(X.shape[0], 1, X.shape[1])
    
    # 標籤
    y_label = df['label'].values if 'label' in df.columns else np.zeros(len(df))
    
    return X, X_cnn, y_label, scalers, label_encoders, selected_features, final_scaler

# === 刪除舊 preprocess_data ===
# def preprocess_data(df, numerical_cols, categorical_cols, scalers, label_encoders, selected_features, final_scaler):
#     # Remove unnecessary columns
#     columns_to_drop = ['attack_cat']
#     df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

#     # Label Encoding for categorical variables (same as aclr_KDD.py)
#     X_categorical = pd.DataFrame()
#     for col in categorical_cols:
#         if col in df.columns:
#             valid_values = set(label_encoders[col].classes_)
#             df[col] = df[col].fillna('').apply(lambda x: x if x in valid_values else '')
#             X_categorical[col] = label_encoders[col].transform(df[col].fillna(''))
#         else:
#             X_categorical[col] = np.zeros(len(df))

#     # Process numerical columns (same as aclr_KDD.py)
#     X_numeric = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    
#     # Handle outliers (IQR method) - same as aclr_KDD.py
#     def handle_outliers(df, columns, method='iqr'):
#         df_clean = df.copy()
#         for col in columns:
#             if method == 'iqr':
#                 Q1 = df_clean[col].quantile(0.25)
#                 Q3 = df_clean[col].quantile(0.75)
#                 IQR = Q3 - Q1
#                 lower_bound = Q1 - 1.5 * IQR
#                 upper_bound = Q3 + 1.5 * IQR
#                 df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
#             elif method == 'zscore':
#                 mean = df_clean[col].mean()
#                 std = df_clean[col].std()
#                 df_clean[col] = df_clean[col].clip(mean - 3*std, mean + 3*std)
#         return df_clean

#     X_numeric = handle_outliers(X_numeric, numerical_cols, method='iqr')

#     # Apply log transformation and scaling (same as aclr_KDD.py)
#     X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
#     for col in numerical_cols:
#         # 對偏態分佈的特徵進行對數轉換
#         if X_numeric[col].skew() > 1:
#             X_numeric[col] = np.log1p(X_numeric[col] - X_numeric[col].min() + 1e-6)
#         X_numeric_scaled[col] = scalers[col].transform(X_numeric[[col]]).ravel()

#     # 確保數值型態且無NaN (same as aclr_KDD.py)
#     X_numeric_scaled = X_numeric_scaled.apply(pd.to_numeric, errors='coerce').fillna(0)

#     # Select features (same as aclr_KDD.py)
#     X_numeric_scaled = X_numeric_scaled[selected_features]

#     # Create interaction features (same as aclr_KDD.py)
#     interaction_features = {}
#     for i in range(len(selected_features)):
#         for j in range(i + 1, len(selected_features)):
#             feat1, feat2 = selected_features[i], selected_features[j]
#             interaction_features[f'{feat1}_{feat2}_inter'] = X_numeric_scaled[feat1] * X_numeric_scaled[feat2]
#     X_numeric_scaled = pd.concat([X_numeric_scaled] + [pd.Series(v, name=k) for k, v in interaction_features.items()], axis=1)

#     # Combine features (same as aclr_KDD.py)
#     X = np.concatenate([X_numeric_scaled.values, X_categorical.values], axis=1).astype(np.float32)

#     # Final standardization (same as aclr_KDD.py)
#     X = final_scaler.transform(X)

#     # Reshape for CNN
#     X_cnn = X.reshape(X.shape[0], 1, X.shape[1])
    
#     # 處理標籤：將 labels 轉換為二元分類 (same as aclr_KDD.py)
#     if 'labels' in df.columns:
#         # 檢查原始標籤分布
#         print("原始標籤分布:")
#         print(df['labels'].value_counts())
        
#         # 將 normal 設為 0，其他設為 1
#         y = (df['labels'] != 'normal').astype(np.float32)
        
#         # 檢查轉換後的標籤分布
#         print("轉換後標籤分布:")
#         print(f"0 (normal): {np.sum(y==0)}")
#         print(f"1 (attack): {np.sum(y==1)}")
#         print(f"比例: 0={np.mean(y==0):.3f}, 1={np.mean(y==1):.3f}")
#     else:
#         y = None

#     return X, X_cnn, y

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # Load saved artifacts
    try:
        print("正在加載預處理器...")
        scalers = joblib.load('feature_scalers.pkl')
        print("已加載 feature_scalers.pkl")
        
        print("正在加載標籤編碼器...")
        label_encoders = joblib.load('label_encoders.pkl')
        print("已加載 label_encoders.pkl")
        
        print("正在加載選擇的特徵...")
        selected_features = joblib.load('selected_features.pkl')
        print("已加載 selected_features.pkl")
        
        print("正在加載最終標準化器...")
        final_scaler = joblib.load('final_scaler.pkl')
        print("已加載 final_scaler.pkl")
        
        print("正在加載模型配置...")
        try:
            with open('model_config.pkl', 'rb') as f:
                model_config = pickle.load(f)
            print("已加載 model_config.pkl")
            print(f"模型配置: {model_config}")
        except Exception as e:
            print(f"加載模型配置時發生錯誤: {e}")
            print("嘗試從測試數據推斷輸入大小...")
            X_test = np.load('X_test.npy')
            model_config = {'input_size': X_test.shape[1]}
            print(f"推斷的輸入大小: {model_config['input_size']}")
    except FileNotFoundError as e:
        print(f"錯誤：找不到文件 {e}")
        print("請確保已經運行 aclr.py 訓練模型")
        return
    except Exception as e:
        print(f"加載文件時發生錯誤：{e}")
        print("請檢查所有必要的文件是否存在")
        return

    input_size = model_config['input_size']
    categorical_cols = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    numerical_cols = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

    # 只執行跨資料集微調
    run_cross_dataset_fine_tuning(input_size, categorical_cols, numerical_cols, scalers, label_encoders, selected_features, final_scaler, device)

def run_cross_dataset_fine_tuning(input_size, categorical_cols, numerical_cols, scalers, 
                                 label_encoders, selected_features, final_scaler, device):
    """NB15單一資料集微調（直接用KDD ACLR最佳參數，不再Optuna搜尋）"""
    print("\n" + "="*50)
    print("NB15 單一資料集 Fine Tuning")
    print("="*50)
    
    # ====== 請在這裡填入KDD ACLR最佳參數 ======
    kdd_ann_params = {
        'hidden_sizes': [128, 64],  # 範例，請用你KDD最佳hidden_sizes
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32
    }
    kdd_cnn_params = {
        'num_filters': [64, 32],  # 範例
        'kernel_sizes': [3, 3],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32
    }
    kdd_lstm_params = {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32
    }
    kdd_rnn_params = {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32
    }
    # ======================================
    
    # === 新增：強制載入 NB15 訓練時的前處理器 ===
    import joblib
    scalers = joblib.load('feature_scalers.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    selected_features = joblib.load('selected_features.pkl')
    final_scaler = joblib.load('final_scaler.pkl')
    # ======================================

    try:
        print("正在加載 NB15 訓練資料集...")
        nb15_df = pd.read_csv('NSL_KDD_to_UNSW (1).csv')
        X_nb15, X_nb15_cnn, y_nb15, _, _, _, _ = preprocess_data_aclr_style(
            nb15_df, scalers, label_encoders, selected_features, final_scaler, fit=False
        )
        if y_nb15 is None:
            print("NB15 資料集無標籤，無法微調")
            return
        print(f"NB15 資料集: {X_nb15.shape[0]} 樣本")
    except FileNotFoundError:
        print("找不到 NSL_KDD_to_UNSW (1).csv")
        return
    except Exception as e:
        print(f"處理 NSL_KDD_to_UNSW (1).csv 時發生錯誤: {e}")
        return

    # 分割訓練/驗證集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_nb15, y_nb15, test_size=0.2, random_state=42, stratify=y_nb15)
    X_cnn_train, X_cnn_val = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]), X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

    # 1. 直接用KDD ACLR最佳參數訓練四個base model
    print("\n直接用KDD ACLR最佳參數訓練ANN base model...")
    ann_model = FineTunableANNModel(input_size, kdd_ann_params['hidden_sizes'], kdd_ann_params['dropout_rate'], kdd_ann_params['learning_rate']).to(device)
    train_dataset = BotnetDataset(X_train, np.array(y_train))
    val_dataset = BotnetDataset(X_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=kdd_ann_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kdd_ann_params['batch_size'])
    config = {'learning_rate': kdd_ann_params['learning_rate']}
    train_model_with_config(ann_model, train_loader, val_loader, device, 10, config)
    # 不保存微調後權重

    print("\n直接用KDD ACLR最佳參數訓練CNN base model...")
    cnn_model = FineTunableCNNModel(input_size, kdd_cnn_params['num_filters'], kdd_cnn_params['kernel_sizes'], kdd_cnn_params['dropout_rate'], kdd_cnn_params['learning_rate']).to(device)
    train_dataset = BotnetDataset(X_cnn_train, np.array(y_train))
    val_dataset = BotnetDataset(X_cnn_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=kdd_cnn_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kdd_cnn_params['batch_size'])
    config = {'learning_rate': kdd_cnn_params['learning_rate']}
    train_model_with_config(cnn_model, train_loader, val_loader, device, 10, config)
    # 不保存微調後權重

    print("\n直接用KDD ACLR最佳參數訓練LSTM base model...")
    lstm_model = FineTunableLSTMModel(input_size, kdd_lstm_params['hidden_size'], kdd_lstm_params['num_layers'], kdd_lstm_params['dropout_rate'], kdd_lstm_params['learning_rate']).to(device)
    train_dataset = BotnetDataset(X_train, np.array(y_train))
    val_dataset = BotnetDataset(X_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=kdd_lstm_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kdd_lstm_params['batch_size'])
    config = {'learning_rate': kdd_lstm_params['learning_rate']}
    train_model_with_config(lstm_model, train_loader, val_loader, device, 10, config)
    # 不保存微調後權重

    print("\n直接用KDD ACLR最佳參數訓練RNN base model...")
    rnn_model = FineTunableRNNModel(input_size, kdd_rnn_params['hidden_size'], kdd_rnn_params['num_layers'], kdd_rnn_params['dropout_rate'], kdd_rnn_params['learning_rate']).to(device)
    train_dataset = BotnetDataset(X_train, np.array(y_train))
    val_dataset = BotnetDataset(X_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=kdd_rnn_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kdd_rnn_params['batch_size'])
    config = {'learning_rate': kdd_rnn_params['learning_rate']}
    train_model_with_config(rnn_model, train_loader, val_loader, device, 10, config)
    # 不保存微調後權重

    # 2. 用最佳base model預測組成meta feature
    def get_pred(model, X, is_cnn=False):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = X[i:i+256]
                if is_cnn and len(xb.shape) == 2:
                    xb = xb.reshape(xb.shape[0], 1, xb.shape[1])
                xb = torch.FloatTensor(xb).to(device)
                out = model(xb)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds).flatten()
    
    ann_pred_train = get_pred(ann_model, X_train)
    cnn_pred_train = get_pred(cnn_model, X_cnn_train, is_cnn=True)
    lstm_pred_train = get_pred(lstm_model, X_train)
    rnn_pred_train = get_pred(rnn_model, X_train)
    meta_X_train = np.column_stack([ann_pred_train, cnn_pred_train, lstm_pred_train, rnn_pred_train])
    
    ann_pred_val = get_pred(ann_model, X_val)
    cnn_pred_val = get_pred(cnn_model, X_cnn_val, is_cnn=True)
    lstm_pred_val = get_pred(lstm_model, X_val)
    rnn_pred_val = get_pred(rnn_model, X_val)
    meta_X_val = np.column_stack([ann_pred_val, cnn_pred_val, lstm_pred_val, rnn_pred_val])

    # 3. Optuna 微調 meta model（XGBoost）
    import xgboost as xgb
    def objective_xgb(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42
        }
        model = xgb.XGBClassifier(**param, use_label_encoder=False, verbosity=0)
        model.fit(meta_X_train, y_train, eval_set=[(meta_X_val, y_val)], verbose=False)
        pred = model.predict(meta_X_val)
        return f1_score(y_val, pred)
    
    print("\nOptuna搜尋XGBoost meta model最佳參數...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_xgb, n_trials=10)
    print("最佳參數：", study.best_params)
    print("最佳F1：", study.best_value)
    best_param = study.best_params
    meta_model = xgb.XGBClassifier(**best_param, use_label_encoder=False, verbosity=0)
    meta_model.fit(np.vstack([meta_X_train, meta_X_val]), np.concatenate([y_train, y_val]))

    # 4. NB15 驗證集混淆矩陣與指標
    meta_val_pred_proba = meta_model.predict_proba(meta_X_val)[:, 1]
    meta_val_pred = (meta_val_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_val, meta_val_pred)
    acc = accuracy_score(y_val, meta_val_pred)
    prec = precision_score(y_val, meta_val_pred)
    rec = recall_score(y_val, meta_val_pred)
    f1 = f1_score(y_val, meta_val_pred)
    fpr, tpr, _ = roc_curve(y_val, meta_val_pred_proba)
    auc_score = auc(fpr, tpr)
    
    print("\nNB15 驗證集 Meta Model 結果:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    # === seaborn.heatmap 混淆矩陣繪圖 ===
    tn, fp, fn, tp = cm.ravel()
    cm_tp_fn_fp_tn = np.array([[tp, fn], [fp, tn]])
    plt.figure(figsize=(6,5))
    ax = sns.heatmap(cm_tp_fn_fp_tn, annot=False, fmt='g', cmap='Greens', cbar=True,
                     xticklabels=['Predicted Positive', 'Predicted Negative'],
                     yticklabels=['Actual Positive', 'Actual Negative'])
    labels = [['TP', 'FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.5, f"{labels[i][j]}\n{cm_tp_fn_fp_tn[i, j]}",
                    color='black', ha='center', va='center', fontsize=13, fontweight='bold')
    plt.title('Confusion Matrix (TP/FN/FP/TN)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_NB15_meta_TP_FN_FP_TN.png')
    plt.close()

    # 儲存最佳參數（檔名統一final_開頭）
    pickle.dump(kdd_ann_params, open('final_ann_params.pkl', 'wb'))
    pickle.dump(kdd_cnn_params, open('final_cnn_params.pkl', 'wb'))
    pickle.dump(kdd_lstm_params, open('final_lstm_params.pkl', 'wb'))
    pickle.dump(kdd_rnn_params, open('final_rnn_params.pkl', 'wb'))
    
    # 儲存meta model參數
    pickle.dump(best_param, open('final_meta_model_params.pkl', 'wb'))
    pickle.dump(meta_model, open('final_meta_model.pkl', 'wb'))
    #pickle.dump(meta_model, open('final_meta_model.pkl', 'wb'))  # 保存NB15 meta model

if __name__ == "__main__":
    main()