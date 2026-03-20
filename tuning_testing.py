
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import time
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'

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

# Model definitions (unchanged)
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

class BotnetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model_with_config(model, train_loader, val_loader, device, epochs, config):
    """使用給定配置訓練模型，新增每epoch的指標追蹤"""
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    # 新增：儲存每epoch的指標
    metrics = {'accuracy': [], 'f1': [], 'loss': []}
    
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
        
        # 計算驗證集指標
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets)
        val_preds_binary = (val_preds > 0.5).astype(int)
        acc = accuracy_score(val_targets, val_preds_binary)
        f1 = f1_score(val_targets, val_preds_binary) if len(np.unique(val_targets)) > 1 else 0.0
        
        # 儲存指標
        metrics['accuracy'].append(acc)
        metrics['f1'].append(f1)
        metrics['loss'].append(val_loss)
        
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
    
    return {
        'val_loss': best_val_loss,
        'accuracy': acc,
        'f1_score': f1,
        'predictions': val_preds,
        'metrics': metrics  # 新增：返回每epoch的指標
    }

def preprocess_data_aclr_style(df, scalers=None, label_encoders=None, selected_features=None, final_scaler=None, fit=True):
    """與aclr.py完全相同的前處理邏輯"""
    df = df.drop('id', axis=1, errors='ignore')
    columns_to_drop = ['attack_cat']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    all_columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'labels']
    categorical_cols = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    label_col = 'labels'
    numerical_cols = [col for col in all_columns if col not in categorical_cols + [label_col]]
    
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
    
    X_numeric = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    
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
    
    if fit or scalers is None:
        scalers = {}
        X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
        for col in numerical_cols:
            if X_numeric[col].skew() > 1:
                X_numeric[col] = np.log1p(X_numeric[col] - X_numeric[col].min() + 1e-6)
            scalers[col] = RobustScaler()
            X_numeric_scaled[col] = scalers[col].fit_transform(X_numeric[[col]]).ravel()
    else:
        X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
        for col in numerical_cols:
            if X_numeric[col].skew() > 1:
                X_numeric[col] = np.log1p(X_numeric[col] - X_numeric[col].min() + 1e-6)
            X_numeric_scaled[col] = scalers[col].transform(X_numeric[[col]]).ravel()
    
    def select_features(X, y, threshold=0.01):
        correlations = []
        for col in X.columns:
            correlation = np.abs(np.corrcoef(X[col], y)[0, 1])
            correlations.append((col, correlation))
        selected_features = [col for col, corr in correlations if corr > threshold]
        return selected_features
    
    def create_interaction_features(X, selected_features):
        interaction_features = {}
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                feat1, feat2 = selected_features[i], selected_features[j]
                interaction_features[f'{feat1}_{feat2}_inter'] = X[feat1] * X[feat2]
        if interaction_features:
            X_inter = pd.concat([X] + [pd.Series(v, name=k) for k, v in interaction_features.items()], axis=1)
        else:
            X_inter = X
        return X_inter
    
    if fit or selected_features is None:
        y_label = df['labels'].values if 'labels' in df.columns else np.zeros(len(df))
        selected_features = select_features(X_numeric_scaled, y_label)
    
    X_numeric_scaled = X_numeric_scaled[selected_features]
    X_numeric_scaled = create_interaction_features(X_numeric_scaled, selected_features)
    
    X = np.concatenate([X_numeric_scaled, X_categorical.values], axis=1).astype(np.float32)
    
    if fit or final_scaler is None:
        final_scaler = StandardScaler()
        X = final_scaler.fit_transform(X)
    else:
        X = final_scaler.transform(X)
    
    X_cnn = X.reshape(X.shape[0], 1, X.shape[1])
    
    y_label = df['labels'].values if 'labels' in df.columns else np.zeros(len(df))
    if y_label.dtype == object or not np.issubdtype(y_label.dtype, np.number):
        y_label = np.array([0 if str(x).lower() == 'normal' else 1 for x in y_label], dtype=np.float32)
    else:
        y_label = y_label.astype(np.float32)
    return X, X_cnn, y_label, scalers, label_encoders, selected_features, final_scaler

def train_model_with_config(model, train_loader, val_loader, device, epochs, config):
    """Train model with selective fine-tuning and differential learning rates."""
    criterion = nn.BCELoss()
    
    # 定義差別學習率
    learning_rate = config['learning_rate']
    param_groups = []
    
    # 根據模型類型凍結早期層並設置差別學習率
    if isinstance(model, FineTunableANNModel):
        # 凍結第一個全連接層
        for i, layer in enumerate(model.model):
            if isinstance(layer, nn.Linear) and i == 0:
                for param in layer.parameters():
                    param.requires_grad = False
        # 分層學習率：全連接層使用較大學習率，其他層（如果未凍結）使用較小學習率
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'model.0' not in n and p.requires_grad], 'lr': learning_rate},  # 後期層
            {'params': [p for n, p in model.named_parameters() if 'model.0' in n and p.requires_grad], 'lr': learning_rate * 0.1}  # 早期層（如果未凍結）
        ]
    
    elif isinstance(model, FineTunableCNNModel):
        # 凍結第一個卷積層
        for param in model.conv_layers[0].parameters():
            param.requires_grad = False
        # 分層學習率：全連接層和後期卷積層使用較大學習率
        param_groups = [
            {'params': model.fc.parameters(), 'lr': learning_rate},  # 全連接層
            {'params': [p for p in model.conv_layers[1:].parameters() if p.requires_grad], 'lr': learning_rate * 0.5},  # 後期卷積層
            {'params': [p for p in model.conv_layers[0].parameters() if p.requires_grad], 'lr': learning_rate * 0.1}  # 早期卷積層（如果未凍結）
        ]
    
    elif isinstance(model, FineTunableLSTMModel):
        # 凍結第一層 LSTM
        for name, param in model.lstm.named_parameters():
            if 'l0' in name:  # 第一層 LSTM
                param.requires_grad = False
        # 分層學習率：全連接層使用較大學習率，LSTM 層使用較小學習率
        param_groups = [
            {'params': model.fc.parameters(), 'lr': learning_rate},  # 全連接層
            {'params': [p for p in model.lstm.parameters() if p.requires_grad], 'lr': learning_rate * 0.1}  # LSTM 層
        ]
    
    elif isinstance(model, FineTunableRNNModel):
        # 凍結第一層 RNN
        for name, param in model.rnn.named_parameters():
            if 'l0' in name:  # 第一層 RNN
                param.requires_grad = False
        # 分層學習率：全連接層使用較大學習率，RNN 層使用較小學習率
        param_groups = [
            {'params': model.fc.parameters(), 'lr': learning_rate},  # 全連接層
            {'params': [p for p in model.rnn.parameters() if p.requires_grad], 'lr': learning_rate * 0.1}  # RNN 層
        ]
    
    # 初始化優化器
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    
    # 進度條
    pbar = tqdm(range(epochs), desc=f"Training {model.__class__.__name__}", leave=False)
    
    for epoch in pbar:
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
        
        train_loss /= len(train_loader)
        
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
        
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets)
        val_preds_binary = (val_preds > 0.5).astype(int)
        accuracy = accuracy_score(val_targets, val_preds_binary)
        f1 = f1_score(val_targets, val_preds_binary)
        
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(accuracy)
        metrics['val_f1'].append(f1)
        
        pbar.set_description(f"Training {model.__class__.__name__} (Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}, Val F1: {f1:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            pbar.set_description(f"Training {model.__class__.__name__} (Early Stopping at Epoch {epoch+1})")
            break
    
    return {
        'val_loss': best_val_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': val_preds,
        'metrics': metrics
    }

def objective_xgb(trial, meta_X_train, y_train, meta_X_val, y_val):
    """Optuna objective function for XGBoost with per-epoch metrics."""
    param = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'error'],  # <-- 移到這裡
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
    eval_set = [(meta_X_train, y_train), (meta_X_val, y_val)]
    # 兼容舊版 xgboost，移除所有可能不支援的參數
    model.fit(
        meta_X_train, y_train,
        eval_set=eval_set,
        verbose=False)
    
    # Collect per-epoch (boosting round) metrics
    evals_result = model.evals_result()
    n_rounds = len(evals_result['validation_0']['logloss'])
    metrics = {
        'epoch': list(range(1, n_rounds + 1)),
        'train_loss': evals_result['validation_0']['logloss'],
        'val_loss': evals_result['validation_1']['logloss'],
        'val_accuracy': [1 - x for x in evals_result['validation_1']['error']],
        'val_f1': []
    }
    
    # Calculate F1-score for each boosting round
    for i in range(n_rounds):
        # 兼容不同版本的 xgboost
        try:
            # 新版 xgboost (>=1.6) 支援 iteration_range
            pred = model.predict(meta_X_val, iteration_range=(0, i + 1))
        except TypeError:
            try:
                # 舊版支援 ntree_limit
                pred = model.predict(meta_X_val, ntree_limit=i + 1)
            except TypeError:
                # 如果都不支援，直接用完整模型預測
                pred = model.predict(meta_X_val)
        f1 = f1_score(y_val, pred)
        metrics['val_f1'].append(f1)
    
    pred = model.predict(meta_X_val)
    return f1_score(y_val, pred), metrics

def print_metrics_table(metrics_dict, model_name):
    """Print metrics in a tabular format."""
    print(f"\n{model_name} Epoch-wise Metrics:")
    print("-" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val Accuracy':>12} | {'Val F1':>12}")
    print("-" * 60)
    for i in range(len(metrics_dict['epoch'])):
        print(f"{metrics_dict['epoch'][i]:>6} | {metrics_dict['train_loss'][i]:>12.4f} | "
              f"{metrics_dict['val_loss'][i]:>12.4f} | {metrics_dict['val_accuracy'][i]:>12.4f} | "
              f"{metrics_dict['val_f1'][i]:>12.4f}")
    print("-" * 60)

def plot_metrics(metrics_dict, model_name, output_dir='plots'):
    """Plot training and validation metrics for a model."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_dict['epoch'], metrics_dict['train_loss'], label='Training Loss', marker='o')
    plt.plot(metrics_dict['epoch'], metrics_dict['val_loss'], label='Validation Loss', marker='s')
    plt.xlabel('Epoch', fontname='Times New Roman', fontsize=12)
    plt.ylabel('Loss', fontname='Times New Roman', fontsize=12)
    plt.title(f'{model_name} Loss over Epochs', fontname='Times New Roman', fontsize=14)
    plt.legend(prop={'family': 'Times New Roman'})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_loss.png'))
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_dict['epoch'], metrics_dict['val_accuracy'], label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch', fontname='Times New Roman', fontsize=12)
    plt.ylabel('Accuracy', fontname='Times New Roman', fontsize=12)
    plt.title(f'{model_name} Validation Accuracy over Epochs', fontname='Times New Roman', fontsize=14)
    plt.legend(prop={'family': 'Times New Roman'})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_accuracy.png'))
    plt.close()
    
    # Plot F1-Score
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_dict['epoch'], metrics_dict['val_f1'], label='Validation F1-Score', marker='s')
    plt.xlabel('Epoch', fontname='Times New Roman', fontsize=12)
    plt.ylabel('F1-Score', fontname='Times New Roman', fontsize=12)
    plt.title(f'{model_name} Validation F1-Score over Epochs', fontname='Times New Roman', fontsize=14)
    plt.legend(prop={'family': 'Times New Roman'})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_f1_score.png'))
    plt.close()

def run_cross_dataset_fine_tuning(input_size, categorical_cols, numerical_cols, scalers, 
                                 label_encoders, selected_features, final_scaler, device):
    process_start_time = time.time()  # Start timing the entire process
    print("\n" + "="*50)
    print("NB15 Single Dataset Fine Tuning")
    print("="*50)
    
    # Load KDD ACLR best parameters (as provided)
    kdd_ann_params = {
        'hidden_sizes': [128, 64],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32
    }
    kdd_cnn_params = {
        'num_filters': [64, 32],
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
    
    # Load NB15 dataset
    try:
        print("Loading NB15 training dataset...")
        t0 = time.time()
        nb15_df = pd.read_csv('mqttiot_to_kdd.csv')
        X_nb15, X_nb15_cnn, y_nb15, _, _, _, _ = preprocess_data_aclr_style(
            nb15_df, scalers, label_encoders, selected_features, final_scaler, fit=False
        )
        print(f"Data loading and preprocessing time: {time.time() - t0:.2f} seconds")
        if y_nb15 is None:
            print("NB15 dataset has no labels, cannot fine-tune")
            return
        print(f"NB15 dataset: {X_nb15.shape[0]} samples")
    except FileNotFoundError:
        print("Cannot find nb15_kdd_train.csv")
        return
    except Exception as e:
        print(f"Error processing nb15_kdd_train.csv: {e}")
        return

    # 切分訓練集與驗證集
    t0 = time.time()
    X_train, X_val, y_train, y_val = train_test_split(
        X_nb15, y_nb15, test_size=0.2, random_state=42, stratify=y_nb15)

    # 對訓練集進行 SMOTE 過採樣
    from imblearn.over_sampling import SMOTE  # 新增：匯入 SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_cnn_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_cnn_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    print(f"資料切分與 SMOTE 時間: {time.time() - t0:.2f} 秒")

    # Dictionary to store metrics for all models
    all_metrics = {}

    # 1. Train ANN base model with KDD ACLR best parameters
    print("\nTraining ANN base model with KDD ACLR best parameters...")
    t0 = time.time()
    ann_model = FineTunableANNModel(input_size, kdd_ann_params['hidden_sizes'], kdd_ann_params['dropout_rate'], kdd_ann_params['learning_rate']).to(device)
    train_dataset = BotnetDataset(X_train, np.array(y_train))
    val_dataset = BotnetDataset(X_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=kdd_ann_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kdd_ann_params['batch_size'])
    config = {'learning_rate': kdd_ann_params['learning_rate']}
    ann_results = train_model_with_config(ann_model, train_loader, val_loader, device, 10, config)
    all_metrics['ANN'] = ann_results['metrics']
    print_metrics_table(ann_results['metrics'], 'ANN')
    print(f"ANN base model training time: {time.time() - t0:.2f} seconds")

    # 2. Train CNN base model with KDD ACLR best parameters
    print("\nTraining CNN base model with KDD ACLR best parameters...")
    t0 = time.time()
    cnn_model = FineTunableCNNModel(input_size, kdd_cnn_params['num_filters'], kdd_cnn_params['kernel_sizes'], kdd_cnn_params['dropout_rate'], kdd_cnn_params['learning_rate']).to(device)
    train_dataset = BotnetDataset(X_cnn_train, np.array(y_train))
    val_dataset = BotnetDataset(X_cnn_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=kdd_cnn_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kdd_cnn_params['batch_size'])
    config = {'learning_rate': kdd_cnn_params['learning_rate']}
    cnn_results = train_model_with_config(cnn_model, train_loader, val_loader, device, 10, config)
    all_metrics['CNN'] = cnn_results['metrics']
    print_metrics_table(cnn_results['metrics'], 'CNN')
    print(f"CNN base model training time: {time.time() - t0:.2f} seconds")

    # 3. Train LSTM base model with KDD ACLR best parameters
    print("\nTraining LSTM base model with KDD ACLR best parameters...")
    t0 = time.time()
    lstm_model = FineTunableLSTMModel(input_size, kdd_lstm_params['hidden_size'], kdd_lstm_params['num_layers'], kdd_lstm_params['dropout_rate'], kdd_lstm_params['learning_rate']).to(device)
    train_dataset = BotnetDataset(X_train, np.array(y_train))
    val_dataset = BotnetDataset(X_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=kdd_lstm_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kdd_lstm_params['batch_size'])
    config = {'learning_rate': kdd_lstm_params['learning_rate']}
    lstm_results = train_model_with_config(lstm_model, train_loader, val_loader, device, 10, config)
    all_metrics['LSTM'] = lstm_results['metrics']
    print_metrics_table(lstm_results['metrics'], 'LSTM')
    print(f"LSTM base model training time: {time.time() - t0:.2f} seconds")

    # 4. Train RNN base model with KDD ACLR best parameters
    print("\nTraining RNN base model with KDD ACLR best parameters...")
    t0 = time.time()
    rnn_model = FineTunableRNNModel(input_size, kdd_rnn_params['hidden_size'], kdd_rnn_params['num_layers'], kdd_rnn_params['dropout_rate'], kdd_rnn_params['learning_rate']).to(device)
    train_dataset = BotnetDataset(X_train, np.array(y_train))
    val_dataset = BotnetDataset(X_val, np.array(y_val))
    train_loader = DataLoader(train_dataset, batch_size=kdd_rnn_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kdd_rnn_params['batch_size'])
    config = {'learning_rate': kdd_rnn_params['learning_rate']}
    rnn_results = train_model_with_config(rnn_model, train_loader, val_loader, device, 10, config)
    all_metrics['RNN'] = rnn_results['metrics']
    print_metrics_table(rnn_results['metrics'], 'RNN')
    print(f"RNN base model training time: {time.time() - t0:.2f} seconds")

    # Generate plots for base models
    for model_name in ['ANN', 'CNN', 'LSTM', 'RNN']:
        plot_metrics(all_metrics[model_name], model_name)

    # 5. Generate meta features
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
    
    t0 = time.time()
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
    print(f"Meta feature generation time: {time.time() - t0:.2f} seconds")

    # 6. Fine-tune XGBoost meta model with Optuna
    print("\nSearching for optimal XGBoost meta model parameters with Optuna...")
    t0 = time.time()
    study = optuna.create_study(direction='maximize')
    best_f1 = -1
    best_metrics = None
    
    def wrapped_objective(trial):
        nonlocal best_f1, best_metrics
        f1, metrics = objective_xgb(trial, meta_X_train, y_train, meta_X_val, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = metrics
        return f1
    
    study.optimize(wrapped_objective, n_trials=10)
    print(f"Optuna meta model tuning time: {time.time() - t0:.2f} seconds")
    print("Best parameters:", study.best_params)
    print("Best F1-score:", study.best_value)
    
    # Print and plot meta model metrics
    all_metrics['XGBoost'] = best_metrics
    print_metrics_table(best_metrics, 'XGBoost')
    plot_metrics(best_metrics, 'XGBoost')
    
    # Train final meta model with best parameters
    t0 = time.time()
    best_param = study.best_params
    # 確保 eval_metric 也加到 best_param
    best_param['eval_metric'] = ['logloss', 'error']
    meta_model = xgb.XGBClassifier(**best_param, use_label_encoder=False, verbosity=0)
    meta_model.fit(np.vstack([meta_X_train, meta_X_val]), np.concatenate([y_train, y_val]))
    print(f"Final meta model training time: {time.time() - t0:.2f} seconds")

    # 7. Evaluate meta model on validation set
    t0 = time.time()
    meta_val_pred_proba = meta_model.predict_proba(meta_X_val)[:, 1]
    meta_val_pred = (meta_val_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_val, meta_val_pred, labels=[0,1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if cm.shape == (1,1):
            if np.unique(y_val)[0] == 0:
                tn = cm[0,0]
            else:
                tp = cm[0,0]
        print("[Warning] Validation set contains only one class, confusion matrix incomplete.")
    
    acc = accuracy_score(y_val, meta_val_pred)
    prec = precision_score(y_val, meta_val_pred)
    rec = recall_score(y_val, meta_val_pred)
    f1 = f1_score(y_val, meta_val_pred)
    DR = tp / (tp + fn) if (tp + fn) > 0 else 0
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Plot confusion matrix
    cm_tp_fn_fp_tn = np.array([[tp, fn], [fp, tn]])
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm_tp_fn_fp_tn, annot=False, fmt='g', cmap='Greens', cbar=True,
                     xticklabels=['Predicted Positive', 'Predicted Negative'],
                     yticklabels=['Actual Positive', 'Actual Negative'])
    labels = [['TP', 'FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.5, f"{labels[i][j]}\n{cm_tp_fn_fp_tn[i, j]}",
                    color='black', ha='center', va='center', fontsize=13, fontweight='bold', family='Times New Roman')
    plt.title('Confusion Matrix (TP/FN/FP/TN)', fontname='Times New Roman')
    plt.ylabel('True Label', fontname='Times New Roman')
    plt.xlabel('Predicted Label', fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('confusion_matrix_NB15_meta_TP_FN_FP_TN.png')
    plt.close()
    
    print(f"NB15 validation set inference time: {time.time() - t0:.2f} seconds")
    print("\nNB15 Validation Set Meta Model Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Detection Rate (DR): {DR:.4f}")
    print(f"  False Positive Rate (FPR): {FPR:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    # Print total process time
    total_time = time.time() - process_start_time
    print(f"\nTotal fine-tuning process time: {total_time:.2f} seconds")

    # Save parameters
    pickle.dump(kdd_ann_params, open('final_ann_params.pkl', 'wb'))
    pickle.dump(kdd_cnn_params, open('final_cnn_params.pkl', 'wb'))
    pickle.dump(kdd_lstm_params, open('final_lstm_params.pkl', 'wb'))
    pickle.dump(kdd_rnn_params, open('final_rnn_params.pkl', 'wb'))
    pickle.dump(best_param, open('final_meta_model_params.pkl', 'wb'))
    pickle.dump(meta_model, open('final_meta_model.pkl', 'wb'))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load saved artifacts
    try:
        print("Loading preprocessors...")
        scalers = joblib.load('feature_scalers.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        selected_features = joblib.load('selected_features.pkl')
        final_scaler = joblib.load('final_scaler.pkl')
        print("Loading model configuration...")
        with open('model_config.pkl', 'rb') as f:
            model_config = pickle.load(f)
        print(f"Model configuration: {model_config}")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure aclr.py has been run to train the models")
        return
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Please check if all necessary files exist")
        return

    input_size = model_config['input_size']
    categorical_cols = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    numerical_cols = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 
                      'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
                      'num_shells', 'num_access_files', 'num_outbound_cmds', 'count', 'srv_count', 
                      'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
                      'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                      'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
                      'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
                      'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

    run_cross_dataset_fine_tuning(input_size, categorical_cols, numerical_cols, scalers, 
                                 label_encoders, selected_features, final_scaler, device)

if __name__ == "__main__":
    main()