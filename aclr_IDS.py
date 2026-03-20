import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch._dynamo
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import random
from imblearn.over_sampling import SMOTE
from collections import Counter

# 固定隨機種子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 動態類別權重計算器
class DynamicClassWeighting:
    def __init__(self, initial_weights=None, update_frequency=1, momentum=0.9):
        self.initial_weights = initial_weights or {}
        self.current_weights = self.initial_weights.copy()
        self.update_frequency = update_frequency
        self.momentum = momentum
        self.epoch_count = 0
        self.weight_history = []
        
    def calculate_class_weights(self, y_true, y_pred_proba, method='focal_adaptive'):
        if method == 'focal_adaptive':
            return self._focal_adaptive_weights(y_true, y_pred_proba)
        elif method == 'confidence_based':
            return self._confidence_based_weights(y_true, y_pred_proba)
        elif method == 'error_rate_based':
            return self._error_rate_based_weights(y_true, y_pred_proba)
        else:
            return self._balanced_weights(y_true)
    
    def _focal_adaptive_weights(self, y_true, y_pred_proba):
        weights = {}
        unique_classes = np.unique(y_true)
        for class_id in unique_classes:
            class_mask = (y_true == class_id)
            if class_mask.sum() == 0:
                continue
            class_probs = y_pred_proba[class_mask]
            if class_id == 1:
                confidence = class_probs
            else:
                confidence = 1 - class_probs
            avg_difficulty = 1 - np.mean(confidence)
            base_weight = 1.0 / (class_mask.sum() + 1e-6)
            difficulty_factor = 1 + avg_difficulty * 2
            weights[class_id] = base_weight * difficulty_factor
        return weights
    
    def _confidence_based_weights(self, y_true, y_pred_proba):
        weights = {}
        unique_classes = np.unique(y_true)
        for class_id in unique_classes:
            class_mask = (y_true == class_id)
            if class_mask.sum() == 0:
                continue
            class_probs = y_pred_proba[class_mask]
            if class_id == 1:
                confidence = class_probs
            else:
                confidence = 1 - class_probs
            avg_confidence = np.mean(confidence)
            base_weight = 1.0 / (class_mask.sum() + 1e-6)
            confidence_factor = 2 - avg_confidence
            weights[class_id] = base_weight * confidence_factor
        return weights
    
    def _error_rate_based_weights(self, y_true, y_pred_proba):
        weights = {}
        unique_classes = np.unique(y_true)
        for class_id in unique_classes:
            class_mask = (y_true == class_id)
            if class_mask.sum() == 0:
                continue
            class_probs = y_pred_proba[class_mask]
            class_preds = (class_probs > 0.5).astype(int)
            class_true = y_true[class_mask]
            error_rate = np.mean(class_preds != class_true)
            base_weight = 1.0 / (class_mask.sum() + 1e-6)
            error_factor = 1 + error_rate * 3
            weights[class_id] = base_weight * error_factor
        return weights
    
    def _balanced_weights(self, y_true):
        weights = {}
        unique_classes = np.unique(y_true)
        total_samples = len(y_true)
        for class_id in unique_classes:
            class_count = (y_true == class_id).sum()
            weights[class_id] = total_samples / (len(unique_classes) * class_count)
        return weights
    
    def update_weights(self, new_weights):
        if not self.current_weights:
            self.current_weights = new_weights
        else:
            for class_id in new_weights:
                if class_id in self.current_weights:
                    self.current_weights[class_id] = (
                        self.momentum * self.current_weights[class_id] + 
                        (1 - self.momentum) * new_weights[class_id]
                    )
                else:
                    self.current_weights[class_id] = new_weights[class_id]
        self.weight_history.append(self.current_weights.copy())
    
    def should_update(self):
        return self.epoch_count % self.update_frequency == 0
    
    def get_current_weights(self):
        return self.current_weights.copy()
    
    def get_weight_history(self):
        return self.weight_history

# 動態Focal Loss
class DynamicFocalLoss(nn.Module):
    def __init__(self, class_weighting, alpha=0.25, gamma=2.0, reduction='mean'):
        super(DynamicFocalLoss, self).__init__()
        self.class_weighting = class_weighting
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.class_weighting.current_weights:
            weights = torch.ones_like(targets)
            for class_id, weight in self.class_weighting.current_weights.items():
                weights[targets == class_id] = weight
            focal_loss = focal_loss * weights
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 原始Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Dataset definitions
class BotnetDataset(Dataset):
    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNNDataset(Dataset):
    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
        if len(X.shape) == 3:
            self.X = torch.FloatTensor(X)
        else:
            self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.FloatTensor(y)
        self.len = len(self.X)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        if idx >= self.len:
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {self.len}")
        return self.X[idx], self.y[idx]

# Model definitions
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
        return x.view(1, -1).size(1)
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
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

# Evaluation function
def evaluate_model(model, val_loader, device='cpu', class_weighting=None):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    
    if class_weighting:
        criterion = DynamicFocalLoss(class_weighting)
    else:
        criterion = FocalLoss()
        
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            val_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds.squeeze() == y_batch).sum().item()
            total += y_batch.size(0)
            all_targets.extend(y_batch.cpu().numpy())
            all_predictions.extend(outputs.squeeze().cpu().numpy())
    
    return val_loss / len(val_loader), correct / total, np.array(all_targets), np.array(all_predictions)

# Training function
def train_model(model, train_loader, val_loader, device='cpu', epochs=10, model_name="", 
                use_dynamic_weighting=False, weight_update_frequency=2):
    class_weighting = None
    if use_dynamic_weighting:
        class_weighting = DynamicClassWeighting(update_frequency=weight_update_frequency)
        criterion = DynamicFocalLoss(class_weighting)
    else:
        criterion = FocalLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    model.to(device)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    weight_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        all_train_targets = []
        all_train_predictions = []
        
        pbar = tqdm(train_loader, desc=f"{model_name} | Epoch {epoch+1}/{epochs}", leave=False)
        for X_batch, y_batch in pbar:
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds.squeeze() == y_batch).sum().item()
            total += y_batch.size(0)
            all_train_targets.extend(y_batch.detach().cpu().numpy())
            all_train_predictions.extend(outputs.detach().squeeze().cpu().numpy())
            pbar.set_postfix({'Loss': loss.item(), 'Acc': f"{(correct / total):.4f}"})

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        
        val_loss, val_acc, val_targets, val_predictions = evaluate_model(
            model, val_loader, device, class_weighting
        )
        
        if use_dynamic_weighting and class_weighting.should_update():
            new_weights = class_weighting.calculate_class_weights(
                val_targets, val_predictions, method='focal_adaptive'
            )
            class_weighting.update_weights(new_weights)
            class_weighting.epoch_count += 1
            weight_history.append(class_weighting.get_current_weights().copy())
            print(f"[{model_name}] Epoch {epoch+1} - 動態權重更新: {new_weights}")
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if use_dynamic_weighting and weight_history:
        plot_weight_history(weight_history, model_name)
    
    return train_losses, train_accs, val_losses, val_accs

def plot_weight_history(weight_history, model_name):
    if not weight_history:
        return
    plt.figure(figsize=(10, 6))
    epochs = range(len(weight_history))
    all_classes = set()
    for weights in weight_history:
        all_classes.update(weights.keys())
    
    for class_id in sorted(all_classes):
        weights = [w.get(class_id, 0) for w in weight_history]
        plt.plot(epochs, weights, marker='o', label=f'Class {class_id}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Class Weight')
    plt.title(f'{model_name} - Dynamic Class Weights Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name.lower()}_dynamic_weights.png')
    plt.close()

# Prediction function
def predict(model, X, device='cpu', batch_size=256):
    model.eval()
    predictions = []
    total_samples = len(X)
    
    try:
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                current_batch_size = min(batch_size, total_samples - i)
                batch_X = X[i:i + current_batch_size]
                X_tensor = torch.FloatTensor(batch_X).to(device)
                batch_pred = model(X_tensor)
                batch_pred = batch_pred.cpu().numpy()
                predictions.append(batch_pred)
                del X_tensor
                del batch_pred
                if device == 'cuda':
                    torch.cuda.empty_cache()
                if (i + current_batch_size) % (batch_size * 10) == 0:
                    progress = (i + current_batch_size) / total_samples * 100
                    print(f"預測進度: {progress:.1f}%")
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            if device == 'cuda':
                torch.cuda.empty_cache()
            print("GPU內存不足，嘗試使用更小的批處理大小...")
            return predict(model, X, device, batch_size // 2)
        raise e
    
    return np.concatenate(predictions).flatten()

# 數據增強函數
def augment_data(X, y, noise_level=0.05, mask_prob=0.1):
    X_aug = X.copy()
    y_aug = y.copy()
    mask = np.random.random(X.shape) < mask_prob
    X_aug[mask] = 0
    return X_aug, y_aug

# Main training logic
# ... [Previous imports, class definitions, and functions remain unchanged] ...

# Main training logic
def main():
    set_seed(42)
    # 定義特徵欄位
    feature_COLUMNS = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
        'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
        'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
        'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
        'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
        'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
        'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
        'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
        'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
        'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
        'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
        'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
        'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
        'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ]
    label_col = 'Label'

    # 讀取所有 CSV 檔案
    csv_files = [
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv'
    ]
    
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            dfs.append(df)
            print(f"成功載入 {file}，形狀: {df.shape}")
        except FileNotFoundError:
            print(f"檔案 {file} 不存在，請檢查路徑")
            continue
        except Exception as e:
            print(f"載入 {file} 時發生錯誤: {e}")
            continue
    
    if not dfs:
        raise ValueError("無法載入任何資料檔案，請檢查檔案路徑和格式")
    
    # 合併並立即重設索引
    df = pd.concat(dfs, ignore_index=True)
    print(f"合併後資料集形狀: {df.shape}")
    
    # Downsample the dataset to reduce memory usage
    print("Downsampling dataset to 10% of original size to reduce memory usage...")
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    print(f"Downsampled dataset shape: {df.shape}")
    
    # 保留原始索引以確保一致性
    original_index = df.index
    print(f"Downsampled index length: {len(original_index)}")

    # 移除不需要的欄位
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # 處理欄位名稱中的空格
    df.columns = df.columns.str.strip()
    feature_COLUMNS = [col.strip() for col in feature_COLUMNS]
    label_col = label_col.strip()

    # 確保所有特徵欄位存在
    missing_cols = [col for col in feature_COLUMNS + [label_col] if col not in df.columns]
    if missing_cols:
        print(f"警告：以下欄位在資料集中不存在，將被忽略：{missing_cols}")
        feature_COLUMNS = [col for col in feature_COLUMNS if col in df.columns]
    
    print("所有欄位名稱：", df.columns.tolist())
    print(df[label_col].value_counts())
    
    # 將 Label 欄位轉為二元分類：BENIGN=0，其餘=1
    df[label_col] = (df[label_col] != 'BENIGN').astype(int)
    print('二元分類後標籤分布:', df[label_col].value_counts())

    # 定義數值型和類別型特徵
    categorical_cols = ['Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
                       'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
                       'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count']
    numerical_cols = [col for col in feature_COLUMNS if col not in categorical_cols]

    print(f"Feature count: {len(numerical_cols) + len(categorical_cols)}")
    print("Numerical features:", numerical_cols)
    print("Categorical features:", categorical_cols)

    # 驗證downsampled資料
    print(f"Downsampled df shape before feature extraction: {df.shape}")

    # Label Encoding for categorical variables
    label_encoders = {}
    X_categorical = pd.DataFrame(index=original_index)
    for col in categorical_cols:
        if col in df.columns:
            label_encoders[col] = LabelEncoder()
            df[col] = df[col].astype(str).fillna('')
            values = df[col].tolist()
            if '' not in values:
                values.append('')
            label_encoders[col].fit(values)
            X_categorical[col] = label_encoders[col].transform(df[col]).astype(np.float32)
        else:
            print(f"警告：類別欄位 {col} 不存在，填充0")
            X_categorical[col] = 0  # 填充0以保持一致

    print(f"X_categorical shape after encoding: {X_categorical.shape}")

    # 數值型特徵的處理
    X_numeric = df[numerical_cols].copy().reset_index(drop=True)
    print(f"X_numeric shape before processing: {X_numeric.shape}")
    
    # 確保數值特徵與原始索引一致
    X_numeric = X_numeric.loc[original_index].apply(pd.to_numeric, errors='coerce').astype(np.float32)
    print(f"X_numeric shape after index alignment and type conversion: {X_numeric.shape}")

    # 處理異常值
    def handle_outliers(df, columns, method='iqr'):
        df_clean = df.copy()
        for col in columns:
            if col not in df_clean.columns:
                continue
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
    print(f"X_numeric shape after outlier handling: {X_numeric.shape}")

    # 處理NaN值
    X_numeric = X_numeric.fillna(0)
    print(f"X_numeric shape after NaN handling: {X_numeric.shape}")

    # 特徵縮放
    scalers = {}
    X_numeric_scaled = pd.DataFrame(index=original_index)
    for col in numerical_cols:
        if col not in X_numeric.columns:
            print(f"警告：數值欄位 {col} 不存在，填充0")
            X_numeric_scaled[col] = 0
            continue
        if X_numeric[col].skew() > 1:
            X_numeric[col] = np.log1p(X_numeric[col] - X_numeric[col].min() + 1e-6)
        scalers[col] = RobustScaler()
        X_numeric_scaled[col] = scalers[col].fit_transform(X_numeric[[col]]).ravel().astype(np.float32)

    print(f"X_numeric_scaled shape after scaling: {X_numeric_scaled.shape}")

    # 確保數值型態且無NaN
    X_numeric_scaled = X_numeric_scaled.fillna(0).astype(np.float32)
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(np.float32)

    # 驗證數據行數
    print(f"X_numeric_scaled final shape: {X_numeric_scaled.shape}")
    print(f"X_categorical final shape: {X_categorical.shape}")
    if X_numeric_scaled.shape[0] != X_categorical.shape[0]:
        raise ValueError(f"Row mismatch: X_numeric_scaled has {X_numeric_scaled.shape[0]} rows, X_categorical has {X_categorical.shape[0]} rows")

    # 特徵選擇
    def select_features(X, y, threshold=0.01):
        correlations = []
        for col in X.columns:
            if X[col].std() == 0:
                continue
            correlation = np.abs(np.corrcoef(X[col], y)[0, 1])
            if not np.isnan(correlation):
                correlations.append((col, correlation))
        # Select top N features to reduce memory usage
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features = [col for col, corr in correlations[:10]]  # Limit to top 10 features
        return selected_features

    # 修改特徵交互函數以減少記憶體使用
    def create_interaction_features(X, selected_features, max_interactions=10, chunk_size=100000):
        interaction_features = {}
        count = 0
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                if count >= max_interactions:
                    break
                feat1, feat2 = selected_features[i], selected_features[j]
                interaction_features[f'{feat1}_{feat2}_inter'] = np.zeros(X.shape[0], dtype=np.float32)
                for start in range(0, X.shape[0], chunk_size):
                    end = min(start + chunk_size, X.shape[0])
                    interaction_features[f'{feat1}_{feat2}_inter'][start:end] = (X[feat1][start:end] * X[feat2][start:end]).astype(np.float32)
                count += 1
            if count >= max_interactions:
                break
        X_inter = pd.concat([X] + [pd.Series(v, name=k, dtype=np.float32, index=X.index) for k, v in interaction_features.items()], axis=1)
        return X_inter

    # 應用特徵選擇
    selected_features = select_features(X_numeric_scaled, df[label_col])
    print(f"Selected features: {selected_features}")
    X_numeric_scaled = X_numeric_scaled[selected_features]

    # 創建特徵交互（限制數量）
    X_numeric_scaled = create_interaction_features(X_numeric_scaled, selected_features, max_interactions=5)

    # 合併特徵
    print(f"X_numeric_scaled shape before concatenation: {X_numeric_scaled.shape}")
    print(f"X_categorical shape before concatenation: {X_categorical.shape}")
    X = np.concatenate([X_numeric_scaled.values, X_categorical.values], axis=1).astype(np.float32)
    y = df[label_col].values

    # 最終的資料標準化
    final_scaler = StandardScaler()
    X = final_scaler.fit_transform(X).astype(np.float32)

    # 為CNN模型準備數據
    X_cnn_full = X.reshape(X.shape[0], 1, X.shape[1])

    print(f"CNN全量數據形狀: {X_cnn_full.shape}")

    # 保存全量數據
    print("\n正在保存全量數據...")
    try:
        np.save('X_full.npy', X)
        print("已保存 X_full.npy")
        np.save('y_full.npy', y)
        print("已保存 y_full.npy")
        np.save('X_cnn_full.npy', X_cnn_full)
        print("已保存 X_cnn_full.npy")
        print("全量數據保存完成")
    except Exception as e:
        print(f"保存全量數據時發生錯誤: {e}")
        raise e

    # 保存預處理器
    print("\n保存預處理器...")
    joblib.dump(scalers, 'feature_scalers.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(final_scaler, 'final_scaler.pkl')
    joblib.dump(selected_features, 'selected_features.pkl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # K-Fold Cross-Validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    results = []
    all_fpr = []
    all_tpr = []
    all_auc = []
    all_y_val = []
    all_pred_proba = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n===== Fold {fold + 1} =====")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 只對訓練集做資料增強
        X_train_aug, y_train_aug = augment_data(X_train, y_train)
        X_train = np.vstack([X_train, X_train_aug]).astype(np.float32)
        y_train = np.concatenate([y_train, y_train_aug]).astype(np.float32)

        # CNN資料
        X_cnn_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_cnn_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        # 創建數據加載器
        train_dataset = BotnetDataset(X_train, y_train)
        val_dataset = BotnetDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=32, drop_last=True)

        cnn_train_dataset = CNNDataset(X_cnn_train, y_train)
        cnn_val_dataset = CNNDataset(X_cnn_val, y_val)
        cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32, shuffle=True, drop_last=True)
        cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32, drop_last=True)

        # 訓練模型
        print(f"\n--- Fold {fold + 1} 訓練 ANN 模型 ---")
        ann = ANNModel(X.shape[1]).to(device)
        train_model(ann, train_loader, val_loader, device, epochs=10, model_name=f"ANN_Fold{fold+1}", 
                   use_dynamic_weighting=True, weight_update_frequency=2)
        ann_pred = predict(ann, X_val, device, batch_size=256)

        print(f"\n--- Fold {fold + 1} 訓練 CNN 模型 ---")
        cnn = CNNModel(X.shape[1]).to(device)
        train_model(cnn, cnn_train_loader, cnn_val_loader, device, epochs=10, model_name=f"CNN_Fold{fold+1}", 
                   use_dynamic_weighting=True, weight_update_frequency=2)
        cnn_pred = predict(cnn, X_cnn_val, device, batch_size=256)

        print(f"\n--- Fold {fold + 1} 訓練 RNN 模型 ---")
        rnn = RNNModel(X.shape[1]).to(device)
        train_model(rnn, train_loader, val_loader, device, epochs=10, model_name=f"RNN_Fold{fold+1}", 
                   use_dynamic_weighting=True, weight_update_frequency=2)
        rnn_pred = predict(rnn, X_val, device, batch_size=256)

        print(f"\n--- Fold {fold + 1} 訓練 LSTM 模型 ---")
        lstm = LSTMModel(X.shape[1]).to(device)
        train_model(lstm, train_loader, val_loader, device, epochs=10, model_name=f"LSTM_Fold{fold+1}", 
                   use_dynamic_weighting=True, weight_update_frequency=2)
        lstm_pred = predict(lstm, X_val, device, batch_size=256)

        meta_X = np.column_stack([ann_pred, cnn_pred, rnn_pred, lstm_pred]).astype(np.float32)
        if len(np.unique(y_val)) < 2:
            print(f"Fold {fold+1} 的驗證集只有一個類別，meta model 直接用 base model 平均。")
            final_pred_proba = np.mean(meta_X, axis=1)
            final_pred = (final_pred_proba > 0.5).astype(int)
        else:
            meta_model = LogisticRegression()
            meta_model.fit(meta_X, y_val)
            final_pred = meta_model.predict(meta_X)
            final_pred_proba = meta_model.predict_proba(meta_X)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, final_pred_proba)
        roc_auc = auc(fpr, tpr)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auc.append(roc_auc)
        all_y_val.extend(y_val)
        all_pred_proba.extend(final_pred_proba)

        acc = accuracy_score(y_val, final_pred)
        prec = precision_score(y_val, final_pred)
        rec = recall_score(y_val, final_pred)
        f1 = f1_score(y_val, final_pred)

        results.append({
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'AUC': roc_auc
        })

    results_df = pd.DataFrame(results)
    print("\nModel Performance Metrics (3-Fold with Dynamic Class Weighting):")
    print(results_df)
    print("\nAverage Performance:")
    print(results_df.mean())

    # Plot average ROC curve
    all_y_val = np.array(all_y_val)
    all_pred_proba = np.array(all_pred_proba)
    mean_fpr, mean_tpr, _ = roc_curve(all_y_val, all_pred_proba)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='red', lw=2, linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Folds (with Dynamic Class Weighting)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_folds_dynamic.png')
    plt.close()

    # Train final models
    print("\n===== Training Final Models with Dynamic Class Weighting =====")
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_final_aug, y_train_final_aug = augment_data(X_train_final, y_train_final)
    X_train_final = np.vstack([X_train_final, X_train_final_aug]).astype(np.float32)
    y_train_final = np.concatenate([y_train_final, y_train_final_aug]).astype(np.float32)

    X_cnn_train_final = X_train_final.reshape(X_train_final.shape[0], 1, X_train_final.shape[1])
    X_cnn_val_final = X_val_final.reshape(X_val_final.shape[0], 1, X_val_final.shape[1])

    train_dataset_final = BotnetDataset(X_train_final, y_train_final)
    val_dataset_final = BotnetDataset(X_val_final, y_val_final)
    train_loader_final = DataLoader(train_dataset_final, batch_size=32, shuffle=True, drop_last=True)
    val_loader_final = DataLoader(val_dataset_final, batch_size=32, drop_last=True)

    cnn_train_dataset_final = CNNDataset(X_cnn_train_final, y_train_final)
    cnn_val_dataset_final = CNNDataset(X_cnn_val_final, y_val_final)
    cnn_train_loader_final = DataLoader(cnn_train_dataset_final, batch_size=32, shuffle=True, drop_last=True)
    cnn_val_loader_final = DataLoader(cnn_val_dataset_final, batch_size=32, drop_last=True)

    print("\n--- 訓練最終 ANN 模型 ---")
    final_ann = ANNModel(X.shape[1]).to(device)
    train_model(final_ann, train_loader_final, val_loader_final, device, epochs=15, model_name="Final_ANN", 
               use_dynamic_weighting=True, weight_update_frequency=3)
    torch.save(final_ann.state_dict(), 'final_ann.pth')

    print("\n--- 訓練最終 CNN 模型 ---")
    final_cnn = CNNModel(X.shape[1]).to(device)
    train_model(final_cnn, cnn_train_loader_final, cnn_val_loader_final, device, epochs=15, model_name="Final_CNN", 
               use_dynamic_weighting=True, weight_update_frequency=3)
    torch.save(final_cnn.state_dict(), 'final_cnn.pth')

    print("\n--- 訓練最終 RNN 模型 ---")
    final_rnn = RNNModel(X.shape[1]).to(device)
    train_model(final_rnn, train_loader_final, val_loader_final, device, epochs=15, model_name="Final_RNN", 
               use_dynamic_weighting=True, weight_update_frequency=3)
    torch.save(final_rnn.state_dict(), 'final_rnn.pth')

    print("\n--- 訓練最終 LSTM 模型 ---")
    final_lstm = LSTMModel(X.shape[1]).to(device)
    train_model(final_lstm, train_loader_final, val_loader_final, device, epochs=15, model_name="Final_LSTM", 
               use_dynamic_weighting=True, weight_update_frequency=3)
    torch.save(final_lstm.state_dict(), 'final_lstm.pth')

    # Generate predictions for meta-model training
    ann_pred_train = predict(final_ann, X_train_final, device, batch_size=256)
    cnn_pred_train = predict(final_cnn, X_cnn_train_final, device, batch_size=256)
    rnn_pred_train = predict(final_rnn, X_train_final, device, batch_size=256)
    lstm_pred_train = predict(final_lstm, X_train_final, device, batch_size=256)

    # Train meta-model
    meta_X_train = np.column_stack([ann_pred_train, cnn_pred_train, rnn_pred_train, lstm_pred_train]).astype(np.float32)
    smote = SMOTE(random_state=42)
    meta_X_train_res, y_res = smote.fit_resample(meta_X_train, y_train_final)
    scale_pos_weight = (y_res == 0).sum() / (y_res == 1).sum()
    final_meta_model = XGBClassifier(
        n_estimators=100, 
        scale_pos_weight=scale_pos_weight, 
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42,
        # L1 正则化 (Lasso)
        reg_alpha=0.1,
        # L2 正则化 (Ridge)
        reg_lambda=1.0,
        # 其他正则化参数
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.1
    )
    final_meta_model.fit(meta_X_train_res, y_res)

    # Save the improved meta model
    joblib.dump(final_meta_model, 'final_meta_model.pkl')

    # Save input size for model initialization
    with open('model_config.pkl', 'wb') as f:
        pickle.dump({'input_size': X.shape[1]}, f)

    # Meta model performance on training set
    print("\n===== Meta Model Performance on Training Set =====")
    meta_train_pred = final_meta_model.predict(meta_X_train_res)
    acc = accuracy_score(y_res, meta_train_pred)
    prec = precision_score(y_res, meta_train_pred)
    rec = recall_score(y_res, meta_train_pred)
    f1 = f1_score(y_res, meta_train_pred)
    print(f"[MetaModel] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    print("\nModels and artifacts saved successfully.")
    print("動態類別權重訓練完成！")

if __name__ == "__main__":
    main()