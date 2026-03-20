import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch._dynamo
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import CalibratedClassifierCV
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import random
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
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
        self.layer_norm = nn.LayerNorm(64)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.layer_norm(x)
        return self.sigmoid(self.fc(x))

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.layer_norm = nn.LayerNorm(64)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.layer_norm(x)
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

# Random Masking 工具
def _apply_random_masking(X_batch: torch.Tensor, mask_prob: float, mask_value: float = 0.0) -> torch.Tensor:
    """
    訓練時的隨機元素遮罩 (element-wise)。
    """
    if mask_prob <= 0:
        return X_batch
    element_mask = (torch.rand_like(X_batch) < mask_prob)
    return X_batch.masked_fill(element_mask, mask_value)

# Training function
def train_model(model, train_loader, val_loader, device='cpu', epochs=10, model_name="", 
                use_dynamic_weighting=False, weight_update_frequency=2, l1_lambda=0.0, l2_lambda=0.0,
                enable_three_stage_random_masking=False, mask_value=0.0):
    """
    訓練模型，支持動態類別權重、Early Stopping 和三階段 Random Masking 策略
    
    Args:
        use_dynamic_weighting: 是否使用動態類別權重
        weight_update_frequency: 權重更新頻率
        l1_lambda: L1正則化係數
        l2_lambda: L2正則化係數
        enable_three_stage_random_masking: 是否啟用三階段 Random Masking 策略
        mask_value: 被遮罩的填充值
    """
    class_weighting = None
    if use_dynamic_weighting:
        class_weighting = DynamicClassWeighting(update_frequency=weight_update_frequency)
        criterion = DynamicFocalLoss(class_weighting)
    else:
        criterion = FocalLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=l2_lambda)
    model.to(device)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
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
        
        # 計算當前 epoch 的三階段 Random Masking 機率
        if enable_three_stage_random_masking:
            if epoch < 10:  # 前期 (1-10): 固定 5%
                current_mask_prob = 0.05
                stage_name = "前期固定"
            elif epoch < 30:  # 中期 (11-30): 每 epoch 增加 0.5%，從 5% 到 15%
                # 中期起始於 epoch=10 (0-based epoch=9? 等)，但調整為 epoch <10 是0-9 (1-10)
                # epoch 從0開始，所以 epoch 0-9: 前期, 10-29: 中期, 30+: 後期
                # 中期20 epochs，從5%增加到15%，每epoch增加0.005 (10%/20=0.005)
                increase_per_epoch = 0.005
                epochs_in_middle = epoch - 9  # epoch=10: epochs_in_middle=1, mask=0.05 + 0.005
                current_mask_prob = 0.05 + (epochs_in_middle * increase_per_epoch)
                stage_name = "中期漸增"
            else:  # 後期 (31+): 固定 15%
                current_mask_prob = 0.15
                stage_name = "後期固定"
            
            if epoch == 0 or epoch == 9 or epoch == 29 or epoch == epochs - 1:
                print(f"[{model_name}] Epoch {epoch+1} - {stage_name} Random Masking: 機率={current_mask_prob:.3f}")
        
        pbar = tqdm(train_loader, desc=f"{model_name} | Epoch {epoch+1}/{epochs}", leave=False)
        for X_batch, y_batch in pbar:
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 應用 Random Masking（如果啟用）
            if enable_three_stage_random_masking and current_mask_prob > 0:
                # 確保支援 2D (ANN/RNN/LSTM) 和 3D (CNN)
                if X_batch.dim() == 3:  # CNN: (batch, channels, features)
                    # 只遮罩特徵維度
                    mask = (torch.rand_like(X_batch[:, 0, :]) < current_mask_prob).unsqueeze(1)
                    X_batch = X_batch.masked_fill(mask, mask_value)
                else:  # 2D: (batch, features)
                    X_batch = _apply_random_masking(X_batch, current_mask_prob, mask_value)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                loss = loss + l1_lambda * l1_norm
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"[{model_name}] Epoch {epoch+1} - 新的最佳驗證損失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"[{model_name}] Epoch {epoch+1} - 驗證損失未改善 ({patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print(f"[{model_name}] Early Stopping 觸發！驗證損失 {patience} 個 epoch 未改善")
            print(f"[{model_name}] 恢復最佳模型狀態（驗證損失: {best_val_loss:.4f}）")
            model.load_state_dict(best_model_state)
            break
        
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
    
    if best_model_state is not None and patience_counter < patience:
        print(f"[{model_name}] 訓練完成，使用最佳模型狀態（驗證損失: {best_val_loss:.4f}）")
        model.load_state_dict(best_model_state)
    
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
def augment_data(X, y, noise_level=0.05):
    """數據增強：添加高斯噪聲"""
    X_aug = X.copy()
    y_aug = y.copy()
    noise = np.random.normal(0, noise_level, X.shape)
    X_aug = X_aug + noise
    return X_aug, y_aug

# New Per-Domain Calibrator Class
class PerDomainCalibrator:
    def __init__(self, method='sigmoid'):
        self.method = method
        self.calibrators = {}
        self.domains = []
    
    def fit(self, X, y, domains):
        """
        Fit a calibrator for each domain.
        X: model predictions (probabilities)
        y: true labels
        domains: domain identifier for each sample (e.g., protocol_type)
        """
        self.domains = np.unique(domains)
        for domain in self.domains:
            mask = (domains == domain)
            if mask.sum() < 10:  # Skip domains with too few samples
                print(f"Skipping calibration for domain {domain}: too few samples ({mask.sum()})")
                continue
            X_domain = X[mask].reshape(-1, 1)
            y_domain = y[mask]
            calibrator = CalibratedClassifierCV(
                base_estimator=None,  # No base estimator needed for Platt Scaling
                method=self.method,
                cv='prefit'  # Use prefit since we're calibrating probabilities directly
            )
            calibrator.fit(X_domain, y_domain)
            self.calibrators[domain] = calibrator
            print(f"Calibrator trained for domain {domain} with {mask.sum()} samples")
    
    def predict_proba(self, X, domains):
        """
        Calibrate probabilities for each sample based on its domain.
        Returns calibrated probabilities.
        """
        calibrated_probs = np.zeros_like(X)
        for domain in self.domains:
            mask = (domains == domain)
            if domain not in self.calibrators:
                print(f"No calibrator for domain {domain}, using raw probabilities")
                calibrated_probs[mask] = X[mask]
            else:
                X_domain = X[mask].reshape(-1, 1)
                calibrated_probs[mask] = self.calibrators[domain].predict_proba(X_domain)[:, 1]
        return calibrated_probs
    
    def save(self, filename):
        """Save calibrators to a file."""
        joblib.dump(self.calibrators, filename)
        print(f"Calibrators saved to {filename}")
    
    def load(self, filename):
        """Load calibrators from a file."""
        self.calibrators = joblib.load(filename)
        self.domains = list(self.calibrators.keys())
        print(f"Calibrators loaded from {filename}")

def main():
    set_seed(42)
    # Read and preprocess data
    training = pd.read_csv('kdd_test.csv')
    testing = pd.read_csv('kdd_train.csv')
    df = pd.concat([training, testing]).reset_index(drop=True)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    print("所有欄位名稱：", df.columns.tolist())
    print(df['labels'].value_counts())
    columns_to_drop = ['attack_cat']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    df['labels'] = (df['labels'] != 'normal').astype(int)
    print('二元分類後標籤分布:', df['labels'].value_counts())

    selected_feature_names = ['src_bytes', 'dst_bytes', 'duration', 'same_srv_rate', 'protocol_type', 'flag']
    categorical_cols = ['protocol_type', 'flag']
    label_col = 'labels'
    numerical_cols = ['src_bytes', 'dst_bytes', 'duration', 'same_srv_rate']

    print(f"Feature count: {len(numerical_cols) + len(categorical_cols)}")
    print("Numerical features:", numerical_cols)
    print("Categorical features:", categorical_cols)

    # Store protocol_type for per-domain calibration
    protocol_types = df['protocol_type'].astype(str).fillna('').values

    label_encoders = {}
    X_categorical = pd.DataFrame()
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = df[col].astype(str).fillna('')
        values = df[col].tolist()
        if '' not in values:
            values.append('')
        label_encoders[col].fit(values)
        X_categorical[col] = label_encoders[col].transform(df[col])

    X_numeric = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

    for col in ['src_bytes', 'dst_bytes', 'duration']:
        if col in X_numeric.columns:
            X_numeric[col] = np.log1p(X_numeric[col])

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

    scalers = {}
    X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
    for col in numerical_cols:
        scalers[col] = RobustScaler()
        X_numeric_scaled[col] = scalers[col].fit_transform(X_numeric[[col]]).ravel()

    X_numeric_scaled = X_numeric_scaled.apply(pd.to_numeric, errors='coerce').fillna(0)
    df['labels'] = pd.to_numeric(df['labels'], errors='coerce').fillna(0)

    X = np.concatenate([X_numeric_scaled.values, X_categorical.values], axis=1).astype(np.float32)
    final_scaler = StandardScaler()
    X = final_scaler.fit_transform(X)

    X_aug, y_aug = augment_data(X, df['labels'].values)
    X = np.vstack([X, X_aug])
    y = np.concatenate([df['labels'].values, y_aug])
    # Augment protocol_types accordingly
    protocol_types = np.concatenate([protocol_types, protocol_types])

    print("\n保存預處理器...")
    joblib.dump(scalers, 'feature_scalers.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(final_scaler, 'final_scaler.pkl')
    joblib.dump(selected_feature_names, 'selected_features.pkl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # Apply SMOTE and ensure protocol_types alignment
    smote = SMOTE(random_state=42)
    # Create a DataFrame to combine X, y, and protocol_types with sequential indices
    feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
    temp_df = pd.DataFrame(X, columns=feature_columns)
    temp_df['label'] = y
    temp_df['protocol_type'] = protocol_types
    # Reset index to ensure sequential indices (0 to len(X)-1)
    temp_df = temp_df.reset_index(drop=True)
    # Apply SMOTE to features and labels
    X_resampled, y_resampled = smote.fit_resample(temp_df[feature_columns], temp_df['label'])
    # Assign protocol_types to resampled data
    # Create a mapping of original indices to protocol_types
    original_indices = np.arange(len(X))  # Sequential indices: 0 to len(X)-1
    protocol_type_map = dict(zip(original_indices, protocol_types))
    # Initialize protocol_types_resampled
    protocol_types_resampled = []
    # For each resampled sample, find the nearest original sample's index
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X)  # Fit on original features
    for i in range(len(X_resampled)):
        if i < len(X):  # Original sample
            protocol_types_resampled.append(protocol_type_map[i])
        else:  # Synthetic sample
            # Find the nearest original sample
            distances, indices = nn.kneighbors([X_resampled[i]], n_neighbors=1)
            nearest_idx = indices[0][0]  # Index of the nearest original sample
            protocol_types_resampled.append(protocol_type_map[nearest_idx])
    protocol_types_resampled = np.array(protocol_types_resampled)

    # Verify lengths
    print(f"X_resampled shape: {X_resampled.shape}")
    print(f"y_resampled length: {len(y_resampled)}")
    print(f"protocol_types_resampled length: {len(protocol_types_resampled)}")
    if len(X_resampled) != len(y_resampled) or len(X_resampled) != len(protocol_types_resampled):
        raise ValueError(f"Inconsistent sample sizes: X_resampled ({len(X_resampled)}), y_resampled ({len(y_resampled)}), protocol_types_resampled ({len(protocol_types_resampled)})")

    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )
    # Split protocol_types accordingly
    _, protocol_types_val = train_test_split(
        protocol_types_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    train_dataset = BotnetDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_dataset = BotnetDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, drop_last=True)

    X_cnn_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_cnn_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    cnn_train_dataset = CNNDataset(X_cnn_train, y_train)
    cnn_val_dataset = CNNDataset(X_cnn_val, y_val)
    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32, shuffle=True, drop_last=True)
    cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32, drop_last=True)

    print("\n===== Training Final Models on All Data (with Dynamic Class Weighting) =====")
    train_dataset_full = BotnetDataset(X_resampled, y_resampled)
    train_loader_full = DataLoader(train_dataset_full, batch_size=32, shuffle=True, drop_last=True)
    val_dataset_full = BotnetDataset(X_resampled, y_resampled)
    val_loader_full = DataLoader(val_dataset_full, batch_size=32, drop_last=True)

    print("\n--- 訓練最終 ANN 模型 ---")
    final_ann = ANNModel(X.shape[1]).to(device)
    train_model(final_ann, train_loader_full, val_loader_full, device, epochs=50, model_name="Final_ANN",
                use_dynamic_weighting=True, weight_update_frequency=3, l1_lambda=0.0, l2_lambda=0.001,
                enable_three_stage_random_masking=True, mask_value=0.0)
    torch.save(final_ann.state_dict(), 'final_ann.pth')

    print("\n--- 訓練最終 CNN 模型 ---")
    final_cnn = CNNModel(X.shape[1]).to(device)
    train_model(final_cnn, train_loader_full, val_loader_full, device, epochs=50, model_name="Final_CNN",
                use_dynamic_weighting=True, weight_update_frequency=3, l1_lambda=0.0, l2_lambda=0.001,
                enable_three_stage_random_masking=True, mask_value=0.0)
    torch.save(final_cnn.state_dict(), 'final_cnn.pth')

    print("\n--- 訓練最終 RNN 模型 ---")
    final_rnn = RNNModel(X.shape[1]).to(device)
    train_model(final_rnn, train_loader_full, val_loader_full, device, epochs=50, model_name="Final_RNN",
                use_dynamic_weighting=True, weight_update_frequency=3, l1_lambda=0.0, l2_lambda=0.005,
                enable_three_stage_random_masking=True, mask_value=0.0)
    torch.save(final_rnn.state_dict(), 'final_rnn.pth')

    print("\n--- 訓練最終 LSTM 模型 ---")
    final_lstm = LSTMModel(X.shape[1]).to(device)
    train_model(final_lstm, train_loader_full, val_loader_full, device, epochs=50, model_name="Final_LSTM",
                use_dynamic_weighting=True, weight_update_frequency=3, l1_lambda=0.0, l2_lambda=0.005,
                enable_three_stage_random_masking=True, mask_value=0.0)
    torch.save(final_lstm.state_dict(), 'final_lstm.pth')

    print("\n--- 訓練 Meta-model ---")
    ann_pred_train = predict(final_ann, X_resampled, device, batch_size=256)
    cnn_pred_train = predict(final_cnn, X_resampled, device, batch_size=256)
    rnn_pred_train = predict(final_rnn, X_resampled, device, batch_size=256)
    lstm_pred_train = predict(final_lstm, X_resampled, device, batch_size=256)

    meta_X_train = np.column_stack([ann_pred_train, cnn_pred_train, rnn_pred_train, lstm_pred_train])
    scale_pos_weight = (y_resampled == 0).sum() / (y_resampled == 1).sum()

    final_meta_model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.0,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.1
    )
    final_meta_model.fit(meta_X_train, y_resampled)

    # Per-Domain Calibration
    print("\n--- 進行 Per-Domain Calibration ---")
    calibrator = PerDomainCalibrator(method='sigmoid')
    meta_pred_proba = final_meta_model.predict_proba(meta_X_train)[:, 1]
    calibrator.fit(meta_pred_proba, y_resampled, protocol_types_resampled)
    calibrated_meta_pred_proba = calibrator.predict_proba(meta_pred_proba, protocol_types_resampled)
    
    # Evaluate calibrated predictions
    final_meta_pred = (calibrated_meta_pred_proba > 0.5).astype(int)
    final_acc = accuracy_score(y_resampled, final_meta_pred)
    final_prec = precision_score(y_resampled, final_meta_pred)
    final_rec = recall_score(y_resampled, final_meta_pred)
    final_f1 = f1_score(y_resampled, final_meta_pred)
    final_auc = roc_auc_score(y_resampled, calibrated_meta_pred_proba)
    print("\nFinal Meta-model Performance (Calibrated, on all training data):")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Precision: {final_prec:.4f}")
    print(f"Recall: {final_rec:.4f}")
    print(f"F1-Score: {final_f1:.4f}")
    print(f"AUC: {final_auc:.4f}")

    # Save models and calibrator
    joblib.dump(final_meta_model, 'final_meta_model.pkl')
    calibrator.save('per_domain_calibrator.pkl')

    with open('model_config.pkl', 'wb') as f:
        pickle.dump({'input_size': X.shape[1]}, f)

    print("\nModels, calibrator, and artifacts saved successfully.")
    print("動態類別權重與 Per-Domain Calibration 訓練完成！")

if __name__ == "__main__":
    main()