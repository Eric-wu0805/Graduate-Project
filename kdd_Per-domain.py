import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
from imblearn.over_sampling import SMOTE

# ======================
# Per-Domain Calibrator Class
# ======================
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
                base_estimator=None,
                method=self.method,
                cv='prefit'
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

# ======================
# 條件式 SMOTE 函數
# ======================
def apply_smote_if_needed(X, y, domains, threshold=0.35):
    """
    若資料不平衡（正類或負類佔比小於 threshold）則執行 SMOTE。
    """
    positive_ratio = np.mean(y == 1)
    negative_ratio = np.mean(y == 0)
    print(f"類別分布：正類 = {positive_ratio:.3f}, 負類 = {negative_ratio:.3f}")
    if positive_ratio < threshold or negative_ratio < threshold:
        print("資料不平衡，執行 SMOTE 過採樣...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        # Resample domains to match
        _, domains_resampled = smote.fit_resample(X, domains)
        print("SMOTE 過採樣後形狀:", X_resampled.shape, y_resampled.shape)
        return X_resampled, y_resampled, domains_resampled
    else:
        print("資料分布均衡，跳過 SMOTE 處理。")
        return X, y, domains

# Model definitions (same as in aclr.py)
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

# Preprocess data (modified to retain protocol_type)
def preprocess_data_aclr_style(df, scalers, label_encoders, selected_features, final_scaler):
    # 只保留指定特徵
    selected_feature_names = ['src_bytes', 'dst_bytes', 'duration', 'same_srv_rate', 'protocol_type', 'flag']
    categorical_cols = ['protocol_type', 'flag']
    numerical_cols = ['src_bytes', 'dst_bytes', 'duration', 'same_srv_rate']
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    columns_to_drop = ['attack_cat']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Retain protocol_type for calibration
    protocol_types = df['protocol_type'].astype(str).fillna('').values

    # Label Encoding
    X_categorical = pd.DataFrame()
    for col in categorical_cols:
        if col in df.columns:
            valid_values = set(label_encoders[col].classes_)
            df[col] = df[col].fillna('').apply(lambda x: x if x in valid_values else '')
            X_categorical[col] = label_encoders[col].transform(df[col].fillna(''))
        else:
            X_categorical[col] = np.zeros(len(df))

    # 數值特徵
    X_numeric = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    # 先對src_bytes、dst_bytes、duration做log1p轉換
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

    X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
    for col in numerical_cols:
        X_numeric_scaled[col] = scalers[col].transform(X_numeric[[col]]).ravel()
    X_numeric_scaled = X_numeric_scaled.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 不做特徵選擇與交互，直接用這五個特徵
    X = np.concatenate([X_numeric_scaled.values, X_categorical.values], axis=1).astype(np.float32)
    X = final_scaler.transform(X)
    X_cnn = X.reshape(X.shape[0], 1, X.shape[1])

    # 標籤
    if 'labels' in df.columns:
        y = (df['labels'] != 'normal').astype(np.float32)
    else:
        y = None

    return X, X_cnn, y, protocol_types

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # Load saved artifacts
    scalers = joblib.load('feature_scalers.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    selected_features = joblib.load('selected_features.pkl')
    final_scaler = joblib.load('final_scaler.pkl')
    with open('model_config.pkl', 'rb') as f:
        model_config = pickle.load(f)
    final_meta_model = joblib.load('final_meta_model.pkl')  # 使用XGBoost
    input_size = model_config['input_size']
    final_ann = ANNModel(input_size).to(device)
    final_cnn = CNNModel(input_size).to(device)
    final_rnn = RNNModel(input_size).to(device)
    final_lstm = LSTMModel(input_size).to(device)

    final_ann.load_state_dict(torch.load('final_ann.pth', map_location=device))
    final_cnn.load_state_dict(torch.load('final_cnn.pth', map_location=device))
    final_rnn.load_state_dict(torch.load('final_rnn.pth', map_location=device))
    final_lstm.load_state_dict(torch.load('final_lstm.pth', map_location=device))

    try:
        cross_df = pd.read_csv('nb15_kdd_train.csv')
        # cross_df = pd.read_csv('cic2017_merged_to_kdd.csv')
    except FileNotFoundError:
        print("Cross-dataset file not found. Skipping cross-dataset evaluation.")
        return

    # Preprocess with domain retention
    X_cross, X_cross_cnn, y_cross, protocol_types = preprocess_data_aclr_style(
        cross_df, scalers, label_encoders, selected_features, final_scaler)

    # 條件式 SMOTE 過採樣
    X_cross, y_cross, protocol_types = apply_smote_if_needed(X_cross, y_cross, protocol_types)
    X_cross_cnn = X_cross.reshape(X_cross.shape[0], 1, X_cross.shape[1])

    # 預測
    print("[Final Meta Classifier] Evaluating on Cross-Dataset...")
    X_stacked_cross = []
    best_models = [(final_ann, "ann"), (final_cnn, "cnn"), (final_rnn, "rnn"), (final_lstm, "lstm")]

    for model, model_type in best_models:
        preds = predict(model, X_cross_cnn if model_type == "cnn" else X_cross, device)
        X_stacked_cross.append(preds)

    X_stacked_cross = np.array(X_stacked_cross).T
    final_preds = final_meta_model.predict(X_stacked_cross)
    final_probs = final_meta_model.predict_proba(X_stacked_cross)[:, 1]

    # Per-Domain Calibration
    print("\n--- Performing Per-Domain Calibration ---")
    calibrator = PerDomainCalibrator(method='sigmoid')
    calibrator.fit(final_probs, y_cross, protocol_types)
    calibrated_probs = calibrator.predict_proba(final_probs, protocol_types)

    # Evaluate raw predictions
    acc = accuracy_score(y_cross, final_preds)
    prec = precision_score(y_cross, final_preds)
    rec = recall_score(y_cross, final_preds)
    f1 = f1_score(y_cross, final_preds)
    fpr, tpr, _ = roc_curve(y_cross, final_probs)
    roc_auc = auc(fpr, tpr)

    print("\nRaw Performance (Cross-Dataset):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {roc_auc:.4f}")

    # Evaluate calibrated predictions
    calibrated_preds = (calibrated_probs > 0.5).astype(int)
    cal_acc = accuracy_score(y_cross, calibrated_preds)
    cal_prec = precision_score(y_cross, calibrated_preds)
    cal_rec = recall_score(y_cross, calibrated_preds)
    cal_f1 = f1_score(y_cross, calibrated_preds)
    cal_fpr, cal_tpr, _ = roc_curve(y_cross, calibrated_probs)
    cal_roc_auc = auc(cal_fpr, cal_tpr)

    print("\nCalibrated Performance (Cross-Dataset):")
    print(f"Accuracy:  {cal_acc:.4f}")
    print(f"Precision: {cal_prec:.4f}")
    print(f"Recall:    {cal_rec:.4f}")
    print(f"F1 Score:  {cal_f1:.4f}")
    print(f"AUC:       {cal_roc_auc:.4f}")

    # Confusion matrix for raw predictions
    cm_cross = confusion_matrix(y_cross, final_preds)
    print("\n混淆矩陣 (跨數據集, Raw):")
    print(cm_cross)

    if cm_cross.shape == (2, 2):
        tn, fp, fn, tp = cm_cross.ravel()
        cm_tp_fn_fp_tn = np.array([[tp, fn], [fp, tn]])
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm_tp_fn_fp_tn, annot=False, fmt='g', cmap='Greens', cbar=True,
                         xticklabels=['Predicted Positive', 'Predicted Negative'],
                         yticklabels=['Actual Positive', 'Actual Negative'])
        labels = [['TP', 'FN'], ['FP', 'TN']]
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.5, f"{labels[i][j]}\n{cm_tp_fn_fp_tn[i, j]}",
                        color='black', ha='center', va='center', fontsize=13, fontweight='bold')
        plt.title('Confusion Matrix (Raw, Cross-Dataset)')
        plt.tight_layout()
        plt.savefig('confusion_matrix_cross_raw_TP_FN_FP_TN.png')
        plt.close()

    plt.figure()
    sns.heatmap(cm_cross, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix (Raw, Cross-Dataset)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_cross_raw.png')
    plt.close()

    # Confusion matrix for calibrated predictions
    cm_calibrated = confusion_matrix(y_cross, calibrated_preds)
    print("\n混淆矩陣 (跨數據集, Calibrated):")
    print(cm_calibrated)

    if cm_calibrated.shape == (2, 2):
        tn, fp, fn, tp = cm_calibrated.ravel()
        cm_tp_fn_fp_tn = np.array([[tp, fn], [fp, tn]])
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm_tp_fn_fp_tn, annot=False, fmt='g', cmap='Greens', cbar=True,
                         xticklabels=['Predicted Positive', 'Predicted Negative'],
                         yticklabels=['Actual Positive', 'Actual Negative'])
        labels = [['TP', 'FN'], ['FP', 'TN']]
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.5, f"{labels[i][j]}\n{cm_tp_fn_fp_tn[i, j]}",
                        color='black', ha='center', va='center', fontsize=13, fontweight='bold')
        plt.title('Confusion Matrix (Calibrated, Cross-Dataset)')
        plt.tight_layout()
        plt.savefig('confusion_matrix_cross_calibrated_TP_FN_FP_TN.png')
        plt.close()

    plt.figure()
    sns.heatmap(cm_calibrated, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix (Calibrated, Cross-Dataset)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_cross_calibrated.png')
    plt.close()

    # ROC curve for both raw and calibrated
    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'Raw ROC (AUC = {roc_auc:.4f})')
    plt.plot(cal_fpr, cal_tpr, color='blue', lw=2, label=f'Calibrated ROC (AUC = {cal_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Cross-Dataset)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_cross_calibrated.png')
    plt.close()

    # Save calibrator and test data
    calibrator.save('per_domain_calibrator_cross.pkl')
    np.save('X_test.npy', X_cross)
    np.save('y_test.npy', y_cross)
    np.save('X_cnn_test.npy', X_cross_cnn)
    np.save('protocol_types_test.npy', protocol_types)
    print("\n測試數據和校準器保存完成")

if __name__ == "__main__":
    main()