import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # Changed from StandardScaler to MinMaxScaler

# Conditional SMOTE function
def apply_smote_if_needed(X, y, threshold=0.35):
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
        print("SMOTE 過採樣後形狀:", X_resampled.shape, y_resampled.shape)
        return X_resampled, y_resampled
    else:
        print("資料分布均衡，跳過 SMOTE 處理。")
        return X, y

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

# Preprocess data (updated for consistency with MinMaxScaler)
def preprocess_data(df, numerical_cols, categorical_cols, scalers, label_encoders, selected_features, final_scaler):
    selected_feature_names = ['src_bytes', 'dst_bytes', 'duration', 'dst_host_srv_diff_host_rate', 'flag']
    categorical_cols = ['flag']
    numerical_cols = [col for col in selected_feature_names if col not in categorical_cols + ['labels']]
    columns_to_drop = ['attack_cat']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    X_categorical = pd.DataFrame()
    for col in categorical_cols:
        if col in df.columns:
            valid_values = set(label_encoders[col].classes_)
            df[col] = df[col].fillna('').apply(lambda x: x if x in valid_values else '')
            X_categorical[col] = label_encoders[col].transform(df[col].fillna(''))
        else:
            X_categorical[col] = np.zeros(len(df))

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

    X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
    for col in numerical_cols:
        X_numeric_scaled[col] = scalers[col].transform(X_numeric[[col]]).ravel()

    X_numeric_scaled = X_numeric_scaled.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = np.concatenate([X_numeric_scaled.values, X_categorical.values], axis=1).astype(np.float32)

    # Use MinMaxScaler for final scaling
    X = final_scaler.transform(X)

    X_cnn = X.reshape(X.shape[0], 1, X.shape[1])
    
    if 'labels' in df.columns:
        print("原始標籤分布:")
        print(df['labels'].value_counts())
        y = (df['labels'] != 'normal').astype(np.float32)
        print("轉換後標籤分布:")
        print(f"0 (normal): {np.sum(y==0)}")
        print(f"1 (attack): {np.sum(y==1)}")
        print(f"比例: 0={np.mean(y==0):.3f}, 1={np.mean(y==1):.3f}")
    else:
        y = None

    return X, X_cnn, y

# Preprocess data (ACLR style, updated to use MinMaxScaler)
def preprocess_data_aclr_style(df, scalers, label_encoders, selected_features, final_scaler):
    selected_feature_names = ['src_bytes', 'dst_bytes', 'duration', 'same_srv_rate', 'protocol_type', 'flag']
    categorical_cols = ['protocol_type', 'flag']
    numerical_cols = ['src_bytes', 'dst_bytes', 'duration', 'same_srv_rate']
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    columns_to_drop = ['attack_cat']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    X_categorical = pd.DataFrame()
    for col in categorical_cols:
        if col in df.columns:
            valid_values = set(label_encoders[col].classes_)
            df[col] = df[col].fillna('').apply(lambda x: x if x in valid_values else '')
            X_categorical[col] = label_encoders[col].transform(df[col].fillna(''))
        else:
            X_categorical[col] = np.zeros(len(df))

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

    X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
    for col in numerical_cols:
        X_numeric_scaled[col] = scalers[col].transform(X_numeric[[col]]).ravel()
    
    X_numeric_scaled = X_numeric_scaled.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = np.concatenate([X_numeric_scaled.values, X_categorical.values], axis=1).astype(np.float32)
    
    # Use MinMaxScaler for final scaling
    X = final_scaler.transform(X)
    
    X_cnn = X.reshape(X.shape[0], 1, X.shape[1])

    if 'labels' in df.columns:
        y = (df['labels'] != 'normal').astype(np.float32)
    else:
        y = None

    return X, X_cnn, y

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # Load saved artifacts
    scalers = joblib.load('feature_scalers.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    selected_features = joblib.load('selected_features.pkl')
    final_scaler = joblib.load('final_scaler.pkl')  # Assumes MinMaxScaler from training
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
    except FileNotFoundError:
        print("Cross-dataset file not found. Skipping cross-dataset evaluation.")
        return

    X_cross, X_cross_cnn, y_cross = preprocess_data_aclr_style(
        cross_df, scalers, label_encoders, selected_features, final_scaler)

    # 條件式 SMOTE 過採樣
    X_cross, y_cross = apply_smote_if_needed(X_cross, y_cross)
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

    acc = accuracy_score(y_cross, final_preds)
    prec = precision_score(y_cross, final_preds)
    rec = recall_score(y_cross, final_preds)
    f1 = f1_score(y_cross, final_preds)
    fpr, tpr, _ = roc_curve(y_cross, final_probs)
    roc_auc = auc(fpr, tpr)

    cm_cross = confusion_matrix(y_cross, final_preds)
    print("\n混淆矩陣 (跨數據集):")
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
        plt.title('Confusion Matrix (TP/FN/FP/TN)')
        plt.tight_layout()
        plt.savefig('confusion_matrix_cross_TP_FN_FP_TN.png')
        plt.close()

    plt.figure()
    sns.heatmap(cm_cross, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix (Cross-Dataset)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_cross.png')
    plt.close()

    print(f"Cross-Dataset Accuracy:  {acc:.4f}")
    print(f"Cross-Dataset Precision: {prec:.4f}")
    print(f"Cross-Dataset Recall:    {rec:.4f}")
    print(f"Cross-Dataset F1 Score:  {f1:.4f}")
    print(f"Cross-Dataset AUC:       {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_cross.png')
    plt.close()

    # 儲存測試數據
    np.save('X_test.npy', X_cross)
    np.save('y_test.npy', y_cross)
    np.save('X_cnn_test.npy', X_cross_cnn)
    print("\n測試數據保存完成")

if __name__ == "__main__":
    main()