import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import seaborn as sns

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

# Preprocess data (same as in aclr_KDD.py)
def preprocess_data(df, numerical_cols, categorical_cols, scalers, label_encoders, selected_features, final_scaler):
    # Remove unnecessary columns
    columns_to_drop = ['attack_cat']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Label Encoding for categorical variables (same as aclr_KDD.py)
    X_categorical = pd.DataFrame()
    for col in categorical_cols:
        if col in df.columns:
            valid_values = set(label_encoders[col].classes_)
            df[col] = df[col].fillna('').apply(lambda x: x if x in valid_values else '')
            X_categorical[col] = label_encoders[col].transform(df[col].fillna(''))
        else:
            X_categorical[col] = np.zeros(len(df))

    # Process numerical columns (same as aclr_KDD.py)
    X_numeric = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    
    # Handle outliers (IQR method) - same as aclr_KDD.py
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

    # Apply log transformation and scaling (same as aclr_KDD.py)
    X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
    for col in numerical_cols:
        # 對偏態分佈的特徵進行對數轉換
        if X_numeric[col].skew() > 1:
            X_numeric[col] = np.log1p(X_numeric[col] - X_numeric[col].min() + 1e-6)
        X_numeric_scaled[col] = scalers[col].transform(X_numeric[[col]]).ravel()

    # 確保數值型態且無NaN (same as aclr_KDD.py)
    X_numeric_scaled = X_numeric_scaled.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Select features (same as aclr_KDD.py)
    X_numeric_scaled = X_numeric_scaled[selected_features]

    # Create interaction features (same as aclr_KDD.py)
    interaction_features = {}
    for i in range(len(selected_features)):
        for j in range(i + 1, len(selected_features)):
            feat1, feat2 = selected_features[i], selected_features[j]
            interaction_features[f'{feat1}_{feat2}_inter'] = X_numeric_scaled[feat1] * X_numeric_scaled[feat2]
    X_numeric_scaled = pd.concat([X_numeric_scaled] + [pd.Series(v, name=k) for k, v in interaction_features.items()], axis=1)

    # Combine features (same as aclr_KDD.py)
    X = np.concatenate([X_numeric_scaled.values, X_categorical.values], axis=1).astype(np.float32)

    # Final standardization (same as aclr_KDD.py)
    X = final_scaler.transform(X)

    # Reshape for CNN
    X_cnn = X.reshape(X.shape[0], 1, X.shape[1])
    
    # 處理標籤：將 labels 轉換為二元分類 (same as aclr_KDD.py)
    if 'labels' in df.columns:
        # 檢查原始標籤分布
        print("原始標籤分布:")
        print(df['labels'].value_counts())
        
        # 將 normal 設為 0，其他設為 1
        y = (df['labels'] != 'normal').astype(np.float32)
        
        # 檢查轉換後的標籤分布
        print("轉換後標籤分布:")
        print(f"0 (normal): {np.sum(y==0)}")
        print(f"1 (attack): {np.sum(y==1)}")
        print(f"比例: 0={np.mean(y==0):.3f}, 1={np.mean(y==1):.3f}")
    else:
        y = None

    return X, X_cnn, y

def preprocess_data_aclr_style(df, scalers, label_encoders, selected_features, final_scaler):
    # 刪除 id、attack_cat
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    columns_to_drop = ['attack_cat']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    categorical_cols = ['Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
                       'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
                       'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['labels']]

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
        if X_numeric[col].skew() > 1:
            X_numeric[col] = np.log1p(X_numeric[col] - X_numeric[col].min() + 1e-6)
        X_numeric_scaled[col] = scalers[col].transform(X_numeric[[col]]).ravel()
    X_numeric_scaled = X_numeric_scaled.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 特徵選擇
    X_numeric_scaled = X_numeric_scaled[selected_features]

    # 特徵交互
    interaction_features = {}
    for i in range(len(selected_features)):
        for j in range(i + 1, len(selected_features)):
            feat1, feat2 = selected_features[i], selected_features[j]
            interaction_features[f'{feat1}_{feat2}_inter'] = X_numeric_scaled[feat1] * X_numeric_scaled[feat2]
    X_numeric_scaled = pd.concat([X_numeric_scaled] + [pd.Series(v, name=k) for k, v in interaction_features.items()], axis=1)

    # 合併
    X = np.concatenate([X_numeric_scaled.values, X_categorical.values], axis=1).astype(np.float32)
    X = final_scaler.transform(X)
    X_cnn = X.reshape(X.shape[0], 1, X.shape[1])

    # 標籤
    if 'labels' in df.columns:
        y = (df['labels'] != 'normal').astype(np.float32)
    else:
        y = None

    return X, X_cnn, y


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
        with open('model_config.pkl', 'rb') as f:
            model_config = pickle.load(f)
        print("已加載 model_config.pkl")
        print(f"模型配置: {model_config}")
        print("正在加載元模型...")
        final_meta_model = joblib.load('final_meta_model.pkl')
        print("已加載 final_meta_model.pkl")
    except Exception as e:
        print(f"加載文件時發生錯誤：{e}")
        return

    input_size = model_config['input_size']
    # Initialize models
    final_ann = ANNModel(input_size).to(device)
    final_cnn = CNNModel(input_size).to(device)
    final_rnn = RNNModel(input_size).to(device)
    final_lstm = LSTMModel(input_size).to(device)
    print("模型初始化完成")

    # Load model weights
    try:
        final_ann.load_state_dict(torch.load('final_ann.pth', map_location=device))
        final_cnn.load_state_dict(torch.load('final_cnn.pth', map_location=device))
        final_rnn.load_state_dict(torch.load('final_rnn.pth', map_location=device))
        final_lstm.load_state_dict(torch.load('final_lstm.pth', map_location=device))
        print("模型權重加載成功")
    except Exception as e:
        print(f"模型權重加載失敗：{e}")
        return

    # === Cross-Dataset Evaluation ===
    try:
        # Define the exact columns you want to load
        # Ensure 'Label' is also included if it's your target variable
        # Note: 'protocol_type', 'service', 'flag', etc., are implicitly handled
        #       by preprocess_data_aclr_style if they are in the list.
        #       We include 'Label' here for the target variable.
        columns_to_load = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Fwd Packets Length Total', 'Bwd Packets Length Total', 'Fwd Packet Length Max',
            'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
            'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
            'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
            'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
            'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
            'Down/Up Ratio', 'Avg Packet Size', 'Avg Fwd Segment Size',
            'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
            'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init Fwd Win Bytes',
            'Init Bwd Win Bytes', 'Fwd Act Data Packets', 'Fwd Seg Size Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
            'Idle Std', 'Idle Max', 'Idle Min', 'Label' # Add 'Label' here
        ]
        
        # Read only the specified columns
        cross_df = pd.read_csv('merged_nsl_to_cic2017.csv', usecols=columns_to_load)
        print("已加載 merged_nsl_to_cic2017.csv，並只讀取指定列。")
    except FileNotFoundError:
        print("Cross-dataset file not found. Skipping cross-dataset evaluation.")
        return
    except KeyError as e:
        print(f"錯誤：指定的列在 CSV 文件中找不到：{e}. 請檢查列名是否正確。")
        return


    X_cross, X_cross_cnn, y_cross = preprocess_data_aclr_style(cross_df, scalers, label_encoders, selected_features, final_scaler)
    print("X_cross shape:", X_cross.shape)
    print("X_cross_cnn shape:", X_cross_cnn.shape)
    print("y_cross shape:", y_cross.shape)
    # ... (rest of the main function remains the same) ...

    # 對跨資料集資料進行 SMOTE 過採樣
    from imblearn.over_sampling import SMOTE  # 新增：匯入 SMOTE
    smote = SMOTE(random_state=42)
    X_cross, y_cross = smote.fit_resample(X_cross, y_cross)
    X_cross_cnn = X_cross.reshape(X_cross.shape[0], 1, X_cross.shape[1])
    print("SMOTE 過採樣後 X_cross shape:", X_cross.shape)
    print("SMOTE 過採樣後 X_cross_cnn shape:", X_cross_cnn.shape)
    print("SMOTE 過採樣後 y_cross shape:", y_cross.shape)

    # Validate CNN input shape for cross-dataset
    if X_cross_cnn.shape[2] != input_size:
        print(f"錯誤：X_cross_cnn 的特徵維度 {X_cross_cnn.shape[2]} 與預期的輸入長度 {input_size} 不匹配")
        print("請檢查跨數據集預處理或重新生成數據")
        return

    # Generate predictions for cross-dataset
    print("[Final Meta Classifier] Evaluating on Cross-Dataset...")
    X_stacked_cross = []
    best_models = [(final_ann, "ann"), (final_cnn, "cnn"), (final_rnn, "rnn"), (final_lstm, "lstm")]

    for model, model_type in best_models:
        if model_type == "cnn":
            preds = predict(model, X_cross_cnn, device)
        else:
            preds = predict(model, X_cross, device)
        X_stacked_cross.append(preds)

    # Stack predictions and classify
    X_stacked_cross = np.array(X_stacked_cross).T
    final_preds = final_meta_model.predict(X_stacked_cross)
    final_probs = final_meta_model.predict_proba(X_stacked_cross)[:, 1]

    # Compute metrics
    acc = accuracy_score(y_cross, final_preds)
    prec = precision_score(y_cross, final_preds)
    rec = recall_score(y_cross, final_preds)
    f1 = f1_score(y_cross, final_preds)
    fpr, tpr, _ = roc_curve(y_cross, final_probs)
    roc_auc = auc(fpr, tpr)

    # 混淆矩陣（跨資料集）
    cm_cross = confusion_matrix(y_cross, final_preds)
    print("\n混淆矩陣 (跨數據集):")
    print(cm_cross)
    if cm_cross.shape == (2, 2):
        tn, fp, fn, tp = cm_cross.ravel()
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        # 重新排列為 [[TP, FN], [FP, TN]]
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
        plt.savefig('confusion_matrix_cross_TP_FN_FP_TN.png')
        plt.close()
    else:
        print("警告：混淆矩陣不是二元分類格式，無法顯示 TN/FP/FN/TP")
    plt.figure()
    # plt.imshow(cm_cross, interpolation='nearest', cmap=plt.cm.Greens)
    sns.heatmap(cm_cross, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix (Cross-Dataset)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_cross.png')
    plt.close()

    print(f"Cross-Dataset Accuracy:  {acc:.4f}")
    print(f"Cross-Dataset Precision: {prec:.4f}")
    print(f"Cross-Dataset Recall:    {rec:.4f}")
    print(f"Cross-Dataset F1 Score:  {f1:.4f}")
    print(f"Cross-Dataset AUC:       {roc_auc:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'Cross-Dataset ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Cross-Dataset')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_cross.png')
    plt.close()

    # Save input size for model initialization in test.py
    with open('model_config.pkl', 'wb') as f:
        pickle.dump({'input_size': X_cross.shape[1]}, f)

    # 保存測試數據
    print("\n正在保存測試數據...")
    np.save('X_test.npy', X_cross)
    np.save('y_test.npy', y_cross)
    np.save('X_cnn_test.npy', X_cross_cnn)
    print("測試數據保存完成")

    print("\nModels and artifacts saved successfully.")

   
    