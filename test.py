import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

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

# Preprocess data (same as in aclr.py)
def preprocess_data(df, numerical_cols, categorical_cols, scalers, label_encoders, selected_features, final_scaler):
    # Remove unnecessary columns
    columns_to_drop = ['attack_cat']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Label Encoding for categorical variables
    X_categorical = pd.DataFrame()
    for col in categorical_cols:
        if col in df.columns:
            valid_values = set(label_encoders[col].classes_)
            df[col] = df[col].fillna('').apply(lambda x: x if x in valid_values else '')
            X_categorical[col] = label_encoders[col].transform(df[col].fillna(''))
        else:
            X_categorical[col] = np.zeros(len(df))

    # Process numerical columns
    X_numeric = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

    # Handle outliers (IQR method)
    for col in numerical_cols:
        Q1 = X_numeric[col].quantile(0.25)
        Q3 = X_numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_numeric[col] = X_numeric[col].clip(lower_bound, upper_bound)

    # Apply log transformation and scaling
    X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
    for col in numerical_cols:
        if X_numeric[col].skew() > 1:
            X_numeric[col] = np.log1p(X_numeric[col] - X_numeric[col].min() + 1e-6)
        X_numeric_scaled[col] = scalers[col].transform(X_numeric[[col]]).ravel()

    # Select features
    X_numeric_scaled = X_numeric_scaled[selected_features]

    # Create interaction features
    interaction_features = {}
    for i in range(len(selected_features)):
        for j in range(i + 1, len(selected_features)):
            feat1, feat2 = selected_features[i], selected_features[j]
            interaction_features[f'{feat1}_{feat2}_inter'] = X_numeric_scaled[feat1] * X_numeric_scaled[feat2]
    X_numeric_scaled = pd.concat([X_numeric_scaled] + [pd.Series(v, name=k) for k, v in interaction_features.items()], axis=1)

    # Combine features
    X = np.concatenate([X_numeric_scaled.values, X_categorical.values], axis=1).astype(np.float32)

    # Final standardization
    X = final_scaler.transform(X)

    # Reshape for CNN
    X_cnn = X.reshape(X.shape[0], 1, X.shape[1])
    y = df['label'].values.astype(np.float32) if 'label' in df.columns else None

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
        
        print("正在加載元模型...")
        final_meta_model = joblib.load('final_meta_model.pkl')
        print("已加載 final_meta_model.pkl")
        
    except FileNotFoundError as e:
        print(f"錯誤：找不到文件 {e}")
        print("請確保已經運行 aclr.py 訓練模型")
        return
    except Exception as e:
        print(f"加載文件時發生錯誤：{e}")
        print("請檢查所有必要的文件是否存在")
        return

    input_size = model_config['input_size']
    categorical_cols = ['proto', 'service', 'state']
    numerical_cols = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
                      'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 
                      'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 
                      'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
                      'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 
                      'ct_srv_dst', 'is_sm_ips_ports']

    # Initialize models
    print("\n正在初始化模型...")
    print(f"輸入特徵數量: {input_size}")
    
    final_ann = ANNModel(input_size).to(device)
    print("ANN 模型初始化完成")
    
    final_cnn = CNNModel(input_size).to(device)
    print(f"CNN 模型初始化完成，輸入長度: {input_size}")
    
    final_rnn = RNNModel(input_size).to(device)
    print("RNN 模型初始化完成")
    
    final_lstm = LSTMModel(input_size).to(device)
    print("LSTM 模型初始化完成")

    # Inspect CNN model weights
    print("\n正在檢查 CNN 模型權重...")
    try:
        checkpoint = torch.load('final_cnn.pth', map_location=device)
        for name, param in checkpoint.items():
            print(f"Weight {name}: {param.shape}")
    except Exception as e:
        print(f"無法檢查 CNN 模型權重：{e}")
        return

    # Load model weights
    try:
        print("\n正在加載模型權重...")
        
        print("正在加載 ANN 模型...")
        final_ann.load_state_dict(torch.load('final_ann.pth', map_location=device))
        print("ANN 模型加載成功")
        
        print("正在加載 CNN 模型...")
        final_cnn.load_state_dict(torch.load('final_cnn.pth', map_location=device))
        print("CNN 模型加載成功")
        
        print("正在加載 RNN 模型...")
        final_rnn.load_state_dict(torch.load('final_rnn.pth', map_location=device))
        print("RNN 模型加載成功")
        
        print("正在加載 LSTM 模型...")
        final_lstm.load_state_dict(torch.load('final_lstm.pth', map_location=device))
        print("LSTM 模型加載成功")
        
    except FileNotFoundError as e:
        print(f"錯誤：找不到模型文件 {e}")
        print("請確保已經運行 aclr.py 訓練模型")
        return
    except RuntimeError as e:
        print(f"加載模型時發生錯誤：{e}")
        print("請檢查 CNNModel 架構是否與訓練時一致，並確認 input_size 是否正確")
        print("建議運行 aclr.py 重新訓練模型")
        return

    # Load test data
    try:
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        X_cnn_test = np.load('X_cnn_test.npy')
    except FileNotFoundError as e:
        print(f"錯誤：找不到測試數據文件 {e}")
        return

    # Validate CNN input shape
    print("\n正在驗證測試數據形狀...")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("X_cnn_test shape:", X_cnn_test.shape)
    if X_cnn_test.shape[2] != input_size:
        print(f"錯誤：X_cnn_test 的特徵維度 {X_cnn_test.shape[2]} 與預期的輸入長度 {input_size} 不匹配")
        print("請檢查測試數據預處理或重新生成 X_test.npy 和 X_cnn_test.npy")
        return

    # Test set evaluation
    print("\n===== Final Evaluation on Test Set =====")
    
    # Generate predictions
    print("\n正在生成預測...")
    try:
        print("正在使用 ANN 模型進行預測...")
        ann_pred_test = predict(final_ann, X_test, device)
        print("ANN 模型預測完成")

        print("正在使用 CNN 模型進行預測...")
        cnn_pred_test = predict(final_cnn, X_cnn_test, device)
        print("CNN 模型預測完成")

        print("正在使用 RNN 模型進行預測...")
        rnn_pred_test = predict(final_rnn, X_test, device)
        print("RNN 模型預測完成")

        print("正在使用 LSTM 模型進行預測...")
        lstm_pred_test = predict(final_lstm, X_test, device)
        print("LSTM 模型預測完成")

        # Stack predictions for meta-model
        print("正在組合預測結果...")
        meta_X_test = np.column_stack([ann_pred_test, cnn_pred_test, rnn_pred_test, lstm_pred_test])
        print("正在使用元模型進行最終預測...")
        final_pred_test = final_meta_model.predict(meta_X_test)
        final_pred_proba_test = final_meta_model.predict_proba(meta_X_test)[:, 1]
        print("預測完成")

    except Exception as e:
        print(f"預測過程中發生錯誤：{e}")
        return

    # Compute metrics
    print("\n正在計算評估指標...")
    test_acc = accuracy_score(y_test, final_pred_test)
    test_prec = precision_score(y_test, final_pred_test)
    test_rec = recall_score(y_test, final_pred_test)
    test_f1 = f1_score(y_test, final_pred_test)
    fpr_test, tpr_test, _ = roc_curve(y_test, final_pred_proba_test)
    test_auc = auc(fpr_test, tpr_test)

    # Output results
    print("\nFinal Test Set Performance:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"AUC: {test_auc:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Test Set')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_test.png')
    plt.close()

    # Cross-dataset evaluation
    print("\n=== Cross-Dataset Evaluation ===")
    try:
        cross_df = pd.read_csv('NSL_KDD_to_UNSW (1).csv')
    except FileNotFoundError:
        print("Cross-dataset file not found. Skipping cross-dataset evaluation.")
        return

    X_cross, X_cross_cnn, y_cross = preprocess_data(cross_df, numerical_cols, categorical_cols, scalers, label_encoders, selected_features, final_scaler)
    print("X_cross shape:", X_cross.shape)
    print("X_cross_cnn shape:", X_cross_cnn.shape)
    print("y_cross shape:", y_cross.shape)

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
        pickle.dump({'input_size': X_test.shape[1]}, f)

    # 保存測試數據
    print("\n正在保存測試數據...")
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    np.save('X_cnn_test.npy', X_cnn_test)
    print("測試數據保存完成")

    print("\nModels and artifacts saved successfully.")

if __name__ == "__main__":
    main()