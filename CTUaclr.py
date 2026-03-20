import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd
import numpy as np

# 讀取資料
training = pd.read_csv('CTU13_Attack_Traffic.csv')
testing = pd.read_csv('CTU13_Normal_Traffic.csv')
df = pd.concat([training, testing])

# 移除不需要的欄位
#columns_to_drop = ['attack_cat']
#df = df.drop(columns=columns_to_drop)

# Label Encoding 欄位
#categorical_cols = ['proto', 'service', 'state']
numerical_cols = df.columns.drop(['Label']).tolist()

#print(f"特徵數量：{len(numerical_cols) + len(categorical_cols)}")
#print("數值型特徵：", numerical_cols)
#print("類別型特徵：", categorical_cols)

# 對類別變數進行 Label Encoding
#label_encoders = {}
#X_categorical = pd.DataFrame()
#for col in categorical_cols:
    #[col] = LabelEncoder()
    #X_categorical[col] = label_encoders[col].fit_transform(df[col])

# 將數值型欄位轉換為浮點數
X_numeric = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
# 填充缺失值（如果有）
X_numeric = X_numeric.fillna(X_numeric.mean())

# 標準化
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# 合併特徵
X = np.concatenate([X_numeric_scaled], axis=1).astype(np.float32)

# 確保所有特徵值都在合理範圍內
X = np.clip(X, -10, 10)  # 限制極端值
X = (X - X.min()) / (X.max() - X.min())  # 正規化到 0-1 範圍

X_cnn = X.reshape(X.shape[0], 1, X.shape[1])
y = df['Label'].values.astype(np.float32)

# 資料分割
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_cnn_train_full, X_cnn_test = train_test_split(X_cnn, test_size=0.3, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
X_cnn_train, X_cnn_val = train_test_split(X_cnn_train_full, test_size=0.2, random_state=42)

# Dataset 定義
class BotnetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 模型定義
class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout 來防止過擬合
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
        self.conv1 = nn.Conv1d(1, 64, 3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 32, 3)
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

# 評估函數
def evaluate_model(model, val_loader, device='cpu'):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            val_loss += loss.item()
            preds = (outputs > 0.8).float()
            correct += (preds.squeeze() == y_batch).sum().item()
            total += y_batch.size(0)
    return val_loss / len(val_loader), correct / total

# 訓練函數
def train_model(model, train_loader, val_loader, device='cpu', epochs=10, model_name=""):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
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
            preds = (outputs > 0.8).float()
            correct += (preds.squeeze() == y_batch).sum().item()
            total += y_batch.size(0)
            pbar.set_postfix({'Loss': loss.item(), 'Acc': f"{(correct / total):.4f}"})

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# 預測函數
def predict(model, X, device='cpu'):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        return predictions.cpu().numpy().flatten()

# 主程式與 Stacking
kf = KFold(n_splits=3, shuffle=True, random_state=42)
results = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n===== Fold {fold + 1} =====")
    X_train, X_val = X[train_idx], X[val_idx]
    X_train_cnn, X_val_cnn = X_cnn[train_idx], X_cnn[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_dataset = BotnetDataset(X_train, y_train)
    val_dataset = BotnetDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    cnn_train_dataset = CNNDataset(X_train_cnn, y_train)
    cnn_val_dataset = CNNDataset(X_val_cnn, y_val)
    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32, shuffle=True)
    cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32)

    ann = ANNModel(X.shape[1]).to(device)
    train_model(ann, train_loader, val_loader, device, epochs=10, model_name="ANN")
    ann_pred = predict(ann, X_val, device)

    cnn = CNNModel(X.shape[1]).to(device)
    train_model(cnn, cnn_train_loader, cnn_val_loader, device, epochs=10, model_name="CNN")
    cnn_pred = predict(cnn, X_val_cnn, device)

    rnn = RNNModel(X.shape[1]).to(device)
    train_model(rnn, train_loader, val_loader, device, epochs=10, model_name="RNN")
    rnn_pred = predict(rnn, X_val, device)

    lstm = LSTMModel(X.shape[1]).to(device)
    train_model(lstm, train_loader, val_loader, device, epochs=10, model_name="LSTM")
    lstm_pred = predict(lstm, X_val, device)

    meta_X = np.column_stack([ann_pred, cnn_pred, rnn_pred, lstm_pred])
    meta_model = LogisticRegression()
    meta_model.fit(meta_X, y_val)
    final_pred = meta_model.predict(meta_X)

    acc = accuracy_score(y_val, final_pred)
    prec = precision_score(y_val, final_pred)
    rec = recall_score(y_val, final_pred)
    f1 = f1_score(y_val, final_pred)

    results.append({
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

results_df = pd.DataFrame(results)
print("\nModel Performance Metrics (3-Fold):")
print(results_df)
print("\nAverage Performance:")
print(results_df.mean())