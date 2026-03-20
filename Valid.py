import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch._dynamo

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib

# Focal Loss implementation
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
        self.X = torch.FloatTensor(X)
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

# Evaluation function
def evaluate_model(model, val_loader, device='cpu'):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
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
    return val_loss / len(val_loader), correct / total

# Training function
def train_model(model, train_loader, val_loader, device='cpu', epochs=10, model_name=""):
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
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
            preds = (outputs > 0.5).float()
            correct += (preds.squeeze() == y_batch).sum().item()
            total += y_batch.size(0)
            pbar.set_postfix({'Loss': loss.item(), 'Acc': f"{(correct / total):.4f}"})

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Prediction function
def predict(model, X, device='cpu'):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        return predictions.cpu().numpy().flatten()

# Main training logic
def main():
    # Read and preprocess data
    training = pd.read_csv('UNSW_NB15_training-set.csv')  # Adjust path
    testing = pd.read_csv('UNSW_NB15_testing-set.csv')    # Adjust path
    df = pd.concat([training, testing]).drop('id', axis=1).reset_index(drop=True)

    # Remove unnecessary columns
    columns_to_drop = ['attack_cat']
    df = df.drop(columns=columns_to_drop)

    # Define feature columns
    categorical_cols = ['proto', 'service', 'state']
    numerical_cols = df.columns.drop(categorical_cols + ['label']).tolist()

    print(f"Feature count: {len(numerical_cols) + len(categorical_cols)}")
    print("Numerical features:", numerical_cols)
    print("Categorical features:", categorical_cols)

    # Label Encoding for categorical variables
    label_encoders = {}
    X_categorical = pd.DataFrame()
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        values = df[col].fillna('').astype(str).tolist()
        if '' not in values:
            values.append('')
        label_encoders[col].fit(values)
        X_categorical[col] = label_encoders[col].transform(df[col].fillna(''))

    # Convert numerical columns to float and handle missing values
    X_numeric = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    X_numeric = X_numeric.fillna(X_numeric.mean())

    # Standardize numerical features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)

    # Combine features
    X = np.concatenate([X_numeric_scaled, X_categorical.values], axis=1).astype(np.float32)

    # Clip and normalize features
    X = np.clip(X, -10, 10)
    X = (X - X.min()) / (X.max() - X.min())

    # Reshape for CNN
    X_cnn = X.reshape(X.shape[0], 1, X.shape[1])
    y = df['label'].values.astype(np.float32)

    # Data splitting (corrected to split only X_cnn)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_cnn_train_full, X_cnn_test = train_test_split(X_cnn, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    X_cnn_train, X_cnn_val = train_test_split(X_cnn_train_full, test_size=0.2, random_state=42)

    # Save test data for use in test.py
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    np.save('X_cnn_test.npy', X_cnn_test)

    # Save scaler and label encoders
    joblib.dump(scaler, 'scaler.pkl')
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3-Fold Cross-Validation
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
        X_train_cnn, X_val_cnn = X_cnn[train_idx], X_cnn[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = BotnetDataset(X_train, y_train)
        val_dataset = BotnetDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=32, drop_last=True)

        cnn_train_dataset = CNNDataset(X_train_cnn, y_train)
        cnn_val_dataset = CNNDataset(X_val_cnn, y_val)
        cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32, shuffle=True, drop_last=True)
        cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32, drop_last=True)

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
    print("\nModel Performance Metrics (3-Fold):")
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
    plt.title('ROC Curves for All Folds')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_folds.png')
    plt.close()

    # Train final models
    print("\n===== Training Final Models =====")
    train_dataset_full = BotnetDataset(X_train_full, y_train_full)
    train_loader_full = DataLoader(train_dataset_full, batch_size=32, shuffle=True, drop_last=True)
    val_dataset = BotnetDataset(X_test, y_test)
    val_loader = DataLoader(val_dataset, batch_size=32, drop_last=True)

    cnn_train_dataset_full = CNNDataset(X_cnn_train_full, y_train_full)
    cnn_train_loader_full = DataLoader(cnn_train_dataset_full, batch_size=32, shuffle=True, drop_last=True)
    cnn_val_dataset = CNNDataset(X_cnn_test, y_test)
    cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32, drop_last=False)

    final_ann = ANNModel(X.shape[1]).to(device)
    train_model(final_ann, train_loader_full, val_loader, device, epochs=10, model_name="Final ANN")
    torch.save(final_ann.state_dict(), 'final_ann.pth')

    final_cnn = CNNModel(X.shape[1]).to(device)
    train_model(final_cnn, cnn_train_loader_full, cnn_val_loader, device, epochs=10, model_name="Final CNN")
    torch.save(final_cnn.state_dict(), 'final_cnn.pth')

    final_rnn = RNNModel(X.shape[1]).to(device)
    train_model(final_rnn, train_loader_full, val_loader, device, epochs=10, model_name="Final RNN")
    torch.save(final_rnn.state_dict(), 'final_rnn.pth')

    final_lstm = LSTMModel(X.shape[1]).to(device)
    train_model(final_lstm, train_loader_full, val_loader, device, epochs=10, model_name="Final LSTM")
    torch.save(final_lstm.state_dict(), 'final_lstm.pth')

    # Generate predictions for meta-model training
    ann_pred_train = predict(final_ann, X_train_full, device)
    cnn_pred_train = predict(final_cnn, X_cnn_train_full, device)
    rnn_pred_train = predict(final_rnn, X_train_full, device)
    lstm_pred_train = predict(final_lstm, X_train_full, device)

    # Train meta-model
    meta_X_train = np.column_stack([ann_pred_train, cnn_pred_train, rnn_pred_train, lstm_pred_train])
    final_meta_model = LogisticRegression()
    final_meta_model.fit(meta_X_train, y_train_full)
    joblib.dump(final_meta_model, 'final_meta_model.pkl')

    # Save input size for model initialization in test.py
    with open('model_config.pkl', 'wb') as f:
        pickle.dump({'input_size': X.shape[1]}, f)

    print("\nModels and artifacts saved successfully.")

if __name__ == "__main__":
    main()