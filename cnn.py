# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras import models, layers

# 讀取資料集
training = pd.read_csv('UNSW_NB15_training-set.csv')
testing = pd.read_csv('UNSW_NB15_testing-set.csv')

# 合併資料，移除 id 欄
df = pd.concat([training, testing]).drop('id', axis=1).reset_index(drop=True)

# 分割特徵與標籤
# 特徵與標籤分開
y = df['label'].values
categorical_cols = ['proto', 'service', 'state', 'attack_cat', 
                    'is_ftp_login', 'ct_flw_http_mthd', 'is_sm_ips_ports']

# 正確的 One-Hot Encoding 流程
X_categorical = pd.get_dummies(df[categorical_cols], drop_first=False)
X_numeric = df.drop(columns=categorical_cols + ['label'])
X = pd.concat([X_numeric, X_categorical], axis=1)

# 標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 分割訓練集、驗證集、測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 調整資料形狀以符合 CNN 輸入需求
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),  # Layer 1: 64 filters
    layers.MaxPooling1D(pool_size=2),  # MaxPooling (2x2)
    layers.Dropout(0.3),

    layers.Conv1D(32, kernel_size=3, activation='relu'),  # Layer 2: 32 filters
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Output layer
])

# 編譯模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 訓練模型（根據表格設定）
history = model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_val_cnn, y_val))

# 模型評估
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# 預測
y_pred_prob = model.predict(X_test_cnn)
y_pred_labels = (y_pred_prob >= 0.5).astype(int)

# 計算指標
accuracy = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels)
recall = recall_score(y_test, y_pred_labels)
f1 = f1_score(y_test, y_pred_labels)

# 以表格格式輸出結果
results_df = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-Score': [f1]
})
print("\nModel Performance Metrics:")
print(results_df.to_string(index=False))
