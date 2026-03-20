# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from keras import models, layers  # Use 'keras' directly if you are using TensorFlow 2.10 or later

# 讀取資料集
training = pd.read_csv('UNSW_NB15_training-set.csv')
testing = pd.read_csv('UNSW_NB15_testing-set.csv')

# 合併 training 和 testing 並移除 id 欄位
df = pd.concat([training, testing]).drop('id', axis=1).reset_index(drop=True)

# 要進行 One-Hot Encoding 的類別欄位
categorical_cols = ['proto', 'service', 'state', 'attack_cat', 
                    'is_ftp_login', 'ct_flw_http_mthd', 'is_sm_ips_ports']

# 分割標籤
y = df['label'].values

# One-Hot Encoding 與保留數值欄位
X_categorical = pd.get_dummies(df[categorical_cols], drop_first=False)
X_numeric = df.drop(columns=categorical_cols + ['label'])
X = pd.concat([X_numeric, X_categorical], axis=1)

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割訓練集、驗證集、測試集（使用標準化後的資料）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 確保格式為 float32（避免 TensorFlow 錯誤）
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)

# 建立人工神經網路模型
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # 二元分類
])

# 編譯模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 訓練模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# 模型評估
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# 預測
y_pred_prob = model.predict(X_test)
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
