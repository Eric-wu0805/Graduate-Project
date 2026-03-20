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
X = df.drop(columns=['attack_cat', 'label'])
y = df['label'].values

# 類別特徵進行 One-Hot Encoding
X = pd.get_dummies(X)

# 特徵標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割訓練集、驗證集、測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 調整資料形狀以符合 RNN 輸入需求
X_train_rnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val_rnn = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
X_test_rnn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the RNN model, now with hyperparameters from Table 5
model = models.Sequential([
    layers.SimpleRNN(64, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]), return_sequences=True),
    layers.Dropout(0.2),
    layers.SimpleRNN(64),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 訓練模型（根據表格設定）
history = model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_data=(X_val_rnn, y_val))

# 模型評估
test_loss, test_accuracy = model.evaluate(X_test_rnn, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# 預測
y_pred_prob = model.predict(X_test_rnn)
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

