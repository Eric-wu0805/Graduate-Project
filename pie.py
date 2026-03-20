import matplotlib.pyplot as plt
import matplotlib

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch._dynamo
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
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
def plot_pie(before, after, labels):
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            count = int(round(pct * total / 100.0))
            # 百分比在上，數量在下
            return f"{pct:.1f}%\n{count}"
        return my_autopct

    # 固定顏色 (0 = 藍, 1 = 橘)
    colors = ['#1f77b4', '#ff7f0e']

    # SMOTE 前
    axes[0].pie(
        before,
        labels=labels,
        autopct=make_autopct(before),
        startangle=90,
        colors=colors
    )
    axes[0].set_title("Before SMOTE", fontsize=14)

    # SMOTE 後
    axes[1].pie(
        after,
        labels=labels,
        autopct=make_autopct(after),
        startangle=90,
        colors=colors
    )
    axes[1].set_title("After SMOTE", fontsize=14)

    plt.suptitle("Class Distribution Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # 存檔 (png格式)
    plt.savefig("smote_comparison.png", dpi=300, bbox_inches="tight")

    # 顯示
    plt.show()

def augment_data(X, y, noise_level=0.05):
    X_aug = X.copy()
    y_aug = y.copy()
    noise = np.random.normal(0, noise_level, X.shape)
    X_aug = X_aug + noise
    return X_aug, y_aug
def main():
    # ... 你原本的程式碼 ...
    #set_seed(42)
    # 讀取和預處理數據
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

    for col in ['src_bytes', 'dst_bytes']:
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

    # 計算特徵重要性（使用 XGBoost）
    scale_pos_weight = (df['labels'] == 0).sum() / (df['labels'] == 1).sum()
    xgb = XGBClassifier(
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
    xgb.fit(X, df['labels'])
    feature_importances = xgb.feature_importances_
    # 標準化重要性分數到 [0, 1]
    feature_importances = (feature_importances - feature_importances.min()) / (feature_importances.max() - feature_importances.min() + 1e-6)
    feature_weights = {i: imp for i, imp in enumerate(feature_importances)}
    print("特徵重要性（XGBoost）：", {selected_feature_names[i]: imp for i, imp in feature_weights.items()})

    X_aug, y_aug = augment_data(X, df['labels'].values)
    X = np.vstack([X, X_aug])
    y = np.concatenate([df['labels'].values, y_aug])

    print("\n保存預處理器...")
    joblib.dump(scalers, 'feature_scalers.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(final_scaler, 'final_scaler.pkl')
    joblib.dump(selected_feature_names, 'selected_features.pkl')
    joblib.dump(feature_weights, 'feature_weights.pkl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # ===== 畫出 SMOTE 前後分布 =====
    before_counts = pd.Series(y).value_counts().sort_index()
    after_counts = pd.Series(y_resampled).value_counts().sort_index()

    plot_pie(before_counts, after_counts, labels=['Normal (0)', 'Attack (1)'])
if __name__ == "__main__":
    main()
