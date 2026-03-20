import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

plt.rcParams['font.family'] = 'Times New Roman'  # ✅ 設定字體

def main():
   
    # === 讀取資料 ===
    training = pd.read_csv('kdd_test.csv')
    testing = pd.read_csv('kdd_train.csv')
    df = pd.concat([training, testing]).reset_index(drop=True)

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    df = df.drop(['attack_cat'], axis=1, errors='ignore')
    df['labels'] = (df['labels'] != 'normal').astype(int)

    numerical_cols = ['src_bytes', 'dst_bytes', 'duration', 'same_srv_rate']
    categorical_cols = ['protocol_type', 'flag']

    # === 類別欄位編碼 ===
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = df[col].astype(str).fillna('')
        df[col] = label_encoders[col].fit_transform(df[col])

    # === 數值欄位轉數值 & 對數轉換 ===
    X_numeric = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    for col in ['src_bytes', 'dst_bytes']:
        if col in X_numeric.columns:
            X_numeric[col] = np.log1p(X_numeric[col])

    # === 處理離群值 ===
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
        return df_clean

    X_numeric = handle_outliers(X_numeric, numerical_cols, method='iqr')

        # === RobustScaler 前後分布圖 ===
    fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(10, 8))
    fig.suptitle('Distribution Before and After RobustScaler', fontsize=14, fontweight='bold')

    X_numeric_scaled = pd.DataFrame(index=X_numeric.index)
    scalers = {}

    for i, col in enumerate(numerical_cols):
        # 原始分布
        axes[i, 0].hist(X_numeric[col], bins=40, color='skyblue', edgecolor='black')
        axes[i, 0].set_title(f'{col} (Before)', fontsize=10)
        axes[i, 0].set_xlabel(col, fontsize=9)
        axes[i, 0].set_ylabel('Frequency', fontsize=9)

        # 進行 RobustScaler
        scalers[col] = RobustScaler()
        X_numeric_scaled[col] = scalers[col].fit_transform(X_numeric[[col]]).ravel()

        # 標準化後分布
        axes[i, 1].hist(X_numeric_scaled[col], bins=40, color='lightgreen', edgecolor='black')
        axes[i, 1].set_title(f'{col} (After)', fontsize=10)
        axes[i, 1].set_xlabel(col, fontsize=9)
        axes[i, 1].set_ylabel('Frequency', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ✅ 儲存成 PNG 圖檔（高解析度）
    plt.savefig('robust_scaler_distribution.png', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main()
