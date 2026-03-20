import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
import matplotlib.pyplot as plt
import os
#from torch import set_seed

def main():
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

    # 清理無限值和 NaN 值
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_numeric_scaled = X_numeric_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 新增功能：繪製穩定縮放器前後的數值分布圖並保存為圖片
    def plot_distribution_before_after(X_numeric, X_numeric_scaled, numerical_cols, output_dir='plots'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for col in numerical_cols:
            plt.figure(figsize=(12, 6))
            
            # 縮放前
            plt.subplot(1, 2, 1)
            plt.hist(X_numeric[col], bins=50, color='skyblue', alpha=0.7, density=True)
            plt.title(f'{col} Before Scaling')
            plt.xlabel(col)
            plt.ylabel('Count (time)')
            
            # 縮放後
            plt.subplot(1, 2, 2)
            plt.hist(X_numeric_scaled[col], bins=50, color='salmon', alpha=0.7, density=True)
            plt.title(f'{col} After Robust Scaling')
            plt.xlabel(col)
            plt.ylabel('Count (time)')
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'{col}_distribution.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Distribution plot for {col} saved to {output_path}")

    # 執行繪圖功能
    plot_distribution_before_after(X_numeric, X_numeric_scaled, numerical_cols)

if __name__ == "__main__":
    main()