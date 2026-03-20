import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from random import seed as set_seed

def plot_data_distribution(data_before, data_after, columns, title_prefix, font='Times New Roman'):
    plt.rc('font', family=font)  # 設置全局字體為 Times New Roman
    for col in columns:
        plt.figure(figsize=(12, 5))
        
        # Preprocess data to handle inf values
        data_before_clean = data_before[col].replace([np.inf, -np.inf], np.nan).dropna()
        data_after_clean = data_after[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Before transformation
        plt.subplot(1, 2, 1)
        plt.hist(data_before_clean, bins=30, density=True, alpha=0.7)
        plt.title(f'{title_prefix} {col} (Before Log)', fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.ylabel('Density', fontsize=10)
        
        # After transformation (for src_bytes and dst_bytes)
        if col in ['src_bytes', 'dst_bytes']:
            plt.subplot(1, 2, 2)
            plt.hist(data_after_clean, bins=30, density=True, alpha=0.7)
            plt.title(f'{title_prefix} {col} (After Log)', fontsize=12)
            plt.xlabel(col, fontsize=10)
            plt.ylabel('Density', fontsize=10)
        
        plt.tight_layout()
        plt.show()

def main():
    set_seed(42)
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

    # Save data before log transformation for plotting
    X_numeric_before = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    
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

    # Apply log transformation
    for col in ['src_bytes', 'dst_bytes']:
        if col in X_numeric.columns:
            X_numeric[col] = np.log1p(X_numeric[col])
    
    # Plot data distributions
    plot_data_distribution(X_numeric_before, X_numeric, numerical_cols, 'Distribution of')

if __name__ == "__main__":
    main()