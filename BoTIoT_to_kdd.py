import pandas as pd
import numpy as np

# 1. 讀取 BoTIoT 欄位名稱
def get_bo_columns():
    with open('data_names.csv', encoding='utf-8') as f:
        return [c.strip() for c in f.readline().strip().split(',') if c.strip()]

# 2. 讀取 BoTIoT 原始資料
def load_bo_data(filename, bo_cols):
    return pd.read_csv(filename, names=bo_cols, header=0)

# 3. 缺值補0，能轉數值就轉
def clean_data(df):
    df = df.fillna(0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

# 4. 自動處理 label（二元化）
def binarize_label(df):
    def label_map(x):
        if str(x).strip().lower() == 'normal':
            return 'normal'
        else:
            return 'attack'
    if 'category' in df.columns:
        df['label'] = df['category'].apply(label_map)
    elif 'attack' in df.columns:
        df['label'] = df['attack'].apply(label_map)
    else:
        raise ValueError("找不到 category 或 attack 欄位，請確認原始資料！")
    return df

# 5. 取得 KDD 欄位順序
def get_kdd_columns():
    with open('kdd_test.csv', encoding='utf-8') as f:
        return [c.strip() for c in f.readline().strip().split(',') if c.strip()]

# 6. 自動補齊缺失欄位
def align_columns(df, kdd_cols):
    for col in kdd_cols:
        if col not in df.columns:
            df[col] = 0
    return df[kdd_cols]

if __name__ == '__main__':
    bo_cols = get_bo_columns()
    df = load_bo_data('BoTIoT.csv', bo_cols)
    df = clean_data(df)
    df = binarize_label(df)
    kdd_cols = get_kdd_columns()
    df = align_columns(df, kdd_cols)
    df.to_csv('BoTIoT_to_kdd.csv', index=False)
    print('轉換完成，已輸出 BoTIoT_to_kdd.csv') 