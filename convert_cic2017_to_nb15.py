import pandas as pd
import os

# CIC2017 -> NB15 欄位對應（請根據實際需求補齊）
column_map = {
    'Flow Duration': 'dur',
    'Total Fwd Packets': 'sbytes',
    'Total Backward Packets': 'dbytes',
    # ... 其餘欄位 ...
}

# CIC2017攻擊名稱 -> NB15攻擊大類（可根據需要細分）
attack_cat_map = {
    'BENIGN': 'Normal',
    'DoS': 'DoS',
    'PortScan': 'Reconnaissance',
    'DDoS': 'DoS',
    'Bot': 'Backdoor',
    'Brute Force': 'BruteForce',
    'Web Attack': 'Web',
    # ... 其餘攻擊 ...
}

def convert_label(row):
    label = str(row['Label'])
    if label == 'BENIGN':
        return 0, 'Normal'
    for k in attack_cat_map:
        if k != 'BENIGN' and k in label:
            return 1, attack_cat_map[k]
    return 1, 'Other'

def convert_and_merge_cic2017_to_nb15():
    files = [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    ]
    all_dfs = []
    for file in files:
        if not os.path.exists(file):
            print(f"找不到檔案: {file}")
            continue
        print(f"讀取: {file}")
        df = pd.read_csv(file)
        # 欄位對應
        df = df.rename(columns=column_map)
        # 標籤轉換
        if 'Label' in df.columns:
            df['label'], df['attack_cat'] = zip(*df.apply(convert_label, axis=1))
        else:
            print(f"{file} 缺少 Label 欄位，跳過標籤轉換")
            df['label'] = 1
            df['attack_cat'] = 'Unknown'
        # 只保留NB15需要的欄位（可根據NB15格式調整）
        nb15_cols = list(column_map.values()) + ['label', 'attack_cat']
        nb15_cols = [col for col in nb15_cols if col in df.columns]
        df = df[nb15_cols]
        all_dfs.append(df)
    if not all_dfs:
        print("沒有可合併的資料！")
        return
    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv('cic2017_merged_to_nb15.csv', index=False)
    print("已合併並儲存為 cic2017_merged_to_nb15.csv")

if __name__ == "__main__":
    convert_and_merge_cic2017_to_nb15() 