import pandas as pd
import os
import numpy as np

# NB15標準欄位
nb15_columns = [
    'id','dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports','attack_cat','label'
]

# CIC2017攻擊名稱 -> NB15攻擊大類
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
    label = str(row['Label']) if 'Label' in row else ''
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
    ]
    all_dfs = []
    for file in files:
        if not os.path.exists(file):
            print(f"找不到檔案: {file}")
            continue
        print(f"讀取: {file}")
        df = pd.read_csv(file)
        # 標籤轉換
        if 'Label' in df.columns:
            df['label'], df['attack_cat'] = zip(*df.apply(convert_label, axis=1))
        else:
            df['label'] = 1
            df['attack_cat'] = 'Unknown'
        # 建立NB15格式DataFrame
        nb15_df = pd.DataFrame()
        for col in nb15_columns:
            # 直接對應欄位
            if col in df.columns:
                nb15_df[col] = df[col]
            # 常見自動對應
            elif col == 'dur' and 'Flow Duration' in df.columns:
                nb15_df[col] = df['Flow Duration']
            elif col == 'spkts' and 'Total Fwd Packets' in df.columns:
                nb15_df[col] = df['Total Fwd Packets']
            elif col == 'dpkts' and 'Total Backward Packets' in df.columns:
                nb15_df[col] = df['Total Backward Packets']
            elif col == 'sbytes' and 'Total Fwd Bytes' in df.columns:
                nb15_df[col] = df['Total Fwd Bytes']
            elif col == 'dbytes' and 'Total Backward Bytes' in df.columns:
                nb15_df[col] = df['Total Backward Bytes']
            elif col == 'smean' and 'Fwd Packet Length Mean' in df.columns:
                nb15_df[col] = df['Fwd Packet Length Mean']
            elif col == 'dmean' and 'Bwd Packet Length Mean' in df.columns:
                nb15_df[col] = df['Bwd Packet Length Mean']
            elif col == 'label':
                nb15_df[col] = df['label']
            elif col == 'attack_cat':
                nb15_df[col] = df['attack_cat']
            else:
                nb15_df[col] = 0
        # 數值轉換
        for col in nb15_df.columns:
            if col not in ['proto','service','state','attack_cat']:
                nb15_df[col] = pd.to_numeric(nb15_df[col], errors='coerce').fillna(0)
        all_dfs.append(nb15_df)
    if not all_dfs:
        print("沒有可合併的資料！")
        return
    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv('cic2017_merged_to_nb15.csv', index=False)
    print("已合併並儲存為 cic2017_merged_to_nb15.csv")

if __name__ == "__main__":
    convert_and_merge_cic2017_to_nb15()
