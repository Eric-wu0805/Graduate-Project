import pandas as pd
import numpy as np
import os
from datetime import datetime

# UNSW-NB15 標準欄位
nb15_columns = [
    'id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 
    'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 
    'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 
    'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 
    'attack_cat', 'label'
]

# CIC2017 攻擊名稱 -> NB15 攻擊大類映射
attack_cat_map = {
    'BENIGN': 'Normal',
    'DoS': 'DoS',
    'PortScan': 'Reconnaissance', 
    'DDoS': 'DoS',
    'Bot': 'Backdoor',
    'Brute Force': 'BruteForce',
    'Web Attack': 'Web',
    'Infiltration': 'Exploits',
    'Web Attack – Brute Force': 'BruteForce',
    'Web Attack – XSS': 'Web',
    'Web Attack – Sql Injection': 'Web',
    'Heartbleed': 'Exploits'
}

# CIC2017 欄位 -> NB15 欄位映射
field_mapping = {
    'Flow Duration': 'dur',
    'Total Fwd Packets': 'spkts', 
    'Total Backward Packets': 'dpkts',
    'Total Length of Fwd Packets': 'sbytes',
    'Total Length of Bwd Packets': 'dbytes',
    'Fwd Packet Length Mean': 'smean',
    'Bwd Packet Length Mean': 'dmean',
    'Flow Bytes/s': 'rate',
    'Flow Packets/s': 'rate_pkts',
    'Flow IAT Mean': 'sinpkt',
    'Fwd IAT Mean': 'sinpkt',
    'Bwd IAT Mean': 'dinpkt',
    'Fwd IAT Std': 'sjit',
    'Bwd IAT Std': 'djit',
    'Fwd Header Length': 'swin',
    'Bwd Header Length': 'dwin',
    'Init_Win_bytes_forward': 'swin',
    'Init_Win_bytes_backward': 'dwin',
    'Fwd Avg Bytes/Bulk': 'stcpb',
    'Bwd Avg Bytes/Bulk': 'dtcpb',
    'Active Mean': 'tcprtt',
    'Idle Mean': 'synack'
}

def convert_label(row):
    """將 CIC2017 標籤轉換為 NB15 格式 - 正常為0，其他為1"""
    label = str(row['Label']) if 'Label' in row else ''
    label = label.strip()
    
    if label == 'BENIGN':
        return 0, 'Normal'  # 正常流量標籤為0
    
    for attack_type, nb15_category in attack_cat_map.items():
        if attack_type != 'BENIGN' and attack_type in label:
            return 1, nb15_category  # 攻擊流量標籤為1
    
    return 1, 'Other'  # 未知類型標籤為1

def calculate_derived_features(df):
    """計算派生特徵"""
    features = {}
    
    # 計算傳輸速率 (bytes/s)
    if 'dur' in df.columns and 'sbytes' in df.columns and 'dbytes' in df.columns:
        features['rate'] = (df['sbytes'] + df['dbytes']) / (df['dur'] + 1e-6)
    
    # 計算 TTL 相關特徵 (如果沒有直接對應)
    if 'sttl' not in df.columns:
        features['sttl'] = 64  # 預設 TTL 值
    if 'dttl' not in df.columns:
        features['dttl'] = 64
    
    # 計算負載相關特徵
    if 'sload' not in df.columns and 'sbytes' in df.columns and 'dur' in df.columns:
        features['sload'] = df['sbytes'] / (df['dur'] + 1e-6)
    if 'dload' not in df.columns and 'dbytes' in df.columns and 'dur' in df.columns:
        features['dload'] = df['dbytes'] / (df['dur'] + 1e-6)
    
    # 計算封包丟失率 (預設為0)
    if 'sloss' not in df.columns:
        features['sloss'] = 0
    if 'dloss' not in df.columns:
        features['dloss'] = 0
    
    return features

def convert_cic2017_to_nb15(df, file_name):
    """將單個 CIC2017 數據集轉換為 NB15 格式"""
    print(f"正在轉換 {file_name}...")
    
    # 清理列名 (移除空格)
    df.columns = df.columns.str.strip()
    
    # 創建 NB15 格式的 DataFrame，預先初始化所有欄位
    nb15_df = pd.DataFrame(index=range(len(df)))
    
    # 初始化所有NB15欄位為預設值
    for col in nb15_columns:
        if col in ['proto', 'service', 'state']:
            nb15_df[col] = 'unknown'
        elif col == 'attack_cat':
            nb15_df[col] = 'Other'
        else:
            nb15_df[col] = 0
    
    # 添加 ID 欄位
    nb15_df['id'] = range(len(df))
    
    # 標籤轉換 - 確保正常為0，其他為1
    if 'Label' in df.columns:
        nb15_df['label'], nb15_df['attack_cat'] = zip(*df.apply(convert_label, axis=1))
    else:
        print(f"警告: {file_name} 缺少 Label 欄位")
        nb15_df['label'] = 1  # 預設為攻擊
        nb15_df['attack_cat'] = 'Unknown'
    
    # 欄位映射
    for cic_col, nb15_col in field_mapping.items():
        if cic_col in df.columns:
            nb15_df[nb15_col] = df[cic_col]
    
    # 計算派生特徵
    derived_features = calculate_derived_features(nb15_df)
    for col, values in derived_features.items():
        nb15_df[col] = values
    
    # 確保所有數值欄位為數值型態並補0
    numeric_columns = [col for col in nb15_columns if col not in ['proto', 'service', 'state', 'attack_cat']]
    for col in numeric_columns:
        nb15_df[col] = pd.to_numeric(nb15_df[col], errors='coerce').fillna(0)
    
    # 最終檢查：確保沒有缺失值
    nb15_df = nb15_df.fillna(0)
    
    # 確保所有NB15欄位都存在並按順序排列
    for col in nb15_columns:
        if col not in nb15_df.columns:
            if col in ['proto', 'service', 'state']:
                nb15_df[col] = 'unknown'
            elif col == 'attack_cat':
                nb15_df[col] = 'Other'
            else:
                nb15_df[col] = 0
    
    # 按 NB15 欄位順序排列
    nb15_df = nb15_df[nb15_columns]
    
    print(f"轉換完成: {file_name} -> {len(nb15_df)} 筆記錄")
    return nb15_df

def merge_cic2017_datasets():
    """合併所有 CIC2017 數據集並轉換為 NB15 格式"""
    print("開始轉換 CIC2017 數據集為 NB15 格式...")
    print("=" * 60)
    
    # 8個 CIC2017 數據集文件
    cic2017_files = [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv', 
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
    ]
    
    all_dfs = []
    total_records = 0
    
    for file_name in cic2017_files:
        if not os.path.exists(file_name):
            print(f"警告: 找不到文件 {file_name}")
            continue
            
        try:
            print(f"\n正在讀取 {file_name}...")
            df = pd.read_csv(file_name)
            print(f"原始數據: {len(df)} 筆記錄")
            
            # 轉換為 NB15 格式
            nb15_df = convert_cic2017_to_nb15(df, file_name)
            all_dfs.append(nb15_df)
            total_records += len(nb15_df)
            
        except Exception as e:
            print(f"處理 {file_name} 時發生錯誤: {e}")
            continue
    
    if not all_dfs:
        print("錯誤: 沒有成功轉換任何數據集！")
        return
    
    # 合併所有數據集
    print(f"\n正在合併 {len(all_dfs)} 個數據集...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # 重新分配 ID
    merged_df['id'] = range(len(merged_df))
    
    # 最終確保沒有缺失值
    merged_df = merged_df.fillna(0)
    
    # 顯示統計信息
    print(f"\n合併完成！")
    print(f"總記錄數: {len(merged_df):,}")
    print(f"總欄位數: {len(merged_df.columns)}")
    
    # 顯示標籤分布
    print(f"\n標籤分布:")
    label_counts = merged_df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(merged_df)) * 100
        print(f"  Label {label}: {count:,} ({percentage:.2f}%)")
    
    print(f"\n攻擊類型分布:")
    attack_counts = merged_df['attack_cat'].value_counts()
    for attack, count in attack_counts.items():
        percentage = (count / len(merged_df)) * 100
        print(f"  {attack}: {count:,} ({percentage:.2f}%)")
    
    # 檢查缺失值
    print(f"\n缺失值檢查:")
    null_counts = merged_df.isnull().sum()
    if null_counts.sum() > 0:
        print("發現缺失值:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  {col}: {count}")
    else:
        print("沒有缺失值")
    
    # 保存合併後的數據集
    output_file = 'cic2017_merged_to_nb15.csv'
    print(f"\n正在保存到 {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    print(f"保存完成！文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # 顯示前幾行數據作為驗證
    print(f"\n前5行數據預覽:")
    print(merged_df.head())
    
    return merged_df

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"開始時間: {start_time}")
    
    try:
        merged_data = merge_cic2017_datasets()
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n完成時間: {end_time}")
        print(f"總耗時: {duration}")
        
    except Exception as e:
        print(f"程序執行時發生錯誤: {e}")
        import traceback
        traceback.print_exc() 