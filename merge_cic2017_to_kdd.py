import pandas as pd
import numpy as np
from glob import glob
import os

def merge_and_convert_cic2017_to_kdd():
    """
    合併所有 CIC 2017 文件並轉換為 NSL-KDD 格式
    """
    print("開始合併和轉換 CIC 2017 到 NSL-KDD 格式...")
    
    # 讀取所有 CIC 2017 資料檔案
    cic_files = [
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Monday-WorkingHours.pcap_ISCX.csv'
    ]
    
    all_data = []
    for file in cic_files:
        try:
            print(f"正在讀取 {file}...")
            df = pd.read_csv(file)
            # 移除列名中的空格
            df.columns = df.columns.str.strip()
            all_data.append(df)
            print(f"成功讀取 {file}, 形狀: {df.shape}")
        except FileNotFoundError:
            print(f"找不到檔案: {file}")
            continue
        except Exception as e:
            print(f"讀取 {file} 時發生錯誤: {e}")
            continue
    
    if not all_data:
        print("沒有成功讀取任何檔案！")
        return
    
    # 合併所有資料
    print("\n合併所有資料...")
    df = pd.concat(all_data, ignore_index=True)
    print(f"合併後資料形狀: {df.shape}")
    
    # 創建 NSL-KDD 格式的 DataFrame
    kdd_df = pd.DataFrame()
    
    # 基本特徵對應
    print("\n轉換基本特徵...")
    kdd_df['duration'] = df['Flow Duration']  # 流量持續時間
    kdd_df['protocol_type'] = 'tcp'  # 假設都是 TCP
    kdd_df['service'] = 'http'  # 假設都是 HTTP
    kdd_df['flag'] = 'SF'  # 假設連接狀態都是 SF (normal establishment)
    kdd_df['src_bytes'] = df['Total Fwd Packets']
    kdd_df['dst_bytes'] = df['Total Backward Packets']
    kdd_df['land'] = 0  # 假設沒有 land 攻擊
    kdd_df['wrong_fragment'] = df['Fwd Packets/s']
    kdd_df['urgent'] = 0
    
    # 內容特徵
    print("設置內容特徵...")
    content_features = [
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login'
    ]
    for feature in content_features:
        kdd_df[feature] = 0
    
    # 流量統計特徵
    print("轉換流量統計特徵...")
    kdd_df['count'] = df['Flow Packets/s']
    kdd_df['srv_count'] = df['Flow Bytes/s']
    kdd_df['serror_rate'] = df['Flow IAT Mean']
    kdd_df['srv_serror_rate'] = df['Flow IAT Std']
    kdd_df['rerror_rate'] = df['Flow IAT Max']
    kdd_df['srv_rerror_rate'] = df['Flow IAT Min']
    kdd_df['same_srv_rate'] = df['Fwd IAT Mean']
    kdd_df['diff_srv_rate'] = df['Fwd IAT Std']
    kdd_df['srv_diff_host_rate'] = df['Fwd IAT Max']
    
    # 主機統計特徵
    print("轉換主機統計特徵...")
    kdd_df['dst_host_count'] = df['Bwd IAT Mean']
    kdd_df['dst_host_srv_count'] = df['Bwd IAT Std']
    kdd_df['dst_host_same_srv_rate'] = df['Bwd IAT Max']
    kdd_df['dst_host_diff_srv_rate'] = df['Bwd IAT Min']
    kdd_df['dst_host_same_src_port_rate'] = df['Active Mean']
    kdd_df['dst_host_srv_diff_host_rate'] = df['Active Std']
    kdd_df['dst_host_serror_rate'] = df['Active Max']
    kdd_df['dst_host_srv_serror_rate'] = df['Active Min']
    kdd_df['dst_host_rerror_rate'] = df['Idle Mean']
    kdd_df['dst_host_srv_rerror_rate'] = df['Idle Std']
    
    # 標籤處理
    print("\n處理標籤...")
    print("原始 CIC 2017 標籤分布:")
    print(df['Label'].value_counts())
    
    # 將標籤轉換為二分類
    print("\n轉換為二分類...")
    kdd_df['labels'] = df['Label'].apply(lambda x: 'normal' if x == 'BENIGN' else 'attack')
    
    print("\n轉換後標籤分布:")
    print(kdd_df['labels'].value_counts())
    
    # 檢查並處理缺失值
    print("\n檢查缺失值...")
    null_counts = kdd_df.isnull().sum()
    print("缺失值統計:")
    print(null_counts[null_counts > 0])
    
    # 用 0 填充缺失值
    kdd_df = kdd_df.fillna(0)
    
    # 保存為 NSL-KDD 格式的 CSV
    output_file = 'cic2017_merged_to_kdd.csv'
    kdd_df.to_csv(output_file, index=False)
    print(f"\n已保存 {output_file} ({len(kdd_df)} 筆)")
    
    # 顯示最終統計信息
    print("\n最終資料統計:")
    print(f"總樣本數: {len(kdd_df)}")
    print(f"特徵數量: {len(kdd_df.columns)}")
    print("\n標籤分布:")
    print(kdd_df['labels'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    
    print("\n轉換完成！")

if __name__ == "__main__":
    merge_and_convert_cic2017_to_kdd() 