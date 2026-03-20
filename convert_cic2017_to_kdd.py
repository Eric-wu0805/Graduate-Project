import pandas as pd
import numpy as np
from glob import glob
import os

def convert_cic2017_to_kdd():
    """
    將 CIC 2017 資料集轉換成 NSL-KDD 格式的 CSV
    """
    print("開始轉換 CIC 2017 到 NSL-KDD 格式...")
    
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
    df = pd.concat(all_data, ignore_index=True)
    print(f"合併後資料形狀: {df.shape}")
    
    # 創建 NSL-KDD 格式的 DataFrame
    kdd_df = pd.DataFrame()
    
    # 基本特徵對應
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
    kdd_df['hot'] = 0
    kdd_df['num_failed_logins'] = 0
    kdd_df['logged_in'] = 0
    kdd_df['num_compromised'] = 0
    kdd_df['root_shell'] = 0
    kdd_df['su_attempted'] = 0
    kdd_df['num_root'] = 0
    kdd_df['num_file_creations'] = 0
    kdd_df['num_shells'] = 0
    kdd_df['num_access_files'] = 0
    kdd_df['num_outbound_cmds'] = 0
    kdd_df['is_host_login'] = 0
    kdd_df['is_guest_login'] = 0
    
    # 流量統計特徵
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
    print("\n原始 CIC 2017 標籤分布:")
    print(df[' Label'].value_counts())
    
    # CIC 2017 的標籤對應到 NSL-KDD 格式
    label_mapping = {
        'BENIGN': 'normal',
        'DDoS': 'neptune',
        'PortScan': 'ipsweep',
        'Bot': 'satan',
        'Infiltration': 'buffer_overflow',
        'Web Attack – Brute Force': 'guess_passwd',
        'Web Attack – XSS': 'xss',
        'Web Attack – Sql Injection': 'sqlattack'
    }
    
    # 應用標籤映射
    kdd_df['labels'] = df[' Label'].map(label_mapping).fillna('normal')
    
    print("\n轉換後 NSL-KDD 格式標籤分布:")
    print(kdd_df['labels'].value_counts())
    
    # 檢查並處理缺失值
    print("\n檢查缺失值...")
    null_counts = kdd_df.isnull().sum()
    print("缺失值統計:")
    print(null_counts[null_counts > 0])
    
    # 用 0 填充缺失值
    kdd_df = kdd_df.fillna(0)
    
    # 保存為 NSL-KDD 格式的 CSV
    output_file = 'cic2017_to_kdd.csv'
    kdd_df.to_csv(output_file, index=False)
    print(f"\n已保存 {output_file} ({len(kdd_df)} 筆)")
    
    print("轉換完成！")

if __name__ == "__main__":
    convert_cic2017_to_kdd() 