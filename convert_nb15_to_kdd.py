import pandas as pd
import numpy as np

def convert_nb15_to_kdd():
    """
    將 UNSW-NB15 資料集轉換成 KDD 格式的 CSV
    """
    print("開始轉換 UNSW-NB15 到 KDD 格式...")
    
    # 讀取 UNSW-NB15 資料
    try:
        training = pd.read_csv('UNSW_NB15_training-set.csv')
        testing = pd.read_csv('UNSW_NB15_testing-set.csv')
        print("成功讀取 UNSW-NB15 資料")
    except FileNotFoundError as e:
        print(f"找不到檔案: {e}")
        return
    
    # 合併訓練和測試資料
    df = pd.concat([training, testing]).reset_index(drop=True)
    print(f"合併後資料形狀: {df.shape}")
    
    # 先印出 UNSW-NB15 的實際欄位名稱
    print("UNSW-NB15 實際欄位名稱:")
    print(df.columns.tolist())
    
    # UNSW-NB15 欄位對應到 KDD 格式
    # KDD 欄位順序: duration, protocol_type, service, flag, src_bytes, dst_bytes, land, 
    # wrong_fragment, urgent, hot, num_failed_logins, logged_in, num_compromised, 
    # root_shell, su_attempted, num_root, num_file_creations, num_shells, 
    # num_access_files, num_outbound_cmds, is_host_login, is_guest_login, 
    # count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, 
    # same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count, 
    # dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, 
    # dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, 
    # dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, 
    # dst_host_srv_rerror_rate, labels
    
    # 創建 KDD 格式的 DataFrame
    kdd_df = pd.DataFrame()
    
    # 基本特徵對應 - 使用安全的對應方式
    kdd_df['duration'] = df.get('dur', 0)
    kdd_df['protocol_type'] = df.get('proto', 'tcp')
    kdd_df['service'] = df.get('service', 'http')
    kdd_df['flag'] = df.get('state', 'SF')
    kdd_df['src_bytes'] = df.get('sbytes', 0)
    kdd_df['dst_bytes'] = df.get('dbytes', 0)
    kdd_df['land'] = 0  # UNSW-NB15 沒有 land 欄位，設為 0
    kdd_df['wrong_fragment'] = 0  # 設為 0
    kdd_df['urgent'] = 0  # 設為 0
    kdd_df['hot'] = 0  # 設為 0
    kdd_df['num_failed_logins'] = 0  # 設為 0
    kdd_df['logged_in'] = 0  # 設為 0
    kdd_df['num_compromised'] = 0  # 設為 0
    kdd_df['root_shell'] = 0  # 設為 0
    kdd_df['su_attempted'] = 0  # 設為 0
    kdd_df['num_root'] = 0  # 設為 0
    kdd_df['num_file_creations'] = 0  # 設為 0
    kdd_df['num_shells'] = 0  # 設為 0
    kdd_df['num_access_files'] = 0  # 設為 0
    kdd_df['num_outbound_cmds'] = 0  # 設為 0
    kdd_df['is_host_login'] = 0  # 設為 0
    kdd_df['is_guest_login'] = 0  # 設為 0
    
    # 統計特徵對應 - 使用安全的對應方式
    kdd_df['count'] = df.get('sttl', 0)
    kdd_df['srv_count'] = df.get('dttl', 0)
    kdd_df['serror_rate'] = df.get('sload', 0)
    kdd_df['srv_serror_rate'] = df.get('dload', 0)
    kdd_df['rerror_rate'] = df.get('sloss', 0)
    kdd_df['srv_rerror_rate'] = df.get('dloss', 0)
    kdd_df['same_srv_rate'] = df.get('sinpkt', 0)
    kdd_df['diff_srv_rate'] = df.get('dinpkt', 0)
    kdd_df['srv_diff_host_rate'] = df.get('sjit', 0)
    kdd_df['dst_host_count'] = df.get('djit', 0)
    kdd_df['dst_host_srv_count'] = df.get('swin', 0)
    kdd_df['dst_host_same_srv_rate'] = df.get('dwin', 0)
    kdd_df['dst_host_diff_srv_rate'] = df.get('stcpb', 0)
    kdd_df['dst_host_same_src_port_rate'] = df.get('dtcpb', 0)
    kdd_df['dst_host_srv_diff_host_rate'] = df.get('smeansz', 0)
    kdd_df['dst_host_serror_rate'] = df.get('dmeansz', 0)
    kdd_df['dst_host_srv_serror_rate'] = df.get('trans_depth', 0)
    kdd_df['dst_host_rerror_rate'] = df.get('response_body_len', 0)
    kdd_df['dst_host_srv_rerror_rate'] = df.get('ct_srv_src', 0)
    
    # 標籤處理
    print("原始 UNSW-NB15 標籤分布:")
    print(df['attack_cat'].value_counts())
    
    # 將 UNSW-NB15 的 attack_cat 對應到 KDD 格式的 labels
    # UNSW-NB15 的標籤對應到 KDD 格式
    label_mapping = {
        'Normal': 'normal',
        'Generic': 'neptune',
        'Exploits': 'satan',
        'Fuzzers': 'ipsweep',
        'DoS': 'smurf',
        'Reconnaissance': 'nmap',
        'Analysis': 'portsweep',
        'Backdoor': 'back',
        'Shellcode': 'buffer_overflow',
        'Worms': 'warezclient'
    }
    
    # 應用標籤映射
    kdd_df['labels'] = df['attack_cat'].map(label_mapping).fillna('normal')
    
    print("轉換後 KDD 格式標籤分布:")
    print(kdd_df['labels'].value_counts())
    
    # 檢查轉換結果
    print(f"轉換後資料形狀: {kdd_df.shape}")
    print("KDD 格式欄位:", kdd_df.columns.tolist())
    
    # 保存為 KDD 格式的 CSV（不分割）
    kdd_df.to_csv('nb15_kdd_train.csv', index=False)
    
    print(f"已保存 nb15_kdd_train.csv ({len(kdd_df)} 筆)")
    print("轉換完成！")

if __name__ == "__main__":
    convert_nb15_to_kdd() 