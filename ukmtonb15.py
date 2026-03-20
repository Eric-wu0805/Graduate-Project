import pandas as pd
import numpy as np
import os

try:
    # 讀取原始 IDS20 資料集
    print("正在讀取原始數據...")
    input_file = 'UKM-IDS20 Training set.csv'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到輸入文件: {input_file}")
    
    df = pd.read_csv(input_file)
    
    # 打印原始數據的列名，用於調試
    print("\n原始數據的列名:")
    print(df.columns.tolist())
    
    # 建立新的 DataFrame 符合 UNSW-NB15 欄位命名（NB15）
    unsw_df = pd.DataFrame()
    
    # 列名映射字典
    column_mapping = {
        'dur': ['dur', 'duration', 'time'],
        'proto': ['proto', 'protocol', 'trnspt'],
        'service': ['service', 'srvs', 'app'],
        'state': ['state', 'conn_state'],
        'spkts': ['spkts', 'src_pkts', 'source_packets'],
        'dpkts': ['dpkts', 'dst_pkts', 'dest_packets'],
        'sbytes': ['sbytes', 'src_bytes', 'source_bytes'],
        'dbytes': ['dbytes', 'dst_bytes', 'dest_bytes'],
        'sttl': ['sttl', 'src_ttl', 'source_ttl'],
        'dttl': ['dttl', 'dst_ttl', 'dest_ttl'],
        'sload': ['sload', 'src_load', 'source_load'],
        'dload': ['dload', 'dst_load', 'dest_load'],
        'sloss': ['sloss', 'src_loss', 'source_loss'],
        'dloss': ['dloss', 'dst_loss', 'dest_loss'],
        'sinpkt': ['sinpkt', 'src_inpkt', 'source_inpkt'],
        'dinpkt': ['dinpkt', 'dst_inpkt', 'dest_inpkt'],
        'sjit': ['sjit', 'src_jit', 'source_jit'],
        'djit': ['djit', 'dst_jit', 'dest_jit'],
        'swin': ['swin', 'src_win', 'source_win'],
        'stcpb': ['stcpb', 'src_tcpb', 'source_tcpb'],
        'dtcpb': ['dtcpb', 'dst_tcpb', 'dest_tcpb'],
        'dwin': ['dwin', 'dst_win', 'dest_win'],
        'tcprtt': ['tcprtt', 'tcp_rtt'],
        'synack': ['synack', 'syn_ack'],
        'ackdat': ['ackdat', 'ack_data'],
        'smean': ['smean', 'src_mean', 'source_mean'],
        'dmean': ['dmean', 'dst_mean', 'dest_mean'],
        'trans_depth': ['trans_depth', 'transaction_depth'],
        'response_body_len': ['response_body_len', 'resp_body_len'],
        'ct_srv_src': ['ct_srv_src', 'service_src_count'],
        'ct_state_ttl': ['ct_state_ttl', 'state_ttl_count'],
        'ct_dst_ltm': ['ct_dst_ltm', 'dst_host_count'],
        'ct_src_dport_ltm': ['ct_src_dport_ltm', 'src_dport_count'],
        'ct_dst_sport_ltm': ['ct_dst_sport_ltm', 'dst_sport_count'],
        'ct_dst_src_ltm': ['ct_dst_src_ltm', 'dst_src_count'],
        'is_ftp_login': ['is_ftp_login', 'ftp_login'],
        'ct_ftp_cmd': ['ct_ftp_cmd', 'ftp_cmd_count'],
        'ct_flw_http_mthd': ['ct_flw_http_mthd', 'http_method_count'],
        'ct_src_ltm': ['ct_src_ltm', 'src_host_count'],
        'ct_srv_dst': ['ct_srv_dst', 'service_dst_count'],
        'is_sm_ips_ports': ['is_sm_ips_ports', 'same_ip_port']
    }
    
    # 標籤列映射
    label_mapping = {
        'label': ['label', 'Binary', 'binary', 'is_attack'],
        'attack_cat': ['attack_cat', 'Class', 'attack_type', 'attack_category']
    }
    
    def find_column(df, possible_names):
        """查找可能的列名"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    # 基本列處理
    print("\n處理基本列...")
    unsw_df['id'] = range(1, len(df) + 1)
    
    # 處理每個特徵列
    for nb15_col, possible_names in column_mapping.items():
        col_name = find_column(df, possible_names)
        if col_name:
            unsw_df[nb15_col] = df[col_name]
            print(f"找到列 {col_name} 映射到 {nb15_col}")
        else:
            # 設置默認值
            if nb15_col in ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl']:
                unsw_df[nb15_col] = 0
            elif nb15_col in ['proto', 'service', 'state']:
                unsw_df[nb15_col] = 'unknown'
            else:
                unsw_df[nb15_col] = 0
            print(f"警告：找不到列 {nb15_col}，使用默認值")
    
    # 處理標籤列
    print("\n處理標籤列...")
    for nb15_col, possible_names in label_mapping.items():
        col_name = find_column(df, possible_names)
        if col_name:
            unsw_df[nb15_col] = df[col_name]
            print(f"找到列 {col_name} 映射到 {nb15_col}")
        else:
            if nb15_col == 'label':
                unsw_df[nb15_col] = 0
                print("警告：找不到標籤列，使用默認值 0")
            else:
                unsw_df[nb15_col] = 'normal'
                print("警告：找不到攻擊類型列，使用默認值 'normal'")
    
    # 計算派生特徵
    print("\n計算派生特徵...")
    # 傳輸速率
    unsw_df['rate'] = (unsw_df['sbytes'] + unsw_df['dbytes']) / (unsw_df['dur'] + 1)
    
    # 平均位元組
    unsw_df['src_avg_byts'] = unsw_df['sbytes'] / (unsw_df['spkts'] + 1)
    unsw_df['dst_avg_byts'] = unsw_df['dbytes'] / (unsw_df['dpkts'] + 1)
    
    # 保存轉換結果
    output_file = 'Converted_IDS20_to_NB15.csv'
    print(f"\n正在保存轉換後的數據到 {output_file}...")
    try:
        unsw_df.to_csv(output_file, index=False)
        print(f"轉換完成，已儲存為 {output_file}")
        
        # 打印數據統計信息
        print("\n數據統計信息:")
        print(f"總行數: {len(unsw_df)}")
        print(f"特徵數量: {len(unsw_df.columns)}")
        print("\n標籤分布:")
        print(unsw_df['label'].value_counts())
        print("\n攻擊類型分布:")
        print(unsw_df['attack_cat'].value_counts())
        
    except Exception as e:
        print(f"保存文件時發生錯誤: {str(e)}")
        print(f"當前工作目錄: {os.getcwd()}")
        print(f"嘗試保存到絕對路徑...")
        abs_path = os.path.abspath(output_file)
        unsw_df.to_csv(abs_path, index=False)
        print(f"已保存到: {abs_path}")

except FileNotFoundError as e:
    print(f"錯誤：{str(e)}")
except Exception as e:
    print(f"發生錯誤: {str(e)}")
    print("請檢查輸入文件格式是否正確")
