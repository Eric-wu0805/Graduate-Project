import pandas as pd
import numpy as np
import os

def convert_ukm_to_kdd(input_file, output_file):
    """
    將 UKM-IDS20 資料集轉換成 KDD 格式的 CSV
    
    Args:
        input_file: 輸入的UKM-IDS20 CSV文件
        output_file: 輸出的KDD格式CSV文件
    """
    print(f"開始轉換 {input_file} 到 KDD 格式...")
    
    # 讀取 UKM-IDS20 資料
    try:
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            print(f"成功讀取 {input_file}")
        else:
            print(f"找不到檔案: {input_file}")
            return False
        
        print(f"原始資料形狀: {df.shape}")
        print("原始資料列名:")
        print(df.columns.tolist())
        
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return False
    
    # 創建 KDD 格式的 DataFrame
    # KDD 格式的標準列名
    kdd_columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'labels'
    ]
    
    kdd_df = pd.DataFrame(columns=kdd_columns)
    
    # 基本特徵對應
    print("處理基本特徵...")
    kdd_df['duration'] = df.get('dur', 0)
    
    # 協議類型對應
    protocol_mapping = {
        1: 'icmp',
        6: 'tcp', 
        17: 'udp',
        256: 'other'
    }
    kdd_df['protocol_type'] = df.get('trnspt', 6).map(protocol_mapping).fillna('tcp')
    
    # 服務對應
    kdd_df['service'] = df.get('srvs', 'http')
    
    # 連接狀態對應
    flag_mapping = {
        0: 'SF',  # 正常連接
        1: 'S0',  # SYN 攻擊
        2: 'S1',  # SYN 攻擊
        3: 'S2',  # SYN 攻擊
        4: 'S3',  # SYN 攻擊
        5: 'OTH'  # 其他
    }
    kdd_df['flag'] = df.get('flag_n', 0).map(flag_mapping).fillna('SF')
    
    # 位元組數
    kdd_df['src_bytes'] = df.get('src_byts', 0)
    kdd_df['dst_bytes'] = df.get('dst_byts', 0)
    
    # 設置默認值為 0 的欄位
    default_zero_fields = [
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login'
    ]
    
    for field in default_zero_fields:
        kdd_df[field] = 0
    
    # 統計特徵對應
    print("處理統計特徵...")
    kdd_df['count'] = df.get('src_pkts', 0)  # 源端封包數
    kdd_df['srv_count'] = df.get('dst_pkts', 0)  # 目標端封包數
    
    # 錯誤率相關特徵
    kdd_df['serror_rate'] = df.get('flag_sign', 0)  # SYN 錯誤率
    kdd_df['srv_serror_rate'] = df.get('flag_synrst', 0)  # SYN RST 錯誤率
    kdd_df['rerror_rate'] = df.get('flag_arst', 0)  # RST 錯誤率
    kdd_df['srv_rerror_rate'] = df.get('flag_uc', 0)  # 緊急錯誤率
    
    # 服務相關特徵
    kdd_df['same_srv_rate'] = df.get('flag_a', 0)  # 相同服務率
    kdd_df['diff_srv_rate'] = df.get('flag_othr', 0)  # 不同服務率
    kdd_df['srv_diff_host_rate'] = df.get('no_lnkd', 0)  # 服務不同主機率
    
    # 主機相關特徵
    kdd_df['dst_host_count'] = df.get('dst_host_count', 0)
    kdd_df['dst_host_srv_count'] = df.get('host_dst _count', 0)  # 注意空格
    kdd_df['dst_host_same_srv_rate'] = df.get('rtt_first_ack', 0)
    kdd_df['dst_host_diff_srv_rate'] = df.get('rtt_avg', 0)
    kdd_df['dst_host_same_src_port_rate'] = df.get('avg_t_sent', 0)
    kdd_df['dst_host_srv_diff_host_rate'] = df.get('avg_t_got', 0)
    kdd_df['dst_host_serror_rate'] = df.get('repeated', 0)
    kdd_df['dst_host_srv_serror_rate'] = df.get('fst_src_sqc', 0)
    kdd_df['dst_host_rerror_rate'] = df.get('fst_dst_sqc', 0)
    kdd_df['dst_host_srv_rerror_rate'] = df.get('src_re', 0)
    
    # 標籤處理
    print("處理標籤...")
    print("原始標籤分布:")
    print(df['Class'].value_counts())
    
    # 將 UKM-IDS20 的標籤對應到 KDD 格式
    label_mapping = {
        'Normal': 'normal',
        'UDP data flood': 'neptune',
        'ARP poisining': 'smurf',
        'TCP flood': 'back',
        'BeEF HTTP exploits': 'warezclient',
        'Mass HTTP requests': 'teardrop',
        'Metasploit exploits': 'satan',
        'Port scanning': 'nmap'
    }
    
    # 應用標籤映射
    kdd_df['labels'] = df['Class'].map(label_mapping).fillna('normal')
    
    print("轉換後 KDD 格式標籤分布:")
    print(kdd_df['labels'].value_counts())
    
    # 檢查轉換結果
    print(f"轉換後資料形狀: {kdd_df.shape}")
    print("KDD 格式欄位:", kdd_df.columns.tolist())
    
    # 檢查是否有缺失值
    print("\n檢查缺失值:")
    missing_values = kdd_df.isnull().sum()
    if missing_values.sum() > 0:
        print("發現缺失值:")
        print(missing_values[missing_values > 0])
        # 填充缺失值
        kdd_df = kdd_df.fillna(0)
        print("已填充缺失值為 0")
    
    # 保存為 KDD 格式的 CSV
    print(f"\n正在保存轉換後的數據到 {output_file}...")
    try:
        kdd_df.to_csv(output_file, index=False)
        print(f"轉換完成，已儲存為 {output_file}")
        
        # 打印數據統計信息
        print("\n數據統計信息:")
        print(f"總行數: {len(kdd_df)}")
        print(f"特徵數量: {len(kdd_df.columns)}")
        print("\n標籤分布:")
        print(kdd_df['labels'].value_counts())
        
        return True
        
    except Exception as e:
        print(f"保存文件時發生錯誤: {str(e)}")
        print(f"當前工作目錄: {os.getcwd()}")
        print(f"嘗試保存到絕對路徑...")
        abs_path = os.path.abspath(output_file)
        kdd_df.to_csv(abs_path, index=False)
        print(f"已保存到: {abs_path}")
        return True

def main():
    """
    主函數：轉換訓練集和測試集
    """
    print("=== UKM-IDS20 到 KDD 格式轉換程序 ===\n")
    
    # 檢查並轉換測試集
    test_input = 'UKM-IDS20 Testing set.csv'
    test_output = 'ukm_to_kdd_test.csv'
    
    if os.path.exists(test_input):
        print("1. 轉換測試集...")
        success = convert_ukm_to_kdd(test_input, test_output)
        if success:
            print("✓ 測試集轉換成功\n")
        else:
            print("✗ 測試集轉換失敗\n")
    else:
        print(f"✗ 找不到測試集文件: {test_input}\n")
    
    # 檢查並轉換訓練集
    train_input = 'UKM-IDS20 Training set.csv'
    train_output = 'ukm_to_kdd_train.csv'
    
    if os.path.exists(train_input):
        print("2. 轉換訓練集...")
        success = convert_ukm_to_kdd(train_input, train_output)
        if success:
            print("✓ 訓練集轉換成功\n")
        else:
            print("✗ 訓練集轉換失敗\n")
    else:
        print(f"✗ 找不到訓練集文件: {train_input}\n")
    
    print("=== 轉換程序完成 ===")
    
    # 顯示生成的文件信息
    print("\n生成的文件:")
    for file in [test_output, train_output]:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  {file} ({size:.1f} KB)")

if __name__ == "__main__":
    main() 