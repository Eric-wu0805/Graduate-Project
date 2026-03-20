import pandas as pd
import numpy as np

def convert_mqttiot_to_kdd():
    """
    將 MQTTIoT 資料集轉換成 KDD 格式的 CSV
    能數值轉換就轉換，不能轉換就補0
    """
    print("開始轉換 MQTTIoT 到 KDD 格式...")
    
    # 讀取 MQTTIoT 資料
    try:
        df = pd.read_csv('MQTTIoT.csv')
        print("成功讀取 MQTTIoT 資料")
        print(f"原始資料形狀: {df.shape}")
        
        # 依照指定數量分別抽樣
        sample_counts = {
            'legitimate': 330204,
            'flood': 58356,
            'slowite': 44864,
            'malformed': 35295,
            'dos': 16455,
            'bruteforce': 9960
        }
        sampled_df_list = []
        for label, count in sample_counts.items():
            sub = df[df['target'].str.lower() == label]
            if len(sub) < count:
                print(f"警告: {label} 資料只有 {len(sub)} 筆，不足 {count}，將全部取用。")
                sampled_df_list.append(sub)
            else:
                sampled_df_list.append(sub.sample(n=count, random_state=42))
        df = pd.concat(sampled_df_list).reset_index(drop=True)
        print(f"依照指定分布抽樣後資料形狀: {df.shape}")
    except FileNotFoundError as e:
        print(f"找不到檔案: {e}")
        return
    
    # 印出 MQTTIoT 的欄位名稱
    print("MQTTIoT 欄位名稱:")
    print(df.columns.tolist())
    
    # KDD 格式的標準欄位順序
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
    
    # 創建 KDD 格式的 DataFrame
    kdd_df = pd.DataFrame()
    
    # 基本特徵對應 - 嘗試從 MQTTIoT 中找到對應的欄位
    # duration - 使用 tcp.time_delta 作為持續時間
    kdd_df['duration'] = df.get('tcp.time_delta', 0)
    
    # protocol_type - MQTT 通常使用 TCP，所以設為 tcp
    kdd_df['protocol_type'] = 'tcp'
    
    # service - MQTT 服務
    kdd_df['service'] = 'mqtt'
    
    # flag - 使用 tcp.flags 轉換
    def convert_tcp_flags(flags):
        if pd.isna(flags) or flags == 0:
            return 'SF'
        # 簡單的標誌轉換邏輯
        if '0x00000010' in str(flags):
            return 'SF'
        elif '0x00000018' in str(flags):
            return 'S1'
        elif '0x00000014' in str(flags):
            return 'S2'
        else:
            return 'SF'
    
    kdd_df['flag'] = df['tcp.flags'].apply(convert_tcp_flags)
    
    # src_bytes 和 dst_bytes - 使用 tcp.len 作為源字節數
    kdd_df['src_bytes'] = df.get('tcp.len', 0)
    kdd_df['dst_bytes'] = df.get('tcp.len', 0)  # 簡化處理
    
    # 其他基本特徵設為 0
    kdd_df['land'] = 0
    kdd_df['wrong_fragment'] = 0
    kdd_df['urgent'] = 0
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
    
    # 統計特徵對應 - 嘗試從 MQTTIoT 中找到對應的欄位
    # count - 使用 mqtt.len 作為計數
    kdd_df['count'] = df.get('mqtt.len', 0)
    
    # srv_count - 使用 mqtt.msgid 作為服務計數
    kdd_df['srv_count'] = df.get('mqtt.msgid', 0)
    
    # serror_rate - 使用 mqtt.conack.flags 作為錯誤率
    kdd_df['serror_rate'] = df.get('mqtt.conack.flags', 0)
    
    # srv_serror_rate - 使用 mqtt.conack.flags.reserved
    kdd_df['srv_serror_rate'] = df.get('mqtt.conack.flags.reserved', 0)
    
    # rerror_rate - 使用 mqtt.conack.flags.sp
    kdd_df['rerror_rate'] = df.get('mqtt.conack.flags.sp', 0)
    
    # srv_rerror_rate - 使用 mqtt.conack.val
    kdd_df['srv_rerror_rate'] = df.get('mqtt.conack.val', 0)
    
    # same_srv_rate - 使用 mqtt.conflag.cleansess
    kdd_df['same_srv_rate'] = df.get('mqtt.conflag.cleansess', 0)
    
    # diff_srv_rate - 使用 mqtt.conflag.passwd
    kdd_df['diff_srv_rate'] = df.get('mqtt.conflag.passwd', 0)
    
    # srv_diff_host_rate - 使用 mqtt.conflag.qos
    kdd_df['srv_diff_host_rate'] = df.get('mqtt.conflag.qos', 0)
    
    # dst_host_count - 使用 mqtt.conflag.reserved
    kdd_df['dst_host_count'] = df.get('mqtt.conflag.reserved', 0)
    
    # dst_host_srv_count - 使用 mqtt.conflag.retain
    kdd_df['dst_host_srv_count'] = df.get('mqtt.conflag.retain', 0)
    
    # dst_host_same_srv_rate - 使用 mqtt.conflag.uname
    kdd_df['dst_host_same_srv_rate'] = df.get('mqtt.conflag.uname', 0)
    
    # dst_host_diff_srv_rate - 使用 mqtt.conflag.willflag
    kdd_df['dst_host_diff_srv_rate'] = df.get('mqtt.conflag.willflag', 0)
    
    # dst_host_same_src_port_rate - 使用 mqtt.conflags
    kdd_df['dst_host_same_src_port_rate'] = df.get('mqtt.conflags', 0)
    
    # dst_host_srv_diff_host_rate - 使用 mqtt.dupflag
    kdd_df['dst_host_srv_diff_host_rate'] = df.get('mqtt.dupflag', 0)
    
    # dst_host_serror_rate - 使用 mqtt.hdrflags
    kdd_df['dst_host_serror_rate'] = df.get('mqtt.hdrflags', 0)
    
    # dst_host_srv_serror_rate - 使用 mqtt.kalive
    kdd_df['dst_host_srv_serror_rate'] = df.get('mqtt.kalive', 0)
    
    # dst_host_rerror_rate - 使用 mqtt.msg
    kdd_df['dst_host_rerror_rate'] = df.get('mqtt.msg', 0)
    
    # dst_host_srv_rerror_rate - 使用 mqtt.msgtype
    kdd_df['dst_host_srv_rerror_rate'] = df.get('mqtt.msgtype', 0)
    
    # 標籤處理
    print("原始 MQTTIoT 標籤分布:")
    print(df['target'].value_counts())
    
    # 將 MQTTIoT 的 target 對應到 KDD 格式的 labels
    # MQTTIoT 的標籤對應到 KDD 格式
    label_mapping = {
        'legitimate': 'normal',
        'dos': 'smurf',  # DoS 攻擊對應到 smurf
        'slowite': 'neptune',  # 慢速攻擊對應到 neptune
        'flood': 'back',  # 洪水攻擊對應到 back
        'bruteforce': 'guess_passwd',  # 暴力破解對應到 guess_passwd
        'malformed': 'buffer_overflow'  # 畸形包對應到 buffer_overflow
    }
    
    # 應用標籤映射
    kdd_df['labels'] = df['target'].map(label_mapping).fillna('normal')
    
    print("轉換後 KDD 格式標籤分布:")
    print(kdd_df['labels'].value_counts())
    
    # 確保所有數值欄位都是數值類型
    numeric_columns = [col for col in kdd_df.columns if col not in ['protocol_type', 'service', 'flag', 'labels']]
    for col in numeric_columns:
        kdd_df[col] = pd.to_numeric(kdd_df[col], errors='coerce').fillna(0)
    
    # 檢查轉換結果
    print(f"轉換後資料形狀: {kdd_df.shape}")
    print("KDD 格式欄位:", kdd_df.columns.tolist())
    
    # 檢查數值範圍
    print("\n數值欄位統計:")
    for col in numeric_columns[:5]:  # 只顯示前5個欄位
        print(f"{col}: min={kdd_df[col].min()}, max={kdd_df[col].max()}, mean={kdd_df[col].mean():.2f}")
    
    # 保存為 KDD 格式的 CSV
    output_filename = 'mqttiot_to_kdd.csv'
    kdd_df.to_csv(output_filename, index=False)
    
    print(f"\n已保存 {output_filename} ({len(kdd_df)} 筆)")
    print("轉換完成！")
    
    # 顯示轉換摘要
    print(f"\n轉換摘要:")
    print(f"- 原始資料筆數: {len(df)}")
    print(f"- 轉換後資料筆數: {len(kdd_df)}")
    print(f"- 數值欄位數: {len(numeric_columns)}")
    print(f"- 分類欄位數: 4 (protocol_type, service, flag, labels)")

if __name__ == "__main__":
    convert_mqttiot_to_kdd() 