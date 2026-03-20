import pandas as pd

# 載入原始 CSV 檔案
df = pd.read_csv("UKM-IDS20 Testing set.csv")

# 修正欄位名稱中的空白（如有）
df.columns = df.columns.str.strip()

# 建立欄位對應字典
column_mapping = {
    'dur': 'duration',
    'trnspt': 'protocol_type',
    'srvs': 'service',
    'flag_n': 'flag',
    'flag_arst': 'src_bytes',
    'flag_uc': 'dst_bytes',
    'flag_sign': 'land',
    'flag_synrst': 'wrong_fragment',
    'flag_a': 'urgent',
    'flag_othr': 'hot',
    'src_pkts': 'num_failed_logins',
    'dst_pkts': 'logged_in',
    'urg_bits': 'num_compromised',
    'push_pkts': 'root_shell',
    'no_lnkd': 'su_attempted',
    'arp': 'num_root',
    'src_ttl': 'num_file_creations',
    'dst_ttl': 'num_shells',
    'pkts_dirctn': 'num_access_files',
    'src_byts': 'num_outbound_cmds',
    'dst_byts': 'is_host_login',
    'src_avg_byts': 'is_guest_login',
    'dst_avg_byts': 'count',
    'strt_t': 'srv_count',
    'end_t': 'serror_rate',
    'dst_host_count': 'srv_serror_rate',
    'host_dst _count': 'rerror_rate',
    'rtt_first_ack': 'srv_rerror_rate',
    'rtt_avg': 'same_srv_rate',
    'avg_t_sent': 'diff_srv_rate',
    'avg_t_got': 'srv_diff_host_rate',
    'repeated': 'dst_host_same_srv_rate',
    'fst_src_sqc': 'dst_host_diff_srv_rate',
    'fst_dst_sqc': 'dst_host_same_src_port_rate',
    'src_re': 'dst_host_srv_diff_host_rate',
    'dst_re': 'dst_host_serror_rate',
    'src_fast_re': 'dst_host_srv_serror_rate',
    'dst_fast_re': 'dst_host_rerror_rate',
    'ovrlp_count': 'dst_host_srv_rerror_rate',
    'Class': 'labels',
    'Binary': 'label_binary'
}

# 重命名欄位
df = df.rename(columns=column_mapping)

# 將缺失值補為 0
df = df.fillna(0)

# 儲存轉換後的資料
df.to_csv("ukmtonsl.csv", index=False)

print("✅ 欄位轉換與缺失值補 0 完成，輸出檔案為 converted_output_UKM-IDS20 Testing set.csv")
