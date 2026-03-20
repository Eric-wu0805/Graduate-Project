import pandas as pd
import os
import warnings

# === CONFIG ===
INPUT_CSV = "CTU13_Normal_Traffic.csv"  # 輸入的 CSV 檔案名稱
OUTPUT_CSV = "CTU13_Normal.csv"  # 輸出的 CSV 檔案名稱

# === Step 0: bottleneck 版本提示 ===
try:
    import bottleneck
    from packaging import version
    if version.parse(bottleneck.__version__) < version.parse("1.3.6"):
        warnings.warn("⚠️ 建議更新 bottleneck 至 1.3.6 以上版本，以避免 pandas 相容性問題。")
except ImportError:
    warnings.warn("⚠️ 未安裝 bottleneck，建議安裝以加速 pandas 運算。")

# === Step 1: 讀取 Flow CSV ===
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"❌ 找不到 {INPUT_CSV}，請確認路徑正確或先使用 cicflowmeter 產生 Flow CSV。")

df = pd.read_csv(INPUT_CSV)
print(f"[✔] 已讀取資料，共 {len(df)} 筆")

# === Step 2: 檢查資料是否有 IP 欄位 ===
print("[🧪] 檢測到的欄位有：")
for col in df.columns:
    print("-", col)

# === Step 3: 根據 Label 設置標註欄位 ===
# Label欄位是1就表示是Bot，否則是Normal
df['label'] = df['Label'].apply(lambda x: 1 if x == 1 else 0)  # 如果 Label 是 1，則表示 Bot
df['attack_cat'] = df['label'].apply(lambda x: 'Bot' if x == 1 else 'Normal')  # 根據 label 標註 attack_cat

# === Step 4: 欄位對齊 ===
column_map = {
    'Flow ID': 'id',
    'Flow Duration': 'dur',
    'Protocol': 'proto',
    'Service': 'service',
    'State': 'state',
    'Tot Fwd Pkts': 'spkts',
    'Tot Bwd Pkts': 'dpkts',
    'TotLen Fwd Pkts': 'sbytes',
    'TotLen Bwd Pkts': 'dbytes',
    'Flow Byts/s': 'rate',
    'Fwd Header Len': 'sttl',
    'Bwd Header Len': 'dttl',
    'Fwd Seg Size Avg': 'sload',
    'Bwd Seg Size Avg': 'dload',
    'Fwd Seg Size Min': 'sloss',
    'Bwd Seg Size Min': 'dloss',
    'Fwd IAT Min': 'sinpkt',
    'Bwd IAT Min': 'dinpkt',
    'Fwd IAT Std': 'sjit',
    'Bwd IAT Std': 'djit',
    'Init Fwd Win Byts': 'swin',
    'Fwd Act Data Pkts': 'stcpb',
    'Bwd Act Data Pkts': 'dtcpb',
    'Init Bwd Win Byts': 'dwin',
    'Flow IAT Mean': 'tcprtt',
    'Fwd IAT Mean': 'synack',
    'Bwd IAT Mean': 'ackdat',
    'Fwd Pkt Len Mean': 'smean',
    'Bwd Pkt Len Mean': 'dmean',
    'Fwd Pkt Len Max': 'trans_depth',
    'Fwd Pkt Len Min': 'response_body_len',
    'Flow Duration': 'ct_srv_src',
    'TotLen Fwd Pkts': 'ct_state_ttl',
    'Fwd Header Len': 'ct_dst_ltm',
    'Bwd Header Len': 'ct_src_dport_ltm',
    'Fwd Seg Size Avg': 'ct_dst_sport_ltm',
    'Bwd Seg Size Avg': 'ct_dst_src_ltm',
    'Fwd PSH Flags': 'is_ftp_login',
    'Bwd PSH Flags': 'ct_ftp_cmd',
    'Fwd URG Flags': 'ct_flw_http_mthd',
    'Bwd URG Flags': 'ct_src_ltm',
    'Active Mean': 'ct_srv_dst',
    'Idle Mean': 'is_sm_ips_ports',
}

# 補欄位
for original_col in column_map:
    if original_col not in df.columns:
        df[original_col] = -1

# 欄位重新命名與排序
df = df.rename(columns=column_map)
final_columns = list(column_map.values()) + ['attack_cat', 'label']
df = df[final_columns]

# === Step 5: 匯出 CSV ===
df.to_csv(OUTPUT_CSV, index=False)
print(f"[✅] 成功轉換並儲存至：{OUTPUT_CSV}")
