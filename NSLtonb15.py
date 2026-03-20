import pandas as pd, numpy as np

df = pd.read_csv('NSL-KDD.csv')
nb = pd.DataFrame()
nb['id'] = np.arange(1, len(df)+1)
nb['dur'] = df['duration']
nb['proto'] = df['protocol_type']
nb['service'] = df['service']
nb['state'] = df['flag']

nb['spkts'] = 0; nb['dpkts'] = 0  # 无装包计数
nb['sbytes'] = df['src_bytes']
nb['dbytes'] = df['dst_bytes']
nb['rate'] = (df['src_bytes'] + df['dst_bytes']) / (df['duration'] + 1)
nb['sttl'] = nb['dttl'] = 0  # NSL 无 TTL 数据

# 损失/错误
nb['sloss'] = df['wrong_fragment']
nb['dloss'] = df['urgent']
nb['sinpkt'] = df['src_bytes'] / (nb['spkts'] + 1)
nb['dinpkt'] = df['dst_bytes'] / (nb['dpkts'] + 1)

# 设置默认 0
for c in ['sjit','djit','swin','stcpb','dwin','dtcpb','tcprtt','synack','ackdat',
          'smean','dmean','trans_depth','response_body_len',
          'ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm',
          'ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd',
          'ct_src_ltm','ct_srv_dst','is_sm_ips_ports']:
    nb[c] = 0

# 标签映射
nb['attack_cat'] = df['attack_type']
nb['label'] = df['label'].apply(lambda x: 0 if x=='normal' else 1)

nb.to_csv('NSL_to_UNSWNB15.csv', index=False)
