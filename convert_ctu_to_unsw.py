import pandas as pd
import numpy as np

def convert_ctu_to_unsw():
    # 读取CTU13数据集
    print("正在读取CTU13数据集...")
    normal_df = pd.read_csv('CTU13_Normal_Traffic.csv')
    attack_df = pd.read_csv('CTU13_Attack_Traffic.csv')
    
    # 合并数据集
    print("正在合并数据集...")
    combined_df = pd.concat([normal_df, attack_df], ignore_index=True)
    
    # 新建UNSW-NB15格式的DataFrame
    print("正在转换为UNSW-NB15格式...")
    unsw_columns = [
        'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
        'service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth',
        'res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports',
        'ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
        'ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label'
    ]
    unsw_df = pd.DataFrame(columns=unsw_columns)
    
    # 字段映射
    unsw_df['dur'] = combined_df['Flow Duration']
    unsw_df['sbytes'] = combined_df['TotLen Fwd Pkts']
    unsw_df['dbytes'] = combined_df['TotLen Bwd Pkts']
    unsw_df['Sload'] = combined_df['Flow Byts/s']
    unsw_df['Spkts'] = combined_df['Tot Fwd Pkts']
    unsw_df['Dpkts'] = combined_df['Tot Bwd Pkts']
    unsw_df['smeansz'] = combined_df['Fwd Pkt Len Mean']
    unsw_df['dmeansz'] = combined_df['Bwd Pkt Len Mean']
    unsw_df['Sintpkt'] = combined_df['Flow IAT Mean']
    unsw_df['Dintpkt'] = combined_df['Bwd IAT Mean']
    unsw_df['Label'] = combined_df['Label']
    
    # attack_cat字段
    unsw_df['attack_cat'] = combined_df['Label'].apply(lambda x: 'Normal' if x==0 else 'Attack')
    
    # 其余字段填空或0
    for col in unsw_columns:
        if col not in unsw_df.columns:
            if col == 'Label':
                continue
            if col in ['is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']:
                unsw_df[col] = 0
            else:
                unsw_df[col] = ''
    
    # 按UNSW-NB15顺序保存
    unsw_df = unsw_df[unsw_columns]
    print("正在保存转换后的数据集...")
    unsw_df.to_csv('CTU13_to_UNSW.csv', index=False)
    print("转换完成！")

if __name__ == "__main__":
    convert_ctu_to_unsw() 