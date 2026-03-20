import pyshark
import pandas as pd
from collections import defaultdict
from datetime import datetime

pcap_file = 'botnet-capture-20110819-bot.pcap'
cap = pyshark.FileCapture(pcap_file, use_json=True, include_raw=False)

flows = defaultdict(list)

for packet in cap:
    try:
        if 'IP' not in packet:
            continue

        src_ip = packet.ip.src
        dst_ip = packet.ip.dst
        proto = packet.transport_layer or packet.highest_layer
        time = packet.sniff_time
        length = int(packet.length)

        src_port = getattr(packet[proto.lower()], 'srcport', '0')
        dst_port = getattr(packet[proto.lower()], 'dstport', '0')

        key = tuple(sorted([
            (src_ip, src_port, dst_ip, dst_port, proto),
            (dst_ip, dst_port, src_ip, src_port, proto)
        ]))[0]

        flows[key].append({
            'time': time,
            'length': length,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
        })
    except Exception:
        continue

cap.close()

flow_features = []

for flow_id, (key, packets) in enumerate(flows.items()):
    src_ip, src_port, dst_ip, dst_port, proto = key
    times = [p['time'] for p in packets]
    lengths = [p['length'] for p in packets]

    start_time = min(times)
    end_time = max(times)
    duration = (end_time - start_time).total_seconds()
    total_packets = len(packets)
    total_bytes = sum(lengths)
    avg_pkt_size = total_bytes / total_packets if total_packets > 0 else 0
    rate = total_bytes / duration if duration > 0 else 0

    # 模擬欄位（未完整）
    flow_features.append({
        'id': flow_id,
        'dur': duration,
        'proto': proto,
        'service': '-',  # 暫缺
        'state': '-',    # 暫缺
        'spkts': total_packets,
        'dpkts': total_packets,  # 暫無方向分離，先填同樣的
        'sbytes': total_bytes,
        'dbytes': total_bytes,
        'rate': rate,
        'sttl': 0,
        'dttl': 0,
        'sload': 0,
        'dload': 0,
        'sloss': 0,
        'dloss': 0,
        'sinpkt': 0,
        'dinpkt': 0,
        'sjit': 0,
        'djit': 0,
        'swin': 0,
        'stcpb': 0,
        'dtcpb': 0,
        'dwin': 0,
        'tcprtt': 0,
        'synack': 0,
        'ackdat': 0,
        'smean': avg_pkt_size,
        'dmean': avg_pkt_size,
        'trans_depth': 0,
        'response_body_len': 0,
        'ct_srv_src': 0,
        'ct_state_ttl': 0,
        'ct_dst_ltm': 0,
        'ct_src_dport_ltm': 0,
        'ct_dst_sport_ltm': 0,
        'ct_dst_src_ltm': 0,
        'is_ftp_login': 0,
        'ct_ftp_cmd': 0,
        'ct_flw_http_mthd': 0,
        'ct_src_ltm': 0,
        'ct_srv_dst': 0,
        'is_sm_ips_ports': 0,
        'attack_cat': 'Normal',  # 預設為 Normal
        'label': 0               # 預設為 Benign
    })

df = pd.DataFrame(flow_features)
df.to_csv('unsw_style_flows.csv', index=False)
print("已輸出 UNSW-NB15 欄位風格的 flow 特徵檔：unsw_style_flows.csv")
