# UKM-IDS20 到 KDD 格式轉換程序

## 概述

這個程序將 UKM-IDS20 入侵檢測數據集轉換為標準的 KDD Cup 1999 格式，以便與現有的機器學習模型和工具兼容。

## 文件說明

### 轉換程序
- `ukm_to_kdd.py` - 基本轉換程序（僅轉換測試集）
- `ukm_to_kdd_complete.py` - 完整轉換程序（轉換訓練集和測試集）

### 輸入文件
- `UKM-IDS20 Testing set.csv` - UKM-IDS20 測試集
- `UKM-IDS20 Training set.csv` - UKM-IDS20 訓練集

### 輸出文件
- `ukm_to_kdd_test.csv` - 轉換後的 KDD 格式測試集
- `ukm_to_kdd_train.csv` - 轉換後的 KDD 格式訓練集

## 使用方法

### 方法一：使用完整轉換程序（推薦）

```bash
python ukm_to_kdd_complete.py
```

這個程序會自動：
1. 檢查並轉換測試集
2. 檢查並轉換訓練集
3. 顯示轉換統計信息
4. 報告生成的文件大小

### 方法二：使用基本轉換程序

```bash
python ukm_to_kdd.py
```

這個程序只轉換測試集。

## 特徵映射

### 基本特徵對應
| UKM-IDS20 欄位 | KDD 欄位 | 說明 |
|----------------|----------|------|
| dur | duration | 連接持續時間 |
| trnspt | protocol_type | 協議類型 (1→icmp, 6→tcp, 17→udp, 256→other) |
| srvs | service | 服務類型 |
| flag_n | flag | 連接狀態 (0→SF, 1→S0, 2→S1, 3→S2, 4→S3, 5→OTH) |
| src_byts | src_bytes | 源端位元組數 |
| dst_byts | dst_bytes | 目標端位元組數 |

### 統計特徵對應
| UKM-IDS20 欄位 | KDD 欄位 | 說明 |
|----------------|----------|------|
| src_pkts | count | 源端封包數 |
| dst_pkts | srv_count | 目標端封包數 |
| flag_sign | serror_rate | SYN 錯誤率 |
| flag_synrst | srv_serror_rate | SYN RST 錯誤率 |
| flag_arst | rerror_rate | RST 錯誤率 |
| flag_uc | srv_rerror_rate | 緊急錯誤率 |
| flag_a | same_srv_rate | 相同服務率 |
| flag_othr | diff_srv_rate | 不同服務率 |
| no_lnkd | srv_diff_host_rate | 服務不同主機率 |

### 主機特徵對應
| UKM-IDS20 欄位 | KDD 欄位 | 說明 |
|----------------|----------|------|
| dst_host_count | dst_host_count | 目標主機連接數 |
| host_dst _count | dst_host_srv_count | 目標主機服務連接數 |
| rtt_first_ack | dst_host_same_srv_rate | 目標主機相同服務率 |
| rtt_avg | dst_host_diff_srv_rate | 目標主機不同服務率 |
| avg_t_sent | dst_host_same_src_port_rate | 目標主機相同源端口率 |
| avg_t_got | dst_host_srv_diff_host_rate | 目標主機服務不同主機率 |
| repeated | dst_host_serror_rate | 目標主機錯誤率 |
| fst_src_sqc | dst_host_srv_serror_rate | 目標主機服務錯誤率 |
| fst_dst_sqc | dst_host_rerror_rate | 目標主機重置錯誤率 |
| src_re | dst_host_srv_rerror_rate | 目標主機服務重置錯誤率 |

### 標籤映射
| UKM-IDS20 標籤 | KDD 標籤 | 說明 |
|----------------|----------|------|
| Normal | normal | 正常流量 |
| UDP data flood | neptune | UDP 洪水攻擊 |
| ARP poisining | smurf | ARP 毒化攻擊 |
| TCP flood | back | TCP 洪水攻擊 |
| BeEF HTTP exploits | warezclient | BeEF HTTP 漏洞利用 |
| Mass HTTP requests | teardrop | 大量 HTTP 請求 |
| Metasploit exploits | satan | Metasploit 漏洞利用 |
| Port scanning | nmap | 端口掃描 |

## 數據統計

### 測試集統計
- 總行數：2,579
- 特徵數量：42
- 文件大小：376.6 KB

### 訓練集統計
- 總行數：10,308
- 特徵數量：42
- 文件大小：1,503.5 KB

## KDD 格式欄位

轉換後的數據包含以下 42 個欄位：

1. duration - 連接持續時間
2. protocol_type - 協議類型
3. service - 服務類型
4. flag - 連接狀態
5. src_bytes - 源端位元組數
6. dst_bytes - 目標端位元組數
7. land - 是否為本地連接
8. wrong_fragment - 錯誤片段數
9. urgent - 緊急位數
10. hot - 熱門服務數
11. num_failed_logins - 登入失敗次數
12. logged_in - 是否已登入
13. num_compromised - 受損主機數
14. root_shell - 是否獲得 root shell
15. su_attempted - su 嘗試次數
16. num_root - root 訪問次數
17. num_file_creations - 文件創建次數
18. num_shells - shell 啟動次數
19. num_access_files - 文件訪問次數
20. num_outbound_cmds - 出站命令數
21. is_host_login - 是否為主機登入
22. is_guest_login - 是否為訪客登入
23. count - 連接數
24. srv_count - 服務連接數
25. serror_rate - SYN 錯誤率
26. srv_serror_rate - 服務 SYN 錯誤率
27. rerror_rate - REJ 錯誤率
28. srv_rerror_rate - 服務 REJ 錯誤率
29. same_srv_rate - 相同服務率
30. diff_srv_rate - 不同服務率
31. srv_diff_host_rate - 服務不同主機率
32. dst_host_count - 目標主機連接數
33. dst_host_srv_count - 目標主機服務連接數
34. dst_host_same_srv_rate - 目標主機相同服務率
35. dst_host_diff_srv_rate - 目標主機不同服務率
36. dst_host_same_src_port_rate - 目標主機相同源端口率
37. dst_host_srv_diff_host_rate - 目標主機服務不同主機率
38. dst_host_serror_rate - 目標主機錯誤率
39. dst_host_srv_serror_rate - 目標主機服務錯誤率
40. dst_host_rerror_rate - 目標主機重置錯誤率
41. dst_host_srv_rerror_rate - 目標主機服務重置錯誤率
42. labels - 標籤

## 注意事項

1. **缺失值處理**：程序會自動將缺失值填充為 0
2. **數據類型**：所有數值特徵都轉換為數值類型
3. **標籤映射**：未知的攻擊類型會被映射為 'normal'
4. **文件編碼**：輸出文件使用 UTF-8 編碼

## 錯誤處理

程序包含完整的錯誤處理機制：
- 文件不存在檢查
- 數據讀取錯誤處理
- 保存文件錯誤處理
- 缺失值檢測和處理

## 依賴庫

- pandas
- numpy
- os (Python 內建模組)

## 版本信息

- 程序版本：1.0
- 支持格式：KDD Cup 1999
- 兼容性：Python 3.6+ 