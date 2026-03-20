import pandas as pd
import numpy as np
import powerlaw

# 載入資料集
training = pd.read_csv('kdd_test.csv')
testing = pd.read_csv('kdd_train.csv')

# 合併資料集
df = pd.concat([training, testing], ignore_index=True)

# 定義要分析的特徵
features = ['src_bytes', 'dst_bytes', 'duration']

print("開始進行冪律分佈統計檢定...")
print("-" * 30)

for feature in features:
    # 篩選掉零值，因為冪律分佈通常適用於正數
    data = df[df[feature] > 0][feature].copy()
    
    # 確保資料點足夠多才進行分析
    if len(data) < 50:
        print(f"'{feature}': 資料點不足 ({len(data)} 個)，跳過分析。")
        print("-" * 30)
        continue
    
    print(f"分析特徵: '{feature}'")
    
    # 擬合資料到冪律分佈
    # `fit.power_law.alpha` 是估計的 alpha
    # `fit.power_law.xmin` 是估計的 xmin
    fit = powerlaw.Fit(data, discrete=True)
    
    print(f"  - 擬合結果: alpha = {fit.power_law.alpha:.4f}, xmin = {fit.power_law.xmin:.2f}")
    
    # 進行對數似然比檢定，將冪律模型與其他模型進行比較
    # 這裡我們比較冪律與指數分佈 (exponential)
    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f"  - 冪律 vs. 指數: 對數似然比 R = {R:.4f}, p-value = {p:.4f}")
    
    # 冪律與對數常態分佈 (lognormal)
    R, p = fit.distribution_compare('power_law', 'lognormal')
    print(f"  - 冪律 vs. 對數常態: 對數似然比 R = {R:.4f}, p-value = {p:.4f}")
    
    print("-" * 30)