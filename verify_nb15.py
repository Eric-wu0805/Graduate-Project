import pandas as pd
import numpy as np

def verify_nb15_file():
    """驗證生成的NB15文件"""
    try:
        # 讀取文件
        df = pd.read_csv('cic2017_merged_to_nb15.csv')
        
        print("=" * 50)
        print("CIC2017 轉換為 NB15 格式驗證報告")
        print("=" * 50)
        
        # 基本信息
        print(f"文件大小: {len(df):,} 筆記錄")
        print(f"欄位數量: {len(df.columns)}")
        print(f"文件大小: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        
        # 標籤分布
        print(f"\n標籤分布:")
        label_counts = df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  Label {label}: {count:,} ({percentage:.2f}%)")
        
        # 攻擊類型分布
        print(f"\n攻擊類型分布:")
        attack_counts = df['attack_cat'].value_counts()
        for attack, count in attack_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {attack}: {count:,} ({percentage:.2f}%)")
        
        # 缺失值檢查
        print(f"\n缺失值檢查:")
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls == 0:
            print("  ✓ 沒有缺失值")
        else:
            print(f"  ✗ 發現 {total_nulls} 個缺失值")
            for col, count in null_counts[null_counts > 0].items():
                print(f"    {col}: {count}")
        
        # 數據類型檢查
        print(f"\n數據類型檢查:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        print(f"  數值欄位: {len(numeric_cols)} 個")
        print(f"  分類欄位: {len(categorical_cols)} 個")
        
        # 檢查標籤是否正確 (正常為0，其他為1)
        print(f"\n標籤正確性檢查:")
        normal_label_0 = df[(df['attack_cat'] == 'Normal') & (df['label'] == 0)].shape[0]
        attack_label_1 = df[(df['attack_cat'] != 'Normal') & (df['label'] == 1)].shape[0]
        total_correct = normal_label_0 + attack_label_1
        
        print(f"  正常流量標籤為0: {normal_label_0:,}")
        print(f"  攻擊流量標籤為1: {attack_label_1:,}")
        print(f"  正確標籤總數: {total_correct:,}")
        print(f"  標籤正確率: {total_correct/len(df)*100:.2f}%")
        
        # 顯示前幾行數據
        print(f"\n前5行數據預覽:")
        print(df.head())
        
        # 顯示欄位列表
        print(f"\nNB15欄位列表:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\n✓ 驗證完成！")
        
    except FileNotFoundError:
        print("錯誤: 找不到 cic2017_merged_to_nb15.csv 文件")
    except Exception as e:
        print(f"驗證時發生錯誤: {e}")

if __name__ == "__main__":
    verify_nb15_file() 