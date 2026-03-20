import pandas as pd
import numpy as np
import os

def merge_ukm_kdd_files():
    """
    合併轉換後的 UKM-IDS20 KDD 格式訓練集和測試集
    """
    print("=== 合併 UKM-IDS20 KDD 格式文件 ===\n")
    
    # 檢查文件是否存在
    train_file = 'ukm_to_kdd_train.csv'
    test_file = 'ukm_to_kdd_test.csv'
    output_file = 'ukm_to_kdd_merged.csv'
    
    files_to_check = [train_file, test_file]
    existing_files = []
    
    for file in files_to_check:
        if os.path.exists(file):
            existing_files.append(file)
            size = os.path.getsize(file) / 1024  # KB
            print(f"✓ 找到文件: {file} ({size:.1f} KB)")
        else:
            print(f"✗ 找不到文件: {file}")
    
    if len(existing_files) == 0:
        print("\n錯誤：沒有找到任何可合併的文件！")
        print("請先運行轉換程序生成 KDD 格式文件。")
        return False
    
    # 讀取並合併文件
    print(f"\n開始合併 {len(existing_files)} 個文件...")
    
    merged_data = []
    
    for i, file in enumerate(existing_files, 1):
        try:
            print(f"正在讀取文件 {i}/{len(existing_files)}: {file}")
            df = pd.read_csv(file)
            print(f"  - 讀取成功: {len(df)} 行, {len(df.columns)} 列")
            
            # 添加數據源標識
            df['data_source'] = file.replace('.csv', '').replace('ukm_to_kdd_', '')
            merged_data.append(df)
            
        except Exception as e:
            print(f"  - 讀取失敗: {str(e)}")
            return False
    
    # 合併所有數據
    print("\n正在合併數據...")
    try:
        merged_df = pd.concat(merged_data, ignore_index=True)
        print(f"合併完成: {len(merged_df)} 行, {len(merged_df.columns)} 列")
        
        # 重新排列列，將 data_source 放在最後
        cols = [col for col in merged_df.columns if col != 'data_source'] + ['data_source']
        merged_df = merged_df[cols]
        
    except Exception as e:
        print(f"合併失敗: {str(e)}")
        return False
    
    # 顯示合併後的統計信息
    print("\n=== 合併後數據統計 ===")
    print(f"總行數: {len(merged_df)}")
    print(f"總列數: {len(merged_df.columns)}")
    
    # 顯示各數據源的統計
    print("\n各數據源統計:")
    source_stats = merged_df['data_source'].value_counts()
    for source, count in source_stats.items():
        print(f"  {source}: {count} 行")
    
    # 顯示標籤分布
    print("\n標籤分布:")
    label_stats = merged_df['labels'].value_counts()
    for label, count in label_stats.items():
        percentage = (count / len(merged_df)) * 100
        print(f"  {label}: {count} 行 ({percentage:.1f}%)")
    
    # 檢查數據質量
    print("\n數據質量檢查:")
    missing_values = merged_df.isnull().sum()
    if missing_values.sum() > 0:
        print("發現缺失值:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"  {col}: {missing} 個缺失值")
    else:
        print("✓ 沒有缺失值")
    
    # 檢查數值範圍
    print("\n數值特徵統計:")
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'data_source']
    if len(numeric_cols) > 0:
        print(merged_df[numeric_cols].describe())
    
    # 保存合併後的文件
    print(f"\n正在保存合併後的文件到 {output_file}...")
    try:
        merged_df.to_csv(output_file, index=False)
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"✓ 保存成功: {output_file} ({file_size:.1f} KB)")
        
        # 顯示文件信息
        print(f"\n=== 生成文件信息 ===")
        print(f"文件名: {output_file}")
        print(f"文件大小: {file_size:.1f} KB")
        print(f"總行數: {len(merged_df)}")
        print(f"總列數: {len(merged_df.columns)}")
        print(f"列名: {list(merged_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"保存失敗: {str(e)}")
        return False

def main():
    """
    主函數
    """
    success = merge_ukm_kdd_files()
    
    if success:
        print("\n=== 合併程序完成 ===")
        print("✓ 所有文件已成功合併")
        print("✓ 可以使用合併後的文件進行機器學習訓練")
    else:
        print("\n=== 合併程序失敗 ===")
        print("✗ 請檢查錯誤信息並重試")

if __name__ == "__main__":
    main() 