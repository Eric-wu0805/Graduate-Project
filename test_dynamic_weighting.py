#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動態類別權重測試腳本
測試DynamicClassWeighting類的各種功能
"""

import numpy as np
import matplotlib.pyplot as plt
from aclr_KDD import DynamicClassWeighting, DynamicFocalLoss, FocalLoss
import torch
import torch.nn as nn

def test_dynamic_class_weighting():
    """測試動態類別權重計算器"""
    print("=== 測試動態類別權重計算器 ===")
    
    # 創建模擬數據
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # 創建不平衡數據集（80% 負類，20% 正類）
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    y_pred_proba = np.random.random(n_samples)
    
    # 調整預測概率使其更真實
    for i in range(n_samples):
        if y_true[i] == 1:
            # 正類樣本，預測概率應該偏向1
            y_pred_proba[i] = np.random.beta(3, 1)
        else:
            # 負類樣本，預測概率應該偏向0
            y_pred_proba[i] = np.random.beta(1, 3)
    
    print(f"數據分佈: 負類 {np.sum(y_true == 0)}, 正類 {np.sum(y_true == 1)}")
    print(f"平均預測概率: {np.mean(y_pred_proba):.4f}")
    
    # 初始化動態權重計算器
    weighting = DynamicClassWeighting(update_frequency=2, momentum=0.9)
    
    # 測試不同的權重計算方法
    methods = ['focal_adaptive', 'confidence_based', 'error_rate_based', 'balanced']
    
    for method in methods:
        print(f"\n--- 測試 {method} 方法 ---")
        weights = weighting.calculate_class_weights(y_true, y_pred_proba, method=method)
        print(f"計算得到的權重: {weights}")
        
        # 更新權重
        weighting.update_weights(weights)
        print(f"更新後的權重: {weighting.get_current_weights()}")
    
    # 測試權重歷史記錄
    print(f"\n權重更新次數: {len(weighting.get_weight_history())}")
    
    return weighting

def test_dynamic_focal_loss():
    """測試動態Focal Loss"""
    print("\n=== 測試動態Focal Loss ===")
    
    # 創建模擬數據
    batch_size = 32
    input_size = 10
    
    # 創建不平衡的標籤
    y_true = torch.tensor([0] * 25 + [1] * 7, dtype=torch.float32)
    y_pred = torch.rand(32, 1)
    
    # 初始化動態權重計算器
    weighting = DynamicClassWeighting()
    weighting.update_weights({0: 1.0, 1: 3.0})  # 給正類更高權重
    
    # 測試動態Focal Loss
    dynamic_criterion = DynamicFocalLoss(weighting)
    dynamic_loss = dynamic_criterion(y_pred, y_true)
    
    # 測試原始Focal Loss
    original_criterion = FocalLoss()
    original_loss = original_criterion(y_pred, y_true)
    
    print(f"原始Focal Loss: {original_loss.item():.4f}")
    print(f"動態Focal Loss: {dynamic_loss.item():.4f}")
    print(f"權重影響: {dynamic_loss.item() / original_loss.item():.4f}")

def visualize_weight_evolution():
    """可視化權重演變過程"""
    print("\n=== 可視化權重演變 ===")
    
    # 模擬多個epoch的權重變化
    weighting = DynamicClassWeighting(update_frequency=1)
    
    # 模擬訓練過程中的權重更新
    for epoch in range(10):
        # 模擬不同epoch的預測結果
        y_true = np.random.choice([0, 1], size=100, p=[0.8, 0.2])
        y_pred_proba = np.random.random(100)
        
        # 根據epoch調整預測質量（模擬模型學習過程）
        if epoch < 5:
            # 早期epoch，預測較差
            for i in range(100):
                if y_true[i] == 1:
                    y_pred_proba[i] = np.random.beta(1, 2)  # 偏向0
                else:
                    y_pred_proba[i] = np.random.beta(2, 1)  # 偏向1
        else:
            # 後期epoch，預測較好
            for i in range(100):
                if y_true[i] == 1:
                    y_pred_proba[i] = np.random.beta(3, 1)  # 偏向1
                else:
                    y_pred_proba[i] = np.random.beta(1, 3)  # 偏向0
        
        # 計算並更新權重
        weights = weighting.calculate_class_weights(y_true, y_pred_proba, method='focal_adaptive')
        weighting.update_weights(weights)
        weighting.epoch_count += 1
        
        print(f"Epoch {epoch+1}: 權重 = {weighting.get_current_weights()}")
    
    # 繪製權重變化圖
    weight_history = weighting.get_weight_history()
    epochs = range(len(weight_history))
    
    plt.figure(figsize=(12, 8))
    
    # 繪製權重變化
    plt.subplot(2, 2, 1)
    for class_id in [0, 1]:
        weights = [w.get(class_id, 0) for w in weight_history]
        plt.plot(epochs, weights, marker='o', label=f'Class {class_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Class Weight')
    plt.title('Dynamic Class Weights Evolution')
    plt.legend()
    plt.grid(True)
    
    # 繪製權重比例
    plt.subplot(2, 2, 2)
    for class_id in [0, 1]:
        weights = [w.get(class_id, 0) for w in weight_history]
        plt.plot(epochs, weights, marker='s', label=f'Class {class_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.title('Weight Values Over Time')
    plt.legend()
    plt.grid(True)
    
    # 繪製權重差異
    plt.subplot(2, 2, 3)
    weight_diff = []
    for w in weight_history:
        diff = w.get(1, 0) - w.get(0, 0)
        weight_diff.append(diff)
    plt.plot(epochs, weight_diff, marker='^', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Weight Difference (Class 1 - Class 0)')
    plt.title('Weight Difference Evolution')
    plt.grid(True)
    
    # 繪製權重比率
    plt.subplot(2, 2, 4)
    weight_ratio = []
    for w in weight_history:
        ratio = w.get(1, 0) / (w.get(0, 0) + 1e-6)
        weight_ratio.append(ratio)
    plt.plot(epochs, weight_ratio, marker='d', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Weight Ratio (Class 1 / Class 0)')
    plt.title('Weight Ratio Evolution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dynamic_weighting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("權重演變分析圖已保存為 'dynamic_weighting_analysis.png'")

def test_weighting_methods_comparison():
    """比較不同權重計算方法的效果"""
    print("\n=== 比較不同權重計算方法 ===")
    
    # 創建模擬數據
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    y_pred_proba = np.random.random(n_samples)
    
    # 調整預測概率
    for i in range(n_samples):
        if y_true[i] == 1:
            y_pred_proba[i] = np.random.beta(2, 1)
        else:
            y_pred_proba[i] = np.random.beta(1, 2)
    
    methods = ['focal_adaptive', 'confidence_based', 'error_rate_based', 'balanced']
    results = {}
    
    for method in methods:
        weighting = DynamicClassWeighting()
        weights = weighting.calculate_class_weights(y_true, y_pred_proba, method=method)
        results[method] = weights
        
        print(f"{method}: {weights}")
    
    # 可視化比較
    plt.figure(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35
    
    class_0_weights = [results[method].get(0, 0) for method in methods]
    class_1_weights = [results[method].get(1, 0) for method in methods]
    
    plt.bar(x - width/2, class_0_weights, width, label='Class 0', alpha=0.8)
    plt.bar(x + width/2, class_1_weights, width, label='Class 1', alpha=0.8)
    
    plt.xlabel('Weighting Method')
    plt.ylabel('Class Weight')
    plt.title('Comparison of Different Weighting Methods')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weighting_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("權重方法比較圖已保存為 'weighting_methods_comparison.png'")

if __name__ == "__main__":
    print("開始測試動態類別權重功能...")
    
    # 運行所有測試
    test_dynamic_class_weighting()
    test_dynamic_focal_loss()
    visualize_weight_evolution()
    test_weighting_methods_comparison()
    
    print("\n所有測試完成！") 