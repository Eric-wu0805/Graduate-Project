"""
测试XGBoost正则化参数效果的脚本

这个脚本展示了不同正则化参数设置对模型性能的影响。
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns

def create_test_data(n_samples=10000, n_features=50, n_informative=20, n_redundant=10):
    """
    创建测试数据集
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # 添加一些噪声特征
    noise_features = np.random.randn(n_samples, 20)
    X = np.hstack([X, noise_features])
    
    return X, y

def test_regularization_effects(X, y):
    """
    测试不同正则化参数对模型性能的影响
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 定义不同的正则化参数组合
    regularization_configs = [
        {
            'name': '无正则化',
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'max_depth': 10,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        },
        {
            'name': '仅L1正则化',
            'reg_alpha': 0.5,
            'reg_lambda': 0.0,
            'max_depth': 10,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        },
        {
            'name': '仅L2正则化',
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'max_depth': 10,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        },
        {
            'name': 'L1+L2正则化',
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        {
            'name': '强正则化',
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'max_depth': 4,
            'subsample': 0.7,
            'colsample_bytree': 0.7
        }
    ]
    
    results = []
    
    for config in regularization_configs:
        print(f"\n测试配置: {config['name']}")
        print(f"参数: reg_alpha={config['reg_alpha']}, reg_lambda={config['reg_lambda']}")
        
        # 创建模型
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            reg_alpha=config['reg_alpha'],
            reg_lambda=config['reg_lambda'],
            max_depth=config['max_depth'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 计算特征重要性
        feature_importance = model.feature_importances_
        non_zero_features = np.sum(feature_importance > 0.001)  # 非零特征数量
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"非零特征数量: {non_zero_features}/{len(feature_importance)}")
        
        results.append({
            'config_name': config['name'],
            'reg_alpha': config['reg_alpha'],
            'reg_lambda': config['reg_lambda'],
            'max_depth': config['max_depth'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'non_zero_features': non_zero_features,
            'model': model
        })
    
    return results

def plot_regularization_comparison(results):
    """
    绘制正则化效果比较图
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 提取数据
    config_names = [r['config_name'] for r in results]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # 绘制性能指标
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ax = axes[i//2, i%2]
        bars = ax.bar(config_names, values, color='skyblue', alpha=0.7)
        ax.set_title(f'{metric.upper()} Score')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance_comparison(results):
    """
    绘制特征重要性比较图
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        if i >= len(axes):
            break
            
        model = result['model']
        feature_importance = model.feature_importances_
        
        # 选择前20个最重要的特征
        top_indices = np.argsort(feature_importance)[-20:]
        top_importance = feature_importance[top_indices]
        
        ax = axes[i]
        bars = ax.barh(range(len(top_indices)), top_importance, color='lightcoral', alpha=0.7)
        ax.set_title(f"{result['config_name']}\n非零特征: {result['non_zero_features']}")
        ax.set_xlabel('特征重要性')
        ax.set_ylabel('特征索引')
        
        # 添加正则化参数信息
        ax.text(0.02, 0.98, f'L1: {result["reg_alpha"]}, L2: {result["reg_lambda"]}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 隐藏多余的子图
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_overfitting(results, X_train, y_train, X_test, y_test):
    """
    分析过拟合情况
    """
    print("\n" + "="*60)
    print("过拟合分析")
    print("="*60)
    
    for result in results:
        model = result['model']
        
        # 训练集性能
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        
        # 测试集性能
        test_accuracy = result['accuracy']
        test_f1 = result['f1']
        
        # 计算性能差异
        accuracy_diff = train_accuracy - test_accuracy
        f1_diff = train_f1 - test_f1
        
        print(f"\n{result['config_name']}:")
        print(f"  训练集 Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        print(f"  测试集 Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        print(f"  性能差异 Accuracy: {accuracy_diff:.4f}, F1: {f1_diff:.4f}")
        
        # 判断过拟合程度
        if accuracy_diff > 0.1 or f1_diff > 0.1:
            print(f"  ⚠️  可能存在过拟合")
        elif accuracy_diff < 0.02 and f1_diff < 0.02:
            print(f"  ✅ 泛化性能良好")
        else:
            print(f"  ⚠️  需要进一步调优")

def main():
    """
    主函数
    """
    print("XGBoost 正则化参数效果测试")
    print("="*50)
    
    # 创建测试数据
    print("创建测试数据集...")
    X, y = create_test_data(n_samples=10000, n_features=50, n_informative=20, n_redundant=10)
    print(f"数据集形状: {X.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 测试正则化效果
    print("\n开始测试正则化效果...")
    results = test_regularization_effects(X, y)
    
    # 绘制比较图
    print("\n绘制性能比较图...")
    plot_regularization_comparison(results)
    
    # 绘制特征重要性比较
    print("\n绘制特征重要性比较图...")
    plot_feature_importance_comparison(results)
    
    # 分析过拟合情况
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    analyze_overfitting(results, X_train, y_train, X_test, y_test)
    
    # 输出总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    best_result = max(results, key=lambda x: x['f1'])
    print(f"最佳F1分数: {best_result['config_name']} (F1: {best_result['f1']:.4f})")
    
    most_sparse = min(results, key=lambda x: x['non_zero_features'])
    print(f"最稀疏模型: {most_sparse['config_name']} (非零特征: {most_sparse['non_zero_features']})")
    
    print("\n建议:")
    print("1. 如果模型过拟合，增加L2正则化 (reg_lambda)")
    print("2. 如果特征过多，增加L1正则化 (reg_alpha)")
    print("3. 结合使用L1和L2正则化通常效果最佳")
    print("4. 使用交叉验证确定最佳参数组合")

if __name__ == "__main__":
    main() 