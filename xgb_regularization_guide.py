"""
XGBoost 正则化参数调优指南

本文件展示了如何在XGBoost中正确使用L1和L2正则化，以及如何调优这些参数。
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import optuna

def create_optimized_xgb_model(X_train, y_train, X_val, y_val, method='optuna'):
    """
    创建带有优化正则化参数的XGBoost模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        method: 优化方法 ('optuna', 'grid_search', 'random_search')
    
    Returns:
        优化后的XGBoost模型
    """
    
    if method == 'optuna':
        return optimize_with_optuna(X_train, y_train, X_val, y_val)
    elif method == 'grid_search':
        return optimize_with_grid_search(X_train, y_train, X_val, y_val)
    elif method == 'random_search':
        return optimize_with_random_search(X_train, y_train, X_val, y_val)
    else:
        return create_default_xgb_model(X_train, y_train)

def optimize_with_optuna(X_train, y_train, X_val, y_val, n_trials=50):
    """
    使用Optuna优化XGBoost参数
    """
    def objective(trial):
        params = {
            # 基础参数
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            
            # L1 正则化 (Lasso) - 控制特征稀疏性
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            
            # L2 正则化 (Ridge) - 控制模型复杂度
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            
            # 其他正则化参数
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            
            # 类别不平衡处理
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            
            # 其他参数
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"最佳参数: {study.best_params}")
    print(f"最佳F1分数: {study.best_value:.4f}")
    
    # 使用最佳参数训练最终模型
    best_params = study.best_params
    best_params.update({
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    })
    
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    return final_model

def optimize_with_grid_search(X_train, y_train, X_val, y_val):
    """
    使用网格搜索优化XGBoost参数
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.1, 0.2],
        'reg_alpha': [0.0, 0.1, 0.5],  # L1正则化
        'reg_lambda': [0.5, 1.0, 2.0],  # L2正则化
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3]
    }
    
    base_params = {
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = XGBClassifier(**base_params)
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳F1分数: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def optimize_with_random_search(X_train, y_train, X_val, y_val, n_iter=20):
    """
    使用随机搜索优化XGBoost参数
    """
    param_distributions = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'reg_alpha': [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0],  # L1正则化
        'reg_lambda': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],  # L2正则化
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5]
    }
    
    base_params = {
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = XGBClassifier(**base_params)
    random_search = RandomizedSearchCV(
        model, param_distributions, n_iter=n_iter, cv=3, 
        scoring='f1', n_jobs=-1, verbose=1, random_state=42
    )
    random_search.fit(X_train, y_train)
    
    print(f"最佳参数: {random_search.best_params_}")
    print(f"最佳F1分数: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def create_default_xgb_model(X_train, y_train):
    """
    创建默认的XGBoost模型（带有合理的正则化参数）
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        # L1 正则化 (Lasso) - 促进特征稀疏性
        reg_alpha=0.1,
        # L2 正则化 (Ridge) - 防止过拟合
        reg_lambda=1.0,
        # 其他正则化参数
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        # 类别不平衡处理
        scale_pos_weight=scale_pos_weight,
        # 其他参数
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model_performance(model, X_test, y_test):
    """
    评估模型性能
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("模型性能评估:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def analyze_feature_importance(model, feature_names=None):
    """
    分析特征重要性
    """
    importance = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance))]
    
    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("特征重要性排名 (前10):")
    print(feature_importance_df.head(10))
    
    return feature_importance_df

def regularization_parameter_guide():
    """
    正则化参数使用指南
    """
    guide = """
    XGBoost 正则化参数使用指南:
    
    1. L1 正则化 (reg_alpha):
       - 作用: 促进特征稀疏性，自动进行特征选择
       - 适用场景: 特征数量很多，需要特征选择
       - 推荐值: 0.0 - 2.0
       - 效果: 值越大，特征越稀疏
    
    2. L2 正则化 (reg_lambda):
       - 作用: 防止过拟合，控制模型复杂度
       - 适用场景: 模型过拟合，需要泛化能力
       - 推荐值: 0.5 - 3.0
       - 效果: 值越大，模型越简单
    
    3. 其他正则化参数:
       - max_depth: 控制树的深度 (3-10)
       - min_child_weight: 控制叶子节点最小权重 (1-10)
       - subsample: 样本采样比例 (0.6-1.0)
       - colsample_bytree: 特征采样比例 (0.6-1.0)
       - colsample_bylevel: 每层特征采样比例 (0.6-1.0)
    
    4. 参数调优建议:
       - 先调 reg_lambda (L2)，再调 reg_alpha (L1)
       - 使用交叉验证避免过拟合
       - 监控训练集和验证集性能差异
       - 考虑使用早停机制 (early_stopping_rounds)
    
    5. 常见问题:
       - 过拟合: 增加 reg_lambda, 减少 max_depth
       - 欠拟合: 减少 reg_lambda, 增加 n_estimators
       - 特征过多: 增加 reg_alpha 进行特征选择
    """
    
    print(guide)
    return guide

if __name__ == "__main__":
    # 示例用法
    print("XGBoost 正则化参数调优指南")
    print("=" * 50)
    
    # 显示正则化参数指南
    regularization_parameter_guide()
    
    print("\n使用示例:")
    print("1. 创建默认模型: create_default_xgb_model(X_train, y_train)")
    print("2. 使用Optuna优化: create_optimized_xgb_model(X_train, y_train, X_val, y_val, 'optuna')")
    print("3. 使用网格搜索: create_optimized_xgb_model(X_train, y_train, X_val, y_val, 'grid_search')")
    print("4. 使用随机搜索: create_optimized_xgb_model(X_train, y_train, X_val, y_val, 'random_search')") 