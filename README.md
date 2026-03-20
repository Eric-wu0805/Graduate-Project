# ACRL - Adaptive Class-weighted Resampling Learning for Network Intrusion Detection

ACRL is an advanced Network Intrusion Detection System (NIDS) project that utilizes ensemble deep learning, dynamic class weighting, and multi-dataset integration to achieve high-performance threat detection across various network environments.

## 🚀 Key Features

- **Ensemble Deep Learning**: Combines ANN, CNN, RNN, and LSTM models with a meta-classifier (XGBoost/Logistic Regression) for robust predictions.
- **Dynamic Class Weighting**: Implements an adaptive weighting mechanism to handle class imbalance in real-time during training.
- **Multi-Dataset Support**: Integrated preprocessing and evaluation for major datasets:
  - UNSW-NB15
  - NSL-KDD
  - CIC-IDS2017
  - Bot-IoT
  - MQTT-IoT
  - CTU-13
- **Advanced Preprocessing**: Includes automated feature scaling (Robust/Standard), outlier handling (IQR), and dimensionality reduction.

## 📁 Repository Structure

The repository is organized into several functional components:

### 1. Model Architectures
- `ann.py`, `cnn.py`, `rnn.py`, `lstm.py`: Base implementations for deep learning models.
- `GNN.py`: Implementation of Graph Neural Networks for flow-based detection.
- `aclr.py`: The core Adaptive Class-weighted Resampling Learning framework.

### 2. Training & Optimization
- `aclr_KDD.py`, `aclr_IDS.py`, `aclr_Robust_new.py`: Specialized training scripts for different datasets and normalization techniques.
- `tuning_NB15.py`, `aclr_optuna_kdd.py`: Hyperparameter optimization scripts using Optuna.

### 3. Data Preprocessing
- `Preprocess.py`: Core data cleaning and transformation pipeline.
- `convert_*.py`: Scripts to convert various dataset formats to a unified structure.
- `merge_*.py`: Data aggregation and merging utilities.

### 4. Evaluation & Validation
- `test.py`: Comprehensive evaluation script for ensemble and base models.
- `Valid.py`: Validation script for model consistency.
- `kdd_test_less_feature_new.py`: Optimized testing script for reduced feature sets.

### 5. Visualization
- `chart.py`, `picture.py`, `pie.py`: Utilities for generating performance plots and distribution charts.
- `structure_pic.py`: Visualization for model architectures.

## 🛠️ Getting Started

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage
1. **Preprocessing**: Use `Preprocess.py` or specific `convert_*.py` scripts to prepare your data.
2. **Training**: Run `aclr.py` to start the ensemble training process with dynamic weighting.
3. **Evaluation**: Use `test.py` or `kdd_test.py` to evaluate trained models on test sets.

## 📝 Note on Data Storage
To maintain repository efficiency, large data files (`.csv`, `.npy`, `.pth` > 100MB) are excluded via `.gitignore`. Please ensure datasets are stored locally in the root directory for scripts to function correctly.
