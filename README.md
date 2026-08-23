# ACRL — Adaptive Class-weighted Resampling Learning for Network Intrusion Detection

ACRL is a graduate research project that builds a **stacked deep-learning ensemble Network Intrusion Detection System (NIDS)**. It combines four base neural networks (ANN, CNN, RNN, LSTM) with a gradient-boosted meta-classifier, trains them with a dynamic, focal-loss-based class-weighting scheme to cope with severe class imbalance, and evaluates cross-dataset generalization across six public intrusion-detection datasets that have all been normalized into a common KDD-style schema.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Repository Layout](#repository-layout)
- [Included Artifacts (models & pickles)](#included-artifacts-models--pickles)
- [Getting Started](#getting-started)
- [Typical Workflow](#typical-workflow)
- [Highlighted Pipeline: aclr_Robust_new.py](#highlighted-pipeline-aclr_robust_newpy)
- [Script Reference](#script-reference)
- [Supplementary Documentation](#supplementary-documentation)
- [Notes](#notes)

## Overview

Network intrusion detection datasets are collected under different capture tools, feature extractors, and labeling conventions (NSL-KDD, UNSW-NB15, CIC-IDS2017, Bot-IoT, MQTT-IoT, CTU-13, UKM-IDS20, …), which makes it hard to train one model that generalizes across environments. This project addresses that in three stages:

1. **Dataset unification** — dedicated `convert_*.py` / `*_to_kdd.py` / `*_to_nb15.py` scripts translate each source dataset's native fields into a shared 41/42-feature KDD-Cup-99-style schema (or a UNSW-NB15-style schema), so that models trained on one dataset can be evaluated on another.
2. **Ensemble training with dynamic class weighting** — four base learners (ANN, 1D-CNN, RNN, LSTM) are trained in parallel on the same features, using a custom `DynamicFocalLoss` whose per-class weights are recalculated every few epochs based on the validation-set confusion pattern (`DynamicClassWeighting`). Their four output probabilities are stacked into a small meta-feature vector.
3. **Meta-classification** — the stacked predictions are fed into a meta-learner (typically `XGBClassifier` with tuned L1/L2 regularization, `scale_pos_weight` for imbalance, and SMOTE oversampling; some variants swap in `LogisticRegression` or `RandomForestClassifier`) to produce the final Normal/Attack decision.

The project also includes many experimental variants (per-domain calibration, reduced feature sets, alternative scalers, Optuna-based hyperparameter tuning, cross-dataset "zero-shot" evaluation) that were used to explore what makes the ensemble generalize best.

## Architecture

```
                 ┌──────────────┐
   raw traffic → │ Preprocessing │  label encoding, IQR outlier clipping,
   (per dataset) │   pipeline    │  skew log-transform, Robust/MinMax/Quantile
                 └──────┬───────┘  scaling, correlation-based feature
                        │           selection, pairwise feature interactions
             ┌──────────┼────────────┬────────────┐
             ▼          ▼            ▼            ▼
          ANNModel   CNNModel    RNNModel     LSTMModel      (base learners,
        (FC 64→32)  (Conv1d x2)  (RNN, 64)   (LSTM, 64)       DynamicFocalLoss)
             │          │            │            │
             └──────────┴─────┬──────┴────────────┘
                               ▼
                  stacked probabilities (meta-features)
                               │
                               ▼
                 Meta-classifier (XGBoost / LogisticRegression /
                 RandomForest, SMOTE-balanced, L1/L2 regularized)
                               │
                               ▼
                    Normal (0) vs. Attack (1)
```

Each base model is defined identically across `model.py`, `aclr*.py`, and the `*test*.py` evaluation scripts:

- **ANNModel** — `Linear(in→64) → ReLU → Dropout → Linear(64→32) → ReLU → Dropout → Linear(32→1) → Sigmoid`
- **CNNModel** — two `Conv1d` + `MaxPool1d` blocks over the feature vector treated as a 1‑D signal, followed by a fully-connected head
- **RNNModel** — a single-layer `nn.RNN(hidden=64)` over the feature vector treated as a length‑1 sequence
- **LSTMModel** — a single-layer `nn.LSTM(hidden=64)` in the same style

Training uses `FocalLoss` / `DynamicFocalLoss` (focal loss with an adaptively-updated per-class weight, computed via one of several strategies: `focal_adaptive`, `confidence_based`, `error_rate_based`, or plain inverse-frequency `balanced`), `AdamW`, and K-Fold cross-validation. A separate `GNN.py` script currently only loads the `Mireu-Lab/NSL-KDD` dataset via Hugging Face `datasets` and is a stub for future graph-based modeling.

## Datasets

The pipeline is designed to unify and cross-evaluate across:

| Dataset | Role in this repo |
|---|---|
| **NSL-KDD** | Primary training/testing schema (41 features + `labels`); `kdd_train.csv` / `kdd_test.csv` are checked into the repo |
| **UNSW-NB15** | Alternate primary schema (`proto`, `service`, `state`, `attack_cat`, …) used by `model.py`, `Preprocess.py`, `aclr.py` |
| **CIC-IDS2017** | Converted to KDD/NB15 schema via `convert_cic2017_to_kdd.py`, `cic2017toNB15.py`, `cic2017_to_nb15_complete.py`, `merge_cic2017_to_kdd.py` |
| **Bot-IoT** | Converted via `BoTIoT_to_kdd.py` |
| **MQTT-IoT** | Converted via `convert_mqttiot_to_kdd.py` |
| **CTU-13** | Converted via `convert_ctu_to_unsw.py`, `ctu53_to_unsw.py`, trained directly via `CTUaclr.py`; raw flows can be produced from `.pcap` captures with `pcaptocsv.py` |
| **UKM-IDS20** | Converted via `ukm_to_kdd.py` / `ukm_to_kdd_complete.py` / `ukmtonb15.py` / `ukmtonslp.py`, merged via `merge_ukm_kdd.py` (see [`UKM_to_KDD_README.md`](UKM_to_KDD_README.md) and [`合併報告.md`](合併報告.md) for the full field mapping and merge statistics) |

`nb15_kdd_train.csv` is a KDD-schema export built from UNSW-NB15 for cross-dataset evaluation; it is one of the few large CSVs kept in the repo (see [Notes](#notes)).

## Repository Layout

```
.
├── Model definitions
│   ├── model.py            TensorFlow/Keras version of the ANN+CNN+LSTM+RNN stacked model
│   ├── ann.py, cnn.py, rnn.py, lstm.py   Standalone PyTorch model/training scripts per architecture
│   └── GNN.py               Stub for a future Graph Neural Network approach (loads NSL-KDD only)
│
├── Core ACRL training pipelines (PyTorch, dynamic class weighting)
│   ├── aclr.py               Baseline pipeline on UNSW-NB15
│   ├── aclr_KDD.py           Baseline pipeline on the KDD schema (primary/most complete script)
│   ├── aclr_IDS.py           Generic IDS-schema training variant
│   ├── aclr_KDD_smote_nc.py  KDD variant using SMOTE-NC (mixed categorical/numeric oversampling)
│   ├── aclr_MinMax.py / aclr_Quantile.py / aclr_Robust.py
│   │                          Same pipeline with different feature-scaling strategies
│   ├── aclr_Robust_new.py     Most advanced single-file pipeline: 6-feature focused input,
│   │                          RobustScaler + StandardScaler, XGBoost-importance-guided
│   │                          "three-stage" random feature masking, early stopping, and
│   │                          explicit L1/L2 regularization (see dedicated section below)
│   ├── aclr_Per-Domain.py / kdd_Per-domain.py
│   │                          Adds per-domain probability calibration (`CalibratedClassifierCV`)
│   │                          for better cross-dataset transfer
│   ├── aclr_less_feature.py / aclr_less_feature_lr.py / aclr_less_feature_rf.py / aclr_less_new.py
│   │                          Reduced-feature-set variants (fewer input columns, lighter models)
│   ├── aclr_optuna_kdd.py    Optuna-driven hyperparameter search on top of the KDD pipeline
│   ├── aclr_uplevel.py       Extended/"upgraded" pipeline variant
│   └── CTUaclr.py            Pipeline specialized for the CTU-13 dataset
│
├── Dataset conversion & merging (→ unified KDD / NB15 schema)
│   ├── convert_cic2017_to_kdd.py, convert_cic2017_to_nb15.py, cic2017toNB15.py,
│   │   cic2017_to_nb15_complete.py, merge_cic2017_to_kdd.py, convert_to_cic2017.py
│   ├── convert_ctu_to_unsw.py, ctu53_to_unsw.py
│   ├── convert_mqttiot_to_kdd.py
│   ├── convert_nb15_to_kdd.py, NSLtonb15.py
│   ├── ukm_to_kdd.py, ukm_to_kdd_complete.py, ukmtonb15.py, ukmtonslp.py, merge_ukm_kdd.py
│   ├── BoTIoT_to_kdd.py
│   └── pcaptocsv.py           Extracts flow-level features from raw `.pcap` captures via `pyshark`
│
├── Preprocessing
│   ├── Preprocess.py          Minimal label-encoding + standardization example
│   └── KDD_minMax.py          KDD pipeline variant using `MinMaxScaler`
│
├── Evaluation & testing
│   ├── test.py                 Loads trained models/artifacts and evaluates on held-out + cross-dataset data
│   ├── IDS_testing.py          Cross-dataset evaluation against a merged NSL-KDD→CIC2017 file
│   ├── kdd_test.py / kdd_new_testing.py / kdd_test_less_feature.py / kdd_test_less_feature_new.py
│   │                            KDD-schema evaluation variants (incl. reduced-feature and
│   │                            Matthews-correlation-coefficient reporting)
│   ├── Valid.py                 Validation-set consistency checks
│   ├── tuning_NB15.py / tuning_testing.py
│   │                            Grid/random search + Optuna hyperparameter tuning utilities
│   ├── test_dynamic_weighting.py  Unit-style tests for `DynamicClassWeighting`/`DynamicFocalLoss`
│   ├── test_xgb_regularization.py / xgb_regularization_guide.py
│   │                            Demonstrates and benchmarks XGBoost L1/L2 regularization settings
│   │                            (see XGBoost_Regularization_README.md)
│   └── verify_nb15.py           Sanity-checks a generated NB15-format CSV
│
├── Analysis & visualization
│   ├── chart.py, picture.py, pie.py, number.py, log.py, power_law.py
│   │                            Distribution / attack-category / power-law plots (Times New Roman styled)
│   ├── structure_pic.py         Renders an architecture diagram via the `diagrams` package
│   └── analyze_attack_features.py (+ _fixed / _improved variants)
│                                  Exports per-attack-type feature usage to Excel via `openpyxl`
│
├── Data & trained artifacts (see below)
│   ├── kdd_train.csv, kdd_test.csv, nb15_kdd_train.csv
│   ├── *.pth (PyTorch model weights), *.pkl (scalers/encoders/meta-models), trained_model.h5
│
├── Untitled13.ipynb            Exploratory notebook mirroring the KDD ACRL pipeline
├── BlazorApp.slnx              .NET solution file referencing an external Blazor front-end
│                                (`../blazor/BlazorApp/BlazorApp.csproj`, not included in this repo)
├── requirements.txt
├── UKM_to_KDD_README.md        Field-mapping documentation for the UKM-IDS20 → KDD converter (Chinese)
├── XGBoost_Regularization_README.md  L1/L2 regularization tuning guide for the meta-model (Chinese)
└── 合併報告.md                  UKM-IDS20 merge report / dataset statistics (Chinese)
```

## Included Artifacts (models & pickles)

Because training these models is expensive, a number of pre-trained artifacts are checked in:

| Pattern | Description |
|---|---|
| `final_ann.pth`, `final_cnn.pth`, `final_rnn.pth`, `final_lstm.pth` | Base-learner weights from the most recent `aclr*.py` run |
| `final_meta_model.pkl`, `final_meta_model_rf.pkl` | Fitted meta-classifiers (XGBoost / Random-Forest variants) |
| `final_scaler.pkl`, `feature_scalers.pkl`, `robust_scaler.pkl`, `standard_scaler.pkl` | Fitted `sklearn` scalers used to reproduce preprocessing at inference time |
| `label_encoders.pkl`, `onehot_encoder.pkl` | Fitted categorical encoders |
| `selected_features.pkl`, `selected_meta_features.pkl` | Feature-selection results from training |
| `feature_weights.pkl` | Normalized XGBoost feature-importance scores produced by `aclr_Robust_new.py`, used to drive its importance-aware random-masking curriculum |
| `model_config.pkl`, `*_params.pkl` | Saved hyperparameters/input sizes needed to reconstruct a model before loading its `state_dict` |
| `KDD_final_*.pth/.pkl`, `NB15_final_*.pth/.pkl` | Dataset-specific final model/meta-model sets (KDD schema vs. NB15 schema) |
| `ANN_best.pth`, `CNN_best.pth`, `RNN_best.pth`, `LSTM_best.pth`, `tuned_*.pth`, `best_FineTunable*Model.pth`, `best_ann_model.pth` | Checkpoints from hyperparameter-tuning / fine-tuning runs |
| `trained_model.h5` | Keras/TensorFlow model from `model.py` |

These files let you run the evaluation scripts without retraining, as long as the expected input CSVs are present.

## Getting Started

### Prerequisites

- Python 3.9+ recommended
- A CUDA-capable GPU is optional but speeds up training considerably (`torch.cuda.is_available()` is checked automatically)

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the core stack (`torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `tqdm`, `joblib`, `imbalanced-learn`, `lightgbm`, `catboost`). Several scripts additionally import packages that are **not** listed there and must be installed separately depending on which script you run:

- `xgboost` — meta-classifier used throughout the `aclr*.py` family
- `optuna` — hyperparameter tuning (`aclr_optuna_kdd.py`, `tuning_NB15.py`, `tuning_testing.py`, `xgb_regularization_guide.py`)
- `seaborn` — confusion-matrix heatmaps in the `*test*.py` scripts
- `tensorflow` — `model.py`
- `openpyxl` — `analyze_attack_features*.py`
- `powerlaw` — `power_law.py`
- `diagrams` (plus Graphviz) — `structure_pic.py`
- `pyshark` (plus Wireshark/tshark) — `pcaptocsv.py`
- `datasets` (Hugging Face) — `GNN.py`

### Data

Only `kdd_train.csv`, `kdd_test.csv`, and `nb15_kdd_train.csv` are committed (everything else matching `*.csv` is excluded via `.gitignore`, along with `*.png`, `*.npy`, `*.pcap`, and `*.xlsx`). To reproduce the full multi-dataset pipeline you will need to download the original datasets yourself (NSL-KDD, UNSW-NB15, CIC-IDS2017, Bot-IoT, MQTT-IoT, CTU-13, UKM-IDS20) and place them in the repository root under the filenames each script expects (see the `pd.read_csv(...)` calls at the top of each script).

## Typical Workflow

1. **Convert/unify a new dataset** (optional, only needed for datasets other than the ones already provided) — run the matching `convert_*.py` / `*_to_kdd.py` / `*_to_nb15.py` script, then merge splits with the corresponding `merge_*.py` script if applicable.
2. **Preprocess & train the ensemble** — run one of the `aclr*.py` pipelines (start with `aclr_KDD.py` for the KDD schema or `aclr.py` for UNSW-NB15). Each run will:
   - clean and scale features (outlier clipping, log-transform, scaling, correlation-based selection, pairwise interaction terms)
   - train ANN/CNN/RNN/LSTM base learners with `DynamicFocalLoss` over K folds
   - stack their outputs and fit the XGBoost (or LR/RF) meta-classifier with SMOTE-balanced data
   - save all trained weights, the meta-model, and the fitted preprocessors (`.pth` / `.pkl` files) to the working directory
3. **Evaluate** — run `test.py`, `kdd_test.py`, `IDS_testing.py`, or one of the other `*test*.py` scripts to reload the saved artifacts, score them on a held-out split, and (if a cross-dataset CSV is present) test generalization to a different dataset. These scripts print accuracy/precision/recall/F1/AUC and save ROC-curve and confusion-matrix plots.
4. **Visualize / analyze** — use `chart.py`, `picture.py`, `pie.py`, `structure_pic.py`, `power_law.py`, or `analyze_attack_features*.py` to generate distribution plots, an architecture diagram, or per-attack-type feature reports.
5. **Tune** (optional) — use `aclr_optuna_kdd.py`, `tuning_NB15.py`, `tuning_testing.py`, or `xgb_regularization_guide.py` to search hyperparameters, and `test_xgb_regularization.py` to compare regularization settings.

## Highlighted Pipeline: `aclr_Robust_new.py`

`aclr_Robust_new.py` is the most refined single-file training pipeline in the repo. It reuses the ACRL ensemble design (ANN + CNN + RNN + LSTM base learners stacked into an XGBoost meta-classifier) but adds several regularization and curriculum-learning techniques on top of the baseline `aclr.py` / `aclr_KDD.py` scripts:

- **Focused feature set** — trains on `kdd_train.csv` / `kdd_test.csv` (KDD schema) but, instead of using all 41 columns, restricts the model to six hand-picked features: `src_bytes`, `dst_bytes`, `duration`, `same_srv_rate` (numerical) plus `protocol_type`, `flag` (categorical). `src_bytes`/`dst_bytes` get a `log1p` transform before IQR outlier clipping, per-feature `RobustScaler`, and a final `StandardScaler` over the combined numeric + label-encoded categorical matrix.
- **XGBoost-derived feature importance** — before training the neural nets, a quick `XGBClassifier` is fit on the full preprocessed data purely to obtain `feature_importances_`, which are min-max normalized to `[0, 1]` and saved as `feature_weights.pkl`. These weights are *not* used for feature selection here — they drive the masking curriculum described below.
- **Gaussian data augmentation** — `augment_data()` adds small Gaussian noise (`noise_level=0.05`) to a copy of the training matrix and appends it to the original data before SMOTE, roughly doubling the sample count.
- **SMOTE oversampling + stratified split** — the augmented data is balanced with `SMOTE`, then split 80/20 (stratified) into train/validation sets.
- **LayerNorm-augmented recurrent models** — unlike the base `RNNModel`/`LSTMModel` used elsewhere in the repo, this script's `RNNModel` and `LSTMModel` apply `nn.LayerNorm(64)` to the recurrent hidden state before the final linear+sigmoid head, for more stable training.
- **Three-stage random feature masking** (`enable_three_stage_random_masking=True`) — a curriculum-style input dropout applied only during training, implemented in `_apply_random_masking()`:
  - **Epochs 1–10**: fixed 5% element-wise mask probability.
  - **Epochs 11–30**: probability ramps linearly from 5% up to 20% (`+0.75%` per epoch).
  - **Epochs 31+**: fixed at 20%.
  - The per-feature mask probability is further scaled by `1 − feature_importance`, so the six input features are masked less often the more important XGBoost judged them to be (`feature_weights` from the step above).
- **Explicit L1/L2 regularization** — `l2_lambda` is passed as `weight_decay` to the `AdamW` optimizer (0.001 for ANN/CNN, 0.005 for RNN/LSTM in `main()`), and an optional manual L1 penalty (`l1_lambda`) can be added directly to the loss.
- **Early stopping** — training runs up to 50 epochs per base model, but stops early (patience = 5 epochs without validation-loss improvement) and restores the best-validation-loss weights before saving.
- **Dynamic class weighting** — same `DynamicClassWeighting` / `DynamicFocalLoss` mechanism as the rest of the ACRL family (`focal_adaptive` method, updated every 3 epochs).
- **Meta-model** — stacks the four base learners' probabilities and fits an `XGBClassifier` (`reg_alpha=0.1`, `reg_lambda=1.0`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`) with `scale_pos_weight` set from the class ratio, then reports Accuracy/Precision/Recall/F1/AUC on the training data.

**Outputs**: `final_ann.pth`, `final_cnn.pth`, `final_rnn.pth`, `final_lstm.pth`, `final_meta_model.pkl`, `model_config.pkl`, plus the fitted preprocessors (`feature_scalers.pkl`, `label_encoders.pkl`, `final_scaler.pkl`, `selected_features.pkl`) and, uniquely to this script, `feature_weights.pkl` (the XGBoost-derived importance scores used for masking). Because it shares the `final_*` artifact filenames with the other `aclr*.py` scripts, running it will overwrite artifacts produced by a previous run of a different variant.

```bash
python aclr_Robust_new.py
```

## Script Reference

See [Repository Layout](#repository-layout) above for a categorized index of all scripts. A few naming conventions to note:

- **`aclr_*` / `*aclr*`** — training pipelines implementing the core ACRL method; the suffix indicates the dataset, scaler, or feature-set variant used.
- **`*_to_kdd.py` / `*_to_nb15.py` / `convert_*`** — one-off or reusable dataset-format converters.
- **`*test*.py` / `Valid.py` / `verify_*.py`** — evaluation, validation, and sanity-check scripts.
- Files prefixed `KDD_` or `NB15_` in the artifact set indicate which schema/dataset the saved model was trained on.

## Supplementary Documentation

- [`UKM_to_KDD_README.md`](UKM_to_KDD_README.md) — detailed field-by-field mapping used to convert UKM-IDS20 into the KDD schema (Chinese).
- [`合併報告.md`](合併報告.md) — statistics and validation report for the merged UKM-IDS20 KDD-format dataset (Chinese).
- [`XGBoost_Regularization_README.md`](XGBoost_Regularization_README.md) — guide to the L1 (`reg_alpha`) / L2 (`reg_lambda`) regularization settings used in the meta-classifier, with tuning heuristics and troubleshooting tips (Chinese).

## Notes

- Large data files (`.csv` other than the three whitelisted training/testing sets, `.npy`, `.pcap`, `.png`, `.xlsx`) are excluded via `.gitignore` to keep the repository lightweight; regenerate them locally using the conversion/training scripts.
- `BlazorApp.slnx` references a `../blazor/BlazorApp/BlazorApp.csproj` project that lives outside this repository and is not included here — it is not required to run the Python/ML pipeline.
- `GNN.py` currently only loads the `Mireu-Lab/NSL-KDD` dataset from the Hugging Face Hub; it does not yet implement a graph-based model despite the filename.
- Random seeds are fixed (`set_seed(42)` / `np.random.seed(42)` / `torch.manual_seed(42)`) throughout the training scripts for reproducibility.
