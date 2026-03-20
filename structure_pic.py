from diagrams import Diagram, Cluster
from diagrams.aws.ml import SageMaker, TrainingJob
from diagrams.aws.storage import S3
from diagrams.aws.compute import EC2
from diagrams.onprem.ml import Jupyter
from diagrams.onprem.database import PostgreSQL
from diagrams.generic.device import Server
from diagrams.generic.network import Firewall
from diagrams.custom import Custom

# 自定義圖標
preprocessing_icon = "./icons/preprocessing.png" # 假設有自定義圖標
smote_icon = "./icons/smote.png"

with Diagram("Botnet Detection Stacked Ensemble", show=False, direction="LR"):

    with Cluster("Data Preprocessing"):
        raw_data = S3("Raw CSV Data")
        
        with Cluster("Feature Engineering"):
            fe_1 = Custom("Data Cleaning\n(NaN/Outliers)", preprocessing_icon)
            fe_2 = Custom("Feature Scaling\n(RobustScaler/StandardScaler)", preprocessing_icon)
            fe_3 = Custom("Categorical Encoding\n(LabelEncoder)", preprocessing_icon)
            fe_4 = Custom("Feature Selection\n& Augmentation", preprocessing_icon)
            
        data_processing = [fe_1, fe_2, fe_3, fe_4]
        
        balancing = Custom("Data Balancing\n(SMOTE)", smote_icon)
        
        preprocessed_data = S3("Preprocessed Data")
        
        raw_data >> data_processing >> balancing >> preprocessed_data

    with Cluster("Base Models"):
        # 四個深度學習模型作為 Base Models
        with Cluster("Neural Network Models"):
            ann = TrainingJob("ANNModel\n(FocalLoss)")
            cnn = TrainingJob("CNNModel\n(FocalLoss)")
            rnn = TrainingJob("RNNModel\n(FocalLoss)")
            lstm = TrainingJob("LSTMModel\n(FocalLoss)")

        base_models = [ann, cnn, rnn, lstm]
        
        preprocessed_data >> base_models

    with Cluster("Meta-model Training"):
        # Meta-model 的訓練過程
        meta_data_gen = EC2("Generate Meta-Features\n(Base Model Predictions)")
        meta_model_trainer = TrainingJob("XGBoost Classifier\n(Meta-model)")
        
        base_models >> meta_data_gen
        meta_data_gen >> meta_model_trainer

    with Cluster("Final Model"):
        final_model = PostgreSQL("Saved Models\n(.pth, .pkl)")
        
        # 儲存 Base Models
        base_models >> final_model
        
        # 儲存 Meta-model
        meta_model_trainer >> final_model

    with Cluster("Deployment/Inference"):
        inference_server = Server("Inference Server")
        
        # 部署流程
        final_model >> inference_server
        
        # 流量檢測
        internet_traffic = Firewall("Internet Traffic")
        inference_server << internet_traffic