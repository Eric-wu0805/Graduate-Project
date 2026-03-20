import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, SimpleRNN, Input, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
import uuid

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Data Preprocessing
def preprocess_unsw_nb15(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Handle missing values
    df = df.fillna('NaN')
    
    # Label encoding for categorical features
    categorical_columns = df.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Separate features and target
    X = df.drop('attack_cat', axis=1)  # Assuming 'label' is the target column
    y = df['attack_cat']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Reshape for CNN and LSTM inputs
    X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test, y_train, y_test, X_train_cnn, X_test_cnn

# 2. Define Individual Models
def create_ann(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def create_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def create_lstm(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=False)(inputs)
    x = Dense(32, activation='sigmoid')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def create_rnn(input_shape):
    inputs = Input(shape=input_shape)
    x = SimpleRNN(64, return_sequences=False)(inputs)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# 3. Create Stacking ACLR Model
def create_aclr_model(input_shape, input_shape_cnn):
    # Create individual models
    ann_model = create_ann(input_shape)
    cnn_model = create_cnn(input_shape_cnn)
    lstm_model = create_lstm(input_shape_cnn)
    rnn_model = create_rnn(input_shape_cnn)
    
    # Define inputs
    input_ann = Input(shape=input_shape)
    input_cnn_lstm_rnn = Input(shape=input_shape_cnn)
    
    # Get outputs from each model
    ann_output = ann_model(input_ann)
    cnn_output = cnn_model(input_cnn_lstm_rnn)
    lstm_output = lstm_model(input_cnn_lstm_rnn)
    rnn_output = rnn_model(input_cnn_lstm_rnn)
    
    # Concatenate all outputs
    combined = Concatenate()([ann_output, cnn_output, lstm_output, rnn_output])
    
    # Meta-learner
    x = Dense(64, activation='relu')(combined)
    x = Dense(32, activation='relu')(x)
    final_output = Dense(1, activation='sigmoid')(x)
    
    # Create final model
    model = Model(inputs=[input_ann, input_cnn_lstm_rnn], outputs=final_output)
    
    return model

# 4. Training and Evaluation
def train_and_evaluate(model, X_train, X_train_cnn, y_train, X_test, X_test_cnn, y_test, epochs=30, batch_size=32):
    # Compile model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    history = model.fit(
        [X_train, X_train_cnn], y_train,
        validation_data=([X_test, X_test_cnn], y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate model
    y_pred = (model.predict([X_test, X_test_cnn]) > 0.5).astype(int)
    y_pred_proba = model.predict([X_test, X_test_cnn])
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, history

def train_and_evaluate(model, X_train, X_train_cnn, y_train, X_test, X_test_cnn, y_test, epochs=30, batch_size=32):
    # Compile model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(
        [X_train, X_train_cnn], y_train,
        validation_data=([X_test, X_test_cnn], y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # === 評估 Training Set ===
    y_train_pred = (model.predict([X_train, X_train_cnn]) > 0.5).astype(int)
    y_train_proba = model.predict([X_train, X_train_cnn])
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_proba)
    }

    # === 評估 Test Set ===
    y_test_pred = (model.predict([X_test, X_test_cnn]) > 0.5).astype(int)
    y_test_proba = model.predict([X_test, X_test_cnn])
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }

    return train_metrics, test_metrics, history


# 5. Main Execution
def main():
    dataset_path = 'UNSW_NB15_training-set.csv'
    X_train, X_test, y_train, y_test, X_train_cnn, X_test_cnn = preprocess_unsw_nb15(dataset_path)
    
    aclr_model = create_aclr_model(input_shape=X_train.shape[1], input_shape_cnn=X_train_cnn.shape[1:])
    
    train_metrics, test_metrics, history = train_and_evaluate(
        aclr_model,
        X_train, X_train_cnn, y_train,
        X_test, X_test_cnn, y_test,
        epochs=1,  # 可以先設少一點測試效果
        batch_size=32
    )
    
    print("\nTraining Set Performance:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nTest Set Performance:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")



if __name__ == "__main__":
    main()
  