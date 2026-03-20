import pandas as pd
import numpy as np
from glob import glob
import os

def merge_and_convert_kdd_to_cic2017():
    """
    Merge all NSL-KDD files and convert to CICIDS2017 format
    """
    print("Starting merge and conversion of NSL-KDD to CICIDS2017 format...")

    # Define NSL-KDD files (adjust paths as needed)
    kdd_files = [
        'kdd_test.csv',
        'kdd_train.csv'
    ]

    # Define NSL-KDD column names (41 features + label)
    kdd_columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'labels'
    ]

    all_data = []
    for file in kdd_files:
        try:
            print(f"Reading {file}...")
            df = pd.read_csv(file, header=None, names=kdd_columns)
            all_data.append(df)
            print(f"Successfully read {file}, shape: {df.shape}")
        except FileNotFoundError:
            print(f"File not found: {file}")
            continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    if not all_data:
        print("No files were successfully read!")
        return

    # Merge all data
    print("\nMerging all data...")
    df = pd.concat(all_data, ignore_index=True)
    print(f"Merged data shape: {df.shape}")

    # Create CICIDS2017 format DataFrame
    cic_df = pd.DataFrame()

    # Define CICIDS2017 features (based on typical CICIDS2017 dataset)
    cic_columns = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packets Length Total',
        'Bwd Packets Length Total', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
        'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
        'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min',
        'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
        'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Avg Packet Size',
        'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
        'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
        'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
        'Subflow Bwd Bytes', 'Init Fwd Win Bytes', 'Init Bwd Win Bytes', 'Fwd Act Data Packets',
        'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
        'Idle Std', 'Idle Max', 'Idle Min', 'Label'
    ]

    # Initialize CICIDS2017 DataFrame with zeros
    for col in cic_columns:
        cic_df[col] = 0

    # Map basic features
    print("\nMapping basic features...")
    cic_df['Flow Duration'] = df['duration']
    cic_df['Total Fwd Packets'] = df['src_bytes']
    cic_df['Total Backward Packets'] = df['dst_bytes']
    cic_df['Fwd Packets/s'] = df['wrong_fragment']

    # Map flow-based features (reverse mapping from original script)
    print("Mapping flow-based features...")
    cic_df['Flow Packets/s'] = df['count']
    cic_df['Flow Bytes/s'] = df['srv_count']
    cic_df['Flow IAT Mean'] = df['serror_rate']
    cic_df['Flow IAT Std'] = df['srv_serror_rate']
    cic_df['Flow IAT Max'] = df['rerror_rate']
    cic_df['Flow IAT Min'] = df['srv_rerror_rate']
    cic_df['Fwd IAT Mean'] = df['same_srv_rate']
    cic_df['Fwd IAT Std'] = df['diff_srv_rate']
    cic_df['Fwd IAT Max'] = df['srv_diff_host_rate']
    cic_df['Bwd IAT Mean'] = df['dst_host_count']
    cic_df['Bwd IAT Std'] = df['dst_host_srv_count']
    cic_df['Bwd IAT Max'] = df['dst_host_same_srv_rate']
    cic_df['Bwd IAT Min'] = df['dst_host_diff_srv_rate']
    cic_df['Active Mean'] = df['dst_host_same_src_port_rate']
    cic_df['Active Std'] = df['dst_host_srv_diff_host_rate']
    cic_df['Active Max'] = df['dst_host_serror_rate']
    cic_df['Active Min'] = df['dst_host_srv_serror_rate']
    cic_df['Idle Mean'] = df['dst_host_rerror_rate']
    cic_df['Idle Std'] = df['dst_host_srv_rerror_rate']

    # Set default values for unmapped features
    print("Setting default values for unmapped features...")
    cic_df['Fwd PSH Flags'] = df['flag'].apply(lambda x: 1 if x == 'PSH' else 0)
    cic_df['SYN Flag Count'] = df['flag'].apply(lambda x: 1 if x == 'SYN' else 0)
    cic_df['RST Flag Count'] = df['flag'].apply(lambda x: 1 if x == 'RST' else 0)
    cic_df['ACK Flag Count'] = df['flag'].apply(lambda x: 1 if x == 'ACK' else 0)
    cic_df['URG Flag Count'] = df['urgent']

    # Map labels
    print("\nProcessing labels...")
    print("Original NSL-KDD label distribution:")
    print(df['labels'].value_counts())

    # Convert labels to CICIDS2017 format (binary: BENIGN or ATTACK)
    print("\nConverting to binary classification...")
    cic_df['Label'] = df['labels'].apply(lambda x: 'BENIGN' if x == 'normal' else 'ATTACK')

    # Optional: Map specific NSL-KDD attack types to CICIDS2017 attack types
    # Example mapping (can be extended based on dataset specifics)
    attack_mapping = {
        'neptune': 'DoS',
        'smurf': 'DDoS',
        'back': 'DoS',
        'teardrop': 'DoS',
        'pod': 'DoS',
        'land': 'DoS',
        'satan': 'PortScan',
        'ipsweep': 'PortScan',
        'portsweep': 'PortScan',
        'nmap': 'PortScan'
        # Add more mappings as needed
    }
    cic_df['Label'] = df['labels'].apply(lambda x: attack_mapping.get(x, 'ATTACK') if x != 'normal' else 'BENIGN')

    print("\nConverted label distribution:")
    print(cic_df['Label'].value_counts())

    # Check and handle missing values
    print("\nChecking for missing values...")
    null_counts = cic_df.isnull().sum()
    print("Missing value counts:")
    print(null_counts[null_counts > 0])

    # Fill missing values with 0
    cic_df = cic_df.fillna(0)

    # Save to CICIDS2017 format CSV
    output_file = 'kdd_to_cic2017_converted.csv'
    cic_df.to_csv(output_file, index=False)
    print(f"\nSaved {output_file} ({len(cic_df)} records)")

    # Display final statistics
    print("\nFinal data statistics:")
    print(f"Total samples: {len(cic_df)}")
    print(f"Number of features: {len(cic_df.columns)}")
    print("\nLabel distribution:")
    print(cic_df['Label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

    print("\nConversion completed!")

if __name__ == "__main__":
    merge_and_convert_kdd_to_cic2017()