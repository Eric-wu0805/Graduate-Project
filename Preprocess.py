import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# 假設你的數據已經在一個 CSV 文件中
df = pd.read_csv('UNSW_NB15_training-set.csv')

# 預處理數據的函式
def preprocess_data(df):
    """預處理數據"""
    if df is None:
        return None, None
        
    # 處理分類特徵
    categorical_columns = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column].astype(str))
    #print(df)
    # 分離特徵和標籤
    if 'attack_cat' in df.columns:
        y = df['attack_cat']
        X = df.drop(['attack_cat'], axis=1)
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
   # print(y)
    # 標準化特徵
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# 調用預處理函式
X_scaled, y = preprocess_data(df)

# 查看結果
print(X_scaled)  # 標準化後的特徵數據
#print(y)         # 標籤（攻擊類別）
