import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocessing(df: pd.DataFrame):
    df = df[df['FlowPkts/s'] != np.inf]
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, axis=1, inplace=True)
    df = df.drop_duplicates()

    # Create sorted ip pairs
    ip_pairs = [tuple(sorted([src_ip, dst_ip])) for src_ip, dst_ip in zip(df['SrcIP'], df['DstIP'])]
    numerical_columns = [col for col in df.columns if col not in ['Label', 'SrcIP', 'DstIP'] and df[col].dtype in ['int64', 'float64']]

    # Create data
    X = df[numerical_columns]
    y = df['Label']
    
    X_train, X_test, y_train, y_test, pairs_train, pairs_test = train_test_split(
        X, y, ip_pairs, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, pairs_train, pairs_test