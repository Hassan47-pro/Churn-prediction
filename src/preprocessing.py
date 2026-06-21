import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

def preprocess_data(df, target_col='Churn', test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline for Telco churn dataset.
    Returns: X_train_res, X_test, y_train_res, y_test, feature_names
    """
    df = df.copy()
    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    # Drop customerID
    df = df.drop(columns=['customerID'])
    # Derived features
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['HasPremiumSupport'] = ((df['OnlineSecurity'] == 'Yes') | (df['TechSupport'] == 'Yes')).astype(int)
    # Encode target
    df[target_col] = (df[target_col] == 'Yes').astype(int)
    # One-hot encode categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    # SMOTENC: identify categorical column indices after encoding (all bool/uint8)
    cat_indices = [i for i, dtype in enumerate(X_train.dtypes) if dtype in ['uint8', 'bool']]
    smote = SMOTENC(categorical_features=cat_indices, random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, X_test, y_train_res, y_test, X.columns.tolist()