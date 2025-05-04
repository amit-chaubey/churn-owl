"""
Data processing module for the Customer Churn Predictor project.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)

def load_data(filename: str) -> pd.DataFrame:
    """
    Load data from the raw data directory.
    
    Args:
        filename (str): Name of the file to load
        
    Returns:
        pd.DataFrame: Loaded data
    """
    file_path = RAW_DATA_DIR / filename
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and converting data types.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Create a copy to avoid modifying the original data
    df = df.copy()
    
    # Convert TotalCharges to numeric, handling any non-numeric values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values in TotalCharges with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Convert target variable to binary
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
    
    return df

def create_preprocessing_pipeline() -> Pipeline:
    """
    Create a preprocessing pipeline for numerical and categorical features.
    
    Returns:
        Pipeline: Preprocessing pipeline
    """
    # Create preprocessing steps for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
    
    return preprocessor

def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save processed data to the processed data directory.
    
    Args:
        df (pd.DataFrame): Data to save
        filename (str): Name of the file to save
    """
    file_path = PROCESSED_DATA_DIR / filename
    df.to_csv(file_path, index=False) 