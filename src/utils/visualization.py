"""
Visualization utilities for the Customer Churn Predictor project.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_feature_importance(model, feature_names: List[str], top_n: int = 10):
    """
    Plot feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (List[str]): List of feature names
        top_n (int): Number of top features to plot
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, features: Optional[List[str]] = None):
    """
    Plot correlation matrix for numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (Optional[List[str]]): List of features to include
    """
    if features:
        df = df[features]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(df: pd.DataFrame, features: List[str], target_col: str):
    """
    Plot distributions of features by target class.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (List[str]): List of features to plot
        target_col (str): Target column name
    """
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4*n_features))
    
    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, hue=target_col, ax=axes[i])
        axes[i].set_title(f'{feature} Distribution by {target_col}')
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray):
    """
    Plot ROC curve.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show() 