"""
Model training module for the Customer Churn Predictor project.
"""
import joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from src.config import (
    MODEL_PARAMS,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN
)

class ModelTrainer:
    """Class for training and evaluating machine learning models."""
    
    def __init__(self, model_name: str = "random_forest"):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        self.model = self._create_model()
        self.preprocessor = None
        
    def _create_model(self) -> RandomForestClassifier:
        """
        Create a model instance with specified parameters.
        
        Returns:
            RandomForestClassifier: Model instance
        """
        return RandomForestClassifier(
            **MODEL_PARAMS[self.model_name],
            random_state=RANDOM_STATE
        )
    
    def train(self, X_train, y_train, preprocessor=None):
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target
            preprocessor: Optional preprocessor to use
        """
        self.preprocessor = preprocessor
        if preprocessor is not None:
            X_train = preprocessor.fit_transform(X_train)
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted values
        """
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def save_model(self, model_path: str):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor
        }, model_path)
    
    @classmethod
    def load_model(cls, model_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            ModelTrainer: ModelTrainer instance with loaded model
        """
        saved_model = joblib.load(model_path)
        trainer = cls()
        trainer.model = saved_model['model']
        trainer.preprocessor = saved_model['preprocessor']
        return trainer 