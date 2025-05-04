"""
Tests for the model trainer module.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.models.model_trainer import ModelTrainer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return X, y

def test_model_initialization():
    """Test model initialization."""
    trainer = ModelTrainer()
    assert trainer.model_name == "random_forest"
    assert trainer.model is not None
    assert trainer.preprocessor is None

def test_model_training(sample_data):
    """Test model training."""
    X, y = sample_data
    trainer = ModelTrainer()
    trainer.train(X, y)
    assert trainer.model is not None

def test_model_prediction(sample_data):
    """Test model prediction."""
    X, y = sample_data
    trainer = ModelTrainer()
    trainer.train(X, y)
    
    predictions = trainer.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)

def test_model_probability_prediction(sample_data):
    """Test model probability prediction."""
    X, y = sample_data
    trainer = ModelTrainer()
    trainer.train(X, y)
    
    probabilities = trainer.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert all(0 <= prob <= 1 for prob in probabilities.flatten())

def test_model_evaluation(sample_data):
    """Test model evaluation."""
    X, y = sample_data
    trainer = ModelTrainer()
    trainer.train(X, y)
    
    metrics = trainer.evaluate(X, y)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert 'confusion_matrix' in metrics
    
    # Check metric values are within valid ranges
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1 