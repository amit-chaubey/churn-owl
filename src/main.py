"""
Main script for the Customer Churn Predictor project.
"""
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)
from src.data.data_processor import (
    load_data,
    preprocess_data,
    create_preprocessing_pipeline,
    save_processed_data
)
from src.models.model_trainer import ModelTrainer
from src.utils.visualization import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_roc_curve
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the customer churn prediction pipeline."""
    try:
        # Load and preprocess data
        logger.info("Loading data...")
        df = load_data("customer_churn.csv")
        
        logger.info("Preprocessing data...")
        df = preprocess_data(df)
        
        # Save processed data
        save_processed_data(df, "processed_data.csv")
        
        # Split features and target
        X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        y = df[TARGET_COLUMN]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Create and fit preprocessing pipeline
        logger.info("Creating preprocessing pipeline...")
        preprocessor = create_preprocessing_pipeline()
        
        # Train model
        logger.info("Training model...")
        trainer = ModelTrainer()
        trainer.train(X_train, y_train, preprocessor)
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = trainer.evaluate(X_test, y_test)
        logger.info(f"Model metrics: {metrics}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        # Plot feature importance
        feature_names = (preprocessor.named_transformers_['cat']
                        .named_steps['onehot']
                        .get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names = list(feature_names) + NUMERICAL_FEATURES
        plot_feature_importance(trainer.model, feature_names)
        
        # Plot confusion matrix
        y_pred = trainer.predict(X_test)
        plot_confusion_matrix(y_test, y_pred)
        
        # Plot correlation matrix
        plot_correlation_matrix(df, NUMERICAL_FEATURES)
        
        # Plot feature distributions
        plot_feature_distributions(df, NUMERICAL_FEATURES, TARGET_COLUMN)
        
        # Plot ROC curve
        y_pred_proba = trainer.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_pred_proba)
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model("models/customer_churn_model.joblib")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 