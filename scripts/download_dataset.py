"""
Script to download and prepare the Telco Customer Churn dataset.
"""
import os
import pandas as pd
import requests
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset URL (Telco Customer Churn dataset from Kaggle)
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def download_dataset():
    """Download the dataset and save it to the raw data directory."""
    try:
        # Create raw data directory if it doesn't exist
        raw_data_dir = Path("data/raw")
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the dataset
        logger.info("Downloading dataset...")
        response = requests.get(DATASET_URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the dataset
        dataset_path = raw_data_dir / "customer_churn.csv"
        with open(dataset_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Dataset downloaded successfully to {dataset_path}")
        
        # Verify the dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info("\nDataset columns:")
        for col in df.columns:
            logger.info(f"- {col}")
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    download_dataset() 