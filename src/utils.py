"""
Utility Functions
Helper functions for the PDF classifier
"""
import pandas as pd
from pathlib import Path
import logging
from project_config import LABELS_PATH, RESULTS_DIR  


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_labels(labels_path: Path = LABELS_PATH) -> pd.DataFrame:  
    """
    Load labels from CSV file
    """
    if not Path(labels_path).exists(): # Check if file exists
        logger.error(f"Labels file not found: {labels_path}") 
        return None # Return None if file does not exist
    
    df = pd.read_csv(labels_path) # Read CSV into DataFrame
    logger.info(f"Loaded labels from {labels_path}") # Log success message
    return df # Return DataFrame of labels


def save_predictions(filenames, predictions, scores, output_path=None): 
    """
    Save model predictions to CSV
    """
    if output_path is None: 
        output_path = RESULTS_DIR / 'predictions.csv' # Default output path  
    
    df = pd.DataFrame({
        'filename': filenames, # PDF filenames
        'prediction': predictions, # Numeric predictions: 1 for useful, 0 for not useful
        'anomaly_score': scores, # Higher means more likely 'useful'
        'label': ['useful' if p == 1 else 'not_useful' for p in predictions] # Map numeric predictions to string labels
    })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    df.to_csv(output_path, index=False) # Save DataFrame to CSV
    logger.info(f"Predictions saved to {output_path}") 
    
    return df

