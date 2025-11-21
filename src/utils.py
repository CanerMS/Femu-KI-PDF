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
    if not Path(labels_path).exists():
        logger.error(f"Labels file not found: {labels_path}")
        return None
    
    df = pd.read_csv(labels_path)
    logger.info(f"Loaded labels from {labels_path}")
    return df


def save_predictions(filenames, predictions, scores, output_path=None): 
    """
    Save model predictions to CSV
    """
    if output_path is None:
        output_path = RESULTS_DIR / 'predictions.csv'  
    
    df = pd.DataFrame({
        'filename': filenames,
        'prediction': predictions,
        'anomaly_score': scores,
        'label': ['useful' if p == 1 else 'not_useful' for p in predictions]
    })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return df

