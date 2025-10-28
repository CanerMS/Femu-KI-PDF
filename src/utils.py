"""
Utility Functions
Helper functions for the PDF classifier
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_predictions(filenames, predictions, scores, output_path='results/predictions.csv'):
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

def load_labels(labels_path='data/labels.csv'):
    if Path(labels_path).exists():
        df = pd.read_csv(labels_path)
        logger.info(f"Loaded labels from {labels_path}")
        return df
    else:
        logger.warning(f"Labels file not found: {labels_path}")
        return None

def create_labels_template(pdf_files, output_path='data/labels.csv'):
    df = pd.DataFrame({
        'filename': [f.name for f in pdf_files],
        'label': ['not_useful'] * len(pdf_files)
    })
    df.to_csv(output_path, index=False)
    logger.info(f"Labels template created at {output_path}")
    return df
