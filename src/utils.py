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
    """
    Save model predictions to a CSV file.

    Args:
        filenames: List of PDF filenames.
        predictions: List of predicted labels (0 or 1).
        scores: List of anomaly scores.
        output_path: Path to the output CSV file.
    """
    df = pd.DataFrame({ # Create DataFrame for predictions
        'filename': filenames,
        'prediction': predictions,
        'anomaly_score': scores,
        'label': ['useful' if p == 1 else 'not_useful' for p in predictions] # Map 1 to 'useful', 0 to 'not_useful'
    })
    
    # Ensure output directory exists, for 'results/predictions.csv', create 'results/' if missing, useful when running first time
    Path(output_path).parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists, parents=True creates all necessary parent directories, here = results/ 
    df.to_csv(output_path, index=False) # Save DataFrame to CSV, index not needed
    logger.info(f"Predictions saved to {output_path}") # Log info message
    return df 

def load_labels(labels_path='data/labels.csv'):
    if Path(labels_path).exists():
        df = pd.read_csv(labels_path)  # Reads and loads CSV → panda's DataFrame
        logger.info(f"Loaded labels from {labels_path}")
        return df
    else:
        logger.warning(f"Labels file not found: {labels_path}")
        return None

def create_labels_template(pdf_files, output_path='data/labels.csv'):
    df = pd.DataFrame({
        'filename': [f.stem for f in pdf_files], # .stem gets filename without extension from Path object
        # f is Path object, f.stem gets filename only
        'label': ['not_useful'] * len(pdf_files) # Default all to 'not_useful'
    })
    df.to_csv(output_path, index=False)
    print(df['label'].value_counts())  # Display labels column
    logger.info(f"Labels template created at {output_path}")
    return df
