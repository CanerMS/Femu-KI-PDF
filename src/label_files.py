'''
Automatically label PDFs using an external labeling service.
'''

import sys
import logging
import pandas as pd

sys.path.insert(0, 'src')

from loader import PDFLoader
from pathlib import Path
from project_config import RAW_PDFS_DIR, USEFUL_PDFS_DIR, LABELS_PATH, LOGS_DIR, TEST_SIZE, RANDOM_STATE 

if not LOGS_DIR.exists():  # Ensure logs directory exists
    LOGS_DIR.mkdir() # Create logs directory if it doesn't exist

logging.basicConfig( # Configure logging
    filename=LOGS_DIR / 'label_pdfs.log', # Log file path 
    filemode='a', # Append mode
    level=logging.INFO, # Log level 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Log format
    datefmt='%Y-%m-%d %H:%M:%S' # Date format
)

logger = logging.getLogger(__name__) # Get logger instance


def create_labels(): 
    """
    Create labels for PDFs in RAW_PDFS_DIR (not useful) and USEFUL_PDFS_DIR (useful).
    """
    logger.info("=" * 60)
    logger.info("PDF Labeling - Supervised Mode")
    logger.info("=" * 60)
    
    # Load PDFs from both directories
    loader_not_useful = PDFLoader(pdf_dir=RAW_PDFS_DIR)  # Load 'not useful' PDFs
    loader_useful = PDFLoader(pdf_dir=USEFUL_PDFS_DIR)   # Load 'useful' PDFs
    
    not_useful_files = loader_not_useful.get_pdf_files()  # Get 'not useful' PDFs
    useful_files = loader_useful.get_pdf_files() # Get 'useful' PDFs

    logger.info(f"Found {len(not_useful_files)} 'not useful' PDFs") # Log count of 'not useful' PDFs 
    logger.info(f"Found {len(useful_files)} 'useful' PDFs") # Log count of 'useful' PDFs
            
    if len(useful_files) == 0: # Check if there are any 'useful' PDFs
        logger.warning("No 'useful' PDF files found in data/useful_pdfs/. Please add PDFs and rerun.") # Log warning
        logger.info("Exiting labeling process.") # Log exit message
        return

    # Combine all PDFs and labels
    all_pdfs = not_useful_files + useful_files # Combine lists of PDFs
    all_labels_list = ['not_useful'] * len(not_useful_files) + ['useful'] * len(useful_files) # Create corresponding labels

    # Create DataFrame
    df = pd.DataFrame({ # Create DataFrame with filenames and labels
        'filename': [f.name for f in all_pdfs], # Extract filenames
        'label': all_labels_list # Corresponding labels
    })

    # Split into train/test
    from sklearn.model_selection import train_test_split # Import train_test_split function

    train_df, test_df = train_test_split( # Split DataFrame into train and test sets
        df, # DataFrame to split
        test_size=TEST_SIZE, # Proportion of test set
        stratify=df['label'], # Stratify by label
        random_state=RANDOM_STATE  # Random state for reproducibility
    )

    train_df['split'] = 'train'      # 75%
    test_df['split'] = 'test'        # 25%

    all_labels = pd.concat([train_df, test_df], ignore_index=True) # Combine train and test DataFrames
    all_labels = all_labels.sort_values(by='filename').reset_index(drop=True) # Sort by filename

    # Save labels
    all_labels.to_csv(LABELS_PATH, index=False) # Save labels to CSV
    logger.info(f"\nLabels saved to {LABELS_PATH}") # Log success message
    
    logger.info("=" * 60) # Separator
    logger.info(f"Labels saved to {LABELS_PATH} successfully.") # Log success message
    logger.info(f"\nDataset Summary:") # Log dataset summary
    logger.info(f"  Total: {len(all_labels)} PDFs") # Log total count
    logger.info(f"  Training: {len(train_df)} ({len(train_df)/len(all_labels)*100:.1f}%)")  # Log training count
    logger.info(f"  Testing: {len(test_df)} ({len(test_df)/len(all_labels)*100:.1f}%)") # Log testing count
    logger.info(f"\nClass Distribution:") # Log class distribution
    for split in ['train', 'test']: # Log distribution per split
        split_data = all_labels[all_labels['split'] == split] # Filter data for the split
        for label in ['not_useful', 'useful']: # Log count per label
            count = len(split_data[split_data['label'] == label]) # Count of each label
            logger.info(f"  {split} - {label}: {count}") # Log count
    
    logger.info("\nNext step: Run 'python main.py' to train the model") # Log next step message
    
    return all_labels

if __name__ == "__main__":
    create_labels()