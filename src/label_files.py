'''
Automatically label PDFs using an external labeling service.
'''

import sys
import logging
import pandas as pd

sys.path.insert(0, 'src')

from loader import PDFLoader, TXTLoader
from typing import Literal
from project_config import *

if not LOGS_DIR.exists():  # Ensure logs directory exists
    LOGS_DIR.mkdir() # Create logs directory if it doesn't exist

logging.basicConfig( # Configure logging
    filename=LOGS_DIR / 'label_files.log', # Log file path 
    filemode='a', # Append mode
    level=logging.INFO, # Log level 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Log format
    datefmt='%Y-%m-%d %H:%M:%S' # Date format
)

logger = logging.getLogger(__name__) # Get logger instance


def create_labels(file_type: Literal['pdf', 'txt'] = 'pdf'): 
    """
    Create labels for {file_type.upper()}s in RAW_PDFS_DIR or RAW_TXTS_DIR (not useful) and USEFUL_PDFS_DIR 
    or USEFUL_TXTS_DIR (useful).
    """

    if file_type == 'pdf': 
        loader_not_useful = PDFLoader(pdf_dir=RAW_PDFS_DIR)
        loader_useful = PDFLoader(pdf_dir=USEFUL_PDFS_DIR)
    elif file_type == 'txt':
        loader_not_useful = TXTLoader(txt_dir=RAW_TXTS_DIR)
        loader_useful = TXTLoader(txt_dir=USEFUL_TXTS_DIR)
    else:
        raise ValueError(f"Invalid file_type: {file_type}")
    
    logger.info("=" * 60)
    logger.info(f"{file_type.upper()} Labeling - Supervised Mode")
    logger.info("=" * 60)
    not_useful_files = loader_not_useful.get_files()  # Get 'not useful' files
    useful_files = loader_useful.get_files() # Get 'useful' files

    logger.info(f"Found {len(not_useful_files)} 'not useful' {file_type.upper()}") # Log count of 'not useful' files
    logger.info(f"Found {len(useful_files)} 'useful' {file_type.upper()}") # Log count of 'useful' files
            
    if len(useful_files) == 0: # Check if there are any 'useful' files
        logger.warning(f"No 'useful' {file_type.upper()} files found in data/useful_pdfs/ or data/useful_txts/. Please add {file_type.upper()}s and rerun.") # Log warning
        logger.info("Exiting labeling process.") # Log exit message
        return

    # Combine all files and labels
    all_files = not_useful_files + useful_files # Combine lists of files
    all_labels_list = ['not_useful'] * len(not_useful_files) + ['useful'] * len(useful_files) # Create corresponding labels

    # Create DataFrame
    df = pd.DataFrame({ # Create DataFrame with filenames and labels
        'filename': [f.name for f in all_files], # Extract filenames
        'label': all_labels_list # Corresponding labels
    })

    # Split into train/test
    from sklearn.model_selection import train_test_split # Import train_test_split function

    train_df, test_df = train_test_split( # Split DataFrame into train and test sets
        df, # DataFrame to split
        test_size=TEST_SIZE, # Proportion of test set
        stratify=df['label'], # Stratify by label guarantees proportional representation between splits, for example if 20% of data is useful, both train and test will have 20% useful
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
    logger.info(f"  Total: {len(all_labels)} {file_type.upper()}s") # Log total count
    logger.info(f"  Training: {len(train_df)} ({len(train_df)/len(all_labels)*100:.1f}%)")  # Log training count
    logger.info(f"  Testing: {len(test_df)} ({len(test_df)/len(all_labels)*100:.1f}%)") # Log testing count
    logger.info(f"\nClass Distribution:") # Log class distribution
    for split in ['train', 'test']: # Log distribution per split
        split_data = all_labels[all_labels['split'] == split] # Filter data for the split
        for label in ['not_useful', 'useful']: # Log count per label
            count = len(split_data[split_data['label'] == label]) # Count of each label
            logger.info(f"  {split} - {label}: {count}") # Log count
    
    logger.info(f"\nNext step: Run 'python main.py' to train the model with FILE_TYPE='{file_type}'") # Log next step message
    
    return all_labels

if __name__ == "__main__":
    create_labels()