'''
Automatically label PDFs using an external labeling service.
'''

import sys
sys.path.insert(0, 'src')

from loader import PDFLoader
import pandas as pd
from pathlib import Path
import logging

if not Path('logs').exists():  # Ensure logs/ directory exists
    Path('logs').mkdir()


logging.basicConfig(
    filename = 'logs/label_pdfs.log', # Log file path
    filemode = 'a', # Append mode
    level = logging.INFO, # Show INFO, WARNING, ERROR, CRITICAL messages, hide DEBUG messages
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Log format
    datefmt='%Y-%m-%d %H:%M:%S' # Timestamp format, e.g., 2025-11-11 14:30:15 
    # asctime = timestamp, when log was created
    # name = module name where log was created, here label_pdfs.py
    # levelname = log level (INFO, WARNING, etc.)
    # message = your log message explaining what happened
)

logger = logging.getLogger(__name__)
 

def create_labels():
    """
    Create labels for PDFs in the data/raw_pdfs directory.
    Splits PDFs into train and test sets (50/50) and labels all as 'not_useful'.
    Saves labels to data/labels/pdf_labels.csv.
    """
    logger.info("=" * 60)
    logger.info("PDF LABELING - AUTOMATIC TRAIN-TEST SPLIT")
    logger.info("=" * 60)

    # Load all PDFs
    logger.info("loading PDFs from data/raw_pdfs...")
    loader = PDFLoader(pdf_dir="data/raw_pdfs")
    pdf_files = loader.get_pdf_files()

    if len(pdf_files) == 0:
        logger.warning("No PDF files found in the specified directory.")
        logger.info("Exiting labeling process.")
        return

    logger.info(f"Found {len(pdf_files)} PDF files. Preparing to label...") # Log number of found PDFs

    # Split into train and test sets separating it 50/50 

    train_pdfs, test_pdfs = loader.split_train_test(test_size=0.5, random_seed=42)

    logger.info(f"Train set {len(train_pdfs)} training PDFs")
    logger.info(f"Test set {len(test_pdfs)} testing PDFs")

    # Create traning labels

    logger.info("Creating labels for training PDFs...")

    train_labels = pd.DataFrame({
        'filename': [f.name for f in train_pdfs],
        'label' : ['not_useful'] * len(train_pdfs),
        'split': ['train'] * len(train_pdfs)
    })

    logger.info("Creating labels for testing PDFs...")

    # Create testing labels
    test_labels = pd.DataFrame({
        'filename' : [f.name for f in test_pdfs], # Get filenames for test PDFs
        'label' : ['not_useful'] * len(test_pdfs),
        'split': ['test'] * len(test_pdfs)
    })

    logger.info("Combining train and test labels...")

    # Combine
    all_labels = pd.concat([train_labels, test_labels], ignore_index=True) # Combine train and test labels into a single DataFrame
    all_labels = all_labels.sort_values(by='filename').reset_index(drop=True) # Sort by filename for consistency

    # Save to CSV
    output_path = Path('data/labels.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    all_labels.to_csv(output_path, index=False)
    logger.info(f"Label saved to {output_path} successfully.")

    # Print summary
    
    logger.info("=" * 60)
    logger.info(f"Labels saved to {output_path}")
    logger.info("\nDataset Summary:\n")
    logger.info(f"  Training samples: {len(train_labels)} ({len(train_labels)/len(pdf_files)*100:.1f}%)") # Log training samples and percentage
    logger.info(f"  Testing samples: {len(test_labels)} ({len(test_labels)/len(pdf_files)*100:.1f}%)") # Log testing samples and percentage
    logger.info(f" Total samples: {len(pdf_files)}")


    logger.info("\nLabel Distribution:")
    label_counts = all_labels.groupby(['split', 'label']).size() # Count labels in the combined DataFrame

    for (split, label), count in label_counts.items():
        logger.info(f"  {split} - {label}: {count}") # Log count for each split and label combination

    # Show first 10 rows of the labels
    logger.info("\nFirst 10 label entries:")
    print(all_labels.head(10))
    logger.info("\n Next step: Run 'python main.py' to train the model")

    return all_labels # Return the DataFrame containing all labels

if __name__ == "__main__":
    try:
        create_labels()
    except Exception as e:
        logger.error(f"An error occurred during labeling: {e}", exc_info=True)
        sys.exit(1)


