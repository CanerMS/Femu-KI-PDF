"""
Configuration Module
Centralized configuration for the PDF classifier
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent # Project root directory
DATA_DIR = BASE_DIR / "data" # Data directory
RESULTS_DIR = BASE_DIR / "results" # Results directory
LOGS_DIR = BASE_DIR / "logs" # Logs directory

# Data directories
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs" # Directory for raw PDFs
USEFUL_PDFS_DIR = DATA_DIR / "useful_pdfs" # Directory for useful PDFs
EXTRACTED_TEXTS_DIR = DATA_DIR / "extracted_texts" # Directory for extracted texts
LABELS_PATH = DATA_DIR / "labels.csv" # Path for labels CSV

# Model parameters
MAX_FEATURES = 2000 # Maximum number of features for vectorizer
NGRAM_RANGE = (1, 2) # N-gram range for text vectorization
TEST_SIZE = 0.25 # Proportion of data for test set
RANDOM_STATE = 42 # Random state for reproducibility
SMOTE_THRESHOLD = 5  # Imbalance ratio threshold

# Model hyperparameters
N_ESTIMATORS = 100 # Number of trees in Random Forest
MAX_DEPTH = 10 # Maximum depth of each tree
MIN_SAMPLES_SPLIT = 5 # Minimum samples required to split a node

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, LOGS_DIR, RAW_PDFS_DIR,     # List of directories to create
                  USEFUL_PDFS_DIR, EXTRACTED_TEXTS_DIR]: # Iterate through each directory path
    dir_path.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist