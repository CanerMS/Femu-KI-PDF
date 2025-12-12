"""# read PDF files and extract text content
PDF Loader Module
Loads PDF files from the raw_pdfs directory
"""
import os
from pathlib import Path
from typing import List, Dict, Literal
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from project_config import RAW_PDFS_DIR, RAW_TXTS_DIR, USEFUL_PDFS_DIR, USEFUL_TXTS_DIR 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Base Class for Loaders
class UnifiedLoader():
    """ Unified Loader for PDF and TXT files and more when needed """
    def __init__(self, 
                data_dir: Path = RAW_PDFS_DIR, 
                useful_dir: Path = USEFUL_PDFS_DIR,
                file_type: Literal['pdf', 'txt'] = 'pdf' # 'pdf' or 'txt' Literal type hint 
                ):
        """
        Initialize Unified Loader
        
        Args:
            data_dir: Main data directory (e.g., raw_pdfs or raw_txts)
            useful_dir: Useful data directory (e.g., useful_pdfs or useful_txts)
            file_type: 'pdf' or 'txt'
        """
        self.data_dir = Path(data_dir)
        self.useful_dir = Path(useful_dir)
        self.file_type = file_type
        self.extension = f".{file_type}" # File extension based on type

        #Validate directories
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        logger.info(f"Initialized {file_type.upper()} loader: ") # Log initialization message
        logger.info(f"  Main dir: {self.data_dir}")
        logger.info(f"  Useful dir: {self.useful_dir}")

    def get_files(self) -> List[Path]:
        """
        Get all files (PDF or TXT) from the main directory

        Returns:
            List of Path objects for files
        """
        files = list(self.data_dir.glob(f"*{self.extension}")) # Get all files with the specified extension
        logger.info(f"Found {len(files)} {self.file_type.upper()} files in {self.data_dir}")
        return sorted(files) # it sorts the list of files alphabetically

    def get_file_info(self) -> List[Dict[str, str]]:
        """
        Get information about all files (PDF or TXT)
        Returns:
        List of dictionaries with file metadata
        """
        files = self.get_files() # Get all files
        file_info = [] # Initialize list to hold file metadata

        for file_path in files: # Iterate over each file
            info = {
                'filename': file_path.name, # filename
                'path': str(file_path), # Full path as string
                'size_kb': file_path.stat().st_size / 1024 # Size in kilobytes
            }
            file_info.append(info)

        return file_info

    def get_files_by_split(self, split: str, labels_df) -> List[Path]:
        """
        Get files (PDF or TXT) for a specific split (train/test) based on labels DataFrame
        Args:
        split: 'train' or 'test'
        labels_df: DataFrame containing 'filename', 'label' and 'split' columns
        Returns:
        List of Path objects for the specified split
        """
        split_filenames = labels_df[labels_df['split'] == split]['filename'].tolist() # Get filenames for the split, e.g., 'train' or 'test'

        files = [] # Initialize list to hold file paths
        for filename in split_filenames: # Iterate over each filename
            file_path = self.data_dir / filename # Construct full path

            if file_path.exists(): # If file exists in main directory
                files.append(file_path) # Add to list
            else:
                # Check in useful directory
                useful_path = self.useful_dir / filename  # Use useful directory
                if useful_path.exists(): # If file exists in useful directory
                    files.append(useful_path) # Add to list
                else:
                    logger.warning(f"{self.file_type.upper()} file not found for {filename}")

        logger.info(f"Found {len(files)} {self.file_type.upper()}s for split '{split}'")
        return files

    def get_labels_for_files(self, files: List[Path], labels_df) -> List[int]:
        """
        Get numeric labels (0/1) for the given files (PDF or TXT) based on labels DataFrame

        Args:
            files: List of Path objects for files
            labels_df: DataFrame containing 'filename' and 'label' columns
    """
    
        label_map = {'not_useful': 0, 'useful': 1} # Map string labels to numeric
        labels = [] # Initialize list to hold labels

        for file_path in files:
            label_row = labels_df[labels_df['filename'] == file_path.name] # Find label for the file
            if not label_row.empty:
                label_str = label_row.iloc[0]['label'] # Get label string, e.g., 'useful' or 'not_useful'
                # iloc is used to access the first row of the filtered DataFrame
            
                labels.append(label_map[label_str]) # Default to 0 if label not found
            else:
                logger.warning(f"No label found for {file_path.name}, defaulting to 0")
                labels.append(0)
        return labels

    def split_train_test(self, test_size: float = 0.5, random_seed: int = 42) -> tuple: # returns train and test splits
        """
        Split files (PDF or TXT) into train and test sets

        Args:
            test_size: Proportion of data for testing (default: 0.5 for 50/50 split)
            random_seed: Random seed for reproducibility
        Returns:
            Tuple of (train_files, test_files)
        """
        import random

        files = self.get_files() # Get all files

        # without seed, different runs would produce different splits
        random.seed(random_seed) # Set random seed for reproducibility
        shuffled = files.copy() # Create a copy to shuffle
        random.shuffle(shuffled) # Shuffle the list of files

        split_idx = int(len(shuffled) * (1 - test_size)) # Calculate split index
        train_files = shuffled[:split_idx] # First part for training
        test_files = shuffled[split_idx:] # Second part for testing
        logger.info(f"Split: {len(train_files)} training, {len(test_files)} testing")
        return train_files, test_files # because of the shuffle, the files are randomly assigned to train and test sets

    def get_all_files(self, labels_df) -> List[Path]:
        """
        Get all files (PDF or TXT) from both main and useful directories

        Args:
            labels_df: DataFrame containing 'filename' column
        Returns:
            List of Path objects for all files
        """
        all_filenames = labels_df['filename'].tolist()
        files = []
        
        for filename in all_filenames:
            # Check main directory first (e.g., raw_pdfs)
            file_path = self.data_dir / filename
            if file_path.exists():
                files.append(file_path)
            else:
                # Check useful directory (e.g., useful_pdfs)
                useful_path = self.useful_dir / filename
                if useful_path.exists():
                    files.append(useful_path)
                else:
                    logger.warning(f"File not found: {filename} (checked both {self.data_dir} and {self.useful_dir})")
        
        logger.info(f"Loaded {len(files)}/{len(all_filenames)} {self.file_type.upper()}s from labels")
        return files
    
    def get_all_labels(self, labels_df) -> List[int]:
        """
        Get ALL numeric labels (0/1) from labels DataFrame
        
        Args:
            labels_df: DataFrame with 'label' column
        
        Returns:
            List of numeric labels (0=not_useful, 1=useful)
        """
        label_map = {'not_useful': 0, 'useful': 1}
        labels = [label_map[label] for label in labels_df['label']]
        
        logger.info(f"Loaded {len(labels)} labels: {labels.count(0)} not_useful, {labels.count(1)} useful")
        return labels
        

class PDFLoader(UnifiedLoader):
    """PDF Loader (backwards compatible)"""
    def __init__(self, 
                pdf_dir: Path = RAW_PDFS_DIR, 
                useful_dir: Path = USEFUL_PDFS_DIR):
        """
        Initialize PDF Loader
        
        Args:
            pdf_dir: Directory containing PDF files
            useful_dir: Directory containing useful PDF files
        """
        super().__init__(
            data_dir=pdf_dir, 
            useful_dir=useful_dir, 
            file_type='pdf'
            )
        
class TXTLoader(UnifiedLoader):
    """TXT Loader"""
    def __init__(self, 
                txt_dir: Path = RAW_TXTS_DIR, 
                useful_dir: Path = USEFUL_TXTS_DIR):
        """
        Initialize TXT Loader
        
        Args:
            txt_dir: Directory containing TXT files
            useful_dir: Directory containing useful PDF files
        """
        super().__init__(
            data_dir=txt_dir, 
            useful_dir=useful_dir, 
            file_type='txt'
            )



if __name__ == "__main__":
    # Test PDF loader
    print("\n" + "="*50)
    print("Testing PDF Loader")
    print("="*50)
    pdf_loader = PDFLoader()
    pdf_info = pdf_loader.get_file_info()
    print(f"Found {len(pdf_info)} PDFs")
    if pdf_info:
        print("\nFirst 3 PDFs:")
        for info in pdf_info[:3]:
            print(f"  - {info['filename']} ({info['size_kb']:.2f} KB)")
    
    # Test TXT loader
    print("\n" + "="*50)
    print("Testing TXT Loader")
    print("="*50)
    txt_loader = TXTLoader()
    txt_info = txt_loader.get_file_info()
    print(f"Found {len(txt_info)} TXTs")
    if txt_info:
        print("\nFirst 3 TXTs:")
        for info in txt_info[:3]:
            print(f"  - {info['filename']} ({info['size_kb']:.2f} KB)")