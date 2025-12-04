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

from project_config import RAW_PDFS_DIR, RAW_TXTS_DIR, USEFUL_PDFS_DIR 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Base Class for Loaders
class UnifiedLoader():
    """ Unified Loader for PDF and TXT files"""
    def __init__(self, 
                pdf_dir: Path = RAW_PDFS_DIR, 
                txt_dir: Path = RAW_TXTS_DIR,
                file_type: Literal['pdf', 'txt'] = 'pdf' # 'pdf' or 'txt' Literal type hint 
                ):
        """
        Initialize Unified Loader
        
        :param self: Description
        :param pdf_dir: Directory containing PDF files
        :type pdf_dir: Path
        :param txt_dir: Directory containing TXT files
        :type txt_dir: Path
        """
        self.pdf_loader = PDFLoader(pdf_dir=pdf_dir)
        self.txt_loader = TXTLoader(txt_dir=txt_dir)

class TXTLoader():
    """ Handles Loading TXT files from directory"""
    def __init__(self, txt_dir: Path = RAW_TXTS_DIR):
        """
        Initialize TXT loader
        
        :param self: Description
        :param txt_dir: Description
        :type txt_dir: Path
        """
        super().__init__(txt_dir=txt_dir)
        self.txt_dir = Path(txt_dir)
        if not self.txt_dir.exists():
            raise ValueError(f"TXT directory not found: {txt_dir}")
        
    def get_text_files(self) -> List[Path]:
        """
        Get all TXT files from the directory
        
        :return: List of Path objects for TXT files
        :rtype: List[Path]
        """
        txt_files = list(self.txt_dir.glob("*.txt")) # Get all .txt files in the directory
        logger.info(f"Found {len(txt_files)} TXT files in {self.txt_dir}")
        return sorted(txt_files) # it sorts the list of TXT files alphabetically
        
    def get_txt_info(self) -> List[Dict[str, str]]:
        """
        Get information about all TXTs

        :return: List of dictionaries with TXT metadata
        :rtype: List[Dict[str, str]]
        """
        txt_files = self.get_text_files() # Get all TXT files
        txt_info = [] # Initialize list to hold TXT metadata
        for txt_path in txt_files: # Iterate over each TXT file
            info = {
                'filename': txt_path.name, # filename
                'path': str(txt_path), # Full path as string
                'size_kb': txt_path.stat().st_size / 1024 # Size in kilobytes
            }
            txt_info.append(info)
        
        return txt_info
    
    def read_txt_file(self, txt_path: Path) -> str:
        """
        Read the content of a TXT file
        
        :param txt_path: Path to the TXT file
        :type txt_path: Path
        
        :return: Content of the TXT file as a string
        :rtype: str
        """
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read() # Read the entire content of the TXT file
        return content
    
    def get_txt_by_split(self, split: str, labels_df) -> List[Path]:
        """
        Get TXTs for a specific split (train/test) based on labels DataFrame
        
        :param split: 'train' or 'test'
        :type split: str
        :param labels_df: DataFrame containing 'filename' and 'split' columns
        :type labels_df: pd.DataFrame
        
        :return: List of Path objects for the specified split
        :rtype: List[Path]
        """
        split_filenames = labels_df[labels_df['split'] == split]['filename'].tolist() # Get filenames for the split, e.g., 'train' or 'test'
        
        txt_files = [] # Initialize list to hold TXT file paths
        for filename in split_filenames: # Iterate over each filename
            txt_path = self.txt_dir / filename # Construct full path
            
            if txt_path.exists(): # If file exists
                txt_files.append(txt_path) # Add to list
            else:
                logger.warning(f"TXT file not found for {filename}")
        
        logger.info(f"Found {len(txt_files)} TXTs for split '{split}'")
        return txt_files


class PDFLoader():
    """Handles loading PDF files from directory"""

    def __init__(self, pdf_dir: Path = RAW_PDFS_DIR): 
        """
        Initialize PDF loader
        
        Args:
            pdf_dir: Directory containing PDF files
        """
        self.pdf_dir = Path(pdf_dir)
        if not self.pdf_dir.exists():
            raise ValueError(f"PDF directory not found: {pdf_dir}")
    
    def get_pdf_files(self) -> List[Path]:
        """
        Get all PDF files from the directory
        
        Returns:
            List of Path objects for PDF files
        """
        
        pdf_files = list(self.pdf_dir.glob("*.pdf")) # Get all .pdf files in the directory
        logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_dir}")
        return sorted(pdf_files) # it sorts the list of PDF files alphabetically
    
    def get_pdf_info(self) -> List[Dict[str, str]]: # returns list of dictionaries with PDF metadata
        """
        Get information about all PDFs
        
        Returns:
            List of dictionaries with PDF metadata
        """
        pdf_files = self.get_pdf_files() # Get all PDF files
        pdf_info = [] # Initialize list to hold PDF metadata
        
        for pdf_path in pdf_files: # Iterate over each PDF file
            info = {
                'filename': pdf_path.name, # filename
                'path': str(pdf_path), # Full path as string 
                'size_kb': pdf_path.stat().st_size / 1024 # Size in kilobytes
            }
            pdf_info.append(info)
        
        return pdf_info
    
    def get_pdfs_by_split(self, split: str, labels_df) -> List[Path]:
        """
        Get PDFs for a specific split (train/test) based on labels DataFrame
        Checks both data/raw_pdfs and data/useful_pdfs directories
        
        Args:
            split: 'train' or 'test'
            labels_df: DataFrame containing 'filename', 'label' and 'split' columns
            
        Returns:
            List of Path objects for the specified split
        """
        split_filenames = labels_df[labels_df['split'] == split]['filename'].tolist() # Get filenames for the split, e.g., 'train' or 'test'
        
        pdf_files = [] # Initialize list to hold PDF file paths
        for filename in split_filenames: # Iterate over each filename
            pdf_path = self.pdf_dir / filename # Check in raw_pdfs directory

            if pdf_path.exists(): # If file exists in raw_pdfs
                pdf_files.append(pdf_path) # Add to list
            else:
                # Check in useful_pdfs directory
                useful_path = USEFUL_PDFS_DIR / filename  # Use constant
                if useful_path.exists(): # If file exists in useful_pdfs
                    pdf_files.append(useful_path) # Add to list
                else:
                    logger.warning(f"PDF file not found for {filename}")

        logger.info(f"Found {len(pdf_files)} PDFs for split '{split}'")
        return pdf_files
    
    def get_labels_for_pdfs(self, pdf_files: List[Path], labels_df) -> List[int]:
        """
        Get numeric labels (0/1) for the given PDF files based on labels DataFrame

        Args:
            pdf_files: List of Path objects for PDF files
            labels_df: DataFrame containing 'filename' and 'label' columns
        """
        
        label_map = {'not_useful': 0, 'useful': 1} # Map string labels to numeric
        labels = [] # Initialize list to hold labels

        for pdf_path in pdf_files:
            label_row = labels_df[labels_df['filename'] == pdf_path.name] # Find label for the PDF
            if not label_row.empty:
                label_str = label_row.iloc[0]['label'] # Get label string, e.g., 'useful' or 'not_useful'
                # iloc is used to access the first row of the filtered DataFrame
                
                labels.append(label_map[label_str]) # Default to 0 if label not found
            else:
                logger.warning(f"No label found for {pdf_path.name}, defaulting to 0")
                labels.append(0)

        return labels

    def split_train_test(self, test_size: float = 0.5, random_seed: int = 42) -> tuple: # returns train and test splits
        """
        Split PDFs into train and test sets
        
        Args:
            test_size: Proportion of data for testing (default: 0.5 for 50/50 split)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_files, test_files)
        """
        
        import random
        
        pdf_files = self.get_pdf_files() # Get all PDF files
        
        # without seed, different runs would produce different splits
        random.seed(random_seed) # Set random seed for reproducibility
        shuffled = pdf_files.copy() # Create a copy to shuffle
        random.shuffle(shuffled) # Shuffle the list of PDF files
        
        split_idx = int(len(shuffled) * (1 - test_size)) # Calculate split index
        train_files = shuffled[:split_idx] # First part for training
        test_files = shuffled[split_idx:] # Second part for testing
        
        logger.info(f"Split: {len(train_files)} training, {len(test_files)} testing")
        return train_files, test_files


if __name__ == "__main__":
    # Test the loader
    loader = PDFLoader() # Initialize loader with default directory
    pdf_info = loader.get_pdf_info() # Get metadata for all PDFs
    print(f"\nFound {len(pdf_info)} PDFs") # Log number of found PDFs
    
    if pdf_info:
        print("\nFirst 3 PDFs:") # Display first 3 PDFs
        for info in pdf_info[:3]: # Show info for first 3 PDFs
            print(f"  - {info['filename']} ({info['size_kb']:.2f} KB)") # Log filename and size
