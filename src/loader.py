"""# read PDF files and extract text content
PDF Loader Module
Loads PDF files from the raw_pdfs directory
"""
import os
from pathlib import Path
from typing import List, Dict
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from project_config import RAW_PDFS_DIR, USEFUL_PDFS_DIR 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class PDFLoader:
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
        
        pdf_files = []
        for filename in split_filenames:
            pdf_path = self.pdf_dir / filename

            if pdf_path.exists():
                pdf_files.append(pdf_path)
            else:
                # Check in useful_pdfs directory
                useful_path = USEFUL_PDFS_DIR / filename  # Use constant
                if useful_path.exists():
                    pdf_files.append(useful_path)
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
        labels = []

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
