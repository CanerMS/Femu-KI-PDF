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
                file_type: Literal['pdf', 'txt'] = 'pdf' #  Type hint: Default is 'pdf'
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
        logger.info(f"Main dir: {self.data_dir}")
        logger.info(f"Useful dir: {self.useful_dir}")

    def get_files(self) -> List[Path]:
        """
        Get all files (PDF or TXT) from the main directory

        Returns:
            List of Path objects for files
        """
        files = list(self.data_dir.glob(f"*{self.extension}")) # Get all files with the specified extension
        logger.info(f"Found {len(files)} {self.file_type.upper()} files in {self.data_dir}")
        return sorted(files) # it sorts the list of files alphabetically
    
    # for testing purposes 
    def get_file_info(self) -> List[Dict[str, str]]:
        """Get file information (name, size) - ONLY FOR TESTING"""
        files = self.get_files()
        info = []
        
        for file_path in files:
            info.append({
                'filename': file_path.name,
                'size_kb': file_path.stat().st_size / 1024
            })
        
        return info


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
            file_path = self.data_dir / filename # Construct full path for every file for example raw_pdfs/filename1.pdf

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
        Returns:
            List of numeric labels (0=not_useful, 1=useful)
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
    
        

class PDFLoader(UnifiedLoader): # inherits from UnifiedLoader
    """PDF Loader"""
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
            ) # change file_type to 'pdf'
        
class TXTLoader(UnifiedLoader): # inherits from UnifiedLoader
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
            ) # change file_type to 'txt'



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
    print("\n" + "="*60)
    print("Testing TXT Loader")
    print("="*60)
    txt_loader = TXTLoader()
    txt_info = txt_loader.get_file_info()
    print(f"Found {len(txt_info)} TXTs")
    if txt_info:
        print("\nFirst 3 TXTs:") 
        for info in txt_info[:3]:
            print(f"  - {info['filename']} ({info['size_kb']:.2f} KB)")