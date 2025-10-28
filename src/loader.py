"""# read PDF files and extract text content
PDF Loader Module
Loads PDF files from the raw_pdfs directory
"""
import os
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLoader:
    """Handles loading PDF files from directory"""
    
    def __init__(self, pdf_dir: str = "data/raw_pdfs"):
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
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_dir}")
        return sorted(pdf_files)
    
    def get_pdf_info(self) -> List[Dict[str, str]]:
        """
        Get information about all PDFs
        
        Returns:
            List of dictionaries with PDF metadata
        """
        pdf_files = self.get_pdf_files()
        pdf_info = []
        
        for pdf_path in pdf_files:
            info = {
                'filename': pdf_path.name,
                'path': str(pdf_path),
                'size_kb': pdf_path.stat().st_size / 1024
            }
            pdf_info.append(info)
        
        return pdf_info
    
    def split_train_test(self, test_size: float = 0.5, random_seed: int = 42) -> tuple:
        """
        Split PDFs into train and test sets
        
        Args:
            test_size: Proportion of data for testing (default: 0.5 for 50/50 split)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_files, test_files)
        """
        import random
        
        pdf_files = self.get_pdf_files()
        random.seed(random_seed)
        shuffled = pdf_files.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - test_size))
        train_files = shuffled[:split_idx]
        test_files = shuffled[split_idx:]
        
        logger.info(f"Split: {len(train_files)} training, {len(test_files)} testing")
        return train_files, test_files


if __name__ == "__main__":
    # Test the loader
    loader = PDFLoader()
    pdf_info = loader.get_pdf_info()
    print(f"\nFound {len(pdf_info)} PDFs")
    
    if pdf_info:
        print("\nFirst 3 PDFs:")
        for info in pdf_info[:3]:
            print(f"  - {info['filename']} ({info['size_kb']:.2f} KB)")
