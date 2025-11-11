"""
PDF Text Extractor Module
Extracts text content from PDF files
"""
from pathlib import Path
from typing import Dict, List
import logging


# pdf plumber is preferred for text, tables and metadata from PDFs extraction due to better handling of complex PDFs

try:
    import pdfplumber # Check if pdfplumber is installed
    USE_PDFPLUMBER = True
except ImportError: # Fallback to PyPDF2 if pdfplumber is not available
    USE_PDFPLUMBER = False
    try:
        # PdfReader is a part of PyPDF2, which is a common library for PDF handling
        from PyPDF2 import PdfReader 
    except ImportError:
        raise ImportError("Please install either pdfplumber or PyPDF2: pip install pdfplumber")

logging.basicConfig(level=logging.INFO) # Configure logging
logger = logging.getLogger(__name__) # Create logger for this module


class PDFExtractor:
    """Extracts text from PDF files"""

    # self is the instance of the class
    
    def __init__(self, output_dir: str = "data/extracted_texts"): # initializer constructor
        """
        Initialize extractor
        
        Args:
            output_dir: Directory to save extracted text files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a single PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try: # because some PDFs may be corrupted or unreadable, PDFPlumber is preferred, better handling complex PDFs
            if USE_PDFPLUMBER:
                return self._extract_with_pdfplumber(pdf_path)
            else:
                return self._extract_with_pypdf2(pdf_path) # Fallback to PyPDF2
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path.name}: {str(e)}")
            return ""
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file using pdfplumber.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text as a string.
        """
        text = [] # List to hold text from each page
        with pdfplumber.open(pdf_path) as pdf: # Open PDF with pdfplumber
            for page in pdf.pages: # Iterate through pages 
                page_text = page.extract_text() # Extract text from page
                if page_text: # Check if text is not None
                    text.append(page_text) # Append page text to list
        return "\n".join(text) # Join all page texts into single string
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str: 
        """Extract using PyPDF2"""
        text = []
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    
    def extract_and_save(self, pdf_path: Path) -> Path:
        """
        Extract text and save to file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Path to saved text file
        """
        text = self.extract_text_from_pdf(pdf_path)
        
        # Save to text file
        txt_filename = pdf_path.stem + ".txt" 
        txt_path = self.output_dir / txt_filename # Path to save text file 
        
        txt_path.write_text(text, encoding='utf-8')
        logger.info(f"Extracted {len(text)} characters from {pdf_path.name}")
        
        return txt_path
    
    def extract_batch(self, pdf_files: List[Path]) -> Dict[str, str]:
        """
        Extract text from multiple PDFs
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            Dictionary mapping filenames to extracted text
        """
        results = {}
        
        for pdf_path in pdf_files:
            text = self.extract_text_from_pdf(pdf_path)
            results[pdf_path.name] = text
            
            # Also save to file
            self.extract_and_save(pdf_path)
        
        logger.info(f"Extracted text from {len(results)} PDFs")
        return results


if __name__ == "__main__":
    # Test the extractor
    from loader import PDFLoader
    
    loader = PDFLoader()
    pdf_files = loader.get_pdf_files()[:3]  # Test with first 3 PDFs
    
    extractor = PDFExtractor()
    results = extractor.extract_batch(pdf_files)
    
    for filename, text in results.items():
        print(f"\n{filename}: {len(text)} characters")
        print(f"Preview: {text[:200]}...")
