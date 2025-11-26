"""
PDF Text Extractor Module
Extracts text content from PDF files
"""
from pathlib import Path # Import Path for file path manipulations
from typing import Dict, List # Import Dict and List for type hinting
import logging # Import logging for logging messages
from project_config import EXTRACTED_TEXTS_DIR # Import configuration for extracted texts directory


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
    """Extracts text from PDF files into plain text format"""

    # self is the instance of the class
    
    def __init__(self, output_dir: Path = EXTRACTED_TEXTS_DIR): # initializer constructor
        """
        Initialize extractor
        
        Args:
            output_dir: Directory to save extracted text files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized PDFExtractor with output directory: {self.output_dir}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a single PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        # Check if cached text file exists

        txt_file = self.output_dir / (f"{pdf_path.stem}.txt")

        if txt_file.exists():
            try:
                cached_text = txt_file.read_text(encoding='utf-8')
                if cached_text.strip():
                    logger.info(f"Loaded from cache: {pdf_path.name} ({len(cached_text)} characters)")
                    return cached_text
                else:
                    logger.warning(f"Cache file is empty, re-extracting: {pdf_path.name}")
            except Exception as e:
                logger.warning(f"Failed to read cache file, re-extracting: {pdf_path.name}. Error: {e}")    
            
        # Extract if not cached
        logger.info(f"Extracting text from: {pdf_path.name}")
        try:
            if USE_PDFPLUMBER: # Use pdfplumber if available
                text = self._extract_with_pdfplumber(pdf_path) # Use pdfplumber for extraction
            else: # Fallback to PyPDF2, because pdfplumber is not available
                text = self._extract_with_pypdf2(pdf_path) # Use PyPDF2 for extraction
        except Exception as e: # Handle extraction errors
            logger.error(f"Error extracting {pdf_path.name}: {str(e)}") # Log error message
            return "" # Return empty string on error
        
        # Save extracted text to cache
        if text.strip():  # Only save if text is not empty
            txt_file.write_text(text, encoding='utf-8')
            logger.info(f"Saved extracted text to cache: {txt_file.name} ({len(text)} characters)")
        
        return text

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
        text = [] # List to hold text from each page
        reader = PdfReader(str(pdf_path)) # Open PDF with PyPDF2
        for page in reader.pages: # Iterate through pages
            page_text = page.extract_text() # Extract text from page
            if page_text: # Check if text is not None
                text.append(page_text) # Append page text to list
        return "\n".join(text) # Join all page texts into single string
    
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
        
        txt_path.write_text(text, encoding='utf-8') # Write extracted text to file, utf-8 means support for all characters 
        logger.info(f"Extracted {len(text)} characters from {pdf_path.name}") 
        
        return txt_path # Return path to saved text file
    
    def extract_batch(self, pdf_files: List[Path]) -> Dict[str, str]: # returns dictionary mapping filenames to extracted text
        """
        Extract text from multiple PDFs
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            Dictionary mapping filenames to extracted text
        """
        results = {}
        total = len(pdf_files)
        cached_count = 0
        extracted_count = 0

        logger.info(f"Processing batch of {total} PDFs for text extraction.")
        
        for i, pdf_path in enumerate(pdf_files, 1): # Iterate over each PDF file
            txt_file = self.output_dir / f"{pdf_path.stem}.txt"
            was_cached = txt_file.exists() and txt_file.stat().st_size > 0
            
            # Extract (will use cache if available and valid)
            text = self.extract_text_from_pdf(pdf_path) # Extract text
            results[pdf_path.name] = text # Store in results dictionary
            
            # Track caching stats
            if was_cached:
                cached_count += 1
            else:
                extracted_count += 1

            # progress logging
            if i % 25 == 0 or i == total:
                logger.info(f"Processed {i}/{total} PDFs. Cached: {cached_count}, Newly Extracted: {extracted_count}")

        # final summary
        logger.info(f"Batch complete! Total PDFs: {total}, Cached: {cached_count}, Newly Extracted: {extracted_count}")
        
       
        return results
    
    def clear_cache(self):
        """
        Clear all cached extracted text files
        """
        if self.output_dir.exists():
            count = 0
            for txt_file in self.output_dir.glob("*.txt"):
                txt_file.unlink()
                count += 1
            logger.info(f"Cleared {count} cached text files from {self.output_dir}")

    def get_cache_stats(self) -> Dict:
        """
        get statistics about cached extracted text files
        """
        if not self.output_dir.exists():
            return {'cached_files': 0, 'total_size_kb': 0}
        
        txt_files = list(self.output_dir.glob("*.txt"))
        total_size = sum(f.stat().st_size for f in txt_files)

        return {
            'cached_files': len(txt_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(self.output_dir)
        }


if __name__ == "__main__":
    # Test the extractor
    from loader import PDFLoader
    
    # Show current cache stats
    extractor = PDFExtractor()
    stats = extractor.get_cache_stats()
    print(f"Current Cache Stats: ")
    print(f"  Cached Files: {stats['cached_files']}")
    print(f"  Total Size (MB): {stats['total_size_mb']}")
    print(f"  Cache Directory: {stats['cache_dir']}")

    # Test extraction on sample PDFs
    loader = PDFLoader()
    pdf_files = loader.get_pdf_files()[:3] # Get first 3 PDFs for testing

    print("\nExtracting text from sample PDFs...")
    results = extractor.extract_batch(pdf_files)
    
    print("\nExtraction Results:")
    for filename, text in results.items():
        print(f"\n{filename}: {len(text)} characters")
        print(f"Preview: {text[:200]}...")