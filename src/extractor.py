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

        txt_file = self.output_dir / (f"{pdf_path.stem}.txt") # Corresponding text file path

        if txt_file.exists(): # Load from cache if available
            try:
                cached_text = txt_file.read_text(encoding='utf-8') # Read cached text
                if cached_text.strip(): # Check if cached text is not empty
                    logger.info(f"Loaded from cache: {pdf_path.name} ({len(cached_text)} characters)") # Log cache load
                    return cached_text # Return cached text
                else:
                    logger.warning(f"Cache file is empty, re-extracting: {pdf_path.name}")
            except Exception as e:
                logger.warning(f"Failed to read cache file, re-extracting: {pdf_path.name}. Error: {e}")    
            
        # Extract if not cached
        logger.info(f"Extracting text from: {pdf_path.name}")
        try:
            if USE_PDFPLUMBER: # Use pdfplumber if available, better extraction quality
                text = self._extract_with_pdfplumber(pdf_path) # Use pdfplumber for extraction
            else: # Fallback to PyPDF2, because pdfplumber is not available
                text = self._extract_with_pypdf2(pdf_path) # Use PyPDF2 for extraction
        except Exception as e: # Handle extraction errors
            logger.error(f"Error extracting {pdf_path.name}: {str(e)}") # Log error message
            return "" # Return empty string on error
        
        # Save extracted text to cache
        if text.strip():  # Only save if text is not empty
            txt_file.write_text(text, encoding='utf-8') # Write extracted text to file
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
        results = {} # Initialize results dictionary
        total = len(pdf_files) # Total number of PDFs to process
        cached_count = 0 # Count of cached files used
        extracted_count = 0 # Count of newly extracted files

        logger.info(f"Processing batch of {total} PDFs for text extraction.")
        
        for i, pdf_path in enumerate(pdf_files, 1): # Iterate over each PDF file
            txt_file = self.output_dir / f"{pdf_path.stem}.txt" # Corresponding text file path
            was_cached = txt_file.exists() and txt_file.stat().st_size > 0 # Check if cached file exists and is non-empty, stat().st_size gets file size in bytes
            
            # Extract (will use cache if available and valid)
            text = self.extract_text_from_pdf(pdf_path) # Extract text
            results[pdf_path.stem] = text # Store in results dictionary
            
            # Track caching stats
            if was_cached: # If text was loaded from cache
                cached_count += 1 # Increment cached count
            else:
                extracted_count += 1 # Increment extracted count

            # progress logging
            if i % 25 == 0 or i == total: # Log progress every 25 PDFs or at the end
                logger.info(f"Processed {i}/{total} PDFs. Cached: {cached_count}, Newly Extracted: {extracted_count}") # Log progress

        # final summary
        logger.info(f"Batch complete! Total PDFs: {total}, Cached: {cached_count}, Newly Extracted: {extracted_count}")
        
       
        return results # Return results dictionary
    
    def clear_cache(self):
        """
        Clear all cached extracted text files
        """
        if self.output_dir.exists(): # Check if output directory exists
            count = 0
            for txt_file in self.output_dir.glob("*.txt"): # Iterate over all text files in output directory
                txt_file.unlink() # Delete the text file
                count += 1 # Increment count of deleted files
            logger.info(f"Cleared {count} cached text files from {self.output_dir}") # Log number of files cleared

    def get_cache_stats(self) -> Dict: # returns dictionary with cache statistics
        """
        get statistics about cached extracted text files
        """
        if not self.output_dir.exists(): # Check if output directory exists
            return {'cached_files': 0, 'total_size_kb': 0} # Return zero stats if directory doesn't exist
        
        txt_files = list(self.output_dir.glob("*.txt")) # Get list of all text files in output directory
        total_size = sum(f.stat().st_size for f in txt_files) # Calculate total size of all text files in bytes

        return {
            'cached_files': len(txt_files), # Number of cached text files
            'total_size_mb': round(total_size / (1024 * 1024), 2), # Total size in megabytes
            'cache_dir': str(self.output_dir) # Cache directory path
        }

class TXTExtractor:
    """
    Extracts text from TXT files 
    Simple reader, no complex parsing needed
    """
    
    def __init__(self):
        """Initialize TXT extractor"""
        logger.info("Initialized TXTExtractor")
    
    def extract_text(self, txt_path: Path) -> str:
        """Read TXT file directly"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"Read {len(text)} characters from {txt_path.name}")
            return text
            
        except UnicodeDecodeError:
            # Fallback encoding
            logger.warning(f"UTF-8 failed for {txt_path.name}, trying latin-1")
            try:
                with open(txt_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                return text
            except Exception as e:
                logger.error(f"Error reading {txt_path.name}: {str(e)}")
                return ""
        
        except Exception as e:
            logger.error(f"Error reading {txt_path.name}: {str(e)}")
            return ""
    
    def extract_batch(self, txt_files: List[Path]) -> Dict[str, str]:
        """Read multiple TXT files"""
        results = {}
        total = len(txt_files)

        for i, txt_path in enumerate(txt_files, 1):
            text = self.extract_text(txt_path)
            if text:
                results[txt_path.stem] = text
        
            if i % 25 == 0 or i == total:
                logger.info(f"Processed {i}/{total} TXT files.")
        
        return results
    
class UnifiedExtractor:
    """
    Unified extractor that can handle both PDF and TXT files
    Automatically detects file type based on extension
    """
    
    def __init__(self, output_dir: Path = EXTRACTED_TEXTS_DIR):
        """
        Initialize unified extractor
        
        Args:
            output_dir: Directory for PDF extraction cache
        """
        self.pdf_extractor = PDFExtractor(output_dir=output_dir)
        self.txt_extractor = TXTExtractor()
        logger.info("Initialized UnifiedExtractor (supports PDF and TXT)")
    
    def extract_text(self, file_path: Path) -> str:
        """
        Extract text from file (auto-detect PDF or TXT)
        
        Args:
            file_path: Path to PDF or TXT file
            
        Returns:
            Extracted text as string
        """
        if file_path.suffix.lower() == '.pdf':
            return self.pdf_extractor.extract_text(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self.txt_extractor.extract_text(file_path)
        else:
            logger.error(f"Unsupported file type: {file_path.suffix}")
            return ""
    
    def extract_batch(self, files: List[Path]) -> Dict[str, str]:
        """
        Extract text from multiple files (mixed PDF and TXT)
        
        Args:
            files: List of file paths (PDF and/or TXT)
            
        Returns:
            Dictionary mapping file stems to extracted text
        """
        # Separate by file type
        pdf_files = [f for f in files if f.suffix.lower() == '.pdf']
        txt_files = [f for f in files if f.suffix.lower() == '.txt']
        
        results = {}
        
        # Extract PDFs
        if pdf_files:
            logger.info(f"Extracting {len(pdf_files)} PDF files...")
            pdf_results = self.pdf_extractor.extract_batch(pdf_files)
            results.update(pdf_results)
        
        # Extract TXTs
        if txt_files:
            logger.info(f"Reading {len(txt_files)} TXT files...")
            txt_results = self.txt_extractor.extract_batch(txt_files)
            results.update(txt_results)
        
        logger.info(f"Total extracted: {len(results)} files")
        return results

if __name__ == "__main__":
    # Test the extractors
    print("="*60)
    print("Testing PDF and TXT Extractors")
    print("="*60)
    
    # Test PDFExtractor
    print("\n[1] Testing PDFExtractor...")
    from loader import PDFLoader
    
    pdf_extractor = PDFExtractor()
    stats = pdf_extractor.get_cache_stats()
    print(f"PDF Cache Stats:")
    print(f"  Cached Files: {stats['cached_files']}")
    print(f"  Total Size (MB): {stats['total_size_mb']}")
    
    pdf_loader = PDFLoader()
    pdf_files = pdf_loader.get_files()[:2]
    
    if pdf_files:
        print(f"\nExtracting {len(pdf_files)} sample PDFs...")
        pdf_results = pdf_extractor.extract_batch(pdf_files)
        for filename, text in pdf_results.items():
            print(f"  {filename}: {len(text)} characters")
    
    # Test TXTExtractor
    print("\n[2] Testing TXTExtractor...")
    from loader import TXTLoader
    
    txt_extractor = TXTExtractor()
    txt_loader = TXTLoader()
    txt_files = txt_loader.get_files()[:2]
    
    if txt_files:
        print(f"\nReading {len(txt_files)} sample TXTs...")
        txt_results = txt_extractor.extract_batch(txt_files)
        for filename, text in txt_results.items():
            print(f"  {filename}: {len(text)} characters")
    else:
        print("  No TXT files found in data/raw_txts/")
    
    # Test UnifiedExtractor
    print("\n[3] Testing UnifiedExtractor...")
    unified_extractor = UnifiedExtractor()
    
    all_files = pdf_files + txt_files
    if all_files:
        print(f"\nExtracting {len(all_files)} mixed files (PDF + TXT)...")
        unified_results = unified_extractor.extract_batch(all_files)
        for filename, text in unified_results.items():
            print(f"  {filename}: {len(text)} characters")