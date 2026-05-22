import pytest
from src.extractor import PDFExtractor
from pathlib import Path
import sys

def test_pdfextractor_init(tmp_path):
    
    dummy_dir = tmp_path / "fake_output"
    
    extractor = PDFExtractor(dummy_dir)

    assert dummy_dir == extractor.output_dir # "." to look inside an object
    assert dummy_dir.exists()

# What happens if it finds a cache file
def test_extract_text_from_cache(tmp_path):
    dummy_dir = tmp_path / "fake_output"
    
    extractor = PDFExtractor(dummy_dir)

    # define the file inside the folder
    cache_file = dummy_dir / "my_document.txt"

    # write text into the file
    cache_file.write_text("I am going to work today.")

    fake_pdf_path = Path("my_document.pdf")
    result = extractor.extract_text_from_pdf()

