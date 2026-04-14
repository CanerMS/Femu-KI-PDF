import pytest
from pathlib import Path
import sys

# adding src. so that the moduls can be saved to there
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loader import UnifiedLoader, PDFLoader, TXTLoader

# test successfull
def test_unified_loader_initialization(tmp_path):
    """Test if the loader initialized with accurate parameters"""

    # Temporary path
    dummy_data = tmp_path / "dummy_data"
    dummy_useful = tmp_path / "dummy_useful"
    
    # Create the directories manually
    dummy_data.mkdir()
    dummy_useful.mkdir()

    loader = UnifiedLoader(data_dir=dummy_data, useful_dir=dummy_useful, file_type='txt')

    assert loader.file_type == 'txt'
    assert loader.data_dir == dummy_data
    assert loader.useful_dir == dummy_useful

# test successfull
def test_pdf_loader_default_type():
    """Controls if the PDFLoader set to type pdf as default parameter"""

    loader = PDFLoader()

    assert isinstance(loader, UnifiedLoader)

# test successfull
def test_text_loader_default_type():
    """Controls if the TXTLoader set the type txt as default parameter"""

    loader = TXTLoader()

    assert isinstance(loader, UnifiedLoader)