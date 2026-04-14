import pytest
from preprocess import *
from pathlib import Path
import sys

# src recognize
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocess import TextPreprocessor

# test is successfull 
def test_empty_value():

    preprocessor = TextPreprocessor()

    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text("       ") == ""
    assert preprocessor.clean_text(None) == ""

# test is successfull 
def test_remove_urls_and_emails():

    preprocessor = TextPreprocessor()

    text = "Please contact admin@femu.com or visit https://github.com for more information"
    clean_text = preprocessor.clean_text(text)

    assert "admin@femu.com" not in clean_text
    assert "htpps://github.com" not in clean_text
    
    assert "please contact or visit for more information" in clean_text