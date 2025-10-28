"""
Text Preprocessing Module
Cleans and prepares text for feature extraction
"""
import re
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, lowercase: bool = True, remove_special_chars: bool = True):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        if self.lowercase:
            text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        logger.info(f"Preprocessing {len(texts)} documents")
        return [self.clean_text(text) for text in texts]
