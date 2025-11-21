"""
Text Preprocessing Module
Cleans and prepares text for feature extraction
Removes noisy elements like special characters, lowercases text, etc.
"""
import re
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, lowercase: bool = True, remove_special_chars: bool = True): 
        """
        Initialize the text preprocessor.

        Args:
            lowercase: Whether to convert text to lowercase.
            remove_special_chars: Whether to remove special characters.
            specified by bool = True: Whether to apply specified preprocessing steps.
        """
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
    
    def clean_text(self, text: str) -> str:
        if not text: # Check for empty text
            return "" # Handle empty text by returning empty string
        text = re.sub(r'\s+', ' ', text) # Normalize whitespace
        if self.lowercase: # Convert to lowercase if specified
            text = text.lower() # Convert to lowercase
        text = re.sub(r'http\S+|www\S+', '', text) # Remove URLs
        text = re.sub(r'\S+@\S+', '', text) # Remove email addresses 
        if self.remove_special_chars: # Remove special characters if specified
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) # Remove special characters
            # Replaces punctuation, symbols such as #, !, ?, etc. with space
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        logger.info(f"Preprocessing {len(texts)} documents")
        cleaned_texts = [self.clean_text(text) for text in texts]
        logger.info("Preprocessing complete")
        return cleaned_texts
