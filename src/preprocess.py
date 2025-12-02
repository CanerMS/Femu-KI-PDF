"""
Text Preprocessing Module
Cleans and prepares text for feature extraction and machine learning
Removes noisy elements like special characters, lowercases text, etc.
"""
import re # regular expressions for text cleaning
from typing import List # type hinting
import logging # logging for info messages

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

        """
        Clean and preprocess text by:
        1. Normalizing whitespace
        2. Converting to lowercase
        3. Removing author sections
        4. Removing noise words
        5. Removing figure references
        6. Removing URLs and emails
        7. Removing standalone numbers
        8. Removing special characters
        Args:
        text: Raw text to clean
    
        Returns:
        Cleaned text string
        """
        
        if not text: # Check for empty text
            return "" # Handle empty text by returning empty string
        
        text = re.sub(r'\s+', ' ', text) # Normalize whitespace

        if self.lowercase: # Convert to lowercase if specified
            text = text.lower() # Convert to lowercase
        
        
        if 'author' in text and 'contribution' in text:
            parts = re.split(r'\bauthor[s]?\s+contribution[s]?\b', text, flags=re.IGNORECASE)
            if len(parts) > 1:
                text = parts[0]  # Keep text before 'Author contribution' section

        noise_words = [
        'education', 'studying', 'diploma', 'degree',
        'university', 'institute', 'college', 'school',
        'received', 'obtained', 'graduated', 'phd', 'bachelor', 'master'
        ]
    
        for word in noise_words:
            text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)

        text = re.sub(r'\b\d+[a-z]\b', '', text, flags=re.IGNORECASE)  # 2b, 3a, 5g
        text = re.sub(r'\bfig\s*\d+\b', '', text, flags=re.IGNORECASE)  # fig 1, fig2
        text = re.sub(r'\btable\s*\d+\b', '', text, flags=re.IGNORECASE)  # table 1
    
        text = re.sub(r'\[\d+\]', '', text)  # [1], [2]
        text = re.sub(r'http[s]?://\S+', '', text)  # URLs
        text = re.sub(r'\S+@\S+', '', text)  # Emails
    
        text = re.sub(r'\b\d+\b(?!\.\d)', '', text) # Remove standalone numbers (not part of decimals)

        if self.remove_special_chars: # Remove special characters if specified
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) # Keep only alphanumeric and spaces

        text = re.sub(r'\s+', ' ', text).strip() # Final whitespace normalization
        return text
    
    # More features can be added as needed for text preprocessing
    
    def preprocess_batch(self, texts: List[str], filenames: List[str] = None) -> List[str]: # preprocess a list of texts
        '''
        Preprocess a batch of texts.
        Args:
            texts: List of text documents to preprocess.
            filenames: Optional list of filenames corresponding to the texts.
        Returns:
            List of preprocessed text documents.
        '''
        
        
        logger.info(f"Preprocessing {len(texts)} documents") # Log number of documents to preprocess

        cleaned_texts = [] # List to hold cleaned texts
        total = len(texts) # Total number of texts

        for i, text in enumerate(texts, 1): # Iterate over each text
            # Colourful Progress Bar
            progress = (i / total) 
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)

            if filenames and i <= len(filenames):
                logger.info(f"[{bar}] {i}/{total} ({progress*100:.1f}%) | {filenames[i-1]}")
            else:
                logger.info(f"[{bar}] {i}/{total} ({progress*100:.1f}%)")

            cleaned = self.clean_text(text) # Clean the text
            cleaned_texts.append(cleaned) # Append cleaned text to list

            reduction = (1 - len(cleaned) / len(text)) * 100 if len(text) > 0 else 0 # Calculate reduction in size
            logger.info(f"  Original length: {len(text)} chars, Cleaned length: {len(cleaned)} chars, Reduction: {reduction:.1f}%") # Log lengths

        logger.info("Preprocessing complete") # Log completion of preprocessing
        return cleaned_texts # Return list of cleaned texts
