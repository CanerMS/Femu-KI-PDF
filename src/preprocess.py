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
        if not text: # Check for empty text
            return "" # Handle empty text by returning empty string
        text = re.sub(r'\s+', ' ', text) # Normalize whitespace
        if self.lowercase: # Convert to lowercase if specified
            text = text.lower() # Convert to lowercase
         # 2. Academic references and citations
        text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)  # [1], [2-5]
        text = re.sub(r'\b(?:figure|fig|table|tab)\.?\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:section|sec|chapter|ch)\.?\s*\d+(?:\.\d+)*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:equation|eq)\.?\s*\(?\d+\)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]', '', text)  # Dipnotlar

        # Authors et al. (2020), Smith and Jones (2019)
        text = re.sub(r'\bAuthor[s]?\s+contribution[s]?:.*', '', text, flags=re.IGNORECASE | re.DOTALL)  # Author contributions bölümü
        text = re.sub(r'[^\n]*(?:received|studying|obtained)\s+(?:his|her|their)\s+(?:education|degree|bachelor|master|phd|diploma)[^\n]*', '', text, flags=re.IGNORECASE)  # Education bilgileri
        text = re.sub(r'[^\n]*(?:research|his|her|their)\s+(?:focuses|interests|is|include)[^\n]*', '', text, flags=re.IGNORECASE)  # Research interests
        text = re.sub(r'\b(?:University|Institute|Hospital|Department|College|School|Laboratory)[^\n]*\d{5,}[^\n]*', '', text, flags=re.IGNORECASE)  # Affiliations + postal codes
    
        # 3. Header/Footer
        text = re.sub(r'\bRunning\s+head:.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'all\s+rights\s+reserved', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\|\s*Vol\.?\s*\d+', '', text)

        text = re.sub(r'\b(?:University|Institute|Hospital|Department|College|School|Laboratory)[^\n]*\d{5,}[^\n]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE) # Page numbers in header/footer
        text = re.sub(r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', '', text)
        text = re.sub(r'\b\d{4}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b', '', text)

    
        # 4. Bibliographic identifiers
        # URLs, DOIs, ISBNs, ISSNs, PMIDs, arXiv
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'https://doi.org/\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'\bdoi:\s*\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bisbn[-:\s]*[\d-]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bissn[-:\s]*[\d-]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpmid:\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\barxiv:\s*\d+\.\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:Received|Accepted|Published|Revised):\s*\d+[^\n]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\n]*Corresponding\s+author[s]?[^\n]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\n]*co-first\s+author[s]?[^\n]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAuthor[s]?:\s*[^\n]+', '', text, flags=re.IGNORECASE)


    
        # 5. Dates and numbers
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
        text = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '', text)
        text = re.sub(r'\d+', '', text)  # Remove all numbers
    
        # 6. Contact information
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\Author[s]?:\s*\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpage\s*\d+\b', '', text, flags=re.IGNORECASE)
    
        # 7. Special characters and symbols
        text = re.sub(r'©|\(c\)|®|™', '', text)
        text = re.sub(r'[$€£¥₹]\s*\d+(?:\.\d+)?', '', text)
        text = re.sub(r'\d+\.?\d*\s*%', '', text)
        text = re.sub(r'[≤≥±×÷√∞≈≠∑∫∂∇]', ' ', text)
        text = re.sub(r'[→←⇒⇐↑↓⇔]', ' ', text)
    
        # 8. Repeated characters
        text = re.sub(r'\.{3,}', ' ', text)
        text = re.sub(r'-{3,}', ' ', text)
        text = re.sub(r'_{3,}', ' ', text)
    
        # 9. PDF artifacts
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # electro-\nnic → electronic
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff')
        text = text.replace('ﬃ', 'ffi')
        text = text.replace('ﬄ', 'ffl')
    



        if self.remove_special_chars: # Remove special characters if specified
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) # Remove special characters
            # Replaces punctuation, symbols such as #, !, ?, etc. with space
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace and trim
        return text
    
    # More features can be added as needed for text preprocessing
    
    def preprocess_batch(self, texts: List[str]) -> List[str]: # preprocess a list of texts
        logger.info(f"Preprocessing {len(texts)} documents") # Log number of documents to preprocess
        cleaned_texts = [self.clean_text(text) for text in texts] # Clean each text in the list
        logger.info("Preprocessing complete") # Log completion of preprocessing
        return cleaned_texts # Return list of cleaned texts
