"""
Feature Extraction Module
Creates TF-IDF features from text
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    
    """
    Handles TF-IDF feature extraction from text
    Uses sklearn's TfidfVectorizer
    """
    
    def __init__(self, max_features: int = 2000, ngram_range: tuple = (1, 2)): # tuple 1,2 means unigrams and bigrams
        self.vectorizer = TfidfVectorizer( # Sklearn TF-IDF tool for converting text to numerical features 
            max_features=max_features, # Keep only top N features by TF-IDF score
            ngram_range=ngram_range, # Consider unigrams and bigrams as specified
            min_df=3, # Ignore terms that appear in only one document
            stop_words='english' # Remove common English stop words (like "the", "is", etc.
        )

        """
        Indicates if the vectorizer has been fitted
        keep only the top 1000 features (words) by TF-IDF score
        considers via ngram_range unigrams ("invoice", "payment") and bigrams ("invoice payment")
        ignores terms that appear in only one document (min_df=2)
        tf means, how frequently a word appears in a document
        idf means, how important a word is across all documents and rares words get higher weights       
         """
        # if a word either appears very often and is common (like "the", "is", "and") 
        # or appears very rarely (like a typo), it gets lower weight

        self.is_fitted = False

    
    
    def fit_transform(self, texts: list): # provides the ability to train and learn vocabulary
        """
        Fit the vectorizer on training texts and transform them to feature matrix
        Args:
            texts: List of text documents to fit and transform
        Returns:
            TF-IDF feature matrix (sparse)
        """

        logger.info("Fitting TF-IDF vectorizer and transforming texts")
        features = self.vectorizer.fit_transform(texts) # Learning happens here, Learn vocabulary and transform texts to feature matrix
        self.is_fitted = True # Mark as fitted after learning vocabulary from training texts
        logger.info(f"Created feature matrix: {features.shape}") 
        return features
    
    def transform(self, texts: list):
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet")
        return self.vectorizer.transform(texts) # Use existing vocabulary to transform new texts, doesn't learn new words
    
    def get_feature_names(self):
        """
        Get the feature names (words) extracted by the vectorizer
        """
      
        if self.is_fitted:
            return self.vectorizer.get_feature_names_out()
        return []
