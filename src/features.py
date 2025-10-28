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
    def __init__(self, max_features: int = 1000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2
        )
        self.is_fitted = False
    
    def fit_transform(self, texts: list):
        logger.info("Fitting TF-IDF vectorizer and transforming texts")
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        logger.info(f"Created feature matrix: {features.shape}")
        return features
    
    def transform(self, texts: list):
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet")
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        if self.is_fitted:
            return self.vectorizer.get_feature_names_out()
        return []
