"""
Feature Extraction Module
Creates TF-IDF features from text
"""
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from project_config import MAX_FEATURES, NGRAM_RANGE, CUSTOM_STOP_WORDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    
    """
    Handles TF-IDF feature extraction from text
    Uses sklearn's TfidfVectorizer
    """
    
    def __init__(self, max_features: int = MAX_FEATURES, ngram_range: tuple = NGRAM_RANGE): # tuple 1,2 means unigrams and bigrams
        # Combine English stop words with custom
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        all_stop_words = list(ENGLISH_STOP_WORDS) + CUSTOM_STOP_WORDS
        
        logger.info(f"Initializing TfidfVectorizer with {len(all_stop_words)} stop words")
        logger.info(f"Custom stop words: {len(CUSTOM_STOP_WORDS)} words")

        self.vectorizer = TfidfVectorizer( # Sklearn TF-IDF tool for converting text to numerical features 
            max_features=max_features, # Keep only top N features by TF-IDF score
            ngram_range=ngram_range, # Consider unigrams and bigrams as specified
            min_df=5, # Ignore terms that appear in only one document
            stop_words=all_stop_words, # Remove common English stop words (like "the", "is", etc.
            token_pattern=r'\b[a-z]{2,}\b' # Tokens must be at least 2 letters long (ignore single letters and numbers
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
        self.feature_names = None

    
    # here it learns and returns the feature matrix by scanning the texts
    
    def fit_transform(self, texts: list): # provides the ability to train and learn vocabulary
        """
        Fit the vectorizer on training texts and transform them to feature matrix
        Scans the texts to learn the vocabulary and idf weights
        Then transforms texts into TF-IDF feature matrix in numerical format
        Args:
            texts: List of text documents to fit and transform
        Returns:
            TF-IDF feature matrix (sparse)
        """

        logger.info("Fitting TF-IDF vectorizer and transforming texts")
        features = self.vectorizer.fit_transform(texts) # Learning happens here, Learn vocabulary and transform texts to feature matrix
        self.is_fitted = True # Mark as fitted after learning vocabulary from training texts
        self.feature_names = self.vectorizer.get_feature_names_out() # Save feature names after fitting
        self._save_feature_names()
        logger.info(f"Created feature matrix: {features.shape}") 
        return features
    
    # here it only returns the feature matrix by using existing vocabulary
    def _save_feature_names(self):
        """
        Save the feature names (words) extracted by the vectorizer
        """
        if self.feature_names is None:
            logger.warning("Feature names not available. Vectorizer may not be fitted yet.")
            return
        
        features_df = pd.DataFrame({
            'feature_index': range(len(self.feature_names)),
            'feature_name': self.feature_names
        })



        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'selected_features.csv'

        features_df.to_csv(output_path, index = False)
        logger.info(f"{len(self.feature_names)} Saved feature names to {output_path}")

        logger.info("\nFirst 50 selected features: ")
        for i, name in enumerate(self.feature_names[:50]):
            logger.info(f"  {i}: {name}")

    def get_top_features(self, X, y, top_n=50):
        """
        Get top N features based on average TF-IDF scores for positive class
        
        Args:
            X: TF-IDF feature matrix
            y: Labels array
            top_n: Number of top features to return
            
        Returns:
            List of top N feature names
        """
        logger.info(f"Calculating {top_n} top features based on TF-IDF scores")

        if self.feature_names is None:
            logger.warning("Vectorizer not fitted yet. Cannot get feature names.")
            return
        
        # Average TF-IDF scores for class
        X_Dense = X.toarray() # Convert sparse matrix to dense for easier manipulation
        useful_mask = (y == 1) # Mask for useful class
        not_useful_mask = (y == 0) # Mask for not useful class

        useful_mean = X_Dense[useful_mask].mean(axis=0) # Mean TF-IDF for useful
        not_useful_mean = X_Dense[not_useful_mask].mean(axis=0) # Mean TF-IDF for not useful
        diff = useful_mean - not_useful_mean # Difference in means

        # Top Features
        top_useful_indices = np.argsort(diff)[-top_n:][::-1] # Indices of top N useful features
        top_not_useful_indices = np.argsort(diff)[:top_n] # Indices of top N not useful features

        # Create DataFrames for better visualization
        useful_features = pd.DataFrame({
            'feature_index': top_useful_indices,
            'feature_name': [self.feature_names[i] for i in top_useful_indices], 
            'useful_mean_tfidf': [useful_mean[i] for i in top_useful_indices], # mean TF-IDF in useful class
            'not_useful_mean_tfidf': [not_useful_mean[i] for i in top_useful_indices], # mean TF-IDF in not useful class
            'difference': [diff[i] for i in top_useful_indices]
        })

        not_useful_features = pd.DataFrame({
            'feature_index': top_not_useful_indices,
            'feature_name': [self.feature_names[i] for i in top_not_useful_indices],
            'useful_mean_tfidf': [useful_mean[i] for i in top_not_useful_indices],
            'not_useful_mean_tfidf': [not_useful_mean[i] for i in top_not_useful_indices],
            'difference': [diff[i] for i in top_not_useful_indices]
        })

        output_dir = Path('results')
        useful_features.to_csv(output_dir / 'top_useful_features.csv', index=False)
        not_useful_features.to_csv(output_dir / 'top_not_useful_features.csv', index=False)

        logger.info("\n" + "="*60)
        logger.info(f"Top {top_n} Features for 'useful' class:")
        logger.info("="*60)
        for idx, row in useful_features.head(20).iterrows():
            logger.info(f"{idx+1:2d}. {row['feature_name']:25s} | "
                       f"useful: {row['useful_mean_tfidf']:.4f} | "
                       f"not_useful: {row['not_useful_mean_tfidf']:.4f} | "
                       f"diff: {row['difference']:+.4f}")
        
        logger.info("\n" + "="*60)
        logger.info(f"TOP {top_n} FEATURES FOR 'NOT_USEFUL' CLASS:")
        logger.info("="*60)
        for idx, row in not_useful_features.head(20).iterrows():
            logger.info(f"{idx+1:2d}. {row['feature_name']:25s} | "
                       f"useful: {row['useful_mean_tfidf']:.4f} | "
                       f"not_useful: {row['not_useful_mean_tfidf']:.4f} | "
                       f"diff: {row['difference']:+.4f}")
        
        logger.info(f"\nFull lists saved to:")
        logger.info(f"   - {output_dir / 'top_useful_features.csv'}")
        logger.info(f"   - {output_dir / 'top_not_useful_features.csv'}")
        
        return useful_features, not_useful_features


    def transform(self, texts: list): # transforms new texts to numerical features using existing vocabulary
        if not self.is_fitted: # Check if vectorizer is fitted
            raise ValueError("Vectorizer not fitted yet") # Can't transform if not fitted
        logger.info("Transforming texts using existing TF-IDF vocabulary")
        return self.vectorizer.transform(texts) # Use existing vocabulary to transform new texts, doesn't learn new words
    
    def get_feature_names(self):
        """
        Get the feature names (words) extracted by the vectorizer
        """
      
        if self.is_fitted:
            return self.vectorizer.get_feature_names_out() # Return feature names if fitted
        return [] # Return empty list if not fitted yet
