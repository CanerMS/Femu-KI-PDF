from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticFeatureExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2'): # small, fast, free
        self.model = SentenceTransformer(model_name)

    def extract_embeddings(self, texts):
        """
        Extract semantic embeddings using a pre-trained transformer model
        Semantic vectors capture contextual meaning of texts.
        """
        return self.model.encode(texts, convert_to_numpy=True)
    
    def combine_with_tfidf(self, tfidf_features, semantic_features):
        """TF-IDF + Semantic concatenate"""
        return np.hstack([tfidf_features, semantic_features])
    
    # TODO: Make this function more efficient 