"""
Anomaly Detection Model Module
Uses Isolation Forest for detecting useful PDFs
"""
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Anomaly Detection using Isolation Forest
    Isolation Forest isolates anomalies instead of profiling normal data
    """
    def __init__(self, contamination: float = 'auto', random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination, # Proportion of anomalies in the data
            random_state=random_state, # Seed for reproducibility means results can be replicated
            n_estimators=100 # Number of trees in the forest
        )
        self.is_trained = False
    
    def train(self, X_train):
        logger.info(f"Training Isolation Forest on {X_train.shape[0]} samples")
        self.model.fit(X_train)
        self.is_trained = True
        logger.info("Training complete")
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        predictions = self.model.predict(X)
        # Convert: -1 (anomaly/useful) to 1, 1 (normal/not useful) to 0
        predictions = np.where(predictions == -1, 1, 0) # Map Isolation Forest output to binary labels
        return predictions
    
    def predict_scores(self, X):
        if not self.is_trained: # check if model is trained
            raise ValueError("Model not trained yet")
        return self.model.score_samples(X)
    
    def evaluate(self, X_test, y_test=None):
        predictions = self.predict(X_test)
        scores = self.predict_scores(X_test)
        
        logger.info(f"Predicted {np.sum(predictions)} anomalies out of {len(predictions)}")
        
        if y_test is not None:
            logger.info("\nClassification Report:")
            print(classification_report(y_test, predictions))
            logger.info("\nConfusion Matrix:")
            print(confusion_matrix(y_test, predictions))
        
        return predictions, scores
    
    def save_model(self, path: str):
        joblib.dump(self.model, path) # Save the trained model to disk
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        self.model = joblib.load(path) # Load the trained model from disk
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
