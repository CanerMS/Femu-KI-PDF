"""
Anomaly Detection Model Module
Uses Isolation Forest for detecting useful PDFs
"""
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from project_config import N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT, RANDOM_STATE
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
            contamination=contamination, # Proportion of anomalies in the data, its auto by default, 0.01 -> 20 useful docs, 0.1 -> 24 useful docs
            random_state=random_state, # Seed for reproducibility means results can be replicated
            n_estimators=100 # Number of trees in the forest
        )
        self.is_trained = False
    
    def train(self, X_train): # 
        logger.info(f"Training Isolation Forest on {X_train.shape[0]} samples")
        self.model.fit(X_train)
        self.is_trained = True
        logger.info("Training complete")
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        predictions = self.model.predict(X) 
        # Convert: -1 (anomaly/useful) to 0, 1 (normal/not useful) to 0
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

class PDFClassifier:
    """
    PDF Classifier supporting both supervised and unsupervised methods
    """
    def __init__(self, mode='supervised', contamination=0.1, random_state= RANDOM_STATE):
        """
        mode_supervised: 'supervised' or 'unsupervised' (anomaly detection)
        contamination: Proportion of anomalies for unsupervised mode
        random_state: Seed for reproducibility
        """
        self.mode = mode
        self.random_state = random_state

        if mode == 'supervised': # supervised classification
            self.model = RandomForestClassifier( # Using Random Forest for supervised classification
                n_estimators=100, # Number of trees
                max_depth=10, # To prevent overfitting
                min_samples_split=5, # To prevent overfitting
                min_samples_leaf=2, # To prevent overfitting
                class_weight={0: 1, 1: 15},  # Adjust class weights to handle imbalance
                random_state=RANDOM_STATE, # Seed for reproducibility
                n_jobs=-1 # Use all available cores
            )
            logger.info("Initialized supervised classifier with class_weight={0: 1, 1: 15}") # Log model initialization
            

        else:
            self.model = IsolationForest(
                contamination=contamination, # Proportion of anomalies in the data
                random_state=random_state, # Seed for reproducibility
                n_estimators=100 # Number of trees
            )

        self.is_trained = False

    def train(self, X_train, y_train=None):
        """
        Train the model

        Args:
            X_train: Feature matrix for training
            y_train: Labels for supervised training (None for unsupervised)
        """
        if self.mode == 'supervised':
            if y_train is None:
                raise ValueError("y_train must be provided for supervised training") # Supervised training requires labels
            logger.info(f"Training supervised model on {X_train.shape[0]}   samples") # Log training info
            logger.info(f"Class distribution: {np.bincount(y_train)}") # Log class distribution
            self.model.fit(X_train, y_train) # Fit supervised model
        else:
            logger.info(f"Training Isolation Forest on {X_train.shape[0]} samples") # Log training info
            self.model.fit(X_train) # Fit unsupervised model

        self.is_trained = True
        logger.info("Training complete")

    def predict(self, X):
        """
        Predict using the trained model

        Args:
            X: Feature matrix for prediction

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet") # Ensure model is trained before prediction

        if self.mode == 'supervised':
            return self.model.predict(X) # Predict using supervised model
        else:
            predictions = self.model.predict(X)
            predictions = np.where(predictions == -1, 1, 0) # Map Isolation Forest output to binary labels
            return predictions
    
    def predict_proba(self, X):
        """
        Predict probabilities using the trained supervised model

        Args:
            X: Feature matrix for prediction

        Returns:
            Probability estimates
        """
        if self.mode != 'supervised':
            raise ValueError("Probability prediction is only available in supervised mode") # Probabilities only for supervised

        if not self.is_trained:
            raise ValueError("Model not trained yet") # Ensure model is trained before prediction

        return self.model.predict_proba(X)
    
    def predict_scores(self, X):
        """
        Get anomaly scores (unsupervised) or probabilities (supervised)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if self.mode == 'supervised':
            proba = self.model.predict_proba(X)
            return proba[:, 1] # Return probability of positive class
        else:
            return self.model.score_samples(X)
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        """
        predictions = self.predict(X_test)
        scores = self.predict_scores(X_test)

        logger.info(f"Predicted {np.sum(predictions)} positive samples out of {len(predictions)}")

        if y_test is not None:  
            logger.info("\nClassification Report:")
            print(classification_report(y_test, predictions, target_names=['not_useful', 'useful']))
            logger.info("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, predictions)
            print(cm)
            logger.info(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
            logger.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

        return predictions, scores
    
    def save_model(self, path: str):
        joblib.dump(self.model, path) # Save the trained model to disk
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str): 
        self.model = joblib.load(path) # Load the trained model from disk
        self.is_trained = True
        logger.info(f"Model loaded from {path}")