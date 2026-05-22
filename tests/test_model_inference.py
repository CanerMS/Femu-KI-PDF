import pytest
import joblib
import numpy as np
import scipy.sparse as sp
import os
import sys
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "src"))

from src.model import LogisticRegressionClassifier
from src.preprocess import TextPreprocessor

# Models need to be uploaded
MODEL_PATH = "results/txt_logistic_regression_93_1_classifier.joblib"
TFIDF_DICT_PATH = "results/txt_tfidf_vocabulary_93_1.joblib"
SCALER_PATH = "results/txt_scaler_93_1.joblib"

@pytest.fixture
def loaded_pipeline():

    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
    assert os.path.exists(TFIDF_DICT_PATH), f"TFIDF file not found: {TFIDF_DICT_PATH}"
    assert os.path.exists(SCALER_PATH), f"Scaler file not found: {SCALER_PATH}"

    # Use the custom class 
    classifier = LogisticRegressionClassifier()
    classifier.load_model(MODEL_PATH)
    
    tfidf = joblib.load(TFIDF_DICT_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    return classifier, tfidf, scaler

def test_model_runs(loaded_pipeline):
    """Checks if model, tfidf, scaler successfully uploaded."""
    classifier, tfidf, scaler = loaded_pipeline
    
    assert classifier is not None
    assert classifier.is_trained is True
    assert hasattr(tfidf, "transform")
    assert hasattr(scaler, "transform")

def test_model_versions_match(loaded_pipeline):
    """
    Checks if the number of the expected features by Model compatible with the number of the Scaler's feature
    """
    classifier, tfidf, scaler = loaded_pipeline
    
    # Model keeps scikit learn model in classifier model
    sk_model = classifier.model 
    model_expected_features = sk_model.n_features_in_
    scaler_expected_features = scaler.n_features_in_
    
    # The dimension must be compatible, especially here, a version error can occur
    assert model_expected_features == scaler_expected_features, \
        f"Version incompability: Model expects {model_expected_features} feature, but scaler expects {scaler_expected_features}."

def test_prediction_with_empty_text(loaded_pipeline):
    """If an empty string shows up, the pipeline shoul not crash."""
    classifier, tfidf, scaler = loaded_pipeline
    preprocessor = TextPreprocessor()
    
    empty_text = ""
    clean_text = preprocessor.clean_text(empty_text)
    
    # TF-IDF process
    tf_vec = tfidf.transform([clean_text])
    
    # Create a mock vector for a reference to semantic
    # The dimension of the needed semantic vector is the difference TFIDF dimension and scaler dimension
    
    tfidf_size = tf_vec.shape[1]

    scaler_size = scaler.n_features_in_
    semantic_size = scaler_size - tfidf_size
    
    sem_vec = np.zeros((1, semantic_size))
    
    # Concatenate the vectors and scale them
    if sp.issparse(tf_vec):
        final_vector = sp.hstack((tf_vec, sp.csr_matrix(sem_vec)))
    else:
        final_vector = np.hstack((tf_vec, sem_vec))
        
    final_vector = scaler.transform(final_vector)
    
    # Can it be predicted without crash
    pred = classifier.predict(final_vector)[0]
    score = classifier.predict_scores(final_vector)[0]
    
    # Predict either 0 or 1
    assert pred in [0, 1]
    assert 0.0 <= score <= 1.0


def test_confidence_threshold_routing():
    """ 
    Logical test for the functionality Confidence Threshold
    """
    CONFIDENCE_THRESHOLD = 75.0
    
    # Situation 1: Model score is high, but negativity  (0.1 -> Model Prediction = 0, score = (1-0.1)*100 = 90%)
    prediction1 = 0
    raw_score1 = 0.1
    calculated_confidence1 = (1.0 - raw_score1) * 100 if prediction1 == 0 else raw_score1 * 100
    
    assert calculated_confidence1 >= CONFIDENCE_THRESHOLD
    # Aim "NOT USEFUL" 
    
    # Situation 2: Undecisive positive (0.6 -> Model Prediction = 1, score = %60)
    prediction2 = 1
    raw_score2 = 0.6
    calculated_confidence2 = (1.0 - raw_score2) * 100 if prediction2 == 0 else raw_score2 * 100
    
    assert calculated_confidence2 < CONFIDENCE_THRESHOLD
    # Here to check, that the decision is manual check