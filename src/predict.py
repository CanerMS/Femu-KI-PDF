import sys
import shutil  # Dosyaları taşımak için
from pathlib import Path
import logging
import joblib
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))

from project_config import FEATURE_MODE
from extractor import PDFExtractor, TXTExtractor
from preprocess import TextPreprocessor
from semantic import SciBERTSemanticFeatureExtractor
from model import LogisticRegressionClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Which model would you like to use?
    MODEL_PATH = "results/txt_logistic_regression_93_1_classifier.joblib"
    
    # Which tf-idf model would you like to combine?
    TFIDF_DICT_PATH = "results/txt_tfidf_vocabulary_93_1.joblib"

    # Which scaler model would you like to use?
    SCALER_PATH = "results/txt_scaler_93_1.joblib"
    
    FILE_TYPE = 'pdf'
    CONFIDENCE_THRESHOLD = 75.0  # Human in the loop threshold
    
    SILENT_MODE = True # Flag
        
    if SILENT_MODE:  
        logger.setLevel(logging.ERROR)
    
    FLAG_EX = False # for more explanations

    
    # 2. Arrange the files
    TARGET_DIR = Path("data/to_test_files")  # Which pdf/txt would you like to test?
    SORTED_DIR = Path("data/sorted_pdfs")    
    
    # Create aim directory
    DIR_USEFUL = SORTED_DIR / "Useful"
    DIR_NOT_USEFUL = SORTED_DIR / "Not_Useful"
    DIR_MANUAL_CHECK = SORTED_DIR / "Manual_Check"
    
    for d in [TARGET_DIR, DIR_USEFUL, DIR_NOT_USEFUL, DIR_MANUAL_CHECK]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Uploaded Model: {MODEL_PATH}")

    # 3. Load Model and tools
    model = LogisticRegressionClassifier()
    try:
        model.load_model(MODEL_PATH)
    except FileNotFoundError:
        logger.error("Error: Trained model could not be found!")
        return

    # Upload TF-IDF Dict 
    tfidf_extractor = None
    if FEATURE_MODE in ['tfidf', 'combined']:
        try:
            tfidf_extractor = joblib.load(TFIDF_DICT_PATH)
            logger.info(f"TF-IDF Dict uploaded successfully: {TFIDF_DICT_PATH}")
        except FileNotFoundError:
            logger.error(f"Critical Error: Feature_Mode='{FEATURE_MODE} but TF-IDF doesn't exist")
            return
        
    scaler = None
    if FEATURE_MODE == 'combined':
        try:
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"Scaler uploaded successfully: {SCALER_PATH}")
        except FileNotFoundError:
            logger.error(f"Critical Error: Feature_Mode='{FEATURE_MODE}' but Scaler doesn't exist")
            return
    
    preprocessor = TextPreprocessor()
    semantic_extractor = SciBERTSemanticFeatureExtractor()

    if FILE_TYPE == 'pdf':
        extractor = PDFExtractor(output_dir=Path("data/temp_extract"))
    else:
        extractor = TXTExtractor()

    files = list(TARGET_DIR.glob(f"*.{FILE_TYPE}"))
    if not files:
        logger.warning(f"No {FILE_TYPE} in: {TARGET_DIR}")
        return
    
    logger.info(f"In total {len(files)} will be predicted\n")

    # 4. Predict and divide to directories
    for file_path in files:
        logger.info(f"{file_path.name} checking...")
        
        # Extract the text
        if FILE_TYPE == 'pdf':
            text = extractor.extract_text_from_pdf(file_path)
        else:
            text = extractor.extract_text(file_path)
            
        clean_text = preprocessor.clean_text(text)
        
        final_vector = None
        if FEATURE_MODE == 'semantic':
            final_vector = semantic_extractor.extract_embeddings([clean_text], [file_path.stem])
        elif FEATURE_MODE == 'tfidf':
            final_vector = tfidf_extractor.transform([clean_text])
        elif FEATURE_MODE == 'combined':
            sem_vec = semantic_extractor.extract_embeddings([clean_text], [file_path.stem])
            tf_vec = tfidf_extractor.transform([clean_text])

            if sp.issparse(tf_vec):
                final_vector = sp.hstack((tf_vec, sp.csr_matrix(sem_vec)))
            else:    
                final_vector = np.hstack((tf_vec, sem_vec))
        

        # Scale before predicting
            if FEATURE_MODE == 'combined' and scaler is not None:
                final_vector = scaler.transform(final_vector)

        # Prediction and confident rate
        try:
            prediction = model.predict(final_vector)[0]
            score = model.predict_scores(final_vector)[0]
        except ValueError as e:
            logger.error(f"Convergence error")
            return 
    
        result = "USEFUL" if prediction == 1 else "NOT USEFUL"
        
        if prediction == 1:
            score_percentage = score * 100
        else:
            score_percentage = (1.0 - score) * 100

        if score_percentage < CONFIDENCE_THRESHOLD:
            final_score = 0.0
            aim_directory = DIR_MANUAL_CHECK
            if FLAG_EX:
                explanation = f""
        elif result == "USEFUL":
            final_score = score_percentage
            aim_directory = DIR_USEFUL
            if FLAG_EX:
                explanation = f"Positive score, archive to {DIR_USEFUL}"
        else: 
            final_score = -score_percentage # Negative score
            aim_directory = DIR_NOT_USEFUL
            if FLAG_EX:
                explanation = f"Negative score, archive to {DIR_NOT_USEFUL}"
        
        # if not enough confident
        if score_percentage < CONFIDENCE_THRESHOLD:
            # Human control
            aim_directory = DIR_MANUAL_CHECK
            explanation = "Human control needed"
        elif result == "USEFUL":
            aim_directory = DIR_USEFUL
            explanation = "Overconfident!"
        else:
            aim_directory = DIR_NOT_USEFUL
            explanation = "Overconfident, archive!"
            
        # Carry the file
        try:
            shutil.move(str(file_path), str(aim_directory / file_path.name))

            if SILENT_MODE:
                print(f"{final_score:.2f}")
            else:
                logger.info(f" -> Decision: {result} (%{score_percentage:.1f} Trust) | {explanation}\n")

            
        except Exception as e:
            logger.error(f" -> Eroor while {file_path.name} was carried: {e}\n")

    logger.info("You can see the results in 'data/sorted_pdfs' .")

if __name__ == "__main__":
    main()


