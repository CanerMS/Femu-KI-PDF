import sys
from pathlib import Path
import logging
import joblib
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

# adding source into path
sys.path.insert(0, str(Path(__file__).parent))

from project_config import (
    FEATURE_MODE, RESULTS_DIR, 
    EXTRACTED_RAW_PDFS_DIR, EXTRACTED_USEFUL_PDFS_DIR,
    EXTRACTED_RAW_TXTS_DIR, EXTRACTED_USEFUL_TXTS_DIR
)
from extractor import PDFExtractor, TXTExtractor
from preprocess import TextPreprocessor
from semantic import SciBERTSemanticFeatureExtractor
from model import LogisticRegressionClassifier
from visualize_cm import plot_confusion_matrix_advanced
from utils import load_labels

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Configuration and Path Integration
    MODEL_PATH = "results/txt_logistic_regression_93_1_classifier.joblib"
    TFIDF_DICT_PATH = "results/txt_tfidf_vocabulary_93_1.joblib"
    SCALER_PATH = "results/txt_scaler_93_1.joblib"
    FILE_TYPE = 'pdf' # or txt
    
    TARGET_DIR = Path("data/to_test_files")
    
    logger.info("="*60)
    logger.info(f"NEW BATCH EVALUATION STARTING NOW... ({FILE_TYPE.upper()})")
    logger.info("="*60)

    # 2. Upload real labels 
    labels_df = load_labels()
    if labels_df is None:
        logger.error("labels.csv could not be found for the evaluation")
        return

    # 3. Upload tools and model
    model = LogisticRegressionClassifier()
    try:
        model.load_model(MODEL_PATH)
    except FileNotFoundError:
        logger.error("Error: Trained model could not be found!")
        return

    tfidf_extractor = None
    if FEATURE_MODE in ['tfidf', 'combined']:
        tfidf_extractor = joblib.load(TFIDF_DICT_PATH)
        
    scaler = None
    if FEATURE_MODE == 'combined':
        scaler = joblib.load(SCALER_PATH)
    
    preprocessor = TextPreprocessor()
    semantic_extractor = SciBERTSemanticFeatureExtractor()

    if FILE_TYPE == 'pdf':
        extractor = PDFExtractor(output_dir=Path("data/temp_extract"))
    else:
        extractor = TXTExtractor()

    files = list(TARGET_DIR.glob(f"*.{FILE_TYPE}"))
    if not files:
        logger.warning(f"That in {TARGET_DIR} to evaluate {FILE_TYPE} file doesn't exist.")
        return

    predicted_labels = []
    true_labels = []
    processed_files = []

    logger.info(f"Total {len(files)} files being processed...")

    # 4. Prediction time
    for file_path in files:
        # Check real labels
        file_label_info = labels_df[labels_df['filename'] == file_path.name]
        if file_label_info.empty:
            logger.warning(f"Skipping: No answer for '{file_path.name}' in labels.csv")
            continue
            
        true_label_str = file_label_info.iloc[0]['label']
        y_true = 1 if true_label_str == 'useful' else 0
        true_labels.append(y_true)
        processed_files.append(file_path.name)

        # Extraction
        text = None
        extracted_filename = f"{file_path.stem}.txt"
        
        # If extracted file name
        if FILE_TYPE == 'pdf':
            possible_paths = [
                EXTRACTED_RAW_PDFS_DIR / extracted_filename,
                EXTRACTED_USEFUL_PDFS_DIR / extracted_filename,
                Path("data/temp_extract") / extracted_filename
            ]
        else:
            possible_paths = [
                EXTRACTED_RAW_TXTS_DIR / extracted_filename,
                EXTRACTED_USEFUL_TXTS_DIR / extracted_filename
            ]

        # Check if the files already exist
        for p in possible_paths:
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    text = f.read()
                logger.info(f" Text uploaded through cache: {p.name}")
                break
                
        # If doesn't exist, extract from scratch
        if text is None:
            logger.info("Text extracting from scratch")
            if FILE_TYPE == 'pdf':
                text = extractor.extract_text_from_pdf(file_path)
                extractor.extract_and_save(file_path) # Save the extracted texts
            else:
                text = extractor.extract_text(file_path)
            
        clean_text = preprocessor.clean_text(text)
        
        # Vectorization
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
                
        if FEATURE_MODE == 'combined' and scaler is not None:
            final_vector = scaler.transform(final_vector)

        # Prediction
        prediction = model.predict(final_vector)[0]
        predicted_labels.append(prediction)

    if not processed_files:
        logger.warning("\nEvaluatable files not found!")
        return

    # 5. Visualization
    acc = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=['Not Useful (0)', 'Useful (1)'])
    cm = confusion_matrix(true_labels, predicted_labels)

    # Print in terminal
    logger.info("="*60)
    logger.info("Evaluation Report")
    logger.info("="*60)
    logger.info(f"Evaluated number of files : {len(processed_files)}")
    logger.info(f"General Accuracy  : % {acc * 100:.2f}")
    logger.info("\nDetailed metrics:")
    logger.info("\n" + report)
    
    # Save report to file
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    report_file = RESULTS_DIR / f'evaluation_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write(f"Evaluation Report - {timestamp}\n")
        f.write(f"General Accuracy % {acc * 100:.2f}\n\n")
        f.write(report)
    logger.info(f"Report printed: {report_file}")

    # Save advanced confusion matrix
    plot_path = RESULTS_DIR / f'evaluation_cm_{timestamp}.png'
    plot_confusion_matrix_advanced(
        cm=np.array(cm), 
        class_names=["not_useful", "useful"], 
        output_path=plot_path,
        dataset_label="to_test_files_Batch",
        model="Logistic_Regression_Evaluation"
    )
    logger.info(f"Confusion matrix saved: {plot_path}")
    logger.info("="*60)

if __name__ == "__main__":
    main()