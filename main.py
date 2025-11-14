"""
Main Pipeline
Orchestrates the PDF classification workflow
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src')) # Ensure src/ is in path

from loader import PDFLoader
from extractor import PDFExtractor
from preprocess import TextPreprocessor
from features import FeatureExtractor
from model import AnomalyDetector
from utils import save_predictions, create_labels_template

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("Starting PDF Classification Pipeline")
    logger.info("="*60)
    
    # 1. Load PDFs
    logger.info("\n[Step 1] Loading PDFs...")
    loader = PDFLoader()
    pdf_files = loader.get_pdf_files()
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    if len(pdf_files) == 0:
        logger.error("No PDF files found! Please add PDFs to data/raw_pdfs/")
        return
    
    # Split into train/test (50/50)
    train_files, test_files = loader.split_train_test(test_size=0.5)
    
    # 2. Extract text
    logger.info("\n[Step 2] Extracting text from PDFs...")
    extractor = PDFExtractor()
    
    logger.info("Extracting training data...")
    train_texts_dict = extractor.extract_batch(train_files)
    train_texts = list(train_texts_dict.values())
    
    logger.info("Extracting test data...")
    test_texts_dict = extractor.extract_batch(test_files)
    test_texts = list(test_texts_dict.values())
    
    # 3. Preprocess text
    logger.info("\n[Step 3] Preprocessing text...")
    preprocessor = TextPreprocessor()
    train_texts_clean = preprocessor.preprocess_batch(train_texts)
    test_texts_clean = preprocessor.preprocess_batch(test_texts)
    
    # 4. Extract features
    logger.info("\n[Step 4] Extracting TF-IDF features...")
    feature_extractor = FeatureExtractor(max_features=500)
    X_train = feature_extractor.fit_transform(train_texts_clean)
    X_test = feature_extractor.transform(test_texts_clean)
    
    # 5. Train anomaly detector
    logger.info("\n[Step 5] Training Anomaly Detection Model...")
    detector = AnomalyDetector(contamination=0.1)
    detector.train(X_train)
    
    # 6. Evaluate on test set
    logger.info("\n[Step 6] Evaluating on test set...")
    test_predictions, test_scores = detector.evaluate(X_test)
    
    # 7. Save results
    logger.info("\n[Step 7] Saving results...")
    test_filenames = [f.name for f in test_files]
    results_df = save_predictions(test_filenames, test_predictions, test_scores)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info(f"Total PDFs processed: {len(pdf_files)}")
    logger.info(f"Training set: {len(train_files)} PDFs")
    logger.info(f"Test set: {len(test_files)} PDFs")
    logger.info(f"Detected {test_predictions.sum()} potentially useful PDFs in test set")
    logger.info(f"\nResults saved to: results/predictions.csv")
    logger.info("="*60) # End of pipeline
    
    # Save model
    detector.save_model('results/anomaly_detector.joblib') # Save trained model
    
    return results_df # Return results dataframe

if __name__ == "__main__":
    main()
