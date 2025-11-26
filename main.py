"""
Main Pipeline
Orchestrates the PDF classification workflow
"""
import sys
from pathlib import Path
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from loader import PDFLoader 
from extractor import PDFExtractor
from preprocess import TextPreprocessor
from features import FeatureExtractor
from model import PDFClassifier
from utils import save_predictions, load_labels
from project_config import LABELS_PATH, MAX_FEATURES, SMOTE_THRESHOLD, RESULTS_DIR  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60) 
    logger.info("Starting PDF Classification Pipeline")
    logger.info("="*60)
    
    # 1. Load Labels
    logger.info("\n[Step 1] Loading Labels...")
    labels_df = load_labels(LABELS_PATH) 

    if labels_df is None:
        logger.error("Labels file not found. Please create labels.csv before running the pipeline.")
        return
    
    # 2. Load PDFs
    logger.info("\n[Step 2] Loading PDFs...")
    loader = PDFLoader()

    train_files = loader.get_pdfs_by_split('train', labels_df)
    test_files = loader.get_pdfs_by_split('test', labels_df)
    
    train_labels = loader.get_labels_for_pdfs(train_files, labels_df)
    test_labels = loader.get_labels_for_pdfs(test_files, labels_df)
    
    logger.info(f"Training: {len(train_files)} PDFs ({sum(train_labels)} useful)")
    logger.info(f"Testing: {len(test_files)} PDFs ({sum(test_labels)} useful)")
    
    # Analyze class balance in training set 
    useful_count = sum(train_labels)
    not_useful_count = len(train_labels) - useful_count

    # Safety checks
    if len(train_labels) == 0:
        logger.error("No training data found!")
        return

    if useful_count == 0:
        logger.error("No useful PDFs in training set!")
        logger.error("Please run: py src\\label_pdfs.py")
        return

    if not_useful_count == 0:
        logger.error("No 'not useful' PDFs in training set!")
        return

    imbalance_ratio = not_useful_count / useful_count  # Now safe

    logger.info(f"\nClass Balance:")
    logger.info(f"   Useful: {useful_count} ({useful_count/len(train_labels)*100:.1f}%)")
    logger.info(f"   Not Useful: {not_useful_count} ({not_useful_count/len(train_labels)*100:.1f}%)")
    logger.info(f"   Imbalance Ratio: 1:{imbalance_ratio:.1f}")

    if imbalance_ratio > 5:
        logger.warning("HIGH CLASS IMBALANCE detected!")
        logger.info("Model uses class_weight='balanced' to handle this automatically")
    
    # 3-4. Extract and preprocess (same as before)
    logger.info("\n[Step 3] Extracting text from PDFs...")
    extractor = PDFExtractor()
    train_texts_dict = extractor.extract_batch(train_files)
    test_texts_dict = extractor.extract_batch(test_files)
    
    logger.info("\n[Step 4] Preprocessing text...")
    preprocessor = TextPreprocessor()
    train_texts_clean = preprocessor.preprocess_batch(list(train_texts_dict.values()))
    # val_texts_clean = preprocessor.preprocess_batch(list(val_texts_dict.values()))
    test_texts_clean = preprocessor.preprocess_batch(list(test_texts_dict.values()))
    
    # 5. Extract features
    logger.info("\n[Step 5] Extracting TF-IDF features...")
    feature_extractor = FeatureExtractor(max_features=MAX_FEATURES) 
    X_train = feature_extractor.fit_transform(train_texts_clean)
    X_test = feature_extractor.transform(test_texts_clean)
    
    # Apply SMOTE if severe class imbalance
    if not_useful_count / useful_count > SMOTE_THRESHOLD: 
        logger.info("\n[Step 5.5] Applying SMOTE to balance classes...")
        try:
            from imblearn.over_sampling import SMOTE
            
            k_neighbors = min(5, useful_count - 1)
            
            if k_neighbors < 1:
                logger.warning("⚠️  Not enough minority samples for SMOTE (need at least 2). Skipping.")
            else:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train, train_labels_balanced = smote.fit_resample(X_train, train_labels)
                
                logger.info(f"   Before SMOTE: Useful={useful_count}, Not Useful={not_useful_count}")
                logger.info(f"   After SMOTE: Useful={sum(train_labels_balanced)}, Not Useful={len(train_labels_balanced) - sum(train_labels_balanced)}")
                logger.info(f"   Total samples: {len(train_labels_balanced)}")
                
                train_labels = train_labels_balanced 

        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
            logger.info("Install with: pip install imbalanced-learn")
        except ValueError as e: 
            logger.warning(f"SMOTE failed: {e}. Using class_weight='balanced' only.")
    
    # 6. Train classifier (SUPERVISED)
    logger.info("\n[Step 6] Training Supervised Classifier...")
    classifier = PDFClassifier(mode='supervised', random_state=42)
    classifier.train(X_train, np.array(train_labels))
    
    

    logger.info("\n[Step 6.5] Analyzing Top Features...")

    if classifier.mode == 'supervised':
        feature_names = feature_extractor.get_feature_names()
        importances = classifier.model.feature_importances_
        
        # Get top 10 features
        indices = np.argsort(importances)[::-1][:10]
        
        logger.info("\nTop 10 Most Important Features:")
        for i, idx in enumerate(indices, 1):
            logger.info(f"  {i}. '{feature_names[idx]}': {importances[idx]:.4f}")
    
    # Compare validation vs test performance
    
    # logger.info(f"\nValidation Accuracy: (see classification report above)")
    # logger.info("If validation accuracy is much higher than test, you may be overfitting.")
    
    # 7. Evaluate
    logger.info("\n[Step 7] Evaluating on test set...")
    test_predictions, test_scores = classifier.evaluate(X_test, np.array(test_labels))
    
    # 8. Save results
    logger.info("\n[Step 8] Saving results...")
    test_filenames = [f.name for f in test_files]
    results_df = save_predictions(test_filenames, test_predictions, test_scores)
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)
    
    classifier.save_model(RESULTS_DIR / 'pdf_classifier.joblib') 
    
    return results_df

if __name__ == "__main__":
    main()
